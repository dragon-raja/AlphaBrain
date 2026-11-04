from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


PALIGEMMA_IMAGE_TOKENS_PER_VIEW = 256
_NEG_INF_ADDITIVE = -2.3819763e38


class StandaloneSiglipFeatures:
    """Frozen two-view tower using the same capture/pooling as native Pi0.5.

    Accepts explicitly preprocessed pixels. It does not resize, normalize, fill
    missing cameras, load a manipulation model, or certify cross-model transfer.
    Caller admission must compare these pixels and features with the source
    policy on real observations before this bridge is used in an experiment.
    """
    def __init__(self, tower):
        self.tower=tower.eval().requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, pixels):
        names=('external','wrist')
        if set(pixels)!=set(names):
            raise ValueError('Both real external and wrist views are required')
        for name in names:
            value=pixels[name]
            if (not isinstance(value,torch.Tensor) or tuple(value.shape)!=(1,3,224,224)
                    or value.dtype!=torch.float32 or not torch.isfinite(value).all()
                    or value.min() < -1 or value.max() > 1):
                raise ValueError('Explicit float32 [-1,1] source-policy pixels required')
        tap=SiglipFeatureCapture(self.tower,patch_grid=(16,16),view_names=names)
        with tap:
            for name in names:self.tower(pixel_values=pixels[name])
        features=tap.features()
        if any(tuple(v.shape)!=(1,4,1152) for v in features.values()):
            raise ValueError('Source actor requires the admitted 4608-dimensional tower')
        return {k:v.reshape(1,4608) for k,v in features.items()}


def load_standalone_siglip(checkpoint, *, expected_sha256, vision_config, device):
    """Load only the vision subtree; no language/action expert or downloads.

    Configuration and implementation identity are supplied by the binding.
    Loading weights is not numerical admission and performs no model forward.
    """
    import hashlib
    from safetensors import safe_open
    from transformers import SiglipVisionConfig, SiglipVisionModel
    h=hashlib.sha256()
    with open(checkpoint,'rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    if h.hexdigest()!=expected_sha256:
        raise ValueError('Source visual checkpoint changed')
    prefix='paligemma_with_expert.paligemma.model.vision_tower.'
    with safe_open(checkpoint,framework='pt',device='cpu') as archive:
        state={k[len(prefix):]:archive.get_tensor(k) for k in archive.keys() if k.startswith(prefix)}
    with torch.device('meta'):
        tower=SiglipVisionModel(SiglipVisionConfig(**vision_config))
    tower.load_state_dict(state,strict=True,assign=True)
    # Transformers omits deterministic position_ids from state_dict. Restore
    # that buffer explicitly; assign=True cannot materialize meta buffers.
    embeddings=tower.vision_model.embeddings
    embeddings.position_ids=torch.arange(embeddings.num_positions,device='cpu').expand((1,-1))
    if any(b.is_meta for b in tower.buffers()):
        raise ValueError('Unrecognized meta buffer in source visual architecture')
    # Match the *resolved native runtime*, not the checkpoint dtype alone.
    # NativePolicyRuntime explicitly calls to_bfloat16_for_selected_params
    # after loading, rounding even the float32 checkpoint embeddings through
    # bf16 before restoring their compute dtype. Preserving their original
    # precision here changes the frozen actor's input features (CPU load audit).
    tower=tower.to(device=device,dtype=torch.bfloat16)
    for name,param in tower.named_parameters():
        if name.startswith('vision_model.embeddings.'):
            param.data=param.data.float()
    return StandaloneSiglipFeatures(tower)


class SiglipFeatureCapture:
    """Capture 2x2 pooled tokens during an existing eager policy forward.

    This never invokes the vision tower, changes its output, or changes model
    mode. The caller supplies the actual patch grid and expected view order.
    A capture must surround exactly one serialized policy request. Concurrent
    calls to the same tower and compiled inference require separate admission.
    Features are detached float32 tensors on the original device.
    """

    def __init__(self, vision_tower, *, patch_grid, view_names):
        self.tower = vision_tower
        self.patch_grid = tuple(patch_grid)
        self.view_names = tuple(view_names)
        if (len(self.patch_grid) != 2
                or any(not isinstance(x, int) or x < 2 for x in self.patch_grid)
                or not self.view_names or len(set(self.view_names)) != len(self.view_names)
                or any(not isinstance(x, str) or not x for x in self.view_names)):
            raise ValueError("Explicit patch grid and unique nonempty view names required")
        self._handle = None
        self._features = []

    def _capture(self, module, args, output):
        tokens = output.last_hidden_state
        height, width = self.patch_grid
        if (tokens.ndim != 3 or tokens.shape[1] != height * width
                or not torch.isfinite(tokens).all()):
            raise ValueError("SigLIP token shape/values disagree with declared patch grid")
        if len(self._features) >= len(self.view_names):
            raise ValueError("Unexpected extra vision forward in one policy request")
        grid = tokens.detach().float().transpose(1, 2).reshape(
            tokens.shape[0], tokens.shape[2], height, width)
        pooled = F.adaptive_avg_pool2d(grid, (2, 2)).flatten(2).transpose(1, 2)
        self._features.append(pooled.contiguous())
        # Returning None is essential: do not replace the operation model output.

    def __enter__(self):
        if self._handle is not None:
            raise RuntimeError("Capture is already active")
        self._features = []
        self._handle = self.tower.register_forward_hook(self._capture)
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._handle.remove()
        self._handle = None
        if exc_type is not None:
            self._features = []
        return False

    def features(self):
        if self._handle is not None:
            raise RuntimeError("Read features after the policy request finishes")
        if len(self._features) != len(self.view_names):
            raise ValueError("Missing vision calls; cached features are not a substitute")
        shapes = {tuple(value.shape) for value in self._features}
        if len(shapes) != 1:
            raise ValueError("Views must have matching batch and feature dimensions")
        return dict(zip(self.view_names, self._features))


def pool_pi05_image_views(
    last_hidden: torch.Tensor,
    *,
    num_views: int,
    tokens_per_view: int = PALIGEMMA_IMAGE_TOKENS_PER_VIEW,
) -> torch.Tensor:
    if last_hidden.ndim != 3:
        raise ValueError(f"last_hidden must have shape [B, L, H], got {last_hidden.shape}")
    if num_views < 1:
        raise ValueError("num_views must be positive")
    required_tokens = num_views * tokens_per_view
    if last_hidden.shape[1] < required_tokens:
        raise ValueError(
            f"hidden sequence has {last_hidden.shape[1]} tokens, needs {required_tokens}"
        )
    pooled = [
        last_hidden[:, view * tokens_per_view : (view + 1) * tokens_per_view].mean(dim=1)
        for view in range(num_views)
    ]
    return torch.cat(pooled, dim=-1)


@torch.no_grad()
def extract_pi05_image_feature(vla, example: Mapping[str, object]) -> torch.Tensor:
    """Extract final-layer image features from one deployable Pi0.5 policy input."""
    from AlphaBrain.model.modules.action_model.pi0_flow_matching_head.openpi_inference import (
        make_att_2d_masks,
    )

    images = list(example["image"]) if isinstance(example["image"], (list, tuple)) else [example["image"]]
    prefix_embs, prefix_pad_masks, prefix_att_masks = vla._prepare_prefix_paligemma([dict(example)])
    prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_att_4d = torch.where(
        prefix_att_2d[:, None, :, :],
        0.0,
        _NEG_INF_ADDITIVE,
    )
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    language_model = vla._get_vlm_language_model()
    language_model.config._attn_implementation = "eager"
    output = language_model.forward(
        inputs_embeds=prefix_embs,
        attention_mask=prefix_att_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        use_cache=False,
    )
    return pool_pi05_image_views(output.last_hidden_state, num_views=len(images))
