from __future__ import annotations

from typing import Mapping

import torch


PALIGEMMA_IMAGE_TOKENS_PER_VIEW = 256
_NEG_INF_ADDITIVE = -2.3819763e38


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
