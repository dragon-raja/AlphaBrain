"""Native OpenPI action decoding with read-only same-forward trace/features.

    Optional imports occur only in the explicitly constructed runtime. The
    original native sampler, preprocessing and output transforms remain intact.
"""
from contextlib import contextmanager, nullcontext
import dataclasses
import json
from pathlib import Path
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import sha
from AlphaBrain.research.dsol.data.native_policy_noise import bound_noise, NOISE_RULE


@contextmanager
def velocity_capture(model):
    """Observe actual denoising outputs; restore the original method on failure."""
    original = model.denoise_step
    previous = model.__dict__.get('denoise_step')
    had_override = 'denoise_step' in model.__dict__
    values = []
    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        values.append(result.detach().float().cpu().numpy().copy())
        return result
    model.denoise_step = observed
    try:
        yield values
    finally:
        if had_override:
            model.denoise_step = previous
        else:
            del model.denoise_step


def observation(example):
    images = example['image']
    if len(images) != 2:
        raise ValueError('Native LIBERO needs external and wrist images')
    return {'observation/image':np.asarray(images[0]),
            'observation/wrist_image':np.asarray(images[1]),
            'observation/state':np.asarray(example['state']).reshape(-1),
            'prompt':example['lang']}


class NativePolicyRuntime:
    def __init__(self, config, output, limit):
        import torch
        from openpi.training.config import get_config
        from openpi.policies.policy import Policy
        from openpi.shared.normalize import load
        from openpi import transforms
        self.torch = torch
        self.output, self.limit, self.calls = Path(output), int(limit), 0
        checkpoint = config['models'][config['operation_model']]
        if (checkpoint.get('implementation') != 'openpi_native'
                or checkpoint.get('noise_rule') != NOISE_RULE
                or checkpoint.get('score_channels') != list(range(7))):
            raise ValueError('Explicit native runtime/noise/executed-channel contract required')
        path = Path(checkpoint['path'])
        if sha(path/'model.safetensors') != checkpoint['weights_sha256']:
            raise ValueError('Native operation weights changed')
        norm = path/'assets/physical-intelligence/libero/norm_stats.json'
        if sha(norm) != checkpoint['normalization_sha256']:
            raise ValueError('Native quantile normalization changed')
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        cfg = get_config('pi05_libero')
        cfg = dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,pytorch_compile_mode=None))
        if cfg.model.action_dim != 32 or cfg.model.action_horizon != 10:
            raise ValueError('Native action shape changed')
        model = cfg.model.load_pytorch(cfg,str(path/'model.safetensors'))
        model.paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16')
        model.requires_grad_(False)
        data = cfg.data.create(cfg.assets_dirs,cfg.model)
        if not data.use_quantile_norm:
            raise ValueError('Native quantile decoding required')
        stats = load(path/'assets'/data.asset_id)
        self.policy = Policy(model,
            transforms=[transforms.InjectDefaultPrompt(None),*data.data_transforms.inputs,
                        transforms.Normalize(stats,use_quantiles=True),*data.model_transforms.inputs],
            output_transforms=[*data.model_transforms.outputs,
                transforms.Unnormalize(stats,use_quantiles=True),*data.data_transforms.outputs],
            sample_kwargs={'num_steps':10},metadata=cfg.policy_metadata,
            is_pytorch=True,pytorch_device='cuda:0')
        self.model = model

    def infer(self, example, noise, identity, capture=True, *, instrument=True):
        from AlphaBrain.model.pi05_features import SiglipFeatureCapture
        from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores
        from AlphaBrain.research.dsol.data.flow_noise import tensor_sha256
        if self.calls >= self.limit:
            raise ValueError('Bounded native model trajectory budget exhausted')
        full_noise, seed = bound_noise(noise)
        self.calls += 1
        with (self.output/'calls.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(index=self.calls,**identity,noise_rule=NOISE_RULE,
                extension_seed=str(seed),full_noise_sha256=tensor_sha256(full_noise)))+'\n')
        torch = self.torch
        tower = self.model.paligemma_with_expert.paligemma.vision_tower
        tap = SiglipFeatureCapture(tower,patch_grid=(16,16),view_names=('external','wrist','masked_dummy'))
        torch.cuda.synchronize(); started = time.perf_counter()
        if capture and not instrument:
            raise ValueError('Native uninstrumented control cannot return features')
        with torch.inference_mode(), (tap if capture else nullcontext()), (velocity_capture(self.model) if instrument else nullcontext()) as velocities:
            pred = self.policy.infer(observation(example),noise=full_noise)
        torch.cuda.synchronize()
        actions = np.asarray(pred['actions'],dtype=np.float32)
        if not instrument:
            if actions.shape != (10,7) or not np.isfinite(actions).all():
                raise ValueError('Invalid native control actions')
            return dict(actions=actions,full_noise=full_noise,seconds=time.perf_counter()-started)
        full_trace = np.stack(velocities,axis=1)
        if (full_trace.shape != (1,10,10,32) or actions.shape != (10,7)
                or not np.isfinite(full_trace).all() or not np.isfinite(actions).all()):
            raise ValueError('Native inference shape or finiteness mismatch')
        result = dict(trace=full_trace[0,:,:,:7].copy(),full_trace=full_trace[0],
                      full_noise=full_noise,actions=actions,seconds=time.perf_counter()-started)
        if capture:
            score = compute_accel_scores(result['trace'][None],prefixes=[3])[3]
            if score['degenerate'].any():
                raise ValueError('Degenerate native score')
            result['score'] = float(score['score'][0])
            result.update({k:v.cpu().numpy().reshape(-1) for k,v in tap.features().items()
                           if k in ('external','wrist')})
        return result
