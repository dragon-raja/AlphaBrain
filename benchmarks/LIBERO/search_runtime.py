"""Frozen batch-one manipulation policy shared by search and capacity workers."""
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import sha


def selected_checkpoint(config, *, allow_native=False):
    """Select a named frozen policy; preserve legacy canonical-only manifests."""
    key = config.get('operation_model', 'canonical')
    if not isinstance(key, str) or key not in config['models']:
        raise ValueError('Unknown frozen operation model')
    checkpoint = config['models'][key]
    if not checkpoint.get('path') or not checkpoint.get('weights_sha256'):
        raise ValueError('Frozen operation model needs path and weights hash')
    # Historical intervention established a decoding mismatch for this exact
    # zero-update artifact. Numerical repeatability cannot admit it as a valid
    # manipulation baseline (see the September 23 interpretation audit).
    if checkpoint['weights_sha256'] == 'f91d7bbb1ea1c212570436525d8e087a979a249fee9c6b41feb286139017cb75':
        raise ValueError('Rejected zero-step policy: known quantile/mean-std decoding mismatch')
    implementation = checkpoint.get('implementation', 'alphabrain')
    if implementation != 'alphabrain' and not (allow_native and implementation == 'openpi_native'):
        raise ValueError('Native policy requires its separately admitted runtime; no fallback loader')
    return checkpoint


def create_policy_runtime(config, output, limit):
    checkpoint = selected_checkpoint(config, allow_native=True)
    if checkpoint.get('implementation', 'alphabrain') == 'openpi_native':
        from .native_search_runtime import NativePolicyRuntime
        return NativePolicyRuntime(config, output, limit)
    return PolicyRuntime(config, output, limit)


class PolicyRuntime:
    def __init__(self, config, output, limit):
        import torch
        from AlphaBrain.model.framework.base_framework import BaseFramework
        self.torch = torch
        self.output, self.limit, self.calls = Path(output), int(limit), 0
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        checkpoint = selected_checkpoint(config)
        if sha(Path(checkpoint['path'])/'model.safetensors') != checkpoint['weights_sha256']:
            raise ValueError('Frozen manipulation policy changed')
        self.policy = BaseFramework.from_pretrained(checkpoint['path'], strict_checkpoint=True).to(torch.bfloat16).to('cuda').eval()
        self.policy.gripper_remap = False
        self.policy.requires_grad_(False)

    def infer(self, example, noise, identity, capture=True):
        from AlphaBrain.model.pi05_features import SiglipFeatureCapture
        from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores
        from AlphaBrain.research.dsol.metrics.accel_inference import _validated_trace_output
        if self.calls >= self.limit:
            raise ValueError('Bounded model trajectory budget exhausted')
        self.calls += 1
        with (self.output/'calls.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(index=self.calls, **identity))+'\n')
        tap = SiglipFeatureCapture(self.policy.vlm_interface.model.vision_tower,
            patch_grid=(16,16), view_names=('external','wrist','masked_dummy'))
        torch = self.torch
        torch.cuda.synchronize(); begin = time.perf_counter()
        with torch.inference_mode(), (tap if capture else nullcontext()):
            prediction = self.policy.predict_action(examples=[example], noise=noise[None], return_flow_trace=True)
        torch.cuda.synchronize()
        trace, _, _, actions = _validated_trace_output(prediction, noise[None], candidate_count=1, action_horizon=10, action_dim=7)
        result = dict(trace=trace[0], actions=actions[0], seconds=time.perf_counter()-begin)
        if capture:
            score = compute_accel_scores(trace, prefixes=[3])[3]
            if score['degenerate'].any():
                raise ValueError('Degenerate live score')
            result['score'] = float(score['score'][0])
            result.update({k:v.cpu().numpy().reshape(-1) for k,v in tap.features().items() if k in ('external','wrist')})
        return result


def exact_prediction(value, reference):
    """No tolerance or comparison of aggregate success in place of outputs."""
    for key in ('trace','actions','score','external','wrist'):
        if not np.array_equal(value[key], reference[key]):
            raise ValueError('Frozen batch1 prediction mismatch: '+key)
    for key in ('full_trace','full_noise'):
        if (key in value) != (key in reference) or (key in value and not np.array_equal(value[key], reference[key])):
            raise ValueError('Native full-dimensional prediction mismatch: '+key)
