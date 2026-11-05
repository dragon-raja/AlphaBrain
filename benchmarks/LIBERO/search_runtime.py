"""Frozen batch-one manipulation policy shared by search and capacity workers."""
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import sha


def selected_checkpoint(config, *, allow_native=False, allow_fastwam=False, allow_groot=False,native_family=None):
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
    registered=None
    if native_family is not None:
        from .model_adapters import model_adapter
        if config.get('native_features',{}).get('family')!=native_family:raise ValueError('Native family/capture binding differs')
        registered=model_adapter(native_family).implementation
    if (implementation != 'alphabrain' and not (allow_native and implementation == 'openpi_native')
            and not (allow_fastwam and implementation == 'fastwam_native')
            and not (allow_groot and implementation == 'groot_native') and implementation!=registered):
        raise ValueError('Native policy requires its separately admitted runtime; no fallback loader')
    return checkpoint


def create_policy_runtime(config, output, limit):
    if config.get('score_contract') is not None:
        from .score_prefix_runtime import PrefixRuntime
        from AlphaBrain.research.dsol.data.score_prefix import validate
        spec=validate(config['score_contract']);base=dict(config);base.pop('score_contract')
        expected=config.get('native_features',{}).get('family','pi05')
        if expected!=spec['family']:raise ValueError('Score/model family differs')
        return PrefixRuntime(create_policy_runtime(base,output,limit),spec)
    if config.get('native_features') is not None:
        from .model_adapters import model_adapter
        adapter=model_adapter(config['native_features']['family'])
        selected_checkpoint(config,native_family=adapter.family)
        return adapter.create_runtime(config,output,limit)
    checkpoint = selected_checkpoint(config, allow_native=True, allow_fastwam=True, allow_groot=True)
    if checkpoint.get('implementation') == 'groot_native':
        from .groot_runtime import GrootRuntime
        return GrootRuntime(checkpoint['runtime_contract'],checkpoint['runtime_contract_sha256'],
                            output,limit,instrument=True,**({'native_features':config['native_features']} if config.get('native_features') is not None else {}))
    if checkpoint.get('implementation') == 'fastwam_native':
        from .legacy.fastwam_runtime import FastWAMRuntime
        return FastWAMRuntime(config, output, limit)
    if checkpoint.get('implementation', 'alphabrain') == 'openpi_native':
        from .native_search_runtime import NativePolicyRuntime
        return NativePolicyRuntime(config, output, limit)
    return PolicyRuntime(config, output, limit)


def bind_policy_task(runtime, output, limit):
    """Rebind accounting only; the resident model and scientific state stay fixed.

    Called inside a sequential resident lease after a new center grant. A block
    with a failed/unknown outcome never reaches a later lease.
    """
    from AlphaBrain.research.dsol.execution import dispatch_policy
    dispatch_policy.reject_production('benchmarks/LIBERO/native_view_panel.py:run')
    output = Path(output)
    if (type(limit) is not int or limit < 1 or not output.is_dir()
            or (output/'calls.jsonl').exists() or type(runtime.calls) is not int
            or not 0 <= runtime.calls <= runtime.limit or getattr(runtime,'failed',False)):
        raise RuntimeError('Invalid resident block accounting boundary')
    runtime.output, runtime.limit, runtime.calls = output, limit, 0


class PolicyRuntime:
    def __init__(self, config, output, limit):
        from AlphaBrain.research.dsol.execution import dispatch_policy
        dispatch_policy.reject_production('benchmarks/LIBERO/search_runtime.py:PolicyRuntime.__init__')
        import torch
        from AlphaBrain.model.framework.base_framework import BaseFramework
        self.torch = torch
        self.output, self.limit, self.calls = Path(output), int(limit), 0
        from AlphaBrain.common.torch_runtime import configure_threads
        configure_threads(torch, intra=2, interop=1)
        checkpoint = selected_checkpoint(config)
        if sha(Path(checkpoint['path'])/'model.safetensors') != checkpoint['weights_sha256']:
            raise ValueError('Frozen manipulation policy changed')
        self.policy = BaseFramework.from_pretrained(checkpoint['path'], strict_checkpoint=True).to(torch.bfloat16).to('cuda').eval()
        self.policy.gripper_remap = False
        self.policy.requires_grad_(False)

    def infer(self, example, noise, identity, capture=True):
        from AlphaBrain.research.dsol.execution import dispatch_policy
        dispatch_policy.reject_production('benchmarks/LIBERO/search_runtime.py:PolicyRuntime.infer')
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
    for key in ('full_trace','full_noise','full_actions','timesteps','deltas'):
        if (key in value) != (key in reference) or (key in value and not np.array_equal(value[key], reference[key])):
            raise ValueError('Native full-dimensional prediction mismatch: '+key)
