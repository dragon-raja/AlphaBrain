import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from AlphaBrain.research.dsol.data.native_policy_noise import bound_noise, extend_noise
from benchmarks.LIBERO.native_search_runtime import observation, velocity_capture
from benchmarks.LIBERO.search_runtime import create_policy_runtime, exact_prediction


def test_historical_extension_and_pose_independent_binding():
    from benchmarks.LIBERO.initialization_diagnostic import extend_noise as historical
    noise=np.random.default_rng(8).standard_normal((10,7),dtype=np.float32)
    assert np.array_equal(extend_noise(noise,91),historical(noise,91,32))
    value,seed=bound_noise(noise)
    again,same=bound_noise(noise.astype('>f4'))
    assert seed==same and np.array_equal(value,again)
    assert np.array_equal(value[:,:7],noise)
    changed=noise.copy();changed[0,0]+=1
    assert bound_noise(changed)[1]!=seed
    assert not np.array_equal(bound_noise(changed)[0][:,7:],value[:,7:])
    for bad in (np.zeros((10,32)),np.full((10,7),np.nan)):
        with pytest.raises(ValueError):bound_noise(bad)


def test_native_tensor_noise_extension_does_not_repeat_original_coordinates():
    values=[bound_noise(np.random.default_rng(i).standard_normal((10,7),dtype=np.float32))[0]
            for i in range(256)]
    corr=np.corrcoef(np.stack(values).reshape(-1,32),rowvar=False)
    assert abs(corr[:7,7:]).max()<0.09


def test_velocity_capture_returns_original_tensor_and_restores_on_failure():
    torch=pytest.importorskip('torch')
    class Model:
        def denoise_step(self,x):return x*2
    m=Model();x=torch.tensor([1.,2.]);before=m.denoise_step(x)
    with velocity_capture(m) as trace:
        assert torch.equal(m.denoise_step(x),before)
    assert np.array_equal(trace[0],before.numpy()) and 'denoise_step' not in vars(m)
    override=lambda x:x+3
    m.denoise_step=override
    with pytest.raises(RuntimeError),velocity_capture(m):
        m.denoise_step(x);raise RuntimeError('inference failure')
    assert m.denoise_step is override


@pytest.mark.parametrize('dtype_name', ['float32', 'float64', 'bfloat16'])
def test_deferred_capture_matches_legacy_with_reused_buffers(dtype_name, monkeypatch):
    torch = pytest.importorskip('torch')
    from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores
    dtype = getattr(torch, dtype_name)
    class Model:
        def __init__(self):
            self.buffer = torch.empty((1, 10, 32), dtype=dtype)
        def denoise_step(self, step):
            self.buffer.copy_(torch.arange(320).reshape(1, 10, 32) * (step + 1) / 17 + step ** 2)
            return self.buffer
    transfers = []
    original_cpu = torch.Tensor.cpu
    def cpu(value, *args, **kwargs):
        transfers.append(tuple(value.shape))
        return original_cpu(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'cpu', cpu)
    traces = []
    for mode in ('per_step_cpu', 'deferred_cpu'):
        model = Model(); transfers.clear()
        with velocity_capture(model, transfer=mode) as values:
            for step in range(10):
                returned = model.denoise_step(step)
                assert returned is model.buffer
                if mode == 'deferred_cpu':
                    assert transfers == []
            # A reference-only capture would now lose all original samples.
            model.buffer.fill_(-999)
        assert len(transfers) == (10 if mode == 'per_step_cpu' else 1)
        assert all(isinstance(v, np.ndarray) and v.dtype == np.float32 for v in values)
        traces.append(np.stack(values, axis=1))
        assert 'denoise_step' not in vars(model)
    np.testing.assert_array_equal(*traces)
    scores = [compute_accel_scores(t[:, :, :, :7], prefixes=[3])[3] for t in traces]
    for key in scores[0]:
        np.testing.assert_array_equal(scores[0][key], scores[1][key])


def test_deferred_capture_restores_override_and_discards_failed_trace(monkeypatch):
    torch = pytest.importorskip('torch')
    model = SimpleNamespace(denoise_step=lambda x: x)
    original = model.denoise_step
    def forbidden_cpu(*args, **kwargs):
        raise AssertionError('Failed sampling must not transfer a partial trace')
    monkeypatch.setattr(torch.Tensor, 'cpu', forbidden_cpu)
    with pytest.raises(RuntimeError, match='sampling failed'):
        with velocity_capture(model, transfer='deferred_cpu') as values:
            model.denoise_step(torch.ones(1, 10, 32))
            raise RuntimeError('sampling failed')
    assert values == [] and model.denoise_step is original


def test_deferred_transfer_failure_restores_method(monkeypatch):
    torch = pytest.importorskip('torch')
    model = SimpleNamespace(denoise_step=lambda x: x)
    original = model.denoise_step
    def failed_cpu(*args, **kwargs):
        raise RuntimeError('transfer failed')
    monkeypatch.setattr(torch.Tensor, 'cpu', failed_cpu)
    with pytest.raises(RuntimeError, match='transfer failed'):
        with velocity_capture(model, transfer='deferred_cpu') as values:
            model.denoise_step(torch.ones(1, 10, 32))
    assert values == [] and model.denoise_step is original


def test_unknown_transfer_rejected_before_model_override():
    model = SimpleNamespace(denoise_step=lambda x: x)
    original = model.denoise_step
    with pytest.raises(ValueError, match='transfer mode'):
        with velocity_capture(model, transfer='implicit-auto'):
            pytest.fail('Unknown mode was accepted')
    assert model.denoise_step is original


def test_actual_native_infer_entry_matches_for_both_transfer_modes(monkeypatch, tmp_path):
    """Exercise the real runtime wrapper with CPU tensors and a fake sampler.

    This validates the trace/features/noise/score assembly, not real GPU kernels.
    """
    torch = pytest.importorskip('torch')
    from AlphaBrain.model import pi05_features
    from AlphaBrain.research.dsol.execution import dispatch_policy
    from benchmarks.LIBERO.native_search_runtime import NativePolicyRuntime
    monkeypatch.setattr(dispatch_policy, 'reject_production', lambda entry: None)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    class Tap:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def features(self):
            return {'external': torch.arange(12), 'wrist': torch.arange(12) + 20}
    monkeypatch.setattr(pi05_features, 'SiglipFeatureCapture', Tap)
    noise = np.random.default_rng(42).standard_normal((10, 7), dtype=np.float32)
    example = dict(image=[np.zeros((2, 2, 3), np.uint8)] * 2,
                   state=np.zeros(8), lang='synthetic CPU test')
    results = []
    for mode in ('per_step_cpu', 'deferred_cpu'):
        model = SimpleNamespace(denoise_step=lambda x, step: x * (step + 1) + step ** 2,
            paligemma_with_expert=SimpleNamespace(paligemma=SimpleNamespace(vision_tower=None)))
        def infer(obs, *, noise):
            value = torch.from_numpy(noise)[None]
            for step in range(10):
                value = model.denoise_step(value, step) / 10
            return {'actions': value[0, :, :7].numpy()}
        runtime = NativePolicyRuntime.__new__(NativePolicyRuntime)
        runtime.torch = torch
        runtime.model = model
        runtime.policy = SimpleNamespace(infer=infer)
        runtime.trace_transfer = mode
        runtime.calls = 0; runtime.limit = 1
        runtime.output = tmp_path / mode; runtime.output.mkdir()
        results.append(runtime.infer(example, noise, {'case': 'cpu-test'}))
        assert runtime.calls == 1
        assert results[-1]['full_trace'].shape == (10, 10, 32)
        assert results[-1]['trace'].shape == (10, 10, 7)
        with pytest.raises(ValueError, match='budget exhausted'):
            runtime.infer(example, noise, {'case': 'cpu-test'})
    exact_prediction(results[0], results[1])


def test_unknown_runtime_transfer_rejected_before_model_loading(monkeypatch, tmp_path):
    from AlphaBrain.research.dsol.execution import dispatch_policy
    from benchmarks.LIBERO.native_search_runtime import NativePolicyRuntime
    monkeypatch.setattr(dispatch_policy, 'reject_production', lambda entry: None)
    with pytest.raises(ValueError, match='transfer mode'):
        NativePolicyRuntime({'native_trace_transfer': 'auto'}, tmp_path, 1)


def test_native_observation_keeps_raw_pixels_and_state():
    a=np.arange(12,dtype=np.uint8).reshape(2,2,3)
    e=dict(image=[a,a+1],state=np.arange(8)[None],lang='move')
    out=observation(e)
    assert np.array_equal(out['observation/image'],a)
    assert out['observation/state'].shape==(8,) and out['prompt']=='move'
    with pytest.raises(ValueError):observation(dict(e,image=[a]))


def test_native_factory_is_explicit_and_never_uses_project_loader(monkeypatch,tmp_path):
    from benchmarks.LIBERO import native_search_runtime,search_runtime
    calls=[]
    def fake_native(*args):calls.append('native');return 123
    def fake_project(*args):calls.append('project');return 456
    monkeypatch.setattr(native_search_runtime,'NativePolicyRuntime',fake_native)
    monkeypatch.setattr(search_runtime,'PolicyRuntime',fake_project)
    model=dict(path='unused',weights_sha256='a'*64,implementation='openpi_native')
    cfg=dict(operation_model='native',models={'native':model})
    assert create_policy_runtime(cfg,tmp_path,4)==123 and calls==['native']
    cfg['models']['native']['implementation']='unknown'
    with pytest.raises(ValueError):create_policy_runtime(cfg,tmp_path,4)


def test_full_native_trace_is_part_of_return_consistency():
    a={k:np.ones(2) for k in ('trace','actions','score','external','wrist','full_trace','full_noise')}
    exact_prediction(a,a)
    b=dict(a,full_trace=np.zeros(2))
    with pytest.raises(ValueError,match='full-dimensional'):exact_prediction(a,b)
