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
