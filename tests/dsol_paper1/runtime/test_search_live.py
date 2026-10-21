import numpy as np
import pytest

from benchmarks.LIBERO.search_live import run, verify_example, simulator


def test_live_validation_cannot_substitute_cached_inputs():
    image=np.zeros((2,2,3),np.uint8)
    example=dict(image=[image,image],state=np.zeros(8,np.float32),camera_intrinsics=np.eye(3),
                 camera_to_world_opencv=np.eye(4))
    cached=dict(external_images=image[None],wrist_images=image[None],robot_states=example['state'][None],
                camera_intrinsics=np.eye(3)[None],camera_to_world_opencv=np.eye(4)[None])
    verify_example(example,cached,0)
    changed={**example,'state':np.ones(8,np.float32)}
    with pytest.raises(ValueError,match='robot_states'):
        verify_example(changed,cached,0)


def test_live_requires_explicit_execution(tmp_path):
    with pytest.raises(ValueError,match='Explicit'):
        run('missing',tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_simulator_handshake_failure_reaps_owned_child(tmp_path,monkeypatch):
    import benchmarks.LIBERO.search_live as module
    events=[]
    class Listener:
        def __init__(self,*a,**k):
            self._listener=self;self._socket=self
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def settimeout(self,timeout):pass
        def accept(self):raise TimeoutError('synthetic handshake failure')
    class Child:
        def __init__(self,*a,**k):pass
        def poll(self):return None
        def terminate(self):events.append('terminate')
        def wait(self,timeout=None):events.append('wait');return 0
    monkeypatch.setattr(module,'Listener',Listener)
    monkeypatch.setattr(module.subprocess,'Popen',Child)
    config=dict(sim_site='unused',source='unused',sim_extra_paths=[],sim_python='unused')
    with pytest.raises(TimeoutError,match='synthetic'):
        with simulator('unused',config,0,tmp_path):
            pytest.fail('Failed handshake must not yield an active simulator')
    assert events==['terminate','wait']
