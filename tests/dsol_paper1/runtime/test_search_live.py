import numpy as np
import pytest

from benchmarks.LIBERO.search_live import run, verify_example, verify_prediction, simulator, simulator_environment


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


def test_late_denoising_drift_is_rejected_even_if_early_score_matches():
    reference=dict(trace=np.zeros((2,2,10,10,7),np.float32),actions=np.zeros((2,2,10,7),np.float32))
    value={k:reference[k][1,0].copy() for k in reference}
    verify_prediction(value,reference,1,0)
    value['trace'][-1,0,0]=.01
    with pytest.raises(ValueError,match='trace'):
        verify_prediction(value,reference,1,0)


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
    times=iter((0,181))
    monkeypatch.setattr(module.time,'monotonic',lambda:next(times))
    config=dict(sim_site='unused',source='unused',sim_extra_paths=[],sim_python='unused',
                sim_library_paths=[str(tmp_path)])
    with pytest.raises(TimeoutError,match='deadline'):
        with simulator('unused',config,0,tmp_path):
            pytest.fail('Failed handshake must not yield an active simulator')
    assert events==['terminate','wait']


def test_recovered_libraries_are_explicit_and_child_only(tmp_path,monkeypatch):
    import os
    monkeypatch.setenv('LD_LIBRARY_PATH','inherited')
    config=dict(sim_site='site',source='source',sim_extra_paths=[])
    with pytest.raises(ValueError,match='sim_library_paths'):
        simulator_environment(config)
    config['sim_library_paths']=[str(tmp_path)]
    env=simulator_environment(config)
    assert env['LD_LIBRARY_PATH']==str(tmp_path)+':inherited'
    assert os.environ['LD_LIBRARY_PATH']=='inherited'


def test_dependency_check_does_not_start_simulator(tmp_path,monkeypatch):
    import benchmarks.LIBERO.search_live as module
    seen=[]
    def probe(command,**kwargs):
        seen.append((command,kwargs))
        return type('Result',(),{'stdout':'libMagickWand-6.Q16.so.6\n'})()
    monkeypatch.setattr(module.subprocess,'run',probe)
    config=dict(sim_site='site',source='source',sim_extra_paths=[],sim_python='python',
                sim_library_paths=[str(tmp_path)])
    result=module.check_simulator_dependencies(config)
    assert result['status']=='PASS' and not result['simulator_started']
    assert seen[0][0]==['python','-c','from wand.api import library; print(library._name)']
    assert seen[0][1]['check'] and seen[0][1]['timeout']==60


def test_exited_simulator_reports_log_instead_of_waiting_for_timeout(tmp_path,monkeypatch):
    import benchmarks.LIBERO.search_live as module
    class Listener:
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def accept(self):pytest.fail('Exited child must be detected before accepting')
    class Child:
        def __init__(self,*args,**kwargs):pass
        def poll(self):return 1
    monkeypatch.setattr(module,'Listener',Listener)
    monkeypatch.setattr(module.subprocess,'Popen',Child)
    config=dict(sim_site='site',source='source',sim_extra_paths=[],sim_python='python',
                sim_library_paths=[str(tmp_path)])
    with pytest.raises(RuntimeError,match='simulator.log'):
        with simulator('unused',config,0,tmp_path):
            pytest.fail('Exited child cannot yield an active simulator')
