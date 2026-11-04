from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from benchmarks.Robocasa_tabletop.task_state import RobosuiteTaskState, ObservationEpoch
from benchmarks.Robocasa_tabletop.operation_io import action_dict
from benchmarks.Robocasa_tabletop.task_runtime import prepare_source_view, TaskSpec,create_task,process_fingerprint


class CompositeController:
    def __init__(self):
        self.part_controllers = {'left':SimpleNamespace(sim=object(),interpolator=None,goal=np.arange(7.))}
        self._applied_action_dict = {'left':np.ones(7)}


class Rng:
    def capture(self): return {'counter':7}
    def restore(self,value): assert value == self.capture()


def fixture():
    cache = ObservationEpoch()
    obs = SimpleNamespace(_time_since_last_sample=.1,_current_delay=0.,
                          _current_observed_value=np.ones(3),_sampled=True)
    env = SimpleNamespace(robots=[SimpleNamespace(composite_controller=CompositeController())],
        _observables={'state':obs},timestep=3,cur_time=.15,done=False,_obs_cache={},_ep_meta={'lang':'fixture'},
        rng=np.random.default_rng(5))
    return env,cache,RobosuiteTaskState(env,cache,Rng())


def test_full_python_snapshot_restores_rng_controller_observation_and_invalidates_cache():
    env,cache,hooks = fixture(); snap=hooks.capture(); expected=env.rng.random()
    env.timestep=99;env.robots[0].composite_controller.part_controllers['left'].goal[:]=20
    env._observables['state']._sampled=False;cache.values['frame']='stale';cache.invalidate()
    hooks.restore(snap)
    assert env.timestep==3 and env.rng.random()==expected
    assert np.array_equal(env.robots[0].composite_controller.part_controllers['left'].goal,np.arange(7.))
    assert env._observables['state']._sampled and cache.epoch==2 and cache.values=={}
    hooks.restore(snap); assert cache.epoch==3


def test_unknown_controller_state_and_incomplete_snapshot_fail():
    env,cache,hooks=fixture(); snap=hooks.capture()
    env.robots[0].composite_controller.part_controllers['left'].hidden=object()
    with pytest.raises(ValueError,match='Unaudited'):hooks.capture()
    del env.robots[0].composite_controller.part_controllers['left'].hidden
    del snap['task']['env']['done']
    with pytest.raises(ValueError,match='Incomplete'):hooks.restore(snap)


def test_absolute_action_partition_roundtrip():
    values=np.arange(29,dtype=np.float32)
    keyed=action_dict(values)
    assert list(map(len,keyed.values()))==[7,7,6,6,3]
    assert np.array_equal(np.concatenate(list(keyed.values())),values)
    keyed['action.left_arm'][0]=99;assert values[0]==0
    with pytest.raises(ValueError):action_dict(np.zeros(32))


def test_source_view_binds_assets_before_import_without_editing_source(tmp_path):
    source=tmp_path/'source';models=source/'robocasa/models';models.mkdir(parents=True)
    (models/'__init__.py').write_text('# fixture')
    (models/'assets').mkdir();(models/'assets/old').write_text('old')
    assets=tmp_path/'assets';assets.mkdir();(assets/'new').write_text('new')
    dest=tmp_path/'runtime'
    prepare_source_view(source,dest,assets)
    assert (dest/'robocasa/models/assets/new').read_text()=='new'
    assert (models/'assets/old').read_text()=='old'
    with pytest.raises(FileExistsError):prepare_source_view(source,dest,assets)


def test_task_parameters_cannot_default_or_clip(tmp_path):
    good=TaskSpec('fixture',0,0,0,str(tmp_path),True,20,1440);good.validate()
    for key,value in [('layout',99),('style',12),('seed',-1),('active_head',1),('control_hz',0)]:
        from dataclasses import replace
        with pytest.raises(ValueError):replace(good,**{key:value}).validate()


def test_native_none_torque_buffer_survives_but_arbitrary_object_rejected():
    env,_,hooks=fixture()
    class DeltaBuffer:
        def __init__(self):
            self.dim=3;self.last=np.zeros(3);self.current=np.array(None)
    env.robots[0].recent_torques=DeltaBuffer()
    snapshot=hooks.capture()
    env.robots[0].recent_torques.current=np.ones(3)
    hooks.restore(snapshot)
    assert env.robots[0].recent_torques.current.shape==()
    assert env.robots[0].recent_torques.current.item() is None
    env.robots[0].recent_torques.current=np.array(object())
    with pytest.raises(ValueError,match='Unaudited'):hooks.capture()


def test_interpreter_hash_mismatch_rejected_before_simulator_import(tmp_path):
    expected=process_fingerprint();expected['hash_probe'][0]+=1
    with pytest.raises(ValueError,match='hash seed'):
        create_task(TaskSpec('no construction',0,0,0,str(tmp_path),True,20,10),
            guard=lambda:None,controller_config={},expected_process_fingerprint=expected)
