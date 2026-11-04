import numpy as np
import pytest
from AlphaBrain.research.dsol.acquisition.continuous_search import ContinuousSearch
from tests.dsol_paper1.helpers.local_search import synthetic_session


def session():
    old,source=synthetic_session()
    return ContinuousSearch(old.mount,source,initial_angles=[0,0],physical_state_id='synthetic',wrist=np.ones(4),score_scale=1.,seed=19),source


def test_off_grid_action_stop_return_cost_and_no_candidates():
    c,s=session();o=c.start();assert len(o['geometry'])==1
    o,r,d,_,_=c.step(dict(stop=False,delta_deg=[3.125,-4.375]))
    assert not d and r==pytest.approx(-.01) and len(o['geometry'])==2
    _,r,d,_,i=c.step(dict(stop=True,delta_deg=[0,0]))
    assert d and r==pytest.approx(-.01) and i['queries']==s.calls==3
    assert np.array_equal(c.current,[0,0])


def test_six_decisions_eight_queries_return_and_no_free_zero_moves():
    c,s=session();c.start()
    for _ in range(6):o,r,d,_,i=c.step(dict(stop=False,delta_deg=[1.125,0.]))
    assert d and c.decision==6 and i['queries']==8 and c.reason=='budget'
    c,s=session();c.start();c.step(dict(stop=False,delta_deg=[0,0]))
    assert s.calls==2 and c.queries==2


def test_bounds_failures_and_gym_without_implicit_resets():
    c,s=session();c.start()
    for action in [dict(stop=False,delta_deg=[10.1,0]),dict(stop=True,delta_deg=[1,0])]:
        with pytest.raises(ValueError):c.step(action)
    assert s.calls==1
    s.fail=True
    with pytest.raises(RuntimeError):c.step(dict(stop=False,delta_deg=[1,0]))
    assert c.status=='FAILED' and c.queries==2
    with pytest.raises(ValueError):c.step(dict(stop=True,delta_deg=[0,0]))
    pytest.importorskip('gymnasium')
    from AlphaBrain.research.dsol.acquisition.continuous_search_gym import ContinuousSearchGym
    c,s=session();e=ContinuousSearchGym(4);o,i=e.reset(options={'session':c})
    assert e.observation_space.contains(o)
    o,r,d,_,i=e.step(dict(stop=0,delta_deg=np.array([2.3,-1.2])))
    assert e.observation_space.contains(o) and not d
    with pytest.raises(ValueError):e.reset(options={'session':session()[0]})


def test_head_backend_refuses_missing_calibration():
    from AlphaBrain.research.dsol.acquisition.continuous_search import validate_head_calibration
    with pytest.raises(ValueError):
        validate_head_calibration(world_from_base=np.eye(4),head_from_camera=None,intrinsics=np.eye(3))
    with pytest.raises(ValueError):
        validate_head_calibration(world_from_base=np.eye(4),head_from_camera=np.eye(4),intrinsics=None)


def test_reward_calibration_changes_only_gain_units_not_observations_or_costs():
    from dataclasses import replace
    sessions=[]
    for reward_scale in (1., .1):
        old,source=synthetic_session()
        original=source.query
        def query(original=original):
            measurement=original()
            return replace(measurement,value=replace(measurement.value,
                accel_mean=3.-measurement.value.accel_mean))
        source.query=query
        core=ContinuousSearch(old.mount,source,initial_angles=[0,0],physical_state_id='synthetic',
            wrist=np.ones(4),score_scale=1.,reward_scale=reward_scale,seed=19)
        core.start();sessions.append(core)
    outputs=[c.step(dict(stop=False,delta_deg=[3.,0])) for c in sessions]
    for key in outputs[0][0]:np.testing.assert_array_equal(outputs[0][0][key],outputs[1][0][key])
    assert outputs[0][1]==pytest.approx(1.-.01)
    assert outputs[1][1]==pytest.approx(10.-.01)
    for c in sessions:
        _,cost,_,_,info=c.step(dict(stop=False,delta_deg=[1.,0]))
        assert cost==pytest.approx(-.01)
        _,cost,done,_,info=c.step(dict(stop=True,delta_deg=[0.,0]))
        assert done and cost==pytest.approx(-.01) and info['queries']==4


@pytest.mark.parametrize('scale',[0,-1,float('nan'),float('inf'),True])
def test_invalid_reward_scale_rejected(scale):
    old,source=synthetic_session()
    with pytest.raises(ValueError):
        ContinuousSearch(old.mount,source,initial_angles=[0,0],physical_state_id='synthetic',
            wrist=np.ones(4),score_scale=1.,reward_scale=scale,seed=19)
