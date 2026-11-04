"""Actual search/Gym lifecycle with a synthetic renderer, no model or GPU."""
from dataclasses import replace
import numpy as np
import pytest
pytest.importorskip('gymnasium')
from AlphaBrain.research.dsol.acquisition.continuous_search import ContinuousSearch
from tests.dsol_paper1.helpers.local_search import synthetic_session
from benchmarks.LIBERO.terminal_confirmation import ConfirmedEnv,finish_and_confirm


def session(fail_confirmation=False):
    from AlphaBrain.research.dsol.acquisition.continuous_search_gym import ContinuousSearchGym
    old,source=synthetic_session();original=source.query
    source.noises=np.stack([np.zeros((10,7)),np.ones((10,7))]);source.references={};source.queries=0;source.failed=False
    def query():
        source.queries+=1
        if fail_confirmation and source.noises[0,0,0]==2:raise RuntimeError('Unknown confirmation outcome')
        value=original();score=3.-value.value.accel_mean
        example=dict(pose=source.pose.copy());source.last_example=example
        source.last_values=[dict(score=score)]*2
        source.references.setdefault(tuple(source.pose.ravel()),(example,source.last_values))
        return replace(value,value=replace(value.value,accel_mean=score))
    source.query=query
    core=ContinuousSearch(old.mount,source,initial_angles=[0,0],physical_state_id='synthetic',wrist=np.ones(4),score_scale=1.,seed=19)
    env=ConfirmedEnv(ContinuousSearchGym(4),np.stack([np.full((10,7),2.),np.full((10,7),3.)]));env.reset(options={'session':core})
    return env,source


def test_stop_at_same_input_is_zero_gain_without_fabricated_queries():
    env,s=session();_,r,done,_,info=env.step(dict(stop=1,delta_deg=np.zeros(2)))
    assert done and r==0 and info['queries']==s.queries==1
    assert env.terminal_confirmation['skipped_same_input'] and not env.terminal_confirmation['query_numbers']


def test_real_entry_reserves_hidden_queries_preserves_best_and_observation():
    env,s=session()
    for i in range(4):obs,r,done,_,info=env.step(dict(stop=0,delta_deg=np.array([1.,0.])))
    assert done and info['queries']==s.queries==8 # final return-to-best plus two confirmations
    assert r==pytest.approx(-.04) and len(env.core.records)==5
    assert env.terminal_confirmation['query_numbers']==[7,8]
    assert env.core.current.tolist()==[1.,0.] and env.env.observation_space.contains(obs)
    assert s.noises[:,0,0].tolist()==[0.,1.]
    assert len(s.references)==5 and not env.terminal_confirmation['skipped_same_input']
    assert finish_and_confirm(env.core,env.noises)==env.terminal_confirmation and s.queries==8


def test_confirmation_failure_is_charged_latched_and_never_retried():
    env,s=session(True);env.step(dict(stop=0,delta_deg=np.array([1.,0.])))
    with pytest.raises(RuntimeError,match='Unknown'):env.step(dict(stop=1,delta_deg=np.zeros(2)))
    assert env.core.status=='FAILED' and s.failed and env.core.queries==s.queries==3
    before=s.queries
    with pytest.raises(ValueError,match='explicit audit'):finish_and_confirm(env.core,env.noises)
    assert before==s.queries and s.noises[:,0,0].tolist()==[0.,1.]


def test_search_noise_cannot_be_reused_as_confirmation():
    env,s=session();env.noises=s.noises.copy()
    with pytest.raises(ValueError,match='overlap'):env.step(dict(stop=1,delta_deg=np.zeros(2)))

