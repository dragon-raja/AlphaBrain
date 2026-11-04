"""Scale, cost, noise-floor and checkpoint-boundary tests; no model calls."""
from copy import deepcopy
import math
import pytest
from AlphaBrain.research.dsol.data.online_reward import assign_batch, empty_state, validate_raw
from AlphaBrain.research.dsol.data.reward_pilot import reward_config


def episode(mean=1., std=.01, update=0, factor=1.):
    pairs=[(mean,mean*.7,1),(mean*.7,mean*.3,2),(mean*.3,mean*.3,0)]
    scale=.01214765438717935
    return dict(status='COMPLETE',policy_update=update,
        raw_reward=dict(schema='FORMAL_BATCH_REWARD_V1',initial_mean=mean*factor,
            initial_std=std*factor,score_members=2,collector_scale=scale),
        transitions=[dict(best_previous=a*factor,best_current=b*factor,delta_queries=q,
            reward=(a*factor-b*factor)/scale-.01*q) for a,b,q in pairs])


@pytest.mark.parametrize('kind',['B','G','L'])
def test_immutable_raw_cost_and_telescoping(kind):
    batch=[episode()];old=deepcopy(batch)
    output,state,receipt=assign_batch(batch,reward_config(kind),empty_state(),update=0)
    assert batch==old and state['batches']==1 and receipt['additional_model_calls']==0
    total=sum(t['reward'] for t in output[0]['transitions'])
    expected=math.log(1.1/.4) if kind=='L' else .7/(reward_config(kind)['fixed_scale'] if kind=='B' else 1.)
    assert total==pytest.approx(expected-.03)
    if kind=='B':assert output==batch


@pytest.mark.parametrize('kind',['G','L'])
def test_model_unit_scaling_invariance(kind):
    rewards=[]
    for factor in (1e-8,1.,1e6):
        out,_,r=assign_batch([episode(factor=factor)],reward_config(kind),empty_state(),update=0)
        rewards.append([t['reward'] for t in out[0]['transitions']])
    for row in rewards:assert row==pytest.approx(rewards[0])


def test_cumulative_initial_only_noise_floor_and_no_double_count():
    _,state,_=assign_batch([episode(mean=1.,std=.5)],reward_config('L'),empty_state(),update=0)
    _,state2,r=assign_batch([episode(mean=.1,std=.01,update=1)],reward_config('L'),state,update=1)
    assert r['count']==2 and r['scale']==.55 and r['tau']==pytest.approx(.3775)
    assert state['initials']==[1.] and state2['initials']==[1.,.1]
    with pytest.raises(ValueError,match='boundary'):
        assign_batch([episode(update=1)],reward_config('L'),state2,update=1)


@pytest.mark.parametrize('fault',['reward','query','chain','noise','nan'])
def test_corrupt_raw_fails_closed(fault):
    e=episode()
    if fault=='reward':e['transitions'][0]['reward']+=.1
    if fault=='query':e['transitions'][0]['delta_queries']=3
    if fault=='chain':e['transitions'][1]['best_previous']=.8
    if fault=='noise':e['raw_reward']['initial_std']=2.
    if fault=='nan':e['raw_reward']['initial_mean']=float('nan')
    with pytest.raises(ValueError):validate_raw(e)


def test_undefined_scale_and_flat_landscape():
    e=episode(mean=0.,std=0.)
    with pytest.raises(ValueError,match='positive scale'):
        assign_batch([e],reward_config('G'),empty_state(),update=0)
    e=episode()
    for t in e['transitions']:
        t.update(best_previous=1.,best_current=1.,reward=-.01*t['delta_queries'])
    out,_,_=assign_batch([e],reward_config('L'),empty_state(),update=0)
    assert sum(t['reward'] for t in out[0]['transitions'])==-.03
