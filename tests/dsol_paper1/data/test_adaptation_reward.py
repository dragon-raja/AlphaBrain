"""Small-sample scale, provenance, window and restart behavior on CPU."""
from copy import deepcopy
import json

import pytest

from AlphaBrain.research.dsol.data.adaptation_reward import assign_batch, empty_state, validate_state
from tests.dsol_paper1.helpers.adaptation_reward import config, batch


def test_first_formal_batch_is_used_without_extra_queries_or_mutation():
    cfg=config(); raw=batch(); original=deepcopy(raw); state=empty_state(cfg)
    out,nxt,r=assign_batch(raw,cfg,state,update=0)
    assert raw==original and state==empty_state(cfg)
    assert r['additional_model_calls']==0 and r['denominator']==1.
    assert nxt['count']==8 and len(nxt['initials'])==8
    assert out[0]['transitions'][0]['reward']==pytest.approx(.18)


def test_same_first_batch_then_explicit_fixed_vs_window_divergence():
    results={}
    for mode in ('first_batch_fixed','window'):
        cfg=config(mode); state=empty_state(cfg); scales=[]
        for update in range(6):
            _,state,r=assign_batch(batch(mean=1. if update==0 else 10.,update=update),cfg,state,update=update)
            scales.append(r['denominator'])
        results[mode]=scales
        assert state['count']==48 and len(state['initials'])==32
        assert state['initials']==[10.]*32
    assert results['first_batch_fixed']==[1.]*6
    assert results['window']==[1.,2.,4.,8.,10.,10.]


@pytest.mark.parametrize('factor',[1e-8,.01,1.,1e6])
def test_global_score_units_preserve_rewards_including_disagreement(factor):
    cfg=config()
    out,_,r=assign_batch(batch(mean=factor,std=.8*factor),cfg,empty_state(cfg),update=0)
    assert r['disagreement_floor_active']
    assert r['denominator']==pytest.approx(1.6*factor)
    assert out[0]['transitions'][0]['reward']==pytest.approx(.125-.02)


def test_resume_serialization_is_exact_and_replay_rejected():
    cfg=config(); _,state,_=assign_batch(batch(),cfg,empty_state(cfg),update=0)
    expected=assign_batch(batch(mean=.02,update=1),cfg,state,update=1)
    restored=json.loads(json.dumps(state))
    assert assign_batch(batch(mean=.02,update=1),cfg,restored,update=1)==expected
    with pytest.raises(ValueError,match='boundary'):
        assign_batch(batch(update=1),cfg,expected[1],update=1)


def test_new_domain_cannot_inherit_scale_or_old_window():
    cfg=config(); _,state,_=assign_batch(batch(mean=100.),cfg,empty_state(cfg),update=0)
    other=config(domain='b')
    with pytest.raises(ValueError,match='boundary'):
        assign_batch(batch(update=1),other,state,update=1)
    _,new,r=assign_batch(batch(mean=.01,std=0.),other,empty_state(other),update=0)
    assert r['denominator']==.01 and new['count']==8


@pytest.mark.parametrize('fault',['incomplete','version','failed','nan','corrupt_reward','zero','bad_saved_count'])
def test_invalid_or_unidentifiable_batch_fails_without_mutation(fault):
    cfg=config(); raw=batch(); state=empty_state(cfg)
    if fault=='incomplete': raw.pop()
    if fault=='version': raw[0]['policy_update']=1
    if fault=='failed': raw[0]['status']='FAILED'
    if fault=='nan': raw[0]['raw_reward']['initial_mean']=float('nan')
    if fault=='corrupt_reward': raw[0]['transitions'][0]['reward']+=1
    if fault=='zero': raw=batch(mean=0.,std=0.)
    if fault=='bad_saved_count': state['count']=8
    original=deepcopy(state)
    with pytest.raises(ValueError): assign_batch(raw,cfg,state,update=0)
    assert state==original


def test_candidate_minima_do_not_drive_the_denominator():
    cfg=config(); raw=batch()
    raw[0]['transitions'][0].update(best_current=1e-30,reward=1.-1e-30-.02)
    _,_,r=assign_batch(raw,cfg,empty_state(cfg),update=0)
    assert r['denominator']==1.


def test_saved_zero_scale_and_changed_contract_rejected():
    cfg=config(); _,state,_=assign_batch(batch(),cfg,empty_state(cfg),update=0)
    changed=config('first_batch_fixed')
    with pytest.raises(ValueError): validate_state(state,changed,update=1,count=8)
    state['scale']=0.
    with pytest.raises(ValueError): validate_state(state,cfg,update=1,count=8)
