"""Candidate gain units: invariant contrasts, noise floors and no STOP shrink."""
from copy import deepcopy

import numpy as np
import pytest

from AlphaBrain.research.dsol.data.change_reward import assign_batch, default_config, empty_state


def episode(factor=1., offset=0., update=0, moving=True):
    before = 1.*factor + offset
    after = .99*factor + offset if moving else before
    episode = dict(status='COMPLETE', policy_update=update,
        raw_reward=dict(schema='FORMAL_BATCH_REWARD_V1', initial_mean=before,
                        initial_std=0., score_members=2, collector_scale=1.),
        transitions=[dict(best_previous=before, best_current=after, delta_queries=int(moving),
                          reward=before-after-.01*int(moving))], paired_changes=[])
    if moving:
        episode['paired_changes'] = [dict(query=2, previous=[before,before],current=[after,after])]
    return episode


def test_signed_contrasts_units_and_additive_floor_cancel():
    rewards=[]
    for factor,offset in [(1.,0.),(100.,50.),(.0001,0.)]:
        batch=[episode(factor,offset) for _ in range(8)];old=deepcopy(batch)
        out,state,receipt=assign_batch(batch,default_config(),empty_state(),update=0)
        assert batch==old and receipt['additional_model_calls']==0
        rewards.append(out[0]['transitions'][0]['reward'])
        assert state['batches']==1 and receipt['window_pairs']==8
    assert rewards==pytest.approx([.99]*3)


def test_stop_batch_never_becomes_zero_contrast_and_cannot_resettle():
    _,state,_=assign_batch([episode() for _ in range(8)],default_config(),empty_state(),update=0)
    _,next_state,receipt=assign_batch([episode(update=1,moving=False) for _ in range(8)],default_config(),state,update=1)
    assert receipt['denominator']==state['scale'] and receipt['new_pairs']==0
    assert next_state['changes']==state['changes']
    with pytest.raises(ValueError,match='boundary'):
        assign_batch([episode(update=1) for _ in range(8)],default_config(),next_state,update=1)


def test_pair_disagreement_floor_and_limited_update():
    batch=[episode() for _ in range(8)]
    for e in batch:e['paired_changes'][0]['current']=[.8,1.2]
    _,state,receipt=assign_batch(batch,default_config(),empty_state(),update=0)
    assert receipt['denominator']==pytest.approx(.4)
    _,_,receipt=assign_batch([episode(factor=1e-8,update=1) for _ in range(8)],default_config(),state,update=1)
    assert .2<=receipt['denominator']<=.8


def test_insufficient_evidence_cold_start_and_flat_zero_scale():
    _,_,receipt=assign_batch([episode()],default_config(),empty_state(),update=0)
    assert receipt['denominator']==1. and 'FALLBACK' in receipt['status']
    with pytest.raises(ValueError,match='positive'):
        assign_batch([episode(factor=0,moving=False)],default_config(),empty_state(),update=0)


@pytest.mark.parametrize('fault',['negative','nan','duplicate','missing'])
def test_invalid_pair_evidence_fails_closed(fault):
    e=episode()
    if fault=='negative':e['paired_changes'][0]['current'][0]=-1
    if fault=='nan':e['paired_changes'][0]['current'][0]=np.nan
    if fault=='duplicate':e['paired_changes']*=2
    if fault=='missing':del e['paired_changes']
    with pytest.raises((ValueError,KeyError)):
        assign_batch([e]*8,default_config(),empty_state(),update=0)
