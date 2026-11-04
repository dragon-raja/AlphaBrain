from copy import deepcopy
import pytest
from AlphaBrain.research.dsol.data import confirmed_reward


def sample(gain=.01):
    e=dict(status='COMPLETE',policy_update=0,
        raw_reward=dict(schema='FORMAL_BATCH_REWARD_V1',initial_mean=1.,initial_std=0.,score_members=2,collector_scale=1.),
        transitions=[dict(best_previous=1.,best_current=.99,delta_queries=3,reward=(1.-.99)-.03)],
        paired_changes=[dict(query=2,previous=[1.,1.],current=[.99,.99])])
    e['terminal_confirmation']=dict(schema=confirmed_reward.RECORD,hidden_from_actor=True,queries=4,
        search_noise_hashes=['a'*64,'b'*64],confirmation_noise_hashes=['c'*64,'d'*64],
        initial_input_sha256='e'*64,selected_input_sha256='f'*64,skipped_same_input=False,
        query_numbers=[3,4],initial_scores=[1.,1.],selected_scores=[1.-gain,1.-gain])
    return e


def test_signed_terminal_gain_can_penalize_a_search_that_optimistically_improves():
    batch=[sample(-.01) for _ in range(8)];before=deepcopy(batch)
    out,state,receipt=confirmed_reward.assign_batch(batch,confirmed_reward.config(),confirmed_reward.empty_state(),update=0)
    assert batch==before and state['batches']==1
    assert receipt['denominator']==pytest.approx(.01) and out[0]['transitions'][0]['reward']==pytest.approx(-1.03)


def test_held_out_scores_never_set_online_scale():
    scales=[]
    for gain in (-.5,.5):
        _,_,receipt=confirmed_reward.assign_batch([sample(gain)]*8,confirmed_reward.config(),confirmed_reward.empty_state(),update=0)
        scales.append(receipt['denominator'])
    assert scales==pytest.approx([.01,.01])


@pytest.mark.parametrize('fault',['shared_noise','wrong_count','missing_query','visible'])
def test_bad_confirmation_contract_rejected(fault):
    e=sample();c=e['terminal_confirmation']
    if fault=='shared_noise':c['confirmation_noise_hashes'][0]=c['search_noise_hashes'][0]
    if fault=='wrong_count':c['queries']=3
    if fault=='missing_query':c['query_numbers']=[2,3]
    if fault=='visible':c['hidden_from_actor']=False
    with pytest.raises(ValueError):confirmed_reward.validate_confirmation(e)
