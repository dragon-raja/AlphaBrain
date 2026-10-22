from copy import deepcopy
import numpy as np
import pytest
from AlphaBrain.research.dsol.data.local_search_rollouts import episode_targets, complete_update, minibatch_indices


def test_gae_matches_hand_computation_and_no_next_episode_bootstrap():
    adv, ret = episode_targets([1., 2.], [.5, .25], terminal=True)
    np.testing.assert_allclose(adv, [2.4125, 1.75])
    np.testing.assert_allclose(ret, [2.9125, 2.])
    np.testing.assert_allclose(episode_targets([4.], [2.], terminal=True)[1], [4.])
    with pytest.raises(ValueError): episode_targets([1.], [0.], terminal=False)


def test_exact_episode_count_partial_minibatches_and_pre_action_snapshots():
    transition = dict(observation={'action_mask': np.array([1, 0])}, action=0,
                      reward=-.01, value=.2, log_prob=0.)
    episodes = [dict(status='COMPLETE', transitions=[deepcopy(transition)] * (1 + i % 6)) for i in range(32)]
    rows = complete_update(episodes)
    episodes[0]['transitions'][0]['observation']['action_mask'][0] = 0
    assert rows[0]['observation']['action_mask'][0] == 1
    for batches in minibatch_indices(len(rows), seed=0):
        assert sorted(np.concatenate(batches)) == list(range(len(rows)))
        assert len(batches[-1]) == len(rows) % 64
    with pytest.raises(ValueError): complete_update(episodes[:-1])
    episodes[0]['status'] = 'FAILED'
    with pytest.raises(ValueError): complete_update(episodes)


def test_continuous_snapshot_preserves_branch_latent_and_rejects_clipped_action():
    obs = dict(action_mask=np.array([1]),delta_low=np.array([-2.,-10.]),delta_high=np.array([10.,3.]))
    z = np.array([.2,-.4]); delta=(obs['delta_high']+obs['delta_low'])/2+(obs['delta_high']-obs['delta_low'])/2*np.tanh(z)
    row=dict(observation=obs,action=dict(stop=False,latent=z,delta_deg=delta),reward=.1,value=0.,log_prob=-1.)
    episodes=[dict(status='COMPLETE',transitions=[deepcopy(row)]) for _ in range(32)]
    rows=complete_update(episodes);assert rows[0]['action']['stop'] is False
    episodes[0]['transitions'][0]['action']['delta_deg'][0]=10.
    with pytest.raises(ValueError):complete_update(episodes)
