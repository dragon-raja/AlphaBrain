"""Existing score evidence: return exclusion and corrupted-chain rejection."""
import numpy as np
import pytest
from benchmarks.LIBERO.paired_change_evidence import attach


def prepare(root):
    for i,(x,score) in enumerate([(0,1.),(1,1.1),(0,1.)],1):
        pose=np.eye(4);pose[0,3]=x
        np.savez(root/f'query-{i:02d}.npz',score_0=score,score_1=score,camera_to_world_opencv=pose)
    return dict(status='COMPLETE',policy_update=0,
        raw_reward=dict(schema='FORMAL_BATCH_REWARD_V1',initial_mean=1.,initial_std=0.,
                        score_members=2,collector_scale=1.),
        transitions=[dict(best_previous=1.,best_current=1.,delta_queries=1,reward=-.01),
                     dict(best_previous=1.,best_current=1.,delta_queries=1,reward=-.01)])


def test_return_cost_is_preserved_but_not_a_new_pose_statistic(tmp_path):
    episode=prepare(tmp_path);out=attach(episode,tmp_path)
    assert out['paired_changes']==[dict(query=2,previous=[1.,1.],current=[1.1,1.1])]
    assert len(out['paired_query_evidence'])==3 and 'paired_changes' not in episode
    assert sum(t['delta_queries'] for t in out['transitions'])==2


def test_wrong_raw_members_fail_before_reward_settlement(tmp_path):
    episode=prepare(tmp_path)
    np.savez(tmp_path/'query-02.npz',score_0=.8,score_1=.8,camera_to_world_opencv=np.eye(4))
    with pytest.raises(ValueError,match='best-score step'):attach(episode,tmp_path)
