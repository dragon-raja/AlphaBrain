import pytest
from AlphaBrain.research.dsol.analysis.search_task_alignment import task_rows,paired_cluster_interval,cutoff_pair


def rows():
    return [dict(cohort='B',task='one',initial=i,start=65,repeat=0,method=m,
                 success=(m!='stay'),queries=1 if m=='stay' else 4,replay_success=None)
            for i in (0,1) for m in ('stay','local_accel','ppo')]


def test_missing_replay_not_failure_and_paired_rescue():
    table=task_rows(rows());ppo=next(r for r in table if r['method']=='ppo')
    assert ppo['replay_n']==0 and ppo['replay_success_rate'] is None
    assert ppo['rescue']==2 and ppo['harm']==0
    assert ppo['initial_seed_positive']==2
    assert paired_cluster_interval(rows(),reference='stay',repeats=10)['interval95']==[1.,1.]


def test_incomplete_or_duplicate_pairing_rejected():
    for data in (rows()[:-1],rows()+rows()[:1]):
        with pytest.raises(ValueError):task_rows(data)


def test_replay_disagreements_keep_both_directions():
    data=rows()
    for r in data:r['replay_success']=not r['success']
    table=task_rows(data)
    assert sum(r['replay_pass_live_fail'] for r in table)==2
    assert sum(r['replay_fail_live_pass'] for r in table)==4


def test_cutoff_is_symmetric_and_missing_remains_missing():
    row=dict(success=True,steps=250,replay_success=True,replay_completion_steps=310)
    assert cutoff_pair(row,300)['success'] is True
    assert cutoff_pair(row,300)['replay_success'] is False
    assert cutoff_pair(row,220)['success'] is False
    assert row['success'] and row['replay_success']
    row['replay_success']=None
    assert cutoff_pair(row,220)['replay_success'] is None
