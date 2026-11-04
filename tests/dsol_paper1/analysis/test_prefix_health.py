import pytest
from AlphaBrain.research.dsol.analysis.prefix_health import complete_three_seed_tasks, paired_diagnostic, training_health, development_progress


def row(seed, task, *, success=True, stay=False):
    return dict(seed=seed,task=task,suite='suite',state_key=task,view_id=1,repeat=0,
                success=success,stay_success=stay,queries=8)


def test_partial_third_seed_is_not_completed_task_or_failure():
    expected=[row(s,t) for s in range(3) for t in ['a','b']]
    observed=[row(s,'a') for s in range(3)]+[row(0,'b',success=False)]
    kept,tasks=complete_three_seed_tasks(observed,expected)
    assert len(kept)==3 and tasks==[('suite','a')]
    summary=paired_diagnostic(kept)
    assert summary['policy_success']==1 and summary['net_pp']==100
    assert summary['exploratory_task_cluster_interval_pp'] is None


def test_rescue_and_harm_are_both_retained_and_missing_reference_not_zero():
    summary=paired_diagnostic([row(0,'a'),row(0,'b',success=False,stay=True)])
    assert summary['rescues']==summary['harms']==1 and summary['net_pp']==0
    assert summary['reference_success'] is None and summary['p3_success'] is None


def test_duplicate_seed_trial_or_unknown_task_rejected():
    r=row(0,'a')
    with pytest.raises(ValueError):paired_diagnostic([r,r])
    with pytest.raises(ValueError):complete_three_seed_tasks([r],[row(0,'b')])


def test_optimizer_reward_is_not_provisional_collector_reward():
    episodes=[dict(episode=i,queries=2,scoring_trajectories=4,initial_accel=1.,best_accel=.8,
                   total_reward=999.) for i in range(2048)]
    updates=[dict(update=i,means=dict(approximate_kl=.01,clip_fraction=.1,value_loss=.2,gradient_norm=1.),
        reward_settlement=dict(kind='G',count=32*i,frozen_during_optimizer=True,rewards=[.19]*32,denominator=1.))
        for i in range(1,65)]
    health=training_health(updates,episodes)
    assert health['windows']['last8']['mean_reward']==pytest.approx(.19)
    assert health['points'][0]['mean_initial_accel']==pytest.approx(1.)
    assert health['points'][0]['mean_best_accel']==pytest.approx(.8)
    assert (health['points'][0]['mean_initial_accel']-health['points'][0]['mean_best_accel']
            ==pytest.approx(health['points'][0]['mean_raw_improvement']))
    updates[-1]['reward_settlement']['rewards'][0]=999.
    with pytest.raises(ValueError):training_health(updates,episodes)


def test_development_curve_requires_the_same_full_panel():
    rows=[dict(seed=seed,checkpoint_version=version,task=str(i//8),state_key=str(i),
               view_id=1,repeat=0,queries=8,success=(version==32 and i<32))
          for seed in range(3) for version in (16,32) for i in range(64)]
    result=development_progress(rows)
    assert result['0']['rescues']==32 and result['0']['harms']==0
    assert result['0']['points'][1]['success_rate']==.5
    with pytest.raises(ValueError):development_progress(rows[:-1])


def test_short_screen_endpoint_is_explicit_not_padded_or_convergence():
    episodes=[dict(episode=i,queries=2,scoring_trajectories=4,initial_accel=1.,best_accel=.8,total_reward=999.) for i in range(512)]
    updates=[dict(update=i,means=dict(approximate_kl=.01,clip_fraction=.1,value_loss=.2,gradient_norm=1.),reward_settlement=dict(kind='G',count=32*i,frozen_during_optimizer=True,rewards=[.19]*32,denominator=1.)) for i in range(1,17)]
    with pytest.raises(ValueError):training_health(updates,episodes)
    out=training_health(updates,episodes,expected_updates=16)
    assert len(out['points'])==16 and out['points'][-1]['episodes']==512
    with pytest.raises(ValueError):training_health(updates[:-1],episodes,expected_updates=16)
