import pytest
from AlphaBrain.research.dsol.analysis.continuous_stop_audit import training_summary, paired_recovery


def test_recovery_excludes_nominal_failures_and_counts_collateral_harm():
    flags = [(True,False,True),(True,True,False),(False,False,True),(False,True,True)]
    rows = [dict(state_key='s',repeat=i//2,start=i%2,nominal=n,stay=p,selected=m)
            for i,(n,p,m) in enumerate(flags)]
    result = paired_recovery(rows)
    assert result['unique_nominal_references']==2
    assert result['nominal_success_lost_to_perturbation']==1
    assert result['conditional_recovery_fraction']==1
    assert result['all_rescues']==2
    assert result['net_recovery_on_nominal_success']==0
    with pytest.raises(ValueError,match='duplicate'): paired_recovery(rows+rows[:1])


def test_no_perturbation_damage_has_no_defined_recovery_fraction():
    row=dict(state_key='s',repeat=0,start=0,nominal=True,stay=True,selected=True)
    assert paired_recovery([row])['conditional_recovery_fraction'] is None


def record(i, initial, best, queries):
    reward=(initial-best)/.2-.01*(queries-1)
    return dict(metrics=dict(episode=i,policy_update=0,initial_accel=initial,best_accel=best,
                queries=queries,reason='stop',total_reward=reward,
                total_reward_including_initial=reward-.01,initial_identity=dict(task='task')))


def test_initial_stop_and_return_query_cost_are_distinct():
    result=training_summary([record(0,.1,.1,1),record(1,.1,.099,3)],scale=.2,cost=.01)
    assert result['all']['immediate_stop']==1
    assert result['continued']['return_after_initial']['mean']==pytest.approx(-.015)
    assert result['hindsight_cost_sensitivity']['0.0']['positive']==1
    assert result['hindsight_cost_sensitivity']['0.01']['positive']==0


def test_reward_corruption_and_duplicate_episodes_rejected():
    x=record(0,.1,.1,1)
    with pytest.raises(ValueError,match='Duplicate'):training_summary([x,x],scale=.2,cost=.01)
    x['metrics']['total_reward']=1
    with pytest.raises(ValueError,match='formula'):training_summary([x],scale=.2,cost=.01)
