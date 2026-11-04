from copy import deepcopy
import pytest
from AlphaBrain.research.dsol.analysis.reward_adaptation import summarize,local_incentive_diagnostic


def example():
    episodes=[];searches=[]
    for arm in ('B','G','L'):
        for seed in range(3):
            for initial in range(2):
                method=f'{arm}-seed-{seed}'
                common=dict(method=method,task=f'task{initial}',state_key=f'initial{initial}',view_id=1)
                for repeat in range(2):
                    episodes.append(dict(common,repeat=repeat,success=arm!='B',stay_success=False,nominal_success=True))
                searches.append(dict(common,queries=2,reason='stop',initial_accel=1.,best_accel=.5,
                    validation_scores=[.7,.8],reference_validation_scores=[1.,1.]))
    return episodes,searches


def test_paired_counts_and_shared_baseline_not_independent_repeats():
    episodes,searches=example();s=summarize(episodes,searches,bootstrap_draws=100)
    assert s['baseline']['unique_perturbed_execution_pairs']==4
    assert s['arms']['G']['success_rate']==1 and s['arms']['G']['rescue']==12
    assert s['contrasts']['G-B']['success_difference_pp']==100
    assert s['contrasts']['G-B']['exploratory_crossed_task_seed_interval_pp']==[100,100]


@pytest.mark.parametrize('fault',['missing','duplicate','baseline','diagnostic'])
def test_no_cherry_picking_or_mixed_baseline(fault):
    episodes,searches=example()
    if fault=='missing':episodes.pop()
    if fault=='duplicate':episodes.append(deepcopy(episodes[0]))
    if fault=='baseline':episodes[0]['stay_success']=True
    if fault=='diagnostic':searches[0]['reference_validation_scores']=[]
    with pytest.raises(ValueError):summarize(episodes,searches,bootstrap_draws=100)


def test_local_incentive_counts_favorable_and_unfavorable_flips():
    episodes=[dict(seed=0,policy_update=0,transitions=[t]) for t in [
        dict(best_previous=.1,best_current=.095,delta_queries=1),
        dict(best_previous=3.,best_current=2.98,delta_queries=1),
        dict(best_previous=.1,best_current=.1,delta_queries=0)]]
    result=local_incentive_diagnostic(episodes,{(0,1):dict(scale=1.,tau=.1)})
    assert result['buckets']['below_0.5S']['G_nonpositive_L_positive']==1
    assert result['buckets']['at_least_2S']['L_nonpositive_G_positive']==1
    assert result['model_calls']==0
