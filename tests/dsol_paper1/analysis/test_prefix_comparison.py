from copy import deepcopy
import pytest
from AlphaBrain.research.dsol.analysis.prefix_comparison import compare_endpoint


def fixture():
    rows=[dict(state_key=str(i),start=0,repeat=0,task='t'+str(i//2),queries=1,success=False) for i in range(8)]
    arms={a:deepcopy(rows) for a in ('stay','p3','full')}
    for i in range(4):arms['full'][i]['success']=True
    return arms,{(r['state_key'],0,0) for r in rows}


def test_paired_rescue_and_task_bootstrap_do_not_pool_repeats():
    arms,expected=fixture();v=compare_endpoint(arms,expected=expected,bootstrap=200)
    assert v['contrasts']['full-minus-stay']['rescues']==4
    assert v['contrasts']['full-minus-stay']['delta']==.5 and v['tasks']==4


@pytest.mark.parametrize('change',['missing','duplicate','task','label'])
def test_no_partial_or_mismatched_claims(change):
    arms,expected=fixture()
    if change=='missing':arms['full'].pop()
    if change=='duplicate':arms['full'].append(arms['full'][0])
    if change=='task':arms['full'][0]['task']='another'
    if change=='label':arms['full'][0]['success']=1
    with pytest.raises(ValueError):compare_endpoint(arms,expected=expected)


@pytest.mark.parametrize('update',[1,3])
def test_g_scale_count_is_cumulative_but_rewards_belong_to_current_batch(update):
    from AlphaBrain.research.dsol.analysis.prefix_comparison import learning_batch
    rows=[dict(episode=i,queries=2,initial_accel=1.,best_accel=.9) for i in range(32*(update-1),32*update)]
    u=dict(update=update,reward_settlement=dict(kind='G',count=32*update,rewards=[.09]*32,denominator=1.,frozen_during_optimizer=True),
           means=dict(approximate_kl=.001,clip_fraction=.1,value_loss=.2))
    got=learning_batch(u,rows)
    assert got['episodes']==32*update and got['mean_reward']==pytest.approx(.09)
    assert got['queries']==2 and got['initial_stop_rate']==0
