import pytest

from AlphaBrain.research.dsol.analysis.search_live_summary import summarize_live


def receipt():
    return dict(status='COMPLETE',tasks=[dict(initial=i,method=m,repeat=s,success=s<2,
        historical_same_seed_success=s>=2,queries=1 if m=='stay' else 8,selected=65,steps=300)
        for i in (0,4) for m in ('stay','local_accel','ppo') for s in range(4)])


def test_matched_live_and_history_keep_disagreements():
    result=summarize_live(receipt())
    for row in result['rows']:
        assert row['live_successes']==row['historical_successes']==4
        assert row['agreement']==0
        assert row['historical_fail_live_pass']==row['historical_pass_live_fail']==4
    assert result['rows'][0]['mean_queries']==1


@pytest.mark.parametrize('mutation',['incomplete','missing','duplicate','label','view'])
def test_rejects_incomplete_or_mismatched_pairing(mutation):
    r=receipt()
    if mutation=='incomplete':r['status']='FAILED_STOPPED'
    elif mutation=='missing':r['tasks'].pop()
    elif mutation=='duplicate':r['tasks'][-1]=r['tasks'][0].copy()
    elif mutation=='label':r['tasks'][0].pop('historical_same_seed_success')
    elif mutation=='view':r['tasks'][0]['selected']=2
    with pytest.raises(ValueError):summarize_live(r)
