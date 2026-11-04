import pytest
from AlphaBrain.research.dsol.analysis.target_adaptation import summarize


def test_all_seeds_clustered_within_task_and_damage_preserved():
    rows=[]
    for t in range(8):
        base=dict(state_key=str(t),task=str(t),start=15,repeat=0,queries=1)
        rows.append(dict(base,method='stay',success=t>=4))
        for s in (0,1,2):rows.append(dict(base,method=f'actor:source-window-seed-{s}:u8',success=True))
    result=summarize(rows,expected=len(rows),bootstrap_samples=1000)
    x=result['seed_aggregates']['source-window:u8']
    assert x['tasks']==8 and x['paired_rows']==24 and x['net_rescue']==.5 and x['damaged']==0
    assert result['decision']=='DEVELOPMENT_RESCUE_TARGET_MET'
    with pytest.raises(ValueError):summarize(rows+rows[:1])


def test_partial_cannot_promote():
    rows=[dict(state_key='a',task='task',start=0,repeat=0,queries=0,method='stay',success=True)]
    result=summarize(rows)
    assert result['status']=='PARTIAL' and result['decision']=='PENDING_COMPLETE_PANEL'
