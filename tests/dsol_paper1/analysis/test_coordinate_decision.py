import pytest
from AlphaBrain.research.dsol.analysis.coordinate_validation import summarize
from AlphaBrain.research.dsol.data.coordinate_validation import CRITERIA


def paired_rows():
    return [dict(state_key=f'{task}-{initial}',task=str(task),offset=offset,start=start,seed=seed,repeat=repeat,
        G=True,stay=False,home=False,local=True,reference=True,G_queries=4,local_queries=8)
        for task in range(12) for initial in (40,44,48) for offset in (-15.,0.,15.)
        for start in range(4) for seed in range(3) for repeat in (0,1)]


def test_all_gates_and_task_clusters():
    result=summarize(paired_rows(),criteria=CRITERIA)
    assert result['decision']=='READY_G_MAIN_WITHIN_VALIDATED_SCOPE'
    assert result['contrasts']['home']['paired_task_95_interval_pp']==[100.,100.]
    assert len(result['per_task'])==12


def test_matching_home_is_not_shortcut_clearance():
    rows=paired_rows()
    for row in rows:row['home']=True
    result=summarize(rows,criteria=CRITERIA)
    assert result['decision']=='HOLD_G_MAIN' and not result['gates']['beats_home']


def test_success_alone_does_not_establish_ppo_cost_effectiveness():
    rows=paired_rows()
    for row in rows:row['G_queries']=8;row['local_queries']=4
    result=summarize(rows,criteria=CRITERIA)
    assert result['decision']=='ACTIVE_SEARCH_ONLY_PPO_NOT_ESTABLISHED'


def test_missing_duplicate_and_unpaired_controls_rejected():
    rows=paired_rows()
    with pytest.raises(ValueError):summarize(rows[:-1],criteria=CRITERIA)
    with pytest.raises(ValueError):summarize(rows[:-1]+rows[:1],criteria=CRITERIA)
    rows[0]['stay']=True
    with pytest.raises(ValueError,match='baselines'):summarize(rows,criteria=CRITERIA)
