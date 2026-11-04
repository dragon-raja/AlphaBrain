import pytest
from AlphaBrain.research.dsol.analysis.selection_screen import summarize


def fixture():
    return [dict(state_key=f'native::suite::task{t}::{i}',variant=v,repeat=m,
                 success=v in ('reference','minimum-p3'))
            for t in range(2) for i in range(2) for v in ('reference','stay-23','stay-24','minimum-p3') for m in (0,1)]


def test_reused_selected_endpoints_do_not_inflate_unique_count():
    value=summarize(fixture(),expected_states=4,expected_tasks=2)
    assert value['unique_operations']==32 and value['selected_unique_operations']==8
    assert value['paired_comparisons']==16 and value['rescued']==16 and value['harmed']==0
    assert value['net_pp']==100 and value['task_cluster_interval_pp']==[100,100]


def test_missing_or_duplicate_outcomes_are_not_partial_results():
    rows=fixture()
    with pytest.raises(ValueError):summarize(rows[:-1],expected_states=4,expected_tasks=2)
    with pytest.raises(ValueError):summarize(rows+[rows[0]],expected_states=4,expected_tasks=2)
    with pytest.raises(ValueError):summarize(rows,expected_states=4,expected_tasks=1)
