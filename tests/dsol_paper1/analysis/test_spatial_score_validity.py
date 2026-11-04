import pytest
from AlphaBrain.research.dsol.analysis.spatial_score_validity import summarize


def panel():
    return [dict(state_key='s', task='t', view_id=i, success=[i < 7, i < 7],
                 p3=[float(i), float(i)], p9=[float(-i), float(-i)]) for i in range(27)]


def test_score_direction_and_reference_exclusion():
    result = summarize(panel())
    a = result['aggregate']
    assert a[0]['low_group_gain_task_mean'] == pytest.approx(1-7/27)
    assert a[1]['low_group_gain_task_mean'] == pytest.approx(6/7-6/26)
    assert a[2]['low_group_gain_task_mean'] == pytest.approx(-7/27)
    assert result['per_state'][0]['spearman'] < 0


def test_missing_or_duplicate_cannot_be_reported_complete():
    with pytest.raises(ValueError): summarize(panel()[:-1])
    rows = panel(); rows[-1] = rows[0]
    with pytest.raises(ValueError): summarize(rows)


def test_constant_scores_have_no_correlation_and_stable_ties():
    rows = panel()
    for row in rows: row['p3'] = [1., 1.]
    result = summarize(list(reversed(rows)))
    assert result['per_state'][0]['spearman'] is None
    assert result['per_state'][0]['argmin_view'] == 0
