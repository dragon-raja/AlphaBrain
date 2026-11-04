"""Reject unknown/missing scientific outcomes before spatial aggregation."""
import pytest
from AlphaBrain.research.dsol.analysis.spatial_completion import assemble, summarize


def records():
    expected = [dict(state_key='suite::task::20', view_id=0, repeat=r, angles_deg=[0., 0.]) for r in (0, 1)]
    scores = [dict(state_key='suite::task::20', view=0, angles_deg=[0., 0.],
                   scores=[{'3': 1., '4': 2.}, {'3': 3., '4': 4.}])]
    outcomes = [dict(r, success=bool(r['repeat'])) for r in expected]
    return expected, scores, outcomes


def test_exact_paired_coverage_and_members():
    rows = assemble(*records())
    assert rows[0]['success'] == [False, True]
    assert rows[0]['p3'] == [1., 3.]
    assert rows[0]['p4'] == [2., 4.]
    assert rows[0]['task'] == 'suite::task'


@pytest.mark.parametrize('error', ['missing', 'duplicate', 'unknown', 'pose', 'score_duplicate'])
def test_no_silent_trial_reuse_or_imputation(error):
    expected, scores, outcomes = records()
    if error == 'missing': outcomes.pop()
    elif error == 'duplicate': outcomes.append(outcomes[0])
    elif error == 'unknown': outcomes[0]['success'] = None
    elif error == 'pose': scores[0]['angles_deg'] = [1., 0.]
    elif error == 'score_duplicate': scores.append(scores[0])
    with pytest.raises(ValueError): assemble(expected, scores, outcomes)


def test_diagnostic_scores_keep_independent_endpoint_selection():
    rows = [dict(state_key='suite::task::20', task='suite::task', view_id=i,
                 angles_deg=[float(i % 9), float(i // 9)],
                 p3=[float(i+1)]*2, p4=[float(30-i)]*2,
                 success=[i < 10]*2) for i in range(27)]
    value = summarize(rows)
    assert value['recovery']['p3']['selected_success'] == 1.
    assert value['recovery']['p4']['selected_success'] == 0.
    assert value['finite_pool_subset']['p3']['1'] == pytest.approx(9/26)
    assert value['finite_pool_subset']['p3']['26'] == 1.
    assert value['finite_pool_subset']['p4']['26'] == 0.
    assert value['old_two_starts_success'] == 0.
    assert value['pearson']['p3']['scopes']['nonreference']['favorable_r'] > 0
    assert value['pearson']['p4']['scopes']['nonreference']['favorable_r'] < 0
    low3 = next(v for v in value['score_validity_secondary']['aggregate']
                if v['score'] == 'p3' and v['exclude_reference'])
    assert low3['low_group_gain_task_mean'] == pytest.approx(1-9/26)
