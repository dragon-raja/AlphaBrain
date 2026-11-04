import copy
from itertools import combinations
import numpy as np
import pytest
from AlphaBrain.research.dsol.analysis.spatial_recovery import analyze, uniform_subset_argmin_probabilities


@pytest.mark.parametrize('size', [1, 2, 3, 4])
def test_subset_weights_match_exhaustive_selection_with_ties(size):
    scores = [3., 1., 1., 2.]
    expected = np.zeros(4)
    subsets = list(combinations(range(4), size))
    for subset in subsets:
        expected[min(subset, key=lambda i: (scores[i], i))] += 1 / len(subsets)
    np.testing.assert_allclose(uniform_subset_argmin_probabilities(scores, size), expected)


@pytest.mark.parametrize('scores,size', [([], 1), ([1], 0), ([1], 2),
    ([float('nan')], 1), ([[1]], 1), ([1], True), ([1], 1.0)])
def test_subset_invalid_inputs(scores, size):
    with pytest.raises(ValueError): uniform_subset_argmin_probabilities(scores, size)


def panel():
    # A good point exists in the low-score group but the minimum fails.
    return [dict(state_key='state', task='task', view_id=i, angles_deg=[i, 0],
                 p3=[float(score)] * 2, success=[success] * 2)
            for i, score, success in [(0, 0, True), (1, 1, False),
                                      (2, 2, True), (3, 3, False)]]


def test_existence_is_not_selection_or_additional_independent_rollouts():
    result = analyze(panel(), expected_views=4, low_count=2)
    assert result['reference_success'] == 1
    assert result['selected_success'] == 0
    assert result['selected_distinct_operations'] == 2
    assert result['states_with_low_group_two_of_two'] == 1
    assert result['per_state'][0]['argmin']['view_id'] == 1
    assert result['paired_endpoint_reuse']['comparisons'] == 6
    assert result['paired_endpoint_reuse']['harmed'] == 2
    assert result['paired_endpoint_reuse']['rescued'] == 0
    assert result['paired_endpoint_reuse']['failed_with_lower_score_success_available'] == 2


def test_labels_and_row_order_cannot_select_endpoint():
    rows = panel()
    original = analyze(rows, expected_views=4, low_count=2)['per_state'][0]['argmin']['view_id']
    for row in rows: row['success'] = [not x for x in row['success']]
    assert analyze(list(reversed(rows)), expected_views=4, low_count=2)['per_state'][0]['argmin']['view_id'] == original


@pytest.mark.parametrize('fault', ['duplicate', 'missing', 'nan', 'nonbool'])
def test_invalid_panel_fails(fault):
    rows = copy.deepcopy(panel())
    if fault == 'duplicate': rows[-1] = rows[-2]
    if fault == 'missing': rows.pop()
    if fault == 'nan': rows[1]['p3'][0] = float('nan')
    if fault == 'nonbool': rows[1]['success'][0] = 0
    with pytest.raises(ValueError): analyze(rows, expected_views=4, low_count=2)
