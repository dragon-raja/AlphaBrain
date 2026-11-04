import copy
import numpy as np
import pytest
from AlphaBrain.research.dsol.analysis.spatial_task_average import summarize
from AlphaBrain.research.dsol.analysis.spatial_task_average import pearson, linear_comparison


def test_pearson_is_value_correlation_not_rank_and_sign_is_explicit():
    assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)
    assert pearson([1, 2, 100], [1, 2, 3]) < .9
    assert pearson([1, 1, 1], [1, 2, 3]) is None
    a = linear_comparison(panel(), expected_views=4)
    s = a['scopes']['nonreference']
    assert s['favorable_r'] == pytest.approx(-s['raw_r'])
    assert s['r_squared'] == pytest.approx(s['raw_r']**2)


def test_average_surface_is_not_pooled_correlation_and_score_key_is_honored():
    rows = panel()
    for r in rows:
        r['p9'] = [-x for x in r['p3']]
    a = linear_comparison(rows, expected_views=4)
    b = linear_comparison(rows, score='p9', expected_views=4)
    assert a['scopes']['all']['raw_r'] == pytest.approx(-b['scopes']['all']['raw_r'])
    assert a['scopes']['all']['n_views'] == 4
    assert a['scopes']['all']['pooled_positions'] == 16


def panel():
    rows = []
    for task, states, outcomes in [('a', 1, [True, False, True, False]), ('b', 3, [False]*4)]:
        for state in range(states):
            for view, xy in enumerate([[0, 0], [1, 0], [0, 1], [1, 1]]):
                rows.append(dict(task=task, state_key=f'{task}-{state}', view_id=view,
                    angles_deg=xy, p3=[float(view+1)]*2, success=[outcomes[view]]*2))
    return rows


def test_task_weight_not_number_of_states_and_constant_correlation_is_undefined():
    result = summarize(panel(), expected_views=4)
    assert result['mean']['success'] == [.5, 0, .5, 0]
    assert result['valid_task_correlations'] == 1
    assert result['per_task'][1]['rho_nonreference'] is None


def test_within_state_rank_invariant_to_positive_score_scale_and_order():
    rows = panel()
    before = summarize(rows, expected_views=4)
    for r in rows:
        if r['task'] == 'b': r['p3'] = [x*100+9 for x in r['p3']]
    after = summarize(list(reversed(rows)), expected_views=4)
    np.testing.assert_array_equal(before['mean']['rank'], after['mean']['rank'])
    assert before['mean']['score'] != after['mean']['score']


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'geometry', 'nonbool'])
def test_invalid_identity_and_results_rejected(fault):
    rows = copy.deepcopy(panel())
    if fault == 'missing': rows.pop()
    if fault == 'duplicate': rows[-1] = rows[-2]
    if fault == 'geometry': rows[-1]['angles_deg'] = [2, 2]
    if fault == 'nonbool': rows[-1]['success'] = [0, 1]
    with pytest.raises(ValueError): summarize(rows, expected_views=4)
