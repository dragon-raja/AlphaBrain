import numpy as np
import pytest
from AlphaBrain.research.dsol.analysis.geometry_visibility import aggregate_pairs, constrained_choices, visible_relations


def test_start_filter_keeps_initial_weight_not_start_count():
    v = np.array([[1., 0., 0.], [.5, .5, .5]])
    mask = np.array([[True, False, False], [True, True, True]])
    out = aggregate_pairs(v, np.zeros_like(v), mask, [0, 1])
    assert out['success'] == .75 and out['starts'] == 4
    assert out['per_initial_success'] == [1., .5]


def test_empty_initial_is_reported_not_imputed_as_failure():
    out = aggregate_pairs(np.ones((2, 3)), np.zeros((2, 3)), [[0, 0, 0], [1, 1, 0]], [0, 1])
    assert out['success'] == 1 and out['initials'] == 1 and out['tasks'] == 1
    assert out['per_initial_success'] == [None, 1.]


def test_restricted_choices_exclude_invisible_global_winner():
    result = constrained_choices([[0., 1., 2.]], [[10., 2., 4.]], [[False, True, True]])
    assert result['min_accel'].tolist() == [1]
    assert result['visibility'].tolist() == [2]
    assert result['accel10_visibility'].tolist() == [2]
    with pytest.raises(ValueError):
        constrained_choices([[0.]], [[0.]], [[False]])


def test_correlations_do_not_turn_constant_outcomes_into_zero():
    out = visible_relations(np.ones((1, 12)), np.arange(12)[None], np.ones((1, 12), bool), np.arange(12)[None])[0]
    assert out['all_rho'] is None and out['visible_rho'] is None
    assert out['visible_low_quartile_success'] == 1.
