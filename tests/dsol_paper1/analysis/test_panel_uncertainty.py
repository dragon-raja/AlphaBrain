import numpy as np
import pytest

from AlphaBrain.research.dsol.analysis.panel_uncertainty import paired_panel_intervals


def test_shared_outcomes_cancel_in_paired_comparison():
    panel = np.random.default_rng(7).normal(size=(3, 8, 2))
    result = paired_panel_intervals(panel + .25, panel, draws=500)
    np.testing.assert_allclose(result['fixed_observed_policies_task_initial_ci95'], [.25, .25])
    np.testing.assert_allclose(result['crossed_seed_task_initial_ci95_sensitivity'], [.25, .25])


def test_global_policy_variation_is_not_erased_by_many_tasks():
    left = np.broadcast_to(np.array([-.5, 0, .5])[:, None, None], (3, 80, 2))
    result = paired_panel_intervals(left, np.zeros_like(left), draws=2000)
    np.testing.assert_allclose(result['fixed_observed_policies_task_initial_ci95'], [0, 0])
    lo, hi = result['crossed_seed_task_initial_ci95_sensitivity']
    assert lo <= -1/3 and hi >= 1/3


def test_reproducible_and_no_input_mutation():
    a = np.random.default_rng(9).normal(size=(3, 8, 2)); saved = a.copy()
    assert paired_panel_intervals(a, a*.2, draws=200) == paired_panel_intervals(a, a*.2, draws=200)
    np.testing.assert_array_equal(a, saved)


@pytest.mark.parametrize('shape', [(3, 8), (1, 8, 2), (3, 1, 2), (3, 8, 1)])
def test_reject_unsupported_panel(shape):
    with pytest.raises(ValueError):
        paired_panel_intervals(np.zeros(shape), np.zeros(shape))


def test_reject_missing_outcome():
    a = np.zeros((3, 8, 2)); a[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        paired_panel_intervals(a, np.zeros_like(a))
