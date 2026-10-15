import numpy as np
import pytest
from AlphaBrain.research.dsol.analysis.view_landscape import association


def test_rank_association_sign_and_scale_invariance():
    a = np.array([[1, 2, 3, 4], [20, 30, 40, 50]], float)
    q = np.array([[1, .75, .5, .25], [.75, .5, .25, 0]])
    before = association(a, q)
    assert before['pooled_within_initial_rank_correlation'] == pytest.approx(1)
    after = association(a*np.array([[100], [.1]]), q)
    assert after['pooled_within_initial_rank_correlation'] == pytest.approx(1)


def test_constant_success_not_false_zero_correlation():
    result = association(np.array([[1, 2, 3]]), np.ones((1, 3)))
    assert result['spatial_mean_spearman'] is None
    assert result['pooled_within_initial_rank_correlation'] is None
    assert result['defined_initials'] == 0
