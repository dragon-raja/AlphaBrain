import numpy as np
import pytest
from AlphaBrain.research.dsol.metrics.action_consensus import disagreement


def test_medoid_and_paired_noise():
    a = np.zeros((3, 2, 2, 3))
    a[1] = 1; a[2] = 10
    a[:, 1] += 100  # Common noise offset must cancel, not look inconsistent.
    np.testing.assert_allclose(disagreement(a, 2, slice(0, 3)), [5.5, 5, 9.5])


def test_permutation_scaling_and_horizon():
    a = np.random.default_rng(0).normal(size=(5, 8, 4, 7))
    d = disagreement(a, 2, slice(0, 3)); p = [4, 2, 0, 1, 3]
    np.testing.assert_allclose(disagreement(a[p]*3, 2, slice(0, 3)), d[p]*3)
    a[:, :, 2:] = 999
    np.testing.assert_allclose(disagreement(a, 2, slice(0, 3)), d)


def test_invalid():
    with pytest.raises(ValueError):
        disagreement(np.zeros((1, 8, 5, 3)), 5, slice(0, 3))
