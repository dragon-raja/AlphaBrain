import numpy as np

from AlphaBrain.research.dsol.metrics.flow_structure import components, structure


def test_radial_and_direction_changes_are_distinct():
    radial = np.array([[[1., 0.]], [[2., 0.]], [[3., 0.]]])
    direction = np.array([[[1., 0.]], [[0., 1.]], [[-1., 0.]]])
    a, b = components(radial), components(direction)
    assert a['angular'] == 0 and a['radial'] == a['score']
    assert b['radial'] == 0 and np.isclose(b['angular'], b['score'])


def test_zero_norm_is_not_a_direction_reward():
    x = np.array([[[0., 0.]], [[1., 0.]], [[2., 0.]]])
    a = components(x)
    assert not a['direction_valid'] and np.isnan(a['angular'])
    assert np.isfinite(a['radial'])
    assert np.isnan(components(np.zeros_like(x))['score'])


def test_xr1_ignores_unexecuted_coordinates_and_reconstructs_channels():
    x = np.random.default_rng(4).normal(size=(2, 5, 5, 16, 60))
    a = structure(x, 'xr1')
    x[..., 12:] *= 1000
    b = structure(x, 'xr1')
    np.testing.assert_array_equal(a['baseline__score'], b['baseline__score'])
    np.testing.assert_array_equal(a['baseline__score'], a['near__score'])
    assert np.max(a['baseline__reconstruction_error']) < 1e-12


def test_pi05_scopes_preserve_far_and_inactive_diagnostics():
    x = np.random.default_rng(9).normal(size=(2, 5, 10, 50, 32))
    a = structure(x, 'official')
    assert 'far__score' in a and 'baseline_inactive__score' in a
    assert not np.array_equal(a['baseline__score'], a['near__score'])
    keys = [k for k in a if k.startswith('baseline_') and k.endswith('__accel_contribution')]
    np.testing.assert_allclose(sum(a[k] for k in keys), a['baseline__score'], atol=1e-12)
