import numpy as np
import pytest

from AlphaBrain.research.dsol.analysis.view_structure import camera_distance, score_quartiles, correlation, mean_field_rank_correlation
from AlphaBrain.research.dsol.plotting.view_space import adaptive_limits, absolute_space_fields, spatial_field


def test_quartiles_sort_within_initial_not_across_tasks():
    score = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
    success = np.array([[1, .7, .3, 0], [0, .3, .7, 1]])
    np.testing.assert_allclose(score_quartiles(success, score), [[1, .7, .3, 0]] * 2)
    with pytest.raises(ValueError):
        score_quartiles(success[:, :3], score[:, :3])


def test_distances_use_actual_camera_geometry():
    c = np.broadcast_to(np.eye(4), (1, 2, 4, 4)).copy()
    c[0, 1, :3, 3] = [0, 3, 4]
    c[0, 1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    d, r = camera_distance(c)
    np.testing.assert_allclose(d, [[0, 5]])
    np.testing.assert_allclose(r, [[0, 90]])


def test_adaptive_scale_preserves_all_values_and_constant_fields():
    lo, hi = adaptive_limits([80, 82, 84])
    assert 0 < lo <= 80 < 84 <= hi < 100
    assert adaptive_limits([100, 100]) == (99.5, 100)
    assert adaptive_limits([0, 0]) == (0, .5)
    with pytest.raises(ValueError):
        adaptive_limits([float('nan')])
    with pytest.raises(ValueError):
        adaptive_limits([101])


def test_constant_success_is_undefined_correlation():
    assert correlation(np.arange(4), np.ones(4)) is None


def test_mean_field_correlation_matches_plotted_points_not_initial_median():
    a=np.array([[.1,.2,.3,.4], [.4,.3,.2,.1]])
    q=np.array([[.9,.8,.7,.6], [.1,.2,.3,.5]])
    # The mean Accel field is flat even though each initial has a ranking.
    assert mean_field_rank_correlation(q,a) is None
    a[1]=[.15,.25,.35,.45]
    assert mean_field_rank_correlation(q,a)==correlation(-a.mean(0),q.mean(0))
    assert mean_field_rank_correlation(q,100*a+7)==mean_field_rank_correlation(q,a)


def test_absolute_space_keeps_value_reduction_not_rank_and_excludes_canonical():
    accel = np.array([[[99, .1, .2], [99, .2, .3]], [[99, .05, .1], [99, .05, .1]]])
    q = np.array([[[1, .2, .4], [1, .4, .6]], [[1, .8, .9], [1, 1, .9]]])
    fields = absolute_space_fields(q, accel)
    np.testing.assert_allclose(fields['accel'], [[.15, .25], [.05, .1]])
    np.testing.assert_allclose(fields['success'], [[30, 50], [90, 90]])
    assert fields['success_limits'] == (0, 100)
    assert fields['accel_limits'] == (0, .25)


def test_spatial_field_respects_shared_scale_and_low_value_brightness():
    class Axis:
        def scatter(self, *args, **kwargs):
            return kwargs
    result = spatial_field(Axis(), np.zeros((2, 3)), [.1, .2], limits=(0, .3), cmap='viridis_r')
    assert result['vmin'] == 0 and result['vmax'] == .3
    assert result['cmap'] == 'viridis_r'
