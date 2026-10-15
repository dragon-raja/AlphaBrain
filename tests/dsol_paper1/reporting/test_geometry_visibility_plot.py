import numpy as np
from AlphaBrain.research.dsol.plotting.geometry_visibility import orbit_positions, pivot_from_camera, frustum_corners


def test_both_azimuth_and_elevation_translate_but_radius_is_preserved():
    p = np.array([1., 0., 1.]); pivot = np.zeros(3)
    az = orbit_positions(p, pivot, np.array([-15., 0., 15.]), 0, 1)
    el = orbit_positions(p, pivot, 0, np.array([-15., 0., 15.]), 1)
    np.testing.assert_allclose(az[1], p); np.testing.assert_allclose(el[1], p)
    assert not np.allclose(az[0], p) and not np.allclose(el[0], p)
    np.testing.assert_allclose(np.linalg.norm(az, axis=1), np.linalg.norm(p))
    np.testing.assert_allclose(np.linalg.norm(el, axis=1), np.linalg.norm(p))


def test_saved_opencv_positive_z_is_used_for_forward():
    c = np.eye(4); c[:3, :3] = np.diag([1., -1., -1.]); c[:3, 3] = [1., 2., 3.]
    np.testing.assert_allclose(pivot_from_camera(c, .5), [1., 2., .5])


def test_frustum_boundary_rays_reproject_to_actual_image_corners():
    c = np.eye(4); c[:3, :3] = np.diag([1., -1., -1.]); c[:3, 3] = [1., 2., 3.]
    k = np.array([[170., 0., 112.], [0., 170., 112.], [0., 0., 1.]])
    world = frustum_corners(c, k, 224, 224, depth=.28)
    local = (world-c[:3, 3]) @ c[:3, :3]
    projected = local @ k.T
    np.testing.assert_allclose(projected[:, :2]/projected[:, 2:], [[0, 0], [223, 0], [223, 223], [0, 223]], atol=1e-10)
