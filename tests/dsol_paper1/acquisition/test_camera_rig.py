import unittest
import numpy as np
from AlphaBrain.research.dsol.acquisition.camera_rig import orbit_rig


class CameraRigTests(unittest.TestCase):
    def setUp(self):
        self.p = np.array([[-.1, 0., 1.], [.1, 0., 1.]])
        self.r = np.repeat(np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])[None], 2, axis=0)

    def test_identity(self):
        out = orbit_rig(self.p, self.r)
        np.testing.assert_allclose(out['positions'], self.p)
        np.testing.assert_allclose(out['rotations'], self.r)

    def test_baseline_and_relative_rotation_preserved(self):
        out = orbit_rig(self.p, self.r, azimuth_deg=45, elevation_deg=20, radius_scale=.65)
        self.assertAlmostEqual(np.linalg.norm(out['positions'][1] - out['positions'][0]), .2)
        np.testing.assert_allclose(out['rotations'][0].T @ out['rotations'][1], self.r[0].T @ self.r[1], atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(out['midpoint'] - out['pivot']), .65)

    def test_rejects_invalid_geometry(self):
        for scale in (0., -1., float('nan')):
            with self.assertRaises(ValueError):
                orbit_rig(self.p, self.r, radius_scale=scale)
