import unittest
import numpy as np
from AlphaBrain.research.dsol.selectors.accel_field import fit_accel_field


class AccelFieldTests(unittest.TestCase):
    def test_prediction_gauge_and_input_boundary(self):
        rng = np.random.default_rng(3)
        images, poses = rng.normal(size=(40, 12)), rng.normal(size=(40, 3))
        y = images[:, :1] * np.arange(6.)[None, :] + rng.normal(size=(40, 1))
        a = fit_accel_field(images, poses, y, image_dim=4)
        b = fit_accel_field(images, poses, y + rng.normal(size=(40, 1)) * 100., image_dim=4)
        pred = a.predict(images[:2], poses[:2])
        self.assertEqual(pred.shape, (2, 6))
        np.testing.assert_allclose(pred.mean(-1), 0., atol=1e-10)
        np.testing.assert_allclose(pred, b.predict(images[:2], poses[:2]), atol=1e-10)
        self.assertEqual(a.components.shape, (4, 12))

    def test_pose_baseline_ignores_image(self):
        rng = np.random.default_rng(2)
        x, q, y = rng.normal(size=(20, 8)), rng.normal(size=(20, 3)), rng.normal(size=(20, 5))
        model = fit_accel_field(x, q, y, use_image=False)
        np.testing.assert_array_equal(model.predict(x, q), model.predict(x * 100, q))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            fit_accel_field(np.ones((2, 4)), np.ones((3, 3)), np.ones((2, 4)))
