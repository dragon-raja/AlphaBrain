import unittest
import numpy as np
from AlphaBrain.research.dsol.analysis.statistics import hierarchy
from AlphaBrain.research.dsol.metrics.view_rules import metric_choices
from AlphaBrain.research.dsol.selectors.ridge import fit_ranker, pca_fit


class InitialAnalysisTests(unittest.TestCase):
    def test_oracle_averages_noise_before_selection(self):
        y = np.array([[[[1, 0], [0, 1]]]], dtype=float)
        self.assertEqual(hierarchy(y.mean(axis=-1))["initial_oracle"], 0.5)
        self.assertEqual(y.max(axis=2).mean(), 1.0)

    def test_hierarchy_order(self):
        q = np.random.default_rng(3).random((8, 4, 97))
        h = hierarchy(q)
        self.assertLessEqual(h["canonical"], h["global_best"])
        self.assertLessEqual(h["global_best"], h["task_best"])
        self.assertLessEqual(h["task_best"], h["initial_oracle"])

    def test_metric_ties_use_catalog_order(self):
        a = np.zeros((2, 12))
        v = np.zeros_like(a)
        v[:, 11] = 1
        c = metric_choices(a, v)
        np.testing.assert_array_equal(c["min_accel"], [0, 0])
        np.testing.assert_array_equal(c["visibility"], [11, 11])
        np.testing.assert_array_equal(c["accel10_visibility"], [0, 0])

    def test_pca_fit_does_not_use_test_features(self):
        x = np.random.default_rng(3).normal(size=(8, 10))
        train = [0, 1, 2, 3]
        a = pca_fit(x, train, 2)
        x[4:] += 1e5
        b = pca_fit(x, train, 2)
        np.testing.assert_allclose(a[1], b[1])
        np.testing.assert_allclose(a[2], b[2])

    def test_learner_is_independent_of_test_labels(self):
        rng = np.random.default_rng(2)
        n, k = 32, 7
        geometry = rng.normal(size=(n, k, 3))
        context = rng.normal(size=(n, 10))
        images = rng.normal(size=(n, k, 10))
        tasks = np.repeat(np.arange(8), 4)
        initials = np.tile(np.arange(4), 8)
        q = rng.random((n, k))
        for family in ["geometry_context_ridge", "candidate_image_ridge"]:
            a = fit_ranker(geometry, context, images, tasks, initials, q, family)
            changed = q.copy()
            changed[initials >= 2] = 1000
            b = fit_ranker(geometry, context, images, tasks, initials, changed, family)
            self.assertEqual(a[2], b[2])
            np.testing.assert_allclose(a[0], b[0])


if __name__ == "__main__":
    unittest.main()
