import unittest
import numpy as np
from AlphaBrain.research.dsol.analysis.relative_accel import paired_label_audit, selection_metrics


class RelativeAccelTests(unittest.TestCase):
    def test_shared_noise_cancels(self):
        a = np.arange(5.)[:, None] + np.array([0., 5., -2., 8.])[None, :]
        r = paired_label_audit(a)
        self.assertAlmostEqual(r['paired_to_independent_variance_median'], 0.)
        self.assertAlmostEqual(r['centered_to_raw_noise_variance'], 0.)
        self.assertTrue(all(c['sign_agreement'] == 1. for c in r['split_checks']))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            paired_label_audit(np.zeros((4, 3)))

    def test_selection_metrics(self):
        r = selection_metrics(np.array([[0., 1., 2., 3.], [3., 2., 1., 0.]]), np.arange(4.))
        self.assertEqual(r['low_quartile_hit'], .5)
        self.assertEqual(r['normalized_regret'], .5)

    def test_pair_loss_is_centered_loss(self):
        # Complete equally weighted pairwise MSE adds no independent constraint.
        e = np.random.default_rng(1).normal(size=9)
        i, j = np.triu_indices(len(e), 1)
        self.assertAlmostEqual(np.sum((e[i] - e[j]) ** 2), len(e) * np.sum((e - e.mean()) ** 2))
