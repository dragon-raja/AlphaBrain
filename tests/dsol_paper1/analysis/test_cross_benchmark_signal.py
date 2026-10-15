import unittest
import numpy as np
from AlphaBrain.research.dsol.analysis.cross_benchmark_signal import (
    view_components, noise_harmonization, common_parameter_box, balanced_initial_components,
    rank_field_association)


class CrossBenchmarkSignalTests(unittest.TestCase):
    def test_fixed_deterministic_view_pattern_has_no_repeat_noise(self):
        y=np.broadcast_to(np.array([0.,0.,1.,1.])[None,:,None],(3,4,8))
        p=view_components(y)
        np.testing.assert_allclose(p['signal'],.25);np.testing.assert_allclose(p['noise_of_mean'],0)
        np.testing.assert_allclose(p['flip'],0)

    def test_common_repeat_offset_is_not_view_signal(self):
        y=np.broadcast_to(np.array([0.,1.,0.,1.])[None,None,:],(3,5,4))
        p=view_components(y)
        np.testing.assert_allclose(p['signal'],0);np.testing.assert_allclose(p['observed'],0)

    def test_iid_view_noise_has_zero_expected_signal(self):
        y=np.random.default_rng(42).binomial(1,.4,size=(2000,97,4))
        p=view_components(y)
        self.assertLess(abs(p['signal'].mean()),.001)
        self.assertTrue(np.any(p['signal']<0))

    def test_stochastic_view_signal_matches_known_variance(self):
        probability=np.linspace(.1,.9,97)[None,:,None]
        y=(np.random.default_rng(3).random((2000,97,4))<probability)
        p=view_components(y)
        self.assertAlmostEqual(p['signal'].mean(),np.var(probability),delta=.002)

    def test_subsampling_preserves_deterministic_choices(self):
        y=np.broadcast_to(np.array([1.,0.,0.,1.])[None,:,None],(3,4,32))
        accel=np.broadcast_to(np.arange(4)[None,:,None],(3,4,8))
        r=noise_harmonization(y,accel,[1,2,3],draws=4)
        self.assertEqual(r['empirical_oracle']['mean'],1.)
        self.assertEqual(r['selected4_evaluated_remaining']['mean'],1.)

    def test_parameter_box_excludes_outside_low_score(self):
        y = np.broadcast_to(np.array([1., 1., 0.])[None, :, None], (4, 3, 4))
        cost = np.broadcast_to(np.array([2., 1., 0.])[None, :, None], (4, 3, 8))
        result = common_parameter_box(y, cost, [1, 2], [[0, 0, 1], [30, 15, 1.2], [60, 0, 1]])
        self.assertEqual(result['candidate_count'], 2)
        self.assertEqual(result['extra_count'], 1)
        self.assertEqual(result['min_accel'], 1.)

    def test_initial_specific_variance_zero_for_identical_initials(self):
        y = np.broadcast_to(np.array([0., 1.])[None, :, None], (8, 2, 4))
        result = balanced_initial_components(y, np.repeat([0, 1], 4), draws=4)
        self.assertEqual(result['initial_specific']['mean'], 0.)

    def test_perfect_binary_rank_field(self):
        y = np.broadcast_to(np.array([0., 0., 1., 1.])[None, :, None], (3, 4, 4))
        scores = np.broadcast_to(np.array([1., 1., 0., 0.])[None, :, None], (3, 4, 8))
        result = rank_field_association(y, scores)
        self.assertEqual(result['observed'], 1.)
        self.assertEqual(result['execution_noise_adjusted'], 1.)

    def test_flat_scores_not_associated(self):
        y = np.broadcast_to(np.array([0., 1.])[None, :, None], (3, 2, 4))
        scores = np.ones((3, 2, 8))
        self.assertIsNone(rank_field_association(y, scores)['observed'])


if __name__=='__main__':unittest.main()
