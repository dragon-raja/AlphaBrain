import unittest
import numpy as np

from AlphaBrain.research.dsol.analysis.view_signal_structure import score_stability, fixed_comparisons


class SignalStructureTest(unittest.TestCase):
    def test_identical_noise_halves(self):
        a = np.broadcast_to(np.arange(4)[None, :, None, None], (2, 4, 8, 3)).copy()
        for result in score_stability(a):
            self.assertEqual(result['top1_agreement'], 1)
            self.assertEqual(result['mean_spearman'], 1)

    def test_fixed_selection_does_not_consume_test_labels(self):
        y = np.array([[[1, 1], [0, 0]], [[0, 0], [1, 1]]])
        accel = np.ones((2, 2, 8))
        a = fixed_comparisons(y, accel, ['a', 'a'], [True, False], [0, 1])
        y[1] = 1-y[1]
        b = fixed_comparisons(y, accel, ['a', 'a'], [True, False], [0, 1])
        self.assertEqual(a['global_fixed_view'], b['global_fixed_view'])
        self.assertEqual(a['task_fixed_views'], b['task_fixed_views'])
        self.assertNotEqual(a['success']['task_fixed'], b['success']['task_fixed'])

    def test_partial_outcomes_rejected(self):
        y = np.zeros((2, 2, 4)); y[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            fixed_comparisons(y, np.ones((2, 2, 8)), ['a', 'a'], [True, False], [0, 1])


if __name__ == '__main__':
    unittest.main()
