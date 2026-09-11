import unittest

import numpy as np

from multimodal_sampling_evaluator import analyze_samples


class MultimodalSamplingEvaluatorTest(unittest.TestCase):
    def test_detects_two_suffix_modes_without_prefix_motion(self):
        common = np.zeros((2, 4, 2))
        samples = np.zeros((2, 32, 4, 2))
        samples[:, :16, 2:, 0] = 1.0
        samples[:, 16:, 2:, 0] = -1.0
        result = analyze_samples(samples, common, np.array([2, 2]), np.array([1.0, 0.0]))
        self.assertEqual(result["suffix_mode_coverage"], 1.0)
        self.assertEqual(result["contexts_with_branch_motion_before_feedback"], 0.0)
        self.assertEqual(result["premature_commitment"], 0.0)

    def test_rejects_horizon_outside_chunk(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            analyze_samples(
                np.zeros((1, 32, 4, 2)),
                np.zeros((1, 4, 2)),
                np.array([5]),
                np.array([1.0, 0.0]),
            )


if __name__ == "__main__":
    unittest.main()
