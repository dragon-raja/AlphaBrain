import unittest

import numpy as np

from probe_post_feedback_modes import analyze_candidate_set


class PostFeedbackModeProbeTest(unittest.TestCase):
    def test_best_of_n_detects_available_correct_mode(self):
        attached = np.ones((3, 2), dtype=np.float32)
        slipped = -attached
        chunks = np.stack((attached, slipped, np.zeros_like(attached)))
        result = analyze_candidate_set(
            chunks,
            slipped,
            attached,
            action_steps=3,
            mode_margin=0.02,
        )
        self.assertFalse(result["sample0_correct_mode"])
        self.assertTrue(result["any_correct_mode"])
        self.assertEqual(result["best_correct_index"], 1)
        self.assertEqual(result["best_correct_rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
