import unittest

import numpy as np

from evaluate_libero_deterministic_reach import reference_target, summarize_rows, target_error


class DeterministicReachTest(unittest.TestCase):
    def test_target_error_is_euclidean(self):
        self.assertEqual(target_error([0, 0, 0], [3, 4, 0]), 5.0)

    def test_summary_uses_groups(self):
        rows = [
            {"pair_id": "g0", "success": True, "best_target_error": 0.02, "final_target_error": 0.03},
            {"pair_id": "g1", "success": False, "best_target_error": 0.06, "final_target_error": 0.08},
        ]
        summary = summarize_rows(rows)
        self.assertEqual(summary["group_count"], 2)
        self.assertEqual(summary["deterministic_reach_success"], 0.5)

    def test_reference_target_uses_recorded_eef_and_clamps_step(self):
        reference = {"eef_pose": np.asarray([[0, 1, 2, 9], [3, 4, 5, 9]], dtype=np.float32)}
        target, step = reference_target(reference, 20)
        np.testing.assert_array_equal(target, [3, 4, 5])
        self.assertEqual(step, 1)


if __name__ == "__main__":
    unittest.main()
