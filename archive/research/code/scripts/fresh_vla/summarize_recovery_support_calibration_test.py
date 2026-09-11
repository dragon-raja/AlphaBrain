from __future__ import annotations

import unittest

from summarize_recovery_support_calibration import calibration_gate


class CalibrationGateTest(unittest.TestCase):
    def test_passes_mean_and_seed_count_thresholds(self) -> None:
        gate = calibration_gate(
            {
                "41": {"attached_task_success": 0.31},
                "42": {"attached_task_success": 0.20},
                "43": {"attached_task_success": 0.40},
            }
        )
        self.assertTrue(gate["passed"])

    def test_fails_when_only_one_seed_reaches_threshold(self) -> None:
        gate = calibration_gate(
            {
                "41": {"attached_task_success": 0.19},
                "42": {"attached_task_success": 0.31},
                "43": {"attached_task_success": 0.19},
            }
        )
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["observed_seeds_at_or_above_0_20"], 1)

    def test_fails_nonfinite_or_missing_seed(self) -> None:
        gate = calibration_gate(
            {
                "41": {"attached_task_success": 0.4},
                "42": {"attached_task_success": float("nan")},
            }
        )
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["all_seed_metrics_finite"])


if __name__ == "__main__":
    unittest.main()
