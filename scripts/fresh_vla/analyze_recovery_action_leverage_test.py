import unittest

import numpy as np

from analyze_recovery_action_leverage import (
    decision_metric_matrix,
    two_factor_variance,
)
from evaluate_recovery_segment_oracle import BINARY_METRICS, METRICS


def bridge_outcome(success):
    return {
        metric: bool(success) if metric in BINARY_METRICS else float(success)
        for metric in METRICS
    }


def candidate(values):
    continuations = [
        {"repeat": repeat, "bridge": bridge_outcome(value)}
        for repeat, value in enumerate(values)
    ]
    summary = {
        (f"{metric}_rate" if metric in BINARY_METRICS else metric): float(
            np.mean([bool(value) if metric in BINARY_METRICS else value for value in values])
        )
        for metric in METRICS
    }
    return {
        "decision_heldout_continuations": continuations,
        "decision_heldout_summary": summary,
    }


class RecoveryActionLeverageTest(unittest.TestCase):
    def test_action_only_variance(self):
        matrix = np.asarray([[0.0] * 5, [1.0] * 5, [2.0] * 5, [3.0] * 5])
        result = two_factor_variance(matrix)
        self.assertAlmostEqual(result["action_total_variance_share"], 1.0)
        self.assertAlmostEqual(result["continuation_total_variance_share"], 0.0)
        self.assertAlmostEqual(result["residual_ss"], 0.0)
        self.assertEqual(result["action_changes_candidate_mean"], 1.0)

    def test_continuation_only_variance(self):
        matrix = np.tile(np.arange(5, dtype=np.float64), (4, 1))
        result = two_factor_variance(matrix)
        self.assertAlmostEqual(result["action_total_variance_share"], 0.0)
        self.assertAlmostEqual(result["continuation_total_variance_share"], 1.0)
        self.assertEqual(result["action_changes_candidate_mean"], 0.0)

    def test_balanced_decomposition_includes_interaction_residual(self):
        matrix = np.asarray(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ]
        )
        result = two_factor_variance(matrix)
        self.assertAlmostEqual(
            result["total_ss"],
            result["action_ss"]
            + result["continuation_ss"]
            + result["residual_ss"],
        )
        self.assertGreater(result["residual_ss"], 0.0)

    def test_raw_continuations_recompute_candidate_summary(self):
        decision = {
            "candidates": [
                candidate([False, False, False, False, False]),
                candidate([True, False, True, False, True]),
                candidate([False, False, False, False, False]),
                candidate([True, True, True, True, True]),
            ]
        }
        matrix = decision_metric_matrix(decision, "success", row_label="row")
        self.assertEqual(matrix.shape, (4, 5))
        self.assertEqual(matrix[3].tolist(), [1.0] * 5)
        decision["candidates"][0]["decision_heldout_summary"]["success_rate"] = 1.0
        with self.assertRaisesRegex(ValueError, "disagrees with raw success"):
            decision_metric_matrix(decision, "success", row_label="row")


if __name__ == "__main__":
    unittest.main()
