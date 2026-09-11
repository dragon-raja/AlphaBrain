from __future__ import annotations

import unittest

from summarize_counterfactual_feedback_revision import (
    average_seed_rows,
    mechanism_interpretation,
    seed_primary_metrics,
)


def interval(mean: float, low: float | None = None) -> dict[str, float]:
    return {
        "mean": mean,
        "bootstrap_95_low": mean if low is None else low,
        "bootstrap_95_high": mean,
    }


def primary() -> dict:
    return {
        "slipped_stale_minus_fresh_mse": interval(0.2, 0.1),
        "failure_continuation_reduction": interval(0.3, 0.1),
        "recovery_action_gain": interval(0.3, 0.1),
        "fresh_pair_assignment_correct": interval(0.9, 0.8),
        "attached_fresh_minus_stale_mse": interval(0.0, -0.01),
        "fresh_slipped_failure_continuation": interval(0.1, 0.0),
        "fresh_slipped_recovery_action": interval(0.9, 0.8),
    }


class InterpretationTest(unittest.TestCase):
    def test_confirms_stale_tail_when_fresh_revision_is_specific(self) -> None:
        result = mechanism_interpretation(primary())
        self.assertEqual(result["label"], "STALE_TAIL_CONFIRMED_IMMEDIATE_REPLAN_CAPABLE")
        self.assertTrue(result["execution_staleness_confirmed"])
        self.assertTrue(result["feedback_specificity_supported"])

    def test_reports_joint_gap_when_fresh_still_continues(self) -> None:
        values = primary()
        values["fresh_slipped_failure_continuation"] = interval(0.4, 0.3)
        result = mechanism_interpretation(values)
        self.assertEqual(result["label"], "STALE_TAIL_AND_IMMEDIATE_POLICY_GAP")


class AverageRowsTest(unittest.TestCase):
    def test_averages_policy_seeds_before_group_statistics(self) -> None:
        def payload(seed: int, value: float) -> dict:
            return {
                "rows": [
                    {
                        "pair_id": "g0",
                        "source_initial_state_index": 7,
                        "stale_age": 1,
                        "horizon": 1,
                        "stale_slipped_failure_continuation": value,
                        "fresh_slipped_failure_continuation": 0.0,
                        "fresh_slipped_recovery_action": 1.0,
                        "stale_slipped_recovery_action": 0.0,
                    }
                ],
                "policy_seed": seed,
            }

        row = average_seed_rows([payload(41, 1.0), payload(42, 0.0), payload(43, 1.0)])[0]
        self.assertAlmostEqual(row["stale_slipped_failure_continuation"], 2.0 / 3.0)
        self.assertAlmostEqual(row["failure_continuation_reduction"], 2.0 / 3.0)
        self.assertEqual(row["recovery_action_gain"], 1.0)


class SeedReportTest(unittest.TestCase):
    def test_extracts_primary_metrics_without_pooling_seeds(self) -> None:
        metrics = {
            "slipped_stale_minus_fresh_mse": 0.4,
            "relative_slipped_mse_reduction": 0.8,
            "stale_slipped_recovery_action": 0.1,
            "fresh_slipped_recovery_action": 0.9,
            "fresh_pair_assignment_correct": 1.0,
            "attached_fresh_minus_stale_mse": -0.01,
        }
        payload = {
            "policy_seed": 41,
            "summaries": {"age1_h1": {"means": metrics}},
        }
        self.assertEqual(seed_primary_metrics(payload), {"policy_seed": 41, **metrics})


if __name__ == "__main__":
    unittest.main()
