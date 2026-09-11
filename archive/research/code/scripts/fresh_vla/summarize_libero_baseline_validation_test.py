from __future__ import annotations

import unittest

from summarize_libero_baseline_validation import group_metrics, summarize_across_seeds


def row(pair_id: str, branch: str, success: bool, progress: float) -> dict:
    return {
        "pair_id": pair_id,
        "branch_outcome": branch,
        "execution_horizon": 3,
        "success": success,
        "event_time": 5,
        "final_progress": progress,
        "grasp_subgoal": True,
        "lift_subgoal": True,
        "transport_subgoal": progress >= 0.75,
        "place_subgoal": success,
    }


class BaselineValidationTest(unittest.TestCase):
    def test_group_metrics_keeps_one_paired_snapshot_unit(self) -> None:
        groups = group_metrics(
            [row("g0", "attached", True, 1.0), row("g0", "slipped", False, 0.5)],
            3,
        )
        self.assertEqual(set(groups), {"g0"})
        self.assertEqual(groups["g0"]["overall_task_success"], 0.5)
        self.assertEqual(groups["g0"]["attached_task_success"], 1.0)

    def test_seed_averaging_precedes_group_bootstrap(self) -> None:
        first = group_metrics(
            [row("g0", "attached", True, 1.0), row("g0", "slipped", False, 0.5)], 3
        )
        second = group_metrics(
            [row("g0", "attached", False, 0.5), row("g0", "slipped", False, 0.5)], 3
        )
        summary = summarize_across_seeds({41: first, 42: second}, bootstrap_samples=100)
        self.assertEqual(summary["attached_task_success"]["count"], 1)
        self.assertEqual(summary["attached_task_success"]["mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
