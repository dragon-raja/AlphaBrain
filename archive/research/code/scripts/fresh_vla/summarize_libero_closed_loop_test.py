import unittest

from summarize_libero_closed_loop import group_metrics, paired_group_delta


def row(pair, branch, k, success, recovery=False):
    return {
        "pair_id": pair,
        "branch_outcome": branch,
        "execution_horizon": k,
        "success": success,
        "recovery_success": recovery,
        "failure_continuation": False,
        "premature_commitment": False,
        "recovery_switch_latency": 1 if branch == "slipped" else None,
        "regrasp_success": recovery if branch == "slipped" else None,
        "drop": False,
        "final_progress": 1.0 if success else 0.5,
        "progress_auc": 0.75 if success else 0.25,
        "grasp_subgoal": True,
        "lift_subgoal": True,
        "transport_subgoal": success,
        "place_subgoal": success,
        "event_time": 5,
        "completion_steps": 10,
    }


class ClosedLoopSummaryTest(unittest.TestCase):
    def test_group_metrics_produces_one_row_per_snapshot_group(self):
        isolated = [row("g0", "attached", 2, True), row("g0", "slipped", 2, True, True)]
        end_to_end = [row("g0", "attached", 2, True), row("g0", "slipped", 2, False, False)]
        deterministic = [{"pair_id": "g0", "execution_horizon": 2, "success": True}]
        result = group_metrics(isolated, end_to_end, deterministic, execution_horizon=2)
        self.assertEqual(list(result), ["g0"])
        self.assertEqual(result["g0"]["overall_task_success"], 0.5)
        self.assertEqual(result["g0"]["isolated_recovery_success"], 1.0)
        self.assertEqual(result["g0"]["isolated_regrasp_success"], 1.0)
        self.assertEqual(result["g0"]["final_progress"], 0.75)
        self.assertEqual(result["g0"]["deterministic_reach_success"], 1.0)

    def test_bootstrap_averages_seeds_before_treating_groups_as_units(self):
        baseline = {
            41: {"g0": {"m": 0.0}, "g1": {"m": 1.0}},
            42: {"g0": {"m": 1.0}, "g1": {"m": 0.0}},
        }
        candidate = {
            41: {"g0": {"m": 1.0}, "g1": {"m": 1.0}},
            42: {"g0": {"m": 1.0}, "g1": {"m": 1.0}},
        }
        result = paired_group_delta(baseline, candidate, "m", bootstrap_samples=100, seed=7)
        self.assertEqual(result["group_count"], 2)
        self.assertEqual(result["group_deltas"], [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
