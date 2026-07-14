import unittest

from summarize_libero_reach import aggregate_seed_summaries, reach_group_metrics


def row(pair_id, k, success, error):
    return {
        "pair_id": pair_id,
        "execution_horizon": k,
        "success": success,
        "best_target_error": error,
        "final_target_error": error + 0.01,
        "best_target_progress": 0.2,
        "final_target_progress": 0.19,
        "first_action_mse": 0.1,
        "first_translation_cosine": 0.8,
    }


class ReachSummaryTest(unittest.TestCase):
    def test_group_metrics_filters_execution_horizon(self):
        result = reach_group_metrics(
            [row("g0", 1, True, 0.02), row("g0", 2, False, 0.08)],
            execution_horizon=1,
        )
        self.assertEqual(list(result), ["g0"])
        self.assertEqual(result["g0"]["success"], 1.0)

    def test_group_metrics_rejects_duplicate_group(self):
        with self.assertRaisesRegex(ValueError, "duplicate reach row"):
            reach_group_metrics([row("g0", 1, True, 0.02), row("g0", 1, False, 0.08)], execution_horizon=1)

    def test_aggregate_keeps_seed_values(self):
        first = reach_group_metrics([row("g0", 1, True, 0.02)], execution_horizon=1)["g0"]
        second = reach_group_metrics([row("g0", 1, False, 0.08)], execution_horizon=1)["g0"]
        aggregate = aggregate_seed_summaries([first, second])
        self.assertEqual(aggregate["success"]["seed_values"], [1.0, 0.0])
        self.assertEqual(aggregate["success"]["mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
