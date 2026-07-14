import unittest

from build_libero_episode_windows import (
    assign_window_labels,
    build_quality_report,
    oracle_window_horizon,
)


class EpisodeWindowBuilderTest(unittest.TestCase):
    def test_oracle_horizon_restores_after_feedback(self):
        self.assertEqual(
            oracle_window_horizon(48, feedback_reveal_time=53, action_divergence_time=53, horizon=10),
            5,
        )
        self.assertEqual(
            oracle_window_horizon(53, feedback_reveal_time=53, action_divergence_time=53, horizon=10),
            10,
        )
        self.assertEqual(
            oracle_window_horizon(2, feedback_reveal_time=53, action_divergence_time=53, horizon=10),
            10,
        )

    def test_labels_are_shared_by_window_group_and_shuffled_within_split(self):
        records = []
        for pair, split, oracle in (("a", "train", 2), ("b", "train", 10), ("c", "test", 4)):
            for branch in ("attached", "slipped"):
                records.append(
                    {
                        "sample_id": f"{pair}-{branch}",
                        "window_group_id": pair,
                        "pair_id": pair,
                        "split": split,
                        "oracle_feedback_horizon": oracle,
                        "gripper_transition_horizon": 7,
                    }
                )
        labels = assign_window_labels(records, horizon=10, seed=7)
        self.assertEqual(labels["a-attached"], labels["a-slipped"])
        self.assertEqual(labels["c-attached"]["shuffled_oracle_horizon"], 4)

    def test_shuffled_oracle_preserves_sample_marginal_with_unequal_group_sizes(self):
        records = []
        for group, oracle, branches in (
            ("paired-low", 2, ("attached", "slipped")),
            ("paired-full", 10, ("attached", "slipped")),
            ("single-full", 10, ("slipped",)),
        ):
            for branch in branches:
                records.append(
                    {
                        "sample_id": f"{group}-{branch}",
                        "window_group_id": group,
                        "pair_id": group,
                        "split": "train",
                        "oracle_feedback_horizon": oracle,
                        "gripper_transition_horizon": 7,
                    }
                )
        labels = assign_window_labels(records, horizon=10, seed=19)
        oracle = sorted(labels[row["sample_id"]]["oracle_feedback_horizon"] for row in records)
        shuffled = sorted(labels[row["sample_id"]]["shuffled_oracle_horizon"] for row in records)
        self.assertEqual(shuffled, oracle)

    def test_quality_rejects_low_post_feedback_horizon(self):
        record = {
            "sample_id": "a",
            "window_group_id": "g",
            "pair_id": "p",
            "split": "train",
            "observation": {"agentview_path": "a.jpg", "wrist_path": "w.jpg"},
            "oracle_feedback_horizon": 2,
            "gripper_transition_horizon": 2,
            "is_post_feedback": True,
        }
        labels = {
            "a": {
                "oracle_feedback_horizon": 2,
                "shuffled_oracle_horizon": 2,
            }
        }
        report = build_quality_report([record], labels, horizon=10)
        self.assertFalse(report["checks"]["post_feedback_horizon_restored"])


if __name__ == "__main__":
    unittest.main()
