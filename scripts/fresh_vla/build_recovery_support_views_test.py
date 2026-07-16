from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_recovery_support_views import build_matched_rows, first_stable_true


def sample(sample_id: str, pair_id: str, root: Path) -> tuple[dict, Path]:
    return (
        {
            "sample_id": sample_id,
            "window_group_id": sample_id,
            "pair_id": pair_id,
            "branch_id": "slipped",
            "branch_outcome": "slipped",
            "task": "grasp_slip_full_episode",
            "split": "train",
            "frame_index": 0,
            "observation": {
                "agentview_path": "agent.jpg",
                "wrist_path": "wrist.jpg",
            },
            "robot_state": [0.0] * 8,
            "language_instruction": "put the cream cheese in the bowl",
            "action_chunk": np.zeros((10, 7), dtype=np.float32).tolist(),
            "oracle_feedback_horizon": 10,
        },
        root,
    )


class FirstStableTrueTest(unittest.TestCase):
    def test_returns_inclusive_completion_index(self) -> None:
        self.assertEqual(
            first_stable_true([False, True, False, True, True], start=1, dwell_steps=2),
            4,
        )

    def test_returns_none_without_stable_run(self) -> None:
        self.assertIsNone(first_stable_true([False, True, False], start=0, dwell_steps=2))


class MatchedRowsTest(unittest.TestCase):
    def test_matches_anchor_samples_and_target_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchors = [sample(f"anchor-{index}", f"g{index % 3}", root) for index in range(9)]
            clean = {
                group: [sample(f"clean-{group}-{index}", group, root) for index in range(2)]
                for group in ("g0", "g1", "g2")
            }
            policy = {
                group: [sample(f"policy-{group}-{index}", group, root) for index in range(3)]
                for group in ("g0", "g1", "g2")
            }
            rows, metadata = build_matched_rows(
                anchors,
                clean,
                policy,
                seed=41,
                steps=12,
                horizon=10,
            )

        self.assertTrue(metadata["passed"])
        self.assertEqual(metadata["anchor_count"], 6)
        self.assertEqual(metadata["target_count"], 6)
        base = rows["base_continuation"]
        clean = rows["clean_recovery_replay"]
        policy = rows["policy_state_recovery"]
        self.assertEqual(set(rows), {"base_continuation", "clean_recovery_replay", "policy_state_recovery"})
        for base_row, clean_row, policy_row in zip(base, clean, policy, strict=True):
            self.assertEqual(base_row["slot_type"], clean_row["slot_type"])
            self.assertEqual(clean_row["slot_type"], policy_row["slot_type"])
            if clean_row["slot_type"] == "anchor":
                self.assertEqual(base_row["source_sample_id"], clean_row["source_sample_id"])
                self.assertEqual(clean_row["source_sample_id"], policy_row["source_sample_id"])
            else:
                self.assertEqual(base_row["source_pair_id"], clean_row["source_pair_id"])
                self.assertEqual(clean_row["source_pair_id"], policy_row["source_pair_id"])

    def test_requires_even_step_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = sample("one", "g0", root)
            with self.assertRaisesRegex(ValueError, "even"):
                build_matched_rows(
                    [row],
                    {"g0": [row]},
                    {"g0": [row]},
                    seed=41,
                    steps=3,
                    horizon=10,
                )


if __name__ == "__main__":
    unittest.main()
