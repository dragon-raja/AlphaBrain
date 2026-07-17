import unittest

import numpy as np

from candidate_support import (
    continuation_compatibility,
    physical_compatibility,
    recall_prefix,
    stable_seed,
    summarize_group_rows,
)


class CandidateSupportTest(unittest.TestCase):
    def test_seed_is_stable_and_candidate_specific(self):
        self.assertEqual(stable_seed(41, "g0", 0), stable_seed(41, "g0", 0))
        self.assertNotEqual(stable_seed(41, "g0", 0), stable_seed(41, "g0", 1))

    def test_joint_label_requires_action_and_effect(self):
        result = continuation_compatibility(
            np.zeros((2, 7)),
            np.ones(3),
            np.zeros((2, 7)),
            np.ones((2, 7)),
            np.zeros(3),
            np.ones(3),
            action_margin=0.02,
            effect_margin=0.002,
        )
        self.assertTrue(result["action_compatible"])
        self.assertFalse(result["effect_compatible"])
        self.assertFalse(result["joint_compatible"])

    def test_attached_physical_label_requires_preserved_grasp(self):
        self.assertTrue(
            physical_compatibility(
                "attached",
                teacher_success=True,
                grasp_trace=[True, True],
                empty_lift=False,
                recovery_action_seen=False,
                initial_object_distance=0.1,
                final_object_distance=0.1,
            )
        )
        self.assertFalse(
            physical_compatibility(
                "attached",
                teacher_success=True,
                grasp_trace=[True, False],
                empty_lift=False,
                recovery_action_seen=False,
                initial_object_distance=0.1,
                final_object_distance=0.1,
            )
        )

    def test_slipped_physical_label_rejects_empty_lift(self):
        self.assertFalse(
            physical_compatibility(
                "slipped",
                teacher_success=True,
                grasp_trace=[False, False],
                empty_lift=True,
                recovery_action_seen=True,
                initial_object_distance=0.1,
                final_object_distance=0.09,
            )
        )

    def test_prefix_recall_and_summary_are_group_level(self):
        recall = recall_prefix([False, True, False, False], [1, 4])
        self.assertEqual(recall, {"1": False, "4": True})
        rows = [
            {
                "outcome": "slipped",
                "joint_recall": recall,
                "action_recall": recall,
                "effect_recall": recall,
                "physical_recall": recall,
            },
            {
                "outcome": "slipped",
                "joint_recall": {"1": False, "4": False},
                "action_recall": {"1": False, "4": False},
                "effect_recall": {"1": False, "4": False},
                "physical_recall": {"1": False, "4": False},
            },
        ]
        summary = summarize_group_rows(rows, [1, 4])
        self.assertEqual(summary["slipped"]["joint_recall@4"], 0.5)


if __name__ == "__main__":
    unittest.main()
