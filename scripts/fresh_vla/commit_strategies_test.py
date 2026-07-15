import unittest

import numpy as np
from commit_strategies import (
    CommitController,
    action_disagreement,
    boundary_commit_length,
    build_random_matched_boundaries,
    gripper_commit_length,
    self_consistency_commit_length,
)

GROUPS = [
    {"pair_id": "g0", "action_divergence_time": 4, "feedback_reveal_time": 5},
    {"pair_id": "g1", "action_divergence_time": 7, "feedback_reveal_time": 8},
    {"pair_id": "g2", "action_divergence_time": 9, "feedback_reveal_time": 10},
]


class CommitStrategiesTest(unittest.TestCase):
    def test_boundary_stops_at_boundary_and_returns_to_default_afterward(self):
        self.assertEqual(boundary_commit_length(0, 4), 3)
        self.assertEqual(boundary_commit_length(2, 4), 2)
        self.assertEqual(boundary_commit_length(3, 4), 1)
        self.assertEqual(boundary_commit_length(4, 4), 3)

    def test_gripper_commit_includes_transition_action(self):
        actions = np.zeros((5, 7), dtype=np.float32)
        actions[:, -1] = [-1, 1, 1, 1, 1]
        self.assertEqual(gripper_commit_length(actions, current_gripper_action=-1), 2)

    def test_gripper_commit_detects_transition_on_first_action(self):
        actions = np.zeros((5, 7), dtype=np.float32)
        actions[:, -1] = 1
        self.assertEqual(gripper_commit_length(actions, current_gripper_action=-1), 1)

    def test_random_boundary_mapping_preserves_multiset(self):
        mapped = build_random_matched_boundaries(GROUPS, seed=41)
        self.assertCountEqual(mapped.values(), [4, 7, 9])
        self.assertNotEqual(list(mapped.values()), [4, 7, 9])

    def test_self_consistency_is_prefix_closed(self):
        chunks = np.zeros((8, 4, 7), dtype=np.float32)
        chunks[:, 1, 0] = np.linspace(-1.0, 1.0, 8)
        scores = action_disagreement(chunks)
        self.assertEqual(scores[0], 0.0)
        decision = self_consistency_commit_length(chunks, threshold=0.15)
        self.assertEqual(decision.length, 1)

    def test_controller_never_uses_branch_outcome(self):
        controller = CommitController("oracle_branch_safe_commit", GROUPS, seed=41)
        chunks = np.zeros((1, 50, 7), dtype=np.float32)
        self.assertEqual(controller.decide("g0", global_step=3, sampled_chunks=chunks).length, 3)
        self.assertEqual(controller.decide("g0", global_step=4, sampled_chunks=chunks).length, 3)


if __name__ == "__main__":
    unittest.main()
