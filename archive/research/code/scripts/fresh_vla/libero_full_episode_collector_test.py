import unittest

import numpy as np

from libero_full_episode_collector import (
    FullEpisodeTeacher,
    first_persistent_action_divergence,
    first_visual_reveal,
    merge_prefix_and_continuation,
)


def observation(eef=(0.0, 0.0, 1.0), obj=(0.0, 0.0, 0.9), bowl=(0.2, 0.0, 0.9)):
    return {
        "robot0_eef_pos": np.asarray(eef, dtype=np.float64),
        "cream_cheese_1_pos": np.asarray(obj, dtype=np.float64),
        "akita_black_bowl_1_pos": np.asarray(bowl, dtype=np.float64),
    }


class FullEpisodeTeacherTest(unittest.TestCase):
    def test_slip_enters_recovery_without_branch_specific_teacher(self):
        teacher = FullEpisodeTeacher(observation())
        attached = teacher.decide(observation(), grasped=True, success=False)
        self.assertEqual(attached.phase, "lift")

        slipped = teacher.decide(observation(), grasped=False, success=False)
        self.assertEqual(slipped.phase, "recover_open")
        self.assertTrue(slipped.recovering)
        self.assertEqual(slipped.action[-1], -1.0)

    def test_success_transitions_to_retract(self):
        teacher = FullEpisodeTeacher(observation())
        decision = teacher.decide(observation(), grasped=False, success=True)
        self.assertEqual(decision.phase, "retract")
        self.assertEqual(decision.action[-1], -1.0)

    def test_failed_place_transitions_back_to_recovery(self):
        teacher = FullEpisodeTeacher(observation())
        teacher.phase = "retract"
        teacher.phase_steps = teacher.config.retract_steps
        teacher.place_attempts = 1
        decision = teacher.decide(
            observation(eef=(0.0, 0.0, 0.9 + teacher.config.carry_height)),
            grasped=False,
            success=False,
        )
        self.assertEqual(decision.phase, "recover_open")
        self.assertTrue(decision.recovering)

    def test_persistent_divergence_ignores_one_step_spike(self):
        attached = np.zeros((6, 2), dtype=np.float32)
        slipped = attached.copy()
        slipped[1] = 1.0
        slipped[3:] = 1.0
        self.assertEqual(first_persistent_action_divergence(attached, slipped), 3)

    def test_visual_reveal_starts_at_intervention(self):
        attached = {
            "agentview": np.zeros((5, 2, 2, 3), dtype=np.uint8),
            "wrist": np.zeros((5, 2, 2, 3), dtype=np.uint8),
        }
        slipped = {key: value.copy() for key, value in attached.items()}
        slipped["wrist"][3, 0, 0, 0] = 1
        self.assertEqual(first_visual_reveal(attached, slipped, start=2), 3)

    def test_merge_keeps_one_shared_boundary_observation(self):
        prefix = {
            "agentview": np.zeros((3, 1), dtype=np.uint8),
            "actions": np.zeros((2, 1), dtype=np.float32),
            "action_phases": np.asarray(["a", "b"]),
        }
        continuation = {
            "agentview": np.ones((4, 1), dtype=np.uint8),
            "actions": np.ones((3, 1), dtype=np.float32),
            "action_phases": np.asarray(["c", "d", "e"]),
        }
        merged = merge_prefix_and_continuation(prefix, continuation)
        self.assertEqual(merged["agentview"].shape[0], 6)
        self.assertEqual(merged["actions"].shape[0], 5)
        self.assertEqual(merged["agentview"][2, 0], 1)


if __name__ == "__main__":
    unittest.main()
