import unittest

import numpy as np

from libero_snapshot_collector import (
    action_toward,
    gripper_transition_horizon,
    offset_free_joint_qpos,
    quat_to_axis_angle,
    validate_physical_branches,
)


class LiberoSnapshotCollectorTest(unittest.TestCase):
    def test_action_toward_clips_translation_and_sets_gripper(self) -> None:
        action = action_toward([0.0, 0.0, 0.0], [0.1, -0.025, 0.0], gripper=2.0)
        self.assertTrue(np.array_equal(action, np.array([1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 1.0])))

    def test_gripper_transition_horizon_includes_transition_action(self) -> None:
        actions = np.zeros((5, 7))
        actions[:, -1] = [-1.0, -1.0, 1.0, 1.0, 1.0]
        self.assertEqual(gripper_transition_horizon(actions), 3)

    def test_gripper_transition_horizon_is_full_without_transition(self) -> None:
        actions = np.zeros((5, 7))
        actions[:, -1] = -1.0
        self.assertEqual(gripper_transition_horizon(actions), 5)

    def test_offset_free_joint_changes_only_position(self) -> None:
        qpos = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
        shifted = offset_free_joint_qpos(qpos, [0.5, -0.5, 0.25])
        self.assertTrue(np.array_equal(shifted, np.array([1.5, 1.5, 3.25, 1.0, 0.0, 0.0, 0.0])))

    def test_identity_quaternion_has_zero_axis_angle(self) -> None:
        self.assertTrue(np.array_equal(quat_to_axis_angle([0.0, 0.0, 0.0, 1.0]), np.zeros(3)))

    def test_push_validation_requires_physical_separation(self) -> None:
        metrics = validate_physical_branches(
            "blocked_push",
            {
                "free_slide": [{"object_displacement": 0.08}, {"object_displacement": 0.07}],
                "blocked": [{"object_displacement": 0.01}, {"object_displacement": 0.01}],
            },
        )
        self.assertGreater(metrics["free_over_blocked_ratio"], 3.0)

    def test_push_validation_rejects_non_contact_fixture(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "physical validation"):
            validate_physical_branches(
                "blocked_push",
                {
                    "free_slide": [{"object_displacement": 0.0}],
                    "blocked": [{"object_displacement": 0.01}],
                },
            )


if __name__ == "__main__":
    unittest.main()
