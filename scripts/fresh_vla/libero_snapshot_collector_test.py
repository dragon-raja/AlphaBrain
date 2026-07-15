import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from libero_snapshot_collector import (
    _capture_controller_state,
    _restore_snapshot,
    action_toward,
    gripper_transition_horizon,
    offset_free_joint_qpos,
    quat_to_axis_angle,
    validate_physical_branches,
)


class LiberoSnapshotCollectorTest(unittest.TestCase):
    @staticmethod
    def fake_env():
        interpolator_pos = SimpleNamespace(
            start=np.array([1.0, 2.0, 3.0]),
            goal=np.array([4.0, 5.0, 6.0]),
            step=2,
        )
        interpolator_ori = SimpleNamespace(
            start=np.array([0.1, 0.2, 0.3]),
            goal=np.array([0.4, 0.5, 0.6]),
            step=1,
        )
        controller = SimpleNamespace(
            goal_pos=np.array([0.2, 0.3, 0.4]),
            goal_ori=np.eye(3),
            relative_ori=np.array([0.01, 0.02, 0.03]),
            ori_ref=np.eye(3) * 2,
            interpolator_pos=interpolator_pos,
            interpolator_ori=interpolator_ori,
            update=mock.Mock(),
            reset_goal=mock.Mock(),
        )
        model = SimpleNamespace(
            body_pos=np.zeros((2, 3)),
            geom_friction=np.ones((1, 3)),
        )
        runtime_env = SimpleNamespace(
            cur_time=0.15,
            timestep=3,
            done=False,
            _obs_cache={"robot": np.array([1.0, 2.0])},
            _observables={},
            _get_observations=mock.Mock(return_value={"observation": True}),
        )
        return SimpleNamespace(
            robots=[
                SimpleNamespace(
                    controller=controller,
                    gripper=SimpleNamespace(current_action=np.array([0.5])),
                )
            ],
            sim=SimpleNamespace(model=model),
            reset=mock.Mock(),
            regenerate_obs_from_state=mock.Mock(return_value={"observation": True}),
            env=runtime_env,
        )

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

    def test_runtime_snapshot_restores_osc_and_interpolator_state(self) -> None:
        env = self.fake_env()
        with mock.patch("libero_snapshot_collector._object_geom_ids", return_value=[0]):
            state = _capture_controller_state(env)
            env.robots[0].controller.goal_pos[:] = -1
            env.robots[0].controller.interpolator_pos.step = 0
            env.env.cur_time = 0.0
            env.env._obs_cache = {}
            _restore_snapshot(env, np.zeros((4,)), state)
        self.assertTrue(
            np.array_equal(env.robots[0].controller.goal_pos, np.array([0.2, 0.3, 0.4]))
        )
        self.assertEqual(env.robots[0].controller.interpolator_pos.step, 2)
        self.assertAlmostEqual(env.env.cur_time, 0.15)
        self.assertTrue(np.array_equal(env.env._obs_cache["robot"], np.array([1.0, 2.0])))
        env.robots[0].controller.reset_goal.assert_not_called()

    def test_legacy_snapshot_still_resets_controller_goal(self) -> None:
        env = self.fake_env()
        legacy = {
            "model_body_pos": np.zeros((2, 3)),
            "object_friction": np.ones((1, 3)),
            "gripper_action": np.array([0.5]),
        }
        with mock.patch("libero_snapshot_collector._object_geom_ids", return_value=[0]):
            _restore_snapshot(env, np.zeros((4,)), legacy)
        env.robots[0].controller.reset_goal.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
