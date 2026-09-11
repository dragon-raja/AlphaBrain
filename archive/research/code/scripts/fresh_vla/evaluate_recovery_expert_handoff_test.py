import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import evaluate_recovery_expert_handoff as handoff


def state(*, grasped=False, object_z=0.0, distance=1.0, success=False):
    return {
        "grasped": grasped,
        "object_z": object_z,
        "bowl_xy_distance": distance,
        "success": success,
    }


class FakeEnv:
    def check_success(self):
        return False


class FakePolicy:
    def __init__(self):
        self.seeds = []

    def predict(self, observation, seed):
        del observation
        self.seeds.append(seed)
        return np.zeros((10, 7), dtype=np.float32)


class FakeReplayEnv:
    def __init__(self):
        self.sim_state = np.zeros(2, dtype=np.float64)
        gripper = type("Gripper", (), {})()
        gripper.current_action = np.zeros(2, dtype=np.float64)
        robot = type("Robot", (), {})()
        robot.gripper = gripper
        self.robots = [robot]

    def regenerate_obs_from_state(self, value):
        self.sim_state = np.asarray(value, dtype=np.float64).copy()
        return {"restored": True}

    def get_sim_state(self):
        return self.sim_state.copy()


class RecoveryExpertHandoffTest(unittest.TestCase):
    def test_decision_budget_matches_full_closed_loop_timeout(self):
        self.assertEqual(handoff.DECISION_TOTAL_ACTION_BUDGET, 320)
        self.assertEqual(
            handoff.DECISION_MAX_TEACHER_ACTIONS,
            handoff.DECISION_TOTAL_ACTION_BUDGET,
        )

    def test_milestones_require_ordered_dwell(self):
        values = (0, 0, 0, False, False, False)
        values = handoff._update_milestones(
            state(grasped=True, object_z=0.1, distance=0.01),
            initial_z=0.0,
            dwell_steps=2,
            grasp_run=values[0],
            lift_run=values[1],
            transport_run=values[2],
            regrasp_reached=values[3],
            lift_reached=values[4],
            transport_reached=values[5],
        )
        self.assertEqual(values[3:], (False, False, False))
        values = handoff._update_milestones(
            state(grasped=True, object_z=0.1, distance=0.01),
            initial_z=0.0,
            dwell_steps=2,
            grasp_run=values[0],
            lift_run=values[1],
            transport_run=values[2],
            regrasp_reached=values[3],
            lift_reached=values[4],
            transport_reached=values[5],
        )
        self.assertTrue(values[3])
        self.assertFalse(values[4])
        values = handoff._update_milestones(
            state(grasped=True, object_z=0.1, distance=0.01),
            initial_z=0.0,
            dwell_steps=2,
            grasp_run=values[0],
            lift_run=values[1],
            transport_run=values[2],
            regrasp_reached=values[3],
            lift_reached=values[4],
            transport_reached=values[5],
        )
        self.assertTrue(values[4])
        self.assertFalse(values[5])
        values = handoff._update_milestones(
            state(grasped=True, object_z=0.1, distance=0.01),
            initial_z=0.0,
            dwell_steps=2,
            grasp_run=values[0],
            lift_run=values[1],
            transport_run=values[2],
            regrasp_reached=values[3],
            lift_reached=values[4],
            transport_reached=values[5],
        )
        self.assertTrue(values[5])

    def test_teacher_endpoint_executes_reconstructed_teacher_online(self):
        actions = np.arange(35, dtype=np.float64).reshape(5, 7)
        observed_actions = []
        teacher = mock.Mock()
        teacher.done = False
        teacher.decide.side_effect = [
            SimpleNamespace(action=action) for action in actions
        ]

        def fake_step(env, action):
            del env
            observed_actions.append(np.asarray(action).copy())
            return {}

        with (
            mock.patch.object(handoff, "restore_runtime_snapshot", return_value={}),
            mock.patch.object(
                handoff,
                "make_reconstructed_teacher",
                return_value=teacher,
            ),
            mock.patch.object(handoff, "object_grasped", return_value=False),
            mock.patch.object(handoff, "_step", side_effect=fake_step),
            mock.patch.object(handoff, "capture_runtime_snapshot", return_value={}),
            mock.patch.object(
                handoff,
                "_physical_state",
                return_value=state(),
            ),
            mock.patch.object(
                handoff,
                "_observation_frame",
                return_value=np.zeros((2, 2, 3), dtype=np.uint8),
            ),
        ):
            endpoint = handoff.generate_teacher_endpoint(
                FakeEnv(),
                {},
                {"source": handoff.TEACHER_ACTION_SOURCE},
                method="teacher_h3",
                execution_horizon=3,
                total_action_budget=320,
                max_teacher_actions=320,
                stage_dwell_steps=2,
            )

        self.assertEqual(endpoint["teacher_actions"], 3)
        self.assertTrue(endpoint["criterion_reached"])
        np.testing.assert_array_equal(np.stack(observed_actions), actions[:3])
        self.assertEqual(teacher.decide.call_count, 3)

    def test_feedback_reconstruction_replays_prefix_before_slip_injection(self):
        env = FakeReplayEnv()
        reference = {
            "actions": np.arange(21, dtype=np.float64).reshape(3, 7),
            "sim_state": np.asarray(
                [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [9.0, 9.0]],
                dtype=np.float64,
            ),
            "gripper_action": np.zeros((4, 2), dtype=np.float64),
        }
        replayed = []

        def fake_step(current_env, action):
            replayed.append(np.asarray(action).copy())
            current_env.sim_state += 1.0
            return {}

        with (
            mock.patch.object(handoff, "_restore_recorded_state", return_value={}),
            mock.patch.object(handoff, "_step", side_effect=fake_step),
            mock.patch.object(
                handoff,
                "capture_runtime_snapshot",
                return_value={"snapshot": True},
            ),
        ):
            observation, snapshot, audit = handoff.reconstruct_feedback_snapshot(
                env,
                reference,
                3,
            )

        np.testing.assert_array_equal(np.stack(replayed), reference["actions"])
        np.testing.assert_array_equal(env.sim_state, reference["sim_state"][3])
        self.assertEqual(observation, {"restored": True})
        self.assertEqual(snapshot, {"snapshot": True})
        self.assertEqual(audit["prefix_actions_replayed"], 3)
        self.assertEqual(audit["post_injection_sim_max_abs_delta"], 0.0)

    def test_teacher_state_uses_branch_start_and_feedback_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "slipped.npz"
            np.savez_compressed(
                episode,
                teacher_phase=np.asarray(["lift"] * 5),
                action_phases=np.asarray(["approach", "lift", "lift", "lift"]),
            )
            reference = {
                "eef_pose": np.asarray(
                    [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
                "object_pose": np.asarray(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.7], [0.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
            }
            result = handoff.reconstruct_teacher_state(
                root,
                {
                    "prefix_steps": 1,
                    "event_time": 4,
                    "episode_files": {"slipped": "slipped.npz"},
                },
                reference,
                4,
            )

        self.assertEqual(result["phase"], "lift")
        self.assertEqual(result["phase_steps"], 3)
        self.assertEqual(result["initial_eef_xy"], [0.1, 0.2])
        self.assertEqual(result["initial_object_z"], 0.7)

    def test_milestone_dispatch(self):
        self.assertTrue(
            handoff._milestone_reached(
                "teacher_to_regrasp",
                regrasp_reached=True,
                lift_reached=False,
                transport_reached=False,
            )
        )
        self.assertFalse(
            handoff._milestone_reached(
                "teacher_to_transport",
                regrasp_reached=True,
                lift_reached=True,
                transport_reached=False,
            )
        )
        with self.assertRaisesRegex(ValueError, "no physical milestone"):
            handoff._milestone_reached(
                "teacher_h3",
                regrasp_reached=False,
                lift_reached=False,
                transport_reached=False,
            )

    def test_policy_continuation_uses_global_replan_slots(self):
        endpoint = {
            "endpoint_snapshot": {},
            "trace": [state()] * 13,
            "frames": [np.zeros((2, 2, 3), dtype=np.uint8)] * 13,
            "teacher_actions": 12,
            "next_global_replan_index": 4,
        }
        policy = FakePolicy()
        with (
            mock.patch.object(handoff, "restore_runtime_snapshot", return_value={}),
            mock.patch.object(handoff, "_step", return_value={}),
            mock.patch.object(handoff, "_physical_state", return_value=state()),
            mock.patch.object(
                handoff,
                "_observation_frame",
                return_value=np.zeros((2, 2, 3), dtype=np.uint8),
            ),
        ):
            row = handoff.rollout_policy_continuation(
                FakeEnv(),
                policy,
                endpoint,
                pair_id="pair",
                seed=41,
                repeat=2,
                execution_horizon=3,
                total_action_budget=18,
                stage_dwell_steps=2,
            )
        self.assertEqual(row["policy_actions"], 6)
        self.assertEqual(
            policy.seeds,
            [
                handoff.stable_seed(41, "pair", "expert_handoff", 2, 4),
                handoff.stable_seed(41, "pair", "expert_handoff", 2, 5),
            ],
        )

    def test_teacher_full_sanity_never_hands_control_to_policy(self):
        endpoint = {
            "method": handoff.EXPERT_SANITY_METHOD,
            "trace": [state(), state(success=True)],
            "frames": [np.zeros((2, 2, 3), dtype=np.uint8)] * 2,
            "teacher_actions": 1,
            "criterion_reached": True,
            "teacher_done": False,
            "teacher_success": True,
            "regrasp_reached": True,
            "lift_reached": True,
            "transport_reached": True,
        }
        policy = FakePolicy()
        result = handoff.summarize_method(
            FakeEnv(),
            policy,
            endpoint,
            pair_id="pair",
            seed=41,
            continuation_count=5,
            execution_horizon=3,
            total_action_budget=120,
            stage_dwell_steps=1,
            video_dir=None,
            video_repeats=0,
        )
        self.assertEqual(policy.seeds, [])
        self.assertEqual(result["summary"]["success_rate"], 1.0)
        self.assertEqual(result["continuations"][0]["policy_calls"], 0)


if __name__ == "__main__":
    unittest.main()
