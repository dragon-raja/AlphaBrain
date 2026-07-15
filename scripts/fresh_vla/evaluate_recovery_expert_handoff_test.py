import unittest
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


class RecoveryExpertHandoffTest(unittest.TestCase):
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
