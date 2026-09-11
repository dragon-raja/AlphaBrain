import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np

from evaluate_libero_closed_loop import (
    _evaluation_payload,
    is_failure_continuation,
    is_premature_commitment,
    is_recovery_action,
    progress_fraction,
    stable_seed,
    slip_offset_for_group,
    summarize_rows,
    update_subgoals,
)


class ClosedLoopEvaluatorTest(unittest.TestCase):
    def test_closed_loop_reuses_collector_applied_slip_offset(self):
        group = {
            "source_randomization": {"slip_offset": [1.0, 0.0, 0.0]},
            "branches": {"slipped": {"applied_slip_offset": [0.0, 1.0, 0.0]}},
        }
        self.assertEqual(slip_offset_for_group(group), [0.0, 1.0, 0.0])

    def test_stable_seed_depends_on_group_and_replan(self):
        self.assertEqual(stable_seed("a", 1), stable_seed("a", 1))
        self.assertNotEqual(stable_seed("a", 1), stable_seed("a", 2))

    def test_paired_branches_share_the_same_noise_schedule(self):
        attached = [stable_seed(41, "group", replan) for replan in range(4)]
        slipped = [stable_seed(41, "group", replan) for replan in range(4)]
        self.assertEqual(attached, slipped)

    def test_failure_continuation_requires_ungrasped_closed_lift(self):
        action = np.asarray([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
        self.assertTrue(is_failure_continuation(action, grasped=False))
        self.assertFalse(is_failure_continuation(action, grasped=True))

    def test_recovery_detects_open_or_motion_to_object(self):
        self.assertTrue(
            is_recovery_action(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                grasped=False,
                eef_position=[0.0, 0.0, 1.0],
                object_position=[0.0, 0.0, 0.9],
            )
        )
        self.assertTrue(
            is_recovery_action(
                [0.0, 0.0, -0.8, 0.0, 0.0, 0.0, 1.0],
                grasped=False,
                eef_position=[0.0, 0.0, 1.0],
                object_position=[0.0, 0.0, 0.9],
            )
        )

    def test_premature_commitment_requires_closed_ungrasped_motion(self):
        self.assertTrue(
            is_premature_commitment(
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                grasped=False,
                eef_position=[0.0, 0.0, 1.0],
                bowl_position=[1.0, 0.0, 1.0],
            )
        )

    def test_progress_subgoals_are_cumulative(self):
        observation = {
            "cream_cheese_1_pos": np.asarray([0.0, 0.0, 0.92]),
            "akita_black_bowl_1_pos": np.asarray([0.02, 0.01, 0.90]),
        }
        state = update_subgoals(
            {}, observation, grasped=True, success=False, initial_object_z=0.90
        )
        self.assertEqual(state, {"grasp": True, "lift": True, "transport": True, "place": False})
        self.assertEqual(progress_fraction(state), 0.75)

        moved_away = {
            "cream_cheese_1_pos": np.asarray([0.8, 0.8, 0.90]),
            "akita_black_bowl_1_pos": np.asarray([0.0, 0.0, 0.90]),
        }
        state = update_subgoals(
            state, moved_away, grasped=False, success=True, initial_object_z=0.90
        )
        self.assertTrue(all(state.values()))

    def test_summary_keeps_snapshot_group_count(self):
        rows = [
            {
                "pair_id": "g0",
                "branch_outcome": "attached",
                "success": True,
                "recovery_success": False,
                "failure_continuation": False,
                "premature_commitment": False,
                "recovery_switch_latency": None,
                "regrasp_success": None,
                "drop": False,
                "final_progress": 1.0,
                "progress_auc": 0.8,
                "grasp_subgoal": True,
                "lift_subgoal": True,
                "transport_subgoal": True,
                "place_subgoal": True,
                "event_time": 3,
                "completion_steps": 10,
            },
            {
                "pair_id": "g0",
                "branch_outcome": "slipped",
                "success": False,
                "recovery_success": False,
                "failure_continuation": True,
                "premature_commitment": True,
                "recovery_switch_latency": 2,
                "regrasp_success": False,
                "drop": True,
                "final_progress": 0.5,
                "progress_auc": 0.4,
                "grasp_subgoal": True,
                "lift_subgoal": True,
                "transport_subgoal": False,
                "place_subgoal": False,
                "event_time": 3,
                "completion_steps": 20,
            },
        ]
        summary = summarize_rows(rows)
        self.assertEqual(summary["group_count"], 1)
        self.assertEqual(summary["overall_task_success"], 0.5)
        self.assertEqual(summary["failure_continuation_rate"], 1.0)
        self.assertEqual(summary["overall_drop_rate"], 0.5)
        self.assertEqual(summary["mean_final_progress"], 0.75)

    def test_partial_payload_tracks_progress_without_inventing_groups(self):
        args = Namespace(
            checkpoint=None,
            policy_socket=Path("/tmp/policy.sock"),
            episode_root=Path("/episodes"),
            evaluation="isolated",
            split="test",
            seed=41,
            execution_horizons=(1, 2),
        )
        rows = [
            {
                "pair_id": "g0",
                "branch_outcome": "attached",
                "execution_horizon": 1,
                "success": False,
                "recovery_success": False,
                "failure_continuation": None,
                "premature_commitment": None,
                "recovery_switch_latency": None,
                "completion_steps": 10,
            }
        ]
        payload = _evaluation_payload(args, rows, expected_rows=8, status="partial")
        self.assertEqual(payload["status"], "partial")
        self.assertEqual(payload["completed_rows"], 1)
        self.assertEqual(payload["expected_rows"], 8)
        self.assertEqual(payload["summary"]["1"]["group_count"], 1)
        self.assertEqual(payload["summary"]["2"]["group_count"], 0)


if __name__ == "__main__":
    unittest.main()
