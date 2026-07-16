from __future__ import annotations

import unittest

import numpy as np

from collect_policy_state_recovery import (
    build_quality_report,
    correction_rejection_reason,
    intervention_reason,
    restore_teacher,
    serialize_teacher,
    write_paired_correction_video,
)
from libero_full_episode_collector import FullEpisodeTeacher


class InterventionReasonTest(unittest.TestCase):
    def test_failure_continuation_has_priority(self) -> None:
        reason = intervention_reason(
            [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
            grasped=False,
            eef_position=[0.0, 0.0, 0.0],
            bowl_position=[1.0, 0.0, 1.0],
        )
        self.assertEqual(reason, "failure_continuation")

    def test_returns_none_for_opening_hold(self) -> None:
        reason = intervention_reason(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            grasped=False,
            eef_position=[0.0, 0.0, 0.0],
            bowl_position=[1.0, 0.0, 0.0],
        )
        self.assertIsNone(reason)


class TeacherStateTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        observation = {
            "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3]),
            "cream_cheese_1_pos": np.asarray([0.4, 0.5, 0.6]),
        }
        teacher = FullEpisodeTeacher(observation)
        teacher.phase = "recover_close"
        teacher.phase_steps = 9
        teacher.regrasp_attempts = 1
        state = serialize_teacher(teacher)
        restored = restore_teacher(observation, state)
        self.assertEqual(serialize_teacher(restored), state)


class QualityReportTest(unittest.TestCase):
    def test_passes_valid_minimal_payload(self) -> None:
        reconstruction = {
            "post_injection_sim_max_abs_delta": 0.0,
            "prefix_gripper_max_abs_delta": 0.0,
            "policy_image_max_abs_delta": 0,
            "policy_robot_state_max_abs_delta": 0.0,
        }
        groups = [
            {
                "pair_id": "g0",
                "split": "train",
                "source_initial_state_index": 0,
                "retained": True,
                "trigger_reason": "failure_continuation",
                "policy_prefix_actions": 1,
                "teacher_correction_actions": 20,
                "stable_regrasp_reached": True,
                "feedback_reconstruction": reconstruction,
                "full_teacher_audit": {"success": True},
                "frozen_policy_audit": {"success": True},
            }
        ]
        records = [
            {
                "observation": {"agentview_path": "a", "wrist_path": "w"},
                "action_chunk": np.zeros((10, 7)).tolist(),
            }
        ]
        report = build_quality_report(
            groups,
            records,
            requested_group_count=1,
            minimum_correction_group_rate=0.8,
            paired_video_count=1,
            requested_video_count=1,
        )
        self.assertTrue(report["passed"])

    def test_expected_teacher_failure_is_a_group_level_rejection(self) -> None:
        self.assertEqual(
            correction_rejection_reason(
                RuntimeError("teacher correction did not reach stable regrasp")
            ),
            "teacher correction did not reach stable regrasp",
        )

    def test_unknown_runtime_error_remains_fatal(self) -> None:
        self.assertIsNone(correction_rejection_reason(RuntimeError("unexpected simulator bug")))


class PairedVideoTest(unittest.TestCase):
    def test_rejects_mismatched_frame_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "same frame shape"):
            write_paired_correction_video(
                None,
                [np.zeros((8, 8, 3), dtype=np.uint8)],
                [np.zeros((9, 8, 3), dtype=np.uint8)],
            )


if __name__ == "__main__":
    unittest.main()
