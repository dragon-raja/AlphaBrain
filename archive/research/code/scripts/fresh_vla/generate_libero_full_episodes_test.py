import argparse
import unittest

import numpy as np

from generate_libero_full_episodes import build_quality_report, is_recoverability_failure, slip_offset_candidates


def group(pair_id="g0"):
    return {
        "pair_id": pair_id,
        "event_time": 10,
        "feedback_reveal_time": 10,
        "action_divergence_time": 10,
        "recovery_latency": 0,
        "pre_event_max_abs_error": {"agentview": 0.0, "wrist": 0.0, "robot_state": 0.0, "actions": 0.0},
        "branches": {
            "attached": {"final_success": True, "total_steps": 100, "regrasp_attempts": 0},
            "slipped": {"final_success": True, "total_steps": 140, "regrasp_attempts": 1},
        },
    }


class FullEpisodeGeneratorTest(unittest.TestCase):
    def test_candidate_retry_only_handles_physical_recoverability(self):
        self.assertTrue(is_recoverability_failure(RuntimeError("slipped teacher ended without success")))
        self.assertFalse(is_recoverability_failure(RuntimeError("invalid episode array")))

    def test_slip_candidates_preserve_magnitude_and_start_with_requested_offset(self):
        requested = np.asarray([-0.04, -0.03, -0.005])
        candidates = slip_offset_candidates(requested)
        np.testing.assert_allclose(candidates[0], requested)
        self.assertEqual(len(candidates), 8)
        self.assertEqual(len({tuple(np.round(row, 8)) for row in candidates}), 8)
        for candidate in candidates:
            self.assertAlmostEqual(np.linalg.norm(candidate[:2]), 0.05)
            self.assertEqual(candidate[2], requested[2])

    def test_quality_gate_requires_both_complete_branches(self):
        args = argparse.Namespace(group_count=2)
        report = build_quality_report([group("g0"), group("g1")], args)
        self.assertTrue(report["passed"])
        self.assertEqual(report["metrics"]["slipped_success_rate"], 1.0)

        failed = group("g1")
        failed["branches"]["slipped"]["final_success"] = False
        report = build_quality_report([group("g0"), failed], args)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["slipped_teacher_recovery_success"])

    def test_quality_gate_rejects_visual_leak(self):
        args = argparse.Namespace(group_count=1)
        leaked = group()
        leaked["feedback_reveal_time"] = 9
        report = build_quality_report([leaked], args)
        self.assertFalse(report["checks"]["no_early_visual_leak"])

    def test_quality_gate_rejects_changed_slip_magnitude(self):
        changed = group()
        changed["branches"]["slipped"].update(
            requested_slip_offset=[0.03, 0.04, -0.005],
            applied_slip_offset=[0.01, 0.01, -0.005],
        )
        report = build_quality_report([changed], argparse.Namespace(group_count=1))
        self.assertFalse(report["checks"]["slip_candidate_magnitude_preserved"])


if __name__ == "__main__":
    unittest.main()
