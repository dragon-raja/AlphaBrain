import unittest

import numpy as np

from evaluate_physical_process_oracle import (
    aggregate_outcomes,
    find_stage_indices,
    preference_key,
    summarize_physical_trace,
    summarize_rows,
)


def state(*, grasped=False, z=0.9, distance=0.3, success=False):
    return {
        "grasped": grasped,
        "object_z": z,
        "bowl_xy_distance": distance,
        "success": success,
    }


class PhysicalProcessOracleTest(unittest.TestCase):
    def test_finds_feedback_and_first_post_feedback_regrasp(self):
        reference = {"grasped": np.asarray([0, 1, 0, 0, 1, 1], dtype=bool)}
        self.assertEqual(find_stage_indices(reference, 2), {"feedback": 2, "post_regrasp": 4})

    def test_feedback_next_stage_is_regrasp(self):
        result = summarize_physical_trace(
            [
                state(),
                state(distance=0.31),
                state(grasped=True, distance=0.30),
                state(grasped=True, z=0.92, distance=0.28),
                state(grasped=True, z=0.92, distance=0.27),
            ],
            stage="feedback",
        )
        self.assertTrue(result["next_stage_reached"])
        self.assertEqual(result["first_regrasp_step"], 3)
        self.assertTrue(result["lift_reached"])
        self.assertFalse(result["drop"])

    def test_post_regrasp_has_ordered_lift_then_transport_stages(self):
        lifted_only = summarize_physical_trace(
            [
                state(grasped=True),
                state(grasped=True, z=0.92, distance=0.2),
                state(grasped=True, z=0.92, distance=0.2),
            ],
            stage="post_regrasp",
        )
        transported = summarize_physical_trace(
            [
                state(grasped=True),
                state(grasped=True, z=0.92, distance=0.2),
                state(grasped=True, z=0.92, distance=0.2),
                state(grasped=True, z=0.92, distance=0.07),
                state(grasped=True, z=0.92, distance=0.07),
            ],
            stage="post_regrasp",
        )
        self.assertTrue(lifted_only["next_stage_reached"])
        self.assertFalse(lifted_only["transport_reached"])
        self.assertTrue(transported["next_stage_reached"])
        self.assertTrue(transported["transport_reached"])
        self.assertGreater(preference_key(transported), preference_key(lifted_only))

    def test_preference_avoids_regress_before_nonterminal_progress(self):
        safe = summarize_physical_trace(
            [state(grasped=True), state(grasped=True, z=0.91, distance=0.28)],
            stage="post_regrasp",
        )
        dropped = summarize_physical_trace(
            [state(grasped=True), state(grasped=False, z=0.93, distance=0.2)],
            stage="post_regrasp",
        )
        self.assertGreater(preference_key(safe), preference_key(dropped))

    def test_summary_reports_oracle_gain(self):
        base = {
            "success": False,
            "regress": False,
            "next_stage_reached": False,
            "transport_reached": False,
            "lift_reached": False,
            "stable_grasp_at_end": False,
            "drop": False,
            "progress_auc": 0.0,
            "object_to_bowl_progress": 0.0,
            "object_height_progress": 0.0,
        }
        good = {**base, "next_stage_reached": True, "stable_grasp_at_end": True}
        row = {
            "pair_id": "g0",
            "source_initial_state_index": 3,
            "stage": "feedback",
            "oracle_index": 1,
            "oracle_replay_match": True,
            "unique_outcome_signatures": 2,
            "candidates": [
                {"selection_summary": aggregate_outcomes([base])},
                {"selection_summary": aggregate_outcomes([good])},
            ],
        }
        summary = summarize_rows([row])["feedback"]
        self.assertEqual(summary["candidate_positive_coverage"], 1.0)
        self.assertEqual(summary["oracle_minus_sample0_next_stage"], 1.0)


if __name__ == "__main__":
    unittest.main()
