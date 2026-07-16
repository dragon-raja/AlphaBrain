from __future__ import annotations

import unittest

import numpy as np

from diagnose_counterfactual_feedback_revision import cosine, revision_metrics, summarize_rows


class RevisionMetricsTest(unittest.TestCase):
    def test_detects_correct_counterfactual_revision(self) -> None:
        attached = np.zeros((2, 7), dtype=np.float32)
        slipped = np.ones((2, 7), dtype=np.float32)
        result = revision_metrics(
            stale=attached,
            fresh_attached=attached,
            fresh_slipped=slipped,
            teacher_attached=attached,
            teacher_slipped=slipped,
        )
        self.assertTrue(result["fresh_pair_assignment_correct"])
        self.assertEqual(result["fresh_joint_mse"], 0.0)
        self.assertEqual(result["slipped_stale_minus_fresh_mse"], 1.0)
        self.assertAlmostEqual(result["fresh_revision_alignment"], 1.0)

    def test_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            revision_metrics(
                np.zeros((1, 7)),
                np.zeros((2, 7)),
                np.zeros((1, 7)),
                np.zeros((1, 7)),
                np.zeros((1, 7)),
            )

    def test_cosine_is_none_for_zero_direction(self) -> None:
        self.assertIsNone(cosine(np.zeros(3), np.ones(3)))


class SummarizeRowsTest(unittest.TestCase):
    def test_aggregates_by_age_and_horizon(self) -> None:
        rows = [
            {
                "pair_id": "g0",
                "source_initial_state_index": 0,
                "stale_age": 1,
                "horizon": 1,
                "stale_slipped_mse": 2.0,
                "fresh_slipped_mse": 1.0,
                "fresh_pair_assignment_correct": True,
                "fresh_revision_alignment": None,
            },
            {
                "pair_id": "g1",
                "source_initial_state_index": 1,
                "stale_age": 1,
                "horizon": 1,
                "stale_slipped_mse": 4.0,
                "fresh_slipped_mse": 1.0,
                "fresh_pair_assignment_correct": False,
                "fresh_revision_alignment": None,
            },
        ]
        result = summarize_rows(rows)["age1_h1"]
        self.assertEqual(result["group_count"], 2)
        self.assertEqual(result["source_initial_state_count"], 2)
        self.assertEqual(result["means"]["fresh_pair_assignment_correct"], 0.5)
        self.assertAlmostEqual(result["means"]["relative_slipped_mse_reduction"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
