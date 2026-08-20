from __future__ import annotations

import unittest

import numpy as np

from accel_core import (
    analyze_selected_relations,
    audit_shared_flow_noise,
    compute_accel_scores,
    rank_accel_candidates,
    shared_flow_noise,
)


class AccelCoreTest(unittest.TestCase):
    def test_computes_registered_prefix_formula(self) -> None:
        trace = np.asarray([[[0.0], [1.0], [3.0]]], dtype=np.float32)
        metrics = compute_accel_scores(trace, prefixes=(2, 3))
        self.assertAlmostEqual(float(metrics[2]["score"][0]), 2.0)
        self.assertAlmostEqual(float(metrics[3]["score"][0]), 2.25)
        self.assertFalse(bool(metrics[3]["degenerate"][0]))

    def test_constant_velocity_has_zero_non_degenerate_accel(self) -> None:
        trace = np.ones((2, 10, 3, 2), dtype=np.float32)
        metrics = compute_accel_scores(trace)
        np.testing.assert_array_equal(metrics[3]["score"], np.zeros(2))
        np.testing.assert_array_equal(metrics[3]["degenerate"], np.zeros(2, dtype=bool))

    def test_zero_velocity_is_degenerate_and_not_selected(self) -> None:
        trace = np.zeros((2, 10, 1), dtype=np.float32)
        trace[1] = 1.0
        ranking = rank_accel_candidates(["zero", "valid"], trace)
        self.assertEqual(ranking["selected_candidate_id"], "valid")
        self.assertFalse(ranking["ranking"][0]["accel_3_degenerate"])
        self.assertTrue(ranking["ranking"][1]["accel_3_degenerate"])

    def test_all_zero_velocity_returns_no_valid_candidate(self) -> None:
        ranking = rank_accel_candidates(
            ["left", "right"], np.zeros((2, 10, 1), dtype=np.float32)
        )
        self.assertEqual(ranking["status"], "NO_VALID_CANDIDATE")
        self.assertIsNone(ranking["selected_candidate_id"])

    def test_shared_noise_is_exact_and_reproducible(self) -> None:
        first = shared_flow_noise(
            seed=41, candidate_count=4, action_horizon=5, action_dim=3
        )
        second = shared_flow_noise(
            seed=41, candidate_count=4, action_horizon=5, action_dim=3
        )
        np.testing.assert_array_equal(first, second)
        audit = audit_shared_flow_noise(first)
        self.assertTrue(audit["exactly_shared"])
        self.assertEqual(audit["max_abs_difference"], 0.0)

    def test_unshared_noise_invalidates_ranking(self) -> None:
        trace = np.ones((2, 10, 1), dtype=np.float32)
        noise = np.zeros((2, 2, 1), dtype=np.float32)
        noise[1, 0, 0] = 1.0
        ranking = rank_accel_candidates(
            ["left", "right"], trace, initial_noise=noise
        )
        self.assertEqual(ranking["status"], "INVALID_UNSHARED_FLOW_NOISE")
        self.assertIsNone(ranking["selected_candidate_id"])

    def test_relation_analysis_reports_exact_and_nearest_sets(self) -> None:
        trace = np.ones((3, 10, 1), dtype=np.float32)
        trace[0, 1:, 0] = np.arange(1, 10)
        trace[2, 1:, 0] = np.arange(1, 10) * 2
        ranking = rank_accel_candidates(["canonical", "info", "reveal"], trace)
        candidates = [
            {
                "candidate_id": "canonical",
                "azimuth_deg": 0.0,
                "elevation_deg": 0.0,
                "radius_scale": 1.0,
            },
            {
                "candidate_id": "info",
                "azimuth_deg": 20.0,
                "elevation_deg": 5.0,
                "radius_scale": 1.0,
            },
            {
                "candidate_id": "reveal",
                "azimuth_deg": 21.0,
                "elevation_deg": 5.0,
                "radius_scale": 1.0,
            },
        ]
        relations = analyze_selected_relations(
            ranking,
            candidates,
            {
                "canonical": "canonical",
                "train": ["canonical", "info"],
                "strong_info": "info",
                "reveal": "reveal",
                "oracle": "info",
            },
        )
        self.assertEqual(relations["selected_candidate_id"], "info")
        self.assertTrue(relations["relations"]["strong_info"]["selected_exact_match"])
        self.assertTrue(relations["relations"]["train"]["selected_exact_match"])
        self.assertEqual(
            relations["relations"]["reveal"]["nearest_reference_candidate_id"],
            "reveal",
        )

    def test_relation_analysis_requires_all_frozen_reference_sets(self) -> None:
        ranking = rank_accel_candidates(
            ["canonical"], np.ones((1, 10, 1), dtype=np.float32)
        )
        with self.assertRaisesRegex(ValueError, "missing required relations"):
            analyze_selected_relations(
                ranking,
                [{"candidate_id": "canonical"}],
                {"canonical": "canonical"},
            )


if __name__ == "__main__":
    unittest.main()
