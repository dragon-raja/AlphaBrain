from __future__ import annotations

import unittest

from build_view_repeatability_protocols import (
    categorize_all_states,
    choose_candidates,
    normalized_pose_distance,
    select_state_categories,
)


class ViewRepeatabilityTest(unittest.TestCase):
    def test_pose_distance(self) -> None:
        left = {"pose": {"azimuth_deg": 0, "elevation_deg": 0, "radius_scale": 1}}
        right = {"pose": {"azimuth_deg": 60, "elevation_deg": 0, "radius_scale": 1}}
        self.assertAlmostEqual(normalized_pose_distance(left, right), 1.0)

    def test_selects_sparse_broad_harm_and_mixed(self) -> None:
        rows = []
        for pair_key, canonical, successes in (
            ("fail-sparse", False, 1),
            ("fail-broad", False, 3),
            ("success-harm", True, 1),
            ("success-mixed", True, 2),
        ):
            for index in range(4):
                rows.append(
                    {
                        "pair_key": pair_key,
                        "selected_candidate_id": "canonical" if index == 0 else f"view-{index}",
                        "success": canonical if index == 0 else index < successes,
                    }
                )
        categories = select_state_categories(rows)
        self.assertEqual(categories["canonical_failure_sparse"], "fail-sparse")
        self.assertEqual(categories["canonical_failure_broad"], "fail-broad")
        self.assertEqual(categories["canonical_success_harm"], "success-harm")
        self.assertEqual(categories["canonical_success_mixed"], "success-mixed")

    def test_categorizes_every_state(self) -> None:
        rows = []
        for pair_key, canonical, successful_noncanonical in (
            ("none", False, 0),
            ("sparse", False, 1),
            ("broad", False, 5),
            ("harm", True, 2),
            ("robust", True, 9),
        ):
            for index in range(10):
                rows.append(
                    {
                        "pair_key": pair_key,
                        "selected_candidate_id": (
                            "canonical" if index == 0 else f"view-{index}"
                        ),
                        "success": (
                            canonical
                            if index == 0
                            else index <= successful_noncanonical
                        ),
                    }
                )
        categories = categorize_all_states(rows)
        self.assertEqual(categories["none"], "no_discovery_success")
        self.assertEqual(categories["sparse"], "canonical_failure_sparse")
        self.assertEqual(categories["broad"], "canonical_failure_broad")
        self.assertEqual(categories["harm"], "canonical_success_harm")
        self.assertEqual(categories["robust"], "canonical_success_broad")

    def test_candidate_selection_handles_no_success(self) -> None:
        values = []
        for index in range(10):
            candidate = "canonical" if index == 0 else f"view-{index}"
            values.append(
                {
                    "selected_candidate_id": candidate,
                    "success": False,
                    "pose": {"azimuth_deg": index * 8.0},
                    "initial_metrics": {
                        "task_entity_visibility": {"score": index / 100.0}
                    },
                }
            )
        selected = choose_candidates(values, "view-3", 8)
        self.assertEqual(len(selected), 8)
        self.assertIn("canonical", selected)
        self.assertIn("view-9", selected)


if __name__ == "__main__":
    unittest.main()
