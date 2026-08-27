from __future__ import annotations

import unittest

from build_view_repeatability_protocols import normalized_pose_distance, select_state_categories


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


if __name__ == "__main__":
    unittest.main()
