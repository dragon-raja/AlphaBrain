from __future__ import annotations

import unittest

from evaluate_libero_bind_camera_rule import evaluate_rule, select_rule_rows


class EvaluateCameraRuleTest(unittest.TestCase):
    def test_rule_selects_only_the_configured_edge_pose(self) -> None:
        rows = []
        for pose in ("baseline", "near"):
            for edge in ("red-left", "yellow-right"):
                rows.append(
                    {
                        "camera_pose": pose,
                        "edge_id": edge,
                        "canonical_state_index": 5,
                        "execution_horizon": 3,
                    }
                )
        baseline, selected = select_rule_rows(
            rows,
            {
                "default_pose": "baseline",
                "edge_pose": {"yellow-right": "near"},
            },
        )
        self.assertEqual(len(baseline), 2)
        selected_by_edge = {
            row["edge_id"]: row["source_camera_pose"] for row in selected
        }
        self.assertEqual(selected_by_edge["red-left"], "baseline")
        self.assertEqual(selected_by_edge["yellow-right"], "near")

    def test_mixed_rule_does_not_report_a_single_radius(self) -> None:
        rows = []
        for pose, radius in (("baseline", 1.0), ("near", 0.925)):
            for edge in ("red-left", "yellow-right"):
                rows.append(
                    {
                        "camera_pose": pose,
                        "edge_id": edge,
                        "canonical_state_index": 5,
                        "execution_horizon": 3,
                        "camera_azimuth_deg": 0.0,
                        "camera_elevation_deg": 0.0,
                        "camera_radius_scale": radius,
                        "success": True,
                        "source_selection_success": True,
                        "wrong_source_grasp": False,
                        "lift_success": True,
                        "transport_success": True,
                        "progress": 1.0,
                        "completion_steps": 100,
                    }
                )
        result = evaluate_rule(
            {"rows": rows},
            {
                "default_pose": "baseline",
                "edge_pose": {"yellow-right": "near"},
            },
        )
        summary = next(
            row for row in result["summaries"] if row["camera_pose"] == "camera_rule"
        )
        self.assertIsNone(summary["radius_scale"])
        self.assertEqual(summary["source_camera_poses"], ["baseline", "near"])


if __name__ == "__main__":
    unittest.main()
