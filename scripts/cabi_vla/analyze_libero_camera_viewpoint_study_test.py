from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from analyze_libero_camera_viewpoint_study import (
    analyze_payloads,
    main,
    sampled_true_intervals,
    visibility_phase,
)


def visibility_row(
    value: float,
    *,
    pose: str | None = None,
    axis: str = "azimuth_deg",
    source_clipping: float = 0.0,
    target_clipping: float = 0.0,
    source_center: bool = True,
    target_center: bool = True,
    source_pixels: int = 200,
    target_pixels: int = 200,
    source_patches: int = 4,
    target_patches: int = 4,
    state: int = 0,
) -> dict:
    baseline = pose == "baseline"
    return {
        "edge_id": "red-left",
        "canonical_state_index": state,
        "camera_pose": pose or f"pose_{value:g}",
        "sweep_axis": "baseline" if baseline else axis,
        "sweep_value": 0.0 if baseline else value,
        "camera_azimuth_deg": 0.0 if baseline else value,
        "camera_elevation_deg": 0.0,
        "camera_radius_scale": 1.0,
        "source_center_in_frame": source_center,
        "source_fov_clipping_fraction": source_clipping,
        "source_visible_pixels": source_pixels,
        "source_visible_patch_support": source_patches,
        "target_center_in_frame": target_center,
        "target_fov_clipping_fraction": target_clipping,
        "target_visible_pixels": target_pixels,
        "target_visible_patch_support": target_patches,
    }


class CameraViewpointStudyTest(unittest.TestCase):
    def test_visibility_phase_uses_most_severe_boundary(self) -> None:
        point = {
            "task_visible_pixels_min": 100,
            "task_visible_patch_support_min": 5,
            "task_center_in_frame_all": True,
            "task_fov_clipping_fraction_max": 0.2,
        }
        self.assertEqual(
            visibility_phase(point, minimum_patch_support=4),
            "partial_10_50",
        )
        point["task_center_in_frame_all"] = False
        self.assertEqual(
            visibility_phase(point, minimum_patch_support=4),
            "center_out",
        )
        point["task_visible_pixels_min"] = 0
        self.assertEqual(
            visibility_phase(point, minimum_patch_support=4),
            "disappeared",
        )

    def test_sampled_intervals_preserve_non_monotonic_runs(self) -> None:
        points = [
            {"sweep_value": -3.0, "bad": True},
            {"sweep_value": -2.0, "bad": True},
            {"sweep_value": -1.0, "bad": False},
            {"sweep_value": 0.0, "bad": True},
            {"sweep_value": 1.0, "bad": False},
            {"sweep_value": 2.0, "bad": True},
        ]
        intervals = sampled_true_intervals(points, lambda point: bool(point["bad"]))
        self.assertEqual(
            [(value["start_value"], value["end_value"]) for value in intervals],
            [(-3.0, -2.0), (0.0, 0.0), (2.0, 2.0)],
        )
        self.assertEqual(intervals[0]["upper_clear_value"], -1.0)
        self.assertEqual(intervals[1]["lower_clear_value"], -1.0)
        self.assertEqual(intervals[1]["upper_clear_value"], 1.0)

    def test_boundaries_use_worst_state_and_report_every_interval(self) -> None:
        rows = [
            visibility_row(0.0, pose="baseline", state=0),
            visibility_row(0.0, pose="baseline", state=1),
            visibility_row(-2.0, source_clipping=0.2),
            visibility_row(-1.0, source_clipping=0.2),
            visibility_row(1.0, source_clipping=0.2),
            visibility_row(2.0),
            visibility_row(
                3.0,
                source_clipping=0.6,
                source_center=False,
                source_pixels=0,
                source_patches=0,
            ),
            visibility_row(2.0, state=1, target_pixels=40, target_patches=1),
        ]
        points, boundaries = analyze_payloads(
            [{"status": "complete", "rows": rows}],
            minimum_patch_support=2,
        )
        point_at_two = next(
            point for point in points if point["sweep_value"] == 2.0
        )
        self.assertEqual(point_at_two["task_visible_pixels_min"], 40)
        self.assertEqual(point_at_two["task_visible_patch_support_min"], 1)

        clipping = next(
            boundary
            for boundary in boundaries
            if boundary["object_scope"] == "task"
            and boundary["boundary"] == "clipping_10_percent"
        )
        self.assertEqual(
            [
                (interval["start_value"], interval["end_value"])
                for interval in clipping["intervals"]
            ],
            [(-2.0, -1.0), (1.0, 1.0), (3.0, 3.0)],
        )
        self.assertEqual(clipping["first_reached_lower"], -1.0)
        self.assertEqual(clipping["first_reached_upper"], 1.0)

        low_pixels = next(
            boundary
            for boundary in boundaries
            if boundary["object_scope"] == "task"
            and boundary["boundary"] == "below_64_visible_pixels"
        )
        self.assertEqual(
            [
                (interval["start_value"], interval["end_value"])
                for interval in low_pixels["intervals"]
            ],
            [(2.0, 3.0)],
        )
        expected_intervals = {
            "first_clipping": [(-2.0, -1.0), (1.0, 1.0), (3.0, 3.0)],
            "clipping_50_percent": [(3.0, 3.0)],
            "center_out_of_frame": [(3.0, 3.0)],
            "below_patch_support": [(2.0, 3.0)],
            "fully_disappeared": [(3.0, 3.0)],
        }
        for boundary_name, expected in expected_intervals.items():
            boundary = next(
                value
                for value in boundaries
                if value["object_scope"] == "task"
                and value["boundary"] == boundary_name
            )
            self.assertEqual(
                [
                    (interval["start_value"], interval["end_value"])
                    for interval in boundary["intervals"]
                ],
                expected,
            )

    def test_radius_axis_inserts_baseline_at_one(self) -> None:
        rows = [
            visibility_row(0.0, pose="baseline"),
            visibility_row(0.8, axis="radius_scale"),
            visibility_row(1.2, axis="radius_scale"),
        ]
        points, _ = analyze_payloads([{"status": "complete", "rows": rows}])
        self.assertEqual(
            [point["sweep_value"] for point in points],
            [0.8, 1.0, 1.2],
        )
        baseline = next(point for point in points if point["sweep_value"] == 1.0)
        self.assertEqual(baseline["camera_poses"], ["baseline"])
        self.assertEqual(baseline["neutral_value"], 1.0)

    def test_policy_curve_and_cli_outputs(self) -> None:
        rows = [
            visibility_row(0.0, pose="baseline"),
            visibility_row(-1.0, source_clipping=0.1),
            visibility_row(1.0, source_clipping=0.5),
        ]
        policy_rows = [
            {"edge_id": "red-left", "camera_pose": "baseline", "success": True},
            {"edge_id": "red-left", "camera_pose": "pose_-1", "success": False},
            {"edge_id": "red-left", "camera_pose": "pose_1", "success": True},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fov_path = root / "fov.json"
            policy_path = root / "policy.json"
            output = root / "analysis"
            fov_path.write_text(
                json.dumps({"status": "complete", "rows": rows}) + "\n"
            )
            policy_path.write_text(
                json.dumps({"status": "complete", "rows": policy_rows}) + "\n"
            )

            main(
                [
                    "--fov-json",
                    str(fov_path),
                    "--policy-json",
                    str(policy_path),
                    "--output-dir",
                    str(output),
                    "--minimum-patch-support",
                    "2",
                ]
            )

            report = json.loads((output / "summary.json").read_text())
            by_value = {
                point["sweep_value"]: point for point in report["curve_points"]
            }
            self.assertEqual(by_value[-1.0]["policy_mean"], 0.0)
            self.assertEqual(by_value[0.0]["policy_mean"], 1.0)
            self.assertEqual(by_value[1.0]["policy_mean"], 1.0)
            with (output / "summary.csv").open(newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertTrue(
                any(
                    row["boundary"] == "clipping_50_percent"
                    and row["interval_start"] == "1.0"
                    for row in summary_rows
                )
            )
            with Image.open(output / "camera_viewpoint_curves.png") as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.width, 500)
                self.assertGreater(image.height, 250)
            with Image.open(output / "camera_fov_phase_map.png") as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.width, 600)
                self.assertGreater(image.height, 150)
            with Image.open(output / "curves_by_edge" / "red-left.png") as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.width, 500)
            self.assertTrue((output / "curve_points.csv").is_file())


if __name__ == "__main__":
    unittest.main()
