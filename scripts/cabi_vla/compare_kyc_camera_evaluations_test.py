from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from compare_kyc_camera_evaluations import (
    compare_evaluations,
    index_fov_rows,
    join_episode_rows,
    main,
    paired_state_bootstrap,
    validate_paired_evaluations,
)


def evaluation_row(
    edge: str,
    state: int,
    pose: str,
    *,
    supervised: bool,
    success: bool,
    horizon: int = 3,
) -> dict:
    return {
        "edge_id": edge,
        "canonical_state_index": state,
        "execution_horizon": horizon,
        "camera_pose": pose,
        "action_supervised": supervised,
        "success": success,
        "progress": 1.0 if success else 0.25,
        "source_selection_success": success,
        "lift_success": success,
        "transport_success": success,
        "target_placement_success": success,
    }


def evaluation_payload(rows: list[dict]) -> dict:
    return {"status": "complete", "rows": rows}


def fov_row(
    edge: str,
    state: int,
    pose: str,
    value: float,
    *,
    supervised: bool,
    clipping: float = 0.0,
    pixels: int = 100,
    patches: int = 4,
    center: bool = True,
    axis: str = "azimuth_deg",
) -> dict:
    baseline = pose == "baseline"
    return {
        "edge_id": edge,
        "canonical_state_index": state,
        "camera_pose": pose,
        "action_supervised": supervised,
        "sweep_axis": "baseline" if baseline else axis,
        "sweep_value": 0.0 if baseline else value,
        "camera_azimuth_deg": 0.0 if baseline else value,
        "camera_elevation_deg": 0.0,
        "camera_radius_scale": 1.0,
        "source_center_in_frame": center,
        "source_fov_clipping_fraction": clipping,
        "source_visible_pixels": pixels,
        "source_visible_patch_support": patches,
        "target_center_in_frame": True,
        "target_fov_clipping_fraction": 0.0,
        "target_visible_pixels": 120,
        "target_visible_patch_support": 5,
    }


class CompareKycCameraEvaluationsTest(unittest.TestCase):
    def test_pairing_requires_identical_episode_keys(self) -> None:
        reference = evaluation_payload(
            [evaluation_row("observed", 0, "baseline", supervised=True, success=True)]
        )
        missing = evaluation_payload(
            [evaluation_row("observed", 0, "az_p10", supervised=True, success=True)]
        )
        with self.assertRaisesRegex(ValueError, "episode keys differ"):
            validate_paired_evaluations({"reference": reference, "method": missing})

        duplicate_row = evaluation_row(
            "observed", 0, "baseline", supervised=True, success=True
        )
        duplicate = evaluation_payload([duplicate_row, dict(duplicate_row)])
        with self.assertRaisesRegex(ValueError, "duplicate episode key"):
            validate_paired_evaluations({"reference": duplicate, "method": duplicate})

    def test_fov_join_ignores_horizon_and_assigns_all_strata(self) -> None:
        cases = [
            ("fully", 0.2, 100, 4, True, "fully_supported"),
            ("severe", 0.7, 100, 4, True, "severe_clipping"),
            ("center", 0.2, 100, 4, False, "center_out"),
            ("below", 0.2, 63, 3, True, "below_support"),
            ("gone", 1.0, 0, 0, False, "disappeared"),
        ]
        evaluation_rows = []
        fov_rows = []
        for index, (pose, clipping, pixels, patches, center, _expected) in enumerate(cases):
            evaluation_rows.append(
                evaluation_row(
                    "observed",
                    0,
                    pose,
                    supervised=True,
                    success=True,
                    horizon=1 if index % 2 else 3,
                )
            )
            fov_rows.append(
                fov_row(
                    "observed",
                    0,
                    pose,
                    float(index),
                    supervised=True,
                    clipping=clipping,
                    pixels=pixels,
                    patches=patches,
                    center=center,
                )
            )
        payload = evaluation_payload(evaluation_rows)
        indexed = validate_paired_evaluations({"reference": payload, "method": payload})
        joined = join_episode_rows(
            indexed,
            index_fov_rows([{"status": "complete", "rows": fov_rows}]),
        )
        reference_rows = {
            row["camera_pose"]: row
            for row in joined
            if row["method"] == "reference"
        }
        self.assertEqual(
            {
                pose: reference_rows[pose]["visibility_stratum"]
                for pose, *_rest in cases
            },
            {pose: expected for pose, *_values, expected in cases},
        )
        self.assertEqual(reference_rows["severe"]["task_max_fov_clipping_fraction"], 0.7)
        self.assertEqual(reference_rows["below"]["task_min_visible_pixels"], 63)

    def test_identical_baseline_can_repeat_across_fov_files(self) -> None:
        baseline = fov_row(
            "observed",
            0,
            "baseline",
            0.0,
            supervised=True,
        )
        indexed = index_fov_rows(
            [
                {"status": "complete", "rows": [baseline]},
                {"status": "complete", "rows": [dict(baseline)]},
            ]
        )
        self.assertEqual(len(indexed), 1)
        conflicting = dict(baseline)
        conflicting["source_visible_pixels"] = 99
        with self.assertRaisesRegex(ValueError, "conflicting duplicate FOV key"):
            index_fov_rows(
                [
                    {"status": "complete", "rows": [baseline]},
                    {"status": "complete", "rows": [conflicting]},
                ]
            )

    def test_paired_bootstrap_uses_state_group_means(self) -> None:
        result = paired_state_bootstrap(
            {0: 0.0, 1: 1.0},
            {0: 1.0, 1: 1.0},
            resamples=200,
            seed=7,
        )
        self.assertEqual(result["delta"], 0.5)
        self.assertEqual(result["paired_state_count"], 2)
        self.assertEqual(result["ci95_low"], 0.0)
        self.assertEqual(result["ci95_high"], 1.0)

    def test_cli_generates_json_csv_and_png(self) -> None:
        reference_rows = []
        method_rows = []
        geometry_rows = []
        poses = (("az_m70", -70.0), ("baseline", 0.0), ("az_p70", 70.0))
        for edge, supervised in (("observed", True), ("withheld", False)):
            for state in (0, 1):
                for pose, value in poses:
                    reference_rows.append(
                        evaluation_row(
                            edge,
                            state,
                            pose,
                            supervised=supervised,
                            success=pose == "baseline",
                        )
                    )
                    method_rows.append(
                        evaluation_row(
                            edge,
                            state,
                            pose,
                            supervised=supervised,
                            success=pose != "az_m70",
                        )
                    )
                    geometry_rows.append(
                        fov_row(
                            edge,
                            state,
                            pose,
                            value,
                            supervised=supervised,
                            clipping=0.6 if pose != "baseline" else 0.0,
                        )
                    )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "reference.json"
            method_path = root / "method.json"
            fov_path = root / "fov.json"
            output = root / "result"
            reference_path.write_text(
                json.dumps(evaluation_payload(reference_rows)) + "\n"
            )
            method_path.write_text(json.dumps(evaluation_payload(method_rows)) + "\n")
            fov_path.write_text(
                json.dumps({"status": "complete", "rows": geometry_rows}) + "\n"
            )

            main(
                [
                    "--evaluation",
                    f"reference={reference_path}",
                    "--evaluation",
                    f"method={method_path}",
                    "--fov-json",
                    str(fov_path),
                    "--output-dir",
                    str(output),
                    "--reference",
                    "reference",
                    "--bootstrap-resamples",
                    "100",
                ]
            )

            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["reference_method"], "reference")
            self.assertEqual(summary["episode_counts"]["observed"]["method"], 6)
            self.assertEqual(summary["episode_counts"]["withheld"]["reference"], 6)
            self.assertEqual(summary["bootstrap_resamples"], 100)
            self.assertTrue(
                any(
                    row["method"] == "method"
                    and row["success_delta"] == 1.0
                    and row["camera_pose"] == "az_p70"
                    for row in summary["aggregates"]
                )
            )
            with (output / "episode_rows.csv").open(newline="") as handle:
                episode_csv = list(csv.DictReader(handle))
            self.assertEqual(len(episode_csv), 24)
            self.assertIn("visibility_stratum", episode_csv[0])
            with (output / "aggregate.csv").open(newline="") as handle:
                aggregate_csv = list(csv.DictReader(handle))
            self.assertIn("success_ci95_low", aggregate_csv[0])
            with Image.open(output / "camera_success_curves.png") as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.width, 600)
                self.assertGreater(image.height, 300)


if __name__ == "__main__":
    unittest.main()
