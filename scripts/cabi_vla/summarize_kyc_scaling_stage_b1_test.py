from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from summarize_kyc_scaling_stage_b1 import (
    budget_to_eighty_percent,
    in_training_support,
    normalized_log_auc,
    paired_group_bootstrap,
    select_factorial_budget,
    summarize,
)


class SummarizeKycScalingStageB1Test(unittest.TestCase):
    def test_training_support_is_inclusive(self) -> None:
        self.assertTrue(
            in_training_support(
                {
                    "camera_azimuth_deg": "-60",
                    "camera_elevation_deg": "25",
                    "camera_radius_scale": "1.25",
                }
            )
        )
        self.assertFalse(
            in_training_support(
                {
                    "camera_azimuth_deg": "-60.1",
                    "camera_elevation_deg": "0",
                    "camera_radius_scale": "1",
                }
            )
        )

    def test_budget_selection_rules(self) -> None:
        self.assertEqual(
            select_factorial_budget({10: 0.1, 45: 0.25, 215: 0.8})[
                "selected_budget"
            ],
            45,
        )
        self.assertEqual(
            select_factorial_budget({10: 0.75, 45: 0.8})["selected_budget"],
            10,
        )
        self.assertEqual(
            select_factorial_budget({10: 0.1, 45: 0.15})["status"],
            "BASELINE_INVALID",
        )

    def test_scaling_helpers(self) -> None:
        values = {10: 0.2, 45: 0.5, 215: 0.75, 1000: 0.8}
        self.assertEqual(budget_to_eighty_percent(values), 215)
        self.assertGreater(normalized_log_auc(values), 0.2)
        self.assertLess(normalized_log_auc(values), 0.8)

    def test_bootstrap_pairs_by_snapshot_group(self) -> None:
        rows = []
        for state, control, kyc in ((40, 0.0, 1.0), (41, 1.0, 1.0)):
            for method, value in (("poseaug_control", control), ("kyc", kyc)):
                rows.append(
                    {
                        "method": method,
                        "edge_id": "red-left",
                        "canonical_state_index": state,
                        "execution_horizon": 3,
                        "camera_pose": "baseline",
                        "success": value,
                    }
                )
        result = paired_group_bootstrap(
            rows,
            method="kyc",
            reference="poseaug_control",
            metric="success",
            bootstrap_resamples=1_000,
        )
        self.assertEqual(result["snapshot_group_count"], 2)
        self.assertEqual(result["delta"], 0.5)

    def test_summary_marks_absent_withheld_split_unavailable(self) -> None:
        fields = (
            "method",
            "data_split",
            "visibility_stratum",
            "camera_azimuth_deg",
            "camera_elevation_deg",
            "camera_radius_scale",
            "edge_id",
            "canonical_state_index",
            "execution_horizon",
            "camera_pose",
            "success",
            "transport_success",
            "progress",
            "completion_steps",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for budget in (10, 45):
                output = root / f"n{budget}" / "episode_rows.csv"
                output.parent.mkdir(parents=True)
                with output.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    for method, success in (
                        ("poseaug_control", 0.0),
                        ("kyc", 1.0),
                    ):
                        writer.writerow(
                            {
                                "method": method,
                                "data_split": "observed",
                                "visibility_stratum": "fully_supported",
                                "camera_azimuth_deg": 0.0,
                                "camera_elevation_deg": 0.0,
                                "camera_radius_scale": 1.0,
                                "edge_id": "red-left",
                                "canonical_state_index": 40,
                                "execution_horizon": 3,
                                "camera_pose": "baseline",
                                "success": success,
                                "transport_success": success,
                                "progress": success,
                                "completion_steps": 100,
                            }
                        )
            result = summarize(
                root,
                budgets=(10, 45),
                bootstrap_resamples=100,
            )
        for budget in ("10", "45"):
            primary = result["budget_results"][budget]["primary"]
            self.assertTrue(primary["all"]["available"])
            self.assertTrue(primary["observed"]["available"])
            self.assertFalse(primary["withheld"]["available"])
            self.assertEqual(primary["withheld"]["methods"], {})


if __name__ == "__main__":
    unittest.main()
