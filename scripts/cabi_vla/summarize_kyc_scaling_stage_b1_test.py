from __future__ import annotations

import unittest

from summarize_kyc_scaling_stage_b1 import (
    budget_to_eighty_percent,
    in_training_support,
    normalized_log_auc,
    paired_group_bootstrap,
    select_factorial_budget,
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


if __name__ == "__main__":
    unittest.main()
