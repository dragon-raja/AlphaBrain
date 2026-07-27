from __future__ import annotations

import unittest

from summarize_kyc_camera_seeds import (
    is_training_support,
    paired_group_bootstrap,
    paired_seed_state_bootstrap,
    summarize_context_pair,
    summarize_seed_pairs,
)


def row(
    method: str,
    seed: int,
    state: int,
    pose: str,
    success: float,
    *,
    axis: str = "azimuth_deg",
    value: float = 0.0,
    stratum: str = "fully_supported",
) -> dict:
    return {
        "method": method,
        "seed": seed,
        "edge_id": "red-left",
        "canonical_state_index": state,
        "execution_horizon": 3,
        "camera_pose": pose,
        "sweep_axis": "baseline" if pose == "baseline" else axis,
        "sweep_value": value,
        "visibility_stratum": stratum,
        "success": success,
        "progress": success,
        "source_selection_success": success,
        "lift_success": success,
        "transport_success": success,
        "target_placement_success": success,
    }


class SummarizeKycCameraSeedsTest(unittest.TestCase):
    def test_training_support_includes_edges_and_baseline(self) -> None:
        self.assertTrue(is_training_support(row("kyc", 41, 0, "baseline", 1.0)))
        self.assertTrue(
            is_training_support(
                row("kyc", 41, 0, "az_p60", 1.0, value=60.0)
            )
        )
        self.assertFalse(
            is_training_support(
                row("kyc", 41, 0, "az_p90", 1.0, value=90.0)
            )
        )

    def test_group_bootstrap_averages_seed_repeats_within_state(self) -> None:
        result = paired_group_bootstrap(
            {0: [1.0, 1.0], 1: [0.0, 0.0]},
            resamples=200,
            seed=7,
        )
        self.assertEqual(result["delta"], 0.5)
        self.assertEqual(result["state_count"], 2)
        self.assertEqual(result["ci95_low"], 0.0)
        self.assertEqual(result["ci95_high"], 1.0)

    def test_hierarchical_bootstrap_includes_seed_and_state_units(self) -> None:
        result = paired_seed_state_bootstrap(
            {
                41: {0: [1.0], 1: [0.0]},
                42: {0: [0.0], 1: [-1.0]},
            },
            resamples=500,
            seed=7,
        )
        self.assertEqual(result["delta"], 0.0)
        self.assertEqual(result["seed_count"], 2)
        self.assertEqual(result["state_count"], 2)
        self.assertLess(result["ci95_low"], 0.0)
        self.assertGreater(result["ci95_high"], 0.0)

    def test_poseaug_rgb_context_is_state_paired(self) -> None:
        rows = []
        for state in (0, 1):
            rows.extend(
                [
                    row("poseaug_rgb", 41, state, "az_p60", 0.0, value=60.0),
                    row("kyc", 41, state, "az_p60", 1.0, value=60.0),
                ]
            )
        result = summarize_context_pair(
            rows,
            reference="poseaug_rgb",
            method="kyc",
            scope="fully_supported",
            metric="success",
            bootstrap_resamples=100,
        )
        self.assertEqual(result["delta"], 1.0)
        self.assertEqual(result["paired_state_count"], 2)

    def test_seed_summary_reports_per_seed_and_group_delta(self) -> None:
        seed_rows = {}
        for seed in (41, 42):
            values = []
            for state in (0, 1):
                values.extend(
                    [
                        row(
                            "poseaug_control",
                            seed,
                            state,
                            "baseline",
                            float(state),
                        ),
                        row("kyc", seed, state, "baseline", 1.0),
                        row(
                            "poseaug_control",
                            seed,
                            state,
                            "az_p60",
                            0.0,
                            value=60.0,
                        ),
                        row(
                            "kyc",
                            seed,
                            state,
                            "az_p60",
                            1.0,
                            value=60.0,
                        ),
                    ]
                )
            seed_rows[seed] = values

        summaries = summarize_seed_pairs(seed_rows, bootstrap_resamples=200)
        by_scope = {summary["scope"]: summary for summary in summaries}
        self.assertEqual(
            by_scope["canonical"]["success"]["delta"],
            0.5,
        )
        self.assertEqual(
            by_scope["training_support"]["success"]["delta"],
            0.75,
        )
        self.assertEqual(
            len(by_scope["training_support"]["success"]["per_seed"]),
            2,
        )


if __name__ == "__main__":
    unittest.main()
