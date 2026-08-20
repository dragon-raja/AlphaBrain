from __future__ import annotations

import unittest

from run_constructed_accel_bank import build_candidate_bank, summarize


def _pose(pose_id: str) -> dict:
    return {
        "pose_id": pose_id,
        "azimuth_deg": 0.0,
        "elevation_deg": 0.0,
        "radius_scale": 1.0,
    }


def _catalog() -> dict:
    return {
        "canonical": [_pose("canonical")],
        "broad_training_64": [_pose(f"train_{index:02d}") for index in range(64)],
        "broad_heldout_32": [_pose(f"heldout_{index:02d}") for index in range(32)],
        "diagnostic_crossed_orbit": [_pose("blind")],
        "diagnostic_look_away": [_pose("look_away")],
    }


def _selection() -> dict:
    return {
        "canonical": {"pose_id": "canonical"},
        "strong_info": {"pose_id": "heldout_01"},
        "matched_control": {"pose_id": "heldout_02"},
        "blind": {"pose_id": "blind"},
        "look_away": {"pose_id": "look_away"},
    }


class ConstructedAccelBankTest(unittest.TestCase):
    def test_bank_separates_operational_physical_and_sensor_candidates(self) -> None:
        bank = build_candidate_bank(_catalog(), _selection())
        self.assertEqual(len(bank["operational_ids"]), 97)
        self.assertEqual(len(bank["all_physical_ids"]), 99)
        self.assertEqual(len(bank["all_candidate_ids"]), 102)
        self.assertEqual(len(bank["diagnostic_ids"]), 8)
        self.assertIn("heldout_01", bank["operational_ids"])
        self.assertNotIn("all_camera_blackout", bank["all_physical_ids"])

    def test_rejects_missing_frozen_diagnostic_pose(self) -> None:
        selection = _selection()
        selection["blind"]["pose_id"] = "absent"
        with self.assertRaisesRegex(ValueError, "absent from catalog"):
            build_candidate_bank(_catalog(), selection)

    def test_summary_is_explicitly_descriptive_until_m1_join(self) -> None:
        role_metrics = {
            role: {
                "complete_rank": index + 1,
                "diagnostic_rank": index + 1,
                "delta_visibility": float(index),
            }
            for index, role in enumerate(
                (
                    "canonical",
                    "strong_info",
                    "matched_control",
                    "blind",
                    "look_away",
                    "external_blackout",
                    "wrist_blackout",
                    "all_camera_blackout",
                )
            )
        }
        row = {
            "task_id": "task",
            "selected_candidates": {
                "complete": "canonical",
                "all_physical": "canonical",
                "operational_97": "canonical",
                "diagnostic_shortlist": "canonical",
            },
            "selected_candidate_categories": {
                "complete": "canonical",
                "all_physical": "canonical",
                "operational_97": "canonical",
                "diagnostic_shortlist": "canonical",
            },
            "fixed_state_audit": {
                "physics_state_preserved_across_all_candidates": True
            },
            "role_metrics": role_metrics,
            "relation_diagnostics": {
                "operational_accel_visibility_spearman": 0.5,
                "strong_info_lower_accel_than_matched_control": True,
                "strong_info_lower_accel_than_canonical": False,
            },
        }
        result = summarize([row])
        self.assertEqual(result["status"], "PASS_DESCRIPTIVE_RELATION_ANALYSIS")
        self.assertIn("DEFERRED", result["closed_loop_oracle_relation"])


if __name__ == "__main__":
    unittest.main()
