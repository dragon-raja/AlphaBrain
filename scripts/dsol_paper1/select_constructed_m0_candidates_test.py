from __future__ import annotations

import copy
import unittest

from scripts.dsol_paper1.select_constructed_m0_candidates import build_selection


TEST_PROTOCOL = {
    "strong_info": {
        "episode_quantile": 0.25,
        "minimum_delta": 0.005,
        "maximum_frozen_delta": 0.2,
    },
    "matched_control": {
        "episode_quantile": 0.75,
        "minimum_abs_delta": 0.001,
        "maximum_abs_delta": 0.02,
        "minimum_translation_tolerance_m": 0.03,
        "maximum_translation_tolerance_m": 0.2,
        "minimum_rotation_tolerance_deg": 5.0,
        "maximum_rotation_tolerance_deg": 40.0,
    },
    "population": {
        "minimum_task_count": 1,
        "minimum_val_episodes_per_task": 2,
        "minimum_test_episodes_per_task": 2,
        "minimum_selected_test_snapshots_per_episode": 1,
    },
}


def record(
    pose_id: str,
    group: str,
    delta: float,
    translation: float,
    rotation: float,
    *,
    canonical_score: float = 0.1,
) -> dict:
    return {
        "pose_id": pose_id,
        "group": group,
        "visibility_score": canonical_score + delta,
        "delta_visibility": delta,
        "camera_displacement_from_canonical": {
            "translation_m": translation,
            "rotation_geodesic_deg": rotation,
        },
    }


def snapshot(
    scan_id: str,
    split: str,
    episode_id: str,
    *,
    strong_delta: float = 0.04,
    include_control: bool = True,
    include_extreme: bool = True,
    include_look_away: bool = True,
) -> dict:
    records = [
        record("canonical", "canonical", 0.0, 0.0, 0.0),
        record("broad-strong", "broad_heldout_32", strong_delta, 0.10, 15.0),
    ]
    if include_control:
        records.append(
            record("broad-control", "wide_extrapolation_24", 0.001, 0.11, 16.0)
        )
    if include_extreme:
        records.append(
            record("extreme", "diagnostic_extreme_orbit", -0.09, 0.5, 90.0)
        )
    if include_look_away:
        records.append(
            record("look-away", "diagnostic_look_away", -0.1, 0.6, 120.0)
        )
    return {
        "scan_id": scan_id,
        "split": split,
        "task_id": "task-a",
        "episode_id": episode_id,
        "frame": 0,
        "scan_path": f"/{scan_id}/scan.json",
        "montage_path": f"/{scan_id}/visibility_extremes.png",
        "records": records,
    }


def complete_population() -> list[dict]:
    return [
        snapshot("val-a", "val", "val-episode-a", strong_delta=0.03),
        snapshot("val-b", "val", "val-episode-b", strong_delta=0.05),
        snapshot("test-a", "test", "test-episode-a", strong_delta=0.04),
        snapshot("test-b", "test", "test-episode-b", strong_delta=0.04),
    ]


class SelectConstructedM0CandidatesTest(unittest.TestCase):
    def test_test_measurements_cannot_change_frozen_thresholds(self) -> None:
        first = complete_population()
        second = copy.deepcopy(first)
        for row in second:
            if row["split"] == "test":
                for candidate in row["records"]:
                    if candidate["group"] == "broad_heldout_32":
                        candidate["delta_visibility"] = 0.19
                        candidate["visibility_score"] = 0.29
        output_a = build_selection(first, protocol=TEST_PROTOCOL)
        output_b = build_selection(second, protocol=TEST_PROTOCOL)
        self.assertEqual(output_a["frozen_rules"], output_b["frozen_rules"])
        self.assertFalse(
            output_a["frozen_rules"]["test_values_used_for_freeze"]
        )
        self.assertEqual(output_a["test_application"]["application_count"], 1)

    def test_extreme_candidate_is_never_selected_as_strong_info(self) -> None:
        rows = complete_population()
        for row in rows:
            row["records"].append(
                record("extreme-high", "diagnostic_extreme_orbit", 0.8, 0.2, 60.0)
            )
        output = build_selection(rows, protocol=TEST_PROTOCOL)
        self.assertEqual(output["status"], "PASS")
        for selected in output["selected_snapshot_groups"]:
            strong = selected["conditions"]["strong_info"]
            self.assertIn(
                strong["source_group"],
                {"broad_heldout_32", "wide_extrapolation_24"},
            )
            self.assertNotEqual(strong["source_pose_id"], "extreme-high")

    def test_high_information_near_pose_is_not_a_matched_control(self) -> None:
        rows = complete_population()
        for row in rows:
            row["records"].append(
                record(
                    "near-but-informative",
                    "broad_heldout_32",
                    0.03,
                    0.1001,
                    15.01,
                )
            )
        output = build_selection(rows, protocol=TEST_PROTOCOL)
        self.assertEqual(output["status"], "PASS")
        for selected in output["selected_snapshot_groups"]:
            control = selected["conditions"]["matched_control"]
            self.assertEqual(control["source_pose_id"], "broad-control")
            self.assertLessEqual(abs(control["delta_visibility"]), 0.02)

    def test_many_frames_from_one_episode_count_as_one_unit(self) -> None:
        rows = [
            snapshot(f"val-{index}", "val", "one-val-episode")
            for index in range(8)
        ]
        rows.extend(
            snapshot(f"test-{index}", "test", "one-test-episode")
            for index in range(8)
        )
        output = build_selection(rows, protocol=TEST_PROTOCOL)
        self.assertEqual(output["status"], "HOLD")
        self.assertEqual(
            output["frozen_rules"]["task_rules"]["task-a"][
                "validation_episode_count"
            ],
            1,
        )
        self.assertEqual(output["test_episode_coverage"]["episode_count"], 1)
        self.assertTrue(output["frames_are_not_independent_samples"])

    def test_missing_control_produces_hold_with_explicit_reason(self) -> None:
        rows = complete_population()
        for row in rows:
            if row["split"] == "test":
                row["records"] = [
                    candidate
                    for candidate in row["records"]
                    if candidate["pose_id"] != "broad-control"
                ]
        output = build_selection(rows, protocol=TEST_PROTOCOL)
        self.assertEqual(output["status"], "HOLD")
        self.assertEqual(output["selected_snapshot_group_count"], 0)
        reasons = {
            reason
            for row in output["insufficient_snapshot_groups"]
            for reason in row["reasons"]
        }
        self.assertIn(
            "no_matched_control_under_frozen_validation_rule", reasons
        )
        self.assertEqual(output["manual_visual_audit"]["status"], "PENDING")
        self.assertFalse(output["manual_visual_audit"]["automatically_promoted"])


if __name__ == "__main__":
    unittest.main()
