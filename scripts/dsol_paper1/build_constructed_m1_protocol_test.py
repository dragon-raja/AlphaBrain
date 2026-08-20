from __future__ import annotations

import unittest
from pathlib import Path

from build_constructed_m1_protocol import build


def _condition(role: str, pose: str, group: str, delta: float) -> dict:
    return {
        "condition_role": role,
        "source_pose_id": pose,
        "source_group": group,
        "visibility_score": 0.1 + delta,
        "delta_visibility": delta,
    }


def _selection() -> dict:
    rows = []
    for index in range(2):
        scan_id = f"scan-{index}"
        rows.append(
            {
                "snapshot_group_id": scan_id,
                "scan_id": scan_id,
                "task_id": "task-a",
                "split": "test",
                "source_episode_id": f"demo-{index}",
                "source_frame": index,
                "conditions": {
                    "canonical": _condition("canonical", "canonical", "canonical", 0.0),
                    "strong_info": _condition("strong_info", "strong", "broad_heldout_32", 0.05),
                    "matched_control": _condition("matched_control", "control", "wide_extrapolation_24", 0.001),
                    "blind": _condition("blind", "blind", "diagnostic_extreme_orbit", -0.08),
                    "look_away": _condition("look_away", "look", "diagnostic_look_away", -0.09),
                    "all_camera_blackout": _condition("all_camera_blackout", "all_camera_blackout", "sensor_controls", -0.1),
                },
            }
        )
    return {
        "schema": "dsol_constructed_m0_candidate_selection_v1",
        "status": "PASS",
        "m1_admission_status": "HOLD_MANUAL_AUDIT",
        "selected_snapshot_groups": rows,
    }


class BuildConstructedM1ProtocolTest(unittest.TestCase):
    def test_builds_ten_exact_state_conditions_per_audited_group(self) -> None:
        selection = _selection()
        audit = {
            "schema": "dsol_constructed_m0_manual_visual_audit_v1",
            "status": "PASS",
            "selection_sha256": "selection-hash",
            "records": [
                {"snapshot_group_id": f"scan-{index}", "status": "PASS"}
                for index in range(2)
            ],
        }
        plan = {
            "records": [
                {
                    "scan_id": f"scan-{index}",
                    "diagnostic_role": "state",
                    "suite": "libero_goal",
                    "hdf5": "/data/task.hdf5",
                    "episode_id": f"demo-{index}",
                    "demo_name": f"demo_{index}",
                    "demo_index": index,
                    "frame": index,
                    "stage_fraction": 0.5,
                }
                for index in range(2)
            ]
        }
        catalog = {
            "canonical": [{"pose_id": "canonical"}],
            "poses": [
                {"pose_id": "strong"},
                {"pose_id": "control"},
                {"pose_id": "blind"},
                {"pose_id": "look"},
            ],
        }
        result = build(
            selection=selection,
            selection_sha256="selection-hash",
            audit=audit,
            scan_plan=plan,
            catalog=catalog,
            catalog_path=Path("/tmp/catalog.json"),
            minimum_audited_groups=2,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["selected_state_count"], 2)
        self.assertEqual(result["episode_count"], 20)
        self.assertEqual(
            {row["condition"] for row in result["specs"]},
            {
                "canonical_both",
                "strong_info_both",
                "matched_control_both",
                "blind_both",
                "canonical_external_only",
                "strong_info_external_only",
                "matched_control_external_only",
                "blind_external_only",
                "canonical_wrist_only",
                "all_camera_blackout",
            },
        )
        for scan_id in ("scan-0", "scan-1"):
            rows = [row for row in result["specs"] if row["scan_id"] == scan_id]
            self.assertEqual(len({row["source_state_index"] for row in rows}), 1)
            self.assertTrue(all(row["manual_audit_verified"] for row in rows))

    def test_rejects_audit_for_other_selection(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not identify"):
            build(
                selection=_selection(),
                selection_sha256="selection-hash",
                audit={
                    "schema": "dsol_constructed_m0_manual_visual_audit_v1",
                    "status": "PASS",
                    "selection_sha256": "other",
                    "records": [],
                },
                scan_plan={"records": []},
                catalog={},
                catalog_path=Path("/tmp/catalog.json"),
                minimum_audited_groups=1,
            )


if __name__ == "__main__":
    unittest.main()
