from __future__ import annotations

import unittest

from finalize_constructed_m0_manual_audit import CHECKS, finalize


def _manifest(count: int = 2) -> dict:
    return {
        "schema": "dsol_constructed_m0_manual_audit_render_v1",
        "status": "RENDERED_PENDING_MANUAL_AUDIT",
        "selection_sha256": "selection-hash",
        "records": [
            {
                "snapshot_group_id": f"group-{index}",
                "task_id": "task-a",
                "source_episode_id": f"demo-{index}",
                "source_frame": index,
                "montage_path": f"/{index}.png",
                "montage_sha256": f"hash-{index}",
                "condition_roles": {},
            }
            for index in range(count)
        ],
    }


def _decisions(count: int = 2) -> dict:
    return {
        "schema": "dsol_constructed_m0_manual_visual_decisions_v1",
        "review_mode": "visual",
        "reviewer": "test",
        "reviewed_at_utc": "2026-08-20T00:00:00Z",
        "records": [
            {
                "snapshot_group_id": f"group-{index}",
                "status": "PASS",
                "checks": {name: True for name in CHECKS},
            }
            for index in range(count)
        ],
    }


class FinalizeConstructedM0ManualAuditTest(unittest.TestCase):
    def test_all_checked_records_admit_m1(self) -> None:
        result = finalize(
            render_manifest=_manifest(),
            selection_sha256="selection-hash",
            decisions=_decisions(),
            minimum_groups=2,
        )
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["m1_admission"])
        self.assertFalse(result["automatically_promoted"])

    def test_failed_check_holds_m1(self) -> None:
        decisions = _decisions()
        decisions["records"][0]["checks"]["strong_info_gain_visible"] = False
        result = finalize(
            render_manifest=_manifest(),
            selection_sha256="selection-hash",
            decisions=decisions,
            minimum_groups=2,
        )
        self.assertEqual(result["status"], "HOLD")
        self.assertFalse(result["m1_admission"])


if __name__ == "__main__":
    unittest.main()
