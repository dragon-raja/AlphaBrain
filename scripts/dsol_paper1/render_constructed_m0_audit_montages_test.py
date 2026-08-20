from __future__ import annotations

import unittest

from render_constructed_m0_audit_montages import (
    select_balanced_audit_groups,
    selected_pose_ids,
)
from scan_libero_hdf5_views import _filter_catalog_poses


def _row(task: str, episode: str, frame: int) -> dict:
    return {
        "snapshot_group_id": f"{task}-{episode}-{frame}",
        "scan_id": f"{task}-{episode}-{frame}",
        "task_id": task,
        "source_episode_id": episode,
        "source_frame": frame,
        "conditions": {
            "strong_info": {"source_pose_id": "strong"},
            "matched_control": {"source_pose_id": "control"},
            "blind": {"source_pose_id": "blind"},
            "look_away": {"source_pose_id": "look"},
            "all_camera_blackout": {"source_pose_id": "all_camera_blackout"},
        },
    }


class RenderConstructedM0AuditMontagesTest(unittest.TestCase):
    def test_balances_tasks_and_source_episodes(self) -> None:
        rows = []
        for task in ("a", "b", "c"):
            for episode in ("e1", "e2"):
                rows.extend(_row(task, episode, frame) for frame in range(5))
        chosen = select_balanced_audit_groups(rows, per_task=7)
        self.assertEqual(len(chosen), 21)
        for task in ("a", "b", "c"):
            task_rows = [row for row in chosen if row["task_id"] == task]
            self.assertEqual(len(task_rows), 7)
            self.assertEqual({row["source_episode_id"] for row in task_rows}, {"e1", "e2"})

    def test_selected_pose_ids_excludes_synthetic_blackout(self) -> None:
        self.assertEqual(
            selected_pose_ids(_row("a", "e1", 0)),
            ["strong", "control", "blind", "look"],
        )

    def test_catalog_filter_rejects_unknown_pose(self) -> None:
        poses = [("broad", {"pose_id": "a"}), ("wide", {"pose_id": "b"})]
        self.assertEqual(_filter_catalog_poses(poses, ["b"]), [poses[1]])
        with self.assertRaisesRegex(ValueError, "absent"):
            _filter_catalog_poses(poses, ["missing"])


if __name__ == "__main__":
    unittest.main()
