from __future__ import annotations

import unittest

from render_constructed_m0_audit_montages import (
    select_balanced_audit_groups,
    selected_pose_ids,
)
from scan_libero_hdf5_views import (
    _filter_catalog_poses,
    _remove_materialized_canonical,
)


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
        chosen = select_balanced_audit_groups(rows, per_task=7, target_total=21)
        self.assertEqual(len(chosen), 21)
        for task in ("a", "b", "c"):
            task_rows = [row for row in chosen if row["task_id"] == task]
            self.assertEqual(len(task_rows), 7)
            self.assertEqual({row["source_episode_id"] for row in task_rows}, {"e1", "e2"})

    def test_fills_target_when_one_task_has_few_candidates(self) -> None:
        rows = [_row("a", "e1", frame) for frame in range(2)]
        for task in ("b", "c"):
            for episode in ("e1", "e2"):
                rows.extend(_row(task, episode, frame) for frame in range(10))
        chosen = select_balanced_audit_groups(
            rows, per_task=7, target_total=21
        )
        self.assertEqual(len(chosen), 21)
        counts = {
            task: sum(row["task_id"] == task for row in chosen)
            for task in ("a", "b", "c")
        }
        self.assertEqual(counts["a"], 2)
        self.assertLessEqual(abs(counts["b"] - counts["c"]), 1)

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

    def test_materialized_canonical_is_not_rendered_twice(self) -> None:
        poses = [
            ("canonical", {"pose_id": "canonical"}),
            ("broad", {"pose_id": "broad-a"}),
        ]
        self.assertEqual(_remove_materialized_canonical(poses), [poses[1]])


if __name__ == "__main__":
    unittest.main()
