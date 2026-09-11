import tempfile
import unittest
from pathlib import Path

import numpy as np
from tfrecord.writer import TFRecordWriter

from build_libero_plus_training_view import (
    assign_pose_group_splits,
    camera_pose_group_id,
    scan_tfrecord_offsets,
)


class LiberoPlusTrainingViewTest(unittest.TestCase):
    def test_scan_tfrecord_offsets(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny.tfrecord"
            writer = TFRecordWriter(str(path))
            writer.write({"value": ([1, 2], "int")})
            writer.write({"value": ([3], "int")})
            writer.close()
            offsets = scan_tfrecord_offsets(path)
            self.assertEqual(len(offsets), 2)
            self.assertEqual(offsets[0][0], 0)
            self.assertEqual(sum(length for _, length in offsets), path.stat().st_size)

    def test_pose_groups_are_atomic_and_all_tasks_are_covered(self):
        rows = []
        for task_index in range(3):
            for pose_index in range(30):
                matrix = np.eye(4)
                matrix[0, 3] = pose_index * 0.01
                matrix[1, 3] = task_index * 0.001
                rows.append(
                    {
                        "language_instruction": f"task-{task_index}",
                        "camera_pose_group_id": camera_pose_group_id(matrix),
                    }
                )
        assignments = assign_pose_group_splits(rows, seed=41)
        self.assertEqual(set(assignments.values()), {"train", "val", "test"})
        for split in ("train", "val", "test"):
            tasks = {
                row["language_instruction"]
                for row in rows
                if assignments[row["camera_pose_group_id"]] == split
            }
            self.assertEqual(tasks, {"task-0", "task-1", "task-2"})


if __name__ == "__main__":
    unittest.main()
