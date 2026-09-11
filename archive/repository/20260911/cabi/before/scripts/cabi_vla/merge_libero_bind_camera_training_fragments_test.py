from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from merge_libero_bind_camera_training_fragments import (
    _link_or_copy,
    _safe_relative_path,
    _validate_camera_matrices,
    _validate_camera_shard,
)


def camera_row(index: int = 0) -> dict:
    return {
        "camera_view_index": index,
        "camera_intrinsics": np.eye(3).tolist(),
        "camera_to_world_opencv": np.eye(4).tolist(),
    }


class MergeCameraFragmentsTest(unittest.TestCase):
    def test_rejects_unsafe_paths_and_existing_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsafe"):
            _safe_relative_path("../other.npz")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            target = root / "target"
            source.write_bytes(b"source")
            target.write_bytes(b"existing")
            with self.assertRaisesRegex(FileExistsError, "replace"):
                _link_or_copy(source, target)
            self.assertEqual(target.read_bytes(), b"existing")

    def test_validates_rigid_camera_matrices(self) -> None:
        _validate_camera_matrices(camera_row(), index=0)
        invalid = camera_row()
        invalid["camera_to_world_opencv"][0][0] = 2.0
        with self.assertRaisesRegex(ValueError, "invalid camera-to-world"):
            _validate_camera_matrices(invalid, index=0)

    def test_validates_camera_shard_schema_and_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "camera.npz"
            np.savez_compressed(
                shard,
                agentview=np.zeros((2, 224, 224, 3), dtype=np.uint8),
                wrist=np.zeros((2, 224, 224, 3), dtype=np.uint8),
                robot_state=np.zeros((2, 8), dtype=np.float32),
            )
            _validate_camera_shard(shard, [camera_row(0), camera_row(1)])
            with self.assertRaisesRegex(IndexError, "outside"):
                _validate_camera_shard(shard, [camera_row(2)])


if __name__ == "__main__":
    unittest.main()
