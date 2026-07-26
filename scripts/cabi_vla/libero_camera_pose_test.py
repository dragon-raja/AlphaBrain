from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from libero_camera_pose import (
    load_camera_sweep_config,
    look_at_rotation,
    orbit_pose,
    rotation_matrix_to_wxyz,
)


class LiberoCameraPoseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.position = np.asarray([0.606577, 0.0, 0.96])
        self.pivot = np.asarray([-0.164, 0.0, 0.48])
        self.rotation = look_at_rotation(self.position, self.pivot)
        self.quaternion = rotation_matrix_to_wxyz(self.rotation)
        self.reference = {
            "position": self.position,
            "pivot": self.pivot,
            "quaternion": self.quaternion,
        }

    def test_baseline_preserves_exact_reference_pose(self) -> None:
        result = orbit_pose(self.reference, {"name": "baseline"})
        np.testing.assert_array_equal(result["position"], self.position)
        np.testing.assert_array_equal(result["quaternion"], self.quaternion)

    def test_azimuth_orbit_preserves_radius_and_looks_at_pivot(self) -> None:
        result = orbit_pose(
            self.reference,
            {"name": "az_p30", "azimuth_deg": 30.0},
        )
        self.assertAlmostEqual(
            np.linalg.norm(result["position"] - self.pivot),
            np.linalg.norm(self.position - self.pivot),
        )
        rotation = look_at_rotation(result["position"], self.pivot)
        expected = rotation_matrix_to_wxyz(rotation)
        np.testing.assert_allclose(result["quaternion"], expected, atol=1e-8)

    def test_loader_rejects_duplicate_names(self) -> None:
        payload = {
            "schema_version": 1,
            "camera_name": "agentview",
            "table_plane_z": 0.48,
            "poses": [{"name": "baseline"}, {"name": "baseline"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "unique"):
                load_camera_sweep_config(path)


if __name__ == "__main__":
    unittest.main()
