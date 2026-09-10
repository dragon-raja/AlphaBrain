from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from libero_camera_pose import (
    camera_intrinsics,
    load_camera_sweep_config,
    look_at_rotation,
    opencv_pixels_to_policy,
    orbit_pose,
    plucker_raymap,
    project_world_points,
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

    def test_project_world_points_uses_opencv_camera_axes(self) -> None:
        intrinsics = camera_intrinsics(fovy_deg=90.0, height=100, width=100)
        camera_to_world = np.eye(4)
        pixels, depth = project_world_points(
            np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
        )
        np.testing.assert_allclose(depth, [1.0, 1.0])
        np.testing.assert_allclose(pixels, [[50.0, 50.0], [100.0, 50.0]])

    def test_opencv_to_policy_matches_mujoco_upright_image(self) -> None:
        pixels = opencv_pixels_to_policy(
            np.asarray([[8.7801413201, 153.0246387358]]),
            width=224,
        )
        np.testing.assert_allclose(
            pixels,
            [[214.2198586799, 153.0246387358]],
        )

    def test_plucker_raymap_satisfies_bilinear_constraint(self) -> None:
        intrinsics = camera_intrinsics(fovy_deg=60.0, height=8, width=10)
        camera_to_world = np.eye(4)
        camera_to_world[:3, 3] = [0.5, -0.25, 1.0]
        rays = plucker_raymap(
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            height=8,
            width=10,
            image_transform="none",
        )
        directions = rays[..., :3]
        moments = rays[..., 3:]
        np.testing.assert_allclose(
            np.linalg.norm(directions, axis=-1),
            1.0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.sum(directions * moments, axis=-1),
            0.0,
            atol=1e-6,
        )

    def test_numpy_mujoco_upright_raymap_flips_only_horizontally(self) -> None:
        intrinsics = camera_intrinsics(fovy_deg=60.0, height=2, width=3)
        camera_to_world = np.eye(4)
        untransformed = plucker_raymap(
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            height=2,
            width=3,
            image_transform="none",
        )
        transformed = plucker_raymap(
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
            height=2,
            width=3,
            image_transform="mujoco_upright",
        )
        np.testing.assert_allclose(
            transformed,
            np.flip(untransformed, axis=1),
        )

    def test_loader_expands_dense_one_factor_axes(self) -> None:
        payload = {
            "schema_version": 1,
            "camera_name": "agentview",
            "table_plane_z": 0.48,
            "one_factor_axes": [
                {
                    "parameter": "azimuth_deg",
                    "start": -5.0,
                    "stop": 5.0,
                    "step": 2.5,
                },
                {
                    "parameter": "radius_scale",
                    "values": [0.95, 1.0, 1.05],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.json"
            path.write_text(json.dumps(payload))
            config = load_camera_sweep_config(path)
        names = [pose["name"] for pose in config["poses"]]
        self.assertEqual(
            names,
            [
                "baseline",
                "az_m5",
                "az_m2p5",
                "az_p2p5",
                "az_p5",
                "rad_0950",
                "rad_1050",
            ],
        )


if __name__ == "__main__":
    unittest.main()
