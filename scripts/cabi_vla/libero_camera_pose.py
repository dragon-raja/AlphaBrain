from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


POSE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _as_vector(value: Any, *, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of length {length}")
    return array


def rotation_matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0:
        quaternion *= -1
    return quaternion


def look_at_rotation(
    position: Sequence[float],
    target: Sequence[float],
    *,
    world_up: Sequence[float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    position_vector = _as_vector(position, length=3, name="position")
    target_vector = _as_vector(target, length=3, name="target")
    up_vector = _as_vector(world_up, length=3, name="world_up")
    forward = target_vector - position_vector
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1e-9:
        raise ValueError("camera position and target must differ")
    forward /= forward_norm
    right = np.cross(forward, up_vector)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-9:
        raise ValueError("camera optical axis may not be parallel to world_up")
    right /= right_norm
    camera_up = np.cross(-forward, right)
    camera_up /= np.linalg.norm(camera_up)
    return np.column_stack((right, camera_up, -forward))


def capture_camera_reference(
    env: Any,
    *,
    camera_name: str,
    table_plane_z: float,
) -> dict[str, Any]:
    sim = env.env.sim
    sim.forward()
    camera_id = int(sim.model.camera_name2id(camera_name))
    position = np.asarray(sim.model.cam_pos[camera_id], dtype=np.float64).copy()
    quaternion = np.asarray(sim.model.cam_quat[camera_id], dtype=np.float64).copy()
    rotation = np.asarray(sim.data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3).copy()
    forward = -rotation[:, 2]
    if abs(float(forward[2])) <= 1e-9:
        raise ValueError("camera optical axis does not intersect the table plane")
    distance = (float(table_plane_z) - float(position[2])) / float(forward[2])
    if distance <= 0:
        raise ValueError("table plane lies behind the camera")
    pivot = position + distance * forward
    return {
        "camera_name": camera_name,
        "camera_id": camera_id,
        "position": position,
        "quaternion": quaternion,
        "rotation": rotation,
        "pivot": pivot,
        "table_plane_z": float(table_plane_z),
        "fovy": float(sim.model.cam_fovy[camera_id]),
    }


def orbit_pose(reference: Mapping[str, Any], pose: Mapping[str, Any]) -> dict[str, Any]:
    azimuth_delta = float(pose.get("azimuth_deg", 0.0))
    elevation_delta = float(pose.get("elevation_deg", 0.0))
    radius_scale = float(pose.get("radius_scale", 1.0))
    if not math.isfinite(azimuth_delta) or not math.isfinite(elevation_delta):
        raise ValueError("camera angle offsets must be finite")
    if not math.isfinite(radius_scale) or not 0.5 <= radius_scale <= 2.0:
        raise ValueError("camera radius_scale must be in [0.5, 2.0]")

    base_position = _as_vector(reference["position"], length=3, name="reference position")
    pivot = _as_vector(reference["pivot"], length=3, name="reference pivot")
    relative = base_position - pivot
    radius = float(np.linalg.norm(relative))
    azimuth = math.atan2(float(relative[1]), float(relative[0]))
    elevation = math.asin(float(relative[2]) / radius)
    new_elevation = elevation + math.radians(elevation_delta)
    if not math.radians(-85.0) < new_elevation < math.radians(85.0):
        raise ValueError("camera elevation leaves the supported range")
    new_azimuth = azimuth + math.radians(azimuth_delta)
    new_radius = radius * radius_scale
    position = pivot + new_radius * np.asarray(
        [
            math.cos(new_elevation) * math.cos(new_azimuth),
            math.cos(new_elevation) * math.sin(new_azimuth),
            math.sin(new_elevation),
        ],
        dtype=np.float64,
    )

    is_baseline = (
        abs(azimuth_delta) <= 1e-12
        and abs(elevation_delta) <= 1e-12
        and abs(radius_scale - 1.0) <= 1e-12
    )
    if is_baseline:
        quaternion = _as_vector(
            reference["quaternion"], length=4, name="reference quaternion"
        ).copy()
    else:
        quaternion = rotation_matrix_to_wxyz(look_at_rotation(position, pivot))
    return {
        "position": position,
        "quaternion": quaternion,
        "pivot": pivot.copy(),
        "azimuth_deg": azimuth_delta,
        "elevation_deg": elevation_delta,
        "radius_scale": radius_scale,
    }


def install_camera_pose(
    env: Any,
    reference: Mapping[str, Any],
    pose: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = orbit_pose(reference, pose)
    sim = env.env.sim
    camera_id = int(reference["camera_id"])
    is_baseline = (
        abs(float(resolved["azimuth_deg"])) <= 1e-12
        and abs(float(resolved["elevation_deg"])) <= 1e-12
        and abs(float(resolved["radius_scale"]) - 1.0) <= 1e-12
    )
    if not is_baseline:
        sim.model.cam_pos[camera_id] = resolved["position"]
        sim.model.cam_quat[camera_id] = resolved["quaternion"]
        sim.forward()
    metadata = {
        "camera_name": str(reference["camera_name"]),
        "camera_position": resolved["position"].tolist(),
        "camera_quaternion_wxyz": resolved["quaternion"].tolist(),
        "camera_pivot": resolved["pivot"].tolist(),
        "camera_fovy": float(reference["fovy"]),
        "camera_azimuth_deg": resolved["azimuth_deg"],
        "camera_elevation_deg": resolved["elevation_deg"],
        "camera_radius_scale": resolved["radius_scale"],
    }
    return metadata


def set_camera_pose(
    env: Any,
    reference: Mapping[str, Any],
    pose: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    metadata = install_camera_pose(env, reference, pose)
    env.env._update_observables(force=True)
    observation = env.env._get_observations()
    return observation, metadata


def load_camera_sweep_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("camera sweep config must use schema_version=1")
    camera_name = str(payload.get("camera_name", ""))
    if not camera_name:
        raise ValueError("camera_name is required")
    table_plane_z = float(payload.get("table_plane_z", float("nan")))
    if not math.isfinite(table_plane_z):
        raise ValueError("table_plane_z must be finite")
    poses = payload.get("poses")
    if not isinstance(poses, list) or not poses:
        raise ValueError("camera sweep config requires a non-empty poses list")
    names = []
    normalized = []
    for value in poses:
        if not isinstance(value, dict):
            raise ValueError("each camera pose must be an object")
        name = str(value.get("name", ""))
        if not POSE_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"invalid camera pose name: {name!r}")
        orbit_pose(
            {
                "position": np.asarray([1.0, 0.0, 1.0]),
                "pivot": np.asarray([0.0, 0.0, 0.0]),
                "quaternion": np.asarray([1.0, 0.0, 0.0, 0.0]),
            },
            value,
        )
        names.append(name)
        normalized.append(
            {
                "name": name,
                "azimuth_deg": float(value.get("azimuth_deg", 0.0)),
                "elevation_deg": float(value.get("elevation_deg", 0.0)),
                "radius_scale": float(value.get("radius_scale", 1.0)),
            }
        )
    if len(names) != len(set(names)):
        raise ValueError("camera pose names must be unique")
    if "baseline" not in names:
        raise ValueError("camera sweep config must include a baseline pose")
    return {
        **payload,
        "camera_name": camera_name,
        "table_plane_z": table_plane_z,
        "poses": normalized,
    }
