from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def quaternion_xyzw_to_rotation(quaternion: Sequence[float]) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("quaternion must be a finite xyzw vector")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be positive")
    x, y, z, w = value / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def axis_angle_to_rotation(axis_angle: Sequence[float]) -> np.ndarray:
    value = np.asarray(axis_angle, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("axis angle must be a finite three-vector")
    angle = float(np.linalg.norm(value))
    if angle <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = value / angle
    skew = np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (
        skew @ skew
    )


def eef_transform_from_pose(eef_pose_xyzw: Sequence[float]) -> np.ndarray:
    pose = np.asarray(eef_pose_xyzw, dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError("EEF pose must be a finite xyz+xyzw vector")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_xyzw_to_rotation(pose[3:])
    transform[:3, 3] = pose[:3]
    return transform


def eef_transform_from_robot_state(robot_state: Sequence[float]) -> np.ndarray:
    state = np.asarray(robot_state, dtype=np.float64)
    if state.shape[0] < 6 or not np.all(np.isfinite(state[:6])):
        raise ValueError("robot state must contain finite EEF xyz and axis angle")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = axis_angle_to_rotation(state[3:6])
    transform[:3, 3] = state[:3]
    return transform


def wrist_camera_from_eef(
    eef_to_world: np.ndarray,
    eef_to_wrist_opencv: np.ndarray,
) -> np.ndarray:
    eef = np.asarray(eef_to_world, dtype=np.float64)
    hand_eye = np.asarray(eef_to_wrist_opencv, dtype=np.float64)
    if eef.shape != (4, 4) or hand_eye.shape != (4, 4):
        raise ValueError("EEF and hand-eye transforms must be 4x4")
    return eef @ hand_eye


def rotation_angle_degrees(rotation: np.ndarray) -> float:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def average_rigid_transforms(transforms: Sequence[np.ndarray]) -> np.ndarray:
    values = np.asarray(transforms, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4) or len(values) < 1:
        raise ValueError("transforms must have shape [N, 4, 4]")
    mean_rotation = values[:, :3, :3].sum(axis=0)
    left, _, right = np.linalg.svd(mean_rotation)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = values[:, :3, 3].mean(axis=0)
    return result
