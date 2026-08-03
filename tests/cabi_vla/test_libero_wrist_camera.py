from __future__ import annotations

import numpy as np

from scripts.cabi_vla.libero_wrist_camera import (
    axis_angle_to_rotation,
    eef_transform_from_pose,
    eef_transform_from_robot_state,
    wrist_camera_from_eef,
)


def test_eef_pose_and_axis_angle_state_agree() -> None:
    angle = np.pi / 2.0
    pose = [1.0, 2.0, 3.0, 0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)]
    state = [1.0, 2.0, 3.0, 0.0, 0.0, angle, 0.0, 0.0]
    np.testing.assert_allclose(
        eef_transform_from_pose(pose),
        eef_transform_from_robot_state(state),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        axis_angle_to_rotation([0.0, 0.0, angle]),
        eef_transform_from_pose(pose)[:3, :3],
        atol=1e-12,
    )


def test_wrist_camera_composes_hand_eye_transform() -> None:
    eef = np.eye(4)
    eef[:3, 3] = [1.0, 2.0, 3.0]
    hand_eye = np.eye(4)
    hand_eye[:3, 3] = [0.1, 0.0, -0.2]
    expected = np.eye(4)
    expected[:3, 3] = [1.1, 2.0, 2.8]
    np.testing.assert_allclose(wrist_camera_from_eef(eef, hand_eye), expected)
