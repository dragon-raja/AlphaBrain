from __future__ import annotations

import numpy as np
import pytest

from inspect_libero_plus_camparam_rlds import (
    EXPECTED_FEATURES,
    inspect_record,
    mujoco_camera_to_opencv,
    summarize_records,
)


def _record(length: int = 2) -> dict:
    jpeg = b"\xff\xd8synthetic-jpeg"
    record = {
        "episode_metadata/camera_calibration/primary_cam_extrinsics": np.eye(4).reshape(-1),
        "episode_metadata/file_path": np.asarray(b"/source/demo.hdf5"),
        "steps/action": np.arange(length * 7, dtype=np.float32),
        "steps/discount": np.ones(length, dtype=np.float32),
        "steps/is_first": np.asarray([1] + [0] * (length - 1)),
        "steps/is_last": np.asarray([0] * (length - 1) + [1]),
        "steps/is_terminal": np.asarray([0] * (length - 1) + [1]),
        "steps/language_instruction": np.asarray([b"test task"] * length),
        "steps/observation/image": np.asarray([jpeg] * length),
        "steps/observation/joint_state": np.arange(length * 7, dtype=np.float32),
        "steps/observation/state": np.asarray(
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.04, -0.04],
             [0.1, 0.0, 1.0, 0.0, 0.0, 0.2, 0.04, -0.04]][:length],
            dtype=np.float32,
        ).reshape(-1),
        "steps/observation/wrist_image": np.asarray([jpeg] * length),
        "steps/reward": np.asarray([0.0] * (length - 1) + [1.0], dtype=np.float32),
    }
    assert set(record) == EXPECTED_FEATURES
    return record


def test_mujoco_camera_to_opencv_flips_y_and_z_axes() -> None:
    converted = mujoco_camera_to_opencv(np.eye(4))
    np.testing.assert_allclose(converted, np.diag([1.0, -1.0, -1.0, 1.0]))
    assert np.linalg.det(converted[:3, :3]) == pytest.approx(1.0)


def test_inspect_record_validates_and_reconstructs_wrist_motion() -> None:
    row, action = inspect_record(
        _record(),
        hand_eye=np.eye(4),
        shard_name="shard-0",
        record_index=3,
    )
    assert action.shape == (2, 7)
    assert row["source_basename"] == "demo.hdf5"
    assert row["step_count"] == 2
    assert row["terminal_reward"] == 1.0
    assert row["external_optical_forward_z"] == -1.0
    assert row["wrist_translation_span_m"] == pytest.approx(0.1)
    assert row["wrist_rotation_span_deg"] == pytest.approx(np.degrees(0.2))

    summary = summarize_records([row], [action])
    assert summary["episode_count"] == 1
    assert summary["step_count"] == 2
    assert summary["terminal_success_count"] == 1
    assert summary["external_camera_downward_fraction"] == 1.0


def test_inspect_record_rejects_misaligned_step_arrays() -> None:
    record = _record()
    record["steps/action"] = np.zeros(13, dtype=np.float32)
    with pytest.raises(ValueError, match="steps/action"):
        inspect_record(
            record,
            hand_eye=np.eye(4),
            shard_name="shard-0",
            record_index=0,
        )
