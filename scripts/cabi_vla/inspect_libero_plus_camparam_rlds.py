from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from libero_wrist_camera import (
    eef_transform_from_robot_state,
    rotation_angle_degrees,
    wrist_camera_from_eef,
)


EXPECTED_FEATURES = {
    "episode_metadata/camera_calibration/primary_cam_extrinsics",
    "episode_metadata/file_path",
    "steps/action",
    "steps/discount",
    "steps/is_first",
    "steps/is_last",
    "steps/is_terminal",
    "steps/language_instruction",
    "steps/observation/image",
    "steps/observation/joint_state",
    "steps/observation/state",
    "steps/observation/wrist_image",
    "steps/reward",
}
MUJOCO_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def _rigid_transform(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-5)
        or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
    ):
        raise ValueError(f"{name} is not a rigid transform")
    return matrix


def mujoco_camera_to_opencv(extrinsics: Any) -> np.ndarray:
    """Convert MuJoCo camera-to-world axes to OpenCV camera-to-world axes."""

    matrix = _rigid_transform(extrinsics, name="MuJoCo camera extrinsics")
    return _rigid_transform(
        matrix @ MUJOCO_TO_OPENCV,
        name="OpenCV camera extrinsics",
    )


def _bytes_array(value: Any, *, key: str, length: int) -> np.ndarray:
    array = np.asarray(value).reshape(-1)
    if len(array) != length or array.dtype.kind not in "SO":
        raise ValueError(f"{key} must contain {length} byte strings")
    return array


def _decode_constant_text(value: Any, *, key: str, length: int) -> str:
    array = _bytes_array(value, key=key, length=length)
    decoded = [bytes(item).decode("utf-8") for item in array]
    if len(set(decoded)) != 1:
        raise ValueError(f"{key} changes within one episode")
    return decoded[0]


def _reshape(value: Any, *, key: str, length: int, width: int) -> np.ndarray:
    array = np.asarray(value)
    if array.size != length * width:
        raise ValueError(
            f"{key} contains {array.size} values, expected {length * width}"
        )
    result = np.asarray(array, dtype=np.float32).reshape(length, width)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{key} contains non-finite values")
    return result


def inspect_record(
    record: Mapping[str, Any],
    *,
    hand_eye: np.ndarray,
    shard_name: str,
    record_index: int,
) -> tuple[dict[str, Any], np.ndarray]:
    keys = set(record)
    if keys != EXPECTED_FEATURES:
        raise ValueError(
            f"unexpected TFRecord fields: missing={sorted(EXPECTED_FEATURES - keys)}, "
            f"extra={sorted(keys - EXPECTED_FEATURES)}"
        )

    rewards = np.asarray(record["steps/reward"], dtype=np.float32).reshape(-1)
    length = len(rewards)
    if length <= 0:
        raise ValueError("episode has no steps")
    action = _reshape(record["steps/action"], key="steps/action", length=length, width=7)
    state = _reshape(
        record["steps/observation/state"],
        key="steps/observation/state",
        length=length,
        width=8,
    )
    _reshape(
        record["steps/observation/joint_state"],
        key="steps/observation/joint_state",
        length=length,
        width=7,
    )
    for key in ("steps/discount", "steps/is_first", "steps/is_last", "steps/is_terminal"):
        if np.asarray(record[key]).size != length:
            raise ValueError(f"{key} length does not match rewards")

    agent_images = _bytes_array(
        record["steps/observation/image"],
        key="steps/observation/image",
        length=length,
    )
    wrist_images = _bytes_array(
        record["steps/observation/wrist_image"],
        key="steps/observation/wrist_image",
        length=length,
    )
    if any(not bytes(value).startswith(b"\xff\xd8") for value in agent_images):
        raise ValueError("agent images are not JPEG encoded")
    if any(not bytes(value).startswith(b"\xff\xd8") for value in wrist_images):
        raise ValueError("wrist images are not JPEG encoded")

    language = _decode_constant_text(
        record["steps/language_instruction"],
        key="steps/language_instruction",
        length=length,
    )
    source_path = bytes(
        np.asarray(record["episode_metadata/file_path"]).reshape(()).item()
    ).decode("utf-8")
    raw_external = _rigid_transform(
        np.asarray(
            record["episode_metadata/camera_calibration/primary_cam_extrinsics"],
            dtype=np.float64,
        ).reshape(4, 4),
        name="primary camera extrinsics",
    )
    external = mujoco_camera_to_opencv(raw_external)
    wrist_first = wrist_camera_from_eef(
        eef_transform_from_robot_state(state[0]), hand_eye
    )
    wrist_last = wrist_camera_from_eef(
        eef_transform_from_robot_state(state[-1]), hand_eye
    )
    wrist_delta = np.linalg.inv(wrist_first) @ wrist_last
    is_first = np.asarray(record["steps/is_first"], dtype=np.int64).reshape(-1)
    is_last = np.asarray(record["steps/is_last"], dtype=np.int64).reshape(-1)
    if int(is_first[0]) != 1 or int(is_first.sum()) != 1:
        raise ValueError("episode must have exactly one first step at index zero")
    if int(is_last[-1]) != 1 or int(is_last.sum()) != 1:
        raise ValueError("episode must have exactly one last step at the final index")

    optical_forward = external[:3, 2]
    return (
        {
            "shard": shard_name,
            "record_index": int(record_index),
            "source_basename": Path(source_path).name,
            "language_instruction": language,
            "step_count": int(length),
            "terminal_reward": float(rewards[-1]),
            "external_camera_to_world_opencv": external.tolist(),
            "external_camera_position": external[:3, 3].tolist(),
            "external_optical_forward_z": float(optical_forward[2]),
            "wrist_translation_span_m": float(
                np.linalg.norm(wrist_last[:3, 3] - wrist_first[:3, 3])
            ),
            "wrist_rotation_span_deg": float(
                rotation_angle_degrees(wrist_delta[:3, :3])
            ),
            "agent_jpeg_bytes": int(sum(len(bytes(value)) for value in agent_images)),
            "wrist_jpeg_bytes": int(sum(len(bytes(value)) for value in wrist_images)),
        },
        action,
    )


def summarize_records(rows: Sequence[Mapping[str, Any]], actions: Sequence[np.ndarray]) -> dict[str, Any]:
    if not rows or not actions or len(rows) != len(actions):
        raise ValueError("rows and actions must be non-empty and aligned")
    concatenated = np.concatenate(actions, axis=0).astype(np.float64)
    positions = np.asarray([row["external_camera_position"] for row in rows])
    camera_keys = {
        tuple(np.round(np.asarray(row["external_camera_to_world_opencv"]).reshape(-1), 4))
        for row in rows
    }
    language_counts = Counter(str(row["language_instruction"]) for row in rows)
    return {
        "episode_count": len(rows),
        "step_count": int(sum(int(row["step_count"]) for row in rows)),
        "step_count_min": int(min(int(row["step_count"]) for row in rows)),
        "step_count_max": int(max(int(row["step_count"]) for row in rows)),
        "terminal_success_count": int(
            sum(float(row["terminal_reward"]) > 0.0 for row in rows)
        ),
        "unique_language_count": len(language_counts),
        "language_episode_counts": dict(sorted(language_counts.items())),
        "unique_external_camera_count_rounded_1e4": len(camera_keys),
        "external_camera_position_min": positions.min(axis=0).tolist(),
        "external_camera_position_max": positions.max(axis=0).tolist(),
        "external_camera_downward_fraction": float(
            np.mean([float(row["external_optical_forward_z"]) < 0.0 for row in rows])
        ),
        "wrist_translation_span_median": float(
            np.median([float(row["wrist_translation_span_m"]) for row in rows])
        ),
        "wrist_rotation_span_deg_median": float(
            np.median([float(row["wrist_rotation_span_deg"]) for row in rows])
        ),
        "action_min": concatenated.min(axis=0).tolist(),
        "action_max": concatenated.max(axis=0).tolist(),
        "action_mean": concatenated.mean(axis=0).tolist(),
        "action_std": concatenated.std(axis=0).tolist(),
    }


def _record_loader(path: Path) -> Iterable[Mapping[str, Any]]:
    try:
        from tfrecord.reader import tfrecord_loader
    except ImportError as error:
        raise RuntimeError(
            "Install the isolated TFRecord reader and run this script with "
            "/share/longjunyu/alphabrain/envs/libero-plus-data-v1/bin/python"
        ) from error
    return tfrecord_loader(str(path), None)


def inspect_dataset(
    *,
    dataset_root: Path,
    hand_eye_config: Path,
    max_shards: int | None,
) -> dict[str, Any]:
    shards = sorted(dataset_root.rglob("*.tfrecord-*"))
    if max_shards is not None:
        shards = shards[:max_shards]
    if not shards:
        raise FileNotFoundError(f"no TFRecord shards under {dataset_root}")
    calibration = json.loads(hand_eye_config.read_text())
    hand_eye = _rigid_transform(
        calibration["eef_to_wrist_opencv"],
        name="EEF-to-wrist hand-eye calibration",
    )

    rows: list[dict[str, Any]] = []
    actions: list[np.ndarray] = []
    for shard_index, shard in enumerate(shards, start=1):
        for record_index, record in enumerate(_record_loader(shard)):
            row, action = inspect_record(
                record,
                hand_eye=hand_eye,
                shard_name=str(shard.relative_to(dataset_root)),
                record_index=record_index,
            )
            rows.append(row)
            actions.append(action)
        print(
            f"audited shard {shard_index}/{len(shards)}: "
            f"episodes={len(rows)} path={shard.name}",
            file=sys.stderr,
            flush=True,
        )
    return {
        "schema_version": 1,
        "status": "complete",
        "study": "libero_plus_camparam_rlds_audit",
        "dataset_root": str(dataset_root),
        "hand_eye_config": str(hand_eye_config),
        "coordinate_conversion": {
            "source": "mujoco_camera_to_world_x_right_y_up_minus_z_forward",
            "target": "opencv_camera_to_world_x_right_y_down_z_forward",
            "right_multiply_diagonal": [1.0, -1.0, -1.0, 1.0],
        },
        "shard_count": len(shards),
        "summary": summarize_records(rows, actions),
        "episodes": rows,
    }


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit LIBERO-Plus camera-parameter RLDS")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--hand-eye-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shards", type=int)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    if args.max_shards is not None and args.max_shards <= 0:
        raise ValueError("max shards must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.output}")
    report = inspect_dataset(
        dataset_root=args.dataset_root,
        hand_eye_config=args.hand_eye_config,
        max_shards=args.max_shards,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), **report["summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
