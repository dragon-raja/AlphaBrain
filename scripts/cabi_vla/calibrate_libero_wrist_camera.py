from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from collect_libero_bind_teacher import load_state_bank
from libero_camera_pose import mujoco_camera_calibration
from libero_wrist_camera import (
    average_rigid_transforms,
    eef_transform_from_pose,
    rotation_angle_degrees,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate LIBERO wrist hand-eye pose")
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--source-collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--edge", default="red-right")
    parser.add_argument("--state-indices", default="0,17,34")
    parser.add_argument("--sample-stride", type=int, default=20)
    parser.add_argument("--settle-steps", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--max-translation-residual-m", type=float, default=0.002)
    parser.add_argument("--max-rotation-residual-deg", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    from libero.libero.envs import OffScreenRenderEnv

    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {args.output}")
    state_indices = [int(value) for value in args.state_indices.split(",")]
    if not state_indices or len(state_indices) != len(set(state_indices)):
        raise ValueError("state indices must be a non-empty unique list")
    if args.sample_stride <= 0 or args.resolution <= 0 or args.settle_steps < 0:
        raise ValueError("invalid calibration sampling parameters")

    manifest = json.loads((args.suite_root / "manifest.json").read_text())
    edge = next(row for row in manifest["edges"] if row["edge_id"] == args.edge)
    states = load_state_bank(Path(manifest["canonical_init_states"]))
    relative_transforms = []
    sample_rows = []
    wrist_intrinsics = None
    canonical_wrist = None
    env = OffScreenRenderEnv(
        bddl_file_name=edge["bddl"],
        camera_heights=args.resolution,
        camera_widths=args.resolution,
        horizon=800,
        ignore_done=True,
    )
    try:
        env.seed(args.seed)
        for state_index in state_indices:
            episode_path = (
                args.source_collection
                / "episodes"
                / f"{args.edge}--state-{state_index:02d}.npz"
            )
            with np.load(episode_path, allow_pickle=False) as episode:
                env.reset()
                observation = env.set_init_state(np.asarray(states[state_index]))
                for _ in range(args.settle_steps):
                    observation, _, _, _ = env.step(
                        np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
                    )
                for frame in range(len(episode["eef_pose"])):
                    if frame % args.sample_stride == 0 or frame == len(
                        episode["eef_pose"]
                    ) - 1:
                        calibration = mujoco_camera_calibration(
                            env,
                            camera_name="robot0_eye_in_hand",
                            height=args.resolution,
                            width=args.resolution,
                        )
                        eef_to_world = eef_transform_from_pose(
                            np.concatenate(
                                [
                                    observation["robot0_eef_pos"],
                                    observation["robot0_eef_quat"],
                                ]
                            )
                        )
                        wrist_to_world = np.asarray(
                            calibration["camera_to_world_opencv"],
                            dtype=np.float64,
                        )
                        relative_transforms.append(
                            np.linalg.inv(eef_to_world) @ wrist_to_world
                        )
                        sample_rows.append(
                            {
                                "state_index": state_index,
                                "frame_index": frame,
                            }
                        )
                        if wrist_intrinsics is None:
                            wrist_intrinsics = np.asarray(
                                calibration["intrinsics"],
                                dtype=np.float64,
                            )
                            canonical_wrist = wrist_to_world
                    if frame < len(episode["actions"]):
                        observation, _, _, _ = env.step(
                            np.asarray(episode["actions"][frame], dtype=np.float32)
                        )
    finally:
        env.close()

    hand_eye = average_rigid_transforms(relative_transforms)
    translation_residuals = [
        float(np.linalg.norm(value[:3, 3] - hand_eye[:3, 3]))
        for value in relative_transforms
    ]
    rotation_residuals = [
        rotation_angle_degrees(hand_eye[:3, :3].T @ value[:3, :3])
        for value in relative_transforms
    ]
    max_translation = max(translation_residuals)
    max_rotation = max(rotation_residuals)
    if max_translation > args.max_translation_residual_m:
        raise RuntimeError(f"hand-eye translation residual is too high: {max_translation}")
    if max_rotation > args.max_rotation_residual_deg:
        raise RuntimeError(f"hand-eye rotation residual is too high: {max_rotation}")

    payload = {
        "schema_version": 1,
        "status": "validated",
        "camera_name": "robot0_eye_in_hand",
        "eef_pose_convention": "xyz+xyzw",
        "camera_convention": "opencv_camera_to_world",
        "resolution": args.resolution,
        "intrinsics": wrist_intrinsics.tolist(),
        "eef_to_wrist_opencv": hand_eye.tolist(),
        "canonical_wrist_camera_to_world_opencv": canonical_wrist.tolist(),
        "calibration_source": {
            "suite_root": str(args.suite_root),
            "source_collection": str(args.source_collection),
            "edge": args.edge,
            "state_indices": state_indices,
            "sample_stride": args.sample_stride,
            "sample_count": len(relative_transforms),
            "sample_rows_sha256": hashlib.sha256(
                json.dumps(sample_rows, sort_keys=True).encode()
            ).hexdigest(),
        },
        "quality": {
            "max_translation_residual_m": max_translation,
            "median_translation_residual_m": float(
                np.median(translation_residuals)
            ),
            "max_rotation_residual_deg": max_rotation,
            "median_rotation_residual_deg": float(np.median(rotation_residuals)),
            "translation_gate_m": args.max_translation_residual_m,
            "rotation_gate_deg": args.max_rotation_residual_deg,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), **payload["quality"]}, sort_keys=True))


if __name__ == "__main__":
    main()
