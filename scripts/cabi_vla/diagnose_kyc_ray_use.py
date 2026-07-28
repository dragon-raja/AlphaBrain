from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from collect_libero_bind_teacher import load_state_bank
from evaluate_libero_bind_closed_loop import (
    RemotePolicy,
    parse_state_indices,
    stable_seed,
)
from libero_camera_pose import (
    capture_camera_reference,
    install_camera_pose,
    load_camera_sweep_config,
    mujoco_camera_calibration,
)


def action_difference(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    first = np.asarray(reference, dtype=np.float64)
    second = np.asarray(candidate, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("action chunks must have the same [H, D] shape")
    delta = second - first
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    cosine = (
        float(np.sum(first * second) / denominator)
        if denominator > 1e-12
        else 1.0 if np.allclose(first, second) else 0.0
    )
    return {
        "chunk_rms": float(np.sqrt(np.mean(np.square(delta)))),
        "first_action_rms": float(
            np.sqrt(np.mean(np.square(delta[0])))
        ),
        "max_abs": float(np.max(np.abs(delta))),
        "cosine_similarity": cosine,
    }


def _parse_names(value: str, available: list[str], kind: str) -> list[str]:
    if value == "all":
        return list(available)
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"{kind} must be a non-empty unique list")
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise KeyError(f"unknown {kind}: {unknown}")
    return names


def _calibration_for_pose(
    env: Any,
    reference: Mapping[str, Any],
    pose: Mapping[str, Any],
    *,
    resolution: int,
) -> dict[str, Any]:
    install_camera_pose(env, reference, pose)
    calibration = mujoco_camera_calibration(
        env,
        camera_name=str(reference["camera_name"]),
        height=resolution,
        width=resolution,
    )
    return {
        "camera_intrinsics": np.asarray(calibration["intrinsics"]).tolist(),
        "camera_to_world_opencv": np.asarray(
            calibration["camera_to_world_opencv"]
        ).tolist(),
    }


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether a trained KYC policy uses supplied camera rays"
    )
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--state-indices")
    parser.add_argument("--edges", default="all")
    parser.add_argument("--poses", default="all")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args(args)


def main() -> None:
    from libero.libero.envs import OffScreenRenderEnv

    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic: {args.output}")
    manifest = json.loads((args.suite_root / "manifest.json").read_text())
    states = load_state_bank(Path(manifest["canonical_init_states"]))
    state_indices = parse_state_indices(args.state_indices, manifest, args.split)
    config = load_camera_sweep_config(args.camera_config)
    pose_by_name = {str(pose["name"]): pose for pose in config["poses"]}
    pose_names = _parse_names(
        args.poses,
        list(pose_by_name),
        "camera poses",
    )
    if len(pose_names) < 2:
        raise ValueError("ray diagnostic requires at least two camera poses")
    edge_by_name = {str(edge["edge_id"]): edge for edge in manifest["edges"]}
    edge_names = _parse_names(args.edges, list(edge_by_name), "edges")

    policy = RemotePolicy(args.policy_socket)
    rows = []
    try:
        for edge_name in edge_names:
            edge = edge_by_name[edge_name]
            env = OffScreenRenderEnv(
                bddl_file_name=edge["bddl"],
                camera_heights=args.resolution,
                camera_widths=args.resolution,
                horizon=64,
                ignore_done=True,
            )
            try:
                env.seed(args.seed)
                reference = capture_camera_reference(
                    env,
                    camera_name=config["camera_name"],
                    table_plane_z=config["table_plane_z"],
                )
                calibrations = {
                    name: _calibration_for_pose(
                        env,
                        reference,
                        pose_by_name[name],
                        resolution=args.resolution,
                    )
                    for name in pose_names
                }
                if "baseline" not in calibrations:
                    calibrations["baseline"] = _calibration_for_pose(
                        env,
                        reference,
                        pose_by_name["baseline"],
                        resolution=args.resolution,
                    )
                for pose_index, pose_name in enumerate(pose_names):
                    visual_pose = pose_by_name[pose_name]
                    mismatch_name = pose_names[(pose_index + 1) % len(pose_names)]
                    for state_index in state_indices:
                        env.reset()
                        install_camera_pose(env, reference, visual_pose)
                        observation = env.set_init_state(
                            np.asarray(states[state_index])
                        )
                        for _ in range(8):
                            observation, _, _, _ = env.step(
                                np.asarray(
                                    [0.0] * 6 + [-1.0],
                                    dtype=np.float32,
                                )
                            )
                        inference_seed = stable_seed(
                            args.seed,
                            edge_name,
                            pose_name,
                            state_index,
                        )
                        actions = {
                            "correct": policy.predict(
                                observation,
                                str(edge["language_instruction"]),
                                seed=inference_seed,
                                metadata=calibrations[pose_name],
                            ),
                            "canonical": policy.predict(
                                observation,
                                str(edge["language_instruction"]),
                                seed=inference_seed,
                                metadata=calibrations["baseline"],
                            ),
                            "mismatched": policy.predict(
                                observation,
                                str(edge["language_instruction"]),
                                seed=inference_seed,
                                metadata=calibrations[mismatch_name],
                            ),
                        }
                        row = {
                            "edge_id": edge_name,
                            "canonical_state_index": int(state_index),
                            "visual_pose": pose_name,
                            "mismatched_ray_pose": mismatch_name,
                            "agent_sha256": hashlib.sha256(
                                np.asarray(
                                    observation["agentview_image"]
                                ).tobytes()
                            ).hexdigest(),
                            "wrist_sha256": hashlib.sha256(
                                np.asarray(
                                    observation["robot0_eye_in_hand_image"]
                                ).tobytes()
                            ).hexdigest(),
                            "canonical_vs_correct": action_difference(
                                actions["correct"],
                                actions["canonical"],
                            ),
                            "mismatched_vs_correct": action_difference(
                                actions["correct"],
                                actions["mismatched"],
                            ),
                        }
                        rows.append(row)
                        print(json.dumps(row, sort_keys=True), flush=True)
            finally:
                env.close()
    finally:
        policy.close()

    payload = {
        "schema_version": 1,
        "study": "kyc_ray_use_diagnostic",
        "policy_identity": policy.identity,
        "camera_config": str(args.camera_config),
        "split": args.split,
        "state_indices": state_indices,
        "edges": edge_names,
        "poses": pose_names,
        "seed": args.seed,
        "rows": rows,
    }
    _atomic_write(args.output, payload)
    print(json.dumps({"output": str(args.output), "row_count": len(rows)}))


if __name__ == "__main__":
    main()

