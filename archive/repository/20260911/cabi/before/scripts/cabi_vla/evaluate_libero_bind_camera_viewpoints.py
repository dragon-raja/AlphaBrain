from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from collect_libero_bind_teacher import load_state_bank
from evaluate_libero_bind_closed_loop import (
    RemotePolicy,
    parse_state_indices,
    run_episode,
)
from libero_camera_pose import (
    camera_task_visibility,
    capture_camera_reference,
    install_camera_pose,
    load_camera_sweep_config,
    mujoco_camera_calibration,
)
from libero_scene_cues import (
    SCENE_CUE_MODES,
    capture_scene_cue_reference,
    install_scene_cues,
)


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _parse_names(
    value: str,
    *,
    available: Iterable[str],
    kind: str,
) -> list[str]:
    available_names = list(available)
    if value == "all":
        return available_names
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"{kind} must be a non-empty unique list")
    unknown = sorted(set(names) - set(available_names))
    if unknown:
        raise KeyError(f"unknown {kind}: {unknown}")
    return names


def make_camera_environment_setup(
    reference: Mapping[str, Any],
    pose: Mapping[str, Any],
    *,
    ray_pose: Mapping[str, Any] | None = None,
    scene_reference: Mapping[str, Any] | None = None,
    scene_cue_mode: str = "fixed",
    scene_cue_seed: int = 0,
    scene_sample_id: str = "",
    resolution: int = 224,
):
    def setup(env: Any) -> Mapping[str, Any]:
        scene_metadata = {}
        if scene_reference is not None:
            scene_metadata = install_scene_cues(
                env,
                scene_reference,
                mode=scene_cue_mode,
                seed=scene_cue_seed,
                sample_id=scene_sample_id,
            )
        metadata = install_camera_pose(env, reference, pose)
        result = {
            "camera_pose": str(pose["name"]),
            "policy_ray_pose": str(
                pose["name"] if ray_pose is None else ray_pose["name"]
            ),
            **scene_metadata,
            **metadata,
        }
        if ray_pose is not None:
            install_camera_pose(env, reference, ray_pose)
            calibration = mujoco_camera_calibration(
                env,
                camera_name=str(reference["camera_name"]),
                height=resolution,
                width=resolution,
            )
            result.update(
                {
                    "policy_camera_intrinsics": np.asarray(
                        calibration["intrinsics"]
                    ).tolist(),
                    "policy_camera_to_world_opencv": np.asarray(
                        calibration["camera_to_world_opencv"]
                    ).tolist(),
                }
            )
            install_camera_pose(env, reference, pose)
        else:
            calibration = mujoco_camera_calibration(
                env,
                camera_name=str(reference["camera_name"]),
                height=resolution,
                width=resolution,
            )
            result.update(
                {
                    "policy_camera_intrinsics": np.asarray(
                        calibration["intrinsics"]
                    ).tolist(),
                    "policy_camera_to_world_opencv": np.asarray(
                        calibration["camera_to_world_opencv"]
                    ).tolist(),
                }
            )
        return result

    return setup


def make_dual_camera_metadata_provider(
    *,
    wrist_ray_mode: str,
    resolution: int,
):
    initial_wrist: tuple[list[list[float]], list[list[float]]] | None = None
    previous_wrist: tuple[list[list[float]], list[list[float]]] | None = None

    def provide(
        env: Any,
        _observation: Mapping[str, Any],
        setup_metadata: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        nonlocal initial_wrist, previous_wrist
        external_intrinsics = setup_metadata.get(
            "policy_camera_intrinsics",
            setup_metadata.get("camera_intrinsics"),
        )
        external_to_world = setup_metadata.get(
            "policy_camera_to_world_opencv",
            setup_metadata.get("camera_to_world_opencv"),
        )
        if external_intrinsics is None or external_to_world is None:
            raise KeyError("external camera calibration is unavailable")

        calibration = mujoco_camera_calibration(
            env,
            camera_name="robot0_eye_in_hand",
            height=resolution,
            width=resolution,
        )
        current_wrist = (
            np.asarray(calibration["intrinsics"]).tolist(),
            np.asarray(calibration["camera_to_world_opencv"]).tolist(),
        )
        if initial_wrist is None:
            initial_wrist = current_wrist
        if wrist_ray_mode == "correct":
            selected_wrist = current_wrist
        elif wrist_ray_mode == "initial":
            selected_wrist = initial_wrist
        elif wrist_ray_mode == "lagged":
            selected_wrist = previous_wrist or current_wrist
        else:
            raise ValueError(f"unsupported wrist ray mode: {wrist_ray_mode}")
        previous_wrist = current_wrist
        return {
            "camera_intrinsics_by_view": [
                external_intrinsics,
                selected_wrist[0],
            ],
            "camera_to_world_opencv_by_view": [
                external_to_world,
                selected_wrist[1],
            ],
        }

    return provide


def make_camera_observation_setup(
    *,
    pose_name: str,
    camera_name: str,
    edge: Mapping[str, Any],
    resolution: int,
    minimum_visible_pixels: int,
    baseline_images: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]],
    episode_key: tuple[str, int, int],
):
    def setup(
        env: Any,
        observation: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        agent = np.asarray(observation["agentview_image"])
        wrist = np.asarray(observation["robot0_eye_in_hand_image"])
        if pose_name == "baseline":
            baseline_images[episode_key] = (agent.copy(), wrist.copy())
        if episode_key not in baseline_images:
            raise RuntimeError(
                "baseline pose must run before perturbed poses for camera QC"
            )
        canonical_agent, canonical_wrist = baseline_images[episode_key]
        metadata = {
            "initial_agent_mae_from_baseline": float(
                np.mean(
                    np.abs(agent.astype(np.float64) - canonical_agent.astype(np.float64))
                )
            ),
            "initial_wrist_mae_from_baseline": float(
                np.mean(
                    np.abs(wrist.astype(np.float64) - canonical_wrist.astype(np.float64))
                )
            ),
            "initial_agent_sha256": hashlib.sha256(agent.tobytes()).hexdigest(),
            "initial_wrist_sha256": hashlib.sha256(wrist.tobytes()).hexdigest(),
            **camera_task_visibility(
                env,
                observation,
                camera_name=camera_name,
                source_object=str(edge["source_object"]),
                target_object=str(edge["target_object"]),
                height=resolution,
                width=resolution,
                minimum_pixels=minimum_visible_pixels,
            ),
        }
        return observation, metadata

    return setup


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled fixed-policy LIBERO-Bind camera viewpoint evaluation"
    )
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--policy-socket", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--state-indices")
    parser.add_argument("--edges", default="all")
    parser.add_argument("--poses", default="all")
    parser.add_argument(
        "--ray-mode",
        choices=("correct", "canonical", "next_pose"),
        default="correct",
        help="camera calibration supplied to the policy while RGB stays fixed",
    )
    parser.add_argument(
        "--wrist-ray-mode",
        choices=("correct", "initial", "lagged"),
        default="correct",
        help="wrist calibration supplied at each policy inference",
    )
    parser.add_argument("--execution-horizons", type=int, nargs="+", default=[3])
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--minimum-visible-pixels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--scene-cue-mode",
        choices=SCENE_CUE_MODES,
        default="fixed",
    )
    parser.add_argument("--scene-cue-seed", type=int, default=20260728)
    parser.add_argument("--frame-dir", type=Path)
    parser.add_argument("--frame-poses", default="baseline")
    parser.add_argument("--frame-edges", default="all")
    parser.add_argument("--frame-episodes-per-edge", type=int, default=0)
    return parser.parse_args(args)


def main() -> None:
    from libero.libero.envs import OffScreenRenderEnv

    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {args.output}")
    if any(value not in (1, 2, 3) for value in args.execution_horizons):
        raise ValueError("execution horizons must be selected from 1, 2, 3")
    if args.frame_episodes_per_edge < 0:
        raise ValueError("frame episodes per edge must be non-negative")
    if args.resolution <= 0:
        raise ValueError("resolution must be positive")
    if args.minimum_visible_pixels <= 0:
        raise ValueError("minimum-visible-pixels must be positive")

    manifest = json.loads((args.suite_root / "manifest.json").read_text())
    states = load_state_bank(Path(manifest["canonical_init_states"]))
    state_indices = parse_state_indices(args.state_indices, manifest, args.split)
    config = load_camera_sweep_config(args.camera_config)
    pose_by_name = {str(pose["name"]): pose for pose in config["poses"]}
    pose_names = _parse_names(
        args.poses,
        available=pose_by_name,
        kind="camera poses",
    )
    if "baseline" not in pose_names:
        raise ValueError("camera sweep must include baseline for paired image QC")
    pose_names = ["baseline", *[name for name in pose_names if name != "baseline"]]
    poses = [pose_by_name[name] for name in pose_names]
    if args.ray_mode == "next_pose" and len(poses) < 2:
        raise ValueError("next_pose ray mode requires at least two visual poses")

    edge_by_name = {str(edge["edge_id"]): edge for edge in manifest["edges"]}
    edge_names = _parse_names(args.edges, available=edge_by_name, kind="edges")
    edges = [edge_by_name[name] for name in edge_names]
    frame_poses = set(
        _parse_names(args.frame_poses, available=pose_by_name, kind="frame poses")
    )
    frame_edges = set(
        _parse_names(args.frame_edges, available=edge_by_name, kind="frame edges")
    )

    policy = RemotePolicy(args.policy_socket)
    if max(args.execution_horizons) > policy.horizon:
        raise ValueError("execution horizon exceeds policy chunk")
    rows = []
    baseline_images: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}
    expected = (
        len(edges) * len(poses) * len(state_indices) * len(args.execution_horizons)
    )
    partial = args.output.with_name(f"{args.output.stem}.partial{args.output.suffix}")
    try:
        for edge in edges:
            env = OffScreenRenderEnv(
                bddl_file_name=edge["bddl"],
                camera_heights=args.resolution,
                camera_widths=args.resolution,
                horizon=args.max_steps + 16,
                ignore_done=True,
            )
            try:
                env.seed(args.seed)
                # The constructor has already built and forwarded the MuJoCo model.
                # Avoid an extra reset here so baseline episodes preserve the exact
                # reset / RNG sequence used by the canonical closed-loop evaluator.
                reference = capture_camera_reference(
                    env,
                    camera_name=config["camera_name"],
                    table_plane_z=config["table_plane_z"],
                )
                scene_reference = capture_scene_cue_reference(env)
                for pose_index, pose in enumerate(poses):
                    if args.ray_mode == "correct":
                        ray_pose = None
                    elif args.ray_mode == "canonical":
                        ray_pose = pose_by_name["baseline"]
                    else:
                        ray_pose = poses[(pose_index + 1) % len(poses)]
                    for execution_horizon in args.execution_horizons:
                        for state_position, state_index in enumerate(state_indices):
                            scene_sample_id = (
                                f"{edge['edge_id']}::state-{state_index}::"
                                f"k{execution_horizon}"
                            )
                            environment_setup = make_camera_environment_setup(
                                reference,
                                pose,
                                ray_pose=ray_pose,
                                scene_reference=scene_reference,
                                scene_cue_mode=args.scene_cue_mode,
                                scene_cue_seed=args.scene_cue_seed,
                                scene_sample_id=scene_sample_id,
                                resolution=args.resolution,
                            )
                            episode_key = (
                                str(edge["edge_id"]),
                                int(state_index),
                                int(execution_horizon),
                            )
                            record = (
                                args.frame_dir is not None
                                and pose["name"] in frame_poses
                                and edge["edge_id"] in frame_edges
                                and state_position < args.frame_episodes_per_edge
                            )
                            metrics, frames = run_episode(
                                env,
                                policy,
                                states[state_index],
                                edge,
                                execution_horizon=execution_horizon,
                                max_steps=args.max_steps,
                                seed=args.seed,
                                record_frames=record,
                                environment_setup=environment_setup,
                                episode_setup=make_camera_observation_setup(
                                    pose_name=str(pose["name"]),
                                    camera_name=config["camera_name"],
                                    edge=edge,
                                    resolution=args.resolution,
                                    minimum_visible_pixels=args.minimum_visible_pixels,
                                    baseline_images=baseline_images,
                                    episode_key=episode_key,
                                ),
                                policy_metadata_provider=(
                                    make_dual_camera_metadata_provider(
                                        wrist_ray_mode=args.wrist_ray_mode,
                                        resolution=args.resolution,
                                    )
                                ),
                            )
                            row = {
                                "edge_id": edge["edge_id"],
                                "source_id": edge["source_id"],
                                "target_id": edge["target_id"],
                                "action_supervised": bool(edge["action_supervised"]),
                                "canonical_state_index": state_index,
                                "split": args.split,
                                "execution_horizon": execution_horizon,
                                "ray_mode": args.ray_mode,
                                "wrist_ray_mode": args.wrist_ray_mode,
                                **metrics,
                            }
                            if frames is not None:
                                args.frame_dir.mkdir(parents=True, exist_ok=True)
                                frame_file = (
                                    f"{pose['name']}--{edge['edge_id']}--"
                                    f"state-{state_index:02d}--k{execution_horizon}.npz"
                                )
                                np.savez_compressed(args.frame_dir / frame_file, frames=frames)
                                row["frame_file"] = frame_file
                            rows.append(row)
                            _atomic_write(
                                partial,
                                {
                                    "status": "partial",
                                    "expected_episode_count": expected,
                                    "camera_config": config,
                                    "policy_identity": policy.identity,
                                    "rows": rows,
                                },
                            )
                            print(json.dumps(row, sort_keys=True), flush=True)
            finally:
                env.close()
    finally:
        policy.close()

    payload = {
        "schema_version": 1,
        "status": "complete",
        "study": "libero_bind_external_camera_pose_sweep",
        "suite": str(args.suite_root),
        "camera_config_path": str(args.camera_config),
        "camera_config": config,
        "split": args.split,
        "state_indices": state_indices,
        "edges": edge_names,
        "poses": pose_names,
        "execution_horizons": args.execution_horizons,
        "ray_mode": args.ray_mode,
        "wrist_ray_mode": args.wrist_ray_mode,
        "max_steps": args.max_steps,
        "resolution": args.resolution,
        "minimum_visible_pixels": args.minimum_visible_pixels,
        "seed": args.seed,
        "scene_cue_mode": args.scene_cue_mode,
        "scene_cue_seed": args.scene_cue_seed,
        "expected_episode_count": expected,
        "policy_identity": policy.identity,
        "rows": rows,
    }
    _atomic_write(args.output, payload)
    partial.unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output), "episode_count": len(rows)}))


if __name__ == "__main__":
    main()
