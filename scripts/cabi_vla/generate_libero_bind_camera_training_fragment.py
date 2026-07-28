from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from collect_libero_bind_teacher import load_state_bank, robot_state, upright_image
from libero_camera_pose import (
    camera_task_visibility,
    capture_camera_reference,
    install_camera_pose,
    mujoco_camera_calibration,
)
from libero_scene_cues import (
    SCENE_CUE_MODES,
    capture_scene_cue_reference,
    install_scene_cues,
)


def stable_uniform(seed: int, sample_id: str, field: str) -> float:
    digest = hashlib.sha256(f"{seed}::{sample_id}::{field}".encode()).digest()
    integer = int.from_bytes(digest[:8], "little")
    return integer / float(1 << 64)


def load_randomization_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("camera training config must use schema_version=1")
    probability = float(payload.get("baseline_probability", -1.0))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("baseline_probability must be in [0, 1]")
    ranges = payload.get("ranges")
    expected = {"azimuth_deg", "elevation_deg", "radius_scale"}
    if not isinstance(ranges, Mapping) or set(ranges) != expected:
        raise ValueError(f"ranges must contain exactly {sorted(expected)}")
    normalized_ranges = {}
    for key in sorted(expected):
        values = np.asarray(ranges[key], dtype=np.float64)
        if (
            values.shape != (2,)
            or not np.all(np.isfinite(values))
            or values[1] < values[0]
        ):
            raise ValueError(f"invalid camera training range for {key}")
        normalized_ranges[key] = values.tolist()
    fixed = payload.get("fixed_variables", {})
    resolution = int(fixed.get("resolution", 224))
    if resolution <= 0:
        raise ValueError("fixed_variables.resolution must be positive")
    sampling_unit = str(
        payload.get("sampling_unit", "fixed_episode_camera_pool")
    )
    if sampling_unit not in {"fixed_episode_camera_pool", "global_camera_catalog"}:
        raise ValueError(
            "sampling_unit must be fixed_episode_camera_pool or "
            "global_camera_catalog"
        )
    poses_per_episode = int(payload.get("poses_per_episode", 1))
    if poses_per_episode <= 0:
        raise ValueError("poses_per_episode must be positive")
    camera_catalog_size = int(payload.get("camera_catalog_size", 1))
    if camera_catalog_size <= 0:
        raise ValueError("camera_catalog_size must be positive")
    epoch_replicas = int(payload.get("epoch_replicas", 1))
    if epoch_replicas <= 0:
        raise ValueError("epoch_replicas must be positive")
    scene_cue_mode = str(payload.get("scene_cue_mode", "fixed"))
    if scene_cue_mode not in SCENE_CUE_MODES:
        raise ValueError(f"scene_cue_mode must be one of {SCENE_CUE_MODES}")
    return {
        **payload,
        "seed": int(payload["seed"]),
        "baseline_probability": probability,
        "poses_per_episode": poses_per_episode,
        "sampling_unit": sampling_unit,
        "camera_catalog_size": camera_catalog_size,
        "epoch_replicas": epoch_replicas,
        "scene_cue_mode": scene_cue_mode,
        "scene_cue_seed": int(payload.get("scene_cue_seed", payload["seed"])),
        "ranges": normalized_ranges,
        "camera_name": str(payload["camera_name"]),
        "table_plane_z": float(payload["table_plane_z"]),
        "fixed_variables": {**fixed, "resolution": resolution},
    }


def sample_training_pose(
    *,
    sample_id: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    seed = int(config["seed"])
    if stable_uniform(seed, sample_id, "baseline") < float(
        config["baseline_probability"]
    ):
        return {
            "name": "baseline",
            "azimuth_deg": 0.0,
            "elevation_deg": 0.0,
            "radius_scale": 1.0,
        }
    values = {}
    for key, bounds in config["ranges"].items():
        low, high = map(float, bounds)
        unit = stable_uniform(seed, sample_id, key)
        values[key] = low + unit * (high - low)
    pose_digest = hashlib.sha256(f"{seed}::{sample_id}".encode()).hexdigest()[:12]
    return {"name": f"train-{pose_digest}", **values}


def episode_camera_pool(
    *,
    edge_id: str,
    episode_file: str,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return the finite, fixed camera set associated with one replay episode."""

    count = int(config.get("poses_per_episode", 1))
    return [
        sample_training_pose(
            sample_id=f"{edge_id}::{episode_file}::camera-{index}",
            config=config,
        )
        for index in range(count)
    ]


def global_camera_catalog(*, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return a nested deterministic catalog shared by every replay episode."""

    count = int(config.get("camera_catalog_size", 1))
    return [
        sample_training_pose(
            sample_id=f"global-camera-{index:04d}",
            config=config,
        )
        for index in range(count)
    ]


def training_camera_pool(
    *,
    edge_id: str,
    episode_file: str,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if config.get("sampling_unit", "fixed_episode_camera_pool") == (
        "global_camera_catalog"
    ):
        return global_camera_catalog(config=config)
    return episode_camera_pool(
        edge_id=edge_id,
        episode_file=episode_file,
        config=config,
    )


def camera_variant_index(
    *,
    sample_id: str,
    config: Mapping[str, Any],
    epoch_replica: int = 0,
) -> int:
    if epoch_replica < 0:
        raise ValueError("epoch_replica must be non-negative")
    sampling_unit = config.get(
        "sampling_unit",
        "fixed_episode_camera_pool",
    )
    count = int(
        config.get("camera_catalog_size", 1)
        if sampling_unit == "global_camera_catalog"
        else config.get("poses_per_episode", 1)
    )
    field = (
        "camera_variant"
        if sampling_unit == "fixed_episode_camera_pool" and epoch_replica == 0
        else f"camera_variant_epoch_{epoch_replica}"
    )
    unit = stable_uniform(int(config["seed"]), sample_id, field)
    return min(int(unit * count), count - 1)


def _restore_reference(env: Any, reference: Mapping[str, Any]) -> None:
    sim = env.env.sim
    camera_id = int(reference["camera_id"])
    sim.model.cam_pos[camera_id] = np.asarray(reference["position"])
    sim.model.cam_quat[camera_id] = np.asarray(reference["quaternion"])
    sim.forward()


def _sync_render_state(source_env: Any, render_env: Any) -> None:
    """Mirror physics into a render-only environment without perturbing replay."""

    source = source_env.env.sim
    target = render_env.env.sim
    target.set_state(source.get_state())
    if source.data.mocap_pos.shape == target.data.mocap_pos.shape:
        target.data.mocap_pos[:] = source.data.mocap_pos
        target.data.mocap_quat[:] = source.data.mocap_quat
    if source.data.ctrl.shape == target.data.ctrl.shape:
        target.data.ctrl[:] = source.data.ctrl
    target.forward()


def _render_agentview(
    env: Any,
    *,
    camera_name: str,
    resolution: int,
) -> np.ndarray:
    raw = env.sim.render(
        camera_name=camera_name,
        height=resolution,
        width=resolution,
    )
    return upright_image(raw)


def _render_wrist(
    env: Any,
    *,
    resolution: int,
) -> np.ndarray:
    raw = env.sim.render(
        camera_name="robot0_eye_in_hand",
        height=resolution,
        width=resolution,
    )
    return upright_image(raw)


def _parse_edges(value: str, available: set[str]) -> list[str]:
    if value == "all":
        return sorted(available)
    requested = [part.strip() for part in value.split(",") if part.strip()]
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("edges must be a non-empty unique list")
    unknown = sorted(set(requested) - available)
    if unknown:
        raise KeyError(f"unknown edges: {unknown}")
    return requested


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay LIBERO-Bind teacher actions and rerender randomized cameras"
    )
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--edges", default="all")
    parser.add_argument("--settle-steps", type=int, default=8)
    parser.add_argument("--replay-state-tolerance", type=float, default=5e-2)
    parser.add_argument(
        "--baseline-image-mae-tolerance",
        type=float,
        default=1.0,
        help="maximum mean uint8 pixel error allowed for canonical replay",
    )
    parser.add_argument("--minimum-visible-pixels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--camera-catalog-size", type=int)
    parser.add_argument("--epoch-replicas", type=int)
    parser.add_argument("--scene-cue-mode", choices=SCENE_CUE_MODES)
    parser.add_argument(
        "--record-limit",
        type=int,
        help="optional deterministic prefix limit for smoke validation",
    )
    return parser.parse_args(args)


def main() -> None:
    from libero.libero.envs import OffScreenRenderEnv

    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite camera fragment: {args.output}")
    if args.settle_steps < 0:
        raise ValueError("settle-steps must be non-negative")
    if args.replay_state_tolerance < 0 or args.baseline_image_mae_tolerance < 0:
        raise ValueError("replay tolerances must be non-negative")

    config = load_randomization_config(args.camera_config)
    if args.camera_catalog_size is not None:
        if args.camera_catalog_size <= 0:
            raise ValueError("camera-catalog-size must be positive")
        config["camera_catalog_size"] = int(args.camera_catalog_size)
    if args.epoch_replicas is not None:
        if args.epoch_replicas <= 0:
            raise ValueError("epoch-replicas must be positive")
        config["epoch_replicas"] = int(args.epoch_replicas)
    if args.scene_cue_mode is not None:
        config["scene_cue_mode"] = str(args.scene_cue_mode)
    resolution = int(config["fixed_variables"]["resolution"])
    source_manifest = json.loads((args.training_view / "manifest.json").read_text())
    suite_manifest = json.loads((args.suite_root / "manifest.json").read_text())
    states = load_state_bank(Path(suite_manifest["canonical_init_states"]))
    source_records = [
        json.loads(line)
        for line in (args.training_view / "records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    available_edges = {str(row["edge_id"]) for row in source_records}
    selected_edges = _parse_edges(args.edges, available_edges)
    edge_by_name = {
        str(edge["edge_id"]): edge for edge in suite_manifest["edges"]
    }

    selected: list[tuple[int, dict[str, Any]]] = [
        (index, row)
        for index, row in enumerate(source_records)
        if str(row["edge_id"]) in selected_edges
    ]
    if args.record_limit is not None:
        if args.record_limit <= 0:
            raise ValueError("record-limit must be positive")
        selected = selected[: args.record_limit]
    if not selected:
        raise ValueError("camera fragment contains no records")
    by_edge_episode: dict[str, dict[str, list[tuple[int, dict[str, Any]]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for source_index, row in selected:
        by_edge_episode[str(row["edge_id"])][str(row["episode_file"])].append(
            (source_index, row)
        )

    staging = args.output.parent / f".{args.output.name}.staging-{os.getpid()}"
    views_dir = staging / "camera_views"
    views_dir.mkdir(parents=True, exist_ok=False)
    epoch_replicas = int(config["epoch_replicas"])
    output_records: dict[tuple[int, int], dict[str, Any]] = {}
    replay_state_max_abs = 0.0
    replay_pose_max_abs = 0.0
    baseline_image_mae_max = 0.0
    visibility_counts = defaultdict(int)
    baseline_camera = None
    try:
        for edge_id in selected_edges:
            edge = edge_by_name[edge_id]
            env = OffScreenRenderEnv(
                bddl_file_name=edge["bddl"],
                camera_heights=resolution,
                camera_widths=resolution,
                horizon=800,
                ignore_done=True,
            )
            render_env = OffScreenRenderEnv(
                bddl_file_name=edge["bddl"],
                camera_heights=resolution,
                camera_widths=resolution,
                horizon=800,
                ignore_done=True,
            )
            try:
                env.seed(args.seed)
                render_env.seed(args.seed)
                reference = capture_camera_reference(
                    render_env,
                    camera_name=config["camera_name"],
                    table_plane_z=config["table_plane_z"],
                )
                scene_reference = capture_scene_cue_reference(render_env)
                _restore_reference(render_env, reference)
                calibration = mujoco_camera_calibration(
                    render_env,
                    camera_name=config["camera_name"],
                    height=resolution,
                    width=resolution,
                )
                current_baseline = {
                    "camera_intrinsics": np.asarray(
                        calibration["intrinsics"]
                    ).tolist(),
                    "camera_to_world_opencv": np.asarray(
                        calibration["camera_to_world_opencv"]
                    ).tolist(),
                }
                if baseline_camera is None:
                    baseline_camera = current_baseline
                else:
                    if not np.allclose(
                        baseline_camera["camera_intrinsics"],
                        current_baseline["camera_intrinsics"],
                        atol=1e-9,
                    ) or not np.allclose(
                        baseline_camera["camera_to_world_opencv"],
                        current_baseline["camera_to_world_opencv"],
                        atol=1e-9,
                    ):
                        raise RuntimeError("baseline camera differs across task edges")

                edge_dir = views_dir / edge_id
                edge_dir.mkdir(parents=True, exist_ok=False)
                for episode_file, episode_rows in sorted(
                    by_edge_episode[edge_id].items()
                ):
                    camera_pool = training_camera_pool(
                        edge_id=edge_id,
                        episode_file=episode_file,
                        config=config,
                    )
                    episode_path = Path(source_manifest["source_collection"]) / episode_file
                    with np.load(episode_path, allow_pickle=False) as episode:
                        frame_rows: dict[int, list[tuple[int, dict[str, Any]]]] = (
                            defaultdict(list)
                        )
                        for source_index, row in episode_rows:
                            frame_rows[int(row["frame_index"])].append(
                                (source_index, row)
                            )
                        max_frame = max(frame_rows)
                        state_indices = {
                            int(row["canonical_state_index"])
                            for _, row in episode_rows
                        }
                        if len(state_indices) != 1:
                            raise RuntimeError("episode rows disagree on canonical state")
                        state_index = next(iter(state_indices))

                        env.reset()
                        observation = env.set_init_state(
                            np.asarray(states[state_index])
                        )
                        for _ in range(args.settle_steps):
                            observation, _, _, _ = env.step(
                                np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
                            )

                        rendered: list[np.ndarray] = []
                        rendered_wrist: list[np.ndarray] = []
                        rendered_state: list[np.ndarray] = []
                        shard_rows: list[dict[str, Any]] = []
                        baseline_audited = False
                        for frame in range(max_frame + 1):
                            if frame in frame_rows:
                                state_error = float(
                                    np.max(
                                        np.abs(
                                            robot_state(observation)
                                            - np.asarray(
                                                episode["robot_state"][frame],
                                                dtype=np.float32,
                                            )
                                        )
                                    )
                                )
                                replay_state_max_abs = max(
                                    replay_state_max_abs,
                                    state_error,
                                )
                                if state_error > args.replay_state_tolerance:
                                    raise RuntimeError(
                                        f"teacher replay state drift {state_error} "
                                        f"at {episode_file} frame {frame}"
                                    )
                                current_poses = (
                                    np.concatenate(
                                        [
                                            observation["robot0_eef_pos"],
                                            observation["robot0_eef_quat"],
                                        ]
                                    ),
                                    np.concatenate(
                                        [
                                            observation[
                                                f"{edge['source_object']}_pos"
                                            ],
                                            observation[
                                                f"{edge['source_object']}_quat"
                                            ],
                                        ]
                                    ),
                                    np.concatenate(
                                        [
                                            observation[
                                                f"{edge['target_object']}_pos"
                                            ],
                                            observation[
                                                f"{edge['target_object']}_quat"
                                            ],
                                        ]
                                    ),
                                )
                                recorded_poses = (
                                    episode["eef_pose"][frame],
                                    episode["source_pose"][frame],
                                    episode["target_pose"][frame],
                                )
                                pose_error = max(
                                    float(
                                        np.max(
                                            np.abs(
                                                np.asarray(current)
                                                - np.asarray(recorded)
                                            )
                                        )
                                    )
                                    for current, recorded in zip(
                                        current_poses,
                                        recorded_poses,
                                    )
                                )
                                replay_pose_max_abs = max(
                                    replay_pose_max_abs,
                                    pose_error,
                                )
                                if pose_error > args.replay_state_tolerance:
                                    raise RuntimeError(
                                        f"teacher replay pose drift {pose_error} "
                                        f"at {episode_file} frame {frame}"
                                    )

                                _sync_render_state(env, render_env)
                                if not baseline_audited:
                                    _restore_reference(render_env, reference)
                                    baseline_render = _render_agentview(
                                        render_env,
                                        camera_name=config["camera_name"],
                                        resolution=resolution,
                                    )
                                    baseline_mae = float(
                                        np.mean(
                                            np.abs(
                                                baseline_render.astype(np.float64)
                                                - np.asarray(
                                                    episode["agentview"][frame],
                                                    dtype=np.float64,
                                                )
                                            )
                                        )
                                    )
                                    baseline_image_mae_max = max(
                                        baseline_image_mae_max,
                                        baseline_mae,
                                    )
                                    if baseline_mae > args.baseline_image_mae_tolerance:
                                        raise RuntimeError(
                                            f"baseline replay image MAE {baseline_mae} "
                                            f"at {episode_file} frame {frame}"
                                        )
                                    baseline_audited = True

                                for source_index, row in frame_rows[frame]:
                                    for epoch_replica in range(epoch_replicas):
                                        scene_metadata = install_scene_cues(
                                            render_env,
                                            scene_reference,
                                            mode=str(config["scene_cue_mode"]),
                                            seed=int(config["scene_cue_seed"]),
                                            sample_id=(
                                                f"{edge_id}::{episode_file}::"
                                                f"epoch-{epoch_replica}"
                                            ),
                                        )
                                        variant_index = camera_variant_index(
                                            sample_id=str(row["sample_id"]),
                                            config=config,
                                            epoch_replica=epoch_replica,
                                        )
                                        pose = camera_pool[variant_index]
                                        install_camera_pose(
                                            render_env,
                                            reference,
                                            pose,
                                        )
                                        image = _render_agentview(
                                            render_env,
                                            camera_name=config["camera_name"],
                                            resolution=resolution,
                                        )
                                        wrist = _render_wrist(
                                            render_env,
                                            resolution=resolution,
                                        )
                                        camera = mujoco_camera_calibration(
                                            render_env,
                                            camera_name=config["camera_name"],
                                            height=resolution,
                                            width=resolution,
                                        )
                                        visibility = camera_task_visibility(
                                            render_env,
                                            observation,
                                            camera_name=config["camera_name"],
                                            source_object=str(
                                                edge["source_object"]
                                            ),
                                            target_object=str(
                                                edge["target_object"]
                                            ),
                                            height=resolution,
                                            width=resolution,
                                            minimum_pixels=(
                                                args.minimum_visible_pixels
                                            ),
                                        )
                                        visibility_counts[
                                            "task_centers_in_frame"
                                            if visibility[
                                                "task_centers_in_frame"
                                            ]
                                            else "task_center_out_of_frame"
                                        ] += 1
                                        visibility_counts[
                                            "task_objects_visible"
                                            if visibility[
                                                "task_objects_visible"
                                            ]
                                            else "task_object_under_minimum"
                                        ] += 1

                                        shard_index = len(rendered)
                                        rendered.append(image)
                                        rendered_wrist.append(wrist)
                                        rendered_state.append(
                                            robot_state(observation)
                                        )
                                        source_sample_id = str(row["sample_id"])
                                        sample_id = (
                                            source_sample_id
                                            if epoch_replicas == 1
                                            else (
                                                f"{source_sample_id}::"
                                                f"camera-epoch-{epoch_replica:02d}"
                                            )
                                        )
                                        output_row = {
                                            **row,
                                            "sample_id": sample_id,
                                            "source_sample_id": source_sample_id,
                                            "source_record_index": source_index,
                                            "camera_epoch_replica": epoch_replica,
                                            "camera_view_file": (
                                                f"camera_views/{edge_id}/"
                                                f"{Path(episode_file).stem}.npz"
                                            ),
                                            "camera_view_index": shard_index,
                                            "camera_pose": pose["name"],
                                            "camera_pose_sampling_unit": str(
                                                config["sampling_unit"]
                                            ),
                                            "camera_variant_index": (
                                                variant_index
                                            ),
                                            "camera_catalog_size": len(
                                                camera_pool
                                            ),
                                            "camera_azimuth_deg": float(
                                                pose["azimuth_deg"]
                                            ),
                                            "camera_elevation_deg": float(
                                                pose["elevation_deg"]
                                            ),
                                            "camera_radius_scale": float(
                                                pose["radius_scale"]
                                            ),
                                            "camera_intrinsics": np.asarray(
                                                camera["intrinsics"]
                                            ).tolist(),
                                            "camera_to_world_opencv": np.asarray(
                                                camera[
                                                    "camera_to_world_opencv"
                                                ]
                                            ).tolist(),
                                            "camera_task_centers_in_frame": bool(
                                                visibility[
                                                    "task_centers_in_frame"
                                                ]
                                            ),
                                            "camera_task_objects_visible": bool(
                                                visibility[
                                                    "task_objects_visible"
                                                ]
                                            ),
                                            **scene_metadata,
                                        }
                                        output_records[
                                            (source_index, epoch_replica)
                                        ] = output_row
                                        shard_rows.append(
                                            {
                                                "source_record_index": (
                                                    source_index
                                                ),
                                                "camera_epoch_replica": (
                                                    epoch_replica
                                                ),
                                                "camera_view_index": (
                                                    shard_index
                                                ),
                                            }
                                        )
                            if frame < max_frame:
                                observation, _, _, _ = env.step(
                                    np.asarray(
                                        episode["actions"][frame],
                                        dtype=np.float32,
                                    )
                                )

                        for frame in range(max_frame, len(episode["actions"])):
                            observation, _, _, _ = env.step(
                                np.asarray(
                                    episode["actions"][frame],
                                    dtype=np.float32,
                                )
                            )
                        if not bool(env.check_success()):
                            raise RuntimeError(
                                f"open-loop teacher replay failed: {episode_file}"
                            )
                        np.savez_compressed(
                            edge_dir / f"{Path(episode_file).stem}.npz",
                            agentview=np.asarray(rendered, dtype=np.uint8),
                            wrist=np.asarray(rendered_wrist, dtype=np.uint8),
                            robot_state=np.asarray(
                                rendered_state,
                                dtype=np.float32,
                            ),
                        )
                    print(
                        json.dumps(
                            {
                                "edge_id": edge_id,
                                "episode_file": episode_file,
                                "rendered_records": (
                                    len(episode_rows) * epoch_replicas
                                ),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            finally:
                render_env.close()
                env.close()

        ordered_records = [
            output_records[(index, epoch_replica)]
            for index, _ in selected
            for epoch_replica in range(epoch_replicas)
        ]
        records_text = "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in ordered_records
        )
        (staging / "records.jsonl").write_text(records_text)
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "study": "libero_bind_random_camera_training_fragment",
            "source_training_view": str(args.training_view),
            "source_record_count": len(source_records),
            "source_records_sha256": hashlib.sha256(
                (args.training_view / "records.jsonl").read_bytes()
            ).hexdigest(),
            "selected_edges": selected_edges,
            "selected_source_record_count": len(selected),
            "selected_record_count": len(ordered_records),
            "camera_config_path": str(args.camera_config),
            "camera_config": config,
            "camera_config_sha256": hashlib.sha256(
                json.dumps(
                    config,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "camera_pose_sampling_unit": str(config["sampling_unit"]),
            "camera_epoch_replicas": epoch_replicas,
            "scene_cue_mode": str(config["scene_cue_mode"]),
            "baseline_camera": baseline_camera,
            "replay_state_max_abs": replay_state_max_abs,
            "replay_pose_max_abs": replay_pose_max_abs,
            "baseline_image_mae_max": baseline_image_mae_max,
            "baseline_image_mae_tolerance": args.baseline_image_mae_tolerance,
            "visibility_counts": dict(visibility_counts),
            "records_sha256": hashlib.sha256(records_text.encode()).hexdigest(),
        }
        _write_json(staging / "manifest.json", manifest)
        staging.rename(args.output)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "record_count": len(ordered_records),
                    "replay_state_max_abs": replay_state_max_abs,
                    "replay_pose_max_abs": replay_pose_max_abs,
                    "baseline_image_mae_max": baseline_image_mae_max,
                },
                sort_keys=True,
            )
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
