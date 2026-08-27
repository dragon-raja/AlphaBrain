from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _restore_reference(env: Any, reference: Mapping[str, Any]) -> None:
    sim = env.env.sim
    camera_id = int(reference["camera_id"])
    sim.model.cam_pos[camera_id] = np.asarray(reference["position"])
    sim.model.cam_quat[camera_id] = np.asarray(reference["quaternion"])
    sim.forward()


def _camera_metadata_from_reference(reference: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "camera_name": str(reference["camera_name"]),
        "camera_position": np.asarray(reference["position"], dtype=np.float64).tolist(),
        "camera_quaternion_wxyz": np.asarray(
            reference["quaternion"], dtype=np.float64
        ).tolist(),
        "camera_pivot": np.asarray(reference["pivot"], dtype=np.float64).tolist(),
        "camera_fovy": float(reference["fovy"]),
        "camera_azimuth_deg": 0.0,
        "camera_elevation_deg": 0.0,
        "camera_radius_scale": 1.0,
    }


def _quaternion_geodesic_deg(
    first_wxyz: Iterable[float],
    second_wxyz: Iterable[float],
) -> float:
    first = np.asarray(list(first_wxyz), dtype=np.float64)
    second = np.asarray(list(second_wxyz), dtype=np.float64)
    if first.shape != (4,) or second.shape != (4,):
        raise ValueError("camera quaternions must have four components")
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        raise ValueError("camera quaternions must be nonzero")
    cosine = float(np.dot(first / first_norm, second / second_norm))
    # q and -q encode the same rotation.
    cosine = float(np.clip(abs(cosine), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(cosine))


def _camera_displacement(
    canonical: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, float]:
    canonical_position = np.asarray(
        canonical["camera_position"], dtype=np.float64
    )
    candidate_position = np.asarray(
        candidate["camera_position"], dtype=np.float64
    )
    if canonical_position.shape != (3,) or candidate_position.shape != (3,):
        raise ValueError("camera positions must have three components")
    translation_m = float(np.linalg.norm(candidate_position - canonical_position))
    rotation_deg = _quaternion_geodesic_deg(
        canonical["camera_quaternion_wxyz"],
        candidate["camera_quaternion_wxyz"],
    )
    return {
        "translation_m": translation_m,
        "rotation_geodesic_deg": rotation_deg,
    }


def _local_rotation(*, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    rotation_y = np.asarray(
        [
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ]
    )
    rotation_x = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ]
    )
    return rotation_y @ rotation_x


def _install_look_away(
    env: Any,
    reference: Mapping[str, Any],
    pose: Mapping[str, Any],
) -> dict[str, Any]:
    from libero_camera_pose import (
        look_at_rotation,
        orbit_pose,
        rotation_matrix_to_wxyz,
    )

    base = orbit_pose(
        reference,
        {
            "azimuth_deg": float(pose.get("base_azimuth_deg", 0.0)),
            "elevation_deg": float(pose.get("base_elevation_deg", 0.0)),
            "radius_scale": float(pose.get("radius_scale", 1.0)),
        },
    )
    rotation = look_at_rotation(base["position"], base["pivot"])
    rotation = rotation @ _local_rotation(
        yaw_deg=float(pose.get("yaw_offset_deg", 0.0)),
        pitch_deg=float(pose.get("pitch_offset_deg", 0.0)),
    )
    quaternion = rotation_matrix_to_wxyz(rotation)
    sim = env.env.sim
    camera_id = int(reference["camera_id"])
    sim.model.cam_pos[camera_id] = base["position"]
    sim.model.cam_quat[camera_id] = quaternion
    sim.forward()
    return {
        "camera_position": base["position"].tolist(),
        "camera_quaternion_wxyz": quaternion.tolist(),
        "camera_pivot": base["pivot"].tolist(),
        "camera_azimuth_deg": base["azimuth_deg"],
        "camera_elevation_deg": base["elevation_deg"],
        "camera_radius_scale": base["radius_scale"],
        "yaw_offset_deg": float(pose.get("yaw_offset_deg", 0.0)),
        "pitch_offset_deg": float(pose.get("pitch_offset_deg", 0.0)),
    }


def _render_observation(env: Any) -> Mapping[str, Any]:
    env.env._update_observables(force=True)
    return env.env._get_observations()


def _sensor_control_records(canonical: Mapping[str, Any]) -> list[dict[str, Any]]:
    external = float(canonical["per_camera"]["agentview"]["score"])
    wrist = float(canonical["per_camera"]["robot0_eye_in_hand"]["score"])
    return [
        {
            "pose_id": "external_blackout",
            "group": "sensor_controls",
            "visibility_score": wrist / 2.0,
            "delta_visibility": wrist / 2.0 - float(canonical["score"]),
            "per_camera_scores": {"agentview": 0.0, "robot0_eye_in_hand": wrist},
        },
        {
            "pose_id": "wrist_blackout",
            "group": "sensor_controls",
            "visibility_score": external / 2.0,
            "delta_visibility": external / 2.0 - float(canonical["score"]),
            "per_camera_scores": {"agentview": external, "robot0_eye_in_hand": 0.0},
        },
        {
            "pose_id": "all_camera_blackout",
            "group": "sensor_controls",
            "visibility_score": 0.0,
            "delta_visibility": -float(canonical["score"]),
            "per_camera_scores": {"agentview": 0.0, "robot0_eye_in_hand": 0.0},
        },
    ]


def _filter_catalog_poses(
    poses: list[tuple[str, Mapping[str, Any]]],
    requested_pose_ids: Iterable[str] | None,
) -> list[tuple[str, Mapping[str, Any]]]:
    if requested_pose_ids is None:
        return poses
    requested = {str(value) for value in requested_pose_ids if str(value)}
    if not requested:
        raise ValueError("pose-id filter must not be empty")
    selected = [item for item in poses if str(item[1]["pose_id"]) in requested]
    found = {str(item[1]["pose_id"]) for item in selected}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"requested pose IDs are absent from selected groups: {missing}")
    return selected


def _remove_materialized_canonical(
    poses: list[tuple[str, Mapping[str, Any]]],
) -> list[tuple[str, Mapping[str, Any]]]:
    """Avoid rendering the canonical record twice.

    ``scan`` materializes canonical directly from the restored simulator state
    before applying catalog transforms.  The catalog entry is metadata, not an
    additional candidate.
    """
    return [
        item
        for item in poses
        if not (
            str(item[0]) == "canonical"
            and str(item[1].get("pose_id")) == "canonical"
        )
    ]


def _save_montage(
    *,
    path: Path,
    images: Mapping[str, tuple[np.ndarray, np.ndarray]],
    records: list[Mapping[str, Any]],
    top_k: int,
) -> None:
    from PIL import Image, ImageDraw

    usable = [record for record in records if record["pose_id"] in images]
    ranked = sorted(usable, key=lambda item: float(item["delta_visibility"]))
    chosen = ranked[:top_k] + ranked[-top_k:]
    canonical = next(
        (record for record in usable if record["pose_id"] == "canonical"), None
    )
    if canonical is not None:
        chosen = [canonical] + [
            record for record in chosen if record["pose_id"] != "canonical"
        ]
    deduplicated = {record["pose_id"]: record for record in chosen}
    chosen = list(deduplicated.values())
    if not chosen:
        return

    example_agent, example_wrist = next(iter(images.values()))
    image_height, single_width = example_agent.shape[:2]
    image_width = single_width * 2
    columns = min(5, len(chosen))
    rows = math.ceil(len(chosen) / columns)
    caption_height = 40
    canvas = Image.new(
        "RGB",
        (columns * image_width, rows * (image_height + caption_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, record in enumerate(chosen):
        column = index % columns
        row = index // columns
        left = column * image_width
        top = row * (image_height + caption_height)
        agent, wrist = images[record["pose_id"]]
        display_image = np.concatenate(
            [agent[::-1, ::-1], wrist[::-1, ::-1]], axis=1
        )
        canvas.paste(Image.fromarray(np.ascontiguousarray(display_image)), (left, top))
        draw.text(
            (left + 3, top + image_height + 2),
            f"{record['pose_id']}\ndV={float(record['delta_visibility']):+.5f}",
            fill="black",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _save_pose_images(
    *,
    output_dir: Path,
    images: Mapping[str, tuple[np.ndarray, np.ndarray]],
    requested: str | None,
) -> None:
    if requested is None:
        return
    from PIL import Image

    pose_ids = (
        sorted(images)
        if requested.strip().lower() == "all"
        else [value.strip() for value in requested.split(",") if value.strip()]
    )
    missing = sorted(set(pose_ids) - set(images))
    if missing:
        raise ValueError(f"requested saved pose images are unavailable: {missing}")
    view_dir = output_dir / "views"
    view_dir.mkdir(parents=True, exist_ok=True)
    for pose_id in pose_ids:
        agent, wrist = images[pose_id]
        display_image = np.concatenate(
            [agent[::-1, ::-1], wrist[::-1, ::-1]], axis=1
        )
        Image.fromarray(np.ascontiguousarray(display_image)).save(
            view_dir / f"{pose_id}.png"
        )


def scan(args: argparse.Namespace) -> dict[str, Any]:
    import h5py

    script_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_root))
    sys.path.insert(0, str(script_root.parent / "cabi_vla"))
    from audit_libero_hdf5_restore import _configure_runtime, _decode, _rewrite_model_paths

    runtime = args.runtime.resolve()
    hdf5_path = args.hdf5.resolve()
    output_dir = args.output_dir.resolve()
    _configure_runtime(runtime, hdf5_path.parent.parent, args.config_root.resolve())

    from libero.libero.envs import OffScreenRenderEnv
    from libero_camera_pose import capture_camera_reference, install_camera_pose
    from libero_constructed_view import (
        inject_static_visual_occluder,
        install_constructed_camera_pose,
        paired_task_orbit_poses,
        resolve_static_visual_occluder,
    )
    from libero_visibility import task_entity_visibility

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    groups = [name.strip() for name in args.groups.split(",") if name.strip()]
    poses = []
    for group in groups:
        if group not in catalog:
            raise KeyError(f"catalog has no group {group!r}")
        poses.extend((group, pose) for pose in catalog[group])
    pose_filter = (
        [value.strip() for value in args.pose_ids.split(",") if value.strip()]
        if args.pose_ids
        else None
    )

    suite = hdf5_path.parent.name
    with h5py.File(hdf5_path, "r") as handle:
        demos = sorted(handle["data"].keys())
        demo_name = demos[args.demo_index]
        demo = handle["data"][demo_name]
        frame = args.frame_index
        if frame < 0:
            frame += int(demo["states"].shape[0])
        state = np.asarray(demo["states"][frame])
        model_xml, rewrite_counts = _rewrite_model_paths(
            _decode(demo.attrs["model_file"]), runtime
        )
        bddl_name = Path(_decode(handle["data"].attrs["bddl_file_name"])).name
    bddl = runtime / "libero" / "libero" / "bddl_files" / suite / bddl_name

    scene_construction = None
    if args.construction_spec is not None:
        construction_spec = json.loads(
            args.construction_spec.read_text(encoding="utf-8")
        )
        probe = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_names=("agentview", "robot0_eye_in_hand"),
            camera_heights=args.resolution,
            camera_widths=args.resolution,
            render_gpu_device_id=args.render_gpu,
        )
        try:
            probe.reset()
            probe.reset_from_xml_string(model_xml)
            probe.set_init_state(state)
            original_nq = int(probe.env.sim.model.nq)
            scene_construction = resolve_static_visual_occluder(
                probe, construction_spec
            )
        finally:
            probe.close()
        model_xml = inject_static_visual_occluder(model_xml, scene_construction)
        poses.extend(
            ("constructed_task_orbit", pose)
            for pose in paired_task_orbit_poses(
                scene_construction,
                construction_spec.get("candidate_pairs", []),
            )
        )
    poses = _filter_catalog_poses(poses, pose_filter)
    poses = _remove_materialized_canonical(poses)

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_names=("agentview", "robot0_eye_in_hand"),
        camera_heights=args.resolution,
        camera_widths=args.resolution,
        render_gpu_device_id=args.render_gpu,
    )
    records = []
    images: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    try:
        env.reset()
        env.reset_from_xml_string(model_xml)
        env.set_init_state(state)
        if scene_construction is not None and int(env.env.sim.model.nq) != original_nq:
            raise RuntimeError("visual occluder changed the MuJoCo state dimension")
        initial_task_success = bool(env.check_success())
        entities = list(env.env.obj_of_interest)
        reference = capture_camera_reference(
            env,
            camera_name="agentview",
            table_plane_z=float(catalog["table_plane_z"]),
        )
        canonical_camera = _camera_metadata_from_reference(reference)
        canonical_visibility = task_entity_visibility(
            env,
            entity_names=entities,
            camera_names=("agentview", "robot0_eye_in_hand"),
            height=args.resolution,
            width=args.resolution,
        )
        canonical_observation = _render_observation(env)
        canonical_agent = np.asarray(
            canonical_observation["agentview_image"], dtype=np.uint8
        )
        canonical_wrist = np.asarray(
            canonical_observation["robot0_eye_in_hand_image"], dtype=np.uint8
        )
        images["canonical"] = (canonical_agent, canonical_wrist)
        records.append(
            {
                "pose_id": "canonical",
                "group": "canonical",
                "visibility_score": canonical_visibility["score"],
                "delta_visibility": 0.0,
                "per_camera_scores": {
                    name: value["score"]
                    for name, value in canonical_visibility["per_camera"].items()
                },
                "visibility": canonical_visibility,
                "camera": canonical_camera,
                "camera_displacement_from_canonical": {
                    "translation_m": 0.0,
                    "rotation_geodesic_deg": 0.0,
                },
                "pose": None,
            }
        )

        for group, pose in poses:
            pose_id = str(pose["pose_id"])
            _restore_reference(env, reference)
            try:
                if pose.get("orientation_mode") == "relative_look_away":
                    camera = _install_look_away(env, reference, pose)
                elif pose.get("orientation_mode") == "explicit_world_look_at":
                    camera = install_constructed_camera_pose(env, reference, pose)
                else:
                    camera = install_camera_pose(env, reference, pose)
                observation = _render_observation(env)
                visibility = task_entity_visibility(
                    env,
                    entity_names=entities,
                    camera_names=("agentview", "robot0_eye_in_hand"),
                    height=args.resolution,
                    width=args.resolution,
                )
                image = np.asarray(observation["agentview_image"], dtype=np.uint8)
                wrist_image = np.asarray(
                    observation["robot0_eye_in_hand_image"], dtype=np.uint8
                )
                images[pose_id] = (image, wrist_image)
                records.append(
                    {
                        "pose_id": pose_id,
                        "group": group,
                        "status": "PASS",
                        "visibility_score": visibility["score"],
                        "delta_visibility": visibility["score"]
                        - canonical_visibility["score"],
                        "per_camera_scores": {
                            name: value["score"]
                            for name, value in visibility["per_camera"].items()
                        },
                        "visibility": visibility,
                        "camera": camera,
                        "pose": dict(pose),
                        "camera_displacement_from_canonical": _camera_displacement(
                            canonical_camera,
                            camera,
                        ),
                        "image_mean": float(image.mean()),
                        "image_std": float(image.std()),
                        "wrist_image_mean": float(wrist_image.mean()),
                        "wrist_image_std": float(wrist_image.std()),
                    }
                )
            except Exception as error:
                records.append(
                    {
                        "pose_id": pose_id,
                        "group": group,
                        "status": "INVALID",
                        "error_type": type(error).__name__,
                    }
                )
        records.extend(_sensor_control_records(canonical_visibility))
        black_agent = np.zeros_like(canonical_agent)
        black_wrist = np.zeros_like(canonical_wrist)
        images.update(
            {
                "external_blackout": (black_agent, canonical_wrist),
                "wrist_blackout": (canonical_agent, black_wrist),
                "all_camera_blackout": (black_agent, black_wrist),
            }
        )
    finally:
        env.close()

    valid = [record for record in records if "delta_visibility" in record]
    result = {
        "schema": "dsol_libero_hdf5_view_scan_v1",
        "status": "PASS" if valid else "FAIL",
        "hdf5": str(hdf5_path),
        "suite": suite,
        "demo": demo_name,
        "frame": frame,
        "bddl": str(bddl),
        "task_entities": entities,
        "catalog": str(args.catalog.resolve()),
        "montage_image_transform": "rot180_display_only",
        "groups": groups,
        "requested_pose_ids": pose_filter,
        "asset_path_rewrites": rewrite_counts,
        "scene_construction": scene_construction,
        "initial_task_success": initial_task_success,
        "records": records,
        "valid_records": len(valid),
        "invalid_records": sum(record.get("status") == "INVALID" for record in records),
        "delta_visibility_min": min(float(record["delta_visibility"]) for record in valid),
        "delta_visibility_max": max(float(record["delta_visibility"]) for record in valid),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "scan.json", result)
    _save_montage(
        path=output_dir / "visibility_extremes.png",
        images=images,
        records=records,
        top_k=args.montage_top_k,
    )
    _save_pose_images(
        output_dir=output_dir,
        images=images,
        requested=args.save_pose_images,
    )
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan exact-state LIBERO views using task-entity visibility."
    )
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument(
        "--pose-ids",
        help="Optional comma-separated subset of pose IDs from the selected groups.",
    )
    parser.add_argument("--demo-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--render-gpu", type=int, default=0)
    parser.add_argument("--montage-top-k", type=int, default=6)
    parser.add_argument(
        "--save-pose-images",
        help="Comma-separated pose IDs to save, or 'all'.",
    )
    parser.add_argument(
        "--construction-spec",
        type=Path,
        help=(
            "Optional JSON specification for a state-dependent static visual "
            "occluder and task-centric paired candidates."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    result = scan(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "valid_records": result["valid_records"],
                "invalid_records": result["invalid_records"],
                "delta_visibility_min": result["delta_visibility_min"],
                "delta_visibility_max": result["delta_visibility_max"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
