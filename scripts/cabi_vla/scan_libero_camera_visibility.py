from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from collect_libero_bind_teacher import load_state_bank, upright_image
from evaluate_libero_bind_closed_loop import parse_state_indices
from libero_camera_pose import (
    camera_task_visibility,
    capture_camera_reference,
    install_camera_pose,
    load_camera_sweep_config,
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


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense policy-free LIBERO camera field-of-view scan"
    )
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--state-indices")
    parser.add_argument("--edges", default="all")
    parser.add_argument("--poses", default="all")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--settle-steps", type=int, default=8)
    parser.add_argument("--minimum-visible-pixels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args(args)


def main() -> None:
    from libero.libero.envs import OffScreenRenderEnv

    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite visibility scan: {args.output}")
    if args.resolution <= 0:
        raise ValueError("resolution must be positive")
    if args.settle_steps < 0:
        raise ValueError("settle_steps must be non-negative")
    if args.minimum_visible_pixels <= 0:
        raise ValueError("minimum-visible-pixels must be positive")

    manifest = json.loads((args.suite_root / "manifest.json").read_text())
    states = load_state_bank(Path(manifest["canonical_init_states"]))
    state_indices = parse_state_indices(args.state_indices, manifest, args.split)
    config = load_camera_sweep_config(args.camera_config)

    edge_by_name = {str(edge["edge_id"]): edge for edge in manifest["edges"]}
    edge_names = _parse_names(args.edges, available=edge_by_name, kind="edges")
    edges = [edge_by_name[name] for name in edge_names]
    pose_by_name = {str(pose["name"]): pose for pose in config["poses"]}
    pose_names = _parse_names(args.poses, available=pose_by_name, kind="poses")
    if "baseline" not in pose_names:
        raise ValueError("visibility scan requires baseline for paired image QC")
    pose_names = ["baseline", *[name for name in pose_names if name != "baseline"]]
    poses = [pose_by_name[name] for name in pose_names]

    rows: list[dict[str, Any]] = []
    baseline_images: dict[tuple[str, int], np.ndarray] = {}
    expected = len(edges) * len(poses) * len(state_indices)
    partial = args.output.with_name(f"{args.output.stem}.partial{args.output.suffix}")
    for edge in edges:
        env = OffScreenRenderEnv(
            bddl_file_name=edge["bddl"],
            camera_heights=args.resolution,
            camera_widths=args.resolution,
            horizon=args.settle_steps + 16,
            ignore_done=True,
        )
        try:
            env.seed(args.seed)
            reference = capture_camera_reference(
                env,
                camera_name=config["camera_name"],
                table_plane_z=config["table_plane_z"],
            )
            for pose in poses:
                for state_index in state_indices:
                    env.reset()
                    camera_metadata = install_camera_pose(env, reference, pose)
                    observation = env.set_init_state(np.asarray(states[state_index]))
                    for _ in range(args.settle_steps):
                        observation, _, _, _ = env.step(
                            np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
                        )
                    image = upright_image(observation["agentview_image"])
                    key = (str(edge["edge_id"]), int(state_index))
                    if pose["name"] == "baseline":
                        baseline_images[key] = image.copy()
                    if key not in baseline_images:
                        raise RuntimeError(
                            "baseline pose must run before perturbations for image QC"
                        )
                    baseline = baseline_images[key]
                    visibility = camera_task_visibility(
                        env,
                        observation,
                        camera_name=config["camera_name"],
                        source_object=str(edge["source_object"]),
                        target_object=str(edge["target_object"]),
                        height=args.resolution,
                        width=args.resolution,
                        minimum_pixels=args.minimum_visible_pixels,
                        detailed_geometry=True,
                    )
                    row = {
                        "edge_id": edge["edge_id"],
                        "source_id": edge["source_id"],
                        "target_id": edge["target_id"],
                        "source_object": edge["source_object"],
                        "target_object": edge["target_object"],
                        "action_supervised": bool(edge["action_supervised"]),
                        "canonical_state_index": int(state_index),
                        "split": args.split,
                        "camera_pose": pose["name"],
                        "camera_azimuth_deg": float(pose["azimuth_deg"]),
                        "camera_elevation_deg": float(pose["elevation_deg"]),
                        "camera_radius_scale": float(pose["radius_scale"]),
                        "sweep_axis": str(pose.get("sweep_axis", "explicit")),
                        "sweep_value": float(pose.get("sweep_value", 0.0)),
                        "initial_agent_mae_from_baseline": float(
                            np.mean(
                                np.abs(
                                    image.astype(np.float64)
                                    - baseline.astype(np.float64)
                                )
                            )
                        ),
                        "initial_agent_sha256": hashlib.sha256(
                            image.tobytes()
                        ).hexdigest(),
                        **camera_metadata,
                        **visibility,
                    }
                    rows.append(row)
                    _atomic_write(
                        partial,
                        {
                            "status": "partial",
                            "expected_record_count": expected,
                            "camera_config": config,
                            "rows": rows,
                        },
                    )
                    print(json.dumps(row, sort_keys=True), flush=True)
        finally:
            env.close()

    payload = {
        "schema_version": 1,
        "status": "complete",
        "study": "libero_bind_camera_visibility_scan",
        "suite": str(args.suite_root),
        "camera_config_path": str(args.camera_config),
        "camera_config": config,
        "split": args.split,
        "state_indices": state_indices,
        "edges": edge_names,
        "poses": pose_names,
        "resolution": args.resolution,
        "settle_steps": args.settle_steps,
        "minimum_visible_pixels": args.minimum_visible_pixels,
        "seed": args.seed,
        "expected_record_count": expected,
        "rows": rows,
    }
    _atomic_write(args.output, payload)
    partial.unlink(missing_ok=True)
    print(
        json.dumps(
            {"output": str(args.output), "record_count": len(rows)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
