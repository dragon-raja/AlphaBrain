from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_pi05_libero_plus_views import DUMMY_ACTION, stable_seed
from libero_camera_pose import (
    mujoco_camera_calibration,
    opencv_pixels_to_policy,
    project_world_points,
)


CONDITION_TASK_KEYS = {
    "canonical": "base_task",
    "camera_only": "camera_task_name",
    "background_only": "background_task_name",
    "camera_background": "camera_background_task_name",
}


def policy_centroid(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or not np.any(values):
        raise ValueError("mask must be a non-empty 2D array")
    yy, xx = np.nonzero(values)
    height, width = values.shape
    raw = np.asarray([float(np.mean(xx)), float(np.mean(yy))])
    return np.asarray([width - 1 - raw[0], height - 1 - raw[1]])


def object_alignment_rows(
    env: Any,
    *,
    camera_name: str,
    height: int,
    width: int,
) -> list[dict[str, Any]]:
    sim = env.env.sim
    sim.forward()
    segmentation = np.asarray(
        sim.render(
            camera_name=camera_name,
            width=width,
            height=height,
            depth=False,
            segmentation=True,
        )
    )
    if segmentation.shape != (height, width, 2):
        raise ValueError(f"unexpected segmentation shape: {segmentation.shape}")
    geom_ids = np.where(segmentation[..., 0] == 5, segmentation[..., 1], -1)
    geom_to_instance = {
        int(geom_id): str(instance)
        for geom_id, instance in env.env.model.geom_ids_to_instances.items()
    }
    calibration = mujoco_camera_calibration(
        env,
        camera_name=camera_name,
        height=height,
        width=width,
    )

    rows = []
    for instance in sorted(str(name) for name in env.obj_of_interest):
        ids = sorted(
            geom_id
            for geom_id, mapped_instance in geom_to_instance.items()
            if mapped_instance == instance
        )
        visible = []
        for geom_id in ids:
            mask = geom_ids == geom_id
            pixels = int(np.count_nonzero(mask))
            if pixels:
                visible.append((geom_id, pixels, mask))
        if not visible:
            rows.append(
                {
                    "object_instance": instance,
                    "visible": False,
                    "pixel_count": 0,
                }
            )
            continue
        union = np.logical_or.reduce([item[2] for item in visible])
        weights = np.asarray([item[1] for item in visible], dtype=np.float64)
        centers = np.asarray(
            [sim.data.geom_xpos[item[0]] for item in visible],
            dtype=np.float64,
        )
        world_center = np.average(centers, axis=0, weights=weights)
        projected, depths = project_world_points(
            world_center,
            intrinsics=calibration["intrinsics"],
            camera_to_world=calibration["camera_to_world_opencv"],
        )
        predicted = opencv_pixels_to_policy(projected, width=width)[0]
        observed = policy_centroid(union)
        finite = bool(np.all(np.isfinite(predicted)) and depths[0] > 0)
        rows.append(
            {
                "object_instance": instance,
                "visible": True,
                "pixel_count": int(np.count_nonzero(union)),
                "visible_geom_count": len(visible),
                "world_center": world_center.tolist(),
                "projected_policy_pixel": predicted.tolist(),
                "segmentation_policy_centroid": observed.tolist(),
                "projection_depth": float(depths[0]),
                "finite_projection": finite,
                "pixel_error": (
                    float(np.linalg.norm(predicted - observed)) if finite else None
                ),
            }
        )
    return rows


def render_policy_image(observation: Mapping[str, Any]) -> np.ndarray:
    return np.ascontiguousarray(
        np.asarray(observation["agentview_image"], dtype=np.uint8)[::-1, ::-1]
    )


def run_validation(
    *,
    protocol: Mapping[str, Any],
    bddl_root: Path,
    task_count: int,
    render_gpu: int,
    seed: int,
    resolution: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    tasks = list(protocol["composition_tasks"])[:task_count]
    if not tasks:
        raise ValueError("protocol contains no composition tasks")
    rows = []
    panels = []
    suites: dict[str, Any] = {}
    for task in tasks:
        suite_name = str(task["suite"])
        if suite_name not in suites:
            suites[suite_name] = benchmark.get_benchmark_dict()[suite_name]()
        task_suite = suites[suite_name]
        init_states = task_suite.get_task_init_states(int(task["task_index"]))
        for condition, key in CONDITION_TASK_KEYS.items():
            task_name = str(task[key])
            bddl = bddl_root / suite_name / f"{task_name}.bddl"
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl),
                camera_heights=resolution,
                camera_widths=resolution,
                render_gpu_device_id=render_gpu,
            )
            episode_seed = stable_seed(
                f"ray-alignment::{suite_name}::{task_name}",
                seed=seed,
            )
            env.seed(episode_seed)
            env.reset()
            observation = env.set_init_state(init_states[0])
            for _ in range(10):
                observation, _, _, _ = env.step(DUMMY_ACTION)
            alignment = object_alignment_rows(
                env,
                camera_name="agentview",
                height=resolution,
                width=resolution,
            )
            rows.append(
                {
                    "suite": suite_name,
                    "base_task": str(task["base_task"]),
                    "condition": condition,
                    "camera_task_name": task_name,
                    "objects": alignment,
                }
            )
            panels.append(
                {
                    "title": f"{condition}\n{suite_name}",
                    "image": render_policy_image(observation),
                    "objects": alignment,
                }
            )
            env.close()

    valid = [
        float(obj["pixel_error"])
        for row in rows
        for obj in row["objects"]
        if obj.get("visible") and obj.get("finite_projection")
    ]
    if not valid:
        raise ValueError("no visible task objects had finite projections")
    report = {
        "schema_version": 1,
        "study": "libero_plus_rgb_ray_pixel_alignment",
        "coordinate_convention": {
            "camera": "opencv_camera_to_world_x_right_y_down_z_forward",
            "policy_image_transform": "mujoco_raw_rot180",
            "ray_transform": "horizontal_flip",
        },
        "task_count": len(tasks),
        "condition_count": len(CONDITION_TASK_KEYS),
        "episode_count": len(rows),
        "visible_object_projection_count": len(valid),
        "pixel_error": {
            "median": float(np.median(valid)),
            "p90": float(np.quantile(valid, 0.9)),
            "maximum": float(np.max(valid)),
        },
        "gate": {
            "median_le_8px": bool(np.median(valid) <= 8.0),
            "p90_le_15px": bool(np.quantile(valid, 0.9) <= 15.0),
            "passed": bool(np.median(valid) <= 8.0 and np.quantile(valid, 0.9) <= 15.0),
        },
        "episodes": rows,
    }
    return report, panels


def render_figure(panels: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    columns = len(CONDITION_TASK_KEYS)
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.5 * columns, 3.7 * rows))
    axes_array = np.asarray(axes, dtype=object).reshape(-1)
    for axis, panel in zip(axes_array, panels, strict=False):
        axis.imshow(panel["image"])
        for obj in panel["objects"]:
            if not obj.get("visible") or not obj.get("finite_projection"):
                continue
            px, py = obj["projected_policy_pixel"]
            cx, cy = obj["segmentation_policy_centroid"]
            axis.scatter([px], [py], marker="+", s=90, c="#D62728", linewidths=2)
            axis.scatter([cx], [cy], marker="o", s=45, facecolors="none", edgecolors="#00A6D6", linewidths=1.5)
        axis.set_title(str(panel["title"]), fontsize=9)
        axis.set_axis_off()
    for axis in axes_array[len(panels) :]:
        axis.set_axis_off()
    figure.suptitle("Ray projection (+) vs. task-object segmentation centroid (o)")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate LIBERO RGB/ray alignment")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bddl-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=4)
    parser.add_argument("--render-gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--resolution", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task_count <= 0 or args.resolution <= 0:
        raise ValueError("task count and resolution must be positive")
    if args.output_json.exists() or args.output_figure.exists():
        raise FileExistsError("refusing to overwrite ray-alignment outputs")
    protocol = json.loads(args.protocol.read_text())
    report, panels = run_validation(
        protocol=protocol,
        bddl_root=args.bddl_root,
        task_count=args.task_count,
        render_gpu=args.render_gpu,
        seed=args.seed,
        resolution=args.resolution,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_name(f".{args.output_json.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output_json)
    render_figure(panels, args.output_figure)
    print(json.dumps({"gate": report["gate"], "pixel_error": report["pixel_error"]}, sort_keys=True))


if __name__ == "__main__":
    main()
