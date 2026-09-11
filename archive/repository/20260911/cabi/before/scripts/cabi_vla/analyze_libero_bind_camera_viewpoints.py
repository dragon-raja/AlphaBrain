from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


RATE_METRICS = (
    "success",
    "source_selection_success",
    "wrong_source_grasp",
    "lift_success",
    "transport_success",
)


def wilson_interval(successes: int, count: int, *, z: float = 1.96) -> tuple[float, float]:
    if count <= 0 or not 0 <= successes <= count:
        raise ValueError("Wilson interval requires 0 <= successes <= count")
    rate = successes / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["camera_pose"])].append(row)
    summaries = []
    for pose, values in grouped.items():
        first = values[0]
        completed_steps = [
            float(row["completion_steps"]) for row in values if bool(row["success"])
        ]
        success_count = sum(bool(row["success"]) for row in values)
        wilson_low, wilson_high = wilson_interval(success_count, len(values))
        summary = {
            "camera_pose": pose,
            "episode_count": len(values),
            "state_count": len({int(row["canonical_state_index"]) for row in values}),
            "edge_count": len({str(row["edge_id"]) for row in values}),
            "azimuth_deg": float(first["camera_azimuth_deg"]),
            "elevation_deg": float(first["camera_elevation_deg"]),
            "radius_scale": float(first["camera_radius_scale"]),
            "success_count": success_count,
            "success_wilson95_low": wilson_low,
            "success_wilson95_high": wilson_high,
            "progress": float(np.mean([float(row["progress"]) for row in values])),
            "mean_completion_steps_capped": float(
                np.mean([float(row["completion_steps"]) for row in values])
            ),
            "mean_successful_completion_steps": (
                float(np.mean(completed_steps)) if completed_steps else None
            ),
        }
        for metric in RATE_METRICS:
            summary[metric] = float(np.mean([float(row[metric]) for row in values]))
        summaries.append(summary)
    return sorted(summaries, key=lambda row: row["camera_pose"])


def _group_metric(
    rows: Sequence[Mapping[str, Any]],
    *,
    pose: str,
    metric: str,
) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if str(row["camera_pose"]) == pose:
            grouped[int(row["canonical_state_index"])].append(float(row[metric]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def paired_bootstrap_delta(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    baseline: str,
    metric: str,
    samples: int = 10000,
    seed: int = 20260726,
) -> dict[str, Any]:
    candidate_groups = _group_metric(rows, pose=candidate, metric=metric)
    baseline_groups = _group_metric(rows, pose=baseline, metric=metric)
    group_ids = sorted(set(candidate_groups) & set(baseline_groups))
    if not group_ids:
        raise ValueError("candidate and baseline have no paired state groups")
    differences = np.asarray(
        [candidate_groups[key] - baseline_groups[key] for key in group_ids],
        dtype=np.float64,
    )
    if len(differences) == 1:
        lower = upper = float(differences[0])
    else:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(differences), size=(samples, len(differences)))
        bootstrap = differences[indices].mean(axis=1)
        lower, upper = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    return {
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric,
        "paired_state_count": len(group_ids),
        "delta": float(np.mean(differences)),
        "ci95": [float(lower), float(upper)],
    }


def rank_nonbaseline(summaries: Sequence[Mapping[str, Any]]) -> list[str]:
    candidates = [row for row in summaries if row["camera_pose"] != "baseline"]
    candidates.sort(
        key=lambda row: (
            float(row["success"]),
            float(row["progress"]),
            -float(row["wrong_source_grasp"]),
            float(row["transport_success"]),
            -float(row["mean_completion_steps_capped"]),
        ),
        reverse=True,
    )
    return [str(row["camera_pose"]) for row in candidates]


def _axis_rows(
    summaries: Sequence[Mapping[str, Any]],
    *,
    axis: str,
) -> list[Mapping[str, Any]]:
    rows = []
    for row in summaries:
        azimuth = float(row["azimuth_deg"])
        elevation = float(row["elevation_deg"])
        radius = float(row["radius_scale"])
        if axis == "azimuth" and elevation == 0.0 and radius == 1.0:
            rows.append(row)
        elif axis == "elevation" and azimuth == 0.0 and radius == 1.0:
            rows.append(row)
        elif axis == "radius" and azimuth == 0.0 and elevation == 0.0:
            rows.append(row)
    key = {
        "azimuth": "azimuth_deg",
        "elevation": "elevation_deg",
        "radius": "radius_scale",
    }[axis]
    return sorted(rows, key=lambda row: float(row[key]))


def plot_curves(summaries: Sequence[Mapping[str, Any]], output: Path) -> None:
    width, height = 1500, 500
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), "LIBERO-Bind fixed-policy camera sensitivity (K=3)", fill="black")
    labels = {
        "azimuth": ("Azimuth offset (deg)", "azimuth_deg"),
        "elevation": ("Elevation offset (deg)", "elevation_deg"),
        "radius": ("Radius scale", "radius_scale"),
    }
    metrics = (
        ("success", "Task success", "#0072B2"),
        ("progress", "Subgoal progress", "#E69F00"),
        ("source_selection_success", "Correct grasp", "#009E73"),
        ("transport_success", "Transport", "#CC79A7"),
    )
    panel_width = 470
    panel_top = 70
    plot_top = 105
    plot_bottom = 430
    plot_height = plot_bottom - plot_top
    for panel_index, axis_name in enumerate(("azimuth", "elevation", "radius")):
        panel_left = 20 + panel_index * 495
        plot_left = panel_left + 55
        plot_right = panel_left + panel_width - 20
        rows = _axis_rows(summaries, axis=axis_name)
        x_label, x_key = labels[axis_name]
        x_values = [float(row[x_key]) for row in rows]
        x_min, x_max = min(x_values), max(x_values)
        if x_min == x_max:
            x_min -= 1.0
            x_max += 1.0
        draw.text((panel_left + 190, panel_top), axis_name.title(), fill="black")
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="black", width=2)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="black", width=2)
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = plot_bottom - int(tick * plot_height)
            draw.line((plot_left, y, plot_right, y), fill="#DDDDDD", width=1)
            draw.text((plot_left - 38, y - 6), f"{tick:.2g}", fill="#555555")
        for x_value in x_values:
            x = plot_left + int((x_value - x_min) / (x_max - x_min) * (plot_right - plot_left))
            draw.line((x, plot_bottom, x, plot_bottom + 5), fill="black", width=1)
            draw.text((x - 16, plot_bottom + 9), f"{x_value:g}", fill="#555555")
        for metric, _label, color in metrics:
            points = []
            for x_value, row in zip(x_values, rows, strict=True):
                x = plot_left + int(
                    (x_value - x_min) / (x_max - x_min) * (plot_right - plot_left)
                )
                y = plot_bottom - int(float(row[metric]) * plot_height)
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=color, width=3, joint="curve")
            for x, y in points:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="white")
        baseline_x = 1.0 if axis_name == "radius" else 0.0
        baseline_pixel = plot_left + int(
            (baseline_x - x_min) / (x_max - x_min) * (plot_right - plot_left)
        )
        for y in range(plot_top, plot_bottom, 10):
            draw.line(
                (baseline_pixel, y, baseline_pixel, min(y + 5, plot_bottom)),
                fill="#666666",
                width=1,
            )
        draw.text((panel_left + 175, 465), x_label, fill="black")
    legend_x = 920
    for index, (_metric, label, color) in enumerate(metrics):
        x = legend_x + (index % 2) * 210
        y = 18 + (index // 2) * 22
        draw.line((x, y + 6, x + 24, y + 6), fill=color, width=3)
        draw.text((x + 30, y), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def plot_metric_matrix(
    summaries: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    metrics = [
        "success",
        "source_selection_success",
        "lift_success",
        "transport_success",
        "progress",
        "wrong_source_grasp",
    ]
    poses = [str(row["camera_pose"]) for row in summaries]
    matrix = np.asarray(
        [[float(row[metric]) for row in summaries] for metric in metrics],
        dtype=np.float64,
    )
    metric_labels = [
        "Task success",
        "Correct grasp",
        "Lift",
        "Transport",
        "Progress",
        "Wrong grasp",
    ]
    cell_width = 90
    cell_height = 54
    left = 150
    top = 100
    canvas = Image.new(
        "RGB",
        (left + cell_width * len(poses) + 20, top + cell_height * len(metrics) + 20),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 18), "Closed-loop behavior by camera pose", fill="black")
    for column, pose in enumerate(poses):
        draw.text((left + column * cell_width + 4, 64), pose, fill="black")
    for row_index, label in enumerate(metric_labels):
        draw.text((12, top + row_index * cell_height + 19), label, fill="black")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            color = (
                int(35 + value * 205),
                int(55 + value * 160),
                int(110 - value * 65),
            )
            x0 = left + column_index * cell_width
            y0 = top + row_index * cell_height
            draw.rectangle(
                (x0, y0, x0 + cell_width, y0 + cell_height),
                fill=color,
                outline="white",
            )
            text_color = "white" if value < 0.45 else "black"
            draw.text((x0 + 33, y0 + 20), f"{value:.2f}", fill=text_color)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def summarize_by_edge(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["edge_id"])].append(row)
    values = []
    for edge_id, edge_rows in sorted(grouped.items()):
        for summary in summarize_rows(edge_rows):
            values.append({"edge_id": edge_id, **summary})
    return values


def plot_edge_success_matrix(
    edge_summaries: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    poses = sorted({str(row["camera_pose"]) for row in edge_summaries})
    edges = sorted({str(row["edge_id"]) for row in edge_summaries})
    lookup = {
        (str(row["edge_id"]), str(row["camera_pose"])): float(row["success"])
        for row in edge_summaries
    }
    cell_width = 90
    cell_height = 56
    left = 150
    top = 100
    canvas = Image.new(
        "RGB",
        (left + cell_width * len(poses) + 20, top + cell_height * len(edges) + 20),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 18), "Task success by edge and camera pose", fill="black")
    for column, pose in enumerate(poses):
        draw.text((left + column * cell_width + 4, 64), pose, fill="black")
    for row_index, edge in enumerate(edges):
        draw.text((12, top + row_index * cell_height + 20), edge, fill="black")
        for column_index, pose in enumerate(poses):
            value = lookup[(edge, pose)]
            color = (
                int(35 + value * 205),
                int(55 + value * 160),
                int(110 - value * 65),
            )
            x0 = left + column_index * cell_width
            y0 = top + row_index * cell_height
            draw.rectangle(
                (x0, y0, x0 + cell_width, y0 + cell_height),
                fill=color,
                outline="white",
            )
            text_color = "white" if value < 0.45 else "black"
            draw.text((x0 + 31, y0 + 20), f"{value:.2f}", fill=text_color)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def make_initial_view_sheet(
    rows: Sequence[Mapping[str, Any]],
    frame_dir: Path,
    output: Path,
    *,
    edge_id: str,
) -> None:
    candidates = [
        row
        for row in rows
        if str(row["edge_id"]) == edge_id and "frame_file" in row
    ]
    if not candidates:
        raise ValueError(f"no recorded frames found for contact edge {edge_id}")
    minimum_state = min(int(row["canonical_state_index"]) for row in candidates)
    candidates = [
        row for row in candidates if int(row["canonical_state_index"]) == minimum_state
    ]
    candidates.sort(key=lambda row: str(row["camera_pose"]))
    tile_width = 224
    tile_height = 252
    columns = 4
    rows_count = math.ceil(len(candidates) / columns)
    canvas = Image.new("RGB", (columns * tile_width, rows_count * tile_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(candidates):
        with np.load(frame_dir / str(row["frame_file"]), allow_pickle=False) as archive:
            frame = np.asarray(archive["frames"][0], dtype=np.uint8)
        agent = Image.fromarray(frame[:, :224], mode="RGB")
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        canvas.paste(agent, (x, y + 28))
        draw.rectangle((x, y, x + tile_width, y + 28), fill="black")
        draw.text((x + 6, y + 7), str(row["camera_pose"]), fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze camera viewpoint evaluations")
    parser.add_argument("--evaluation", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poses", default="all")
    parser.add_argument("--frame-dir", type=Path)
    parser.add_argument("--contact-edge", default="red-left")
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {args.output_dir}")
    payloads = [json.loads(path.read_text()) for path in args.evaluation]
    if any(payload.get("status") != "complete" for payload in payloads):
        raise ValueError("all camera evaluations must be complete")
    rows = [row for payload in payloads for row in payload["rows"]]
    available_poses = sorted({str(row["camera_pose"]) for row in rows})
    if args.poses != "all":
        requested_poses = [
            value.strip() for value in args.poses.split(",") if value.strip()
        ]
        unknown = sorted(set(requested_poses) - set(available_poses))
        if not requested_poses or unknown:
            raise ValueError(f"invalid requested camera poses: {unknown}")
        rows = [row for row in rows if str(row["camera_pose"]) in requested_poses]
    if not rows or not any(row["camera_pose"] == "baseline" for row in rows):
        raise ValueError("analysis requires baseline and at least one episode")
    keys = [
        (
            str(row["camera_pose"]),
            str(row["edge_id"]),
            int(row["canonical_state_index"]),
            int(row["execution_horizon"]),
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("camera evaluations contain duplicate episode keys")

    summaries = summarize_rows(rows)
    edge_summaries = summarize_by_edge(rows)
    ranking = rank_nonbaseline(summaries)
    paired = []
    for pose in ranking:
        for metric in ("success", "progress", "wrong_source_grasp"):
            paired.append(
                paired_bootstrap_delta(
                    rows,
                    candidate=pose,
                    baseline="baseline",
                    metric=metric,
                )
            )
    perturbed_rows = [row for row in rows if row["camera_pose"] != "baseline"]
    camera_qc = {
        "baseline_agent_mae_max": max(
            float(row["initial_agent_mae_from_baseline"])
            for row in rows
            if row["camera_pose"] == "baseline"
        ),
        "perturbed_agent_mae_min": min(
            float(row["initial_agent_mae_from_baseline"]) for row in perturbed_rows
        ),
        "wrist_mae_max": max(
            float(row["initial_wrist_mae_from_baseline"]) for row in rows
        ),
    }
    report = {
        "schema_version": 1,
        "evaluations": [str(path) for path in args.evaluation],
        "episode_count": len(rows),
        "state_indices": sorted({int(row["canonical_state_index"]) for row in rows}),
        "edge_ids": sorted({str(row["edge_id"]) for row in rows}),
        "camera_qc": camera_qc,
        "ranking_nonbaseline": ranking,
        "recommended_confirmation_poses": ranking[:2],
        "summaries": summaries,
        "edge_summaries": edge_summaries,
        "paired_group_bootstrap": paired,
    }
    staging = args.output_dir.parent / f".{args.output_dir.name}.staging-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        (staging / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        with (staging / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        with (staging / "edge_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(edge_summaries[0]))
            writer.writeheader()
            writer.writerows(edge_summaries)
        plot_curves(summaries, staging / "camera_sensitivity_curves.png")
        plot_metric_matrix(summaries, staging / "camera_metric_matrix.png")
        plot_edge_success_matrix(
            edge_summaries,
            staging / "edge_success_matrix.png",
        )
        if args.frame_dir is not None:
            make_initial_view_sheet(
                rows,
                args.frame_dir,
                staging / "initial_agent_views.jpg",
                edge_id=args.contact_edge,
            )
        staging.rename(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "recommended_confirmation_poses": ranking[:2],
                "camera_qc": camera_qc,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
