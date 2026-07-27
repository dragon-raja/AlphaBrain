from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


BOOTSTRAP_SEED = 20260727
EPISODE_KEY_FIELDS = (
    "edge_id",
    "canonical_state_index",
    "execution_horizon",
    "camera_pose",
)
METRICS = (
    "success",
    "progress",
    "source_selection_success",
    "lift_success",
    "transport_success",
    "target_placement_success",
)
SUBGOAL_METRICS = METRICS[2:]
TRAINING_SUPPORT = {
    "azimuth": (-60.0, 60.0),
    "azimuth_deg": (-60.0, 60.0),
    "elevation": (-25.0, 25.0),
    "elevation_deg": (-25.0, 25.0),
    "radius": (0.9, 1.25),
    "radius_scale": (0.9, 1.25),
}
AXIS_CAMERA_FIELDS = {
    "azimuth": "camera_azimuth_deg",
    "azimuth_deg": "camera_azimuth_deg",
    "elevation": "camera_elevation_deg",
    "elevation_deg": "camera_elevation_deg",
    "radius": "camera_radius_scale",
    "radius_scale": "camera_radius_scale",
}
METHOD_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def episode_key(row: Mapping[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(row["edge_id"]),
        int(row["canonical_state_index"]),
        int(row["execution_horizon"]),
        str(row["camera_pose"]),
    )


def fov_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(row["edge_id"]),
        int(row["canonical_state_index"]),
        str(row["camera_pose"]),
    )


def _finite_float(value: Any, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _evaluation_rows(payload: Mapping[str, Any], *, method: str) -> list[Mapping[str, Any]]:
    if payload.get("status") != "complete":
        raise ValueError(f"evaluation {method!r} must have status=complete")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"evaluation {method!r} must contain non-empty rows")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"evaluation {method!r} contains a non-object row")
    return rows


def validate_paired_evaluations(
    evaluations: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[tuple[str, int, int, str], Mapping[str, Any]]]:
    """Validate exact episode pairing and return rows indexed by episode key."""
    if len(evaluations) < 2:
        raise ValueError("comparison requires at least two evaluation methods")
    indexed: dict[str, dict[tuple[str, int, int, str], Mapping[str, Any]]] = {}
    for method, payload in evaluations.items():
        rows = _evaluation_rows(payload, method=method)
        method_rows: dict[tuple[str, int, int, str], Mapping[str, Any]] = {}
        for index, row in enumerate(rows):
            missing = sorted(
                (
                    set(EPISODE_KEY_FIELDS)
                    | {"action_supervised", *METRICS}
                )
                - set(row)
            )
            if missing:
                raise ValueError(
                    f"evaluation {method!r} row {index} is missing fields: {missing}"
                )
            key = episode_key(row)
            if key in method_rows:
                raise ValueError(f"evaluation {method!r} has duplicate episode key {key}")
            for metric in METRICS:
                _finite_float(row[metric], field=metric)
            method_rows[key] = row
        indexed[method] = method_rows

    reference_method = next(iter(indexed))
    reference_keys = set(indexed[reference_method])
    for method, rows in indexed.items():
        keys = set(rows)
        if keys != reference_keys:
            missing = sorted(reference_keys - keys)
            extra = sorted(keys - reference_keys)
            raise ValueError(
                f"evaluation {method!r} episode keys differ from "
                f"{reference_method!r}: missing={missing[:5]}, extra={extra[:5]}"
            )
        for key in reference_keys:
            expected = bool(indexed[reference_method][key]["action_supervised"])
            if bool(rows[key]["action_supervised"]) != expected:
                raise ValueError(
                    f"action_supervised differs across methods for episode {key}"
                )
    return indexed


def index_fov_rows(
    payloads: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    required = {
        "edge_id",
        "canonical_state_index",
        "camera_pose",
        "sweep_axis",
        "sweep_value",
    }
    for label in ("source", "target"):
        required.update(
            {
                f"{label}_center_in_frame",
                f"{label}_fov_clipping_fraction",
                f"{label}_projected_pixels_in_frame",
                f"{label}_projected_patch_support",
                f"{label}_external_occlusion_fraction",
                f"{label}_visible_pixels",
                f"{label}_visible_patch_support",
            }
        )
    for payload_index, payload in enumerate(payloads):
        if payload.get("status") != "complete":
            raise ValueError(f"FOV JSON {payload_index} must have status=complete")
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"FOV JSON {payload_index} must contain a rows list")
        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"FOV JSON {payload_index} contains a non-object row")
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(
                    f"FOV row {payload_index}:{row_index} is missing fields: {missing}"
                )
            key = fov_key(row)
            if key in indexed:
                comparison_fields = required | {
                    "action_supervised",
                    *AXIS_CAMERA_FIELDS.values(),
                }
                conflict = any(
                    field in row
                    and field in indexed[key]
                    and row[field] != indexed[key][field]
                    for field in comparison_fields
                )
                if conflict:
                    raise ValueError(f"conflicting duplicate FOV key {key}")
                continue
            _finite_float(row["sweep_value"], field="sweep_value")
            for label in ("source", "target"):
                clipping = _finite_float(
                    row[f"{label}_fov_clipping_fraction"],
                    field=f"{label}_fov_clipping_fraction",
                )
                projected_pixels = int(
                    row[f"{label}_projected_pixels_in_frame"]
                )
                projected_patches = int(
                    row[f"{label}_projected_patch_support"]
                )
                pixels = int(row[f"{label}_visible_pixels"])
                patches = int(row[f"{label}_visible_patch_support"])
                occlusion = _finite_float(
                    row[f"{label}_external_occlusion_fraction"],
                    field=f"{label}_external_occlusion_fraction",
                )
                if not 0.0 <= clipping <= 1.0:
                    raise ValueError(f"FOV key {key} has clipping outside [0, 1]")
                if (
                    projected_pixels < 0
                    or projected_patches < 0
                    or pixels < 0
                    or patches < 0
                ):
                    raise ValueError(f"FOV key {key} has negative support")
                if not 0.0 <= occlusion <= 1.0:
                    raise ValueError(f"FOV key {key} has occlusion outside [0, 1]")
            indexed[key] = row
    if not indexed:
        raise ValueError("FOV inputs contain no rows")
    return indexed


def visibility_stratum(
    *,
    task_min_projected_pixels_in_frame: int,
    task_min_visible_pixels: int,
    task_min_visible_patch_support: int,
    task_centers_in_frame: bool,
    task_max_fov_clipping_fraction: float,
    minimum_patch_support: int = 4,
) -> str:
    """Classify one FOV sample with a deterministic most-severe-first precedence."""
    if minimum_patch_support <= 0:
        raise ValueError("minimum_patch_support must be positive")
    if task_min_projected_pixels_in_frame == 0:
        return "geometrically_out_of_view"
    if task_min_visible_pixels == 0:
        return "fully_occluded"
    if not task_centers_in_frame:
        return "center_out"
    if (
        task_min_visible_pixels < 64
        or task_min_visible_patch_support < minimum_patch_support
    ):
        return "below_support"
    if task_max_fov_clipping_fraction >= 0.5:
        return "severe_clipping"
    return "fully_supported"


def join_episode_rows(
    indexed_evaluations: Mapping[
        str, Mapping[tuple[str, int, int, str], Mapping[str, Any]]
    ],
    fov_rows: Mapping[tuple[str, int, str], Mapping[str, Any]],
    *,
    minimum_patch_support: int = 4,
) -> list[dict[str, Any]]:
    joined = []
    for method, rows in indexed_evaluations.items():
        for key, evaluation_row in sorted(rows.items()):
            edge_id, state_index, execution_horizon, camera_pose = key
            geometry_key = (edge_id, state_index, camera_pose)
            if geometry_key not in fov_rows:
                raise ValueError(
                    f"no FOV row for evaluation episode key {key}; "
                    "FOV matching ignores execution_horizon"
                )
            fov = fov_rows[geometry_key]
            if (
                "action_supervised" in fov
                and bool(fov["action_supervised"])
                != bool(evaluation_row["action_supervised"])
            ):
                raise ValueError(f"FOV action_supervised differs for key {geometry_key}")

            source_clip = float(fov["source_fov_clipping_fraction"])
            target_clip = float(fov["target_fov_clipping_fraction"])
            source_pixels = int(fov["source_visible_pixels"])
            target_pixels = int(fov["target_visible_pixels"])
            source_patches = int(fov["source_visible_patch_support"])
            target_patches = int(fov["target_visible_patch_support"])
            source_projected_pixels = int(
                fov["source_projected_pixels_in_frame"]
            )
            target_projected_pixels = int(
                fov["target_projected_pixels_in_frame"]
            )
            source_projected_patches = int(
                fov["source_projected_patch_support"]
            )
            target_projected_patches = int(
                fov["target_projected_patch_support"]
            )
            source_occlusion = float(fov["source_external_occlusion_fraction"])
            target_occlusion = float(fov["target_external_occlusion_fraction"])
            centers_in_frame = bool(fov["source_center_in_frame"]) and bool(
                fov["target_center_in_frame"]
            )
            task_clip = max(source_clip, target_clip)
            task_projected_pixels = min(
                source_projected_pixels,
                target_projected_pixels,
            )
            task_projected_patches = min(
                source_projected_patches,
                target_projected_patches,
            )
            task_pixels = min(source_pixels, target_pixels)
            task_patches = min(source_patches, target_patches)
            split = (
                "observed"
                if bool(evaluation_row["action_supervised"])
                else "withheld"
            )
            row = {
                "method": method,
                "data_split": split,
                "edge_id": edge_id,
                "canonical_state_index": state_index,
                "execution_horizon": execution_horizon,
                "camera_pose": camera_pose,
                "action_supervised": bool(evaluation_row["action_supervised"]),
                "sweep_axis": str(fov["sweep_axis"]),
                "sweep_value": float(fov["sweep_value"]),
                "source_fov_clipping_fraction": source_clip,
                "target_fov_clipping_fraction": target_clip,
                "source_projected_pixels_in_frame": source_projected_pixels,
                "target_projected_pixels_in_frame": target_projected_pixels,
                "source_projected_patch_support": source_projected_patches,
                "target_projected_patch_support": target_projected_patches,
                "source_external_occlusion_fraction": source_occlusion,
                "target_external_occlusion_fraction": target_occlusion,
                "source_visible_pixels": source_pixels,
                "target_visible_pixels": target_pixels,
                "source_visible_patch_support": source_patches,
                "target_visible_patch_support": target_patches,
                "source_center_in_frame": bool(fov["source_center_in_frame"]),
                "target_center_in_frame": bool(fov["target_center_in_frame"]),
                "task_max_fov_clipping_fraction": task_clip,
                "task_min_projected_pixels_in_frame": task_projected_pixels,
                "task_min_projected_patch_support": task_projected_patches,
                "task_max_external_occlusion_fraction": max(
                    source_occlusion,
                    target_occlusion,
                ),
                "task_min_visible_pixels": task_pixels,
                "task_min_visible_patch_support": task_patches,
                "task_centers_in_frame": centers_in_frame,
                "visibility_stratum": visibility_stratum(
                    task_min_projected_pixels_in_frame=task_projected_pixels,
                    task_min_visible_pixels=task_pixels,
                    task_min_visible_patch_support=task_patches,
                    task_centers_in_frame=centers_in_frame,
                    task_max_fov_clipping_fraction=task_clip,
                    minimum_patch_support=minimum_patch_support,
                ),
            }
            for field in AXIS_CAMERA_FIELDS.values():
                if field in fov:
                    row[field] = float(fov[field])
                elif field in evaluation_row:
                    row[field] = float(evaluation_row[field])
            for metric in METRICS:
                row[metric] = float(evaluation_row[metric])
            joined.append(row)
    return joined


def _state_means(
    rows: Iterable[Mapping[str, Any]],
    metric: str,
) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["canonical_state_index"])].append(float(row[metric]))
    return {
        state: float(np.mean(values))
        for state, values in grouped.items()
    }


def paired_state_bootstrap(
    reference: Mapping[int, float],
    method: Mapping[int, float],
    *,
    resamples: int = 10_000,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not reference or set(reference) != set(method):
        raise ValueError("paired bootstrap requires identical non-empty state groups")
    states = sorted(reference)
    reference_values = np.asarray(
        [float(reference[state]) for state in states],
        dtype=np.float64,
    )
    method_values = np.asarray(
        [float(method[state]) for state in states],
        dtype=np.float64,
    )
    differences = method_values - reference_values
    if len(states) == 1:
        low = high = float(differences[0])
    else:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(states), size=(resamples, len(states)))
        draws = differences[indices].mean(axis=1)
        low, high = np.quantile(draws, [0.025, 0.975]).tolist()
    return {
        "delta": float(np.mean(differences)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "paired_state_count": len(states),
    }


AGGREGATE_DIMENSIONS = (
    "data_split",
    "camera_pose",
    "sweep_axis",
    "sweep_value",
    "visibility_stratum",
)


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str],
    reference: str,
    bootstrap_resamples: int,
) -> list[dict[str, Any]]:
    if reference not in methods:
        raise ValueError(f"reference method {reference!r} is not present")
    grouped: dict[
        tuple[str, str, float, str, str],
        dict[str, list[Mapping[str, Any]]],
    ] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        group_key = tuple(row[field] for field in AGGREGATE_DIMENSIONS)
        grouped[group_key][str(row["method"])].append(row)

    aggregates = []
    for group_key, method_rows in sorted(grouped.items()):
        if set(method_rows) != set(methods):
            raise ValueError(f"aggregate group {group_key} is not method-paired")
        reference_rows = method_rows[reference]
        for method in methods:
            values = method_rows[method]
            result: dict[str, Any] = {
                field: value
                for field, value in zip(
                    AGGREGATE_DIMENSIONS,
                    group_key,
                    strict=True,
                )
            }
            result.update(
                {
                    "method": method,
                    "reference_method": reference,
                    "episode_count": len(values),
                    "state_count": len(
                        {int(row["canonical_state_index"]) for row in values}
                    ),
                    "edge_count": len({str(row["edge_id"]) for row in values}),
                }
            )
            for metric in METRICS:
                reference_mean = float(
                    np.mean([float(row[metric]) for row in reference_rows])
                )
                method_mean = float(np.mean([float(row[metric]) for row in values]))
                paired = paired_state_bootstrap(
                    _state_means(reference_rows, metric),
                    _state_means(values, metric),
                    resamples=bootstrap_resamples,
                    seed=BOOTSTRAP_SEED,
                )
                result[f"{metric}_mean"] = method_mean
                result[f"{metric}_reference_mean"] = reference_mean
                result[f"{metric}_delta"] = paired["delta"]
                result[f"{metric}_ci95_low"] = paired["ci95_low"]
                result[f"{metric}_ci95_high"] = paired["ci95_high"]
            aggregates.append(result)
    return aggregates


def _neutral_axis_value(axis: str, rows: Sequence[Mapping[str, Any]]) -> float:
    field = AXIS_CAMERA_FIELDS.get(axis)
    baseline_rows = [row for row in rows if str(row["camera_pose"]) == "baseline"]
    if field and baseline_rows and all(field in row for row in baseline_rows):
        values = {float(row[field]) for row in baseline_rows}
        if len(values) == 1:
            return values.pop()
    return 1.0 if axis in ("radius", "radius_scale") else 0.0


def success_curve_rows(
    episode_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    axes = sorted(
        {
            str(row["sweep_axis"])
            for row in episode_rows
            if str(row["sweep_axis"]) != "baseline"
        }
    )
    splits = sorted({str(row["data_split"]) for row in episode_rows})
    methods = list(dict.fromkeys(str(row["method"]) for row in episode_rows))
    curves = []
    for split in splits:
        split_rows = [
            row for row in episode_rows if str(row["data_split"]) == split
        ]
        for axis in axes:
            neutral = _neutral_axis_value(axis, split_rows)
            for method in methods:
                method_rows = [
                    row
                    for row in split_rows
                    if str(row["method"]) == method
                    and (
                        str(row["sweep_axis"]) == axis
                        or str(row["camera_pose"]) == "baseline"
                    )
                ]
                grouped: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
                for row in method_rows:
                    value = (
                        neutral
                        if str(row["camera_pose"]) == "baseline"
                        else float(row["sweep_value"])
                    )
                    grouped[value].append(row)
                for value, values in sorted(grouped.items()):
                    curves.append(
                        {
                            "data_split": split,
                            "sweep_axis": axis,
                            "sweep_value": value,
                            "method": method,
                            "success": float(
                                np.mean([float(row["success"]) for row in values])
                            ),
                            "episode_count": len(values),
                        }
                    )
    return curves


def _tick_values(low: float, high: float, *, maximum: int = 5) -> list[float]:
    if math.isclose(low, high):
        return [low]
    return np.linspace(low, high, num=maximum).tolist()


def plot_success_curves(
    curves: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    keys = sorted(
        {
            (str(row["sweep_axis"]), str(row["data_split"]))
            for row in curves
        }
    )
    if not keys:
        raise ValueError("no non-baseline axes are available for plotting")
    splits = [value for value in ("observed", "withheld") if any(k[1] == value for k in keys)]
    axes = sorted({key[0] for key in keys})
    methods = list(dict.fromkeys(str(row["method"]) for row in curves))
    columns = len(splits)
    panel_width = 690
    legend_columns = min(3, len(methods))
    legend_rows = math.ceil(len(methods) / legend_columns)
    panel_height = 370 + max(0, legend_rows - 1) * 22
    canvas = Image.new(
        "RGB",
        (panel_width * columns, panel_height * len(axes)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    colors = (
        "#0072b2",
        "#d55e00",
        "#009e73",
        "#cc79a7",
        "#e69f00",
        "#56b4e9",
        "#000000",
    )
    method_colors = {
        method: colors[index % len(colors)]
        for index, method in enumerate(methods)
    }

    for row_index, axis in enumerate(axes):
        for column_index, split in enumerate(splits):
            panel_rows = [
                row
                for row in curves
                if str(row["sweep_axis"]) == axis
                and str(row["data_split"]) == split
            ]
            left = column_index * panel_width
            top = row_index * panel_height
            plot_left = left + 62
            plot_right = left + panel_width - 28
            plot_top = top + 48
            plot_bottom = top + 286
            draw.text((left + 16, top + 14), f"{split} / {axis}", fill="black")
            if not panel_rows:
                draw.text(
                    (left + panel_width // 2 - 42, top + panel_height // 2),
                    "no episodes",
                    fill="#666666",
                )
                continue
            x_values = [float(row["sweep_value"]) for row in panel_rows]
            support = TRAINING_SUPPORT.get(axis)
            x_min = min(x_values)
            x_max = max(x_values)
            if support is not None:
                x_min = min(x_min, support[0])
                x_max = max(x_max, support[1])
            if math.isclose(x_min, x_max):
                x_min -= 1.0
                x_max += 1.0

            def pixel_x(value: float) -> int:
                return plot_left + round(
                    (value - x_min) / (x_max - x_min) * (plot_right - plot_left)
                )

            def pixel_y(value: float) -> int:
                return plot_bottom - round(value * (plot_bottom - plot_top))

            if support is not None:
                support_left = pixel_x(support[0])
                support_right = pixel_x(support[1])
                draw.rectangle(
                    (support_left, plot_top, support_right, plot_bottom),
                    fill="#e8f3e8",
                )
                draw.text(
                    (support_left + 5, plot_top + 5),
                    f"training support [{support[0]:g}, {support[1]:g}]",
                    fill="#416641",
                )
                draw.line(
                    (support_left, plot_top, support_left, plot_bottom),
                    fill="#6b8e6b",
                    width=1,
                )
                draw.line(
                    (support_right, plot_top, support_right, plot_bottom),
                    fill="#6b8e6b",
                    width=1,
                )
            draw.line(
                (plot_left, plot_top, plot_left, plot_bottom),
                fill="black",
                width=2,
            )
            draw.line(
                (plot_left, plot_bottom, plot_right, plot_bottom),
                fill="black",
                width=2,
            )
            for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
                y = pixel_y(tick)
                draw.line((plot_left, y, plot_right, y), fill="#d8d8d8", width=1)
                draw.text((plot_left - 38, y - 6), f"{tick:g}", fill="#555555")
            for tick in _tick_values(x_min, x_max):
                x = pixel_x(tick)
                draw.line((x, plot_bottom, x, plot_bottom + 4), fill="black", width=1)
                draw.text((x - 17, plot_bottom + 7), f"{tick:g}", fill="#555555")

            for method in methods:
                method_rows = sorted(
                    [
                        row
                        for row in panel_rows
                        if str(row["method"]) == method
                    ],
                    key=lambda row: float(row["sweep_value"]),
                )
                coordinates = [
                    (pixel_x(float(row["sweep_value"])), pixel_y(float(row["success"])))
                    for row in method_rows
                ]
                if len(coordinates) > 1:
                    draw.line(coordinates, fill=method_colors[method], width=3)
                for x, y in coordinates:
                    draw.ellipse(
                        (x - 4, y - 4, x + 4, y + 4),
                        fill=method_colors[method],
                        outline="white",
                    )

            legend_y = top + 321
            for index, method in enumerate(methods):
                legend_column = index % legend_columns
                legend_row = index // legend_columns
                legend_x = left + 62 + legend_column * 195
                method_legend_y = legend_y + legend_row * 22
                draw.line(
                    (
                        legend_x,
                        method_legend_y + 6,
                        legend_x + 22,
                        method_legend_y + 6,
                    ),
                    fill=method_colors[method],
                    width=3,
                )
                draw.text(
                    (legend_x + 28, method_legend_y),
                    method,
                    fill="#333333",
                )
            draw.text(
                (
                    left + panel_width // 2 - 30,
                    top + 349 + max(0, legend_rows - 1) * 22,
                ),
                axis,
                fill="#333333",
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def write_csv_rows(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {output.name}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def compare_evaluations(
    evaluations: Mapping[str, Mapping[str, Any]],
    fov_payloads: Sequence[Mapping[str, Any]],
    *,
    reference: str,
    minimum_patch_support: int = 4,
    bootstrap_resamples: int = 10_000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if reference not in evaluations:
        raise ValueError(f"reference method {reference!r} is not present")
    if minimum_patch_support <= 0:
        raise ValueError("minimum_patch_support must be positive")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    indexed = validate_paired_evaluations(evaluations)
    fov = index_fov_rows(fov_payloads)
    episode_rows = join_episode_rows(
        indexed,
        fov,
        minimum_patch_support=minimum_patch_support,
    )
    methods = list(evaluations)
    aggregates = aggregate_rows(
        episode_rows,
        methods=methods,
        reference=reference,
        bootstrap_resamples=bootstrap_resamples,
    )
    curves = success_curve_rows(episode_rows)
    return episode_rows, aggregates, curves


def parse_evaluation_specs(specs: Sequence[str]) -> dict[str, Path]:
    evaluations: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"evaluation must use METHOD=JSON syntax: {spec!r}")
        method, path_text = spec.split("=", 1)
        if not METHOD_PATTERN.fullmatch(method) or not path_text:
            raise ValueError(f"invalid evaluation specification: {spec!r}")
        if method in evaluations:
            raise ValueError(f"duplicate evaluation method: {method!r}")
        evaluations[method] = Path(path_text)
    if len(evaluations) < 2:
        raise ValueError("at least two --evaluation arguments are required")
    return evaluations


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare paired KYC camera evaluations with FOV geometry"
    )
    parser.add_argument(
        "--evaluation",
        action="append",
        required=True,
        metavar="METHOD=JSON",
    )
    parser.add_argument("--fov-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--minimum-patch-support", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser.parse_args(args)


def main(args: Iterable[str] | None = None) -> None:
    parsed = parse_args(args)
    evaluation_paths = parse_evaluation_specs(parsed.evaluation)
    if parsed.reference not in evaluation_paths:
        raise ValueError(f"reference method {parsed.reference!r} is not present")
    if parsed.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite comparison: {parsed.output_dir}")
    evaluations = {
        method: json.loads(path.read_text())
        for method, path in evaluation_paths.items()
    }
    fov_payloads = [json.loads(path.read_text()) for path in parsed.fov_json]
    episode_rows, aggregates, curves = compare_evaluations(
        evaluations,
        fov_payloads,
        reference=parsed.reference,
        minimum_patch_support=parsed.minimum_patch_support,
        bootstrap_resamples=parsed.bootstrap_resamples,
    )
    split_counts = {
        split: {
            method: sum(
                row["data_split"] == split and row["method"] == method
                for row in episode_rows
            )
            for method in evaluation_paths
        }
        for split in ("observed", "withheld")
    }
    report = {
        "schema_version": 2,
        "study": "kyc_camera_evaluation_comparison",
        "evaluations": {
            method: str(path)
            for method, path in evaluation_paths.items()
        },
        "fov_json": [str(path) for path in parsed.fov_json],
        "methods": list(evaluation_paths),
        "reference_method": parsed.reference,
        "episode_key": list(EPISODE_KEY_FIELDS),
        "fov_join_key": [
            "edge_id",
            "canonical_state_index",
            "camera_pose",
        ],
        "paired_unit": "canonical_state_index",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": parsed.bootstrap_resamples,
        "minimum_visible_pixels": 64,
        "minimum_patch_support": parsed.minimum_patch_support,
        "metrics": list(METRICS),
        "subgoal_metrics": list(SUBGOAL_METRICS),
        "visibility_strata_precedence": [
            "geometrically_out_of_view",
            "fully_occluded",
            "center_out",
            "below_support",
            "severe_clipping",
            "fully_supported",
        ],
        "training_support": {
            "azimuth_deg": [-60.0, 60.0],
            "elevation_deg": [-25.0, 25.0],
            "radius_scale": [0.9, 1.25],
        },
        "episode_counts": split_counts,
        "aggregates": aggregates,
        "success_curves": curves,
    }

    parsed.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = (
        parsed.output_dir.parent / f".{parsed.output_dir.name}.staging-{os.getpid()}"
    )
    staging.mkdir(parents=True, exist_ok=False)
    try:
        (staging / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        write_csv_rows(episode_rows, staging / "episode_rows.csv")
        write_csv_rows(aggregates, staging / "aggregate.csv")
        plot_success_curves(curves, staging / "camera_success_curves.png")
        staging.rename(parsed.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "output_dir": str(parsed.output_dir),
                "methods": list(evaluation_paths),
                "episode_row_count": len(episode_rows),
                "aggregate_count": len(aggregates),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
