from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


OBJECT_SCOPES = ("source", "target", "task")
PIXEL_BOUNDARY = 64
AXIS_CAMERA_FIELDS = {
    "azimuth_deg": "camera_azimuth_deg",
    "elevation_deg": "camera_elevation_deg",
    "radius_scale": "camera_radius_scale",
}
TRAINING_SUPPORT = {
    "azimuth_deg": (-60.0, 60.0),
    "elevation_deg": (-25.0, 25.0),
    "radius_scale": (0.9, 1.25),
}
PHASE_COLORS = {
    "inside": "#2e8b57",
    "partial_under_10": "#9acd32",
    "partial_10_50": "#f0c84b",
    "severe_clipping": "#ed7d31",
    "center_out": "#c050a0",
    "below_support": "#3d85c6",
    "disappeared": "#222222",
}


def _finite_float(value: Any, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _extract_rows(payload: Mapping[str, Any], *, kind: str) -> list[Mapping[str, Any]]:
    if payload.get("status") not in (None, "complete"):
        raise ValueError(f"{kind} JSON must have status=complete")
    rows = payload.get("rows")
    if rows is None and kind == "policy":
        rows = payload.get("edge_summaries", payload.get("summaries"))
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{kind} JSON must contain a rows list")
    return rows


def _validate_visibility_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("visibility scan contains no rows")
    required = {
        "edge_id",
        "camera_pose",
        "sweep_axis",
        "sweep_value",
    }
    for label in ("source", "target"):
        required.update(
            {
                f"{label}_center_in_frame",
                f"{label}_fov_clipping_fraction",
                f"{label}_visible_pixels",
                f"{label}_visible_patch_support",
            }
        )
    for index, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"visibility row {index} is missing fields: {missing}")
        _finite_float(row["sweep_value"], field="sweep_value")
        for label in ("source", "target"):
            clipping = _finite_float(
                row[f"{label}_fov_clipping_fraction"],
                field=f"{label}_fov_clipping_fraction",
            )
            if not 0.0 <= clipping <= 1.0:
                raise ValueError(f"visibility row {index} has clipping outside [0, 1]")
            pixels = int(row[f"{label}_visible_pixels"])
            patches = int(row[f"{label}_visible_patch_support"])
            if pixels < 0 or patches < 0:
                raise ValueError(f"visibility row {index} has negative support")


def _neutral_value(axis: str, baseline_rows: Sequence[Mapping[str, Any]]) -> float:
    camera_field = AXIS_CAMERA_FIELDS.get(axis)
    if camera_field and all(camera_field in row for row in baseline_rows):
        values = {
            _finite_float(row[camera_field], field=camera_field)
            for row in baseline_rows
        }
        if len(values) != 1:
            raise ValueError(f"baseline has inconsistent {camera_field} values")
        return values.pop()
    return 0.0


def _scope_values(
    rows: Sequence[Mapping[str, Any]],
    scope: str,
) -> dict[str, Any]:
    labels = ("source", "target") if scope == "task" else (scope,)
    return {
        f"{scope}_fov_clipping_fraction_max": max(
            float(row[f"{label}_fov_clipping_fraction"])
            for row in rows
            for label in labels
        ),
        f"{scope}_center_in_frame_all": all(
            bool(row[f"{label}_center_in_frame"])
            for row in rows
            for label in labels
        ),
        f"{scope}_visible_pixels_min": min(
            int(row[f"{label}_visible_pixels"])
            for row in rows
            for label in labels
        ),
        f"{scope}_visible_patch_support_min": min(
            int(row[f"{label}_visible_patch_support"])
            for row in rows
            for label in labels
        ),
    }


def summarize_curve_points(
    visibility_rows: Sequence[Mapping[str, Any]],
    *,
    policy_rows: Sequence[Mapping[str, Any]] = (),
    policy_metric: str = "success",
) -> list[dict[str, Any]]:
    """Aggregate scan records into conservative edge/axis curve points."""
    _validate_visibility_rows(visibility_rows)
    edges = sorted({str(row["edge_id"]) for row in visibility_rows})
    axes = sorted(
        {
            str(row["sweep_axis"])
            for row in visibility_rows
            if str(row["sweep_axis"]) != "baseline"
        }
    )
    if not axes:
        raise ValueError("visibility scan contains no non-baseline sweep axes")

    points: list[dict[str, Any]] = []
    for edge_id in edges:
        edge_rows = [
            row for row in visibility_rows if str(row["edge_id"]) == edge_id
        ]
        baseline_rows = [
            row
            for row in edge_rows
            if str(row["sweep_axis"]) == "baseline"
            or str(row["camera_pose"]) == "baseline"
        ]
        if not baseline_rows:
            raise ValueError(f"edge {edge_id!r} has no baseline visibility rows")
        for axis in axes:
            axis_rows = [
                row for row in edge_rows if str(row["sweep_axis"]) == axis
            ]
            if not axis_rows:
                continue
            neutral = _neutral_value(axis, baseline_rows)
            grouped: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
            for row in axis_rows:
                grouped[_finite_float(row["sweep_value"], field="sweep_value")].append(
                    row
                )
            if not any(math.isclose(value, neutral, abs_tol=1e-10) for value in grouped):
                grouped[neutral].extend(baseline_rows)

            for sweep_value, rows in sorted(grouped.items()):
                point: dict[str, Any] = {
                    "edge_id": edge_id,
                    "sweep_axis": axis,
                    "sweep_value": sweep_value,
                    "neutral_value": neutral,
                    "camera_poses": sorted({str(row["camera_pose"]) for row in rows}),
                    "record_count": len(rows),
                    "state_count": len(
                        {
                            int(row["canonical_state_index"])
                            for row in rows
                            if "canonical_state_index" in row
                        }
                    ),
                }
                for scope in OBJECT_SCOPES:
                    point.update(_scope_values(rows, scope))
                points.append(point)

    _attach_policy_curve(points, policy_rows, policy_metric=policy_metric)
    return sorted(
        points,
        key=lambda point: (
            str(point["edge_id"]),
            str(point["sweep_axis"]),
            float(point["sweep_value"]),
        ),
    )


def _attach_policy_curve(
    points: Sequence[dict[str, Any]],
    policy_rows: Sequence[Mapping[str, Any]],
    *,
    policy_metric: str,
) -> None:
    if not policy_rows:
        return
    for index, row in enumerate(policy_rows):
        if "edge_id" not in row or policy_metric not in row:
            raise ValueError(
                f"policy row {index} must contain edge_id and {policy_metric!r}"
            )
        _finite_float(row[policy_metric], field=policy_metric)

    for point in points:
        values = []
        poses = set(point["camera_poses"])
        for row in policy_rows:
            if str(row["edge_id"]) != point["edge_id"]:
                continue
            pose_match = "camera_pose" in row and str(row["camera_pose"]) in poses
            direct_match = (
                str(row.get("sweep_axis", "")) == point["sweep_axis"]
                and "sweep_value" in row
                and math.isclose(
                    float(row["sweep_value"]),
                    float(point["sweep_value"]),
                    abs_tol=1e-10,
                )
            )
            if pose_match or direct_match:
                values.append(float(row[policy_metric]))
        if values:
            point["policy_metric"] = policy_metric
            point["policy_mean"] = sum(values) / len(values)
            point["policy_count"] = len(values)


def sampled_true_intervals(
    points: Sequence[Mapping[str, Any]],
    predicate: Callable[[Mapping[str, Any]], bool],
) -> list[dict[str, Any]]:
    """Return every true run in sorted sampled points, including clear brackets."""
    ordered = sorted(points, key=lambda point: float(point["sweep_value"]))
    matches = [bool(predicate(point)) for point in ordered]
    intervals: list[dict[str, Any]] = []
    index = 0
    while index < len(ordered):
        if not matches[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(ordered) and matches[index + 1]:
            index += 1
        end = index
        values = [float(point["sweep_value"]) for point in ordered[start : end + 1]]
        intervals.append(
            {
                "start_value": values[0],
                "end_value": values[-1],
                "sample_count": len(values),
                "sample_values": values,
                "lower_clear_value": (
                    float(ordered[start - 1]["sweep_value"]) if start > 0 else None
                ),
                "upper_clear_value": (
                    float(ordered[end + 1]["sweep_value"])
                    if end + 1 < len(ordered)
                    else None
                ),
            }
        )
        index += 1
    return intervals


def _boundary_specs(
    scope: str,
    *,
    minimum_patch_support: int,
) -> list[tuple[str, str, str, float, Callable[[Mapping[str, Any]], bool]]]:
    clipping = f"{scope}_fov_clipping_fraction_max"
    center = f"{scope}_center_in_frame_all"
    pixels = f"{scope}_visible_pixels_min"
    patches = f"{scope}_visible_patch_support_min"
    return [
        ("first_clipping", clipping, ">", 0.0, lambda point: float(point[clipping]) > 0.0),
        (
            "clipping_10_percent",
            clipping,
            ">=",
            0.10,
            lambda point: float(point[clipping]) >= 0.10,
        ),
        (
            "clipping_50_percent",
            clipping,
            ">=",
            0.50,
            lambda point: float(point[clipping]) >= 0.50,
        ),
        (
            "center_out_of_frame",
            center,
            "==",
            0.0,
            lambda point: not bool(point[center]),
        ),
        (
            "below_64_visible_pixels",
            pixels,
            "<",
            float(PIXEL_BOUNDARY),
            lambda point: int(point[pixels]) < PIXEL_BOUNDARY,
        ),
        (
            "below_patch_support",
            patches,
            "<",
            float(minimum_patch_support),
            lambda point: int(point[patches]) < minimum_patch_support,
        ),
        (
            "fully_disappeared",
            pixels,
            "==",
            0.0,
            lambda point: int(point[pixels]) == 0,
        ),
    ]


def build_boundaries(
    curve_points: Sequence[Mapping[str, Any]],
    *,
    minimum_patch_support: int,
) -> list[dict[str, Any]]:
    if minimum_patch_support <= 0:
        raise ValueError("minimum_patch_support must be positive")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for point in curve_points:
        grouped[(str(point["edge_id"]), str(point["sweep_axis"]))].append(point)

    boundaries = []
    for (edge_id, axis), points in sorted(grouped.items()):
        points.sort(key=lambda point: float(point["sweep_value"]))
        neutral_values = {float(point["neutral_value"]) for point in points}
        if len(neutral_values) != 1:
            raise ValueError(f"{edge_id}/{axis} has inconsistent neutral values")
        neutral = neutral_values.pop()
        for scope in OBJECT_SCOPES:
            for name, metric, comparison, threshold, predicate in _boundary_specs(
                scope,
                minimum_patch_support=minimum_patch_support,
            ):
                intervals = sampled_true_intervals(points, predicate)
                lower = [
                    float(point["sweep_value"])
                    for point in points
                    if float(point["sweep_value"]) < neutral and predicate(point)
                ]
                upper = [
                    float(point["sweep_value"])
                    for point in points
                    if float(point["sweep_value"]) > neutral and predicate(point)
                ]
                at_neutral = any(
                    math.isclose(
                        float(point["sweep_value"]), neutral, abs_tol=1e-10
                    )
                    and predicate(point)
                    for point in points
                )
                boundaries.append(
                    {
                        "edge_id": edge_id,
                        "sweep_axis": axis,
                        "object_scope": scope,
                        "boundary": name,
                        "metric": metric,
                        "comparison": comparison,
                        "threshold": threshold,
                        "neutral_value": neutral,
                        "first_reached_lower": max(lower) if lower else None,
                        "first_reached_upper": min(upper) if upper else None,
                        "reached_at_neutral": at_neutral,
                        "intervals": intervals,
                    }
                )
    return boundaries


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def write_boundary_csv(boundaries: Sequence[Mapping[str, Any]], output: Path) -> None:
    fields = [
        "edge_id",
        "sweep_axis",
        "object_scope",
        "boundary",
        "metric",
        "comparison",
        "threshold",
        "neutral_value",
        "first_reached_lower",
        "first_reached_upper",
        "reached_at_neutral",
        "interval_index",
        "interval_start",
        "interval_end",
        "interval_sample_count",
        "interval_sample_values",
        "lower_clear_value",
        "upper_clear_value",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for boundary in boundaries:
            intervals = list(boundary["intervals"])
            for interval_index, interval in enumerate(intervals or [None]):
                row = {key: boundary[key] for key in fields[:11]}
                row.update(
                    {
                        "interval_index": interval_index if interval is not None else "",
                        "interval_start": (
                            interval["start_value"] if interval is not None else ""
                        ),
                        "interval_end": (
                            interval["end_value"] if interval is not None else ""
                        ),
                        "interval_sample_count": (
                            interval["sample_count"] if interval is not None else 0
                        ),
                        "interval_sample_values": (
                            json.dumps(interval["sample_values"], separators=(",", ":"))
                            if interval is not None
                            else "[]"
                        ),
                        "lower_clear_value": (
                            interval["lower_clear_value"]
                            if interval is not None
                            else ""
                        ),
                        "upper_clear_value": (
                            interval["upper_clear_value"]
                            if interval is not None
                            else ""
                        ),
                    }
                )
                writer.writerow(row)


def write_curve_csv(points: Sequence[Mapping[str, Any]], output: Path) -> None:
    fields = sorted({key for point in points for key in point})
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in points:
            writer.writerow({key: _csv_value(point.get(key, "")) for key in fields})


def visibility_phase(
    point: Mapping[str, Any],
    *,
    minimum_patch_support: int,
) -> str:
    pixels = int(point["task_visible_pixels_min"])
    patches = int(point["task_visible_patch_support_min"])
    centers = bool(point["task_center_in_frame_all"])
    clipping = float(point["task_fov_clipping_fraction_max"])
    if pixels == 0:
        return "disappeared"
    if not centers:
        return "center_out"
    if pixels < PIXEL_BOUNDARY or patches < minimum_patch_support:
        return "below_support"
    if clipping >= 0.50:
        return "severe_clipping"
    if clipping >= 0.10:
        return "partial_10_50"
    if clipping > 0.0:
        return "partial_under_10"
    return "inside"


def plot_phase_map(
    points: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    minimum_patch_support: int,
) -> None:
    if minimum_patch_support <= 0:
        raise ValueError("minimum_patch_support must be positive")
    axes = sorted({str(point["sweep_axis"]) for point in points})
    edges = sorted({str(point["edge_id"]) for point in points})
    if not axes or not edges:
        raise ValueError("phase map requires non-empty axes and edges")

    width = 980
    legend_height = 72
    panel_header = 45
    row_height = 30
    panel_footer = 48
    panel_height = panel_header + len(edges) * row_height + panel_footer
    canvas = Image.new(
        "RGB",
        (width, legend_height + len(axes) * panel_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), "Worst-case task visibility across test snapshots", fill="black")
    legend_x = 18
    for phase, color in PHASE_COLORS.items():
        draw.rectangle((legend_x, 36, legend_x + 16, 50), fill=color)
        draw.text(
            (legend_x + 21, 36),
            phase.replace("_", " "),
            fill="#333333",
        )
        legend_x += 125

    plot_left = 170
    plot_right = width - 35
    for axis_index, axis in enumerate(axes):
        top = legend_height + axis_index * panel_height
        axis_points = [
            point for point in points if str(point["sweep_axis"]) == axis
        ]
        x_values = sorted({float(point["sweep_value"]) for point in axis_points})
        x_min, x_max = min(x_values), max(x_values)
        if math.isclose(x_min, x_max):
            x_min -= 1.0
            x_max += 1.0

        def pixel_x(value: float) -> int:
            return plot_left + round(
                (value - x_min) / (x_max - x_min) * (plot_right - plot_left)
            )

        support = TRAINING_SUPPORT.get(axis)
        if support is not None:
            support_left = pixel_x(max(x_min, support[0]))
            support_right = pixel_x(min(x_max, support[1]))
            draw.rectangle(
                (
                    support_left,
                    top + panel_header - 5,
                    support_right,
                    top + panel_header + len(edges) * row_height,
                ),
                fill="#eef7ee",
            )
            draw.text(
                (support_left + 4, top + 7),
                f"training support [{support[0]:g}, {support[1]:g}]",
                fill="#416641",
            )
        draw.text((18, top + 8), axis, fill="black")

        for edge_index, edge in enumerate(edges):
            row_top = top + panel_header + edge_index * row_height
            row_bottom = row_top + row_height - 5
            draw.text((18, row_top + 6), edge, fill="#333333")
            values = sorted(
                [
                    point
                    for point in axis_points
                    if str(point["edge_id"]) == edge
                ],
                key=lambda point: float(point["sweep_value"]),
            )
            if not values:
                continue
            samples = [float(point["sweep_value"]) for point in values]
            cell_edges = [x_min]
            cell_edges.extend(
                (left + right) / 2.0
                for left, right in zip(samples[:-1], samples[1:], strict=True)
            )
            cell_edges.append(x_max)
            for point, left_value, right_value in zip(
                values,
                cell_edges[:-1],
                cell_edges[1:],
                strict=True,
            ):
                phase = visibility_phase(
                    point,
                    minimum_patch_support=minimum_patch_support,
                )
                left_px = pixel_x(left_value)
                draw.rectangle(
                    (
                        left_px,
                        row_top,
                        max(pixel_x(right_value), left_px + 1),
                        row_bottom,
                    ),
                    fill=PHASE_COLORS[phase],
                )
            draw.rectangle(
                (plot_left, row_top, plot_right, row_bottom),
                outline="#666666",
                width=1,
            )

        axis_bottom = top + panel_header + len(edges) * row_height
        for tick in np.linspace(x_min, x_max, num=7):
            px = pixel_x(float(tick))
            draw.line((px, axis_bottom, px, axis_bottom + 5), fill="black")
            draw.text((px - 16, axis_bottom + 8), f"{tick:g}", fill="#555555")
        neutral = 1.0 if axis == "radius_scale" else 0.0
        if x_min <= neutral <= x_max:
            px = pixel_x(neutral)
            draw.line(
                (px, top + panel_header - 5, px, axis_bottom),
                fill="#ffffff",
                width=2,
            )
            draw.text(
                (px + 4, top + panel_header - 18),
                "canonical",
                fill="#444444",
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def plot_curves(
    points: Sequence[Mapping[str, Any]],
    boundaries: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for point in points:
        grouped[(str(point["edge_id"]), str(point["sweep_axis"]))].append(point)
    keys = sorted(grouped)
    columns = min(2, len(keys))
    panel_rows = math.ceil(len(keys) / columns)
    panel_width = 720
    panel_height = 390
    canvas = Image.new(
        "RGB",
        (panel_width * columns, panel_height * panel_rows),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    boundary_colors = {
        "first_clipping": "#7f7f7f",
        "clipping_10_percent": "#e69f00",
        "clipping_50_percent": "#d55e00",
        "center_out_of_frame": "#cc79a7",
        "below_64_visible_pixels": "#0072b2",
        "below_patch_support": "#009e73",
        "fully_disappeared": "#000000",
    }
    for panel_index, key in enumerate(keys):
        edge_id, axis = key
        values = sorted(grouped[key], key=lambda point: float(point["sweep_value"]))
        x = [float(point["sweep_value"]) for point in values]
        column = panel_index % columns
        row = panel_index // columns
        panel_left = column * panel_width
        panel_top = row * panel_height
        plot_left = panel_left + 64
        plot_right = panel_left + panel_width - 58
        plot_top = panel_top + 55
        plot_bottom = panel_top + 300
        plot_width = plot_right - plot_left
        plot_height = plot_bottom - plot_top
        x_min = min(x)
        x_max = max(x)
        if math.isclose(x_min, x_max):
            x_min -= 1.0
            x_max += 1.0

        def pixel_x(value: float) -> int:
            return plot_left + round((value - x_min) / (x_max - x_min) * plot_width)

        def pixel_y(value: float) -> int:
            return plot_bottom - round(max(0.0, min(1.0, value)) * plot_height)

        draw.text(
            (panel_left + 18, panel_top + 15),
            f"{edge_id} / {axis}",
            fill="black",
        )
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="black", width=2)
        draw.line(
            (plot_left, plot_bottom, plot_right, plot_bottom),
            fill="black",
            width=2,
        )
        for tick in (0.0, 0.1, 0.5, 1.0):
            y = pixel_y(tick)
            color = "#e0e0e0" if tick not in (0.1, 0.5) else "#c8c8c8"
            draw.line((plot_left, y, plot_right, y), fill=color, width=1)
            draw.text((plot_left - 40, y - 6), f"{tick:g}", fill="#555555")
        tick_divisions = min(6, max(len(x) - 1, 1))
        tick_indices = {
            round(index * (len(x) - 1) / tick_divisions)
            for index in range(tick_divisions + 1)
        }
        tick_indices.add(
            min(
                range(len(x)),
                key=lambda index: abs(
                    x[index] - float(values[0]["neutral_value"])
                ),
            )
        )
        for index in sorted(tick_indices):
            value = x[index]
            px = pixel_x(value)
            draw.line((px, plot_bottom, px, plot_bottom + 4), fill="black", width=1)
            draw.text((px - 14, plot_bottom + 7), f"{value:g}", fill="#555555")

        neutral_x = pixel_x(float(values[0]["neutral_value"]))
        for y in range(plot_top, plot_bottom, 8):
            draw.line(
                (neutral_x, y, neutral_x, min(y + 4, plot_bottom)),
                fill="#666666",
                width=1,
            )

        task_boundaries = [
            boundary
            for boundary in boundaries
            if boundary["edge_id"] == edge_id
            and boundary["sweep_axis"] == axis
            and boundary["object_scope"] == "task"
        ]
        for boundary in task_boundaries:
            for interval in boundary["intervals"]:
                name = str(boundary["boundary"])
                start_x = pixel_x(float(interval["start_value"]))
                end_x = pixel_x(float(interval["end_value"]))
                for px in {start_x, end_x}:
                    draw.line(
                        (px, plot_top, px, plot_bottom),
                        fill=boundary_colors[name],
                        width=1,
                    )

        series = [
            (
                "source clipping",
                "#d55e00",
                [
                    float(point["source_fov_clipping_fraction_max"])
                    for point in values
                ],
                "circle",
            ),
            (
                "target clipping",
                "#0072b2",
                [
                    float(point["target_fov_clipping_fraction_max"])
                    for point in values
                ],
                "square",
            ),
            (
                "pixel support / 64",
                "#56b4e9",
                [
                    min(
                        float(point["task_visible_pixels_min"]) / PIXEL_BOUNDARY,
                        1.0,
                    )
                    for point in values
                ],
                "circle",
            ),
        ]
        for _label, color, y_values, marker in series:
            coordinates = [
                (pixel_x(x_value), pixel_y(y_value))
                for x_value, y_value in zip(x, y_values, strict=True)
            ]
            if len(coordinates) > 1:
                draw.line(coordinates, fill=color, width=3)
            for px, py in coordinates:
                if marker == "square":
                    draw.rectangle((px - 3, py - 3, px + 3, py + 3), fill=color)
                else:
                    draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)

        policy_points = [
            (float(point["sweep_value"]), float(point["policy_mean"]))
            for point in values
            if "policy_mean" in point
        ]
        if policy_points:
            policy_values = [value for _, value in policy_points]
            policy_min = min(policy_values)
            policy_max = max(policy_values)
            rate_scale = all(0.0 <= value <= 1.0 for value in policy_values)
            if not rate_scale and math.isclose(policy_min, policy_max):
                policy_min -= 1.0
                policy_max += 1.0

            def policy_pixel_y(value: float) -> int:
                if rate_scale:
                    return pixel_y(value)
                return plot_bottom - round(
                    (value - policy_min) / (policy_max - policy_min) * plot_height
                )

            coordinates = [
                (pixel_x(x_value), policy_pixel_y(y_value))
                for x_value, y_value in policy_points
            ]
            if len(coordinates) > 1:
                draw.line(coordinates, fill="#000000", width=3)
            for px, py in coordinates:
                draw.polygon(
                    ((px, py - 5), (px + 5, py), (px, py + 5), (px - 5, py)),
                    fill="#000000",
                )
            metric = str(values[0].get("policy_metric", "metric"))
            draw.text(
                (plot_right - 135, panel_top + 32),
                f"policy {metric}",
                fill="#000000",
            )
            if not rate_scale:
                draw.text(
                    (plot_right + 5, plot_top - 5),
                    f"{policy_max:g}",
                    fill="#333333",
                )
                draw.text(
                    (plot_right + 5, plot_bottom - 7),
                    f"{policy_min:g}",
                    fill="#333333",
                )

        legend_y = panel_top + 326
        legend_x = panel_left + 68
        for legend_index, (label, color, _values, _marker) in enumerate(series):
            x0 = legend_x + legend_index * 170
            draw.line((x0, legend_y + 6, x0 + 22, legend_y + 6), fill=color, width=3)
            draw.text((x0 + 28, legend_y), label, fill="#333333")
        draw.text(
            (panel_left + panel_width // 2 - 25, panel_top + 365),
            axis,
            fill="#333333",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def analyze_payloads(
    visibility_payloads: Sequence[Mapping[str, Any]],
    *,
    policy_payloads: Sequence[Mapping[str, Any]] = (),
    minimum_patch_support: int = 1,
    policy_metric: str = "success",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    visibility_rows = [
        row
        for payload in visibility_payloads
        for row in _extract_rows(payload, kind="visibility")
    ]
    policy_rows = [
        row
        for payload in policy_payloads
        for row in _extract_rows(payload, kind="policy")
    ]
    points = summarize_curve_points(
        visibility_rows,
        policy_rows=policy_rows,
        policy_metric=policy_metric,
    )
    boundaries = build_boundaries(
        points,
        minimum_patch_support=minimum_patch_support,
    )
    return points, boundaries


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze dense LIBERO camera FOV boundaries and policy sensitivity"
    )
    parser.add_argument(
        "--fov-json",
        "--visibility",
        dest="fov_json",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--policy-json",
        "--policy-evaluation",
        dest="policy_json",
        type=Path,
        nargs="+",
        default=[],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-patch-support", type=int, default=1)
    parser.add_argument("--policy-metric", default="success")
    return parser.parse_args(args)


def main(args: Iterable[str] | None = None) -> None:
    parsed = parse_args(args)
    if parsed.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {parsed.output_dir}")
    if parsed.minimum_patch_support <= 0:
        raise ValueError("minimum-patch-support must be positive")
    visibility_payloads = [json.loads(path.read_text()) for path in parsed.fov_json]
    policy_payloads = [json.loads(path.read_text()) for path in parsed.policy_json]
    points, boundaries = analyze_payloads(
        visibility_payloads,
        policy_payloads=policy_payloads,
        minimum_patch_support=parsed.minimum_patch_support,
        policy_metric=parsed.policy_metric,
    )
    report = {
        "schema_version": 1,
        "study": "libero_camera_viewpoint_fov_boundaries",
        "fov_json": [str(path) for path in parsed.fov_json],
        "policy_json": [str(path) for path in parsed.policy_json],
        "aggregation": {
            "states": "worst_case",
            "task_objects": "worst_of_source_and_target",
        },
        "minimum_visible_pixels": PIXEL_BOUNDARY,
        "minimum_patch_support": parsed.minimum_patch_support,
        "policy_metric": parsed.policy_metric if parsed.policy_json else None,
        "curve_points": points,
        "boundaries": boundaries,
    }

    staging = (
        parsed.output_dir.parent / f".{parsed.output_dir.name}.staging-{os.getpid()}"
    )
    parsed.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        (staging / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        write_boundary_csv(boundaries, staging / "summary.csv")
        write_curve_csv(points, staging / "curve_points.csv")
        plot_curves(points, boundaries, staging / "camera_viewpoint_curves.png")
        plot_phase_map(
            points,
            staging / "camera_fov_phase_map.png",
            minimum_patch_support=parsed.minimum_patch_support,
        )
        edge_plot_dir = staging / "curves_by_edge"
        edge_plot_dir.mkdir()
        for edge_id in sorted({str(point["edge_id"]) for point in points}):
            plot_curves(
                [
                    point
                    for point in points
                    if str(point["edge_id"]) == edge_id
                ],
                [
                    boundary
                    for boundary in boundaries
                    if str(boundary["edge_id"]) == edge_id
                ],
                edge_plot_dir / f"{edge_id}.png",
            )
        staging.rename(parsed.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "output_dir": str(parsed.output_dir),
                "curve_point_count": len(points),
                "boundary_count": len(boundaries),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
