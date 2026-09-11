from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


METHODS = ("poseaug_rgb_fla", "poseaug_control_fla", "kyc_fla")
METHOD_LABELS = {
    "poseaug_rgb_fla": "PoseAug-RGB + FLA",
    "poseaug_control_fla": "PoseAug-Control + FLA",
    "kyc_fla": "KYC + FLA",
}
METHOD_COLORS = {
    "poseaug_rgb_fla": "#4C956C",
    "poseaug_control_fla": "#355C7D",
    "kyc_fla": "#E76F51",
}
STRATA = ("inside_training_support", "objects_visible", "fully_visible")
STRATUM_LABELS = {
    "inside_training_support": "All support\nposes",
    "objects_visible": "Objects\nvisible",
    "fully_visible": "Objects fully visible\n+ centers in frame",
}
GRID_COLOR = "#D9DEE3"
DEFAULT_CAMERA = {
    "camera_azimuth_deg": 0.0,
    "camera_elevation_deg": 0.0,
    "camera_radius_scale": 1.0,
}


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text())


def _parse_evaluation(value: str) -> tuple[str, Path]:
    method, separator, path = value.partition("=")
    if not separator or method not in METHODS or not path:
        raise ValueError(f"evaluation must be METHOD=PATH with METHOD in {METHODS}")
    return method, Path(path)


def _atomic_save(fig: Any, output: Path, *, dpi: int = 180) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite figure: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    fig.savefig(temporary, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    os.replace(temporary, output)


def _validate_summary(summary: Mapping[str, Any]) -> None:
    if summary.get("status") != "complete":
        raise ValueError("visual alignment summary is incomplete")
    if summary.get("study") != "kyc_pi05_visual_alignment_screen":
        raise ValueError("unexpected visual alignment study")


def render_screen_summary(*, summary: Mapping[str, Any], output: Path) -> None:
    _validate_summary(summary)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    success_axis, effect_axis, ray_axis = axes
    fig.subplots_adjust(left=0.055, right=0.985, top=0.76, bottom=0.20, wspace=0.38)
    fig.suptitle(
        "Pi0.5 visual-alignment screen: matched-data comparison",
        fontsize=15,
        fontweight="semibold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.925,
        "Seed 41, 2,000 updates, randomized scene cues, wrist camera on, K=3",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )

    x = np.arange(len(STRATA), dtype=np.float64)
    width = 0.24
    for method_index, method in enumerate(METHODS):
        values = [
            100.0 * float(summary["strata"][stratum]["methods"][method]["success"])
            for stratum in STRATA
        ]
        offsets = x + (method_index - 1) * width
        bars = success_axis.bar(
            offsets,
            values,
            width=width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        for bar, value in zip(bars, values):
            success_axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.8,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    success_axis.set_xticks(x, [STRATUM_LABELS[stratum] for stratum in STRATA])
    success_axis.set_ylabel("Closed-loop task success (%)")
    success_axis.set_title("(a) Absolute task success")
    success_axis.set_ylim(
        0.0,
        max(
            25.0,
            112.0
            * max(
                float(summary["strata"][stratum]["methods"][method]["success"])
                for stratum in STRATA
                for method in METHODS
            ),
        ),
    )
    success_axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    success_axis.set_axisbelow(True)
    success_axis.spines[["top", "right"]].set_visible(False)
    handles, labels = success_axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.875),
        ncol=3,
        fontsize=8.2,
        frameon=False,
    )

    effects = [
        summary["strata"][stratum]["paired_differences"][
            "kyc_fla_minus_poseaug_control_fla"
        ]["success"]
        for stratum in STRATA
    ]
    y = np.arange(len(STRATA), dtype=np.float64)[::-1]
    for y_value, effect in zip(y, effects):
        delta = 100.0 * float(effect["delta"])
        low = 100.0 * float(effect["ci95_low"])
        high = 100.0 * float(effect["ci95_high"])
        effect_axis.errorbar(
            delta,
            y_value,
            xerr=np.asarray([[delta - low], [high - delta]]),
            fmt="o",
            color=METHOD_COLORS["kyc_fla"],
            ecolor=METHOD_COLORS["kyc_fla"],
            capsize=5,
            markersize=7,
        )
        effect_axis.text(
            high + 0.6,
            y_value,
            f"{delta:+.1f} [{low:+.1f}, {high:+.1f}]",
            va="center",
            fontsize=8,
        )
    effect_axis.axvline(0.0, color="#111111", linewidth=1.0)
    effect_axis.axvline(5.0, color="#6A994E", linestyle="--", linewidth=1.2)
    effect_axis.set_yticks(
        y,
        ["All support poses", "Objects visible", "Fully visible"],
    )
    lows = [100.0 * float(effect["ci95_low"]) for effect in effects]
    highs = [100.0 * float(effect["ci95_high"]) for effect in effects]
    effect_axis.set_xlim(min(-8.0, min(lows) - 2.0), max(12.0, max(highs) + 8.0))
    effect_axis.set_xlabel("KYC - matched Control (percentage points)")
    effect_axis.set_title("(b) Paired KYC effect (95% CI)")
    effect_axis.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    effect_axis.set_axisbelow(True)
    effect_axis.spines[["top", "right"]].set_visible(False)

    ray = summary["ray_diagnostic"]
    ray_values = [
        float(ray["canonical_vs_correct"]["chunk_rms"]),
        float(ray["mismatched_vs_correct"]["chunk_rms"]),
    ]
    bars = ray_axis.bar(
        ["Canonical ray\nvs correct", "Mismatched ray\nvs correct"],
        ray_values,
        color=["#8C8C8C", METHOD_COLORS["kyc_fla"]],
        width=0.58,
    )
    threshold = float(summary["gate"]["minimum_causal_ray_rms"])
    ray_axis.axhline(threshold, color="#6A994E", linestyle="--", linewidth=1.2)
    for bar, value in zip(bars, ray_values):
        ray_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(ray_values + [threshold]) * 0.035,
            f"{value:.5f}",
            ha="center",
            fontsize=9,
        )
    ray_axis.text(
        0.98,
        threshold,
        f" pre-registered gate = {threshold:.3f}",
        transform=ray_axis.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color="#4C7031",
    )
    ray_axis.set_ylabel("Predicted action-chunk RMS")
    ray_axis.set_title("(c) Causal response to ray substitution")
    ray_axis.set_ylim(0.0, max(ray_values + [threshold]) * 1.30)
    ray_axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ray_axis.set_axisbelow(True)
    ray_axis.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5,
        0.035,
        f"Pre-registered screen decision: {summary['gate']['decision']}",
        ha="center",
        fontsize=9.5,
        fontweight="semibold",
    )
    _atomic_save(fig, output)


def _load_evaluation_rows(paths: Mapping[str, Path]) -> dict[str, list[Mapping[str, Any]]]:
    rows = {}
    keys = {}
    for method in METHODS:
        payload = _load_json(paths[method])
        if payload.get("status") != "complete":
            raise ValueError(f"incomplete evaluation for {method}")
        method_rows = list(payload["rows"])
        rows[method] = method_rows
        keys[method] = {
            (
                str(row["edge_id"]),
                int(row["canonical_state_index"]),
                int(row["execution_horizon"]),
                str(row["camera_pose"]),
            )
            for row in method_rows
        }
    if len({frozenset(value) for value in keys.values()}) != 1:
        raise ValueError("visual alignment evaluations are not episode-paired")
    return rows


def _pose_success(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    values: Sequence[float],
) -> list[float]:
    result = []
    for value in values:
        selected = _dimension_rows(rows, field=field, value=value)
        if not selected:
            raise ValueError(f"no rows for {field}={value}")
        result.append(100.0 * float(np.mean([bool(row["success"]) for row in selected])))
    return result


def _pose_visibility(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    values: Sequence[float],
) -> list[float]:
    result = []
    for value in values:
        selected = _dimension_rows(rows, field=field, value=value)
        if not selected:
            raise ValueError(f"no rows for {field}={value}")
        result.append(
            100.0
            * float(
                np.mean(
                    [
                        bool(row["task_objects_fully_visible"])
                        and bool(row["task_centers_in_frame"])
                        for row in selected
                    ]
                )
            )
        )
    return result


def _dimension_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    value: float,
) -> list[Mapping[str, Any]]:
    if field not in DEFAULT_CAMERA:
        raise ValueError(f"unknown camera dimension: {field}")
    return [
        row
        for row in rows
        if np.isclose(float(row[field]), value)
        and all(
            other_field == field
            or np.isclose(float(row[other_field]), default_value)
            for other_field, default_value in DEFAULT_CAMERA.items()
        )
    ]


def render_pose_response(
    *,
    evaluation_paths: Mapping[str, Path],
    output: Path,
) -> None:
    rows = _load_evaluation_rows(evaluation_paths)
    dimensions = (
        ("camera_azimuth_deg", (-60.0, 0.0, 60.0), "Horizontal orbit angle (deg)"),
        ("camera_elevation_deg", (-25.0, 0.0, 25.0), "Vertical angle offset (deg)"),
        ("camera_radius_scale", (0.9, 1.0, 1.25), "Camera distance multiplier"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))
    fig.subplots_adjust(left=0.06, right=0.94, top=0.82, bottom=0.18, wspace=0.32)
    fig.suptitle(
        "Closed-loop response to the external-camera pose",
        fontsize=15,
        fontweight="semibold",
        y=0.96,
    )
    fig.text(
        0.5,
        0.89,
        "Each point aggregates the same 5 snapshot groups and 4 supervised task edges",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    visibility_rows = rows[METHODS[0]]
    for axis, (field, values, xlabel) in zip(axes, dimensions):
        for method in METHODS:
            axis.plot(
                values,
                _pose_success(rows[method], field=field, values=values),
                marker="o",
                linewidth=2.0,
                markersize=5.5,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
            )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Task success (%)")
        axis.set_ylim(0.0, 100.0)
        axis.grid(color=GRID_COLOR, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        visibility_axis = axis.twinx()
        visibility_axis.plot(
            values,
            _pose_visibility(visibility_rows, field=field, values=values),
            color="#111111",
            linestyle="--",
            linewidth=1.2,
            marker="s",
            markersize=4,
            label="Both objects fully visible",
        )
        visibility_axis.set_ylim(0.0, 100.0)
        visibility_axis.set_ylabel("Full-visibility coverage (%)", color="#333333")
        visibility_axis.spines["top"].set_visible(False)
        visibility_axis.tick_params(axis="y", colors="#555555")
    handles, labels = axes[0].get_legend_handles_labels()
    visibility_handle = Line2D(
        [],
        [],
        color="#111111",
        linestyle="--",
        marker="s",
    )
    fig.legend(
        handles + [visibility_handle],
        labels + ["Both objects fully visible"],
        loc="lower center",
        ncol=4,
        fontsize=8.5,
        frameon=False,
    )
    _atomic_save(fig, output)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Pi0.5 KYC visual-alignment screen figures"
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--evaluation", action="append", required=True)
    parser.add_argument("--screen-output", type=Path, required=True)
    parser.add_argument("--pose-output", type=Path, required=True)
    return parser.parse_args(args)


def main(args: Iterable[str] | None = None) -> None:
    parsed = parse_args(args)
    pairs = [_parse_evaluation(value) for value in parsed.evaluation]
    paths = dict(pairs)
    if set(paths) != set(METHODS) or len(pairs) != len(METHODS):
        raise ValueError(f"exactly one evaluation is required for each of {METHODS}")
    summary = _load_json(parsed.summary)
    render_screen_summary(summary=summary, output=parsed.screen_output)
    render_pose_response(evaluation_paths=paths, output=parsed.pose_output)
    print(
        json.dumps(
            {
                "screen_output": str(parsed.screen_output),
                "pose_output": str(parsed.pose_output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
