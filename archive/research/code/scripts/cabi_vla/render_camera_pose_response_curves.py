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


METHOD_SPECS = {
    "base": {
        "label": "Base (fixed-view training, seed 41)",
        "color": "#8C8C8C",
        "linestyle": ":",
        "linewidth": 2.0,
    },
    "poseaug_rgb": {
        "label": "PoseAug-RGB (seed 41)",
        "color": "#5B8E7D",
        "linestyle": "--",
        "linewidth": 2.0,
    },
    "poseaug_control": {
        "label": "PoseAug-Control (3-seed mean)",
        "color": "#355C7D",
        "linestyle": "-",
        "linewidth": 2.3,
    },
    "kyc": {
        "label": "KYC (3-seed mean)",
        "color": "#E76F51",
        "linestyle": "-",
        "linewidth": 2.3,
    },
}

AXES = {
    "azimuth": {
        "title": "Horizontal orbit / azimuth",
        "parameter": "camera_azimuth_deg",
        "poses": ["az_m90", "az_m60", "baseline", "az_p60", "az_p90"],
        "values": [-90.0, -60.0, 0.0, 60.0, 90.0],
        "support": (-60.0, 60.0),
        "xlabel": "Azimuth offset (degrees)",
    },
    "elevation": {
        "title": "Camera height angle / elevation",
        "parameter": "camera_elevation_deg",
        "poses": ["el_m32", "el_m25", "baseline", "el_p25", "el_p32"],
        "values": [-32.0, -25.0, 0.0, 25.0, 32.0],
        "support": (-25.0, 25.0),
        "xlabel": "Elevation offset (degrees)",
    },
    "radius": {
        "title": "Camera distance / radius",
        "parameter": "camera_radius_scale",
        "poses": ["rad_0750", "rad_0900", "baseline", "rad_1250", "rad_1400"],
        "values": [0.75, 0.9, 1.0, 1.25, 1.4],
        "support": (0.9, 1.25),
        "xlabel": "Distance relative to default",
    },
}

GRID_COLOR = "#D9DEE3"
SUPPORT_COLOR = "#DDECDD"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"incomplete camera evaluation: {path}")
    rows = payload["rows"]
    expected = int(payload["expected_episode_count"])
    if len(rows) != expected:
        raise ValueError(f"row-count mismatch in {path}: {len(rows)} != {expected}")
    return rows


def _key(row: Mapping[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(row["edge_id"]),
        int(row["canonical_state_index"]),
        int(row["execution_horizon"]),
        str(row["camera_pose"]),
    )


def _load_method(paths: Sequence[Path]) -> list[list[dict[str, Any]]]:
    evaluations = [_load_rows(path) for path in paths]
    reference_keys = {_key(row) for row in evaluations[0]}
    if len(reference_keys) != len(evaluations[0]):
        raise ValueError(f"duplicate episode keys in {paths[0]}")
    for path, rows in zip(paths[1:], evaluations[1:]):
        keys = {_key(row) for row in rows}
        if keys != reference_keys or len(keys) != len(rows):
            raise ValueError(f"evaluation grid mismatch: {path}")
    return evaluations


def _pose_metric(
    evaluations: Sequence[Sequence[Mapping[str, Any]]],
    *,
    pose: str,
    metric: str,
) -> float:
    per_seed = []
    for rows in evaluations:
        values = [
            float(row[metric])
            for row in rows
            if str(row["camera_pose"]) == pose
        ]
        if not values:
            raise ValueError(f"pose {pose!r} has no rows")
        per_seed.append(float(np.mean(values)))
    return 100.0 * float(np.mean(per_seed))


def _visual_support(rows: Sequence[Mapping[str, Any]], *, pose: str) -> float:
    pose_rows = [row for row in rows if str(row["camera_pose"]) == pose]
    if not pose_rows:
        raise ValueError(f"pose {pose!r} has no rows")
    supported = [
        bool(row["source_center_in_frame"])
        and bool(row["target_center_in_frame"])
        and int(row["source_visible_pixels"]) >= 64
        and int(row["target_visible_pixels"]) >= 64
        for row in pose_rows
    ]
    return 100.0 * float(np.mean(supported))


def _validate_camera_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    axis: Mapping[str, Any],
) -> None:
    parameter = str(axis["parameter"])
    for pose, expected in zip(axis["poses"], axis["values"]):
        values = {
            round(float(row[parameter]), 8)
            for row in rows
            if str(row["camera_pose"]) == pose
        }
        if values != {round(float(expected), 8)}:
            raise ValueError(
                f"unexpected {parameter} for {pose}: {values} != {expected}"
            )


def _support_amplitudes(
    evaluations: Mapping[str, Sequence[Sequence[Mapping[str, Any]]]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for method, method_evaluations in evaluations.items():
        result[method] = {}
        for axis_name, axis in AXES.items():
            low, high = map(float, axis["support"])
            values = [
                _pose_metric(
                    method_evaluations,
                    pose=str(pose),
                    metric="success",
                )
                for pose, value in zip(axis["poses"], axis["values"])
                if low <= float(value) <= high
            ]
            result[method][axis_name] = float(max(values) - min(values))
    return result


def render(
    *,
    evaluations: Mapping[str, Sequence[Sequence[Mapping[str, Any]]]],
    output: Path,
    summary_output: Path,
) -> None:
    reference_rows = evaluations["poseaug_control"][0]
    for axis in AXES.values():
        _validate_camera_values(reference_rows, axis=axis)

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(16.0, 12.0),
        sharex="col",
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.72]},
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.975,
        top=0.84,
        bottom=0.09,
        wspace=0.23,
        hspace=0.18,
    )
    fig.suptitle(
        "Closed-loop response to external-camera pose",
        fontsize=17,
        fontweight="semibold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.925,
        (
            "Same held-out grid: 4 tasks x 10 snapshot groups x 13 camera "
            "poses; K=3, 320-step limit"
        ),
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.legend(
        [
            plt.Line2D(
                [0],
                [0],
                color=spec["color"],
                linestyle=spec["linestyle"],
                linewidth=spec["linewidth"],
                marker="o",
            )
            for spec in METHOD_SPECS.values()
        ],
        [str(spec["label"]) for spec in METHOD_SPECS.values()],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=4,
        frameon=False,
        fontsize=9,
    )

    for column, (axis_name, axis) in enumerate(AXES.items()):
        x = np.asarray(axis["values"], dtype=np.float64)
        support_low, support_high = map(float, axis["support"])
        for row_index in (0, 1, 2):
            plot_axis = axes[row_index, column]
            plot_axis.axvspan(
                support_low,
                support_high,
                color=SUPPORT_COLOR,
                alpha=0.75,
                zorder=0,
            )
            neutral = 1.0 if axis_name == "radius" else 0.0
            plot_axis.axvline(
                neutral,
                color="#333333",
                linewidth=0.9,
                linestyle="--",
                zorder=1,
            )
            plot_axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
            plot_axis.set_axisbelow(True)
            plot_axis.spines[["top", "right"]].set_visible(False)

        axes[0, column].set_title(str(axis["title"]), fontsize=12)
        for method, method_evaluations in evaluations.items():
            spec = METHOD_SPECS[method]
            for row_index, metric in ((0, "success"), (1, "progress")):
                y = np.asarray(
                    [
                        _pose_metric(
                            method_evaluations,
                            pose=str(pose),
                            metric=metric,
                        )
                        for pose in axis["poses"]
                    ],
                    dtype=np.float64,
                )
                axes[row_index, column].plot(
                    x,
                    y,
                    color=spec["color"],
                    linestyle=spec["linestyle"],
                    linewidth=spec["linewidth"],
                    marker="o",
                    markersize=5,
                    zorder=3,
                )

        support = np.asarray(
            [
                _visual_support(reference_rows, pose=str(pose))
                for pose in axis["poses"]
            ],
            dtype=np.float64,
        )
        axes[2, column].plot(
            x,
            support,
            color="#222222",
            linewidth=2.0,
            marker="o",
            markersize=5,
            zorder=3,
        )
        for x_value, y_value in zip(x, support):
            axes[2, column].text(
                x_value,
                min(104.0, y_value + 4.0),
                f"{y_value:.0f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#333333",
            )

        for row_index in (0, 1, 2):
            axes[row_index, column].set_ylim(0.0, 108.0)
            axes[row_index, column].set_xticks(x)
        axes[2, column].set_xlabel(str(axis["xlabel"]))
        axes[0, column].text(
            support_low,
            102.0,
            "training support",
            ha="left",
            va="top",
            fontsize=8,
            color="#557755",
        )

    axes[0, 0].set_ylabel("Full-task success (%)")
    axes[1, 0].set_ylabel("Subgoal progress (%)")
    axes[2, 0].set_ylabel("Initial visual support (%)")
    axes[2, 1].text(
        0.5,
        -0.43,
        (
            "Visual support = both task-object centers inside the external "
            "image and each object has at least 64 visible pixels."
        ),
        transform=axes[2, 1].transAxes,
        ha="center",
        fontsize=8.5,
    )
    fig.text(
        0.5,
        0.025,
        (
            "Base and PoseAug-RGB are seed-41 descriptive curves (40 episodes "
            "per pose). Control and KYC are equal means over seeds 41/42/43 "
            "(120 episodes per pose). Green bands mark the camera-pose "
            "training range."
        ),
        ha="center",
        fontsize=8.5,
    )

    if output.exists():
        raise FileExistsError(f"refusing to overwrite figure: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    fig.savefig(temporary, format="png", dpi=170, facecolor="white")
    plt.close(fig)
    os.replace(temporary, output)

    summary = {
        "schema_version": 1,
        "study": "libero_bind_camera_pose_response_curves",
        "status": "complete",
        "methods": {
            method: {
                "evaluation_count": len(method_evaluations),
                "training_seed_count": len(method_evaluations),
                "curves": {
                    axis_name: {
                        "camera_values": list(map(float, axis["values"])),
                        "poses": list(map(str, axis["poses"])),
                        "success_percent": [
                            _pose_metric(
                                method_evaluations,
                                pose=str(pose),
                                metric="success",
                            )
                            for pose in axis["poses"]
                        ],
                        "progress_percent": [
                            _pose_metric(
                                method_evaluations,
                                pose=str(pose),
                                metric="progress",
                            )
                            for pose in axis["poses"]
                        ],
                    }
                    for axis_name, axis in AXES.items()
                },
            }
            for method, method_evaluations in evaluations.items()
        },
        "visual_support_percent": {
            axis_name: [
                _visual_support(reference_rows, pose=str(pose))
                for pose in axis["poses"]
            ]
            for axis_name, axis in AXES.items()
        },
        "training_support_success_amplitude_pp": _support_amplitudes(
            evaluations
        ),
        "visual_support_definition": (
            "both source and target centers are inside the external image and "
            "each object has at least 64 visible pixels"
        ),
        "figure": str(output),
    }
    if summary_output.exists():
        raise FileExistsError(
            f"refusing to overwrite summary: {summary_output}"
        )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary = summary_output.with_name(
        f".{summary_output.name}.{os.getpid()}.tmp"
    )
    temporary_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_summary, summary_output)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render three-axis camera-pose response curves"
    )
    parser.add_argument("--base-eval", type=Path, required=True)
    parser.add_argument("--poseaug-rgb-eval", type=Path, required=True)
    parser.add_argument(
        "--poseaug-control-evals",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument("--kyc-evals", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args(args)


def main(args: Iterable[str] | None = None) -> None:
    parsed = parse_args(args)
    evaluations = {
        "base": _load_method([parsed.base_eval]),
        "poseaug_rgb": _load_method([parsed.poseaug_rgb_eval]),
        "poseaug_control": _load_method(parsed.poseaug_control_evals),
        "kyc": _load_method(parsed.kyc_evals),
    }
    render(
        evaluations=evaluations,
        output=parsed.output,
        summary_output=parsed.summary_output,
    )
    print(
        json.dumps(
            {
                "output": str(parsed.output),
                "summary_output": str(parsed.summary_output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
