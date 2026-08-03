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

from summarize_kyc_dual_camera_screen import METHODS


LABELS = {
    "dual_rgb_fla": "RGB",
    "dual_control_fla": "Control",
    "external_fla": "External",
    "wrist_fla": "Wrist",
    "dual_fla": "Dual",
}
COLORS = {
    "dual_rgb_fla": "#4C956C",
    "dual_control_fla": "#355C7D",
    "external_fla": "#E9C46A",
    "wrist_fla": "#9C6ADE",
    "dual_fla": "#E76F51",
}
POSES = (
    "baseline",
    "az_m60",
    "az_p60",
    "el_m25",
    "el_p25",
    "rad_0900",
    "rad_1250",
)
POSE_LABELS = ("Default", "Az -60", "Az +60", "El -25", "El +25", "R 0.90", "R 1.25")


def _load(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text())


def _parse_evaluation(value: str) -> tuple[str, Path]:
    method, separator, path = value.partition("=")
    if not separator or method not in METHODS or not path:
        raise ValueError(f"evaluation must be METHOD=PATH with METHOD in {METHODS}")
    return method, Path(path)


def _atomic_save(fig: Any, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite figure: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    fig.savefig(temporary, format="png", dpi=190, facecolor="white")
    plt.close(fig)
    os.replace(temporary, output)


def _pose_success(rows: Sequence[Mapping[str, Any]], pose: str) -> float:
    selected = [row for row in rows if row["camera_pose"] == pose]
    if not selected:
        raise ValueError(f"evaluation is missing pose: {pose}")
    return float(np.mean([float(row["success"]) for row in selected]))


def render(
    *,
    summary: Mapping[str, Any],
    evaluation_paths: Mapping[str, Path],
    output: Path,
) -> None:
    if summary.get("status") != "complete":
        raise ValueError("dual-camera summary is incomplete")
    if summary.get("study") != "kyc_pi05_dual_camera_screen":
        raise ValueError("unexpected dual-camera study")

    evaluations = {}
    reference_keys = None
    for method in METHODS:
        payload = _load(evaluation_paths[method])
        if payload.get("status") != "complete":
            raise ValueError(f"incomplete evaluation for {method}")
        rows = payload["rows"]
        keys = {
            (
                row["edge_id"],
                int(row["canonical_state_index"]),
                int(row["execution_horizon"]),
                row["camera_pose"],
            )
            for row in rows
        }
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError("plot evaluations are not episode-paired")
        evaluations[method] = rows

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.09, hspace=0.38, wspace=0.28)
    fig.suptitle("Pi0.5 dual-camera geometry screen", fontsize=17, fontweight="semibold")
    fig.text(
        0.5,
        0.935,
        "Matched data, seed 41, 2,000 updates, 5 snapshot groups x 4 edges x 7 camera poses, K=3",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )

    absolute_axis, effect_axis, causal_axis, pose_axis = axes.ravel()
    support = summary["strata"]["inside_training_support"]
    absolute = [100.0 * float(support["methods"][method]["success"]) for method in METHODS]
    x = np.arange(len(METHODS))
    bars = absolute_axis.bar(x, absolute, color=[COLORS[method] for method in METHODS])
    absolute_axis.set_xticks(x, [LABELS[method] for method in METHODS])
    absolute_axis.set_ylabel("Closed-loop success (%)")
    absolute_axis.set_title("(a) Absolute task success")
    absolute_axis.set_ylim(0.0, max(25.0, 1.20 * max(absolute)))
    absolute_axis.grid(axis="y", color="#D9DEE3", linewidth=0.8)
    absolute_axis.set_axisbelow(True)
    for bar, value in zip(bars, absolute):
        absolute_axis.text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.1f}", ha="center", fontsize=8)

    effects = (
        ("external_fla_minus_dual_control_fla", "External - Control", COLORS["external_fla"]),
        ("wrist_fla_minus_dual_control_fla", "Wrist - Control", COLORS["wrist_fla"]),
        ("dual_fla_minus_dual_control_fla", "Dual - Control", COLORS["dual_fla"]),
        ("dual_interaction", "Dual interaction", "#2A9D8F"),
    )
    for y, (key, label, color) in enumerate(effects):
        effect = support["paired_differences"][key]["success"]
        delta = 100.0 * float(effect["delta"])
        low = 100.0 * float(effect["ci95_low"])
        high = 100.0 * float(effect["ci95_high"])
        effect_axis.errorbar(
            delta,
            y,
            xerr=np.asarray([[delta - low], [high - delta]]),
            fmt="o",
            color=color,
            capsize=5,
            markersize=7,
        )
    effect_axis.axvline(0.0, color="#222222", linewidth=1.0)
    effect_axis.axvline(5.0, color="#6A994E", linestyle="--", linewidth=1.0)
    effect_axis.set_yticks(np.arange(len(effects)), [effect[1] for effect in effects])
    effect_axis.set_xlabel("Paired success difference (percentage points)")
    effect_axis.set_title("(b) Geometry effects with group bootstrap 95% CI")
    effect_axis.grid(axis="x", color="#D9DEE3", linewidth=0.8)
    effect_axis.set_axisbelow(True)

    causal = summary["wrist_ray_causal_intervention"]["inside_training_support"]
    conditions = ("correct", "initial", "lagged")
    causal_values = [100.0 * float(causal["conditions"][condition]["success"]) for condition in conditions]
    causal_colors = (COLORS["dual_fla"], "#8D99AE", "#6C757D")
    causal_bars = causal_axis.bar(np.arange(3), causal_values, color=causal_colors)
    causal_axis.set_xticks(
        np.arange(3),
        ("Correct wrist ray", "Initial fixed ray", "Previous-call ray (K=3)"),
    )
    causal_axis.set_ylabel("Closed-loop success (%)")
    causal_axis.set_title("(c) Same RGB, wrist-ray intervention")
    causal_axis.set_ylim(0.0, max(25.0, 1.20 * max(causal_values)))
    causal_axis.grid(axis="y", color="#D9DEE3", linewidth=0.8)
    causal_axis.set_axisbelow(True)
    for bar, value in zip(causal_bars, causal_values):
        causal_axis.text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.1f}", ha="center", fontsize=8)

    for method in METHODS:
        values = [100.0 * _pose_success(evaluations[method], pose) for pose in POSES]
        pose_axis.plot(
            np.arange(len(POSES)),
            values,
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            color=COLORS[method],
            label=LABELS[method],
        )
    pose_axis.set_xticks(np.arange(len(POSES)), POSE_LABELS, rotation=20, ha="right")
    pose_axis.set_ylabel("Closed-loop success (%)")
    pose_axis.set_title("(d) Response across external-camera poses")
    pose_axis.grid(color="#D9DEE3", linewidth=0.8)
    pose_axis.set_axisbelow(True)
    pose_axis.legend(ncol=3, fontsize=8, frameon=False)

    for axis in axes.ravel():
        axis.spines[["top", "right"]].set_visible(False)
    _atomic_save(fig, output)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render dual-camera KYC screen")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--evaluation", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    pairs = [_parse_evaluation(value) for value in args.evaluation]
    paths = dict(pairs)
    if len(pairs) != len(METHODS) or set(paths) != set(METHODS):
        raise ValueError("exactly one evaluation is required for every method")
    render(summary=_load(args.summary), evaluation_paths=paths, output=args.output)
    print(json.dumps({"output": str(args.output)}))


if __name__ == "__main__":
    main()
