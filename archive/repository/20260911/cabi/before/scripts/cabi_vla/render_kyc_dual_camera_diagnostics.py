from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "dual_rgb_fla": "RGB",
    "dual_control_fla": "Control",
    "external_fla": "External",
    "wrist_fla": "Wrist",
    "dual_fla": "Dual",
}
METHOD_COLORS = {
    "dual_rgb_fla": "#4C956C",
    "dual_control_fla": "#355C7D",
    "external_fla": "#E9C46A",
    "wrist_fla": "#8E6CBE",
    "dual_fla": "#E76F51",
}
POSE_LABELS = {
    "baseline": "Default",
    "az_m60": "Az -60",
    "az_p60": "Az +60",
    "el_m25": "El -25",
    "el_p25": "El +25",
    "rad_0900": "R 0.90",
    "rad_1250": "R 1.25",
}


def _atomic_save(fig: Any, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite figure: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    fig.savefig(temporary, format="png", dpi=190, facecolor="white")
    plt.close(fig)
    os.replace(temporary, output)


def render(payload: Mapping[str, Any], *, output: Path) -> None:
    if payload.get("status") != "complete":
        raise ValueError("dual-camera diagnostics are incomplete")
    if payload.get("study") != "kyc_pi05_dual_camera_diagnostics":
        raise ValueError("unexpected dual-camera diagnostics study")

    poses = list(payload["pose_order"])
    x = np.arange(len(poses))
    labels = [POSE_LABELS.get(pose, pose) for pose in poses]
    methods = list(METHOD_LABELS)

    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.5))
    fig.subplots_adjust(
        left=0.16,
        right=0.98,
        top=0.90,
        bottom=0.10,
        hspace=0.38,
        wspace=0.28,
    )
    fig.suptitle(
        "Pi0.5 dual-camera KYC diagnostic closure",
        fontsize=17,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.935,
        (
            "Matched initial observations and states; "
            f"seed {payload['seed']}; {int(payload['training_updates']):,} updates; "
            f"K={int(payload['execution_horizon'])}"
        ),
        ha="center",
        fontsize=9.5,
        color="#444444",
    )

    visibility_axis, subgoal_axis, effect_axis, causal_axis = axes.ravel()
    visibility_series = (
        ("task_objects_visible", "Both objects visible", "#2A9D8F"),
        ("task_objects_fully_visible", "Both fully visible", "#E9C46A"),
        ("task_centers_in_frame", "Both centers in frame", "#355C7D"),
    )
    for key, label, color in visibility_series:
        values = [
            100.0 * float(payload["pose_diagnostics"][pose]["visibility"][key])
            for pose in poses
        ]
        visibility_axis.plot(x, values, marker="o", linewidth=2.0, color=color, label=label)
    visibility_axis.set_xticks(x, labels, rotation=20, ha="right")
    visibility_axis.set_ylim(-2.0, 104.0)
    visibility_axis.set_ylabel("Episode fraction (%)")
    visibility_axis.set_title("(a) Initial view boundary and object visibility")
    visibility_axis.legend(frameon=False, fontsize=8)

    subgoals = (
        "source_selection_success",
        "lift_success",
        "transport_success",
        "target_placement_success",
        "success",
    )
    subgoal_labels = ("Select/grasp", "Lift", "Transport", "Place", "Task")
    width = 0.16
    for method_index, method in enumerate(methods):
        values = [100.0 * float(payload["overall"][method][key]) for key in subgoals]
        subgoal_axis.bar(
            np.arange(len(subgoals)) + (method_index - 2) * width,
            values,
            width=width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    subgoal_axis.set_xticks(np.arange(len(subgoals)), subgoal_labels)
    subgoal_axis.set_ylabel("Success (%)")
    subgoal_axis.set_title("(b) Closed-loop subgoal completion")
    subgoal_axis.legend(ncol=3, frameon=False, fontsize=8)

    effects = (
        ("external_at_canonical_wrist", "External | canonical wrist"),
        ("external_at_real_wrist", "External | real wrist"),
        ("wrist_at_canonical_external", "Wrist | canonical external"),
        ("wrist_at_real_external", "Wrist | real external"),
        ("external_average_main_effect", "External mean effect"),
        ("wrist_average_main_effect", "Wrist mean effect"),
        ("interaction", "External x wrist"),
    )
    for effect_index, (key, label) in enumerate(effects):
        result = payload["factorial_effects"][key]["success"]
        delta = 100.0 * float(result["delta"])
        low = 100.0 * float(result["ci95_low"])
        high = 100.0 * float(result["ci95_high"])
        effect_axis.errorbar(
            delta,
            effect_index,
            xerr=np.asarray([[delta - low], [high - delta]]),
            fmt="o",
            color="#355C7D" if effect_index < 6 else "#2A9D8F",
            capsize=4,
            markersize=6,
        )
    effect_axis.axvline(0.0, color="#222222", linewidth=1.0)
    effect_axis.set_yticks(np.arange(len(effects)), [label for _, label in effects])
    effect_axis.set_xlabel("Paired success difference (percentage points)")
    effect_axis.set_title("(c) 2 x 2 camera-geometry factorial effects")

    causal_conditions = (
        ("correct", "Correct wrist ray", "#E76F51"),
        ("initial", "Initial fixed ray", "#8D99AE"),
        ("lagged", "Previous-call ray (K=3)", "#495057"),
    )
    for condition, label, color in causal_conditions:
        values = [
            100.0
            * float(
                payload["pose_diagnostics"][pose]["wrist_ray_intervention"][condition][
                    "success"
                ]
            )
            for pose in poses
        ]
        causal_axis.plot(x, values, marker="o", linewidth=2.0, color=color, label=label)
    causal_axis.set_xticks(x, labels, rotation=20, ha="right")
    causal_axis.set_ylabel("Closed-loop success (%)")
    causal_axis.set_title("(d) Causal wrist-ray intervention by camera pose")
    causal_axis.legend(frameon=False, fontsize=8)

    for axis in axes.ravel():
        axis.grid(color="#D9DEE3", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    _atomic_save(fig, output)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render dual-camera KYC diagnostics")
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    render(json.loads(args.diagnostics.read_text()), output=args.output)
    print(json.dumps({"output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
