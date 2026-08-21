#!/usr/bin/env python3
"""Render reproducible interim figures for the view revalidation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASET_MANIFEST = Path(
    "/share/longjunyu/alphabrain/datasets/dsol-libero-broad-pairs-v1/"
    "quick_gate_seed41_broad64_stride2/manifest.json"
)
DEVELOPMENT_METRICS = Path(
    "/share/longjunyu/alphabrain/experiments/dsol-libero-broad-pairing-v1/"
    "analysis/fixed_condition_seed41_v1/metrics.json"
)
M0_SUMMARY = Path(
    "/share/longjunyu/alphabrain/experiments/dsol-libero-constructed-m0-v1/"
    "operational-three-task-scan-v2/analysis/summary.json"
)
M1_METRICS = Path(
    "/share/longjunyu/alphabrain/experiments/dsol-libero-constructed-m1-v2/"
    "cross-model-analysis/metrics.json"
)
CAMERA_FULL_ROOT = Path(
    "/share/longjunyu/alphabrain/experiments/libero-plus-camera-full-v1"
)
ACCEL_ROOT = Path(
    "/share/longjunyu/alphabrain/experiments/dsol-accel-constructed-v2/m1-joins"
)

COLORS = {
    "navy": "#294C60",
    "blue": "#3D7EA6",
    "teal": "#3F8C7A",
    "gold": "#C79432",
    "red": "#BE5968",
    "gray": "#8A9299",
    "ink": "#1D2731",
    "grid": "#D8DEE4",
}

ARM_LABELS = {
    "canonical_unique": "Canonical\nunique",
    "canonical_repeat": "Canonical\nrepeat",
    "image_augmentation_unique": "Image\naug.",
    "broad_unpaired_practical": "Broad\npractical",
    "broad_unpaired_state_matched": "State\nmatched",
    "broad_paired_fm": "Paired\nFM",
    "broad_paired_consistency": "Paired +\nconsistency",
}

MODEL_LABELS = {
    "official": "Official",
    "broad64-practical": "Broad practical",
    "broad64-state-matched": "State matched",
    "broad64-paired-fm": "Paired FM",
    "broad64-paired-consistency": "Paired + consistency",
}


def load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def annotate_bars(ax: plt.Axes, bars, suffix: str = "") -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(ax.get_ylim()[1] * 0.012, 0.5),
            f"{value:.1f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["ink"],
        )


def plot_data_and_passive(output: Path) -> None:
    dataset = load(DATASET_MANIFEST)
    dev = load(DEVELOPMENT_METRICS)
    official = load(CAMERA_FULL_ROOT / "official-pi05-frozen-camera1599/metrics.json")["summary"]
    broad = load(CAMERA_FULL_ROOT / "broad64-seed41-camera1599/metrics.json")["summary"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    fig.suptitle(
        "Broad-view training: custom diagnostics and official camera benchmark (interim)",
        fontsize=17,
        fontweight="bold",
        color=COLORS["ink"],
    )

    ax = axes[0]
    split_names = ["Train", "Validation", "Test"]
    split_values = [dataset["counts_by_split"][key] for key in ["train", "val", "test"]]
    bars = ax.bar(split_names, split_values, color=[COLORS["blue"], COLORS["gold"], COLORS["teal"]])
    ax.set_title("A. Broad64 paired render records", loc="left", fontweight="bold")
    ax.set_ylabel("Record count")
    ax.set_ylim(0, max(split_values) * 1.18)
    for bar, value in zip(bars, split_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 500, f"{value:,}", ha="center", fontsize=9)
    ax.text(
        0.02,
        0.96,
        f"Total {dataset['record_count']:,} | 8 tasks | status {dataset['status']}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color=COLORS["ink"],
    )
    style_axis(ax)

    ax = axes[1]
    arms = list(ARM_LABELS)
    conditions = ["canonical_both", "broad_heldout_both", "wide_extrapolation_both"]
    condition_labels = ["Canonical", "Broad held-out", "Wide extrapolation"]
    condition_colors = [COLORS["gray"], COLORS["blue"], COLORS["teal"]]
    x = np.arange(len(arms))
    width = 0.25
    for offset, condition, label, color in zip(
        [-width, 0, width], conditions, condition_labels, condition_colors
    ):
        values = [100 * dev["success_rates"][arm][condition] for arm in arms]
        ax.bar(x + offset, values, width, label=label, color=color)
    ax.set_title("B. LIBERO-derived exact-state diagnostic", loc="left", fontweight="bold")
    ax.set_ylabel("Closed-loop success (%)")
    ax.set_xticks(x, [ARM_LABELS[arm] for arm in arms], fontsize=8)
    ax.set_ylim(0, 108)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    ax.text(
        0.02,
        0.96,
        "24 paired episodes; custom protocol, not an official leaderboard score",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    style_axis(ax)

    ax = axes[2]
    suites = ["libero_10", "libero_goal", "libero_object", "libero_spatial", "overall"]
    labels = [
        "LIBERO-10\nsuite",
        "Goal\nsuite",
        "Object\nsuite",
        "Spatial\nsuite",
        "All 1,599\n(Pooled)",
    ]
    official_values = [
        *[100 * official["by_suite"][suite]["success_rate"] for suite in suites[:-1]],
        100 * official["official_pooled"]["success_rate"],
    ]
    broad_values = [
        *[100 * broad["by_suite"][suite]["success_rate"] for suite in suites[:-1]],
        100 * broad["official_pooled"]["success_rate"],
    ]
    x = np.arange(len(labels))
    bars_a = ax.bar(x - 0.19, official_values, 0.38, label="Official Pi0.5", color=COLORS["gray"])
    bars_b = ax.bar(x + 0.19, broad_values, 0.38, label="Broad64 seed 41", color=COLORS["teal"])
    ax.set_title("C. Official LIBERO-Plus Camera track", loc="left", fontweight="bold")
    ax.set_ylabel("Closed-loop success (%)")
    ax.set_xticks(x, labels, rotation=18, ha="right", fontsize=8)
    ax.set_ylim(0, 103)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    annotate_bars(ax, bars_a, "%")
    annotate_bars(ax, bars_b, "%")
    ax.text(
        0.02,
        0.96,
        "All bars are camera-perturbed; first four are suite subsets",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    style_axis(ax)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_m0_m1_accel(output: Path) -> None:
    m0 = load(M0_SUMMARY)
    m1 = load(M1_METRICS)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), constrained_layout=True)
    fig.suptitle(
        "Visibility intervention, full closed loop, and Accel selection (interim)",
        fontsize=17,
        fontweight="bold",
        color=COLORS["ink"],
    )

    ax = axes[0]
    m0_groups = [
        "broad_heldout_32",
        "wide_extrapolation_24",
        "diagnostic_crossed_orbit",
        "diagnostic_extreme_orbit",
        "diagnostic_look_away",
        "sensor_controls",
    ]
    m0_labels = ["Broad held-out", "Wide extrap.", "Crossed orbit", "Extreme", "Look-away", "Sensor ctrl"]
    stats = [m0["group_delta_statistics"][f"test::{group}"] for group in m0_groups]
    positions = np.arange(len(stats))
    for y, stat in zip(positions, stats):
        ax.hlines(y, stat["q05"], stat["q95"], color=COLORS["gray"], linewidth=2)
        ax.hlines(y, stat["q25"], stat["q75"], color=COLORS["blue"], linewidth=7)
        ax.plot(stat["median"], y, "o", color=COLORS["red"], markersize=5)
    ax.axvline(0, color=COLORS["ink"], linewidth=1)
    ax.set_yticks(positions, m0_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Visibility change vs canonical")
    ax.set_title(
        "A. M0 candidate visibility\n180 states / 15,840 candidates",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)

    ax = axes[1]
    models = list(MODEL_LABELS)
    conditions = ["canonical_both", "strong_info_both", "matched_control_both", "blind_both"]
    condition_labels = ["Canonical", "Strong-info", "Matched control", "Blind"]
    condition_colors = [COLORS["gray"], COLORS["teal"], COLORS["gold"], COLORS["red"]]
    lookup = {(row["model"], row["condition"]): row for row in m1["condition_success"]}
    x = np.arange(len(models))
    width = 0.19
    for index, (condition, label, color) in enumerate(
        zip(conditions, condition_labels, condition_colors)
    ):
        values = [100 * lookup[(model, condition)]["state_success_rate"] for model in models]
        ax.bar(x + (index - 1.5) * width, values, width, label=label, color=color)
    ax.set_title(
        "B. M1 full closed-loop gate\n21 states / 6 independent source demonstrations",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )
    ax.set_ylabel("Success (%)")
    ax.set_xticks(x, [MODEL_LABELS[model] for model in models], rotation=18, ha="right", fontsize=8)
    ax.set_ylim(0, 82)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    style_axis(ax)

    ax = axes[2]
    accel_models = [
        "broad64-practical",
        "broad64-state-matched",
        "broad64-paired-fm",
        "broad64-paired-consistency",
    ]
    role_order = ["canonical", "matched_control", "strong_info", "blind"]
    role_colors = [COLORS["gray"], COLORS["gold"], COLORS["teal"], COLORS["red"]]
    left = np.zeros(len(accel_models))
    delta_text = []
    for role, color in zip(role_order, role_colors):
        values = []
        for model in accel_models:
            metrics = load(ACCEL_ROOT / model / "metrics.json")
            values.append(metrics["accel_selected_role_counts"].get(role, 0) / metrics["paired_state_count"] * 100)
            if role == role_order[0]:
                delta_text.append(metrics["accel_minus_canonical_success_pp"])
        ax.barh(
            [MODEL_LABELS[model] for model in accel_models],
            values,
            left=left,
            color=color,
            label=role.replace("_", " ").title(),
        )
        left += np.asarray(values)
    for y, delta in enumerate(delta_text):
        ax.text(101, y, f"SR delta {delta:+.1f} pp", va="center", fontsize=8)
    ax.set_xlim(0, 126)
    ax.set_xlabel("Accel-selected view role (%)")
    ax.set_title(
        "C. Accel view-choice behavior\nMostly canonical; no reliable success gain",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )
    ax.legend(
        frameon=False,
        fontsize=7,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/dsol_paper1/figures"),
    )
    args = parser.parse_args()
    plot_data_and_passive(args.output_dir / "view_revalidation_data_passive_interim.png")
    plot_m0_m1_accel(args.output_dir / "view_revalidation_m0_m1_accel_interim.png")


if __name__ == "__main__":
    main()
