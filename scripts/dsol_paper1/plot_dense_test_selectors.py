#!/usr/bin/env python3
"""Plot independent dense-test selector results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "canonical": "Canonical",
    "visibility_mean": "Visibility mean",
    "visibility_min_entity": "Min-entity visibility",
    "visibility_hmean_entity": "Harmonic visibility",
    "visibility_gain_gated": "Visibility +0.5pp gate",
    "validation_global_fixed": "Val global fixed",
}

TASK_LABELS = {
    "goal_cream_cheese_bowl": "Cream cheese -> bowl",
    "goal_top_drawer_bowl": "Bowl -> top drawer",
    "goal_wine_rack": "Wine bottle -> rack",
    "libero10_book_caddy": "Book -> caddy",
    "libero10_bowl_bottom_drawer": "Bowl -> bottom drawer",
    "libero10_mug_microwave": "Mug -> microwave",
    "object_cream_cheese_basket": "Cream cheese -> basket",
    "spatial_drawer_bowl_plate": "Drawer bowl -> plate",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.analysis.read_text(encoding="utf-8"))
    methods = payload["selector_methods"]
    summary = payload["selector_summary"]
    tasks = sorted(payload["task_summary"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    labels = [METHOD_LABELS.get(method, method) for method in methods]
    y = np.arange(len(methods))

    rates = np.asarray([100 * summary[method]["mean_repeat_success_rate"] for method in methods])
    lows = np.asarray([100 * summary[method]["success_rate_ci_low"] for method in methods])
    highs = np.asarray([100 * summary[method]["success_rate_ci_high"] for method in methods])
    axes[0].barh(y, rates, color="#3E7CB1")
    axes[0].errorbar(
        rates,
        y,
        xerr=np.vstack([rates - lows, highs - rates]),
        fmt="none",
        ecolor="#1F2A37",
        capsize=3,
    )
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 105)
    axes[0].set_xlabel("Three-noise success rate (%)")
    axes[0].set_title("A. Independent test success")
    axes[0].grid(axis="x", alpha=0.2)

    differences = np.asarray(
        [summary[method]["difference_from_canonical_pp"] for method in methods]
    )
    diff_lows = np.asarray([summary[method]["difference_ci_low_pp"] for method in methods])
    diff_highs = np.asarray([summary[method]["difference_ci_high_pp"] for method in methods])
    axes[1].axvline(0, color="#1F2A37", linewidth=1)
    axes[1].errorbar(
        differences,
        y,
        xerr=np.vstack([differences - diff_lows, diff_highs - differences]),
        fmt="o",
        color="#C55A6A",
        ecolor="#1F2A37",
        capsize=3,
    )
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Difference from canonical (pp)")
    axes[1].set_title("B. Paired source-group effect")
    axes[1].grid(axis="x", alpha=0.2)

    matrix = np.asarray(
        [
            [
                100
                * payload["task_summary"][task][method]["mean_repeat_success_rate"]
                for method in methods
            ]
            for task in tasks
        ]
    )
    image = axes[2].imshow(matrix, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    axes[2].set_xticks(np.arange(len(methods)), labels, rotation=45, ha="right")
    axes[2].set_yticks(
        np.arange(len(tasks)), [TASK_LABELS.get(task, task) for task in tasks]
    )
    axes[2].set_title("C. Task-level success (%)")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[2].text(
                column,
                row,
                f"{matrix[row, column]:.0f}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] >= 55 else "#1F2A37",
                fontsize=7,
            )
    figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.03)

    figure.suptitle(
        "Frozen view selectors on untouched source demonstrations",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        f"{payload['states']} states from {payload['source_groups']} source episodes; "
        f"{payload['expected_repeats']} policy-noise repeats per selector. "
        "Selectors were frozen before test outcomes.",
        ha="center",
        color="#44505E",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
