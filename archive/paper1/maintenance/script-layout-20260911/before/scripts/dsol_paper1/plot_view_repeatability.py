#!/usr/bin/env python3
"""Plot multi-seed stability of discovery-selected camera views."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CATEGORY_LABELS = {
    "canonical_success_mixed": "Easy / view-insensitive",
    "canonical_failure_sparse": "Sparse single-seed rescue",
    "canonical_success_harm": "View-harm state",
    "canonical_failure_broad": "Broad single-seed rescue",
    "canonical_success_broad": "Canonical success / broad support",
    "no_discovery_success": "No single-seed success",
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
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.analysis.read_text(encoding="utf-8"))
    with args.candidates.open(encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    expanded = summary["states"] > 8
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 6.2 if expanded else 4.8))

    state_rows = summary["state_rows"]
    if expanded:
        grouped = defaultdict(list)
        for row in state_rows:
            grouped[row["task_id"]].append(row)
        plot_rows = []
        for task_id, values in sorted(grouped.items()):
            plot_rows.append(
                {
                    "task_id": task_id,
                    "canonical_repeat_success_rate": float(
                        np.mean(
                            [row["canonical_repeat_success_rate"] for row in values]
                        )
                    ),
                    "visibility_repeat_success_rate": float(
                        np.mean(
                            [row["visibility_repeat_success_rate"] for row in values]
                        )
                    ),
                    "best_shortlist_repeat_success_rate": float(
                        np.mean(
                            [
                                row["best_shortlist_repeat_success_rate"]
                                for row in values
                            ]
                        )
                    ),
                    "stable_rescue_count": sum(
                        row["canonical_repeat_success_rate"] < 2 / 3
                        and row["best_shortlist_repeat_success_rate"] >= 2 / 3
                        for row in values
                    ),
                    "visibility_rescue_count": sum(
                        row["canonical_repeat_success_rate"] < 2 / 3
                        and row["visibility_repeat_success_rate"] >= 2 / 3
                        for row in values
                    ),
                    "visibility_harm_count": sum(
                        row["canonical_repeat_success_rate"] >= 2 / 3
                        and row["visibility_repeat_success_rate"] < 2 / 3
                        for row in values
                    ),
                }
            )
    else:
        plot_rows = state_rows
    labels = [
        TASK_LABELS.get(row["task_id"], row["task_id"])
        if expanded
        else CATEGORY_LABELS[row["category"]]
        for row in plot_rows
    ]
    y = np.arange(len(labels))
    width = 0.23
    axes[0].barh(
        y - width,
        [100 * row["canonical_repeat_success_rate"] for row in plot_rows],
        height=width,
        label="Canonical",
        color="#3E7CB1",
    )
    axes[0].barh(
        y,
        [100 * row["visibility_repeat_success_rate"] for row in plot_rows],
        height=width,
        label="Visibility top-1",
        color="#3FA37C",
    )
    axes[0].barh(
        y + width,
        [100 * row["best_shortlist_repeat_success_rate"] for row in plot_rows],
        height=width,
        label="Best of 8 (post hoc)",
        color="#C55A6A",
    )
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 105)
    axes[0].set_xlabel("Success across 3 independent noise draws (%)")
    axes[0].set_title("A. Repeatability by task" if expanded else "A. Repeatability by state type")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].grid(axis="x", alpha=0.2)

    if expanded:
        axes[1].barh(
            y - width,
            [row["stable_rescue_count"] for row in plot_rows],
            height=width,
            color="#C55A6A",
            label="Best-of-8 rescue",
        )
        axes[1].barh(
            y,
            [row["visibility_rescue_count"] for row in plot_rows],
            height=width,
            color="#3FA37C",
            label="Visibility rescue",
        )
        axes[1].barh(
            y + width,
            [row["visibility_harm_count"] for row in plot_rows],
            height=width,
            color="#E6B566",
            label="Visibility harm",
        )
        axes[1].set_xlim(0, 4.3)
        axes[1].set_xlabel("States (4 per task)")
        axes[1].set_title("B. Stable rescue and harm (2-of-3)")
        axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    else:
        transition = defaultdict(Counter)
        for row in candidates:
            transition[row["category"]][int(row["repeat_successes"])] += 1
        bottoms = np.zeros(len(labels))
        colors = ["#D7DEE8", "#E6B566", "#67A9CF", "#238B6B"]
        for successes in range(4):
            values = [transition[row["category"]][successes] for row in plot_rows]
            axes[1].barh(
                y,
                values,
                left=bottoms,
                color=colors[successes],
                label=f"{successes}/3",
            )
            bottoms += np.asarray(values)
        axes[1].set_xlim(0, max(8, float(max(bottoms, default=8))))
        axes[1].set_xlabel("Shortlisted candidates")
        axes[1].set_title("B. Candidate stability distribution")
        axes[1].legend(title="Repeat wins", frameon=False, fontsize=8, ncol=2)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.2)

    for discovery, marker, color, label in (
        (0, "o", "#657786", "Discovery failure"),
        (1, "s", "#C55A6A", "Discovery success"),
    ):
        subset = [row for row in candidates if int(row["discovery_success"]) == discovery]
        jitter = np.linspace(-0.06, 0.06, len(subset)) if subset else []
        axes[2].scatter(
            np.asarray([discovery] * len(subset)) + jitter,
            [100 * float(row["repeat_success_rate"]) for row in subset],
            marker=marker,
            color=color,
            alpha=0.8,
            label=label,
        )
    axes[2].set_xticks([0, 1], ["Failed", "Succeeded"])
    axes[2].set_ylim(-5, 105)
    axes[2].set_ylabel("Repeat success rate (%)")
    axes[2].set_xlabel("Outcome in the original single-noise discovery")
    axes[2].set_title("C. Single-run outcome vs repeatability")
    axes[2].grid(axis="y", alpha=0.2)

    figure.suptitle("Expanded dense-view repeatability audit", fontsize=15, fontweight="bold")
    figure.text(
        0.5,
        0.012,
        f"Exploratory shortlist: {summary['states']} physical states x 8 views x "
        f"{summary['expected_noise_draws_per_candidate']} new noise draws. "
        "Best-of-8 remains post-hoc and is not a deployable selector.",
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
