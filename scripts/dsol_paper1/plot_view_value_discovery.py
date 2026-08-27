#!/usr/bin/env python3
"""Plot compact diagnostics from view-value discovery analysis JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text(encoding="utf-8"))

    curves = data["candidate_budget_curves"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    styles = {
        "random_pool_oracle": ("Random k-view pool", "#8A94A3", "--"),
        "visibility_mean": ("Visibility-ranked top-k", "#2D7FB8", "-"),
        "accel_ensemble": ("Accel-ranked top-k", "#C55A6A", "-"),
    }
    for key, (label, color, linestyle) in styles.items():
        rows = curves[key]
        axes[0].plot(
            [row["budget"] for row in rows],
            [100 * row["state_hit_rate"] for row in rows],
            marker="o",
            linewidth=2.2,
            color=color,
            linestyle=linestyle,
            label=label,
        )
    axes[0].set_title("Does top-k contain any successful view?")
    axes[0].set_xlabel("Candidate budget k")
    axes[0].set_ylabel("State hit rate (%)")
    axes[0].set_xticks([row["budget"] for row in curves["visibility_mean"]])
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    diagnostics = data["score_diagnostics"]
    names = ["Visibility", "Min-entity", "Accel"]
    keys = ["visibility_mean", "visibility_min_entity", "accel_ensemble"]
    top1 = [100 * diagnostics[key]["top1_success_rate"] for key in keys]
    auc = [100 * diagnostics[key]["state_conditional_auc"] for key in keys]
    x = range(len(names))
    axes[1].bar([value - 0.18 for value in x], top1, width=0.36, color="#3C87B9", label="Top-1 success")
    axes[1].bar([value + 0.18 for value in x], auc, width=0.36, color="#D39B3A", label="Within-state AUC")
    axes[1].axhline(50, color="#505864", linewidth=1, linestyle="--")
    axes[1].set_title("Current scores barely identify behavior-positive views")
    axes[1].set_xticks(list(x), names)
    axes[1].set_ylabel("Percent (%)")
    axes[1].set_ylim(0, 85)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    for index, value in enumerate(top1):
        axes[1].text(index - 0.18, value + 1.5, f"{value:.1f}", ha="center", fontsize=9)
    for index, value in enumerate(auc):
        axes[1].text(index + 0.18, value + 1.5, f"{value:.1f}", ha="center", fontsize=9)

    fig.suptitle(
        f"A97 canonical-failure discovery: {data['states']} states, "
        f"Oracle@97={100 * data['oracle_at_all_success_rate']:.1f}%",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
