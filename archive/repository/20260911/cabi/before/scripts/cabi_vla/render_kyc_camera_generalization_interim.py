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


CONTROL_COLOR = "#355C7D"
KYC_COLOR = "#E76F51"
RGB_COLOR = "#5B8E7D"
GRID_COLOR = "#D9DEE3"


def render(
    scaling_path: Path,
    official_path: Path,
    output_path: Path,
) -> None:
    scaling: Mapping[str, Any] = json.loads(scaling_path.read_text())
    official: Mapping[str, Any] = json.loads(official_path.read_text())
    if scaling.get("status") != "complete":
        raise ValueError("Stage B1 scaling summary is incomplete")
    if official.get("status") != "complete":
        raise ValueError("official ACT summary is incomplete")

    budgets = sorted(map(int, scaling["budget_results"]))
    control = np.asarray(
        [
            100.0
            * float(
                scaling["budget_results"][str(budget)]["primary"]["all"][
                    "methods"
                ]["poseaug_control"]["success"]
            )
            for budget in budgets
        ]
    )
    kyc = np.asarray(
        [
            100.0
            * float(
                scaling["budget_results"][str(budget)]["primary"]["all"][
                    "methods"
                ]["kyc"]["success"]
            )
            for budget in budgets
        ]
    )
    rgb = {
        budget: 100.0
        * float(
            scaling["budget_results"][str(budget)]["primary"]["all"][
                "methods"
            ]["poseaug_rgb"]["success"]
        )
        for budget in budgets
        if "poseaug_rgb"
        in scaling["budget_results"][str(budget)]["primary"]["all"]["methods"]
    }
    delta_rows = [
        scaling["budget_results"][str(budget)]["primary"]["all"][
            "comparisons"
        ]["kyc_minus_poseaug_control"]["success"]
        for budget in budgets
    ]
    deltas = 100.0 * np.asarray(
        [float(row["delta"]) for row in delta_rows]
    )
    lows = 100.0 * np.asarray(
        [float(row["ci95_low"]) for row in delta_rows]
    )
    highs = 100.0 * np.asarray(
        [float(row["ci95_high"]) for row in delta_rows]
    )

    official_test = official["pose_sets"]["test_cameras"]
    official_values = [
        100.0 * float(official_test["equal_seed_mean"]["image_success"]),
        100.0 * float(official_test["equal_seed_mean"]["kyc_success"]),
    ]
    official_delta = official_test["hierarchical_paired_bootstrap"]

    fig, (positive, curve, forest) = plt.subplots(
        1,
        3,
        figsize=(15.0, 5.2),
        gridspec_kw={"width_ratios": [0.78, 1.15, 1.1]},
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.74,
        bottom=0.22,
        wspace=0.38,
    )
    fig.suptitle(
        "KYC camera conditioning: released-code control and Pi0.5 transfer",
        fontsize=15,
        fontweight="semibold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.885,
        "Separate benchmarks: absolute ACT and Pi0.5 success rates are not directly comparable",
        ha="center",
        fontsize=9,
        color="#444444",
    )

    bars = positive.bar(
        ["Image", "KYC"],
        official_values,
        color=[CONTROL_COLOR, KYC_COLOR],
        width=0.62,
    )
    for bar, value in zip(bars, official_values):
        positive.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.1f}%",
            ha="center",
            fontsize=10,
        )
    positive.text(
        0.5,
        max(official_values) + 9.0,
        (
            f"Paired gain {100.0 * float(official_delta['delta']):+.1f} pp\n"
            f"95% CI [{100.0 * float(official_delta['ci95_low']):+.1f}, "
            f"{100.0 * float(official_delta['ci95_high']):+.1f}]"
        ),
        ha="center",
        fontsize=9,
    )
    positive.set_ylim(0.0, max(official_values) + 22.0)
    positive.set_ylabel("Task success (%)")
    positive.set_title("Official ACT Lift randomized\n3 training seeds")
    positive.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    positive.set_axisbelow(True)
    positive.spines[["top", "right"]].set_visible(False)

    curve.plot(
        budgets,
        control,
        marker="o",
        linewidth=2.0,
        color=CONTROL_COLOR,
        label="PoseAug-Control",
    )
    curve.plot(
        budgets,
        kyc,
        marker="o",
        linewidth=2.0,
        color=KYC_COLOR,
        label="KYC",
    )
    if rgb:
        rgb_budgets = sorted(rgb)
        curve.plot(
            rgb_budgets,
            [rgb[budget] for budget in rgb_budgets],
            marker="o",
            linestyle="--",
            linewidth=1.6,
            color=RGB_COLOR,
            label="PoseAug-RGB",
        )
    curve.set_xscale("log")
    curve.set_xticks(budgets, [str(budget) for budget in budgets])
    curve.set_ylim(0.0, max(max(control), max(kyc)) + 10.0)
    curve.set_xlabel("Training camera catalog size")
    curve.set_ylabel("Task success (%)")
    curve.set_title("Pi0.5 Stage B1 scaling\nseed 41")
    curve.grid(color=GRID_COLOR, linewidth=0.8)
    curve.set_axisbelow(True)
    curve.legend(frameon=False, fontsize=8)
    curve.spines[["top", "right"]].set_visible(False)

    y = np.arange(len(budgets), dtype=np.float64)
    forest.errorbar(
        deltas,
        y,
        xerr=np.vstack([deltas - lows, highs - deltas]),
        fmt="o",
        color=KYC_COLOR,
        ecolor="#3D3D3D",
        elinewidth=1.4,
        capsize=4,
        markersize=7,
    )
    forest.axvline(0.0, color="#111111", linewidth=1.0)
    forest.axvline(10.0, color="#6A994E", linewidth=1.0, linestyle="--")
    forest.set_yticks(y, [f"n={budget}" for budget in budgets])
    forest.invert_yaxis()
    forest.set_xlabel("KYC - Control (percentage points, 95% CI)")
    forest.set_title("Pi0.5 paired effects\n10 snapshot groups")
    forest.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    forest.set_axisbelow(True)
    forest.spines[["top", "right"]].set_visible(False)
    lower = min(-15.0, float(np.min(lows)) - 3.0)
    upper = max(15.0, float(np.max(highs)) + 3.0)
    forest.set_xlim(lower, upper)
    for index, (delta, low, high) in enumerate(zip(deltas, lows, highs)):
        forest.text(
            upper - 0.5,
            index,
            f"{delta:+.1f} [{low:+.1f}, {high:+.1f}]",
            ha="right",
            va="center",
            fontsize=8,
        )

    fig.text(
        0.5,
        0.055,
        (
            "Pi0.5: 520 paired rollouts per model; primary stratum is "
            "fully-supported and inside training camera support. "
            "Stage B1 is exploratory until seeds 42 and 43 confirm it."
        ),
        ha="center",
        fontsize=8.5,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    fig.savefig(temporary, format="png", dpi=170, facecolor="white")
    plt.close(fig)
    os.replace(temporary, output_path)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render official ACT and Pi0.5 KYC interim results"
    )
    parser.add_argument("--scaling-summary", type=Path, required=True)
    parser.add_argument("--official-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(args)


def main(args: Iterable[str] | None = None) -> None:
    parsed = parse_args(args)
    if parsed.output.exists():
        raise FileExistsError(f"refusing to overwrite plot: {parsed.output}")
    render(
        parsed.scaling_summary,
        parsed.official_summary,
        parsed.output,
    )
    print(json.dumps({"output": str(parsed.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
