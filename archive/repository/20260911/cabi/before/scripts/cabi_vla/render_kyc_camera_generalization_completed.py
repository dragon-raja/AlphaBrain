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
FIXED_COLOR = "#355C7D"
CUE_COLOR = "#5B8E7D"
GRID_COLOR = "#D9DEE3"
SEED_COLOR = "#777777"


def _load_complete(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("status", "complete") != "complete":
        raise ValueError(f"incomplete result: {path}")
    return payload


def _atomic_save(fig: Any, output: Path, *, dpi: int = 170) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite figure: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    fig.savefig(temporary, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    os.replace(temporary, output)


def render_main(
    *,
    stage_b1: Mapping[str, Any],
    stage_b2: Mapping[str, Any],
    official: Mapping[str, Any],
    output: Path,
) -> None:
    budgets = sorted(map(int, stage_b1["budget_results"]))
    b1_methods = {
        method: np.asarray(
            [
                100.0
                * float(
                    stage_b1["budget_results"][str(budget)]["primary"]["all"][
                        "methods"
                    ][method]["success"]
                )
                for budget in budgets
                if method
                in stage_b1["budget_results"][str(budget)]["primary"]["all"][
                    "methods"
                ]
            ]
        )
        for method in ("poseaug_control", "kyc", "poseaug_rgb")
    }
    rgb_budgets = [
        budget
        for budget in budgets
        if "poseaug_rgb"
        in stage_b1["budget_results"][str(budget)]["primary"]["all"]["methods"]
    ]

    official_test = official["pose_sets"]["test_cameras"]
    official_values = [
        100.0 * float(official_test["equal_seed_mean"]["image_success"]),
        100.0 * float(official_test["equal_seed_mean"]["kyc_success"]),
    ]
    official_delta = official_test["hierarchical_paired_bootstrap"]

    confirmed_budgets = sorted(map(int, stage_b2["budget_results"]))
    confirmed_control = [
        100.0
        * float(
            stage_b2["budget_results"][str(budget)]["equal_seed_mean"][
                "poseaug_control"
            ]["success"]
        )
        for budget in confirmed_budgets
    ]
    confirmed_kyc = [
        100.0
        * float(
            stage_b2["budget_results"][str(budget)]["equal_seed_mean"]["kyc"][
                "success"
            ]
        )
        for budget in confirmed_budgets
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.0))
    positive, scaling, confirmed, forest = axes.flat
    fig.subplots_adjust(
        left=0.07,
        right=0.975,
        top=0.84,
        bottom=0.11,
        wspace=0.29,
        hspace=0.43,
    )
    fig.suptitle(
        "KYC camera conditioning: completed evidence before the factorial study",
        fontsize=16,
        fontweight="semibold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.915,
        (
            "Released ACT and Pi0.5 are separate benchmarks; their absolute "
            "success rates are not directly comparable"
        ),
        ha="center",
        fontsize=9.5,
        color="#444444",
    )

    bars = positive.bar(
        ["Image", "KYC"],
        official_values,
        color=[CONTROL_COLOR, KYC_COLOR],
        width=0.58,
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
        max(official_values) + 10.0,
        (
            f"Gain {100.0 * float(official_delta['delta']):+.1f} pp\n"
            f"95% CI [{100.0 * float(official_delta['ci95_low']):+.1f}, "
            f"{100.0 * float(official_delta['ci95_high']):+.1f}]"
        ),
        ha="center",
        fontsize=9.5,
    )
    positive.set_ylim(0.0, 87.0)
    positive.set_ylabel("Task success (%)")
    positive.set_title("(a) Released ACT positive control\n3 training seeds")
    positive.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    positive.set_axisbelow(True)
    positive.spines[["top", "right"]].set_visible(False)

    scaling.plot(
        budgets,
        b1_methods["poseaug_control"],
        marker="o",
        linewidth=2.0,
        color=CONTROL_COLOR,
        label="PoseAug-Control",
    )
    scaling.plot(
        budgets,
        b1_methods["kyc"],
        marker="o",
        linewidth=2.0,
        color=KYC_COLOR,
        label="KYC",
    )
    scaling.plot(
        rgb_budgets,
        b1_methods["poseaug_rgb"],
        marker="o",
        linestyle="--",
        linewidth=1.6,
        color=RGB_COLOR,
        label="PoseAug-RGB",
    )
    scaling.set_xscale("log")
    scaling.set_xticks(budgets, [str(budget) for budget in budgets])
    scaling.set_ylim(20.0, 50.0)
    scaling.set_xlabel("Training camera catalog size")
    scaling.set_ylabel("Task success (%)")
    scaling.set_title("(b) Pi0.5 exploratory scaling\nseed 41")
    scaling.grid(color=GRID_COLOR, linewidth=0.8)
    scaling.set_axisbelow(True)
    scaling.legend(frameon=False, fontsize=8.5)
    scaling.spines[["top", "right"]].set_visible(False)

    x = np.arange(len(confirmed_budgets), dtype=np.float64)
    width = 0.34
    control_bars = confirmed.bar(
        x - width / 2,
        confirmed_control,
        width,
        color=CONTROL_COLOR,
        label="PoseAug-Control",
    )
    kyc_bars = confirmed.bar(
        x + width / 2,
        confirmed_kyc,
        width,
        color=KYC_COLOR,
        label="KYC",
    )
    for bar in [*control_bars, *kyc_bars]:
        value = float(bar.get_height())
        confirmed.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.9,
            f"{value:.1f}",
            ha="center",
            fontsize=8.5,
        )
    confirmed.set_xticks(x, [f"n={budget}" for budget in confirmed_budgets])
    confirmed.set_ylim(0.0, 46.0)
    confirmed.set_ylabel("Equal-seed mean success (%)")
    confirmed.set_title("(c) Pi0.5 confirmed scaling\nseeds 41, 42, 43")
    confirmed.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    confirmed.set_axisbelow(True)
    confirmed.legend(frameon=False, fontsize=8.5)
    confirmed.spines[["top", "right"]].set_visible(False)

    forest_rows = []
    for budget in confirmed_budgets:
        result = stage_b2["budget_results"][str(budget)]
        for training_seed in map(str, stage_b2["training_seeds"]):
            effect = result["per_seed"][training_seed]["kyc_minus_control"][
                "success"
            ]
            forest_rows.append(
                {
                    "label": f"n={budget} / seed {training_seed}",
                    "delta": 100.0 * float(effect["delta"]),
                    "low": 100.0 * float(effect["ci95_low"]),
                    "high": 100.0 * float(effect["ci95_high"]),
                    "confirmed": False,
                }
            )
        effect = result["hierarchical_kyc_minus_control"]["success"]
        forest_rows.append(
            {
                "label": f"n={budget} / all seeds",
                "delta": 100.0 * float(effect["delta"]),
                "low": 100.0 * float(effect["ci95_low"]),
                "high": 100.0 * float(effect["ci95_high"]),
                "confirmed": True,
            }
        )
    y_positions = np.asarray([8.0, 7.0, 6.0, 5.0, 3.2, 2.2, 1.2, 0.2])
    for y_position, row in zip(y_positions, forest_rows):
        color = KYC_COLOR if row["confirmed"] else SEED_COLOR
        forest.errorbar(
            row["delta"],
            y_position,
            xerr=np.asarray(
                [
                    [row["delta"] - row["low"]],
                    [row["high"] - row["delta"]],
                ]
            ),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.5 if row["confirmed"] else 1.0,
            capsize=4,
            markersize=7 if row["confirmed"] else 5,
        )
    forest.axvline(0.0, color="#111111", linewidth=1.0)
    forest.axvline(10.0, color="#6A994E", linewidth=1.0, linestyle="--")
    forest.set_yticks(y_positions, [row["label"] for row in forest_rows])
    forest.set_xlim(-25.0, 18.0)
    forest.set_ylim(-0.7, 8.7)
    forest.set_xlabel("KYC - Control (percentage points, 95% CI)")
    forest.set_title("(d) Pi0.5 paired effects\n10 snapshot groups")
    forest.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    forest.set_axisbelow(True)
    forest.spines[["top", "right"]].set_visible(False)
    for y_position, row in zip(y_positions, forest_rows):
        if row["confirmed"]:
            forest.text(
                row["delta"] + (0.8 if row["delta"] >= 0 else -0.8),
                y_position,
                f"{row['delta']:+.1f}",
                ha="left" if row["delta"] >= 0 else "right",
                va="center",
                fontsize=8.5,
                fontweight="semibold",
                color=KYC_COLOR,
            )

    fig.text(
        0.5,
        0.035,
        (
            "Pi0.5 primary stratum: fully supported and inside training camera "
            "support. Confirmed intervals use crossed training-seed x "
            "snapshot-group bootstrap."
        ),
        ha="center",
        fontsize=8.5,
    )
    _atomic_save(fig, output)


def render_leakage(
    *,
    leakage: Mapping[str, Any],
    output: Path,
) -> None:
    metrics = leakage["metrics"]
    conditions = ["Fixed scene", "Cue randomized"]
    colors = [FIXED_COLOR, CUE_COLOR]
    classification = [
        100.0 * float(metrics["fixed"]["accuracy"]),
        100.0 * float(metrics["cue_randomized"]["accuracy"]),
    ]
    chance = 100.0 * float(metrics["fixed"]["chance_accuracy"])
    axes_names = ["Azimuth", "Elevation", "Radius"]
    fixed_r2 = [
        float(metrics["fixed"]["r2"][key])
        for key in ("azimuth_deg", "elevation_deg", "radius_scale")
    ]
    cue_r2 = [
        float(metrics["cue_randomized"]["r2"][key])
        for key in ("azimuth_deg", "elevation_deg", "radius_scale")
    ]

    fig, (classification_axis, regression_axis) = plt.subplots(
        1,
        2,
        figsize=(11.8, 4.9),
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.975,
        top=0.78,
        bottom=0.25,
        wspace=0.3,
    )
    fig.suptitle(
        "Can static scene pixels reveal the external camera pose?",
        fontsize=15,
        fontweight="semibold",
        y=0.95,
    )
    fig.text(
        0.5,
        0.855,
        "Background-only linear probe; held-out physical states and scene seeds",
        ha="center",
        fontsize=9,
        color="#444444",
    )

    bars = classification_axis.bar(
        conditions,
        classification,
        color=colors,
        width=0.58,
    )
    classification_axis.axhline(
        chance,
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        label=f"Chance ({chance:.1f}%)",
    )
    for bar, value in zip(bars, classification):
        classification_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.0,
            f"{value:.1f}%",
            ha="center",
            fontsize=10,
        )
    classification_axis.text(
        0.5,
        77.0,
        (
            "Pose-classification advantage\n"
            f"reduced {100.0 * float(leakage['classification_advantage_reduction']):.1f}%"
        ),
        ha="center",
        fontsize=9,
    )
    classification_axis.set_ylim(0.0, 108.0)
    classification_axis.set_ylabel("13-way camera-pose accuracy (%)")
    classification_axis.set_title("(a) Discrete pose leakage")
    classification_axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    classification_axis.set_axisbelow(True)
    classification_axis.legend(frameon=False, fontsize=8)
    classification_axis.spines[["top", "right"]].set_visible(False)

    x = np.arange(len(axes_names), dtype=np.float64)
    width = 0.34
    fixed_bars = regression_axis.bar(
        x - width / 2,
        fixed_r2,
        width,
        color=FIXED_COLOR,
        label="Fixed scene",
    )
    cue_bars = regression_axis.bar(
        x + width / 2,
        cue_r2,
        width,
        color=CUE_COLOR,
        label="Cue randomized",
    )
    for bar in [*fixed_bars, *cue_bars]:
        value = float(bar.get_height())
        regression_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.2f}",
            ha="center",
            fontsize=8.5,
        )
    regression_axis.set_xticks(x, axes_names)
    regression_axis.set_ylim(0.0, 1.12)
    regression_axis.set_ylabel("Held-out regression R2")
    regression_axis.set_title("(b) Continuous pose leakage")
    regression_axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    regression_axis.set_axisbelow(True)
    regression_axis.legend(
        frameon=False,
        fontsize=8.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
    )
    regression_axis.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5,
        0.055,
        (
            f"{int(leakage['render_count']):,} renders; physical-state max "
            f"change = {float(leakage['physics_state_max_abs']):.1f}; "
            f"cue-suppression validity gate = {'PASS' if leakage['gate_passed'] else 'FAIL'}."
        ),
        ha="center",
        fontsize=8.5,
    )
    _atomic_save(fig, output)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render completed KYC evidence for a research report"
    )
    parser.add_argument("--stage-b1-summary", type=Path, required=True)
    parser.add_argument("--stage-b2-summary", type=Path, required=True)
    parser.add_argument("--official-summary", type=Path, required=True)
    parser.add_argument("--leakage-summary", type=Path, required=True)
    parser.add_argument("--main-output", type=Path, required=True)
    parser.add_argument("--leakage-output", type=Path, required=True)
    return parser.parse_args(args)


def main(args: Iterable[str] | None = None) -> None:
    parsed = parse_args(args)
    stage_b1 = _load_complete(parsed.stage_b1_summary)
    stage_b2 = _load_complete(parsed.stage_b2_summary)
    official = _load_complete(parsed.official_summary)
    leakage = json.loads(parsed.leakage_summary.read_text())
    render_main(
        stage_b1=stage_b1,
        stage_b2=stage_b2,
        official=official,
        output=parsed.main_output,
    )
    render_leakage(
        leakage=leakage,
        output=parsed.leakage_output,
    )
    print(
        json.dumps(
            {
                "main_output": str(parsed.main_output),
                "leakage_output": str(parsed.leakage_output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
