from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONTROL_COLOR = "#355C7D"
KYC_COLOR = "#E76F51"
GRID_COLOR = "#D9DEE3"


def _scope(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    matches = [row for row in payload["summaries"] if row["scope"] == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one summary for scope {name!r}")
    return matches[0]


def render(summary_path: Path, output_path: Path) -> None:
    payload = json.loads(summary_path.read_text())
    if int(payload.get("schema_version", 0)) < 3:
        raise ValueError("KYC gate summary schema version 3 or newer is required")

    supported = _scope(payload, "fully_supported")
    canonical = _scope(payload, "canonical")
    all_rows = _scope(payload, "all")
    success = supported["success"]
    per_seed = success["per_seed"]
    seed_labels = [str(row["seed"]) for row in per_seed] + ["Mean"]
    control = [100.0 * float(row["poseaug_control"]) for row in per_seed]
    kyc = [100.0 * float(row["kyc"]) for row in per_seed]
    control.append(100.0 * float(success["poseaug_control_mean"]))
    kyc.append(100.0 * float(success["kyc_mean"]))

    forest_specs = [
        ("Supported task success", supported["success"]),
        ("Supported transport", supported["transport_success"]),
        ("Supported progress", supported["progress"]),
        ("Canonical task success", canonical["success"]),
        ("All-view task success", all_rows["success"]),
    ]
    deltas = np.asarray(
        [100.0 * float(values["delta"]) for _, values in forest_specs],
        dtype=np.float64,
    )
    lows = np.asarray(
        [100.0 * float(values["ci95_low"]) for _, values in forest_specs],
        dtype=np.float64,
    )
    highs = np.asarray(
        [100.0 * float(values["ci95_high"]) for _, values in forest_specs],
        dtype=np.float64,
    )

    fig, (bars, forest) = plt.subplots(
        1,
        2,
        figsize=(12.0, 5.8),
        gridspec_kw={"width_ratios": [1.0, 1.15]},
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.85,
        bottom=0.19,
        wspace=0.45,
    )
    fig.suptitle(
        "KYC camera conditioning: fixed multi-state gate",
        fontsize=15,
        fontweight="semibold",
        y=0.965,
    )

    x = np.arange(len(seed_labels), dtype=np.float64)
    width = 0.36
    bars.bar(
        x - width / 2,
        control,
        width,
        color=CONTROL_COLOR,
        label="PoseAug-Control",
    )
    bars.bar(x + width / 2, kyc, width, color=KYC_COLOR, label="KYC")
    for index, value in enumerate(control):
        bars.text(
            index - width / 2,
            value + 1.1,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for index, value in enumerate(kyc):
        bars.text(
            index + width / 2,
            value + 1.1,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    bars.set_title("Fully-supported task success")
    bars.set_ylabel("Success (%)")
    bars.set_xlabel("Fine-tuning seed")
    bars.set_xticks(x, seed_labels)
    bars.set_ylim(0.0, max(max(control), max(kyc)) + 12.0)
    bars.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    bars.set_axisbelow(True)
    bars.legend(frameon=False, loc="upper right")
    bars.spines[["top", "right"]].set_visible(False)

    y = np.arange(len(forest_specs), dtype=np.float64)
    errors = np.vstack([deltas - lows, highs - deltas])
    forest.errorbar(
        deltas,
        y,
        xerr=errors,
        fmt="o",
        color=KYC_COLOR,
        ecolor="#3D3D3D",
        elinewidth=1.4,
        capsize=4,
        markersize=7,
    )
    forest.axvline(0.0, color="#111111", linewidth=1.0)
    forest.axvline(
        10.0,
        color="#6A994E",
        linewidth=1.0,
        linestyle="--",
    )
    for index, (delta, low, high) in enumerate(zip(deltas, lows, highs)):
        forest.text(
            max(high + 0.7, 1.0),
            index,
            f"{delta:+.1f} [{low:+.1f}, {high:+.1f}]",
            va="center",
            fontsize=8,
        )
    forest.set_yticks(y, [label for label, _ in forest_specs])
    forest.invert_yaxis()
    forest.set_xlabel("KYC - PoseAug-Control (percentage points, 95% CI)")
    forest.set_title("Crossed seed x state effects")
    forest.set_xlim(-15.0, 27.0)
    forest.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    forest.set_axisbelow(True)
    forest.text(
        10.4,
        -0.52,
        "+10 pp threshold",
        color="#4F772D",
        fontsize=8,
        va="center",
    )
    forest.spines[["top", "right"]].set_visible(False)

    gate_passed = bool(payload["incremental_camera_metadata_gate"]["passed"])
    fig.text(
        0.5,
        0.035,
        (
            f"Incremental camera-metadata gate: "
            f"{'PASS' if gate_passed else 'NOT PASSED'} | "
            "10,000 crossed bootstrap resamples; initial fully-supported "
            "observations only"
        ),
        ha="center",
        fontsize=9,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    fig.savefig(temporary, format="png", dpi=160, facecolor="white")
    plt.close(fig)
    os.replace(temporary, output_path)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the final cross-seed KYC camera gate summary"
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(args)


def main(args: Iterable[str] | None = None) -> None:
    parsed = parse_args(args)
    if parsed.output.exists():
        raise FileExistsError(f"refusing to overwrite plot: {parsed.output}")
    render(parsed.summary, parsed.output)
    print(json.dumps({"output": str(parsed.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
