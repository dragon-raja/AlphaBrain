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
BASE_COLOR = "#8C8C8C"
CAPACITY_COLOR = "#9C6ADE"
GRID_COLOR = "#D9DEE3"


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text())


def _atomic_save(fig: Any, output: Path, *, dpi: int = 170) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite figure: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    fig.savefig(temporary, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    os.replace(temporary, output)


def _initial_joint_frame(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        frames = np.asarray(archive["frames"])
    if frames.ndim != 4 or frames.shape[1:] != (224, 448, 3):
        raise ValueError(f"unexpected paired frame shape in {path}: {frames.shape}")
    frame = frames[0]
    return frame[:, :224], frame[:, 224:]


def render_camera_setup(*, sweep_root: Path, output: Path) -> None:
    frames_root = sweep_root / "frames"
    names = [
        ("baseline", "Default external view", "agent"),
        ("baseline", "Eye-in-hand wrist view", "wrist"),
        ("az_m30", "Azimuth -30 deg", "agent"),
        ("az_p30", "Azimuth +30 deg", "agent"),
        ("el_m12", "Elevation -12 deg", "agent"),
        ("el_p12", "Elevation +12 deg", "agent"),
        ("rad_085", "Distance 0.85x", "agent"),
        ("rad_115", "Distance 1.15x", "agent"),
    ]
    images = []
    for pose, title, view in names:
        agent, wrist = _initial_joint_frame(
            frames_root / f"{pose}--red-left--state-00--k3.npz"
        )
        images.append((wrist if view == "wrist" else agent, title))

    fig, axes = plt.subplots(2, 4, figsize=(14.8, 7.3))
    fig.subplots_adjust(
        left=0.025,
        right=0.985,
        top=0.86,
        bottom=0.06,
        wspace=0.06,
        hspace=0.22,
    )
    fig.suptitle(
        "Actual LIBERO observations used in the camera study",
        fontsize=16,
        fontweight="semibold",
        y=0.96,
    )
    fig.text(
        0.5,
        0.905,
        (
            "Only the external agent camera is perturbed; the wrist image is "
            "unchanged by that intervention"
        ),
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    for axis, (image, title) in zip(axes.flat, images):
        axis.imshow(image)
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.8)
    _atomic_save(fig, output)


def _percentage(success: list[int]) -> float:
    numerator, denominator = map(float, success)
    return 100.0 * numerator / denominator


def render_evidence_story(
    *,
    viewpoint: Mapping[str, Any],
    kyc_study: Mapping[str, Any],
    stage_b2: Mapping[str, Any],
    official: Mapping[str, Any],
    output: Path,
) -> None:
    direct_labels = ["Default\n1.00x", "Closer\n0.925x", "Farther\n1.075x"]
    selected = viewpoint["selected_states0to4"]
    direct_values = [
        _percentage(selected["baseline_success"]),
        _percentage(selected["rad_0925_success"]),
        _percentage(selected["rad_1075_success"]),
    ]

    method_keys = [
        "base",
        "pm_fixed",
        "poseaug_rgb",
        "poseaug_control",
        "kyc",
    ]
    method_labels = [
        "Base",
        "Fixed +\nbranch",
        "PoseAug\nRGB",
        "PoseAug\nControl",
        "KYC",
    ]
    method_values = [
        100.0 * float(kyc_study["seed41_context_success"][key])
        for key in method_keys
    ]

    official_test = official["pose_sets"]["test_cameras"]
    official_values = [
        100.0 * float(official_test["equal_seed_mean"]["image_success"]),
        100.0 * float(official_test["equal_seed_mean"]["kyc_success"]),
    ]
    official_delta = official_test["hierarchical_paired_bootstrap"]

    primary = kyc_study["primary_fully_supported"]["success"]
    effect_rows = [
        {
            "label": "Pi0.5 full gate",
            "delta": 100.0 * float(primary["delta"]),
            "low": 100.0 * float(primary["ci95"][0]),
            "high": 100.0 * float(primary["ci95"][1]),
        }
    ]
    for budget in (10, 45):
        effect = stage_b2["budget_results"][str(budget)][
            "hierarchical_kyc_minus_control"
        ]["success"]
        effect_rows.append(
            {
                "label": f"Pi0.5 n={budget}",
                "delta": 100.0 * float(effect["delta"]),
                "low": 100.0 * float(effect["ci95_low"]),
                "high": 100.0 * float(effect["ci95_high"]),
            }
        )

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.0))
    direct_axis, augmentation_axis, official_axis, effect_axis = axes.flat
    fig.subplots_adjust(
        left=0.07,
        right=0.975,
        top=0.85,
        bottom=0.11,
        wspace=0.28,
        hspace=0.43,
    )
    fig.suptitle(
        "Camera robustness: the completed evidence in one causal sequence",
        fontsize=16,
        fontweight="semibold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.915,
        (
            "Different panels answer different questions; absolute success "
            "rates should not be compared across benchmarks"
        ),
        ha="center",
        fontsize=9.5,
        color="#444444",
    )

    bars = direct_axis.bar(
        direct_labels,
        direct_values,
        color=[CONTROL_COLOR, "#D88C62", "#C9A227"],
        width=0.62,
    )
    for bar, value in zip(bars, direct_values):
        direct_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.0,
            f"{value:.0f}%",
            ha="center",
            fontsize=10,
        )
    direct_axis.set_ylim(0.0, 88.0)
    direct_axis.set_ylabel("Closed-loop task success (%)")
    direct_axis.set_title(
        "(a) Same Pi0.5 weights, only camera distance changes\n"
        "4 tasks x states 0-4"
    )
    direct_axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    direct_axis.set_axisbelow(True)
    direct_axis.spines[["top", "right"]].set_visible(False)

    method_colors = [
        BASE_COLOR,
        CAPACITY_COLOR,
        RGB_COLOR,
        CONTROL_COLOR,
        KYC_COLOR,
    ]
    bars = augmentation_axis.bar(
        method_labels,
        method_values,
        color=method_colors,
        width=0.7,
    )
    for bar, value in zip(bars, method_values):
        augmentation_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.1f}",
            ha="center",
            fontsize=8.5,
        )
    augmentation_axis.set_ylim(0.0, 56.0)
    augmentation_axis.set_ylabel("Fully-supported success (%)")
    augmentation_axis.set_title(
        "(b) Which training ingredient helps?\nPi0.5 seed 41 context"
    )
    augmentation_axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    augmentation_axis.set_axisbelow(True)
    augmentation_axis.spines[["top", "right"]].set_visible(False)

    bars = official_axis.bar(
        ["Image", "KYC"],
        official_values,
        color=[CONTROL_COLOR, KYC_COLOR],
        width=0.58,
    )
    for bar, value in zip(bars, official_values):
        official_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.1f}%",
            ha="center",
            fontsize=10,
        )
    official_axis.text(
        0.5,
        max(official_values) + 9.0,
        (
            f"Gain {100.0 * float(official_delta['delta']):+.1f} pp; "
            f"95% CI [{100.0 * float(official_delta['ci95_low']):+.1f}, "
            f"{100.0 * float(official_delta['ci95_high']):+.1f}]"
        ),
        ha="center",
        fontsize=9.2,
    )
    official_axis.set_ylim(0.0, 86.0)
    official_axis.set_ylabel("Task success (%)")
    official_axis.set_title(
        "(c) Released ACT positive control\nheld-out test cameras, 3 seeds"
    )
    official_axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    official_axis.set_axisbelow(True)
    official_axis.spines[["top", "right"]].set_visible(False)

    y_positions = np.arange(len(effect_rows), dtype=np.float64)[::-1]
    for y_position, row in zip(y_positions, effect_rows):
        effect_axis.errorbar(
            row["delta"],
            y_position,
            xerr=np.asarray(
                [
                    [row["delta"] - row["low"]],
                    [row["high"] - row["delta"]],
                ]
            ),
            fmt="o",
            color=KYC_COLOR,
            ecolor=KYC_COLOR,
            capsize=5,
            elinewidth=1.6,
            markersize=7,
        )
        effect_axis.text(
            row["high"] + 0.5,
            y_position,
            (
                f"{row['delta']:+.1f} "
                f"[{row['low']:+.1f}, {row['high']:+.1f}]"
            ),
            va="center",
            fontsize=8.5,
        )
    effect_axis.axvline(0.0, color="#111111", linewidth=1.0)
    effect_axis.axvline(10.0, color="#6A994E", linewidth=1.0, linestyle="--")
    effect_axis.set_yticks(
        y_positions,
        [row["label"] for row in effect_rows],
    )
    effect_axis.set_xlim(-16.0, 17.0)
    effect_axis.set_ylim(-0.75, len(effect_rows) - 0.25)
    effect_axis.set_xlabel("KYC - matched Control (percentage points)")
    effect_axis.set_title(
        "(d) Incremental value of measured camera geometry\n"
        "dot = estimate; line = 95% CI"
    )
    effect_axis.text(
        -15.5,
        len(effect_rows) - 0.48,
        "Control better",
        ha="left",
        fontsize=8,
        color="#555555",
    )
    effect_axis.text(
        16.5,
        len(effect_rows) - 0.48,
        "KYC better",
        ha="right",
        fontsize=8,
        color="#555555",
    )
    effect_axis.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    effect_axis.set_axisbelow(True)
    effect_axis.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.5,
        0.035,
        (
            "Panel (b) is a seed-41 mechanism comparison. Panel (d) uses "
            "three-seed, snapshot-group-level paired statistics."
        ),
        ha="center",
        fontsize=8.5,
    )
    _atomic_save(fig, output)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the consolidated camera-robustness report figures"
    )
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--viewpoint-summary", type=Path, required=True)
    parser.add_argument("--kyc-study-summary", type=Path, required=True)
    parser.add_argument("--stage-b2-summary", type=Path, required=True)
    parser.add_argument("--official-summary", type=Path, required=True)
    parser.add_argument("--setup-output", type=Path, required=True)
    parser.add_argument("--story-output", type=Path, required=True)
    return parser.parse_args(args)


def main(args: Iterable[str] | None = None) -> None:
    parsed = parse_args(args)
    render_camera_setup(
        sweep_root=parsed.sweep_root,
        output=parsed.setup_output,
    )
    render_evidence_story(
        viewpoint=_load_json(parsed.viewpoint_summary),
        kyc_study=_load_json(parsed.kyc_study_summary),
        stage_b2=_load_json(parsed.stage_b2_summary),
        official=_load_json(parsed.official_summary),
        output=parsed.story_output,
    )
    print(
        json.dumps(
            {
                "setup_output": str(parsed.setup_output),
                "story_output": str(parsed.story_output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
