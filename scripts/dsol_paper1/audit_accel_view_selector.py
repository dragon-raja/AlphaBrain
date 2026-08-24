#!/usr/bin/env python3
"""Audit Accel prefix sensitivity for the fixed-state view selector."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_ROOT = Path(
    "/share/longjunyu/alphabrain/experiments/dsol-accel-constructed-v2"
)
DEFAULT_OUTPUT = ROOT / "docs/dsol_paper1/accel_view_selector_audit.json"
DEFAULT_FIGURE = ROOT / "docs/dsol_paper1/figures/accel_prefix_selector_audit.png"
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

MODEL_ROOTS = {
    "Broad practical": "broad64-practical-seed41-full",
    "Broad state-matched": "broad64-state-matched-seed41-full",
    "Broad paired FM": "broad64-paired-fm-seed41-full",
    "Broad paired consistency": "broad64-paired-consistency-seed41-full",
}
JOIN_NAMES = {
    "Broad practical": "broad64-practical",
    "Broad state-matched": "broad64-state-matched",
    "Broad paired FM": "broad64-paired-fm",
    "Broad paired consistency": "broad64-paired-consistency",
}
EVALUATED_ROLES = ("canonical", "strong_info", "matched_control", "blind")
PREFIXES = tuple(range(2, 11))


def configure_font() -> None:
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        family = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
        plt.rcParams["font.family"] = family
    plt.rcParams["axes.unicode_minus"] = False


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def reference_scaled_scores(npz_path: Path) -> dict[int, dict[str, float]]:
    """Apply the official reference code's in-batch path-coordinate z-score.

    The original detector normalizes flattened denoising-path coordinates before
    taking norms. For this cross-view extension, the fixed-state candidate bank is
    the only available batch, so the scale is estimated over its candidates and
    denoising points. This is a sensitivity audit, not a reproduction of the
    paper's rollout-level calibration population.
    """

    with np.load(npz_path) as artifact:
        candidate_ids = [str(value) for value in artifact["candidate_ids"]]
        velocity = np.asarray(artifact["velocity_trace"], dtype=np.float64)
        initial_noise = np.asarray(artifact["initial_noise"], dtype=np.float64)

    step_size = -1.0 / velocity.shape[1]
    points = [initial_noise]
    current = initial_noise.copy()
    for step in range(velocity.shape[1]):
        current = current + step_size * velocity[:, step]
        points.append(current.copy())
    path = np.stack(points, axis=1)
    flattened_path = path.reshape(path.shape[0], path.shape[1], -1)
    coordinate_scale = flattened_path.std(axis=(0, 1)) + 1e-12

    flattened_velocity = velocity.reshape(velocity.shape[0], velocity.shape[1], -1)
    flattened_velocity = flattened_velocity / coordinate_scale[None, None, :]
    velocity_norm = np.linalg.norm(flattened_velocity, axis=2)
    delta_norm = np.linalg.norm(np.diff(flattened_velocity, axis=1), axis=2)

    result: dict[int, dict[str, float]] = {}
    for prefix in PREFIXES:
        scores = (
            prefix
            * delta_norm[:, : prefix - 1].sum(axis=1)
            / (velocity_norm[:, :prefix].sum(axis=1) + 1e-12)
        )
        result[prefix] = {
            candidate_id: float(score)
            for candidate_id, score in zip(candidate_ids, scores)
        }
    return result


def local_v1_scores(rankings_path: Path) -> dict[int, dict[str, float]]:
    rows = load_json(rankings_path)["diagnostic_shortlist"]["ranking"]
    return {
        prefix: {
            str(row["candidate_id"]): float(row[f"accel_{prefix}"])
            for row in rows
        }
        for prefix in PREFIXES
    }


def summarize_prefix(
    state_artifacts: dict[str, dict],
    outcomes: dict[str, dict],
    *,
    prefix: int,
    score_variant: str,
) -> dict:
    role_counts: Counter[str] = Counter()
    selected_successes = 0
    canonical_successes = 0
    any_successes = 0
    selected_in_success_set = 0
    efficiency_oracle_matches = 0
    efficiency_oracle_states = 0
    group_differences: dict[str, list[float]] = {}

    for pair_key, outcome in outcomes.items():
        artifact = state_artifacts[pair_key]
        scores = artifact[score_variant][prefix]
        role_ids = artifact["role_ids"]
        selected_role = min(
            EVALUATED_ROLES,
            key=lambda role: (scores[role_ids[role]], role),
        )
        role_counts[selected_role] += 1

        selected_success = bool(outcome["outcomes"][selected_role]["success"])
        canonical_success = bool(outcome["outcomes"]["canonical"]["success"])
        successful_roles = [
            role for role, values in outcome["outcomes"].items() if values["success"]
        ]
        selected_successes += int(selected_success)
        canonical_successes += int(canonical_success)
        any_successes += int(bool(successful_roles))
        selected_in_success_set += int(selected_success and bool(successful_roles))

        if successful_roles:
            minimum_steps = min(
                int(outcome["outcomes"][role]["completion_steps"])
                for role in successful_roles
            )
            efficiency_roles = [
                role
                for role in successful_roles
                if int(outcome["outcomes"][role]["completion_steps"])
                == minimum_steps
            ]
            efficiency_oracle_matches += int(selected_role in efficiency_roles)
            efficiency_oracle_states += 1

        group_differences.setdefault(outcome["source_episode_group"], []).append(
            float(selected_success) - float(canonical_success)
        )

    state_count = len(outcomes)
    source_macro_delta = 100.0 * float(
        np.mean([np.mean(values) for values in group_differences.values()])
    )
    return {
        "prefix": prefix,
        "state_count": state_count,
        "source_episode_group_count": len(group_differences),
        "selected_role_counts": dict(sorted(role_counts.items())),
        "selected_success_rate": selected_successes / state_count,
        "canonical_success_rate": canonical_successes / state_count,
        "any_evaluated_view_success_rate": any_successes / state_count,
        "selected_minus_canonical_state_pp": 100.0
        * (selected_successes - canonical_successes)
        / state_count,
        "selected_minus_canonical_source_macro_pp": source_macro_delta,
        "selected_in_success_set_rate_when_any_success": (
            selected_in_success_set / any_successes if any_successes else None
        ),
        "exact_efficiency_oracle_match_rate": (
            efficiency_oracle_matches / efficiency_oracle_states
            if efficiency_oracle_states
            else None
        ),
    }


def audit_model(experiment_root: Path, label: str) -> dict:
    accel_root = experiment_root / MODEL_ROOTS[label]
    join_root = experiment_root / "m1-joins" / JOIN_NAMES[label]
    outcome_rows = load_json(join_root / "state_records.json")
    outcomes = {str(row["pair_key"]): row for row in outcome_rows}

    state_artifacts: dict[str, dict] = {}
    for state_dir in sorted((accel_root / "states").iterdir()):
        rank_record = load_json(state_dir / "rank_record.json")
        pair_key = str(rank_record["pair_key"])
        role_ids = {
            role: str(rank_record["role_metrics"][role]["candidate_id"])
            for role in EVALUATED_ROLES
        }
        state_artifacts[pair_key] = {
            "role_ids": role_ids,
            "local_v1": local_v1_scores(state_dir / "rankings.json"),
            "reference_scaled": reference_scaled_scores(state_dir / "flow_trace.npz"),
        }

    if set(state_artifacts) != set(outcomes):
        raise ValueError(f"state/outcome key mismatch for {label}")

    return {
        "checkpoint_label": label,
        "accel_root": str(accel_root),
        "m1_join_root": str(join_root),
        "variants": {
            variant: [
                summarize_prefix(
                    state_artifacts,
                    outcomes,
                    prefix=prefix,
                    score_variant=variant,
                )
                for prefix in PREFIXES
            ]
            for variant in ("local_v1", "reference_scaled")
        },
    }


def plot_audit(payload: dict, output: Path) -> None:
    configure_font()
    colors = ["#3D7EA6", "#3F8C7A", "#C79432", "#BE5968"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), sharey=True)
    variants = [
        ("local_v1", "A. 本地冻结版本：归一化动作坐标"),
        ("reference_scaled", "B. 官方参考式坐标尺度敏感性"),
    ]
    for axis, (variant, title) in zip(axes, variants):
        for color, model in zip(colors, payload["models"]):
            rows = payload["models"][model]["variants"][variant]
            axis.plot(
                [row["prefix"] for row in rows],
                [row["selected_minus_canonical_state_pp"] for row in rows],
                marker="o",
                linewidth=1.8,
                markersize=4,
                color=color,
                label=model,
            )
        axis.axhline(0, color="#17212B", linewidth=1)
        axis.axvline(3, color="#8A9299", linestyle="--", linewidth=1)
        axis.set_xticks(PREFIXES)
        axis.set_xlabel("去噪前缀 p（共 10 步）")
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.grid(axis="y", color="#D9DFE5", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Accel 所选视角相对 Canonical 的成功率差（百分点）")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(
        "Accel 视角选择对前缀与坐标尺度敏感；没有跨模型稳定正增益",
        x=0.06,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.01,
        "虚线为预注册 p=3。其余前缀均为事后敏感性分析，不能用于挑选最优结果。统计单位：21 个状态。",
        fontsize=8.5,
        color="#63707D",
    )
    fig.tight_layout(rect=(0.04, 0.06, 0.99, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="#F7F8FA")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()

    payload = {
        "schema": "dsol_accel_view_selector_audit_v1",
        "scope": "fixed-state cross-view selection, not original failure-detection reproduction",
        "primary_prefix": 3,
        "prefixes": list(PREFIXES),
        "evaluated_roles": list(EVALUATED_ROLES),
        "models": {
            label: audit_model(args.experiment_root, label) for label in MODEL_ROOTS
        },
        "interpretation": {
            "local_v1": "Paper equation on normalized action velocity traces.",
            "reference_scaled": (
                "Sensitivity audit using the official reference code's path-coordinate "
                "z-score, estimated within each fixed-state candidate bank."
            ),
            "restriction": (
                "No prefix may be selected post hoc from M1 outcomes; p=3 remains the "
                "frozen primary result."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_audit(payload, args.figure)
    print(args.output)
    print(args.figure)


if __name__ == "__main__":
    main()
