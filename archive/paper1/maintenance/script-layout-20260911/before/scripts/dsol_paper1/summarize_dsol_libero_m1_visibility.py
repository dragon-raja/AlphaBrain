#!/usr/bin/env python3
"""Summarize visibility-defined M1 with source-demonstration statistics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


EXPECTED_CONDITIONS = (
    "canonical_both",
    "strong_info_both",
    "matched_control_both",
    "blind_both",
    "canonical_external_only",
    "strong_info_external_only",
    "matched_control_external_only",
    "blind_external_only",
    "canonical_wrist_only",
    "all_camera_blackout",
)
COMPARISONS = (
    ("info_vs_canonical_both", "strong_info_both", "canonical_both"),
    ("control_vs_canonical_both", "matched_control_both", "canonical_both"),
    ("information_specificity_both", "strong_info_both", "matched_control_both"),
    ("info_vs_blind_both", "strong_info_both", "blind_both"),
    (
        "info_vs_canonical_external_only",
        "strong_info_external_only",
        "canonical_external_only",
    ),
    (
        "control_vs_canonical_external_only",
        "matched_control_external_only",
        "canonical_external_only",
    ),
    (
        "information_specificity_external_only",
        "strong_info_external_only",
        "matched_control_external_only",
    ),
    ("info_vs_blind_external_only", "strong_info_external_only", "blind_external_only"),
    ("canonical_wrist_only_vs_both", "canonical_wrist_only", "canonical_both"),
    ("all_blackout_vs_canonical", "all_camera_blackout", "canonical_both"),
)
PHYSICS_STATE_STAGE = "after_set_init_state_before_camera_install_and_wait"


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def paired_bootstrap(
    values: np.ndarray, *, seed: int, samples: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    return np.quantile(values[draws].mean(axis=1), [0.025, 0.975]).tolist()


def load_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        )
    if not rows:
        raise ValueError("no episode rows")
    return rows


def paired_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    result: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        pair_key = str(row["pair_key"])
        condition = str(row["condition"])
        if condition in result[pair_key]:
            raise ValueError(f"duplicate pair condition: {pair_key}::{condition}")
        result[pair_key][condition] = row
    for pair_key, conditions in result.items():
        missing = set(EXPECTED_CONDITIONS) - set(conditions)
        extras = set(conditions) - set(EXPECTED_CONDITIONS)
        if missing or extras:
            raise ValueError(
                f"invalid pair {pair_key}: missing={sorted(missing)} extras={sorted(extras)}"
            )
        stages = {
            row["initial_metrics"].get("physics_state_stage")
            for row in conditions.values()
        }
        if stages != {PHYSICS_STATE_STAGE}:
            raise ValueError(f"invalid physics hash stage for {pair_key}: {stages}")
        hashes = {
            row["initial_metrics"]["physics_state_sha256"]
            for row in conditions.values()
        }
        if len(hashes) != 1:
            raise ValueError(f"paired physics state mismatch: {pair_key}")
    return dict(result)


def source_group_id(pair: Mapping[str, Mapping[str, Any]]) -> str:
    example = next(iter(pair.values()))
    return str(example.get("episode_id_source", example["pair_key"]))


def grouped_differences(
    by_pair: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    left: str,
    right: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    task_by_group: dict[str, str] = {}
    for pair in by_pair.values():
        group = source_group_id(pair)
        grouped[group].append(
            float(pair[left]["success"]) - float(pair[right]["success"])
        )
        task_by_group[group] = str(pair[left]["task_id"])
    rows = [
        {
            "source_episode_group": group,
            "task_id": task_by_group[group],
            "paired_state_count": len(values),
            "mean_success_difference": float(np.mean(values)),
        }
        for group, values in sorted(grouped.items())
    ]
    return np.asarray(
        [row["mean_success_difference"] for row in rows], dtype=np.float64
    ), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()

    rows = load_rows(args.episodes)
    by_pair = paired_rows(rows)
    condition_summary = {}
    for condition in EXPECTED_CONDITIONS:
        selected = [pair[condition] for pair in by_pair.values()]
        successes = sum(bool(row["success"]) for row in selected)
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in selected:
            grouped[str(row.get("episode_id_source", row["pair_key"]))].append(
                float(row["success"])
            )
        condition_summary[condition] = {
            "successes": successes,
            "state_episodes": len(selected),
            "state_success_rate": successes / len(selected),
            "state_wilson_95": wilson(successes, len(selected)),
            "source_episode_groups": len(grouped),
            "source_episode_macro_success_rate": float(
                np.mean([np.mean(values) for values in grouped.values()])
            ),
            "mean_completion_steps": float(
                np.mean([row["completion_steps"] for row in selected])
            ),
        }

    comparison_summary = {}
    comparison_group_rows = []
    for index, (name, left, right) in enumerate(COMPARISONS):
        differences, group_rows = grouped_differences(
            by_pair,
            left=left,
            right=right,
        )
        ci = paired_bootstrap(
            differences,
            seed=args.seed + index,
            samples=args.bootstrap_samples,
        )
        comparison_summary[name] = {
            "left": left,
            "right": right,
            "difference_pp": float(differences.mean() * 100.0),
            "paired_source_episode_bootstrap_95_pp": [value * 100.0 for value in ci],
            "independent_source_episode_groups": len(differences),
            "paired_frame_states": len(by_pair),
        }
        comparison_group_rows.extend(
            {"comparison": name, **row} for row in group_rows
        )

    task_rows = []
    for task_id in sorted({str(row["task_id"]) for row in rows}):
        for condition in EXPECTED_CONDITIONS:
            selected = [
                row
                for row in rows
                if row["task_id"] == task_id and row["condition"] == condition
            ]
            task_rows.append(
                {
                    "task_id": task_id,
                    "condition": condition,
                    "successes": sum(bool(row["success"]) for row in selected),
                    "state_episodes": len(selected),
                    "state_success_rate": float(
                        np.mean([row["success"] for row in selected])
                    ),
                    "source_episode_groups": len(
                        {row["episode_id_source"] for row in selected}
                    ),
                }
            )

    summary = {
        "schema": "dsol_libero_m1_visibility_closed_loop_summary_v1",
        "status": "PASS",
        "analysis_role": "development_quick_gate_not_final_confirmatory",
        "episode_count": len(rows),
        "paired_frame_state_count": len(by_pair),
        "independent_source_episode_group_count": len(
            {source_group_id(pair) for pair in by_pair.values()}
        ),
        "physics_pair_mismatches": 0,
        "physics_state_hash_stage": PHYSICS_STATE_STAGE,
        "conditions": condition_summary,
        "paired_comparisons": comparison_summary,
        "statistical_unit": "source HDF5 demonstration; frame states clustered within source",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    for name, output_rows in (
        ("task_condition_success.csv", task_rows),
        ("comparison_source_groups.csv", comparison_group_rows),
    ):
        with (args.output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
            writer.writeheader()
            writer.writerows(output_rows)

    try:
        import matplotlib.pyplot as plt

        values = [condition_summary[name]["state_success_rate"] * 100 for name in EXPECTED_CONDITIONS]
        figure, axis = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
        axis.bar(range(len(EXPECTED_CONDITIONS)), values, color="#3f7fa8")
        axis.set_xticks(
            range(len(EXPECTED_CONDITIONS)), EXPECTED_CONDITIONS, rotation=30, ha="right"
        )
        axis.set_ylabel("Full-task success (%)")
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.25)
        for index, value in enumerate(values):
            axis.text(index, value + 1, f"{value:.1f}", ha="center", fontsize=8)
        figure.savefig(args.output_dir / "condition_success.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
