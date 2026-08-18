#!/usr/bin/env python3
"""Summarize paired exact-HDF5 DSOL closed-loop episodes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


EXPECTED_CONDITIONS = (
    "canonical_both",
    "canonical_external_only",
    "canonical_wrist_only",
    "broad_heldout_both",
    "broad_heldout_external_only",
    "broad_heldout_wrist_only",
    "wide_extrapolation_both",
)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def paired_bootstrap(values: np.ndarray, *, seed: int, samples: int) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()

    rows = []
    for path in args.episodes:
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if not rows:
        raise ValueError("no episode rows")
    by_pair = defaultdict(dict)
    for row in rows:
        pair = by_pair[str(row["pair_key"])]
        condition = str(row["condition"])
        if condition in pair:
            raise ValueError(f"duplicate pair condition: {row['pair_key']}::{condition}")
        pair[condition] = row
    for pair_key, conditions in by_pair.items():
        missing = set(EXPECTED_CONDITIONS) - set(conditions)
        if missing:
            raise ValueError(f"incomplete pair {pair_key}: missing {sorted(missing)}")

    physics_mismatches = []
    for pair_key, conditions in by_pair.items():
        hashes = {
            row["initial_metrics"]["physics_state_sha256"]
            for row in conditions.values()
        }
        if len(hashes) != 1:
            physics_mismatches.append(pair_key)
    if physics_mismatches:
        raise ValueError(f"paired physics state mismatch: {physics_mismatches[:5]}")

    conditions_summary = {}
    for condition in EXPECTED_CONDITIONS:
        selected = [pair[condition] for pair in by_pair.values()]
        successes = sum(bool(row["success"]) for row in selected)
        conditions_summary[condition] = {
            "successes": successes,
            "episodes": len(selected),
            "success_rate": successes / len(selected),
            "wilson_95": wilson(successes, len(selected)),
            "mean_completion_steps": float(np.mean([row["completion_steps"] for row in selected])),
        }

    comparisons = (
        ("broad_vs_canonical_both", "broad_heldout_both", "canonical_both"),
        ("extrapolation_vs_canonical_both", "wide_extrapolation_both", "canonical_both"),
        ("broad_vs_canonical_external_only", "broad_heldout_external_only", "canonical_external_only"),
        ("broad_vs_canonical_wrist_only", "broad_heldout_wrist_only", "canonical_wrist_only"),
        ("canonical_external_only_vs_both", "canonical_external_only", "canonical_both"),
        ("canonical_wrist_only_vs_both", "canonical_wrist_only", "canonical_both"),
        ("broad_external_only_vs_both", "broad_heldout_external_only", "broad_heldout_both"),
        ("broad_wrist_only_vs_both", "broad_heldout_wrist_only", "broad_heldout_both"),
    )
    comparison_summary = {}
    for index, (name, left, right) in enumerate(comparisons):
        differences = np.asarray(
            [
                float(pair[left]["success"]) - float(pair[right]["success"])
                for pair in by_pair.values()
            ],
            dtype=np.float64,
        )
        comparison_summary[name] = {
            "left": left,
            "right": right,
            "difference_pp": float(differences.mean() * 100.0),
            "paired_group_bootstrap_95_pp": [
                value * 100.0
                for value in paired_bootstrap(
                    differences,
                    seed=args.seed + index,
                    samples=args.bootstrap_samples,
                )
            ],
            "paired_groups": len(differences),
        }

    task_rows = []
    tasks = sorted({str(row["task_id"]) for row in rows})
    for task in tasks:
        for condition in EXPECTED_CONDITIONS:
            selected = [
                row for row in rows if row["task_id"] == task and row["condition"] == condition
            ]
            task_rows.append(
                {
                    "task_id": task,
                    "condition": condition,
                    "successes": sum(bool(row["success"]) for row in selected),
                    "episodes": len(selected),
                    "success_rate": float(np.mean([row["success"] for row in selected])),
                }
            )

    summary = {
        "schema": "dsol_libero_hdf5_closed_loop_summary_v1",
        "status": "PASS",
        "episode_count": len(rows),
        "paired_group_count": len(by_pair),
        "physics_pair_mismatches": 0,
        "conditions": conditions_summary,
        "paired_comparisons": comparison_summary,
        "statistical_unit": "source HDF5 episode / pair_key",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (args.output_dir / "task_condition_success.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=task_rows[0].keys())
        writer.writeheader(); writer.writerows(task_rows)

    try:
        import matplotlib.pyplot as plt

        labels = list(EXPECTED_CONDITIONS)
        values = [conditions_summary[label]["success_rate"] * 100 for label in labels]
        figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
        axis.bar(range(len(labels)), values, color="#3f7fa8")
        axis.set_xticks(range(len(labels)), labels, rotation=28, ha="right")
        axis.set_ylabel("Full-task success (%)")
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.25)
        for index, value in enumerate(values):
            axis.text(index, value + 1, f"{value:.1f}", ha="center", fontsize=9)
        figure.savefig(args.output_dir / "condition_success.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
