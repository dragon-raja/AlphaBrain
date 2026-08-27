#!/usr/bin/env python3
"""Summarize the constructed task-centric closed-loop pilot."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


COMPARISONS = (
    ("strong_info_both", "canonical_both"),
    ("strong_info_both", "matched_control_both"),
    ("strong_info_external_only", "canonical_external_only"),
    ("strong_info_external_only", "matched_control_external_only"),
    ("canonical_wrist_only", "canonical_both"),
    ("all_camera_blackout", "canonical_both"),
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def load_rows(patterns: Iterable[str]) -> list[dict[str, Any]]:
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    rows = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def source_group(row: Mapping[str, Any]) -> str:
    return f"{row['task_id']}::{row['episode_id_source']}"


def group_scores(rows: Sequence[Mapping[str, Any]], condition: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["condition"] == condition:
            grouped[source_group(row)].append(float(bool(row["success"])))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def stratified_bootstrap_difference(
    first: Mapping[str, float],
    second: Mapping[str, float],
    *,
    seed: int,
    samples: int,
) -> tuple[float, float, float]:
    keys = sorted(set(first) & set(second))
    tasks: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        tasks[key.split("::", 1)[0]].append(key)
    observed = float(np.mean([first[key] - second[key] for key in keys]))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        values = []
        for task_keys in tasks.values():
            sampled = rng.choice(task_keys, size=len(task_keys), replace=True)
            values.extend(first[key] - second[key] for key in sampled)
        draws[index] = np.mean(values)
    low, high = np.quantile(draws, [0.025, 0.975])
    return observed, float(low), float(high)


def summarize(
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    expected_conditions = {str(spec["condition"]) for spec in protocol["specs"]}
    expected_pairs = {str(spec["pair_key"]) for spec in protocol["specs"]}
    by_pair: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        pair_key = str(row["pair_key"])
        condition = str(row["condition"])
        if condition in by_pair[pair_key]:
            raise ValueError(f"duplicate condition {condition} for {pair_key}")
        by_pair[pair_key][condition] = row
    if set(by_pair) != expected_pairs:
        raise ValueError("result pair keys do not match the frozen protocol")
    for pair_key, conditions in by_pair.items():
        if set(conditions) != expected_conditions:
            raise ValueError(f"incomplete pair {pair_key}")

    physics_mismatches = 0
    initial_successes = 0
    for conditions in by_pair.values():
        physics = {
            str(row["initial_metrics"]["physics_state_sha256"])
            for row in conditions.values()
        }
        physics_mismatches += int(len(physics) != 1)
        initial_successes += int(
            any(bool(row["initial_metrics"]["initial_task_success"]) for row in conditions.values())
        )

    strict_keys = {
        str(row["pair_key"])
        for row in protocol["selected_states"]
        if row["strong_visibility_pass"] and row["control_visibility_pass"]
    }
    conditions = []
    for condition in sorted(expected_conditions):
        selected = [row for row in rows if row["condition"] == condition]
        strict = [row for row in selected if row["pair_key"] in strict_keys]
        source = group_scores(rows, condition)
        conditions.append(
            {
                "condition": condition,
                "state_count": len(selected),
                "state_success_rate": float(np.mean([bool(row["success"]) for row in selected])),
                "strict_visibility_state_count": len(strict),
                "strict_visibility_success_rate": float(
                    np.mean([bool(row["success"]) for row in strict])
                ),
                "source_episode_count": len(source),
                "source_episode_macro_success_rate": float(np.mean(list(source.values()))),
                "mean_completion_steps_successes": (
                    float(np.mean([row["completion_steps"] for row in selected if row["success"]]))
                    if any(row["success"] for row in selected)
                    else None
                ),
            }
        )

    comparisons = []
    for index, (first_name, second_name) in enumerate(COMPARISONS):
        first = group_scores(rows, first_name)
        second = group_scores(rows, second_name)
        estimate, low, high = stratified_bootstrap_difference(
            first,
            second,
            seed=seed + index,
            samples=bootstrap_samples,
        )
        state_pairs = [conditions for conditions in by_pair.values()]
        rescue = sum(
            bool(pair[first_name]["success"]) and not bool(pair[second_name]["success"])
            for pair in state_pairs
        )
        harm = sum(
            bool(pair[second_name]["success"]) and not bool(pair[first_name]["success"])
            for pair in state_pairs
        )
        comparisons.append(
            {
                "first": first_name,
                "second": second_name,
                "source_episode_macro_difference_pp": 100.0 * estimate,
                "paired_stratified_bootstrap_95_ci_pp": [100.0 * low, 100.0 * high],
                "source_episode_count": len(first),
                "state_rescue_count": rescue,
                "state_harm_count": harm,
                "state_net_rescue": rescue - harm,
            }
        )

    task_conditions = []
    for task_id in sorted({str(row["task_id"]) for row in rows}):
        for condition in sorted(expected_conditions):
            selected = [
                row
                for row in rows
                if row["task_id"] == task_id and row["condition"] == condition
            ]
            task_conditions.append(
                {
                    "task_id": task_id,
                    "condition": condition,
                    "state_count": len(selected),
                    "successes": sum(bool(row["success"]) for row in selected),
                    "success_rate": float(np.mean([bool(row["success"]) for row in selected])),
                }
            )
    return {
        "schema": "dsol_constructed_taskcentric_closed_loop_summary_v1",
        "status": "PASS",
        "episode_count": len(rows),
        "paired_state_count": len(by_pair),
        "strict_visibility_state_count": len(strict_keys),
        "statistical_unit": "source_episode",
        "bootstrap": {
            "method": "paired_stratified_source_episode_bootstrap",
            "samples": bootstrap_samples,
            "seed": seed,
        },
        "integrity": {
            "physics_mismatch_pair_count": physics_mismatches,
            "initial_success_pair_count": initial_successes,
            "complete_condition_pairs": len(by_pair),
        },
        "conditions": conditions,
        "comparisons": comparisons,
        "task_conditions": task_conditions,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_conditions(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row["condition"]).replace("_", "\n") for row in rows]
    values = [100.0 * float(row["state_success_rate"]) for row in rows]
    colors = ["#3B82A0" if "both" in row["condition"] else "#8B9AA8" for row in rows]
    figure, axis = plt.subplots(figsize=(12, 5.5))
    bars = axis.bar(range(len(rows)), values, color=colors)
    axis.set_ylim(0, 105)
    axis.set_ylabel("Closed-loop success rate (%)")
    axis.set_xticks(range(len(rows)), labels, fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize(
        load_rows(args.inputs),
        json.loads(args.protocol.read_text()),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "metrics.json", result)
    write_csv(args.output_dir / "condition_success.csv", result["conditions"])
    write_csv(args.output_dir / "task_condition_success.csv", result["task_conditions"])
    plot_conditions(args.output_dir / "condition_success.png", result["conditions"])
    print(json.dumps({"status": result["status"], "episodes": result["episode_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
