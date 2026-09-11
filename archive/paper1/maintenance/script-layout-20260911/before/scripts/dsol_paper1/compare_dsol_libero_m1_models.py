#!/usr/bin/env python3
"""Compare M1 policies on the same states with source-demo clustered statistics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dsol_paper1.summarize_dsol_libero_m1_visibility import (
    COMPARISONS,
    EXPECTED_CONDITIONS,
    load_rows,
    paired_bootstrap,
    paired_rows,
    source_group_id,
)


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must use NAME=RUN_ROOT")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("run name cannot be empty")
    path = Path(raw_path)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"run root is not a directory: {path}")
    return name, path


def load_run(path: Path) -> dict[str, dict[str, Mapping[str, Any]]]:
    episode_paths = sorted(path.glob("episodes-shard-*.jsonl"))
    if not episode_paths:
        raise ValueError(f"no episode shards in {path}")
    return paired_rows(load_rows(episode_paths))


def validate_same_protocol(
    runs: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
) -> None:
    first_name, first = next(iter(runs.items()))
    reference_keys = set(first)
    for name, by_pair in runs.items():
        if set(by_pair) != reference_keys:
            missing = sorted(reference_keys - set(by_pair))
            extras = sorted(set(by_pair) - reference_keys)
            raise ValueError(
                f"protocol mismatch for {name} vs {first_name}: "
                f"missing={missing[:3]} extras={extras[:3]}"
            )
        for pair_key in reference_keys:
            reference = first[pair_key][EXPECTED_CONDITIONS[0]]
            candidate = by_pair[pair_key][EXPECTED_CONDITIONS[0]]
            fields = ("task_id", "episode_id_source")
            if any(reference.get(field) != candidate.get(field) for field in fields):
                raise ValueError(f"state identity mismatch: {name}::{pair_key}")


def grouped_model_difference(
    left: Mapping[str, Mapping[str, Mapping[str, Any]]],
    right: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    condition: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    tasks: dict[str, str] = {}
    for pair_key, left_pair in left.items():
        group = source_group_id(left_pair)
        grouped[group].append(
            float(left_pair[condition]["success"])
            - float(right[pair_key][condition]["success"])
        )
        tasks[group] = str(left_pair[condition]["task_id"])
    rows = [
        {
            "source_episode_group": group,
            "task_id": tasks[group],
            "paired_state_count": len(values),
            "mean_success_difference": float(np.mean(values)),
        }
        for group, values in sorted(grouped.items())
    ]
    return np.asarray(
        [row["mean_success_difference"] for row in rows], dtype=np.float64
    ), rows


def grouped_within_model_difference(
    by_pair: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    left: str,
    right: str,
) -> np.ndarray:
    grouped: dict[str, list[float]] = defaultdict(list)
    for pair in by_pair.values():
        grouped[source_group_id(pair)].append(
            float(pair[left]["success"]) - float(pair[right]["success"])
        )
    return np.asarray(
        [float(np.mean(values)) for _, values in sorted(grouped.items())],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()

    run_roots = dict(args.run)
    if len(run_roots) != len(args.run):
        raise ValueError("duplicate run name")
    if args.baseline not in run_roots:
        raise ValueError(f"baseline not found: {args.baseline}")
    runs = {name: load_run(path) for name, path in run_roots.items()}
    validate_same_protocol(runs)

    condition_rows = []
    within_rows = []
    model_difference_rows = []
    group_rows = []
    for model_index, (name, by_pair) in enumerate(runs.items()):
        for condition in EXPECTED_CONDITIONS:
            values = np.asarray(
                [float(pair[condition]["success"]) for pair in by_pair.values()],
                dtype=np.float64,
            )
            condition_rows.append(
                {
                    "model": name,
                    "condition": condition,
                    "successes": int(values.sum()),
                    "state_episodes": len(values),
                    "state_success_rate": float(values.mean()),
                }
            )
        for comparison_index, (comparison, left, right) in enumerate(COMPARISONS):
            differences = grouped_within_model_difference(
                by_pair, left=left, right=right
            )
            ci = paired_bootstrap(
                differences,
                seed=args.seed + model_index * 100 + comparison_index,
                samples=args.bootstrap_samples,
            )
            within_rows.append(
                {
                    "model": name,
                    "comparison": comparison,
                    "difference_pp": float(differences.mean() * 100.0),
                    "ci_low_pp": float(ci[0] * 100.0),
                    "ci_high_pp": float(ci[1] * 100.0),
                    "source_episode_groups": len(differences),
                }
            )

    baseline = runs[args.baseline]
    comparison_counter = 0
    for name, by_pair in runs.items():
        if name == args.baseline:
            continue
        for condition in EXPECTED_CONDITIONS:
            differences, rows = grouped_model_difference(
                by_pair, baseline, condition=condition
            )
            ci = paired_bootstrap(
                differences,
                seed=args.seed + 1000 + comparison_counter,
                samples=args.bootstrap_samples,
            )
            model_difference_rows.append(
                {
                    "model": name,
                    "baseline": args.baseline,
                    "condition": condition,
                    "difference_pp": float(differences.mean() * 100.0),
                    "ci_low_pp": float(ci[0] * 100.0),
                    "ci_high_pp": float(ci[1] * 100.0),
                    "source_episode_groups": len(differences),
                }
            )
            group_rows.extend(
                {
                    "model": name,
                    "baseline": args.baseline,
                    "condition": condition,
                    **row,
                }
                for row in rows
            )
            comparison_counter += 1

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "condition_success.csv": condition_rows,
        "within_model_comparisons.csv": within_rows,
        "model_vs_baseline.csv": model_difference_rows,
        "model_vs_baseline_source_groups.csv": group_rows,
    }
    for filename, rows in tables.items():
        with (output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "schema": "dsol_libero_m1_cross_model_summary_v1",
        "status": "PASS",
        "analysis_role": "development_quick_gate_not_final_confirmatory",
        "baseline": args.baseline,
        "models": list(runs),
        "paired_frame_states": len(next(iter(runs.values()))),
        "independent_source_episode_groups": len(
            {
                source_group_id(pair)
                for pair in next(iter(runs.values())).values()
            }
        ),
        "condition_success": condition_rows,
        "within_model_comparisons": within_rows,
        "model_vs_baseline": model_difference_rows,
        "statistical_unit": "source HDF5 demonstration; frame states clustered within source",
        "run_roots": {name: str(path) for name, path in run_roots.items()},
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    try:
        import matplotlib.pyplot as plt

        selected = (
            "canonical_both",
            "strong_info_both",
            "matched_control_both",
            "blind_both",
            "canonical_external_only",
            "canonical_wrist_only",
            "all_camera_blackout",
        )
        model_names = list(runs)
        width = 0.8 / len(model_names)
        x = np.arange(len(selected))
        figure, axis = plt.subplots(figsize=(14, 5.8), constrained_layout=True)
        by_key = {
            (row["model"], row["condition"]): row for row in condition_rows
        }
        for index, name in enumerate(model_names):
            values = [
                by_key[(name, condition)]["state_success_rate"] * 100.0
                for condition in selected
            ]
            axis.bar(
                x - 0.4 + width / 2 + index * width,
                values,
                width,
                label=name,
            )
        axis.set_xticks(x, selected, rotation=25, ha="right")
        axis.set_ylabel("Full-task success (%)")
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
        figure.savefig(output_dir / "cross_model_condition_success.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
