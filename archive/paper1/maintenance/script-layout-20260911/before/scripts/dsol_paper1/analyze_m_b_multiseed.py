from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def load_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 20260819,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("bootstrap requires at least one value")
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(resamples, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "independent_group_count": int(array.size),
        "bootstrap_resamples": resamples,
    }


def validate_camera_rows(rows: Sequence[Mapping[str, Any]], *, expected_count: int) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} camera episodes, found {len(rows)}")
    if any(row.get("condition") != "official_camera" for row in rows):
        raise ValueError("camera-full input contains a non-camera condition")
    keys = [
        (str(row["suite"]), int(row["official_task_id"]), int(row["init_state_index"]))
        for row in rows
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate camera-full episode keys")


def validate_original(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
    expected_trials_per_task: int,
) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} original episodes, found {len(rows)}")
    if any(row.get("condition") != "canonical" for row in rows):
        raise ValueError("Original Full input contains a non-canonical condition")
    keys = [
        (str(row["suite"]), str(row["base_task"]), int(row["init_state_index"]))
        for row in rows
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate Original Full episode keys")
    grouped: dict[tuple[str, str], set[int]] = defaultdict(set)
    for suite, base_task, init_state in keys:
        grouped[(suite, base_task)].add(init_state)
    expected_states = set(range(expected_trials_per_task))
    if any(states != expected_states for states in grouped.values()):
        raise ValueError("incomplete Original Full task/state ledger")


def _episode_files(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("episodes-shard-*.jsonl"))
    if not paths:
        raise ValueError(f"no episode shards in {directory}")
    return paths


def _key(row: Mapping[str, Any], benchmark: str) -> tuple[Any, ...]:
    if benchmark == "camera_full":
        return str(row["suite"]), int(row["official_task_id"]), int(row["init_state_index"])
    return str(row["suite"]), str(row["base_task"]), int(row["init_state_index"])


def _group(row: Mapping[str, Any]) -> str:
    return f"{row['suite']}::{row['base_task']}"


def _validate(rows: Sequence[Mapping[str, Any]], benchmark: str) -> None:
    if benchmark == "camera_full":
        validate_camera_rows(rows, expected_count=1599)
    else:
        validate_original(rows, expected_count=2000, expected_trials_per_task=50)


def _keyed(rows: Sequence[Mapping[str, Any]], benchmark: str) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    return {_key(row, benchmark): row for row in rows}


def _group_differences(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    benchmark: str,
) -> dict[str, float]:
    left_keyed = _keyed(left, benchmark)
    right_keyed = _keyed(right, benchmark)
    if set(left_keyed) != set(right_keyed):
        raise ValueError("paired evaluation ledgers differ")
    grouped: dict[str, list[float]] = defaultdict(list)
    for key in sorted(left_keyed):
        left_row = left_keyed[key]
        right_row = right_keyed[key]
        if _group(left_row) != _group(right_row):
            raise ValueError(f"base-task mismatch for {key}")
        grouped[_group(left_row)].append(
            float(bool(right_row["success"])) - float(bool(left_row["success"]))
        )
    return {name: float(np.mean(values)) for name, values in grouped.items()}


def _rate(rows: Sequence[Mapping[str, Any]]) -> float:
    return float(np.mean([float(bool(row["success"])) for row in rows]))


def _summarize_method(
    baseline: Sequence[Mapping[str, Any]],
    seed_rows: Mapping[int, Sequence[Mapping[str, Any]]],
    benchmark: str,
) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    by_seed_group: dict[int, dict[str, float]] = {}
    for seed, rows in sorted(seed_rows.items()):
        differences = _group_differences(baseline, rows, benchmark)
        by_seed_group[seed] = differences
        clustered = bootstrap_mean_ci(list(differences.values()))
        per_seed[str(seed)] = {
            "success_rate": _rate(rows),
            "delta_vs_official_pp": 100.0 * clustered["mean"],
            "delta_vs_official_ci95_pp": [100.0 * value for value in clustered["ci95"]],
        }
    group_names = set.intersection(*(set(values) for values in by_seed_group.values()))
    if any(set(values) != group_names for values in by_seed_group.values()):
        raise ValueError("base-task groups differ across seeds")
    across_seed_group = [
        float(np.mean([by_seed_group[seed][group] for seed in sorted(by_seed_group)]))
        for group in sorted(group_names)
    ]
    clustered = bootstrap_mean_ci(across_seed_group)
    return {
        "per_seed": per_seed,
        "cross_seed_mean_success_rate": float(
            np.mean([_rate(rows) for rows in seed_rows.values()])
        ),
        "cross_seed_delta_vs_official_pp": 100.0 * clustered["mean"],
        "cross_seed_delta_vs_official_ci95_pp": [
            100.0 * value for value in clustered["ci95"]
        ],
        "independent_task_groups": len(group_names),
        "seed_count": len(seed_rows),
    }


def summarize(
    *,
    benchmark: str,
    baseline: Sequence[Mapping[str, Any]],
    runs: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
    validate: bool = True,
) -> dict[str, Any]:
    if validate:
        _validate(baseline, benchmark)
        for seed_rows in runs.values():
            for rows in seed_rows.values():
                _validate(rows, benchmark)
    methods = {
        method: _summarize_method(baseline, seed_rows, benchmark)
        for method, seed_rows in sorted(runs.items())
    }
    payload: dict[str, Any] = {
        "schema": "dsol_view_revalidation_m_b_multiseed_v1",
        "benchmark": benchmark,
        "baseline_success_rate": _rate(baseline),
        "methods": methods,
    }
    if {"broad64_practical", "broad64_paired_consistency"} <= set(runs):
        practical = runs["broad64_practical"]
        consistency = runs["broad64_paired_consistency"]
        if set(practical) != set(consistency):
            raise ValueError("pairing comparison requires identical training seeds")
        by_seed_group = {
            seed: _group_differences(practical[seed], consistency[seed], benchmark)
            for seed in sorted(practical)
        }
        groups = set.intersection(*(set(values) for values in by_seed_group.values()))
        across_seed = [
            float(np.mean([by_seed_group[seed][group] for seed in sorted(by_seed_group)]))
            for group in sorted(groups)
        ]
        clustered = bootstrap_mean_ci(across_seed)
        per_seed = {
            str(seed): 100.0 * float(np.mean(list(values.values())))
            for seed, values in by_seed_group.items()
        }
        seed_signs = [int(np.sign(value)) for value in per_seed.values()]
        payload["paired_consistency_minus_practical"] = {
            "per_seed_pp": per_seed,
            "all_seeds_same_direction": bool(seed_signs[0])
            and len(set(seed_signs)) == 1,
            "cross_seed_delta_pp": 100.0 * clustered["mean"],
            "cross_seed_ci95_pp": [100.0 * value for value in clustered["ci95"]],
            "passes_preregistered_point_threshold": 100.0 * clustered["mean"] >= 3.0,
        }
    return payload


def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        f"# M-B {payload['benchmark']} 三 seed 汇总",
        "",
        f"- Official success: {payload['baseline_success_rate']:.2%}",
        "",
        "| Method | Mean success | Delta vs Official | 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for method, values in payload["methods"].items():
        ci = values["cross_seed_delta_vs_official_ci95_pp"]
        lines.append(
            f"| {method} | {values['cross_seed_mean_success_rate']:.2%} | "
            f"{values['cross_seed_delta_vs_official_pp']:+.2f}pp | "
            f"[{ci[0]:+.2f},{ci[1]:+.2f}] |"
        )
    pairing = payload.get("paired_consistency_minus_practical")
    if pairing:
        ci = pairing["cross_seed_ci95_pp"]
        lines.extend(
            [
                "",
                "## Pairing contribution",
                "",
                f"- Cross-seed delta: {pairing['cross_seed_delta_pp']:+.2f}pp "
                f"[{ci[0]:+.2f},{ci[1]:+.2f}]",
                f"- Per-seed deltas: {pairing['per_seed_pp']}",
                f"- Same direction: {pairing['all_seeds_same_direction']}",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate M-B paired multi-seed evaluations")
    parser.add_argument("--benchmark", choices=("camera_full", "original_full"), required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--run", nargs=3, action="append", metavar=("METHOD", "SEED", "DIR"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    baseline = load_rows(_episode_files(args.baseline_dir))
    runs: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(dict)
    for method, seed_text, directory_text in args.run:
        seed = int(seed_text)
        if seed in runs[method]:
            raise ValueError(f"duplicate method/seed: {method}/{seed}")
        runs[method][seed] = load_rows(_episode_files(Path(directory_text)))
    payload = summarize(benchmark=args.benchmark, baseline=baseline, runs=runs)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(args.output_report, payload)


if __name__ == "__main__":
    main()
