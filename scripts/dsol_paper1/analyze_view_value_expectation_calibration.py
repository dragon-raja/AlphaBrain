#!/usr/bin/env python3
"""Analyze confirmed state-specific view headroom at calibration stage D."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from build_view_value_expectation_calibration_stage import (
    candidate_summaries,
    load_results,
    validate_explicit_pairing,
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


def cluster_bootstrap(values: Mapping[str, float], *, seed: int, draws: int) -> list[float]:
    groups = sorted(values)
    array = np.asarray([values[group] for group in groups], dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(draws, len(array)))
    return np.quantile(array[indices].mean(axis=1), [0.025, 0.975]).tolist()


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        return [0.0, 1.0]
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [float(center - half), float(center + half)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("stage") != "D" or protocol.get("status") != "PASS":
        raise ValueError("analysis requires a complete stage-D protocol")
    rows = load_results(args.inputs)
    if len(rows) != int(protocol["episode_count"]):
        raise ValueError(f"stage-D matrix incomplete: {len(rows)}/{protocol['episode_count']}")
    validate_explicit_pairing(rows, "D")
    summaries = candidate_summaries(rows)
    metadata = {}
    for row in rows:
        metadata.setdefault(
            row["pair_key"],
            {"task_id": row["task_id"], "source_group": row["source_group"]},
        )
    state_rows = []
    for pair_key, candidates in sorted(summaries.items()):
        noncanonical = [value for key, value in candidates.items() if key != "canonical"]
        if len(noncanonical) != 1:
            raise ValueError("stage D must contain one canonical and one frozen candidate")
        candidate = noncanonical[0]
        canonical = candidates["canonical"]
        gain = candidate["mean_success"] - canonical["mean_success"]
        strong = candidate["mean_success"] >= 0.8 and gain >= 0.2
        state_rows.append(
            {
                "pair_key": pair_key,
                **metadata[pair_key],
                "candidate_id": candidate["candidate_id"],
                "canonical_success": canonical["mean_success"],
                "candidate_success": candidate["mean_success"],
                "success_gain": gain,
                "canonical_progress": canonical["mean_progress"],
                "candidate_progress": candidate["mean_progress"],
                "progress_gain": candidate["mean_progress"] - canonical["mean_progress"],
                "harm_probability": candidate["harm_probability"],
                "strong_state": strong,
            }
        )
    strong_rows = [row for row in state_rows if row["strong_state"]]
    source_advantages = {
        row["source_group"]: row["success_gain"] for row in state_rows
    }
    strong_fraction = len(strong_rows) / len(state_rows)
    gate = (
        strong_fraction >= 0.25
        and len({row["source_group"] for row in strong_rows}) >= 4
        and len({row["task_id"] for row in strong_rows}) >= 3
    )
    result = {
        "schema": "dsol_view_value_expectation_calibration_analysis_v1",
        "status": "STABLE_VIEW_HEADROOM_CONFIRMED" if gate else "VIEW_HEADROOM_NOT_CONFIRMED",
        "state_count": len(state_rows),
        "source_group_count": len(source_advantages),
        "task_count": len({row["task_id"] for row in state_rows}),
        "policy_noise_repeats_per_condition": 64,
        "strong_state_count": len(strong_rows),
        "strong_state_fraction": strong_fraction,
        "strong_state_fraction_wilson_95": wilson(len(strong_rows), len(state_rows)),
        "strong_source_group_count": len({row["source_group"] for row in strong_rows}),
        "strong_task_count": len({row["task_id"] for row in strong_rows}),
        "source_equal_success_gain_pp": float(np.mean(list(source_advantages.values())) * 100),
        "source_cluster_bootstrap_95_pp": [
            value * 100
            for value in cluster_bootstrap(
                source_advantages,
                seed=20260901,
                draws=args.bootstrap_resamples,
            )
        ],
        "gate_thresholds": {
            "candidate_success_at_least": 0.8,
            "candidate_gain_at_least_pp": 20.0,
            "strong_state_fraction_at_least": 0.25,
            "strong_source_groups_at_least": 4,
            "strong_tasks_at_least": 3,
        },
        "states": state_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "analysis.json", result)
    with (args.output_dir / "state_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(state_rows[0]))
        writer.writeheader()
        writer.writerows(state_rows)
    print(json.dumps({key: result[key] for key in ("status", "strong_state_count", "strong_state_fraction")}, sort_keys=True))


if __name__ == "__main__":
    main()
