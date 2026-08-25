#!/usr/bin/env python3
"""Summarize shortlist or exhaustive closed-loop outcomes for Accel Gate A97."""

from __future__ import annotations

import argparse
import glob
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def load_rows(patterns: Sequence[str]) -> list[dict[str, Any]]:
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError("no episode JSONL matched")
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    episode_ids = [str(row["episode_id"]) for row in rows]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("episode IDs are not unique")
    if any(row.get("status") != "complete" for row in rows):
        raise ValueError("all episode rows must be complete")
    return rows


def paired_group_difference(
    rows: Sequence[Mapping[str, Any]], left: str, right: str
) -> tuple[float, dict[str, float]]:
    by_group: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        condition = str(row["condition"])
        if condition not in {left, right}:
            continue
        by_group[str(row["episode_id_source"])][condition].append(float(row["success"]))
    differences = {}
    for group, values in by_group.items():
        if left in values and right in values:
            differences[group] = float(np.mean(values[left]) - np.mean(values[right]))
    if not differences:
        raise ValueError(f"no paired source groups for {left} vs {right}")
    return float(np.mean(list(differences.values()))), differences


def bootstrap_ci(
    values: Sequence[float], *, seed: int = 20260825, samples: int = 20000
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def summarize_shortlist(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    expected = int(protocol["episode_count"])
    if len(rows) != expected:
        raise ValueError(f"expected {expected} episodes, found {len(rows)}")
    conditions = sorted({str(row["condition"]) for row in rows})
    condition_success = {}
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        condition_success[condition] = {
            "episodes": len(selected),
            "successes": sum(bool(row["success"]) for row in selected),
            "state_success_rate": float(np.mean([row["success"] for row in selected])),
            "mean_completion_steps": float(np.mean([row["completion_steps"] for row in selected])),
        }
    comparisons = {}
    for condition in conditions:
        if condition == "canonical":
            continue
        difference, groups = paired_group_difference(rows, condition, "canonical")
        low, high = bootstrap_ci(list(groups.values()))
        comparisons[f"{condition}_vs_canonical"] = {
            "difference_pp": 100 * difference,
            "ci_low_pp": 100 * low,
            "ci_high_pp": 100 * high,
            "source_episode_groups": len(groups),
        }
    by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["pair_key"])].append(row)
    oracle_rate = float(
        np.mean([any(bool(row["success"]) for row in values) for values in by_pair.values()])
    )
    return {
        "condition_success": condition_success,
        "paired_source_bootstrap": comparisons,
        "oracle_at_shortlist_state_rate": oracle_rate,
        "state_count": len(by_pair),
    }


def point_biserial(scores: Sequence[float], successes: Sequence[bool]) -> float | None:
    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(successes, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summarize_oracle(rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> dict[str, Any]:
    expected = int(protocol["episode_count"])
    if len(rows) != expected:
        raise ValueError(f"expected {expected} episodes, found {len(rows)}")
    by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["pair_key"])].append(row)
    state_rows = []
    correlations = []
    for pair_key, values in sorted(by_pair.items()):
        if len(values) != 97:
            raise ValueError(f"oracle state {pair_key} has {len(values)} candidates")
        by_id = {str(row["selected_candidate_id"]): row for row in values}
        canonical = by_id["canonical"]
        successes = [bool(row["success"]) for row in values]
        scores = [float(row["selection_metadata"]["ensemble_accel_3"]) for row in values]
        correlation = point_biserial(scores, successes)
        if correlation is not None:
            correlations.append(correlation)
        state_rows.append(
            {
                "pair_key": pair_key,
                "task_id": str(values[0]["task_id"]),
                "source_episode_id": str(values[0]["episode_id_source"]),
                "canonical_success": bool(canonical["success"]),
                "oracle_at_97_success": any(successes),
                "successful_candidate_count": sum(successes),
                "accel_success_point_biserial": correlation,
                "best_successful_accel_rank": min(
                    (
                        int(row["selection_metadata"]["ensemble_accel_rank"])
                        for row in values
                        if row["success"]
                    ),
                    default=None,
                ),
            }
        )
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in state_rows:
        grouped[str(row["source_episode_id"])].append(row)
    canonical_group_rates = [
        float(np.mean([row["canonical_success"] for row in values]))
        for values in grouped.values()
    ]
    oracle_group_rates = [
        float(np.mean([row["oracle_at_97_success"] for row in values]))
        for values in grouped.values()
    ]
    oracle_group_differences = [
        float(
            np.mean(
                [
                    float(row["oracle_at_97_success"]) - float(row["canonical_success"])
                    for row in values
                ]
            )
        )
        for values in grouped.values()
    ]
    oracle_low, oracle_high = bootstrap_ci(oracle_group_rates)
    oracle_diff_low, oracle_diff_high = bootstrap_ci(oracle_group_differences)
    selected_by_pair = {
        str(state["pair_key"]): {
            str(condition): str(candidate_id)
            for condition, candidate_id in state["selected_candidates"].items()
        }
        for state in protocol["selected_states"]
    }
    pre_registered = {}
    for condition in sorted(next(iter(selected_by_pair.values()))):
        selected_rows = []
        for pair_key, values in by_pair.items():
            candidate_id = selected_by_pair[pair_key][condition]
            by_id = {str(row["selected_candidate_id"]): row for row in values}
            selected_rows.append(by_id[candidate_id])
        selected_groups: dict[str, list[float]] = defaultdict(list)
        for row in selected_rows:
            selected_groups[str(row["episode_id_source"])].append(float(row["success"]))
        group_rates = [float(np.mean(values)) for values in selected_groups.values()]
        low, high = bootstrap_ci(group_rates)
        state_by_pair = {str(row["pair_key"]): row for row in state_rows}
        paired_differences: dict[str, list[float]] = defaultdict(list)
        for row in selected_rows:
            pair_key = str(row["pair_key"])
            paired_differences[str(row["episode_id_source"])].append(
                float(row["success"]) - float(state_by_pair[pair_key]["canonical_success"])
            )
        difference_rates = [float(np.mean(values)) for values in paired_differences.values()]
        difference_low, difference_high = bootstrap_ci(difference_rates)
        pre_registered[condition] = {
            "states": len(selected_rows),
            "successes": sum(bool(row["success"]) for row in selected_rows),
            "state_success_rate": float(np.mean([row["success"] for row in selected_rows])),
            "source_episode_groups": len(selected_groups),
            "source_group_rate": float(np.mean(group_rates)),
            "source_group_ci": [low, high],
            "vs_canonical_source_group_difference": float(np.mean(difference_rates)),
            "vs_canonical_source_group_ci": [difference_low, difference_high],
        }
    return {
        "state_count": len(state_rows),
        "canonical_success_rate": float(np.mean([row["canonical_success"] for row in state_rows])),
        "oracle_at_97_success_rate": float(np.mean([row["oracle_at_97_success"] for row in state_rows])),
        "source_episode_groups": len(grouped),
        "canonical_source_group_rate": float(np.mean(canonical_group_rates)),
        "oracle_at_97_source_group_rate": float(np.mean(oracle_group_rates)),
        "oracle_at_97_source_group_ci": [oracle_low, oracle_high],
        "oracle_vs_canonical_source_group_difference": float(
            np.mean(oracle_group_differences)
        ),
        "oracle_vs_canonical_source_group_ci": [oracle_diff_low, oracle_diff_high],
        "mean_successful_candidate_fraction": float(
            np.mean([row["successful_candidate_count"] / 97 for row in state_rows])
        ),
        "mean_accel_success_point_biserial": (
            None if not correlations else float(np.mean(correlations))
        ),
        "pre_registered_selection_success": pre_registered,
        "state_rows": state_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", nargs="+")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    rows = load_rows(args.episodes)
    mode = str(protocol["mode"])
    analysis = (
        summarize_shortlist(rows, protocol)
        if mode == "shortlist"
        else summarize_oracle(rows, protocol)
    )
    payload = {
        "schema": "dsol_accel_gate_a97_analysis_v1",
        "status": "PASS",
        "mode": mode,
        "model": str(protocol["model"]),
        "protocol": str(args.protocol.resolve()),
        "statistical_unit": protocol["statistical_unit"],
        **analysis,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
