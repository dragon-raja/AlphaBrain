#!/usr/bin/env python3
"""Summarize frozen dense-test selectors across repeated policy noise."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.dsol_paper1.summarize_view_repeatability import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    source_group_from_pair_key,
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


def load_rows(patterns: Sequence[str]) -> list[dict[str, Any]]:
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError("no dense-test episode rows matched")
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    identities = [
        (str(row["episode_id"]), int(row["policy_noise_seed"])) for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("dense-test episode/noise identities are not unique")
    if any(row.get("status") != "complete" for row in rows):
        raise ValueError("all dense-test episodes must be complete")
    return rows


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def summarize(
    rows: Sequence[Mapping[str, Any]], expected_repeats: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    majority = expected_repeats // 2 + 1
    majority_rate = majority / expected_repeats
    by_condition: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        condition = str(row["condition"])
        if not condition.startswith("selector__"):
            raise ValueError(f"unexpected dense-test condition: {condition}")
        method = condition.split("selector__", 1)[1]
        by_condition[(str(row["pair_key"]), method)].append(row)
    counts = {len(values) for values in by_condition.values()}
    if counts != {expected_repeats}:
        raise ValueError(f"selector repeat counts differ from {expected_repeats}: {counts}")

    state_method_rows = []
    for (pair_key, method), values in sorted(by_condition.items()):
        state_method_rows.append(
            {
                "pair_key": pair_key,
                "source_group": source_group_from_pair_key(pair_key),
                "task_id": str(values[0]["task_id"]),
                "selector_method": method,
                "selected_candidate_id": str(values[0]["selected_candidate_id"]),
                "repeat_successes": sum(bool(row["success"]) for row in values),
                "repeat_success_rate": float(
                    np.mean([bool(row["success"]) for row in values])
                ),
                "stable_success": int(
                    sum(bool(row["success"]) for row in values) >= majority
                ),
                "mean_completion_steps": float(
                    np.mean([int(row["completion_steps"]) for row in values])
                ),
            }
        )
    by_state: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in state_method_rows:
        by_state[row["pair_key"]][row["selector_method"]] = row
    methods = sorted({row["selector_method"] for row in state_method_rows})
    if "canonical" not in methods:
        raise ValueError("dense-test selector results omit canonical")
    if any(set(values) != set(methods) for values in by_state.values()):
        raise ValueError("selector methods differ across test states")

    grouped_states: dict[str, list[str]] = defaultdict(list)
    for pair_key, values in by_state.items():
        grouped_states[str(values["canonical"]["source_group"])].append(pair_key)
    group_names = sorted(grouped_states)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_indices = rng.integers(
        0, len(group_names), size=(BOOTSTRAP_RESAMPLES, len(group_names))
    )
    selector_summary = {}
    for method in methods:
        rates = np.asarray(
            [by_state[pair_key][method]["repeat_success_rate"] for pair_key in by_state],
            dtype=np.float64,
        )
        canonical_rates = np.asarray(
            [
                by_state[pair_key]["canonical"]["repeat_success_rate"]
                for pair_key in by_state
            ],
            dtype=np.float64,
        )
        group_rates = []
        group_differences = []
        for group_name in group_names:
            pair_keys = grouped_states[group_name]
            group_rates.append(
                float(np.mean([by_state[key][method]["repeat_success_rate"] for key in pair_keys]))
            )
            group_differences.append(
                float(
                    100
                    * np.mean(
                        [
                            by_state[key][method]["repeat_success_rate"]
                            - by_state[key]["canonical"]["repeat_success_rate"]
                            for key in pair_keys
                        ]
                    )
                )
            )
        group_rates = np.asarray(group_rates)
        group_differences = np.asarray(group_differences)
        rate_draws = group_rates[bootstrap_indices].mean(axis=1)
        difference_draws = group_differences[bootstrap_indices].mean(axis=1)
        rate_low, rate_high = percentile_interval(rate_draws)
        diff_low, diff_high = percentile_interval(difference_draws)
        selector_summary[method] = {
            "mean_repeat_success_rate": float(rates.mean()),
            "success_rate_ci_low": rate_low,
            "success_rate_ci_high": rate_high,
            "difference_from_canonical_pp": float(100 * (rates - canonical_rates).mean()),
            "difference_ci_low_pp": diff_low,
            "difference_ci_high_pp": diff_high,
            "stable_success_state_count": sum(
                by_state[key][method]["stable_success"] for key in by_state
            ),
            "stable_rescue_state_count": sum(
                by_state[key]["canonical"]["repeat_success_rate"] < majority_rate
                and by_state[key][method]["repeat_success_rate"] >= majority_rate
                for key in by_state
            ),
            "stable_harm_state_count": sum(
                by_state[key]["canonical"]["repeat_success_rate"] >= majority_rate
                and by_state[key][method]["repeat_success_rate"] < majority_rate
                for key in by_state
            ),
        }

    task_summary = {}
    task_ids = sorted({row["task_id"] for row in state_method_rows})
    for task_id in task_ids:
        task_summary[task_id] = {}
        for method in methods:
            selected = [
                row
                for row in state_method_rows
                if row["task_id"] == task_id and row["selector_method"] == method
            ]
            task_summary[task_id][method] = {
                "states": len(selected),
                "mean_repeat_success_rate": float(
                    np.mean([row["repeat_success_rate"] for row in selected])
                ),
            }
    payload = {
        "schema": "dsol_dense_test_selector_summary_v1",
        "status": "PASS",
        "evidence_role": "independent_test_frozen_selector_evaluation",
        "selection_uses_test_policy_outcomes": False,
        "episodes": len(rows),
        "states": len(by_state),
        "source_groups": len(group_names),
        "selector_methods": methods,
        "expected_repeats": expected_repeats,
        "stable_success_minimum_repeats": majority,
        "bootstrap": {
            "independent_unit": "source_episode",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "selector_summary": selector_summary,
        "task_summary": task_summary,
        "state_method_rows": state_method_rows,
    }
    return payload, state_method_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", nargs="+")
    parser.add_argument("--expected-repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload, state_rows = summarize(load_rows(args.episodes), args.expected_repeats)
    atomic_json(args.output_dir / "analysis.json", payload)
    write_csv(args.output_dir / "state_method_results.csv", state_rows)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
