#!/usr/bin/env python3
"""Summarize multi-seed repeatability for discovery-selected candidate views."""

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
        raise FileNotFoundError("no repeat episode JSONL matched")
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    episode_ids = [str(row["episode_id"]) for row in rows]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("repeat episode IDs are not unique")
    return rows


def summarize(rows: Sequence[Mapping[str, Any]], expected_seeds: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    majority = expected_seeds // 2 + 1
    by_candidate: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_candidate[(str(row["pair_key"]), str(row["selected_candidate_id"]))].append(row)
    if any(len(values) != expected_seeds for values in by_candidate.values()):
        counts = sorted({len(values) for values in by_candidate.values()})
        raise ValueError(f"candidate repeat counts differ from {expected_seeds}: {counts}")

    candidate_rows = []
    for (pair_key, candidate), values in sorted(by_candidate.items()):
        roles = values[0]["selection_metadata"].get("repeat_selection_roles", [])
        candidate_rows.append(
            {
                "pair_key": pair_key,
                "task_id": str(values[0]["task_id"]),
                "category": str(values[0]["diagnostic_role"]).split("::", 1)[-1],
                "candidate_id": candidate,
                "roles": "|".join(roles),
                "discovery_success": int(bool(values[0]["selection_metadata"]["discovery_success"])),
                "repeat_successes": sum(bool(row["success"]) for row in values),
                "repeat_success_rate": float(np.mean([row["success"] for row in values])),
                "stable_positive_2of3": int(
                    sum(bool(row["success"]) for row in values) >= majority
                ),
                "mean_completion_steps": float(np.mean([row["completion_steps"] for row in values])),
            }
        )

    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_state[row["pair_key"]].append(row)
    state_rows = []
    for pair_key, values in sorted(by_state.items()):
        canonical = next(row for row in values if row["candidate_id"] == "canonical")
        visibility = next(
            row for row in values if "visibility_top1" in row["roles"].split("|")
        )
        best = max(values, key=lambda row: (row["repeat_success_rate"], -row["mean_completion_steps"], row["candidate_id"]))
        state_rows.append(
            {
                "pair_key": pair_key,
                "task_id": values[0]["task_id"],
                "category": values[0]["category"],
                "canonical_repeat_success_rate": canonical["repeat_success_rate"],
                "visibility_repeat_success_rate": visibility["repeat_success_rate"],
                "best_shortlist_candidate_id": best["candidate_id"],
                "best_shortlist_repeat_success_rate": best["repeat_success_rate"],
                "stable_positive_candidates": sum(row["stable_positive_2of3"] for row in values),
                "shortlist_candidates": len(values),
            }
        )

    role_values: dict[str, list[float]] = defaultdict(list)
    for row in candidate_rows:
        for role in row["roles"].split("|"):
            if role:
                role_values[role].append(float(row["repeat_success_rate"]))
    role_summary = {
        role: {"candidate_states": len(values), "mean_repeat_success_rate": float(np.mean(values))}
        for role, values in sorted(role_values.items())
    }
    discovery_positive = [row for row in candidate_rows if row["discovery_success"]]
    discovery_negative = [row for row in candidate_rows if not row["discovery_success"]]
    stable_rescue_states = [
        row
        for row in state_rows
        if row["canonical_repeat_success_rate"] < majority / expected_seeds
        and row["best_shortlist_repeat_success_rate"] >= majority / expected_seeds
    ]
    payload = {
        "schema": "dsol_view_repeatability_summary_v1",
        "status": "PASS",
        "evidence_role": "discovery_repeatability_only",
        "episodes": len(rows),
        "states": len(state_rows),
        "candidates": len(candidate_rows),
        "expected_noise_draws_per_candidate": expected_seeds,
        "stable_positive_minimum_successes": majority,
        "policy_noise_seeds": sorted({int(row["policy_noise_seed"]) for row in rows}),
        "canonical_mean_repeat_success_rate": float(
            np.mean([row["canonical_repeat_success_rate"] for row in state_rows])
        ),
        "visibility_mean_repeat_success_rate": float(
            np.mean([row["visibility_repeat_success_rate"] for row in state_rows])
        ),
        "best_shortlist_oracle_mean_repeat_success_rate": float(
            np.mean([row["best_shortlist_repeat_success_rate"] for row in state_rows])
        ),
        "stable_rescue_state_count": len(stable_rescue_states),
        "stable_rescue_state_fraction": len(stable_rescue_states) / len(state_rows),
        "single_run_transition": {
            "discovery_positive_candidates": len(discovery_positive),
            "discovery_negative_candidates": len(discovery_negative),
            "stable_positive_candidates": sum(
                row["stable_positive_2of3"] for row in candidate_rows
            ),
            "discovery_positive_and_stable": sum(
                row["stable_positive_2of3"] for row in discovery_positive
            ),
            "discovery_negative_but_stable": sum(
                row["stable_positive_2of3"] for row in discovery_negative
            ),
            "discovery_positive_predictive_value": (
                float(
                    np.mean(
                        [row["stable_positive_2of3"] for row in discovery_positive]
                    )
                )
                if discovery_positive
                else None
            ),
        },
        "role_summary": role_summary,
        "state_rows": state_rows,
    }
    return payload, candidate_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", nargs="+")
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload, candidate_rows = summarize(load_rows(args.episodes), args.expected_seeds)
    atomic_json(args.output_dir / "analysis.json", payload)
    write_csv(args.output_dir / "candidate_repeatability.csv", candidate_rows)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
