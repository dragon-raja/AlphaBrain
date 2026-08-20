#!/usr/bin/env python3
"""Join fixed-state Accel rankings with matched constructed-M1 outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from summarize_dsol_libero_m1_visibility import load_rows, paired_bootstrap, paired_rows


ROLE_TO_CONDITION = {
    "canonical": "canonical_both",
    "strong_info": "strong_info_both",
    "matched_control": "matched_control_both",
    "blind": "blind_both",
}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def load_accel(root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for record_path in sorted((root / "states").glob("*/rank_record.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "PASS":
            raise ValueError(f"Accel state did not PASS: {record_path}")
        pair_key = str(record["pair_key"])
        if pair_key in result:
            raise ValueError(f"duplicate Accel pair: {pair_key}")
        rankings = json.loads(
            (record_path.parent / "rankings.json").read_text(encoding="utf-8")
        )
        result[pair_key] = {"record": record, "rankings": rankings}
    if not result:
        raise ValueError(f"no Accel rank records in {root}")
    return result


def _score_by_candidate(rankings: Mapping[str, Any]) -> dict[str, float]:
    return {
        str(row["candidate_id"]): float(row["accel_3"])
        for row in rankings["complete"]["ranking"]
    }


def join_state(
    accel: Mapping[str, Any], m1: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    record = accel["record"]
    physics_hashes = {
        str(row["initial_metrics"]["physics_state_sha256"]) for row in m1.values()
    }
    if physics_hashes != {str(record["fixed_state_audit"]["physics_state_sha256"])}:
        raise ValueError(f"Accel/M1 physical state mismatch: {record['pair_key']}")
    scores = _score_by_candidate(accel["rankings"])
    role_candidates = {
        role: str(record["role_metrics"][role]["candidate_id"])
        for role in ROLE_TO_CONDITION
    }
    role_scores = {role: scores[candidate_id] for role, candidate_id in role_candidates.items()}
    accel_role = min(role_scores, key=lambda role: (role_scores[role], role))
    outcomes = {
        role: {
            "condition": condition,
            "success": bool(m1[condition]["success"]),
            "completion_steps": int(m1[condition]["completion_steps"]),
            "accel_3": role_scores[role],
        }
        for role, condition in ROLE_TO_CONDITION.items()
    }
    successful_roles = [role for role, row in outcomes.items() if row["success"]]
    if successful_roles:
        best_steps = min(outcomes[role]["completion_steps"] for role in successful_roles)
        efficiency_oracle_roles = sorted(
            role
            for role in successful_roles
            if outcomes[role]["completion_steps"] == best_steps
        )
    else:
        best_steps = None
        efficiency_oracle_roles = []
    return {
        "pair_key": str(record["pair_key"]),
        "task_id": str(record["task_id"]),
        "source_episode_group": str(record["source_episode_id"]),
        "physics_state_sha256": next(iter(physics_hashes)),
        "accel_selected_role_evaluated4": accel_role,
        "accel_selected_candidate_id": role_candidates[accel_role],
        "accel_selected_success": outcomes[accel_role]["success"],
        "canonical_success": outcomes["canonical"]["success"],
        "any_evaluated_view_success": bool(successful_roles),
        "successful_roles": successful_roles,
        "accel_selected_in_success_set": accel_role in successful_roles,
        "efficiency_oracle_defined": bool(efficiency_oracle_roles),
        "efficiency_oracle_roles": efficiency_oracle_roles,
        "efficiency_oracle_completion_steps": best_steps,
        "accel_exact_efficiency_oracle_match": accel_role in efficiency_oracle_roles,
        "outcomes": outcomes,
    }


def grouped_success_differences(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in records:
        grouped[str(row["source_episode_group"])].append(
            float(row["accel_selected_success"]) - float(row["canonical_success"])
        )
    return np.asarray(
        [float(np.mean(values)) for _, values in sorted(grouped.items())],
        dtype=np.float64,
    )


def summarize(
    records: Sequence[Mapping[str, Any]], *, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    differences = grouped_success_differences(records)
    ci = paired_bootstrap(differences, seed=seed, samples=bootstrap_samples)
    oracle_defined = [row for row in records if row["efficiency_oracle_defined"]]
    any_success = [row for row in records if row["any_evaluated_view_success"]]
    return {
        "schema": "dsol_constructed_accel_m1_join_v1",
        "status": "PASS",
        "paired_state_count": len(records),
        "independent_source_episode_group_count": len(differences),
        "physics_state_mismatch_count": 0,
        "accel_selected_role_counts": dict(
            Counter(str(row["accel_selected_role_evaluated4"]) for row in records)
        ),
        "canonical_success_rate": float(np.mean([row["canonical_success"] for row in records])),
        "accel_selected_success_rate": float(
            np.mean([row["accel_selected_success"] for row in records])
        ),
        "accel_minus_canonical_success_pp": float(differences.mean() * 100.0),
        "paired_source_episode_bootstrap_95_pp": [value * 100.0 for value in ci],
        "any_evaluated_view_success_rate": float(
            np.mean([row["any_evaluated_view_success"] for row in records])
        ),
        "accel_selected_in_success_set_rate_when_any_success": (
            None
            if not any_success
            else float(np.mean([row["accel_selected_in_success_set"] for row in any_success]))
        ),
        "efficiency_oracle_defined_state_count": len(oracle_defined),
        "accel_exact_efficiency_oracle_match_rate": (
            None
            if not oracle_defined
            else float(
                np.mean(
                    [row["accel_exact_efficiency_oracle_match"] for row in oracle_defined]
                )
            )
        ),
        "oracle_definition": (
            "Among canonical/strong-info/matched-control/blind views, a state has "
            "a success oracle only when at least one view succeeds; the efficiency "
            "oracle is the successful view set with minimum completion steps."
        ),
        "statistical_unit": "source HDF5 demonstration; frame states clustered within source",
        "claim_scope": "frozen evaluated-four relation; look-away and sensor controls lack M1 oracle outcomes",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accel-root", type=Path, required=True)
    parser.add_argument("--m1-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accel = load_accel(args.accel_root)
    episode_paths = sorted(args.m1_root.glob("episodes-shard-*.jsonl"))
    if not episode_paths:
        raise ValueError(f"no M1 episode shards in {args.m1_root}")
    m1 = paired_rows(load_rows(episode_paths))
    if set(accel) != set(m1):
        raise ValueError(
            f"Accel/M1 pair keys differ: accel={len(accel)} m1={len(m1)}"
        )
    records = [join_state(accel[pair_key], m1[pair_key]) for pair_key in sorted(accel)]
    summary = summarize(
        records, bootstrap_samples=args.bootstrap_samples, seed=args.seed
    )
    summary.update(
        {
            "accel_root": str(args.accel_root.resolve()),
            "m1_root": str(args.m1_root.resolve()),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "metrics.json", summary)
    atomic_json(args.output_dir / "state_records.json", records)
    with (args.output_dir / "state_records.csv").open("w", newline="") as handle:
        flat_rows = [
            {
                key: value
                for key, value in row.items()
                if key not in {"outcomes", "successful_roles", "efficiency_oracle_roles"}
            }
            for row in records
        ]
        writer = csv.DictWriter(handle, fieldnames=flat_rows[0].keys())
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
