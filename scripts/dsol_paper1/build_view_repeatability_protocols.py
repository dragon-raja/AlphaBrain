#!/usr/bin/env python3
"""Build multi-seed repeatability protocols from dense-view discovery outcomes."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
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
        raise FileNotFoundError("no discovery episodes matched")
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def normalized_pose_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_pose = left.get("pose") or {}
    right_pose = right.get("pose") or {}
    return math.sqrt(
        ((float(left_pose.get("azimuth_deg", 0.0)) - float(right_pose.get("azimuth_deg", 0.0))) / 60.0) ** 2
        + ((float(left_pose.get("elevation_deg", 0.0)) - float(right_pose.get("elevation_deg", 0.0))) / 25.0) ** 2
        + ((float(left_pose.get("radius_scale", 1.0)) - float(right_pose.get("radius_scale", 1.0))) / 0.30) ** 2
    )


def select_state_categories(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    by_state: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_state[str(row["pair_key"])].append(row)
    summaries = []
    for pair_key, values in by_state.items():
        canonical = next(row for row in values if row["selected_candidate_id"] == "canonical")
        summaries.append(
            {
                "pair_key": pair_key,
                "canonical_success": bool(canonical["success"]),
                "success_fraction": float(np.mean([row["success"] for row in values])),
            }
        )
    failures = sorted(
        [row for row in summaries if not row["canonical_success"]],
        key=lambda row: (row["success_fraction"], row["pair_key"]),
    )
    successes = sorted(
        [row for row in summaries if row["canonical_success"]],
        key=lambda row: (row["success_fraction"], row["pair_key"]),
    )
    if len(failures) < 2 or len(successes) < 2:
        raise ValueError("repeatability requires at least two canonical failures and successes")
    return {
        "canonical_failure_sparse": str(failures[0]["pair_key"]),
        "canonical_failure_broad": str(failures[-1]["pair_key"]),
        "canonical_success_harm": str(successes[0]["pair_key"]),
        "canonical_success_mixed": str(successes[1]["pair_key"]),
    }


def best_global_candidate(rows: Sequence[Mapping[str, Any]]) -> str:
    by_candidate: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_candidate[str(row["selected_candidate_id"])].append(bool(row["success"]))
    return sorted(
        by_candidate,
        key=lambda candidate: (-float(np.mean(by_candidate[candidate])), candidate),
    )[0]


def choose_candidates(
    values: Sequence[Mapping[str, Any]], global_candidate: str, shortlist_size: int
) -> dict[str, list[str]]:
    by_id = {str(row["selected_candidate_id"]): row for row in values}
    successes = [row for row in values if row["success"]]
    failures = [row for row in values if not row["success"]]
    fastest = min(successes, key=lambda row: (row["completion_steps"], row["selected_candidate_id"]))
    visibility_high = max(values, key=lambda row: (row["initial_metrics"]["task_entity_visibility"]["score"], row["selected_candidate_id"]))
    visibility_low = min(values, key=lambda row: (row["initial_metrics"]["task_entity_visibility"]["score"], row["selected_candidate_id"]))
    success_high = max(successes, key=lambda row: (row["initial_metrics"]["task_entity_visibility"]["score"], row["selected_candidate_id"]))
    success_low = min(successes, key=lambda row: (row["initial_metrics"]["task_entity_visibility"]["score"], row["selected_candidate_id"]))
    if failures:
        nearest_failure = min(
            failures,
            key=lambda row: (normalized_pose_distance(row, fastest), row["selected_candidate_id"]),
        )
    else:
        nearest_failure = visibility_low

    requested = [
        ("canonical", "canonical"),
        ("visibility_top1", str(visibility_high["selected_candidate_id"])),
        ("visibility_bottom1", str(visibility_low["selected_candidate_id"])),
        ("discovery_fastest_success", str(fastest["selected_candidate_id"])),
        ("highest_visibility_success", str(success_high["selected_candidate_id"])),
        ("lowest_visibility_success", str(success_low["selected_candidate_id"])),
        ("nearest_failure_to_fastest", str(nearest_failure["selected_candidate_id"])),
        ("discovery_global_fixed", global_candidate),
    ]
    roles: dict[str, list[str]] = defaultdict(list)
    selected = []
    for role, candidate in requested:
        roles[candidate].append(role)
        if candidate not in selected:
            selected.append(candidate)

    while len(selected) < shortlist_size:
        remaining = [candidate for candidate in sorted(by_id) if candidate not in selected]
        candidate = max(
            remaining,
            key=lambda item: (
                min(normalized_pose_distance(by_id[item], by_id[chosen]) for chosen in selected),
                item,
            ),
        )
        selected.append(candidate)
        roles[candidate].append("pose_diverse_fill")
    return {candidate: roles[candidate] for candidate in selected[:shortlist_size]}


def build_protocols(
    discovery_rows: Sequence[Mapping[str, Any]],
    base_protocol: Mapping[str, Any],
    *,
    eval_seeds: Sequence[int],
    shortlist_size: int,
) -> dict[int, dict[str, Any]]:
    categories = select_state_categories(discovery_rows)
    selected_pairs = {pair_key: category for category, pair_key in categories.items()}
    by_state: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in discovery_rows:
        by_state[str(row["pair_key"])].append(row)
    global_candidate = best_global_candidate(discovery_rows)
    base_specs = {
        (str(spec["pair_key"]), str(spec["selected_candidate_id"])): spec
        for spec in base_protocol["specs"]
    }
    selection = {
        pair_key: choose_candidates(by_state[pair_key], global_candidate, shortlist_size)
        for pair_key in selected_pairs
    }

    protocols = {}
    for seed in eval_seeds:
        specs = []
        selected_states = []
        for pair_key, category in sorted(selected_pairs.items()):
            candidates = selection[pair_key]
            selected_states.append(
                {
                    "pair_key": pair_key,
                    "category": category,
                    "selected_candidates": candidates,
                }
            )
            for candidate, roles in candidates.items():
                spec = dict(base_specs[(pair_key, candidate)])
                metadata = dict(spec.get("selection_metadata") or {})
                metadata["repeat_selection_roles"] = roles
                metadata["discovery_success"] = bool(
                    next(
                        row["success"]
                        for row in by_state[pair_key]
                        if row["selected_candidate_id"] == candidate
                    )
                )
                spec["selection_metadata"] = metadata
                spec["diagnostic_role"] = f"view_repeatability::{category}"
                spec["episode_id"] = hashlib.sha256(
                    f"{pair_key}::{candidate}::seed-{seed}".encode()
                ).hexdigest()[:20]
                specs.append(spec)
        protocols[seed] = {
            "schema": "dsol_view_repeatability_protocol_v1",
            "status": "PASS",
            "analysis_role": "discovery_candidate_repeatability",
            "confirmatory_test_eligible": False,
            "selection_uses_policy_outcomes": True,
            "statistical_unit": "source HDF5 demonstration",
            "catalog": base_protocol["catalog"],
            "eval_seed": seed,
            "state_count": len(selected_states),
            "candidate_count_per_state": shortlist_size,
            "episode_count": len(specs),
            "global_fixed_candidate": global_candidate,
            "selected_states": selected_states,
            "specs": specs,
        }
    return protocols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", nargs="+")
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--eval-seeds", default="20260831,20260832,20260833")
    parser.add_argument("--shortlist-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocols = build_protocols(
        load_rows(args.episodes),
        json.loads(args.base_protocol.read_text(encoding="utf-8")),
        eval_seeds=[int(value) for value in args.eval_seeds.split(",")],
        shortlist_size=args.shortlist_size,
    )
    for seed, protocol in protocols.items():
        atomic_json(args.output_dir / f"protocol-seed-{seed}.json", protocol)
    print(
        json.dumps(
            {
                "seeds": sorted(protocols),
                "states": next(iter(protocols.values()))["state_count"],
                "episodes_per_seed": next(iter(protocols.values()))["episode_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
