#!/usr/bin/env python3
"""Freeze held-out selector protocols without using held-out policy outcomes."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from scripts.dsol_paper1.build_view_value_expectation_calibration_stage import (
        candidate_records,
        candidate_summaries,
        load_results,
        load_scans,
        rank_key,
        sha256_file,
        validate_explicit_pairing,
    )
except ModuleNotFoundError:
    from build_view_value_expectation_calibration_stage import (
        candidate_records,
        candidate_summaries,
        load_results,
        load_scans,
        rank_key,
        sha256_file,
        validate_explicit_pairing,
    )


METHODS = (
    "canonical",
    "deterministic_random_candidate",
    "calibration_global_fixed_pose",
    "visibility_increment_selector",
    "entity_visibility_hmean_selector",
    "accel_ensemble_selector",
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def stable_index(label: str, count: int, root_seed: int) -> int:
    value = int.from_bytes(hashlib.sha256(f"{root_seed}::{label}".encode()).digest()[:8], "little")
    return value % count


def entity_hmean(record: Mapping[str, Any]) -> float:
    visibility = record["visibility"]
    cameras = list(visibility["camera_names"])
    values = []
    for entity in visibility["entity_names"]:
        values.append(
            float(
                np.mean([visibility["per_camera"][camera]["entities"][entity]["visible_fraction"] for camera in cameras])
            )
        )
    if not values or any(value <= 0 for value in values):
        return 0.0
    return float(len(values) / sum(1.0 / value for value in values))


def load_accel(patterns: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            for line in Path(path).read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") != "PASS" or int(row.get("ensemble_size", 0)) != 8:
                    raise ValueError("Accel ensemble record is incomplete or not size 8")
                result[row["pair_key"]] = row
    return result


def noncanonical_ids(records: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return sorted(candidate_id for candidate_id in records if candidate_id != "canonical")


def selector_candidates(
    state: Mapping[str, Any],
    scan: Mapping[str, Any],
    *,
    global_fixed: str,
    accel: Mapping[str, Any],
    random_seed: int,
) -> dict[str, str]:
    records = candidate_records(scan)
    noncanonical = noncanonical_ids(records)
    random_candidate = noncanonical[stable_index(state["pair_key"], len(noncanonical), random_seed)]
    visibility = max(
        noncanonical,
        key=lambda candidate_id: (
            float(records[candidate_id]["delta_visibility"]),
            candidate_id,
        ),
    )
    hmean = max(
        noncanonical,
        key=lambda candidate_id: (entity_hmean(records[candidate_id]), candidate_id),
    )
    accel_candidate = str(accel["selected_candidate_id"])
    if accel_candidate not in records:
        raise ValueError("Accel selected a candidate outside the operational bank")
    return {
        "canonical": "canonical",
        "deterministic_random_candidate": random_candidate,
        "calibration_global_fixed_pose": global_fixed,
        "visibility_increment_selector": visibility,
        "entity_visibility_hmean_selector": hmean,
        "accel_ensemble_selector": accel_candidate,
    }


def calibration_global_fixed(summaries: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> str:
    candidates = sorted(next(iter(summaries.values())))
    aggregate = []
    for candidate_id in candidates:
        if candidate_id == "canonical":
            continue
        values = [state[candidate_id] for state in summaries.values()]
        aggregate.append(
            {
                "candidate_id": candidate_id,
                "mean_success": float(np.mean([row["mean_success"] for row in values])),
                "mean_progress": float(np.mean([row["mean_progress"] for row in values])),
                "harm_probability": float(np.mean([row["harm_probability"] for row in values])),
                "mean_success_steps": float(np.mean([row["mean_success_steps"] for row in values])),
            }
        )
    return min(aggregate, key=rank_key)["candidate_id"]


def choose_best_rule(
    calibration_states: Sequence[Mapping[str, Any]],
    calibration_scans: Mapping[str, Mapping[str, Any]],
    calibration_accel: Mapping[str, Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Mapping[str, Any]]],
    global_fixed: str,
    random_seed: int,
) -> tuple[str, dict[str, Any]]:
    rule_values = {method: [] for method in METHODS if method != "canonical"}
    for state in calibration_states:
        selected = selector_candidates(
            state,
            calibration_scans[state["pair_key"]],
            global_fixed=global_fixed,
            accel=calibration_accel[state["pair_key"]],
            random_seed=random_seed,
        )
        for method in rule_values:
            rule_values[method].append(summaries[state["pair_key"]][selected[method]])
    rule_summary = {}
    for method, values in rule_values.items():
        rule_summary[method] = {
            "mean_success": float(np.mean([row["mean_success"] for row in values])),
            "mean_progress": float(np.mean([row["mean_progress"] for row in values])),
            "harm_probability": float(np.mean([row["harm_probability"] for row in values])),
            "mean_success_steps": float(np.mean([row["mean_success_steps"] for row in values])),
        }
    best = min(
        rule_summary,
        key=lambda method: rank_key({"candidate_id": method, **rule_summary[method]}),
    )
    return best, rule_summary


def build_protocol(
    *,
    checkpoint_seed: int,
    states: Sequence[Mapping[str, Any]],
    scans: Mapping[str, Mapping[str, Any]],
    accel: Mapping[str, Mapping[str, Any]],
    global_fixed: str,
    best_rule: str,
    catalog: Path,
    random_seed: int,
) -> dict[str, Any]:
    methods = METHODS if checkpoint_seed == 41 else ("canonical", best_rule)
    specs = []
    state_selections = {}
    for state in states:
        all_selections = selector_candidates(
            state,
            scans[state["pair_key"]],
            global_fixed=global_fixed,
            accel=accel[state["pair_key"]],
            random_seed=random_seed,
        )
        state_selections[state["pair_key"]] = {method: all_selections[method] for method in methods}
        records = candidate_records(scans[state["pair_key"]])
        for method in methods:
            candidate_id = all_selections[method]
            record = records[candidate_id]
            for repeat_id in range(32):
                identity = (
                    f"expectation-v1::heldout-E::seed-{checkpoint_seed}::"
                    f"{state['pair_key']}::{method}::{candidate_id}::{repeat_id}"
                )
                specs.append(
                    {
                        **state,
                        "condition": f"selector__{method}",
                        "selector_method": method,
                        "selected_candidate_id": candidate_id,
                        "pose": record.get("pose"),
                        "scene_construction": scans[state["pair_key"]]["scene_construction"],
                        "sensor_control": "both",
                        "catalog": str(catalog.resolve()),
                        "policy_repeat_id": repeat_id,
                        "noise_bank_id": "E",
                        "checkpoint_seed": checkpoint_seed,
                        "candidate_features": {
                            "catalog_group": record["group"],
                            "visibility_score": float(record["visibility_score"]),
                            "delta_visibility": float(record["delta_visibility"]),
                            "entity_visibility_hmean": entity_hmean(record),
                        },
                        "episode_id": hashlib.sha256(identity.encode()).hexdigest()[:24],
                    }
                )
    return {
        "schema": "dsol_view_value_expectation_heldout_protocol_v1",
        "status": "PASS",
        "split": "heldout_test",
        "checkpoint_seed": checkpoint_seed,
        "bank_id": "E",
        "policy_noise_repeats": 32,
        "selector_methods": list(methods),
        "selector_method_count": len(methods),
        "state_count": len(states),
        "source_group_count": len({state["source_group"] for state in states}),
        "episode_count": len(specs),
        "selection_uses_heldout_policy_outcomes": False,
        "calibration_global_fixed_pose": global_fixed,
        "best_noncanonical_rule": best_rule,
        "state_selections": state_selections,
        "specs": specs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--scan-ledgers", nargs="+", required=True)
    parser.add_argument("--calibration-stage-a-results", nargs="+", required=True)
    parser.add_argument("--accel-results", nargs="+", required=True)
    parser.add_argument("--random-selector-seed", type=int, default=20260931)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    population = json.loads(args.population.read_text())
    calibration_states = population["population"]["calibration"]["states"]
    heldout_states = population["population"]["heldout_test"]["states"]
    scans = load_scans(args.scan_ledgers)
    accel = load_accel(args.accel_results)
    expected_states = {row["pair_key"] for row in [*calibration_states, *heldout_states]}
    if set(accel) != expected_states:
        raise ValueError("Accel ensemble results differ from the frozen population")
    stage_a_rows = load_results(args.calibration_stage_a_results)
    validate_explicit_pairing(stage_a_rows, "A")
    summaries = candidate_summaries(stage_a_rows)
    global_fixed = calibration_global_fixed(summaries)
    best_rule, calibration_rule_summary = choose_best_rule(
        calibration_states,
        scans,
        accel,
        summaries,
        global_fixed,
        args.random_selector_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocols = {}
    for seed in (41, 42, 43):
        payload = build_protocol(
            checkpoint_seed=seed,
            states=heldout_states,
            scans=scans,
            accel=accel,
            global_fixed=global_fixed,
            best_rule=best_rule,
            catalog=args.catalog,
            random_seed=args.random_selector_seed,
        )
        path = args.output_dir / f"heldout-primary-seed{seed}.json"
        atomic_json(path, payload)
        protocols[str(seed)] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "episodes": payload["episode_count"],
        }
    receipt = {
        "schema": "dsol_view_value_expectation_heldout_freeze_v1",
        "status": "PASS",
        "selection_uses_heldout_policy_outcomes": False,
        "calibration_global_fixed_pose": global_fixed,
        "best_noncanonical_rule": best_rule,
        "calibration_rule_summary_four_noise_screening_only": calibration_rule_summary,
        "accel_ensemble_size": 8,
        "protocols": protocols,
    }
    atomic_json(args.output_dir / "heldout-freeze-receipt.json", receipt)
    print(json.dumps({"status": "PASS", "global_fixed": global_fixed, "best_rule": best_rule}, sort_keys=True))


if __name__ == "__main__":
    main()
