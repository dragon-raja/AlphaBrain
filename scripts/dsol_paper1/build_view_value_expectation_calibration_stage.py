#!/usr/bin/env python3
"""Build auditable A/B/C/D calibration stages for view-value expectation."""

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


STAGES = {
    "A": {"bank": "A", "repeats": 4, "candidate_count": 97},
    "B": {"bank": "B", "repeats": 8, "candidate_count": 24},
    "C": {"bank": "C", "repeats": 16, "candidate_count": 6},
    "D": {"bank": "D", "repeats": 64, "candidate_count": 2},
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scans(patterns: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    scans = {}
    for pattern in patterns:
        for ledger in sorted(glob.glob(pattern)):
            with open(ledger, encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("status") != "PASS":
                        raise ValueError(f"visibility scan failed: {row.get('scan_id')}")
                    scan = json.loads((Path(row["output_dir"]) / "scan.json").read_text())
                    if scan.get("status") != "PASS":
                        raise ValueError(f"scan artifact failed: {row.get('scan_id')}")
                    scans[str(row["scan_id"])] = scan
    return scans


def candidate_records(scan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = {
        str(row["pose_id"]): row
        for row in scan["records"]
        if row.get("status", "PASS") == "PASS"
        and str(row.get("pose_id", "")).startswith(("canonical", "broad_train_", "broad_heldout_"))
    }
    if len(records) != 97 or "canonical" not in records:
        raise ValueError(f"scan does not contain the operational 97-view bank: {len(records)}")
    return records


def load_results(patterns: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise FileNotFoundError("no previous-stage episode records matched")
    return rows


def validate_explicit_pairing(rows: Sequence[Mapping[str, Any]], expected_bank: str) -> None:
    noise_hashes = defaultdict(set)
    physics_hashes = defaultdict(set)
    episode_ids = set()
    for row in rows:
        if row["episode_id"] in episode_ids:
            raise ValueError("duplicate episode ID in prior results")
        episode_ids.add(row["episode_id"])
        if row.get("status") != "complete" or not row.get("explicit_flow_noise"):
            raise ValueError("previous stage is incomplete or not explicit-noise")
        if row.get("noise_bank_id") != expected_bank:
            raise ValueError("previous stage used the wrong noise bank")
        if len(row.get("policy_calls", [])) != int(row["inference_calls"]):
            raise ValueError("per-replan policy-call ledger is incomplete")
        state_repeat = (row["pair_key"], int(row["policy_repeat_id"]))
        physics_hashes[row["pair_key"]].add(
            row["initial_metrics"]["physics_state_sha256"]
        )
        for call in row["policy_calls"]:
            key = (*state_repeat, int(call["replan_index"]))
            noise_hashes[key].add(call["noise_sha256"])
            for required in ("noise_seed", "action_chunk_sha256"):
                if required not in call:
                    raise ValueError(f"missing policy-call field: {required}")
    if any(len(values) != 1 for values in physics_hashes.values()):
        raise ValueError("physics state differs across paired views or repeats")
    if any(len(values) != 1 for values in noise_hashes.values()):
        raise ValueError("explicit policy noise differs across paired views")


def candidate_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_key"]), str(row["selected_candidate_id"])].append(row)
    result = defaultdict(dict)
    by_state_repeat = {
        (str(row["pair_key"]), str(row["selected_candidate_id"]), int(row["policy_repeat_id"])): row
        for row in rows
    }
    for (state, candidate), values in grouped.items():
        repeats = {int(row["policy_repeat_id"]) for row in values}
        canonical_by_repeat = {
            repeat: by_state_repeat.get((state, "canonical", repeat)) for repeat in repeats
        }
        harms = [
            bool(base["success"] and not row["success"])
            for row in values
            for base in [canonical_by_repeat[int(row["policy_repeat_id"])]]
            if base is not None
        ]
        successful_steps = [int(row["completion_steps"]) for row in values if row["success"]]
        result[state][candidate] = {
            "candidate_id": candidate,
            "mean_success": float(np.mean([row["success"] for row in values])),
            "mean_progress": float(np.mean([row["normalized_final_progress"] for row in values])),
            "harm_probability": float(np.mean(harms)) if harms else 0.0,
            "mean_success_steps": float(np.mean(successful_steps)) if successful_steps else math.inf,
            "repeat_count": len(repeats),
        }
    return {key: dict(value) for key, value in result.items()}


def rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["mean_success"]),
        -float(row["mean_progress"]),
        float(row["harm_probability"]),
        float(row["mean_success_steps"]),
        str(row["candidate_id"]),
    )


def select_candidates(
    stage: str,
    states: Sequence[Mapping[str, Any]],
    scans: Mapping[str, Mapping[str, Any]],
    previous_rows: Sequence[Mapping[str, Any]] | None,
) -> dict[str, list[str]]:
    if stage == "A":
        return {state["pair_key"]: sorted(candidate_records(scans[state["pair_key"]])) for state in states}
    if previous_rows is None:
        raise ValueError(f"stage {stage} requires previous-stage results")
    summaries = candidate_summaries(previous_rows)
    selections = {}
    for state in states:
        key = state["pair_key"]
        ranked = sorted(summaries[key].values(), key=rank_key)
        if stage == "B":
            train = [row["candidate_id"] for row in ranked if row["candidate_id"].startswith("broad_train_")][:15]
            heldout = [row["candidate_id"] for row in ranked if row["candidate_id"].startswith("broad_heldout_")][:8]
            chosen = ["canonical", *train, *heldout]
        elif stage == "C":
            chosen = ["canonical", *[row["candidate_id"] for row in ranked if row["candidate_id"] != "canonical"][:5]]
        elif stage == "D":
            chosen = ["canonical", next(row["candidate_id"] for row in ranked if row["candidate_id"] != "canonical")]
        else:
            raise ValueError(f"unsupported stage: {stage}")
        if len(chosen) != STAGES[stage]["candidate_count"] or len(set(chosen)) != len(chosen):
            raise ValueError(f"invalid {stage} selection for {key}: {chosen}")
        selections[key] = chosen
    return selections


def build_specs(
    stage: str,
    states: Sequence[Mapping[str, Any]],
    scans: Mapping[str, Mapping[str, Any]],
    selections: Mapping[str, Sequence[str]],
    catalog_path: Path,
) -> list[dict[str, Any]]:
    bank = STAGES[stage]["bank"]
    repeats = STAGES[stage]["repeats"]
    specs = []
    for state in states:
        pair_key = state["pair_key"]
        records = candidate_records(scans[pair_key])
        for candidate_id in selections[pair_key]:
            record = records[candidate_id]
            for repeat_id in range(repeats):
                identity = f"expectation-v1::{stage}::{pair_key}::{candidate_id}::{repeat_id}"
                specs.append(
                    {
                        **state,
                        "condition": f"candidate__{candidate_id}",
                        "diagnostic_role": f"view_value_expectation_calibration_stage_{stage}",
                        "selected_candidate_id": candidate_id,
                        "pose": record.get("pose"),
                        "scene_construction": scans[pair_key]["scene_construction"],
                        "sensor_control": "both",
                        "catalog": str(catalog_path.resolve()),
                        "policy_repeat_id": repeat_id,
                        "noise_bank_id": bank,
                        "candidate_features": {
                            "catalog_group": record["group"],
                            "visibility_score": float(record["visibility_score"]),
                            "delta_visibility": float(record["delta_visibility"]),
                            "per_camera_scores": record.get("per_camera_scores"),
                            "camera_displacement_from_canonical": record.get("camera_displacement_from_canonical"),
                        },
                        "episode_id": hashlib.sha256(identity.encode()).hexdigest()[:24],
                    }
                )
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--scan-ledgers", nargs="+", required=True)
    parser.add_argument("--previous-results", nargs="*")
    parser.add_argument("--previous-protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    population = json.loads(args.population.read_text())
    states = population["population"]["calibration"]["states"]
    scans = load_scans(args.scan_ledgers)
    if set(scans) != {state["pair_key"] for state in [
        *population["population"]["calibration"]["states"],
        *population["population"]["heldout_test"]["states"],
    ]}:
        raise ValueError("visibility scan set differs from the frozen population")
    previous_rows = load_results(args.previous_results) if args.previous_results else None
    previous_stage = chr(ord(args.stage) - 1) if args.stage != "A" else None
    if previous_rows is not None:
        validate_explicit_pairing(previous_rows, expected_bank=previous_stage)
    selections = select_candidates(args.stage, states, scans, previous_rows)
    specs = build_specs(args.stage, states, scans, selections, args.catalog)
    payload = {
        "schema": "dsol_view_value_expectation_calibration_stage_v1",
        "status": "PASS",
        "stage": args.stage,
        "bank_id": STAGES[args.stage]["bank"],
        "candidate_count_per_state": STAGES[args.stage]["candidate_count"],
        "policy_noise_repeats": STAGES[args.stage]["repeats"],
        "state_count": len(states),
        "source_group_count": len({state["source_group"] for state in states}),
        "episode_count": len(specs),
        "selection_uses_current_stage_outcomes": False,
        "selection_previous_stage": previous_stage,
        "selection_rule": {
            "A": "all_97_frozen_candidates",
            "B": "canonical_plus_previous_stage_top15_train_support_and_top8_heldout_by_preregistered_rank",
            "C": "canonical_plus_previous_stage_top5_noncanonical_by_preregistered_rank",
            "D": "canonical_plus_previous_stage_top1_noncanonical_by_preregistered_rank",
        }[args.stage],
        "rank_order": [
            "higher_success_probability",
            "higher_normalized_final_progress",
            "lower_harm_probability",
            "lower_completion_steps_conditional_on_success",
            "lexicographic_candidate_id",
        ],
        "population": str(args.population.resolve()),
        "population_sha256": sha256_file(args.population),
        "catalog": str(args.catalog.resolve()),
        "catalog_sha256": sha256_file(args.catalog),
        "previous_protocol": None if args.previous_protocol is None else str(args.previous_protocol.resolve()),
        "previous_protocol_sha256": None if args.previous_protocol is None else sha256_file(args.previous_protocol),
        "selected_candidates": selections,
        "specs": specs,
    }
    expected = len(states) * STAGES[args.stage]["candidate_count"] * STAGES[args.stage]["repeats"]
    if len(specs) != expected:
        raise AssertionError(f"episode budget mismatch: {len(specs)} != {expected}")
    atomic_json(args.output, payload)
    print(json.dumps({"status": "PASS", "stage": args.stage, "episodes": len(specs)}, sort_keys=True))


if __name__ == "__main__":
    main()
