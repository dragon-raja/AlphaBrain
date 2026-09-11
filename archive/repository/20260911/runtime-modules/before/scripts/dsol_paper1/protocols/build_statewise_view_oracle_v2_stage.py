#!/usr/bin/env python3
"""Build independent P refinement and Q confirmation protocols for oracle v2."""

from __future__ import annotations

# Repository-local CLI bootstrap: path setup only.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[3]
for _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):
    if str(_layout_path) not in _layout_sys.path:
        _layout_sys.path.insert(0, str(_layout_path))


import argparse
import glob
import json
import math
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from scripts.dsol_paper1.protocols.build_statewise_view_oracle_v2 import freeze_json, load_scan_assets, operational_candidates
    from scripts.dsol_paper1.explicit_flow_noise import sha256_file
except ImportError:  # Direct script execution.
    from scripts.dsol_paper1.protocols.build_statewise_view_oracle_v2 import freeze_json, load_scan_assets, operational_candidates
    from scripts.dsol_paper1.explicit_flow_noise import sha256_file


@dataclass
class CandidateAccumulator:
    success_sum: int = 0
    progress_sum: float = 0.0
    harm_sum: int = 0
    success_steps_sum: int = 0
    success_steps_count: int = 0
    repeats: set[int] = field(default_factory=set)


def matched_paths(patterns: Sequence[str]) -> list[Path]:
    paths = []
    for pattern in patterns:
        paths.extend(Path(value) for value in sorted(glob.glob(pattern)))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError("no input episode ledgers matched")
    return paths


def iter_rows(paths: Sequence[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def summarize_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_bank: str, expected_repeats: int
) -> dict[str, dict[str, dict[str, Any]]]:
    canonical = {}
    episode_ids = set()
    for row in rows:
        if row.get("status") != "complete" or not row.get("explicit_flow_noise"):
            raise ValueError("input includes an incomplete or non-explicit-noise episode")
        if row.get("noise_bank_id") != expected_bank:
            raise ValueError("input includes the wrong noise bank")
        episode_id = str(row["episode_id"])
        if episode_id in episode_ids:
            raise ValueError(f"duplicate input episode: {episode_id}")
        episode_ids.add(episode_id)
        if row["selected_candidate_id"] == "canonical":
            canonical[str(row["pair_key"]), int(row["policy_repeat_id"])] = bool(
                row["success"]
            )
    accumulators: dict[tuple[str, str], CandidateAccumulator] = defaultdict(CandidateAccumulator)
    for row in rows:
        state = str(row["pair_key"])
        candidate = str(row["selected_candidate_id"])
        repeat = int(row["policy_repeat_id"])
        base = canonical.get((state, repeat))
        if base is None:
            raise ValueError(f"missing paired canonical outcome: {state}/{repeat}")
        value = accumulators[state, candidate]
        success = bool(row["success"])
        value.success_sum += int(success)
        value.progress_sum += float(row["normalized_final_progress"])
        value.harm_sum += int(base and not success)
        if success:
            value.success_steps_sum += int(row["completion_steps"])
            value.success_steps_count += 1
        value.repeats.add(repeat)
    summaries: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (state, candidate), value in accumulators.items():
        if len(value.repeats) != expected_repeats:
            raise ValueError(
                f"candidate repeat count differs from {expected_repeats}: {state}/{candidate}"
            )
        count = len(value.repeats)
        summaries[state][candidate] = {
            "candidate_id": candidate,
            "repeat_count": count,
            "mean_success": value.success_sum / count,
            "mean_progress": value.progress_sum / count,
            "harm_probability": value.harm_sum / count,
            "mean_success_steps": (
                value.success_steps_sum / value.success_steps_count
                if value.success_steps_count
                else math.inf
            ),
        }
    return {state: dict(values) for state, values in summaries.items()}


def summarize_ledgers(
    paths: Sequence[Path], *, expected_bank: str, expected_repeats: int
) -> dict[str, dict[str, dict[str, Any]]]:
    canonical = {}
    seen = set()
    for row in iter_rows(paths):
        if row.get("status") != "complete" or not row.get("explicit_flow_noise"):
            raise ValueError("input includes an incomplete or non-explicit-noise episode")
        if row.get("noise_bank_id") != expected_bank:
            raise ValueError("input includes the wrong noise bank")
        episode_id = str(row["episode_id"])
        if episode_id in seen:
            raise ValueError(f"duplicate input episode: {episode_id}")
        seen.add(episode_id)
        if row["selected_candidate_id"] == "canonical":
            key = (str(row["pair_key"]), int(row["policy_repeat_id"]))
            if key in canonical:
                raise ValueError(f"duplicate paired canonical outcome: {key}")
            canonical[key] = bool(row["success"])
    accumulators: dict[tuple[str, str], CandidateAccumulator] = defaultdict(CandidateAccumulator)
    seen.clear()
    for row in iter_rows(paths):
        episode_id = str(row["episode_id"])
        if episode_id in seen:
            raise ValueError(f"duplicate input episode: {episode_id}")
        seen.add(episode_id)
        state = str(row["pair_key"])
        candidate = str(row["selected_candidate_id"])
        repeat = int(row["policy_repeat_id"])
        base = canonical.get((state, repeat))
        if base is None:
            raise ValueError(f"missing paired canonical outcome: {state}/{repeat}")
        value = accumulators[state, candidate]
        success = bool(row["success"])
        value.success_sum += int(success)
        value.progress_sum += float(row["normalized_final_progress"])
        value.harm_sum += int(base and not success)
        if success:
            value.success_steps_sum += int(row["completion_steps"])
            value.success_steps_count += 1
        value.repeats.add(repeat)
    summaries: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (state, candidate), value in accumulators.items():
        if len(value.repeats) != expected_repeats:
            raise ValueError(
                f"candidate repeat count differs from {expected_repeats}: {state}/{candidate}"
            )
        count = len(value.repeats)
        summaries[state][candidate] = {
            "candidate_id": candidate,
            "repeat_count": count,
            "mean_success": value.success_sum / count,
            "mean_progress": value.progress_sum / count,
            "harm_probability": value.harm_sum / count,
            "mean_success_steps": (
                value.success_steps_sum / value.success_steps_count
                if value.success_steps_count
                else math.inf
            ),
        }
    return {state: dict(values) for state, values in summaries.items()}


def rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["mean_success"]),
        -float(row["mean_progress"]),
        float(row["harm_probability"]),
        float(row["mean_success_steps"]),
        str(row["candidate_id"]),
    )


def select_noncanonical(
    summaries: Mapping[str, Mapping[str, Mapping[str, Any]]], count: int
) -> dict[str, list[str]]:
    selections = {}
    for state, candidates in summaries.items():
        if "canonical" not in candidates:
            raise ValueError(f"state lacks canonical results: {state}")
        ranked = sorted(
            (row for candidate, row in candidates.items() if candidate != "canonical"),
            key=rank_key,
        )
        if len(ranked) < count:
            raise ValueError(f"state has fewer than {count} noncanonical candidates: {state}")
        selections[state] = [str(row["candidate_id"]) for row in ranked[:count]]
    return selections


def validate_input_audits(patterns: Sequence[str], expected_count: int) -> list[dict[str, Any]]:
    paths = matched_paths(patterns)
    audits = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(audits) != expected_count:
        raise ValueError(f"expected {expected_count} input audits, found {len(audits)}")
    if any(row.get("status") != "PASS_COMPLETE" for row in audits):
        raise ValueError("one or more input integrity audits did not PASS_COMPLETE")
    return [
        {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for path in paths
    ]


def load_selector_freeze(path: Path | None, population_sha256: str) -> dict[str, Any]:
    if path is None:
        raise ValueError("test-stage construction requires a selector-freeze receipt")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != "FROZEN_BEFORE_TEST_OUTCOMES_OPENED":
        raise ValueError("selector-freeze receipt has the wrong status")
    if receipt.get("population_sha256") != population_sha256:
        raise ValueError("selector-freeze receipt references a different population")
    return receipt


def candidate_lookup_for_state(
    state: Mapping[str, Any], scan_assets: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    candidates = operational_candidates(
        scan_assets[str(state["asset_source_pair_key"])]["scan"]
    )
    return {str(row["selected_candidate_id"]): row for row in candidates}


def build_stage(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    population = json.loads(args.population.read_text(encoding="utf-8"))
    if population.get("status") != "PASS":
        raise ValueError("v2 population did not PASS")
    states = population["population"][args.role]["states"]
    state_keys = {str(row["pair_key"]) for row in states}
    input_paths = matched_paths(args.input_ledgers)
    input_bank = "O" if args.stage == "P" else "P"
    expected_repeats = 32
    audit_receipts = validate_input_audits(
        args.input_audits, expected_count=8 if args.stage == "P" else 1
    )
    if args.role == "test":
        selector_freeze = load_selector_freeze(
            args.selector_freeze_receipt, sha256_file(args.population)
        )
    else:
        selector_freeze = None
    summaries = summarize_ledgers(
        input_paths, expected_bank=input_bank, expected_repeats=expected_repeats
    )
    if set(summaries) != state_keys:
        raise ValueError("input outcome state set differs from the requested v2 role")
    expected_candidates = 97 if args.stage == "P" else 9
    if any(len(values) != expected_candidates for values in summaries.values()):
        raise ValueError(
            f"input candidate matrix is not complete at {expected_candidates} per state"
        )
    selection_count = 8 if args.stage == "P" else 1
    selected = select_noncanonical(summaries, selection_count)
    protocol_config = json.loads(args.protocol_config.read_text(encoding="utf-8"))
    legacy_root = Path(protocol_config["legacy_evidence"]["root"])
    scan_assets = load_scan_assets(legacy_root)
    method_selections = {}
    blocks = []
    for state in states:
        pair_key = str(state["pair_key"])
        lookup = candidate_lookup_for_state(state, scan_assets)
        if args.stage == "P":
            methods = {
                "canonical": "canonical",
                **{f"O_top{index + 1}": candidate for index, candidate in enumerate(selected[pair_key])},
            }
        elif args.role == "development":
            methods = {
                "canonical": "canonical",
                "statewise_P_top1": selected[pair_key][0],
            }
        else:
            methods = {
                "canonical": "canonical",
                "statewise_P_top1": selected[pair_key][0],
            }
            for method, method_payload in selector_freeze["methods"].items():
                candidate_id = str(method_payload["state_selections"][pair_key])
                methods[str(method)] = candidate_id
        unknown = set(methods.values()).difference(lookup)
        if unknown:
            raise ValueError(f"selector chose candidates outside the 97-view bank: {unknown}")
        method_selections[pair_key] = methods
        unique_candidates = []
        for candidate_id in dict.fromkeys(methods.values()):
            candidate = dict(lookup[candidate_id])
            candidate["candidate_features"] = {
                **candidate.get("candidate_features", {}),
                "selection_methods": [
                    method for method, value in methods.items() if value == candidate_id
                ],
            }
            unique_candidates.append(candidate)
        scan = scan_assets[str(state["asset_source_pair_key"])]["scan"]
        blocks.append(
            {
                "state": dict(state),
                "scene_construction": scan["scene_construction"],
                "candidates": unique_candidates,
            }
        )
    bank_id = args.stage
    phase = "independent_top8_refinement" if args.stage == "P" else "independent_confirmation"
    protocol = {
        "schema": "dsol_compact_view_matrix_protocol_v1",
        "status": "PASS_FROZEN",
        "phase": phase,
        "role": args.role,
        "diagnostic_role": f"statewise_view_oracle_v2_{args.stage}_{args.role}",
        "episode_identity_prefix": f"oracle-v2::{bank_id}::{args.role}",
        "noise_bank_id": bank_id,
        "policy_repeat_ids": list(range(32 if args.stage == "P" else 64)),
        "sensor_control": "both",
        "state_count": len(states),
        "candidate_count_per_state": sorted(
            {len(block["candidates"]) for block in blocks}
        ),
        "episode_count": sum(
            len(block["candidates"]) * (32 if args.stage == "P" else 64)
            for block in blocks
        ),
        "matrix_order": "state_then_candidate_then_repeat",
        "catalog": str(args.catalog.resolve()),
        "catalog_sha256": sha256_file(args.catalog),
        "population": str(args.population.resolve()),
        "population_sha256": sha256_file(args.population),
        "protocol_config": str(args.protocol_config.resolve()),
        "protocol_config_sha256": sha256_file(args.protocol_config),
        "input_bank": input_bank,
        "input_integrity_audits": audit_receipts,
        "rank_order": [
            "higher_success_probability",
            "higher_normalized_final_progress",
            "lower_harm_probability",
            "lower_completion_steps_conditional_on_success",
            "lexicographic_candidate_id",
        ],
        "selected_noncanonical": selected,
        "method_selections": method_selections,
        "selector_freeze_receipt": (
            None if args.selector_freeze_receipt is None else str(args.selector_freeze_receipt.resolve())
        ),
        "selector_freeze_receipt_sha256": (
            None if args.selector_freeze_receipt is None else sha256_file(args.selector_freeze_receipt)
        ),
        "test_outcomes_used_for_selector_training_or_hyperparameters": False,
        "state_blocks": blocks,
    }
    freeze_receipt = {
        "schema": "dsol_statewise_view_oracle_v2_stage_freeze",
        "status": "PASS_FROZEN_BEFORE_OUTPUT_BANK_OPEN",
        "stage": args.stage,
        "role": args.role,
        "input_bank": input_bank,
        "output_bank": bank_id,
        "input_ledger_files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in input_paths
        ],
        "input_integrity_audits": audit_receipts,
        "population_sha256": sha256_file(args.population),
        "protocol_sha256": None,
        "selected_noncanonical": selected,
        "method_selections": method_selections,
        "test_selector_freeze_verified": args.role != "test" or selector_freeze is not None,
    }
    return protocol, freeze_receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("P", "Q"), required=True)
    parser.add_argument("--role", choices=("development", "test"), required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--input-ledgers", nargs="+", required=True)
    parser.add_argument("--input-audits", nargs="+", required=True)
    parser.add_argument("--selector-freeze-receipt", type=Path)
    parser.add_argument("--output-protocol", type=Path, required=True)
    parser.add_argument("--output-freeze-receipt", type=Path, required=True)
    args = parser.parse_args()
    for field in ("population", "catalog", "protocol_config"):
        setattr(args, field, getattr(args, field).resolve())
    if args.selector_freeze_receipt is not None:
        args.selector_freeze_receipt = args.selector_freeze_receipt.resolve()
    protocol, receipt = build_stage(args)
    freeze_json(args.output_protocol.resolve(), protocol)
    receipt["protocol_sha256"] = sha256_file(args.output_protocol.resolve())
    freeze_json(args.output_freeze_receipt.resolve(), receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "stage": args.stage,
                "role": args.role,
                "episodes": protocol["episode_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
