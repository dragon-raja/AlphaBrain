#!/usr/bin/env python3
"""Analyze frozen Q candidates without treating discovery maxima as effects."""

from __future__ import annotations

# Repository-local CLI bootstrap: path setup only.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[3]
for _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):
    if str(_layout_path) not in _layout_sys.path:
        _layout_sys.path.insert(0, str(_layout_path))


import argparse
import csv
import hashlib
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from scripts.dsol_paper1.protocols.build_statewise_view_oracle_v2_stage import iter_rows, matched_paths
    from scripts.dsol_paper1.explicit_flow_noise import sha256_file
except ImportError:  # Direct script execution.
    from scripts.dsol_paper1.protocols.build_statewise_view_oracle_v2_stage import iter_rows, matched_paths
    from scripts.dsol_paper1.explicit_flow_noise import sha256_file


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def stable_seed(label: str, root_seed: int = 20260903) -> int:
    return int.from_bytes(hashlib.sha256(f"{root_seed}::{label}".encode()).digest()[:8], "little")


def load_outcomes(
    patterns: Sequence[str], expected_states: set[str], expected_banks: set[str]
) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    rows = {}
    banks = set()
    for row in iter_rows(matched_paths(patterns)):
        if row.get("status") != "complete" or not row.get("explicit_flow_noise"):
            raise ValueError("Q analysis input includes incomplete/non-explicit-noise row")
        state = str(row["pair_key"])
        if state not in expected_states:
            raise ValueError(f"Q outcome outside test population: {state}")
        bank = str(row["noise_bank_id"])
        banks.add(bank)
        key = (
            state,
            str(row["selected_candidate_id"]),
            bank,
            int(row["policy_repeat_id"]),
        )
        if key in rows:
            raise ValueError(f"duplicate Q outcome: {key}")
        rows[key] = row
    if banks != expected_banks:
        raise ValueError(f"analysis expected banks {expected_banks}, found {banks}")
    return rows


def stratified_bootstrap(
    values: Mapping[str, float], tasks: Mapping[str, str], *, label: str, resamples: int = 20000
) -> dict[str, Any]:
    by_task = defaultdict(list)
    for state, value in values.items():
        by_task[tasks[state]].append(float(value))
    if len(by_task) != 8 or any(len(group) != 2 for group in by_task.values()):
        raise ValueError("test bootstrap requires two source states for each of eight tasks")
    rng = np.random.default_rng(stable_seed(label))
    draws = np.empty(resamples, dtype=np.float64)
    ordered = [np.asarray(by_task[task], dtype=np.float64) for task in sorted(by_task)]
    for index in range(resamples):
        sampled = [rng.choice(group, size=len(group), replace=True) for group in ordered]
        draws[index] = float(np.mean(np.concatenate(sampled)))
    observed = float(np.mean(list(values.values())))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": observed,
        "ci95": [float(lower), float(upper)],
        "ci_halfwidth": float((upper - lower) / 2),
        "bootstrap_resamples": resamples,
        "P_bootstrap_gt_0": float(np.mean(draws > 0)),
    }


def analyze_checkpoint(
    *,
    checkpoint_seed: int,
    rows: Mapping[tuple[str, str, str, int], Mapping[str, Any]],
    protocol: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
    sample_keys: Sequence[tuple[str, int]],
) -> dict[str, Any]:
    task_by_state = {str(state["pair_key"]): str(state["task_id"]) for state in states}
    methods_by_state = protocol["method_selections"]
    method_names = sorted(
        {method for values in methods_by_state.values() for method in values}
    )
    first32_keys = [("Q", repeat) for repeat in range(32)]
    state_metrics = []
    method_state_values: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for state in states:
        state_key = str(state["pair_key"])
        canonical_id = str(methods_by_state[state_key]["canonical"])
        canonical_rows = [
            rows[state_key, canonical_id, bank, repeat] for bank, repeat in sample_keys
        ]
        canonical_success = np.asarray([value["success"] for value in canonical_rows], dtype=bool)
        for method in method_names:
            candidate_id = str(methods_by_state[state_key][method])
            values = [
                rows[state_key, candidate_id, bank, repeat] for bank, repeat in sample_keys
            ]
            success = np.asarray([value["success"] for value in values], dtype=bool)
            progress = np.asarray(
                [value["normalized_final_progress"] for value in values], dtype=np.float64
            )
            record = {
                "checkpoint_seed": checkpoint_seed,
                "pair_key": state_key,
                "source_group": state["source_group"],
                "task_id": state["task_id"],
                "method": method,
                "candidate_id": candidate_id,
                "success": float(np.mean(success)),
                "progress": float(np.mean(progress)),
                "gain_vs_canonical": float(np.mean(success) - np.mean(canonical_success)),
                "harm_vs_canonical": float(np.mean(canonical_success & ~success)),
                "rescue_vs_canonical": float(np.mean(~canonical_success & success)),
                "first32_success": float(
                    np.mean(
                        [
                            rows[state_key, candidate_id, bank, repeat]["success"]
                            for bank, repeat in first32_keys
                        ]
                    )
                ),
            }
            state_metrics.append(record)
            method_state_values[method][state_key] = record
    population = {}
    for method in method_names:
        records = method_state_values[method]
        population[method] = {
            "success": float(np.mean([row["success"] for row in records.values()])),
            "progress": float(np.mean([row["progress"] for row in records.values()])),
            "gain_vs_canonical": stratified_bootstrap(
                {state: row["gain_vs_canonical"] for state, row in records.items()},
                task_by_state,
                label=f"seed{checkpoint_seed}::{method}::canonical",
            ),
            "harm_vs_canonical": float(
                np.mean([row["harm_vs_canonical"] for row in records.values()])
            ),
            "rescue_vs_canonical": float(
                np.mean([row["rescue_vs_canonical"] for row in records.values()])
            ),
            "first32_to_full_success_shift": float(
                np.mean([row["first32_success"] - row["success"] for row in records.values()])
            ),
        }
    comparisons = {}
    for left, right in (
        ("statewise_P_top1", "canonical"),
        ("statewise_P_top1", "global_fixed"),
        ("geometry_context_ridge", "canonical"),
        ("image_queryable_ridge", "canonical"),
    ):
        if left not in method_state_values or right not in method_state_values:
            continue
        differences = {
            state: method_state_values[left][state]["success"]
            - method_state_values[right][state]["success"]
            for state in task_by_state
        }
        comparisons[f"{left}_minus_{right}"] = stratified_bootstrap(
            differences,
            task_by_state,
            label=f"seed{checkpoint_seed}::{left}::{right}",
        )
    return {
        "checkpoint_seed": checkpoint_seed,
        "noise_repeats": len(sample_keys),
        "noise_banks": sorted({bank for bank, _repeat in sample_keys}),
        "state_count": len(states),
        "population": population,
        "comparisons": comparisons,
        "state_metrics": state_metrics,
    }


def decision(checkpoint: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    comparisons = checkpoint["comparisons"]
    population = checkpoint["population"]
    oracle = comparisons["statewise_P_top1_minus_canonical"]
    dependence = comparisons["statewise_P_top1_minus_global_fixed"]
    oracle_statistical = oracle["ci95"][0] > 0
    oracle_practical = oracle["estimate"] >= 0.05
    dependence_statistical = dependence["ci95"][0] > 0
    dependence_practical = dependence["estimate"] >= 0.03
    learner_gates = {}
    for method in ("geometry_context_ridge", "image_queryable_ridge"):
        comparison = comparisons[f"{method}_minus_canonical"]
        oracle_gain = oracle["estimate"]
        capture = comparison["estimate"] / oracle_gain if oracle_gain > 0 else None
        learner_gates[method] = {
            "statistical": comparison["ci95"][0] > 0,
            "practical_gain_3pp": comparison["estimate"] >= 0.03,
            "oracle_headroom_capture": capture,
            "capture_at_least_half": capture is not None and capture >= 0.5,
            "harm_no_more_than_5pp": population[method]["harm_vs_canonical"] <= 0.05,
        }
        learner_gates[method]["pass"] = all(
            learner_gates[method][key]
            for key in (
                "statistical",
                "practical_gain_3pp",
                "capture_at_least_half",
                "harm_no_more_than_5pp",
            )
        )
    statewise_ids = [
        values["statewise_P_top1"] for values in protocol["method_selections"].values()
    ]
    return {
        "oracle_existence": {
            "statistical": oracle_statistical,
            "practical_gain_5pp": oracle_practical,
            "pass": oracle_statistical and oracle_practical,
        },
        "state_dependence": {
            "statistical": dependence_statistical,
            "practical_regret_3pp": dependence_practical,
            "pass": dependence_statistical and dependence_practical,
        },
        "learner_generalization": learner_gates,
        "statewise_candidate_unique_count": len(set(statewise_ids)),
        "statewise_candidate_frequencies": dict(Counter(statewise_ids)),
    }


def write_state_metrics(path: Path, checkpoints: Sequence[Mapping[str, Any]]) -> None:
    rows = [row for checkpoint in checkpoints for row in checkpoint["state_metrics"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--seed41-ledgers", nargs="+", required=True)
    parser.add_argument("--seed42-ledgers", nargs="+", required=True)
    parser.add_argument("--seed43-ledgers", nargs="+", required=True)
    parser.add_argument("--seed41-reserve-ledgers", nargs="+")
    parser.add_argument("--audits", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    population = json.loads(args.population.read_text(encoding="utf-8"))
    states = population["population"]["test"]["states"]
    state_keys = {str(state["pair_key"]) for state in states}
    audit_receipts = []
    for path in args.audits:
        audit = json.loads(path.read_text(encoding="utf-8"))
        if audit.get("status") != "PASS_COMPLETE":
            raise ValueError(f"Q integrity audit did not PASS_COMPLETE: {path}")
        audit_receipts.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    checkpoints = []
    for seed, patterns in ((41, args.seed41_ledgers), (42, args.seed42_ledgers), (43, args.seed43_ledgers)):
        expected_banks = {"Q"}
        sample_keys = [("Q", repeat) for repeat in range(64)]
        outcomes = load_outcomes(patterns, state_keys, expected_banks)
        if seed == 41 and args.seed41_reserve_ledgers:
            reserve = load_outcomes(args.seed41_reserve_ledgers, state_keys, {"R"})
            overlap = set(outcomes).intersection(reserve)
            if overlap:
                raise ValueError("Q and R outcome identities unexpectedly overlap")
            outcomes.update(reserve)
            expected_banks.add("R")
            sample_keys.extend(("R", repeat) for repeat in range(64))
        checkpoints.append(
            analyze_checkpoint(
                checkpoint_seed=seed,
                rows=outcomes,
                protocol=protocol,
                states=states,
                sample_keys=sample_keys,
            )
        )
    primary_decision = decision(checkpoints[0], protocol)
    primary_comparisons = checkpoints[0]["comparisons"]
    reserve_triggers = {
        "any_primary_ci_halfwidth_over_5pp": any(
            row["ci_halfwidth"] > 0.05 for row in primary_comparisons.values()
        ),
        "any_method_first32_to_full_shift_over_5pp": any(
            abs(row["first32_to_full_success_shift"]) > 0.05
            for row in checkpoints[0]["population"].values()
        ),
    }
    reserve_complete = bool(args.seed41_reserve_ledgers)
    reserve_decision = {
        "schema": "dsol_statewise_view_oracle_v2_reserve_decision",
        "status": (
            "R_COMPLETE"
            if reserve_complete
            else ("ACTIVATE_R" if any(reserve_triggers.values()) else "DO_NOT_ACTIVATE_R")
        ),
        "triggers": reserve_triggers,
        "candidate_set_may_change": False,
        "reserve_complete": reserve_complete,
    }
    result = {
        "schema": "dsol_statewise_view_oracle_v2_analysis",
        "status": "PASS_Q_ANALYZED",
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256_file(args.protocol),
        "population": str(args.population.resolve()),
        "population_sha256": sha256_file(args.population),
        "integrity_audits": audit_receipts,
        "primary_checkpoint_seed": 41,
        "primary_decision": primary_decision,
        "checkpoint_results": checkpoints,
        "cross_checkpoint_scope": "seed41-frozen candidates transferred to seeds42/43; not checkpoint-specific dense oracles",
        "reserve_decision": reserve_decision,
        "discovery_O_or_refinement_P_winner_rates_used_as_final_effect": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "analysis.json", result)
    atomic_json(args.output_dir / "reserve-decision.json", reserve_decision)
    write_state_metrics(args.output_dir / "state-method-metrics.csv", checkpoints)
    print(
        json.dumps(
            {
                "status": result["status"],
                "oracle_gate": primary_decision["oracle_existence"]["pass"],
                "state_dependence_gate": primary_decision["state_dependence"]["pass"],
                "reserve": reserve_decision["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
