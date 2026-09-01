#!/usr/bin/env python3
"""Analyze held-out selector utility and decide whether precision reserve is needed."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from scripts.dsol_paper1.build_view_value_expectation_calibration_stage import (
        load_results,
        validate_explicit_pairing,
    )
except ModuleNotFoundError:
    from build_view_value_expectation_calibration_stage import (
        load_results,
        validate_explicit_pairing,
    )


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        return [0.0, 1.0]
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [float(center - half), float(center + half)]


def task_stratified_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    *,
    seed: int,
    draws: int,
) -> list[float]:
    by_task_group = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_task_group[row["task_id"]][row["source_group"]].append(float(row[value_key]))
    generator = np.random.default_rng(seed)
    task_draws = []
    for task in sorted(by_task_group):
        values = np.asarray(
            [np.mean(by_task_group[task][group]) for group in sorted(by_task_group[task])],
            dtype=np.float64,
        )
        indices = generator.integers(0, len(values), size=(draws, len(values)))
        task_draws.append(values[indices].mean(axis=1))
    population = np.stack(task_draws).mean(axis=0)
    return np.quantile(population, [0.025, 0.975]).tolist()


def task_source_equal_mean(rows: Sequence[Mapping[str, Any]], value_key: str) -> float:
    by_task_group = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_task_group[row["task_id"]][row["source_group"]].append(float(row[value_key]))
    task_values = []
    for task in sorted(by_task_group):
        source_values = [float(np.mean(by_task_group[task][group])) for group in sorted(by_task_group[task])]
        task_values.append(float(np.mean(source_values)))
    return float(np.mean(task_values))


def load_seed_rows(patterns: Sequence[str], protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = load_results(patterns)
    if len(rows) != int(protocol["episode_count"]):
        raise ValueError(f"held-out matrix incomplete: {len(rows)}/{protocol['episode_count']}")
    validate_explicit_pairing(rows, str(protocol["bank_id"]))
    return rows


def state_method_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint_seed: int,
) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["pair_key"], row["selector_method"]].append(row)
    result = []
    states = sorted({row["pair_key"] for row in rows})
    for state in states:
        methods = {method for pair, method in grouped if pair == state}
        canonical = {
            (str(row["noise_bank_id"]), int(row["policy_repeat_id"])): row for row in grouped[state, "canonical"]
        }
        for method in sorted(methods):
            values = grouped[state, method]
            by_repeat = {(str(row["noise_bank_id"]), int(row["policy_repeat_id"])): row for row in values}
            repeats = sorted(by_repeat)
            if repeats != sorted(canonical):
                raise ValueError("selector and canonical repeat sets differ")
            paired = [(canonical[repeat], by_repeat[repeat]) for repeat in repeats]
            success = float(np.mean([selected["success"] for _, selected in paired]))
            canonical_success = float(np.mean([base["success"] for base, _ in paired]))
            harms = [bool(base["success"] and not selected["success"]) for base, selected in paired]
            rescues = [bool(not base["success"] and selected["success"]) for base, selected in paired]
            both_success = [bool(base["success"] and selected["success"]) for base, selected in paired]
            both_failure = [bool(not base["success"] and not selected["success"]) for base, selected in paired]
            first = values[0]
            result.append(
                {
                    "checkpoint_seed": checkpoint_seed,
                    "pair_key": state,
                    "source_group": first["source_group"],
                    "task_id": first["task_id"],
                    "selector_method": method,
                    "selected_candidate_id": first["selected_candidate_id"],
                    "repeat_count": len(repeats),
                    "success": success,
                    "canonical_success": canonical_success,
                    "success_gain": success - canonical_success,
                    "progress": float(np.mean([selected["normalized_final_progress"] for _, selected in paired])),
                    "canonical_progress": float(np.mean([base["normalized_final_progress"] for base, _ in paired])),
                    "harm_probability": float(np.mean(harms)),
                    "rescue_probability": float(np.mean(rescues)),
                    "harm_count": sum(harms),
                    "rescue_count": sum(rescues),
                    "both_success_count": sum(both_success),
                    "both_failure_count": sum(both_failure),
                }
            )
    return result


def summarize_seed(
    state_rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint_seed: int,
    draws: int,
) -> dict[str, Any]:
    methods = sorted({row["selector_method"] for row in state_rows})
    result = {}
    for method in methods:
        rows = [row for row in state_rows if row["selector_method"] == method]
        success_ci = task_stratified_bootstrap(rows, "success", seed=checkpoint_seed * 100 + 1, draws=draws)
        gain_ci = task_stratified_bootstrap(rows, "success_gain", seed=checkpoint_seed * 100 + 2, draws=draws)
        harm_successes = sum(int(row["harm_count"]) for row in rows)
        harm_trials = sum(int(row["repeat_count"]) for row in rows)
        harm_wilson = wilson(harm_successes, harm_trials)
        success_count = sum(round(float(row["success"]) * int(row["repeat_count"])) for row in rows)
        task_summary = {}
        for task in sorted({row["task_id"] for row in rows}):
            task_rows = [row for row in rows if row["task_id"] == task]
            task_summary[task] = {
                "source_group_count": len({row["source_group"] for row in task_rows}),
                "success_rate": task_source_equal_mean(task_rows, "success"),
                "canonical_success_rate": task_source_equal_mean(task_rows, "canonical_success"),
                "success_gain_pp": task_source_equal_mean(task_rows, "success_gain") * 100,
                "harm_probability": task_source_equal_mean(task_rows, "harm_probability"),
                "rescue_probability": task_source_equal_mean(task_rows, "rescue_probability"),
            }
        result[method] = {
            "state_count": len(rows),
            "source_group_count": len({row["source_group"] for row in rows}),
            "task_count": len({row["task_id"] for row in rows}),
            "success_rate": task_source_equal_mean(rows, "success"),
            "success_wilson_95_episode_level_descriptive": wilson(success_count, harm_trials),
            "success_task_stratified_bootstrap_95": success_ci,
            "success_gain_pp": task_source_equal_mean(rows, "success_gain") * 100,
            "success_gain_task_stratified_bootstrap_95_pp": [value * 100 for value in gain_ci],
            "harm_probability": task_source_equal_mean(rows, "harm_probability"),
            "harm_wilson_95": harm_wilson,
            "rescue_probability": task_source_equal_mean(rows, "rescue_probability"),
            "paired_episode_counts": {
                "rescue": sum(int(row["rescue_count"]) for row in rows),
                "harm": harm_successes,
                "both_success": sum(int(row["both_success_count"]) for row in rows),
                "both_failure": sum(int(row["both_failure_count"]) for row in rows),
            },
            "task_summary": task_summary,
        }
    return result


def cross_checkpoint_gate(
    summaries: Mapping[str, Mapping[str, Any]],
    best_rules: Mapping[str, str],
    convergence_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    final_precision: bool,
) -> dict[str, Any]:
    seed_rows = []
    for seed in sorted(summaries, key=int):
        method = best_rules[seed]
        values = summaries[seed][method]
        seed_rows.append(
            {
                "checkpoint_seed": int(seed),
                "selector_method": method,
                "success_rate": values["success_rate"],
                "success_gain_pp": values["success_gain_pp"],
                "success_gain_ci_95_pp": values["success_gain_task_stratified_bootstrap_95_pp"],
                "harm_probability": values["harm_probability"],
            }
        )
    directions = [row["success_gain_pp"] > 0 for row in seed_rows]
    gain = float(np.mean([row["success_gain_pp"] for row in seed_rows]))
    harm = float(np.mean([row["harm_probability"] for row in seed_rows]))
    all_cis_positive = all(row["success_gain_ci_95_pp"][0] > 0 for row in seed_rows)
    gain_ci_precision = all(
        (row["success_gain_ci_95_pp"][1] - row["success_gain_ci_95_pp"][0]) / 2 <= 5.0 for row in seed_rows
    )
    harm_precision = all(
        (summaries[seed][best_rules[seed]]["harm_wilson_95"][1] - summaries[seed][best_rules[seed]]["harm_wilson_95"][0])
        * 50
        <= 5.0
        for seed in summaries
    )
    convergence_precision = True
    if final_precision:
        for seed, rows in convergence_rows.items():
            curve = convergence(rows)
            method = best_rules[seed]
            if "32" not in curve or "64" not in curve:
                convergence_precision = False
                continue
            if abs(curve["64"][method] - curve["32"][method]) * 100 > 5.0:
                convergence_precision = False
    precision_sufficient = gain_ci_precision and harm_precision and convergence_precision
    passed = gain >= 5.0 and all(directions) and all_cis_positive and harm <= 0.05
    status = "SELECTOR_GAIN_CONFIRMED" if passed else "SELECTOR_GAIN_NOT_CONFIRMED"
    if final_precision and not precision_sufficient:
        status = "INCONCLUSIVE_NO_DIRECTIONAL_CLAIM"
    return {
        "status": status,
        "best_rule_frozen_on_calibration": next(iter(best_rules.values())),
        "cross_checkpoint_mean_gain_pp": gain,
        "cross_checkpoint_mean_harm_probability": harm,
        "direction_consistent_positive": all(directions),
        "each_checkpoint_ci_excludes_zero": all_cis_positive,
        "final_precision_halfwidth_at_most_5pp": precision_sufficient,
        "gain_ci_precision_sufficient": gain_ci_precision,
        "harm_ci_precision_sufficient": harm_precision,
        "noise_convergence_precision_sufficient": convergence_precision,
        "seed_results": seed_rows,
        "thresholds": {
            "gain_at_least_pp": 5.0,
            "harm_probability_at_most": 0.05,
            "positive_direction_all_checkpoint_seeds": True,
            "paired_ci_excludes_zero_each_checkpoint_seed": True,
        },
    }


def convergence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    # Recompute directly from episode outcomes so prefixes use the same repeat ordering.
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["pair_key"], row["selector_method"]].append(row)
    output = {}
    for prefix in (4, 8, 16, 32, 64):
        if any(len(values) < prefix for values in grouped.values()):
            continue
        method_values = defaultdict(list)
        for (_state, method), values in grouped.items():
            values = sorted(values, key=lambda row: (row["noise_bank_id"], int(row["policy_repeat_id"])))[:prefix]
            method_values[method].append(float(np.mean([row["success"] for row in values])))
        output[str(prefix)] = {method: float(np.mean(values)) for method, values in method_values.items()}
    return output


def reserve_decision(
    seed_summaries: Mapping[str, Mapping[str, Any]],
    convergence_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    reasons = []
    for seed, methods in seed_summaries.items():
        for method, values in methods.items():
            if method == "canonical":
                continue
            lower, upper = values["success_gain_task_stratified_bootstrap_95_pp"]
            if (upper - lower) / 2 > 5.0:
                reasons.append(
                    {
                        "seed": seed,
                        "method": method,
                        "criterion": "task_stratified_source_cluster_bootstrap_95_ci_halfwidth_exceeds_5pp",
                        "value_pp": (upper - lower) / 2,
                    }
                )
            harm_lower, harm_upper = values["harm_wilson_95"]
            if (harm_upper - harm_lower) * 50 > 5.0:
                reasons.append(
                    {
                        "seed": seed,
                        "method": method,
                        "criterion": "harm_rate_wilson_95_ci_halfwidth_exceeds_5pp",
                        "value_pp": (harm_upper - harm_lower) * 50,
                    }
                )
        curves = convergence(convergence_rows[seed])
        if "16" in curves and "32" in curves:
            for method in curves["32"]:
                shift = abs(curves["32"][method] - curves["16"][method]) * 100
                if shift > 5.0:
                    reasons.append(
                        {
                            "seed": seed,
                            "method": method,
                            "criterion": "absolute_population_estimate_shift_between_first16_and_first32_exceeds_5pp",
                            "value_pp": shift,
                        }
                    )
    return {
        "schema": "dsol_view_value_expectation_reserve_decision_v1",
        "status": "ACTIVATE_BANK_F" if reasons else "PRIMARY_PRECISION_SUFFICIENT",
        "activate_bank_F": bool(reasons),
        "machine_generated_before_bank_F_opened": True,
        "reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for seed in (41, 42, 43):
        parser.add_argument(f"--seed{seed}-protocol", type=Path, required=True)
        parser.add_argument(f"--seed{seed}-results", nargs="+", required=True)
        parser.add_argument(f"--seed{seed}-reserve-protocol", type=Path)
        parser.add_argument(f"--seed{seed}-reserve-results", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    args = parser.parse_args()
    seed_rows = {}
    state_rows = {}
    summaries = {}
    reserve_loaded = {}
    best_rules = {}
    for seed in (41, 42, 43):
        protocol = json.loads(getattr(args, f"seed{seed}_protocol").read_text())
        best_rules[str(seed)] = str(protocol["best_noncanonical_rule"])
        primary_rows = load_seed_rows(getattr(args, f"seed{seed}_results"), protocol)
        reserve_protocol_path = getattr(args, f"seed{seed}_reserve_protocol")
        reserve_patterns = getattr(args, f"seed{seed}_reserve_results")
        if (reserve_protocol_path is None) != (reserve_patterns is None):
            raise ValueError(f"seed {seed} reserve protocol and results must be provided together")
        rows = list(primary_rows)
        reserve_loaded[str(seed)] = False
        if reserve_protocol_path is not None:
            reserve_protocol = json.loads(reserve_protocol_path.read_text())
            if reserve_protocol.get("bank_id") != "F":
                raise ValueError(f"seed {seed} reserve protocol must use bank F")
            rows.extend(load_seed_rows(reserve_patterns, reserve_protocol))
            reserve_loaded[str(seed)] = True
        seed_rows[str(seed)] = rows
        state_rows[str(seed)] = state_method_rows(rows, checkpoint_seed=seed)
        summaries[str(seed)] = summarize_seed(
            state_rows[str(seed)], checkpoint_seed=seed, draws=args.bootstrap_resamples
        )
    if any(reserve_loaded.values()) and not all(reserve_loaded.values()):
        raise ValueError("reserve results must be complete for all checkpoint seeds")
    if len(set(best_rules.values())) != 1:
        raise ValueError("held-out protocols do not share one frozen best rule")
    final_precision = all(reserve_loaded.values())
    decision = (
        {
            "schema": "dsol_view_value_expectation_reserve_decision_v1",
            "status": "BANK_F_COMPLETE",
            "activate_bank_F": True,
            "machine_generated_before_bank_F_opened": False,
            "reasons": [],
        }
        if final_precision
        else reserve_decision(summaries, seed_rows)
    )
    selector_gate = cross_checkpoint_gate(
        summaries,
        best_rules,
        seed_rows,
        final_precision=final_precision,
    )
    result = {
        "schema": "dsol_view_value_expectation_heldout_analysis_v1",
        "status": (
            "FINAL_64_REPEAT_COMPLETE"
            if final_precision
            else "PRIMARY_COMPLETE_RESERVE_REQUIRED"
            if decision["activate_bank_F"]
            else "PRIMARY_COMPLETE_PRECISION_SUFFICIENT"
        ),
        "independent_unit": "source_demonstration_group",
        "policy_noise_is_nested_not_independent_population_sample": True,
        "noise_repeats_per_condition": 64 if final_precision else 32,
        "reserve_bank_loaded": reserve_loaded,
        "checkpoint_seeds": summaries,
        "selector_population_gate": selector_gate,
        "noise_convergence": {seed: convergence(rows) for seed, rows in seed_rows.items()},
        "reserve_decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "primary-analysis.json", result)
    atomic_json(args.output_dir / "reserve-decision.json", decision)
    flat = [row for values in state_rows.values() for row in values]
    with (args.output_dir / "state-method-results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({"status": result["status"], "activate_bank_F": decision["activate_bank_F"]}, sort_keys=True))


if __name__ == "__main__":
    main()
