from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PRIMARY_METRICS = (
    "success",
    "source_selection_success",
    "target_placement_success",
    "wrong_source_grasp",
    "lift_success",
    "transport_success",
    "progress",
)


def row_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["edge_id"]),
        int(row["canonical_state_index"]),
        int(row["execution_horizon"]),
    )


def paired_state_bootstrap(
    baseline: Mapping[int, float],
    method: Mapping[int, float],
    *,
    samples: int = 10_000,
    seed: int = 20260722,
) -> dict[str, float]:
    states = sorted(set(baseline) & set(method))
    if not states or set(baseline) != set(method):
        raise ValueError("paired bootstrap requires identical non-empty state groups")
    differences = np.asarray([method[state] - baseline[state] for state in states])
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(samples, len(states)), replace=True).mean(axis=1)
    return {
        "difference": float(differences.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "state_count": len(states),
    }


def _state_means(rows: Iterable[Mapping[str, Any]], metric: str) -> dict[int, float]:
    values: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        values[int(row["canonical_state_index"])].append(float(row[metric]))
    return {state: float(np.mean(items)) for state, items in values.items()}


def compare_payloads(
    baseline_payload: Mapping[str, Any],
    method_payload: Mapping[str, Any],
    *,
    bootstrap_samples: int = 10_000,
    decision_horizon: int = 3,
) -> dict[str, Any]:
    if baseline_payload.get("status") != "complete" or method_payload.get("status") != "complete":
        raise ValueError("policy comparison requires complete evaluations")
    baseline = {row_key(row): row for row in baseline_payload["rows"]}
    method = {row_key(row): row for row in method_payload["rows"]}
    if set(baseline) != set(method):
        raise ValueError("baseline and method evaluation keys differ")

    results: dict[str, Any] = {}
    horizons = sorted({key[2] for key in baseline})
    for horizon in horizons:
        horizon_result = {}
        for slice_name, predicate in (
            ("all", lambda row: True),
            ("id", lambda row: bool(row["action_supervised"])),
            ("ood", lambda row: not bool(row["action_supervised"])),
        ):
            keys = [
                key
                for key, row in baseline.items()
                if key[2] == horizon and predicate(row)
            ]
            slice_result = {}
            for metric in PRIMARY_METRICS:
                baseline_rows = [baseline[key] for key in keys]
                method_rows = [method[key] for key in keys]
                paired = paired_state_bootstrap(
                    _state_means(baseline_rows, metric),
                    _state_means(method_rows, metric),
                    samples=bootstrap_samples,
                    seed=20260722 + horizon,
                )
                slice_result[metric] = {
                    "baseline": float(np.mean([float(row[metric]) for row in baseline_rows])),
                    "method": float(np.mean([float(row[metric]) for row in method_rows])),
                    **paired,
                }
            horizon_result[slice_name] = slice_result

        by_edge = {}
        for edge_id in sorted({key[0] for key in baseline if key[2] == horizon}):
            keys = [key for key in baseline if key[0] == edge_id and key[2] == horizon]
            by_edge[edge_id] = {
                "baseline_success": float(np.mean([baseline[key]["success"] for key in keys])),
                "method_success": float(np.mean([method[key]["success"] for key in keys])),
            }
            by_edge[edge_id]["difference"] = (
                by_edge[edge_id]["method_success"] - by_edge[edge_id]["baseline_success"]
            )
        horizon_result["by_edge"] = by_edge
        results[f"k{horizon}"] = horizon_result

    decision_key = f"k{decision_horizon}"
    if decision_key not in results:
        raise ValueError(f"decision horizon K={decision_horizon} is absent")
    decision_slice = results[decision_key]
    withheld_differences = [
        row["difference"]
        for edge, row in decision_slice["by_edge"].items()
        if not bool(next(value for key, value in baseline.items() if key[0] == edge)["action_supervised"])
    ]
    baseline_valid = decision_slice["id"]["success"]["baseline"] >= 0.70
    advance = (
        baseline_valid
        and decision_slice["ood"]["success"]["difference"] >= 0.10
        and decision_slice["id"]["success"]["difference"] >= -0.05
        and all(value > 0 for value in withheld_differences)
    )
    decision = (
        "BASELINE_INVALID"
        if not baseline_valid
        else "ADVANCE_TO_FULL_CONTROLS"
        if advance
        else "PILOT_DOES_NOT_CLEAR_MIGRATION_GATE"
    )
    return {
        "schema_version": 1,
        "baseline_policy_identity": baseline_payload.get("policy_identity"),
        "method_policy_identity": method_payload.get("policy_identity"),
        "paired_unit": "canonical_state_index",
        "bootstrap_samples": bootstrap_samples,
        "results": results,
        "decision_horizon": decision_horizon,
        "pilot_decision": decision,
        "note": "K=3 is fixed before the full comparison; K=1/2 are robustness slices. This pilot cannot replace equal-data, fixed-slot, factor-null-dropout, or ablation controls.",
    }


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired LIBERO-Bind policy comparison")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--decision-horizon", type=int, choices=(1, 2, 3), default=3)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite comparison: {args.output}")
    result = compare_payloads(
        json.loads(args.baseline.read_text()),
        json.loads(args.method.read_text()),
        bootstrap_samples=args.bootstrap_samples,
        decision_horizon=args.decision_horizon,
    )
    _atomic_write(args.output, result)
    print(json.dumps({"output": str(args.output), "pilot_decision": result["pilot_decision"]}))


if __name__ == "__main__":
    main()
