from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_libero_closed_loop import _atomic_write_json
from evaluate_physical_process_oracle import aggregate_outcomes
from evaluate_recovery_expert_handoff import (
    EXPERT_SANITY_METHOD,
    HANDOFF_METHODS,
)
from evaluate_recovery_segment_oracle import (
    BINARY_METRICS,
    EXPECTED_CHECKPOINT_SHA256,
    METRICS,
)
from paired_evaluation import bootstrap_summary


EXPECTED_SEEDS = (41, 42, 43)
EXPECTED_GROUPS_PER_SEED = 13
EXPECTED_SOURCE_CLUSTERS = 9


def _summary_value(summary: Mapping[str, Any], metric: str) -> float:
    key = f"{metric}_rate" if metric in BINARY_METRICS else metric
    value = float(summary[key])
    if not np.isfinite(value):
        raise ValueError(f"non-finite summary metric {metric}")
    return value


def _validate_method(
    row: Mapping[str, Any],
    method: str,
    *,
    continuation_count: int,
) -> None:
    result = row["methods"][method]
    expected = 1 if method == EXPERT_SANITY_METHOD else continuation_count
    continuations = result.get("continuations")
    if not isinstance(continuations, list) or len(continuations) != expected:
        raise ValueError(
            f"row {row.get('pair_id')} method {method} requires {expected} continuations"
        )
    repeats = [continuation.get("repeat") for continuation in continuations]
    if repeats != list(range(expected)):
        raise ValueError(
            f"row {row.get('pair_id')} method {method} repeats must be ordered 0..{expected - 1}"
        )
    outcomes = []
    for continuation in continuations:
        outcome = continuation.get("outcome")
        if not isinstance(outcome, Mapping):
            raise ValueError(
                f"row {row.get('pair_id')} method {method} lacks a raw outcome"
            )
        outcomes.append(outcome)
    recomputed = aggregate_outcomes(outcomes)
    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError(
            f"row {row.get('pair_id')} method {method} lacks a summary"
        )
    for metric in METRICS:
        if not np.isclose(
            _summary_value(summary, metric),
            _summary_value(recomputed, metric),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"row {row.get('pair_id')} method {method} summary does not match raw {metric}"
            )
    if method == EXPERT_SANITY_METHOD and any(
        int(continuation["policy_calls"]) != 0 for continuation in continuations
    ):
        raise ValueError("teacher_full must never hand control to the policy")


def load_and_validate(
    paths: Sequence[Path],
    *,
    require_full_grid: bool,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    if not paths:
        raise ValueError("at least one handoff result is required")
    payloads = [json.loads(path.read_text()) for path in paths]
    identity_keys = (
        "schema_version",
        "run_kind",
        "episode_root",
        "split",
        "methods",
        "expert_sanity_method",
        "execution_horizon",
        "total_action_budget",
        "max_teacher_actions",
        "continuations",
        "stage_dwell_steps",
        "teacher_is_privileged_upper_bound",
        "teacher_privileged_inputs",
        "policy_receives_teacher_or_branch_labels",
        "continuation_seed_protocol",
        "git_sha",
    )
    identity = {key: payloads[0].get(key) for key in identity_keys}
    if identity["teacher_is_privileged_upper_bound"] is not True:
        raise ValueError("handoff results must label the teacher as a privileged upper bound")
    if identity["policy_receives_teacher_or_branch_labels"] is not False:
        raise ValueError("handoff policy must not receive teacher or branch labels")
    rows: list[Mapping[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for path, payload in zip(paths, payloads, strict=True):
        if payload.get("status") != "complete":
            raise ValueError(f"incomplete handoff result: {path}")
        payload_seed = int(payload.get("seed", -1))
        if (
            payload.get("run_kind") == "decision"
            and payload.get("git_dirty_at_launch") is not False
        ):
            raise ValueError(f"handoff result was dirty at launch: {path}")
        if not payload.get("git_sha"):
            raise ValueError(f"handoff result lacks Git identity: {path}")
        if payload.get("policy_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256.get(
            payload_seed
        ):
            raise ValueError(f"handoff checkpoint SHA256 mismatch: {path}")
        for key, expected in identity.items():
            if payload.get(key) != expected:
                raise ValueError(f"handoff result identity mismatch for {key}: {path}")
        if tuple(payload.get("methods", ())) != HANDOFF_METHODS:
            raise ValueError(f"unexpected handoff methods: {path}")
        if payload.get("expert_sanity_method") != EXPERT_SANITY_METHOD:
            raise ValueError(f"unexpected expert sanity method: {path}")
        payload_rows = payload.get("rows")
        if not isinstance(payload_rows, list) or len(payload_rows) != int(
            payload["expected_rows"]
        ):
            raise ValueError(f"handoff row count mismatch: {path}")
        if int(payload.get("completed_rows", -1)) != len(payload_rows):
            raise ValueError(f"handoff completion count mismatch: {path}")
        for row in payload_rows:
            if int(row["seed"]) != int(payload["seed"]):
                raise ValueError(f"row seed disagrees with payload: {path}")
            key = (int(row["seed"]), str(row["pair_id"]))
            if key in seen:
                raise ValueError(f"duplicate handoff row {key}")
            seen.add(key)
            order_audit = row.get("method_order_invariance", {})
            if any(float(value) != 0.0 for value in order_audit.values()):
                raise ValueError(f"method-order invariance failed for {key}")
            for method in (*HANDOFF_METHODS, EXPERT_SANITY_METHOD):
                _validate_method(
                    row,
                    method,
                    continuation_count=int(payload["continuations"]),
                )
            rows.append(row)

    if require_full_grid:
        if identity["run_kind"] != "decision":
            raise ValueError("full handoff grid must use decision runs")
        seeds = {int(row["seed"]) for row in rows}
        if seeds != set(EXPECTED_SEEDS):
            raise ValueError(f"full grid requires seeds {EXPECTED_SEEDS}, found {sorted(seeds)}")
        counts = {
            seed: sum(int(row["seed"]) == seed for row in rows)
            for seed in EXPECTED_SEEDS
        }
        if any(count != EXPECTED_GROUPS_PER_SEED for count in counts.values()):
            raise ValueError(f"full grid requires 13 groups per seed, found {counts}")
        source_count = len({int(row["source_initial_state_index"]) for row in rows})
        if source_count != EXPECTED_SOURCE_CLUSTERS:
            raise ValueError(
                f"full grid requires {EXPECTED_SOURCE_CLUSTERS} source clusters, found {source_count}"
            )
    return rows, identity


def _source_cluster_values(
    rows: Sequence[Mapping[str, Any]],
    value_fn,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    by_seed_source: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        source = str(int(row["source_initial_state_index"]))
        by_seed_source[(int(row["seed"]), source)].append(float(value_fn(row)))
    seed_source = {
        key: float(np.mean(values)) for key, values in sorted(by_seed_source.items())
    }
    by_source: dict[str, list[float]] = defaultdict(list)
    by_seed: dict[str, list[float]] = defaultdict(list)
    for (seed, source), value in seed_source.items():
        by_source[source].append(value)
        by_seed[str(seed)].append(value)
    source_values = {
        source: float(np.mean(values)) for source, values in sorted(by_source.items())
    }
    return source_values, by_seed


def hierarchical_summary(
    rows: Sequence[Mapping[str, Any]],
    value_fn,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    source_values, by_seed = _source_cluster_values(rows, value_fn)
    per_seed = {
        run_seed: bootstrap_summary(
            values,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 10_000 + int(run_seed),
        )
        for run_seed, values in sorted(by_seed.items())
    }
    return {
        "independent_unit": "source_initial_state_index",
        "aggregation_order": (
            "mean within (seed, source), then across seeds within source; "
            "bootstrap resamples source clusters"
        ),
        "source_cluster_level": bootstrap_summary(
            list(source_values.values()),
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "per_source_cluster": source_values,
        "per_seed_source_cluster_level": per_seed,
        "per_seed_mean": {
            run_seed: summary["mean"] for run_seed, summary in per_seed.items()
        },
    }


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "row_count": len(rows),
        "source_cluster_count": len(
            {int(row["source_initial_state_index"]) for row in rows}
        ),
        "absolute": {},
        "paired_vs_policy_only": {},
        "teacher_prefix": {},
    }
    for method_index, method in enumerate((*HANDOFF_METHODS, EXPERT_SANITY_METHOD)):
        result["absolute"][method] = {}
        for metric_index, metric in enumerate(METRICS):
            metric_summary = hierarchical_summary(
                rows,
                lambda row, m=method, key=metric: _summary_value(
                    row["methods"][m]["summary"], key
                ),
                bootstrap_samples=bootstrap_samples,
                seed=seed + method_index * 100 + metric_index,
            )
            if metric in BINARY_METRICS:
                metric_summary["percentage_points"] = {
                    "mean": 100.0 * metric_summary["source_cluster_level"]["mean"],
                    "bootstrap_95_low": 100.0
                    * metric_summary["source_cluster_level"]["bootstrap_95_low"],
                    "bootstrap_95_high": 100.0
                    * metric_summary["source_cluster_level"]["bootstrap_95_high"],
                }
            result["absolute"][method][metric] = metric_summary
        result["teacher_prefix"][method] = {
            "teacher_actions": hierarchical_summary(
                rows,
                lambda row, m=method: float(row["methods"][m]["teacher_actions"]),
                bootstrap_samples=bootstrap_samples,
                seed=seed + 5_000 + method_index,
            ),
            "criterion_reached_rate": hierarchical_summary(
                rows,
                lambda row, m=method: float(row["methods"][m]["criterion_reached"]),
                bootstrap_samples=bootstrap_samples,
                seed=seed + 6_000 + method_index,
            ),
        }

    for method_index, method in enumerate(HANDOFF_METHODS[1:], start=1):
        result["paired_vs_policy_only"][method] = {}
        for metric_index, metric in enumerate(METRICS):
            paired = hierarchical_summary(
                rows,
                lambda row, m=method, key=metric: _summary_value(
                    row["methods"][m]["summary"], key
                )
                - _summary_value(row["methods"]["policy_only"]["summary"], key),
                bootstrap_samples=bootstrap_samples,
                seed=seed + 10_000 + method_index * 100 + metric_index,
            )
            if metric in BINARY_METRICS:
                paired["percentage_point_difference"] = {
                    "mean": 100.0 * paired["source_cluster_level"]["mean"],
                    "bootstrap_95_low": 100.0
                    * paired["source_cluster_level"]["bootstrap_95_low"],
                    "bootstrap_95_high": 100.0
                    * paired["source_cluster_level"]["bootstrap_95_high"],
                }
            result["paired_vs_policy_only"][method][metric] = paired

    expert_success = result["absolute"][EXPERT_SANITY_METHOD]["success"]
    result["diagnostic_gate"] = {
        "teacher_controller_valid": bool(
            expert_success["source_cluster_level"]["mean"] >= 0.9
            and all(value >= 0.8 for value in expert_success["per_seed_mean"].values())
        ),
        "teacher_controller_success_rate": expert_success[
            "source_cluster_level"
        ]["mean"],
        "handoff_success_delta": {
            method: result["paired_vs_policy_only"][method]["success"][
                "source_cluster_level"
            ]
            for method in HANDOFF_METHODS[1:]
        },
        "note": (
            "This is a causal sufficiency diagnostic, not a final method decision. "
            "Training remains blocked until the teacher is valid and at least one "
            "deployable handoff stage improves full-task success."
        ),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize expert handoff ladder")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_715)
    parser.add_argument("--require-full-grid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, identity = load_and_validate(
        args.inputs,
        require_full_grid=args.require_full_grid,
    )
    payload = {
        "schema_version": 1,
        "identity": identity,
        "input_files": [str(path) for path in args.inputs],
        "full_grid_required": args.require_full_grid,
        "summary": build_summary(
            rows,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
    }
    _atomic_write_json(args.output, payload)
    print(json.dumps(payload["summary"]["diagnostic_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
