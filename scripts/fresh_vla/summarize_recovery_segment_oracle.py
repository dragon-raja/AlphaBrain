from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_libero_closed_loop import _atomic_write_json
from evaluate_recovery_segment_oracle import (
    BINARY_METRICS,
    METHODS,
    METRICS,
    METRIC_DIRECTIONS,
)
from paired_evaluation import bootstrap_summary


DECISION_RUN_KIND = "decision"
SMOKE_RUN_KIND = "smoke"
EXPECTED_SCHEMA_VERSION = 2
EXPECTED_DECISION_SEEDS = (41, 42, 43)
EXPECTED_DECISION_GROUP_COUNT = 13
EXPECTED_DECISION_SOURCE_CLUSTER_COUNT = 9
EXPECTED_RANDOM4_SCHEDULE_COUNT = 3
EXPECTED_PREREGISTRATION_SHA256 = "d3105ba595e3467f2d2cec5642ca052dea2a692a63ad98b026c06a436ecb167c"
EXPECTED_CHECKPOINT_SHA256 = {
    41: "144a3b3d3dcc8421418564a62059a1038c9a7ef3196ac157f5f9ea1997a31f30",
    42: "98dc52d2ed1983776d218fee7666f3131053d1a55296e93e9f521b1c088ce875",
    43: "5db16350d9835c1f28d01b660dd6e9234bcab3da79abbce1f092e92b08ac9149",
}
EXPECTED_DECISION_CONFIG = {
    "split": "val",
    "sample_count": 4,
    "segment_replans": 4,
    "execution_horizon": 3,
    "segment_action_budget": 12,
    "total_action_budget": 120,
    "lookahead_steps": 30,
    "selection_continuations": 3,
    "decision_heldout_continuations": 5,
    "full_heldout_continuations": 5,
    "stage_dwell_steps": 2,
    "random4_schedule_count": EXPECTED_RANDOM4_SCHEDULE_COUNT,
}

IDENTITY_KEYS = (
    "schema_version",
    "episode_root",
    "split",
    "sample_count",
    "segment_replans",
    "execution_horizon",
    "segment_action_budget",
    "total_action_budget",
    "lookahead_steps",
    "selection_continuations",
    "decision_heldout_continuations",
    "full_heldout_continuations",
    "stage_dwell_steps",
    "random4_schedule_count",
    "replay_sim_tolerance",
    "candidate_pool_tolerance",
    "methods",
    "outcome_metric_order",
    "outcome_metric_directions",
    "git_sha",
    "run_kind",
    "preregistration_sha256",
    "expected_preregistration_sha256",
    "expected_global_rows",
    "expected_global_pair_source_map",
)


def _run_kind(payload: Mapping[str, Any]) -> str:
    if "run_kind" not in payload:
        if payload.get("schema_version") == 1:
            return SMOKE_RUN_KIND
        raise ValueError("schema version 2 recovery-segment result requires run_kind")
    value = payload["run_kind"]
    if value not in {DECISION_RUN_KIND, SMOKE_RUN_KIND}:
        raise ValueError(f"unsupported recovery-segment run_kind: {value!r}")
    return str(value)


def _preregistration_sha256(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("preregistration_sha256", payload.get("prereg_sha256"))
    return str(value) if value is not None else None


def _random4_schedule_count(payload: Mapping[str, Any]) -> int | None:
    value = payload.get(
        "random4_schedule_count",
        payload.get("random_schedule_count", payload.get("random_schedules")),
    )
    return int(value) if value is not None else None


def _heldout_continuation_counts(payload: Mapping[str, Any]) -> tuple[int | None, int | None]:
    legacy = payload.get("heldout_continuations")
    decision = payload.get("decision_heldout_continuations", legacy)
    full = payload.get("full_heldout_continuations", legacy)
    return (
        int(decision) if decision is not None else None,
        int(full) if full is not None else None,
    )


def _expected_global_pair_source_map(payload: Mapping[str, Any]) -> dict[str, str] | None:
    value = payload.get("expected_global_pair_source_map")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("expected_global_pair_source_map must be a mapping")
    return {str(pair_id): _normalize_source(source) for pair_id, source in value.items()}


def _identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["run_kind"] = _run_kind(payload)
    normalized["preregistration_sha256"] = _preregistration_sha256(payload)
    normalized["random4_schedule_count"] = _random4_schedule_count(payload)
    decision_heldout, full_heldout = _heldout_continuation_counts(payload)
    normalized["decision_heldout_continuations"] = decision_heldout
    normalized["full_heldout_continuations"] = full_heldout
    normalized["expected_global_pair_source_map"] = _expected_global_pair_source_map(
        payload
    )
    return {key: normalized.get(key) for key in IDENTITY_KEYS}


def _normalize_source(value: Any) -> str:
    if value is None:
        raise ValueError("source_initial_state_index must not be null")
    return str(value)


def _manifest_grid(episode_root: str, split: str) -> dict[str, str]:
    manifest_path = Path(episode_root) / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"episode manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    result: dict[str, str] = {}
    for group in manifest.get("groups", []):
        if group.get("split") != split:
            continue
        pair_id = str(group["pair_id"])
        if pair_id in result:
            raise ValueError(f"duplicate pair_id in episode manifest: {pair_id}")
        result[pair_id] = _normalize_source(group.get("source_initial_state_index"))
    return result


def _decision_error(path: Path, message: str) -> ValueError:
    return ValueError(f"invalid decision run {path}: {message}")


def _validate_decision_payload(path: Path, payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise _decision_error(path, f"schema_version must be {EXPECTED_SCHEMA_VERSION}")
    if payload.get("status") != "complete":
        raise _decision_error(path, "status is not complete")
    if payload.get("git_dirty_at_launch") is not False:
        raise _decision_error(path, "git worktree clean status is missing or dirty at launch")
    if not payload.get("git_sha"):
        raise _decision_error(path, "git_sha is missing")
    required_schema_fields = (
        "preregistration_sha256",
        "decision_heldout_continuations",
        "full_heldout_continuations",
        "random_schedules",
        "expected_global_rows",
        "expected_global_pair_source_map",
    )
    missing_schema_fields = [key for key in required_schema_fields if key not in payload]
    if missing_schema_fields:
        raise _decision_error(
            path,
            f"schema version 2 fields are missing: {missing_schema_fields}",
        )
    if _preregistration_sha256(payload) != EXPECTED_PREREGISTRATION_SHA256:
        raise _decision_error(path, "preregistration SHA256 mismatch")
    if payload.get("expected_preregistration_sha256") != EXPECTED_PREREGISTRATION_SHA256:
        raise _decision_error(path, "evaluator expected preregistration SHA256 mismatch")
    seed = int(payload.get("seed", -1))
    if seed not in EXPECTED_DECISION_SEEDS:
        raise _decision_error(path, f"unsupported model seed {seed}")
    expected_checkpoint = EXPECTED_CHECKPOINT_SHA256[seed]
    if payload.get("policy_checkpoint_sha256") != expected_checkpoint:
        raise _decision_error(path, f"checkpoint SHA256 mismatch for seed {seed}")
    if payload.get("methods") != list(METHODS):
        raise _decision_error(path, "method order does not match the frozen evaluator schema")
    if payload.get("outcome_metric_order") != list(METRICS):
        raise _decision_error(path, "outcome metric order does not match the frozen evaluator schema")
    if payload.get("outcome_metric_directions") != list(METRIC_DIRECTIONS):
        raise _decision_error(
            path,
            "outcome metric directions do not match the frozen evaluator schema",
        )
    normalized = _identity(payload)
    for key, expected in EXPECTED_DECISION_CONFIG.items():
        if normalized.get(key) != expected:
            raise _decision_error(
                path,
                f"pre-registered config mismatch for {key}: {normalized.get(key)!r} != {expected!r}",
            )
    expected_pair_source_map = _expected_global_pair_source_map(payload)
    if expected_pair_source_map is None:
        raise _decision_error(path, "expected_global_pair_source_map is missing")
    if len(expected_pair_source_map) != EXPECTED_DECISION_GROUP_COUNT:
        raise _decision_error(
            path,
            "expected_global_pair_source_map must contain "
            f"{EXPECTED_DECISION_GROUP_COUNT} groups",
        )
    if len(set(expected_pair_source_map.values())) != EXPECTED_DECISION_SOURCE_CLUSTER_COUNT:
        raise _decision_error(
            path,
            "expected_global_pair_source_map must contain "
            f"{EXPECTED_DECISION_SOURCE_CLUSTER_COUNT} source clusters",
        )
    if payload.get("expected_global_rows") != EXPECTED_DECISION_GROUP_COUNT:
        raise _decision_error(
            path,
            f"expected_global_rows must be {EXPECTED_DECISION_GROUP_COUNT}",
        )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise _decision_error(path, "rows must be a list")
    expected_rows = payload.get("expected_rows")
    completed_rows = payload.get("completed_rows")
    if expected_rows != completed_rows or completed_rows != len(rows):
        raise _decision_error(
            path,
            f"incomplete shard: expected={expected_rows}, completed={completed_rows}, rows={len(rows)}",
        )


def _validate_decision_grid(rows: Sequence[Mapping[str, Any]], identity: Mapping[str, Any]) -> None:
    expected_grid = _manifest_grid(str(identity["episode_root"]), "val")
    expected_sources = set(expected_grid.values())
    if len(expected_grid) != EXPECTED_DECISION_GROUP_COUNT:
        raise ValueError(
            f"val manifest must contain {EXPECTED_DECISION_GROUP_COUNT} groups, found {len(expected_grid)}"
        )
    if len(expected_sources) != EXPECTED_DECISION_SOURCE_CLUSTER_COUNT:
        raise ValueError(
            "val manifest must contain "
            f"{EXPECTED_DECISION_SOURCE_CLUSTER_COUNT} source clusters, found {len(expected_sources)}"
        )
    declared_grid = identity.get("expected_global_pair_source_map")
    if declared_grid != expected_grid:
        raise ValueError(
            "expected_global_pair_source_map does not match the current val manifest"
        )

    by_seed: dict[int, dict[str, str]] = defaultdict(dict)
    for row in rows:
        seed = int(row["seed"])
        pair_id = str(row["pair_id"])
        if pair_id in by_seed[seed]:
            raise ValueError(f"duplicate decision row for seed={seed}, pair_id={pair_id}")
        by_seed[seed][pair_id] = _normalize_source(row.get("source_initial_state_index"))
        if row.get("split") != "val":
            raise ValueError(f"decision row is not from val split: seed={seed}, pair_id={pair_id}")

    if tuple(sorted(by_seed)) != EXPECTED_DECISION_SEEDS:
        raise ValueError(
            f"decision grid must contain seeds {EXPECTED_DECISION_SEEDS}, found {tuple(sorted(by_seed))}"
        )
    for seed in EXPECTED_DECISION_SEEDS:
        if by_seed[seed] != expected_grid:
            missing = sorted(set(expected_grid) - set(by_seed[seed]))
            extra = sorted(set(by_seed[seed]) - set(expected_grid))
            mismatched = sorted(
                pair_id
                for pair_id in set(expected_grid) & set(by_seed[seed])
                if expected_grid[pair_id] != by_seed[seed][pair_id]
            )
            raise ValueError(
                f"seed {seed} does not match the complete val manifest grid: "
                f"missing={missing}, extra={extra}, source_mismatch={mismatched}"
            )


def _smoke_reasons(payloads: Sequence[Mapping[str, Any]]) -> list[str]:
    reasons = {"run_kind is smoke or missing; result is not decision-bearing"}
    for payload in payloads:
        if payload.get("status") != "complete":
            reasons.add("one or more smoke inputs are incomplete")
        if payload.get("git_dirty_at_launch"):
            reasons.add("one or more smoke inputs came from a dirty worktree")
        if not payload.get("git_sha") or not payload.get("policy_checkpoint_sha256"):
            reasons.add("one or more smoke inputs lack complete source/checkpoint identity")
        if _preregistration_sha256(payload) != EXPECTED_PREREGISTRATION_SHA256:
            reasons.add("smoke preregistration hash is missing or does not match the decision protocol")
    return sorted(reasons)


def load_and_validate(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not paths:
        raise ValueError("at least one recovery-segment result is required")
    payloads = [json.loads(path.read_text()) for path in paths]
    for path, payload in zip(paths, payloads):
        if payload.get("schema_version") not in {1, EXPECTED_SCHEMA_VERSION}:
            raise ValueError(f"unsupported recovery-segment schema: {path}")
        if not isinstance(payload.get("rows"), list):
            raise ValueError(f"recovery-segment rows are missing: {path}")

    run_kinds = {_run_kind(payload) for payload in payloads}
    if len(run_kinds) != 1:
        raise ValueError(f"cannot mix decision and smoke inputs: {sorted(run_kinds)}")
    run_kind = next(iter(run_kinds))
    if run_kind == DECISION_RUN_KIND:
        for path, payload in zip(paths, payloads):
            _validate_decision_payload(path, payload)

    identity = _identity(payloads[0])
    for path, payload in zip(paths[1:], payloads[1:]):
        proposed = _identity(payload)
        if proposed != identity:
            raise ValueError(f"evaluation configuration mismatch: {path}")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for path, payload in zip(paths, payloads):
        payload_seed = int(payload.get("seed", -1))
        for row in payload["rows"]:
            if "seed" in row and int(row["seed"]) != payload_seed:
                raise ValueError(f"row seed does not match payload seed in {path}: {row['pair_id']}")
            key = (payload_seed, str(row["pair_id"]))
            if key in seen:
                raise ValueError(f"duplicate seed/group row {key} in {path}")
            seen.add(key)
            rows.append({**row, "seed": payload_seed, "source_file": str(path)})
    if not rows:
        raise ValueError("recovery-segment inputs contain no rows")

    if run_kind == DECISION_RUN_KIND:
        _validate_decision_grid(rows, identity)
        non_decision_reasons: list[str] = []
    else:
        non_decision_reasons = _smoke_reasons(payloads)
    return rows, {
        "identity": identity,
        "run_kind": run_kind,
        "decision_eligible": run_kind == DECISION_RUN_KIND,
        "non_decision_reasons": non_decision_reasons,
    }


def _extract_full_heldout_summary(
    result: Mapping[str, Any],
    *,
    decision_run: bool,
) -> tuple[Mapping[str, Any], str]:
    summary = result.get("full_heldout_summary")
    if isinstance(summary, Mapping):
        return summary, "full_heldout_summary"
    if not decision_run:
        for key in ("natural_outcome_summary", "natural_outcome", "outcome"):
            if isinstance(result.get(key), Mapping):
                return result[key], "natural_outcome_smoke_fallback"
    raise ValueError("method result is missing full_heldout_summary")


def _random4_schedule_results(result: Mapping[str, Any]) -> Sequence[Mapping[str, Any]] | None:
    for key in ("schedule_results", "random_schedule_results", "random_schedules", "schedules"):
        value = result.get(key)
        if isinstance(value, list):
            return value
    summaries = result.get("full_heldout_summaries")
    if isinstance(summaries, list):
        return [{"full_heldout_summary": summary} for summary in summaries]
    return None


def _method_summaries(
    row: Mapping[str, Any],
    method: str,
    *,
    decision_run: bool,
) -> tuple[list[Mapping[str, Any]], str]:
    try:
        result = row["methods"][method]
    except KeyError as exc:
        raise ValueError(f"row {row.get('pair_id')} is missing method {method}") from exc
    if not isinstance(result, Mapping):
        raise ValueError(f"row {row.get('pair_id')} method {method} is not a mapping")

    summary, source = _extract_full_heldout_summary(result, decision_run=decision_run)
    if not decision_run:
        return [summary], source

    if method != "random4":
        heldout_outcomes = result.get("full_heldout_outcomes")
        if not isinstance(heldout_outcomes, list) or len(heldout_outcomes) != 5:
            raise ValueError(
                f"row {row.get('pair_id')} method {method} requires 5 full-heldout outcomes"
            )
        return [summary], source

    schedules = _random4_schedule_results(result)
    if schedules is None:
        raise ValueError("decision random4 must expose three schedule_results")
    if len(schedules) != EXPECTED_RANDOM4_SCHEDULE_COUNT:
        raise ValueError(
            f"decision random4 requires {EXPECTED_RANDOM4_SCHEDULE_COUNT} schedules, found {len(schedules)}"
        )
    declared_count = result.get("random_schedule_count")
    if declared_count is None or int(declared_count) != EXPECTED_RANDOM4_SCHEDULE_COUNT:
        raise ValueError(
            f"row {row.get('pair_id')} random4 schedule count disagrees with its schedule list"
        )
    flattened_outcomes = result.get("full_heldout_outcomes")
    expected_flattened_count = EXPECTED_RANDOM4_SCHEDULE_COUNT * 5
    if not isinstance(flattened_outcomes, list) or len(flattened_outcomes) != expected_flattened_count:
        raise ValueError(
            f"row {row.get('pair_id')} random4 requires {expected_flattened_count} flattened full-heldout outcomes"
        )

    schedule_summaries = []
    identifiers = []
    for schedule in schedules:
        schedule_summary, _ = _extract_full_heldout_summary(
            schedule,
            decision_run=True,
        )
        schedule_summaries.append(schedule_summary)
        schedule_outcomes = schedule.get("full_heldout_outcomes")
        if not isinstance(schedule_outcomes, list) or len(schedule_outcomes) != 5:
            raise ValueError(
                f"row {row.get('pair_id')} random4 schedule requires 5 full-heldout outcomes"
            )
        identifier = schedule.get(
            "random_schedule_index",
            schedule.get("schedule_index", schedule.get("schedule_id")),
        )
        if identifier is not None:
            identifiers.append(str(identifier))
    expected_identifiers = {str(index) for index in range(EXPECTED_RANDOM4_SCHEDULE_COUNT)}
    if set(identifiers) != expected_identifiers:
        raise ValueError(
            f"row {row.get('pair_id')} random4 schedule identifiers must be 0, 1, and 2"
        )
    flattened_identifiers = [
        str(outcome.get("random_schedule_index"))
        for outcome in flattened_outcomes
        if isinstance(outcome, Mapping) and outcome.get("random_schedule_index") is not None
    ]
    if any(flattened_identifiers.count(identifier) != 5 for identifier in expected_identifiers):
        raise ValueError(
            f"row {row.get('pair_id')} flattened random4 outcomes must contain 5 rows per schedule"
        )
    for metric in METRICS:
        aggregate_value = _summary_metric(summary, metric)
        schedule_mean = float(
            np.mean([_summary_metric(item, metric) for item in schedule_summaries])
        )
        if not np.isclose(aggregate_value, schedule_mean, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"row {row.get('pair_id')} random4 aggregate full_heldout_summary "
                f"does not match its schedules for {metric}"
            )
    return [summary], source


def _summary_metric(summary: Mapping[str, Any], metric: str) -> float:
    rate_key = f"{metric}_rate"
    if rate_key in summary:
        value = float(summary[rate_key])
    elif metric in summary:
        value = float(summary[metric])
    else:
        raise ValueError(f"full-heldout summary is missing metric {metric}")
    if not np.isfinite(value):
        raise ValueError(f"full-heldout metric {metric} is non-finite")
    return value


def method_metric_value(
    row: Mapping[str, Any],
    method: str,
    metric: str,
    *,
    decision_run: bool,
) -> float:
    summaries, _ = _method_summaries(row, method, decision_run=decision_run)
    return float(np.mean([_summary_metric(summary, metric) for summary in summaries]))


def _scaled_summary(summary: Mapping[str, Any], scale: float) -> dict[str, Any]:
    return {
        key: (int(value) if key == "count" else float(value) * scale)
        for key, value in summary.items()
    }


def paired_method_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    metric: str,
    bootstrap_samples: int,
    seed: int,
    reference_method: str = "sample0",
    decision_run: bool = True,
) -> dict[str, Any]:
    if method == reference_method:
        raise ValueError("method and reference_method must differ")
    deltas_by_pair: dict[str, list[float]] = defaultdict(list)
    deltas_by_seed_source: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        source_cluster = _normalize_source(row.get("source_initial_state_index"))
        reference = method_metric_value(
            row,
            reference_method,
            metric,
            decision_run=decision_run,
        )
        proposed = method_metric_value(row, method, metric, decision_run=decision_run)
        delta = proposed - reference
        deltas_by_pair[str(row["pair_id"])].append(delta)
        deltas_by_seed_source[(int(row["seed"]), source_cluster)].append(delta)

    seed_source_deltas = {
        key: float(np.mean(values)) for key, values in sorted(deltas_by_seed_source.items())
    }
    source_values: dict[str, list[float]] = defaultdict(list)
    per_seed: dict[str, list[float]] = defaultdict(list)
    for (run_seed, source_cluster), delta in seed_source_deltas.items():
        source_values[source_cluster].append(delta)
        per_seed[str(run_seed)].append(delta)
    source_deltas = {source: float(np.mean(values)) for source, values in sorted(source_values.items())}
    cluster_summary = bootstrap_summary(
        list(source_deltas.values()),
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    per_seed_source_cluster_level = {
        run_seed: bootstrap_summary(
            values,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 10_000 + int(run_seed),
        )
        for run_seed, values in sorted(per_seed.items())
    }
    per_seed_mean = {
        run_seed: summary["mean"]
        for run_seed, summary in per_seed_source_cluster_level.items()
    }
    result: dict[str, Any] = {
        "direction": f"{method}_minus_{reference_method}",
        "metric_source": (
            "full_heldout_summary"
            if decision_run
            else "full_heldout_summary_or_natural_outcome_smoke_fallback"
        ),
        "independent_unit": "source_initial_state_index",
        "random4_schedule_aggregation": (
            "validated_3_schedule_full_heldout_summary_before_source_cluster_pairing"
        ),
        "source_cluster_level": cluster_summary,
        "per_source_cluster": source_deltas,
        "per_pair_diagnostic": {
            pair_id: float(np.mean(values)) for pair_id, values in sorted(deltas_by_pair.items())
        },
        "per_seed_source_cluster_level": per_seed_source_cluster_level,
        "per_seed_mean": per_seed_mean,
        "per_seed_source_cluster_count": {
            run_seed: len(values) for run_seed, values in sorted(per_seed.items())
        },
    }
    if metric in BINARY_METRICS:
        result["absolute_percentage_points"] = {
            "source_cluster_level": _scaled_summary(cluster_summary, 100.0),
            "per_source_cluster": {key: value * 100.0 for key, value in source_deltas.items()},
            "per_pair_diagnostic": {
                key: value * 100.0 for key, value in result["per_pair_diagnostic"].items()
            },
            "per_seed_source_cluster_level": {
                key: _scaled_summary(value, 100.0)
                for key, value in per_seed_source_cluster_level.items()
            },
            "per_seed_mean": {key: value * 100.0 for key, value in per_seed_mean.items()},
        }
    return result


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
    decision_run: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "metric_source": (
            "full_heldout_summary"
            if decision_run
            else "full_heldout_summary_or_natural_outcome_smoke_fallback"
        ),
        "paired_vs_sample0": {},
        "gate_pairwise": {},
    }
    for method_index, method in enumerate(METHODS[1:], start=1):
        result["paired_vs_sample0"][method] = {
            metric: paired_method_summary(
                rows,
                method=method,
                metric=metric,
                bootstrap_samples=bootstrap_samples,
                seed=seed + method_index * 100 + metric_index,
                decision_run=decision_run,
            )
            for metric_index, metric in enumerate(METRICS)
        }

    for comparison_index, reference_method in enumerate(("random4", "myopic_stage"), start=1):
        name = f"receding_oracle_minus_{reference_method}"
        result["gate_pairwise"][name] = {
            metric: paired_method_summary(
                rows,
                method="receding_oracle",
                reference_method=reference_method,
                metric=metric,
                bootstrap_samples=bootstrap_samples,
                seed=seed + 1_000 + comparison_index * 100 + metric_index,
                decision_run=decision_run,
            )
            for metric_index, metric in enumerate(METRICS)
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge fixed-budget recovery-segment Oracle shards")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, validation = load_and_validate(args.inputs)
    decision_run = bool(validation["decision_eligible"])
    payload = {
        "schema_version": 2,
        "status": "complete",
        "run_kind": validation["run_kind"],
        "decision_eligible": decision_run,
        "non_decision_reasons": validation["non_decision_reasons"],
        "input_files": [str(path) for path in args.inputs],
        "evaluation_identity": validation["identity"],
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "group_count": len({str(row["pair_id"]) for row in rows}),
        "source_cluster_count": len(
            {_normalize_source(row["source_initial_state_index"]) for row in rows}
        ),
        "row_count": len(rows),
        "summary": build_summary(
            rows,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            decision_run=decision_run,
        ),
        "rows": rows,
    }
    _atomic_write_json(args.output, payload)


if __name__ == "__main__":
    main()
