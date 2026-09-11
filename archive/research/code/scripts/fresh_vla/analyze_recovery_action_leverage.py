from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluate_libero_closed_loop import _atomic_write_json
from evaluate_recovery_segment_oracle import BINARY_METRICS, METRICS
from paired_evaluation import bootstrap_summary
from summarize_recovery_segment_oracle import load_and_validate


def _outcome_value(outcome: Mapping[str, Any], metric: str) -> float:
    if metric not in outcome:
        raise ValueError(f"continuation outcome is missing {metric}")
    value = float(bool(outcome[metric])) if metric in BINARY_METRICS else float(
        outcome[metric]
    )
    if not np.isfinite(value):
        raise ValueError(f"continuation outcome {metric} is non-finite")
    return value


def decision_metric_matrix(
    decision: Mapping[str, Any],
    metric: str,
    *,
    row_label: str,
) -> np.ndarray:
    candidates = decision.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 4:
        raise ValueError(f"{row_label} decision requires four candidates")
    values = np.empty((4, 5), dtype=np.float64)
    for candidate_index, candidate in enumerate(candidates):
        continuations = candidate.get("decision_heldout_continuations")
        if not isinstance(continuations, list) or len(continuations) != 5:
            raise ValueError(
                f"{row_label} candidate {candidate_index} requires five heldout continuations"
            )
        repeats = []
        for continuation in continuations:
            repeat = continuation.get("repeat")
            if isinstance(repeat, bool) or not isinstance(repeat, int):
                raise ValueError(f"{row_label} heldout repeat must be an integer")
            repeats.append(repeat)
            bridge = continuation.get("bridge")
            if not isinstance(bridge, Mapping):
                raise ValueError(f"{row_label} heldout continuation lacks bridge outcome")
            values[candidate_index, repeat] = _outcome_value(bridge, metric)
        if sorted(repeats) != list(range(5)) or len(set(repeats)) != 5:
            raise ValueError(f"{row_label} heldout repeats must be exactly 0..4")

        summary = candidate.get("decision_heldout_summary")
        if not isinstance(summary, Mapping):
            raise ValueError(f"{row_label} candidate lacks decision_heldout_summary")
        summary_key = f"{metric}_rate" if metric in BINARY_METRICS else metric
        if not np.isclose(
            float(summary[summary_key]),
            float(np.mean(values[candidate_index])),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{row_label} candidate heldout summary disagrees with raw {metric}"
            )
    return values


def two_factor_variance(matrix: np.ndarray) -> dict[str, float]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("two-factor variance requires at least a 2x2 balanced matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("two-factor variance matrix contains non-finite values")
    action_count, continuation_count = values.shape
    grand = float(values.mean())
    action_means = values.mean(axis=1)
    continuation_means = values.mean(axis=0)
    action_ss = float(continuation_count * np.square(action_means - grand).sum())
    continuation_ss = float(
        action_count * np.square(continuation_means - grand).sum()
    )
    residual = values - action_means[:, None] - continuation_means[None, :] + grand
    residual_ss = float(np.square(residual).sum())
    total_ss = float(np.square(values - grand).sum())
    if not np.isclose(
        total_ss,
        action_ss + continuation_ss + residual_ss,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError("balanced ANOVA sum-of-squares decomposition failed")
    action_partial_denominator = action_ss + residual_ss
    continuation_partial_denominator = continuation_ss + residual_ss
    return {
        "action_mean_range": float(action_means.max() - action_means.min()),
        "continuation_mean_range": float(
            continuation_means.max() - continuation_means.min()
        ),
        "action_ss": action_ss,
        "continuation_ss": continuation_ss,
        "residual_ss": residual_ss,
        "total_ss": total_ss,
        "action_total_variance_share": action_ss / total_ss if total_ss > 0.0 else 0.0,
        "continuation_total_variance_share": (
            continuation_ss / total_ss if total_ss > 0.0 else 0.0
        ),
        "action_partial_eta_squared": (
            action_ss / action_partial_denominator
            if action_partial_denominator > 0.0
            else 0.0
        ),
        "continuation_partial_eta_squared": (
            continuation_ss / continuation_partial_denominator
            if continuation_partial_denominator > 0.0
            else 0.0
        ),
        "action_changes_candidate_mean": float(
            float(action_means.max() - action_means.min()) > 1e-12
        ),
    }


def decision_statistics(row: Mapping[str, Any]) -> dict[str, list[dict[str, float]]]:
    decisions = row["methods"]["receding_oracle"]["decisions"]
    if not isinstance(decisions, list) or len(decisions) != 4:
        raise ValueError(f"row {row.get('pair_id')} requires four Oracle decisions")
    return {
        metric: [
            two_factor_variance(
                decision_metric_matrix(
                    decision,
                    metric,
                    row_label=f"row {row.get('pair_id')} decision {decision_index}",
                )
            )
            for decision_index, decision in enumerate(decisions)
        ]
        for metric in METRICS
    }


def hierarchical_statistic(
    rows: Sequence[Mapping[str, Any]],
    statistics: Sequence[Mapping[str, Sequence[Mapping[str, float]]]],
    *,
    metric: str,
    statistic: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    by_seed_source: dict[tuple[int, str], list[float]] = defaultdict(list)
    raw_values = []
    for row, row_statistics in zip(rows, statistics, strict=True):
        values = [float(value[statistic]) for value in row_statistics[metric]]
        raw_values.extend(values)
        by_seed_source[
            (int(row["seed"]), str(int(row["source_initial_state_index"])))
        ].append(float(np.mean(values)))
    seed_source = {
        key: float(np.mean(values)) for key, values in sorted(by_seed_source.items())
    }
    by_source: dict[str, list[float]] = defaultdict(list)
    by_seed: dict[str, list[float]] = defaultdict(list)
    for (run_seed, source), value in seed_source.items():
        by_source[source].append(value)
        by_seed[str(run_seed)].append(value)
    source_values = {
        source: float(np.mean(values)) for source, values in sorted(by_source.items())
    }
    per_seed = {
        run_seed: bootstrap_summary(
            values,
            bootstrap_samples=bootstrap_samples,
            seed=seed + 10_000 + int(run_seed),
        )
        for run_seed, values in sorted(by_seed.items())
    }
    return {
        "raw_decision_count": len(raw_values),
        "raw_mean": float(np.mean(raw_values)),
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
    statistics = [decision_statistics(row) for row in rows]
    names = (
        "action_mean_range",
        "continuation_mean_range",
        "action_total_variance_share",
        "continuation_total_variance_share",
        "action_partial_eta_squared",
        "continuation_partial_eta_squared",
        "action_changes_candidate_mean",
    )
    result = {
        "independent_unit": "source_initial_state_index",
        "aggregation_order": (
            "mean across four decisions, then within (seed, source), then across seeds "
            "within source; bootstrap resamples source clusters"
        ),
        "design": "balanced 4 candidate actions x 5 shared continuation repeats",
        "interpretation_guard": (
            "Variance decomposition is descriptive. It does not treat candidates, "
            "continuations, or decision frames as independent statistical units."
        ),
        "metrics": {},
    }
    for metric_index, metric in enumerate(METRICS):
        result["metrics"][metric] = {
            statistic: hierarchical_statistic(
                rows,
                statistics,
                metric=metric,
                statistic=statistic,
                bootstrap_samples=bootstrap_samples,
                seed=seed + metric_index * 100 + statistic_index,
            )
            for statistic_index, statistic in enumerate(names)
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose recovery outcome variance into action and continuation effects"
    )
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_715)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, validation = load_and_validate(args.inputs)
    if not validation["decision_eligible"]:
        raise ValueError("action-leverage analysis requires a complete decision grid")
    payload = {
        "schema_version": 1,
        "experiment": "recovery_action_vs_continuation_variance",
        "input_files": [str(path) for path in args.inputs],
        "summary": build_summary(
            rows,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
    }
    _atomic_write_json(args.output, payload)
    concise = {
        metric: {
            "action_nonzero_rate": payload["summary"]["metrics"][metric][
                "action_changes_candidate_mean"
            ]["source_cluster_level"]["mean"],
            "action_mean_range": payload["summary"]["metrics"][metric][
                "action_mean_range"
            ]["source_cluster_level"]["mean"],
            "continuation_mean_range": payload["summary"]["metrics"][metric][
                "continuation_mean_range"
            ]["source_cluster_level"]["mean"],
            "action_variance_share": payload["summary"]["metrics"][metric][
                "action_total_variance_share"
            ]["source_cluster_level"]["mean"],
        }
        for metric in METRICS
    }
    print(json.dumps(concise, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
