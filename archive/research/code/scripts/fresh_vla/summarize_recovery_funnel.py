from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

try:
    from scripts.fresh_vla.paired_evaluation import bootstrap_summary
except ModuleNotFoundError:
    from paired_evaluation import bootstrap_summary


SEEDS = (41, 42, 43)
METHODS = (
    "fixed_k1",
    "fixed_k2",
    "fixed_k3",
    "oracle_branch_safe_commit",
    "oracle_feedback_reveal_commit",
    "gripper_commit",
    "random_matched_commit",
    "self_consistency_commit",
)
EVALUATIONS = ("isolated", "end_to_end")
CHAIN_STAGES = ("event", "recovery_action", "regrasp", "transport", "success")
DIAGNOSTIC_STAGES = ("lift", "place")


def stage_indicators(row: Mapping[str, object]) -> dict[str, float]:
    marginal = {
        "event": float(bool(row.get("intervention_triggered")) and row.get("event_time") is not None),
        "recovery_action": float(bool(row.get("recovery_switch_observed"))),
        "regrasp": float(bool(row.get("regrasp_success"))),
        "lift": float(bool(row.get("lift_subgoal"))),
        "transport": float(bool(row.get("transport_subgoal"))),
        "place": float(bool(row.get("place_subgoal"))),
        "success": float(bool(row.get("recovery_success"))),
    }
    chain = {}
    reached = True
    for stage in CHAIN_STAGES:
        reached = reached and bool(marginal[stage])
        chain[stage] = float(reached)
    return {
        **{f"marginal_{stage}": value for stage, value in marginal.items()},
        **{f"chain_{stage}": value for stage, value in chain.items()},
    }


def conditional_rates(indicators: Sequence[Mapping[str, float]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for previous, current in zip(CHAIN_STAGES[:-1], CHAIN_STAGES[1:], strict=True):
        eligible = [row for row in indicators if row[f"chain_{previous}"] == 1.0]
        result[f"{previous}_to_{current}"] = (
            float(np.mean([row[f"chain_{current}"] for row in eligible])) if eligible else None
        )
    return result


def bootstrap_ratio(
    units: Sequence[tuple[float, float]],
    *,
    seed: int,
    samples: int = 10_000,
) -> dict[str, float | int | None]:
    values = np.asarray(units, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("ratio units must be (numerator, denominator) pairs")
    numerator = float(values[:, 0].sum())
    denominator = float(values[:, 1].sum())
    mean = numerator / denominator if denominator else None
    if not len(values) or not denominator:
        return {
            "mean": mean,
            "bootstrap_95_low": None,
            "bootstrap_95_high": None,
            "unit_count": len(values),
            "effective_numerator": numerator,
            "effective_denominator": denominator,
        }
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        sampled = values[rng.integers(0, len(values), size=len(values))]
        sampled_denominator = float(sampled[:, 1].sum())
        if sampled_denominator:
            draws.append(float(sampled[:, 0].sum()) / sampled_denominator)
    return {
        "mean": mean,
        "bootstrap_95_low": float(np.quantile(draws, 0.025)) if draws else None,
        "bootstrap_95_high": float(np.quantile(draws, 0.975)) if draws else None,
        "unit_count": len(values),
        "effective_numerator": numerator,
        "effective_denominator": denominator,
    }


def _load_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"incomplete result: {path}")
    return payload


def load_rows(
    method: str,
    seed: int,
    evaluation: str,
    *,
    baseline_root: Path,
    oracle_root: Path,
) -> list[dict[str, object]]:
    filename = f"closed_loop_{evaluation}.json"
    if method.startswith("fixed_k"):
        horizon = int(method[-1])
        path = baseline_root / f"fresh_closed_loop_full_h_seed{seed}" / filename
        rows = [
            dict(row)
            for row in _load_json(path)["rows"]
            if int(row["execution_horizon"]) == horizon
        ]
    else:
        path = oracle_root / f"oracle_commit_{method}_seed{seed}" / filename
        rows = [dict(row) for row in _load_json(path)["rows"]]
    slipped = [row for row in rows if row["branch_outcome"] == "slipped"]
    if len(slipped) != 13:
        raise ValueError(f"expected 13 slipped test groups for {method} seed {seed} {evaluation}, got {len(slipped)}")
    return slipped


def _bootstrap_by_unit(
    values_by_seed_pair: Mapping[int, Mapping[str, float]],
    pair_to_source: Mapping[str, str],
    *,
    cluster: str,
    seed: int,
) -> dict[str, object]:
    pair_ids = sorted(next(iter(values_by_seed_pair.values())))
    group_values = {
        pair_id: float(np.mean([values_by_seed_pair[run_seed][pair_id] for run_seed in SEEDS]))
        for pair_id in pair_ids
    }
    if cluster == "snapshot_group":
        units = list(group_values.values())
    elif cluster == "source_initial_state":
        source_values: dict[str, list[float]] = defaultdict(list)
        for pair_id, value in group_values.items():
            source_values[pair_to_source[pair_id]].append(value)
        units = [float(np.mean(values)) for values in source_values.values()]
    else:
        raise ValueError(f"unknown cluster: {cluster}")
    return bootstrap_summary(units, seed=seed)


def summarize_method(
    rows_by_seed: Mapping[int, Sequence[Mapping[str, object]]],
    pair_to_source: Mapping[str, str],
    *,
    seed: int,
) -> dict[str, object]:
    indicators_by_seed = {
        run_seed: {str(row["pair_id"]): stage_indicators(row) for row in rows}
        for run_seed, rows in rows_by_seed.items()
    }
    pair_sets = [set(rows) for rows in indicators_by_seed.values()]
    if any(pair_set != pair_sets[0] for pair_set in pair_sets[1:]):
        raise ValueError("test snapshot groups differ across seeds")
    metrics = tuple(indicators_by_seed[SEEDS[0]][next(iter(indicators_by_seed[SEEDS[0]]))])
    absolute = {}
    for metric_index, metric in enumerate(metrics):
        values = {
            run_seed: {pair_id: rows[pair_id][metric] for pair_id in rows}
            for run_seed, rows in indicators_by_seed.items()
        }
        absolute[metric] = {
            "per_seed": {
                str(run_seed): float(np.mean(list(pair_values.values())))
                for run_seed, pair_values in values.items()
            },
            "snapshot_group": _bootstrap_by_unit(
                values,
                pair_to_source,
                cluster="snapshot_group",
                seed=seed + metric_index * 2,
            ),
            "source_initial_state": _bootstrap_by_unit(
                values,
                pair_to_source,
                cluster="source_initial_state",
                seed=seed + metric_index * 2 + 1,
            ),
        }
    conditional_per_seed = {
        str(run_seed): conditional_rates(list(rows.values()))
        for run_seed, rows in indicators_by_seed.items()
    }
    conditional = {}
    for transition_index, (previous, current) in enumerate(
        zip(CHAIN_STAGES[:-1], CHAIN_STAGES[1:], strict=True)
    ):
        pair_units = []
        source_units: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for pair_id in sorted(pair_sets[0]):
            numerator = float(
                np.mean([indicators_by_seed[run_seed][pair_id][f"chain_{current}"] for run_seed in SEEDS])
            )
            denominator = float(
                np.mean([indicators_by_seed[run_seed][pair_id][f"chain_{previous}"] for run_seed in SEEDS])
            )
            pair_units.append((numerator, denominator))
            source_units[pair_to_source[pair_id]].append((numerator, denominator))
        source_mean_units = [
            (
                float(np.mean([value[0] for value in values])),
                float(np.mean([value[1] for value in values])),
            )
            for values in source_units.values()
        ]
        conditional[f"{previous}_to_{current}"] = {
            "per_seed": {
                str(run_seed): {
                    "rate": conditional_per_seed[str(run_seed)][f"{previous}_to_{current}"],
                    "numerator": int(
                        sum(row[f"chain_{current}"] for row in indicators_by_seed[run_seed].values())
                    ),
                    "denominator": int(
                        sum(row[f"chain_{previous}"] for row in indicators_by_seed[run_seed].values())
                    ),
                }
                for run_seed in SEEDS
            },
            "snapshot_group": bootstrap_ratio(
                pair_units,
                seed=seed + 1000 + transition_index * 2,
            ),
            "source_initial_state": bootstrap_ratio(
                source_mean_units,
                seed=seed + 1001 + transition_index * 2,
            ),
        }
    return {
        "absolute": absolute,
        "conditional": conditional,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the post-feedback slip recovery behavior funnel")
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/runs/libero-full-episode-final-v2"),
    )
    parser.add_argument(
        "--oracle-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/runs/libero-oracle-commit-final-v1"),
    )
    parser.add_argument(
        "--episode-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/libero-full-episode-v2-128"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/research-reset/recovery_funnel.json"),
    )
    parser.add_argument("--seed", type=int, default=161803)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.episode_root / "manifest.json").read_text())
    test_groups = [group for group in manifest["groups"] if group["split"] == "test"]
    pair_to_source = {
        str(group["pair_id"]): str(group["source_initial_state_index"])
        for group in test_groups
    }
    payload = {
        "experiment": "post_feedback_recovery_behavior_funnel",
        "statistical_unit": "snapshot group; source_initial_state is reported as a dependence-aware cluster sensitivity check",
        "seeds": list(SEEDS),
        "chain_stages": list(CHAIN_STAGES),
        "diagnostic_marginals": list(DIAGNOSTIC_STAGES),
        "definitions": {
            "marginal": "whether a row ever reached the named event or cumulative subgoal",
            "event": "slipped row with intervention_triggered and a non-null event_time; this does not prove policy feedback awareness",
            "recovery_action": "recovery_switch_observed heuristic: first ungrasped open-gripper or object-directed action",
            "chain": "logical intersection of every preceding named stage; not a temporal or causal proof because first-stage timestamps are unavailable",
            "conditional": "next chain-stage numerator divided by previous chain-stage denominator, recomputed inside each group/source-state bootstrap draw",
            "lift": "reported only as a marginal diagnostic because the schema cannot prove it happened after regrasp",
            "place": "reported only as a marginal diagnostic and is equivalent to environment success in the current evaluator",
        },
        "oracle_deduplication_note": "both Oracle methods are retained even though oracle_equivalence.json establishes row-semantic equivalence",
        "results": {},
    }
    for evaluation_index, evaluation in enumerate(EVALUATIONS):
        payload["results"][evaluation] = {}
        for method_index, method in enumerate(METHODS):
            rows_by_seed = {
                seed: load_rows(
                    method,
                    seed,
                    evaluation,
                    baseline_root=args.baseline_root,
                    oracle_root=args.oracle_root,
                )
                for seed in SEEDS
            }
            payload["results"][evaluation][method] = summarize_method(
                rows_by_seed,
                pair_to_source,
                seed=args.seed + evaluation_index * 10000 + method_index * 100,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "methods": len(METHODS)}, sort_keys=True))


if __name__ == "__main__":
    main()
