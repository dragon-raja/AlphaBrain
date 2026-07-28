from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from summarize_kyc_factorial import group_method_gain
from summarize_kyc_scaling_stage_b1 import (
    BOOTSTRAP_SEED,
    METRICS,
    method_metrics,
    paired_group_bootstrap,
    primary_row,
    read_episode_rows,
)


def hierarchical_group_bootstrap(
    gains_by_seed: Mapping[int, Mapping[int, float]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    training_seeds = sorted(gains_by_seed)
    if not training_seeds:
        raise ValueError("hierarchical bootstrap has no seeds")
    rng = np.random.default_rng(seed)
    distribution = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled_seeds = rng.choice(
            training_seeds,
            size=len(training_seeds),
            replace=True,
        )
        seed_means = []
        for training_seed in sampled_seeds:
            group_values = np.asarray(
                list(gains_by_seed[int(training_seed)].values()),
                dtype=np.float64,
            )
            sampled_groups = rng.choice(
                group_values,
                size=len(group_values),
                replace=True,
            )
            seed_means.append(float(np.mean(sampled_groups)))
        distribution[index] = float(np.mean(seed_means))
    estimate = float(
        np.mean(
            [
                np.mean(list(group_values.values()))
                for group_values in gains_by_seed.values()
            ]
        )
    )
    return {
        "delta": estimate,
        "ci95_low": float(np.quantile(distribution, 0.025)),
        "ci95_high": float(np.quantile(distribution, 0.975)),
        "training_seed_count": len(training_seeds),
        "snapshot_groups_per_seed": {
            str(training_seed): len(gains_by_seed[training_seed])
            for training_seed in training_seeds
        },
        "bootstrap_resamples": resamples,
    }


def analysis_rows_path(root: Path, *, budget: int, seed: int) -> Path:
    if seed == 41:
        return root / f"n{budget}" / "episode_rows.csv"
    return root / f"n{budget}" / f"seed{seed}" / "episode_rows.csv"


def summarize_stage_b2(
    *,
    analysis_root: Path,
    selection: Mapping[str, Any],
    bootstrap_resamples: int,
) -> dict[str, Any]:
    seeds = [41, *map(int, selection["confirmation_seeds"])]
    budget_payloads = {}
    for budget in map(int, selection["training_budgets"]):
        per_seed = {}
        gains_by_metric = {metric: {} for metric in METRICS}
        for seed in seeds:
            rows = [
                row
                for row in read_episode_rows(
                    analysis_rows_path(analysis_root, budget=budget, seed=seed)
                )
                if primary_row(row)
            ]
            per_seed[str(seed)] = {
                "methods": {
                    method: method_metrics(rows, method=method)
                    for method in ("poseaug_control", "kyc")
                },
                "kyc_minus_control": {
                    metric: paired_group_bootstrap(
                        rows,
                        method="kyc",
                        reference="poseaug_control",
                        metric=metric,
                        bootstrap_resamples=bootstrap_resamples,
                        seed=BOOTSTRAP_SEED + budget + seed,
                    )
                    for metric in METRICS
                },
            }
            for metric in METRICS:
                gains_by_metric[metric][seed] = group_method_gain(
                    rows,
                    metric=metric,
                )
        equal_seed_mean = {
            method: {
                metric: float(
                    np.mean(
                        [
                            per_seed[str(seed)]["methods"][method][metric]
                            for seed in seeds
                        ]
                    )
                )
                for metric in METRICS
            }
            for method in ("poseaug_control", "kyc")
        }
        budget_payloads[str(budget)] = {
            "per_seed": per_seed,
            "equal_seed_mean": equal_seed_mean,
            "hierarchical_kyc_minus_control": {
                metric: hierarchical_group_bootstrap(
                    gains_by_metric[metric],
                    resamples=bootstrap_resamples,
                    seed=BOOTSTRAP_SEED + budget + metric_index,
                )
                for metric_index, metric in enumerate(METRICS)
            },
            "scaling_confirmation_budget": (
                budget in selection["scaling_confirmation_budgets"]
            ),
            "factorial_budget": budget == selection["factorial_budget"],
        }
    return {
        "schema_version": 1,
        "status": "complete",
        "study": "kyc_pi05_view_count_scaling_stage_b2",
        "training_seeds": seeds,
        "selection": dict(selection),
        "primary_stratum": (
            "fully_supported_and_inside_training_camera_support"
        ),
        "inference_unit": "canonical_state_index",
        "budget_results": budget_payloads,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": bootstrap_resamples,
    }


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize confirmed KYC Pi0.5 Stage B2 scaling"
    )
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(args)


def main(args: Iterable[str] | None = None) -> None:
    parsed = parse_args(args)
    if parsed.output.exists():
        raise FileExistsError(f"refusing to overwrite Stage B2 summary: {parsed.output}")
    payload = summarize_stage_b2(
        analysis_root=parsed.analysis_root,
        selection=json.loads(parsed.selection.read_text()),
        bootstrap_resamples=parsed.bootstrap_resamples,
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"budgets": list(payload["budget_results"])}, sort_keys=True))


if __name__ == "__main__":
    main()
