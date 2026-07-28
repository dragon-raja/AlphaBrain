from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


BOOTSTRAP_SEED = 20260728
METRICS = (
    "success",
    "transport_success",
    "progress",
    "completion_steps",
)
TRAINING_SUPPORT = {
    "camera_azimuth_deg": (-60.0, 60.0),
    "camera_elevation_deg": (-25.0, 25.0),
    "camera_radius_scale": (0.9, 1.25),
}


def in_training_support(row: Mapping[str, Any]) -> bool:
    return all(
        lower <= float(row[field]) <= upper
        for field, (lower, upper) in TRAINING_SUPPORT.items()
    )


def primary_row(row: Mapping[str, Any], *, data_split: str = "all") -> bool:
    if row["visibility_stratum"] != "fully_supported":
        return False
    if not in_training_support(row):
        return False
    return data_split == "all" or row["data_split"] == data_split


def episode_key(row: Mapping[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(row["edge_id"]),
        int(row["canonical_state_index"]),
        int(row["execution_horizon"]),
        str(row["camera_pose"]),
    )


def read_episode_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no comparison rows in {path}")
    return rows


def paired_group_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    reference: str,
    metric: str,
    bootstrap_resamples: int,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    selected = {
        name: {
            episode_key(row): float(row[metric])
            for row in rows
            if row["method"] == name
        }
        for name in (reference, method)
    }
    if set(selected[reference]) != set(selected[method]):
        raise ValueError(f"{method} and {reference} do not have paired episodes")
    grouped: dict[int, list[float]] = defaultdict(list)
    for key, reference_value in selected[reference].items():
        state_index = key[1]
        grouped[state_index].append(selected[method][key] - reference_value)
    if not grouped:
        raise ValueError("paired bootstrap has no snapshot groups")
    group_deltas = np.asarray(
        [np.mean(values) for _, values in sorted(grouped.items())],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(group_deltas),
        size=(bootstrap_resamples, len(group_deltas)),
    )
    distribution = group_deltas[indices].mean(axis=1)
    return {
        "delta": float(group_deltas.mean()),
        "ci95_low": float(np.quantile(distribution, 0.025)),
        "ci95_high": float(np.quantile(distribution, 0.975)),
        "snapshot_group_count": int(len(group_deltas)),
        "bootstrap_resamples": int(bootstrap_resamples),
    }


def method_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
) -> dict[str, float | int]:
    method_rows = [row for row in rows if row["method"] == method]
    if not method_rows:
        raise ValueError(f"no rows for method {method}")
    result: dict[str, float | int] = {
        "episode_count": len(method_rows),
        "snapshot_group_count": len(
            {int(row["canonical_state_index"]) for row in method_rows}
        ),
    }
    for metric in METRICS:
        result[metric] = float(np.mean([float(row[metric]) for row in method_rows]))
    return result


def budget_to_eighty_percent(
    values: Mapping[int, float],
) -> int:
    maximum = max(values.values())
    threshold = 0.8 * maximum
    return min(budget for budget, value in values.items() if value >= threshold)


def normalized_log_auc(values: Mapping[int, float]) -> float:
    ordered = sorted(values.items())
    x = np.log(np.asarray([budget for budget, _ in ordered], dtype=np.float64))
    y = np.asarray([value for _, value in ordered], dtype=np.float64)
    if len(x) < 2 or math.isclose(float(x[-1]), float(x[0])):
        raise ValueError("log-view AUC requires at least two distinct budgets")
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def select_factorial_budget(control_success: Mapping[int, float]) -> dict[str, Any]:
    ordered = sorted(control_success.items())
    eligible = [budget for budget, success in ordered if success >= 0.20]
    if not eligible:
        return {
            "status": "BASELINE_INVALID",
            "selected_budget": None,
            "rule": "no Control budget reached 20% primary success",
        }
    smallest_budget, smallest_success = ordered[0]
    if smallest_success >= 0.70:
        return {
            "status": "selected",
            "selected_budget": smallest_budget,
            "rule": "smallest budget already reached at least 70% success",
        }
    return {
        "status": "selected",
        "selected_budget": min(eligible),
        "rule": "smallest budget with at least 20% primary success",
    }


def summarize(
    analysis_root: Path,
    *,
    budgets: Sequence[int],
    bootstrap_resamples: int,
) -> dict[str, Any]:
    budget_payloads = {}
    success_by_method: dict[str, dict[int, float]] = defaultdict(dict)
    for budget in budgets:
        rows = read_episode_rows(analysis_root / f"n{budget}" / "episode_rows.csv")
        methods = sorted({str(row["method"]) for row in rows})
        split_payload = {}
        for data_split in ("all", "observed", "withheld"):
            selected_rows = [
                row for row in rows if primary_row(row, data_split=data_split)
            ]
            summaries = {
                method: method_metrics(selected_rows, method=method)
                for method in methods
            }
            comparisons = {}
            for method in methods:
                if method == "poseaug_control":
                    continue
                comparisons[f"{method}_minus_poseaug_control"] = {
                    metric: paired_group_bootstrap(
                        selected_rows,
                        method=method,
                        reference="poseaug_control",
                        metric=metric,
                        bootstrap_resamples=bootstrap_resamples,
                        seed=BOOTSTRAP_SEED + budget,
                    )
                    for metric in METRICS
                }
            split_payload[data_split] = {
                "definition": (
                    "fully_supported_and_inside_training_camera_support"
                ),
                "methods": summaries,
                "comparisons": comparisons,
            }
        canonical_rows = [row for row in rows if row["camera_pose"] == "baseline"]
        canonical = {
            method: method_metrics(canonical_rows, method=method)
            for method in methods
        }
        budget_payloads[str(budget)] = {
            "primary": split_payload,
            "canonical_view": canonical,
        }
        for method in ("poseaug_control", "kyc"):
            success_by_method[method][budget] = float(
                split_payload["all"]["methods"][method]["success"]
            )

    scaling = {
        method: {
            "success_by_budget": {
                str(budget): value for budget, value in sorted(values.items())
            },
            "budget_to_80pct_observed_max": budget_to_eighty_percent(values),
            "normalized_success_auc_over_log_views": normalized_log_auc(values),
        }
        for method, values in success_by_method.items()
    }
    selection = select_factorial_budget(success_by_method["poseaug_control"])
    return {
        "schema_version": 1,
        "status": "complete",
        "study": "kyc_pi05_view_count_scaling_stage_b1",
        "seed": 41,
        "budgets": list(budgets),
        "primary_stratum": {
            "visibility": "fully_supported",
            "camera_support": TRAINING_SUPPORT,
            "inference_unit": "canonical_state_index",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": bootstrap_resamples,
        },
        "budget_results": budget_payloads,
        "scaling": scaling,
        "factorial_budget_selection": selection,
    }


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize preregistered KYC Pi0.5 Stage B1 scaling"
    )
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[10, 45, 215, 1000])
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser.parse_args(args)


def main(args: Iterable[str] | None = None) -> None:
    parsed = parse_args(args)
    if parsed.output.exists():
        raise FileExistsError(f"refusing to overwrite summary: {parsed.output}")
    if parsed.bootstrap_resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    payload = summarize(
        parsed.analysis_root,
        budgets=parsed.budgets,
        bootstrap_resamples=parsed.bootstrap_resamples,
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["factorial_budget_selection"], sort_keys=True))


if __name__ == "__main__":
    main()
