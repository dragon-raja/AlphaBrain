from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from summarize_kyc_scaling_stage_b1 import (
    BOOTSTRAP_SEED,
    METRICS,
    episode_key,
    in_training_support,
    method_metrics,
    paired_group_bootstrap,
)


METHODS = (
    "dual_rgb_fla",
    "dual_control_fla",
    "external_fla",
    "wrist_fla",
    "dual_fla",
)
WRIST_INTERVENTIONS = ("initial", "lagged")
STRATA = ("inside_training_support", "objects_visible", "fully_visible")


def _parse_named_path(
    value: str,
    *,
    allowed: Sequence[str],
    kind: str,
) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or name not in allowed or not path:
        raise ValueError(f"{kind} must be NAME=PATH with NAME in {tuple(allowed)}")
    return name, Path(path)


def _load_evaluation(path: Path, *, method: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"incomplete evaluation for {method}: {path}")
    rows = [{**row, "method": method} for row in payload.get("rows", [])]
    if not rows:
        raise ValueError(f"evaluation has no rows for {method}: {path}")
    return rows


def _validate_paired(rows_by_method: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    keys = {
        method: {episode_key(row) for row in rows}
        for method, rows in rows_by_method.items()
    }
    if len({frozenset(values) for values in keys.values()}) != 1:
        counts = {method: len(values) for method, values in keys.items()}
        raise ValueError(f"evaluations are not episode-paired: {counts}")


def _stratum_rows(
    rows: Sequence[Mapping[str, Any]],
    stratum: str,
) -> list[Mapping[str, Any]]:
    selected = [row for row in rows if in_training_support(row)]
    if stratum == "inside_training_support":
        return selected
    if stratum == "objects_visible":
        return [row for row in selected if bool(row["task_objects_visible"])]
    if stratum == "fully_visible":
        return [
            row
            for row in selected
            if bool(row["task_objects_fully_visible"])
            and bool(row["task_centers_in_frame"])
        ]
    raise ValueError(f"unknown stratum: {stratum}")


def grouped_linear_contrast(
    rows: Sequence[Mapping[str, Any]],
    *,
    coefficients: Mapping[str, float],
    metric: str,
    bootstrap_resamples: int,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    values = {
        method: {
            episode_key(row): float(row[metric])
            for row in rows
            if row["method"] == method
        }
        for method in coefficients
    }
    if len({frozenset(method_values) for method_values in values.values()}) != 1:
        raise ValueError("linear contrast requires paired episodes")

    grouped: dict[int, list[float]] = defaultdict(list)
    first_method = next(iter(coefficients))
    for key in values[first_method]:
        contrast = sum(
            float(coefficient) * values[method][key]
            for method, coefficient in coefficients.items()
        )
        grouped[int(key[1])].append(contrast)
    if not grouped:
        raise ValueError("linear contrast has no snapshot groups")

    group_values = np.asarray(
        [np.mean(grouped[index]) for index in sorted(grouped)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(group_values),
        size=(bootstrap_resamples, len(group_values)),
    )
    distribution = group_values[indices].mean(axis=1)
    return {
        "delta": float(group_values.mean()),
        "ci95_low": float(np.quantile(distribution, 0.025)),
        "ci95_high": float(np.quantile(distribution, 0.975)),
        "snapshot_group_count": int(len(group_values)),
        "bootstrap_resamples": int(bootstrap_resamples),
        "coefficients": dict(coefficients),
    }


def _effect_bundle(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    reference: str,
    bootstrap_resamples: int,
    seed_offset: int,
) -> dict[str, Any]:
    return {
        metric: paired_group_bootstrap(
            rows,
            method=method,
            reference=reference,
            metric=metric,
            bootstrap_resamples=bootstrap_resamples,
            seed=BOOTSTRAP_SEED + seed_offset,
        )
        for metric in METRICS
    }


def summarize(
    *,
    evaluation_paths: Mapping[str, Path],
    intervention_paths: Mapping[str, Path],
    bootstrap_resamples: int,
) -> dict[str, Any]:
    rows_by_method = {
        method: _load_evaluation(evaluation_paths[method], method=method)
        for method in METHODS
    }
    _validate_paired(rows_by_method)
    rows = [row for method in METHODS for row in rows_by_method[method]]

    strata: dict[str, Any] = {}
    comparisons = (
        ("dual_fla", "dual_control_fla"),
        ("external_fla", "dual_control_fla"),
        ("wrist_fla", "dual_control_fla"),
        ("dual_fla", "dual_rgb_fla"),
        ("dual_control_fla", "dual_rgb_fla"),
    )
    for stratum_index, stratum in enumerate(STRATA):
        selected = _stratum_rows(rows, stratum)
        if any(not any(row["method"] == method for row in selected) for method in METHODS):
            raise ValueError(f"{stratum} is missing one or more methods")
        paired_differences = {
            f"{method}_minus_{reference}": _effect_bundle(
                selected,
                method=method,
                reference=reference,
                bootstrap_resamples=bootstrap_resamples,
                seed_offset=100 * stratum_index + comparison_index,
            )
            for comparison_index, (method, reference) in enumerate(comparisons)
        }
        paired_differences["dual_interaction"] = {
            metric: grouped_linear_contrast(
                selected,
                coefficients={
                    "dual_fla": 1.0,
                    "external_fla": -1.0,
                    "wrist_fla": -1.0,
                    "dual_control_fla": 1.0,
                },
                metric=metric,
                bootstrap_resamples=bootstrap_resamples,
                seed=BOOTSTRAP_SEED + 100 * stratum_index + 50,
            )
            for metric in METRICS
        }
        strata[stratum] = {
            "methods": {
                method: method_metrics(selected, method=method) for method in METHODS
            },
            "paired_differences": paired_differences,
        }

    correct_rows = [
        {**row, "method": "correct"} for row in rows_by_method["dual_fla"]
    ]
    intervention_rows = {
        name: _load_evaluation(intervention_paths[name], method=name)
        for name in WRIST_INTERVENTIONS
    }
    _validate_paired({"correct": correct_rows, **intervention_rows})
    causal_rows = correct_rows + [
        row for name in WRIST_INTERVENTIONS for row in intervention_rows[name]
    ]
    causal = {}
    for stratum_index, stratum in enumerate(STRATA):
        selected = _stratum_rows(causal_rows, stratum)
        causal[stratum] = {
            "conditions": {
                condition: method_metrics(selected, method=condition)
                for condition in ("correct", *WRIST_INTERVENTIONS)
            },
            "correct_minus_initial": _effect_bundle(
                selected,
                method="correct",
                reference="initial",
                bootstrap_resamples=bootstrap_resamples,
                seed_offset=1000 + 10 * stratum_index,
            ),
            "correct_minus_lagged": _effect_bundle(
                selected,
                method="correct",
                reference="lagged",
                bootstrap_resamples=bootstrap_resamples,
                seed_offset=1001 + 10 * stratum_index,
            ),
        }

    primary = strata["inside_training_support"]
    baseline_success = max(
        float(primary["methods"]["dual_rgb_fla"]["success"]),
        float(primary["methods"]["dual_control_fla"]["success"]),
    )
    dual_gain = float(
        primary["paired_differences"]["dual_fla_minus_dual_control_fla"][
            "success"
        ]["delta"]
    )
    external_gain = float(
        primary["paired_differences"]["external_fla_minus_dual_control_fla"][
            "success"
        ]["delta"]
    )
    wrist_gain = float(
        primary["paired_differences"]["wrist_fla_minus_dual_control_fla"][
            "success"
        ]["delta"]
    )
    causal_gains = {
        mode: float(causal["inside_training_support"][f"correct_minus_{mode}"]["success"]["delta"])
        for mode in WRIST_INTERVENTIONS
    }

    canonical_rows = [row for row in rows if row["camera_pose"] == "baseline"]
    canonical_dual_gain = float(
        paired_group_bootstrap(
            canonical_rows,
            method="dual_fla",
            reference="dual_control_fla",
            metric="success",
            bootstrap_resamples=bootstrap_resamples,
            seed=BOOTSTRAP_SEED + 2000,
        )["delta"]
    )
    baseline_valid = baseline_success >= 0.20
    geometry_gain = max(dual_gain, external_gain, wrist_gain) >= 0.05
    causal_wrist_use = max(causal_gains.values()) >= 0.05
    normal_view_preserved = canonical_dual_gain >= -0.05
    if not baseline_valid:
        decision = "BASELINE_INVALID"
    elif dual_gain >= 0.05 and causal_wrist_use and normal_view_preserved:
        decision = "ADVANCE_DUAL_CAMERA_CONFIRMATION"
    elif geometry_gain and normal_view_preserved:
        decision = "ADVANCE_SINGLE_VIEW_DIAGNOSIS"
    else:
        decision = "DO_NOT_ADVANCE_FROM_DUAL_CAMERA_SCREEN"

    return {
        "schema_version": 1,
        "status": "complete",
        "study": "kyc_pi05_dual_camera_screen",
        "seed": 41,
        "training_updates": 2000,
        "execution_horizon": 3,
        "inference_unit": "canonical_state_index",
        "methods": list(METHODS),
        "strata": strata,
        "wrist_ray_causal_intervention": causal,
        "gate": {
            "baseline_success": baseline_success,
            "minimum_baseline_success": 0.20,
            "dual_minus_control_success": dual_gain,
            "external_minus_control_success": external_gain,
            "wrist_minus_control_success": wrist_gain,
            "correct_minus_initial_success": causal_gains["initial"],
            "correct_minus_lagged_success": causal_gains["lagged"],
            "canonical_dual_minus_control_success": canonical_dual_gain,
            "minimum_geometry_gain": 0.05,
            "minimum_causal_wrist_gain": 0.05,
            "maximum_normal_view_degradation": 0.05,
            "baseline_valid": baseline_valid,
            "geometry_gain": geometry_gain,
            "causal_wrist_use": causal_wrist_use,
            "normal_view_preserved": normal_view_preserved,
            "decision": decision,
        },
    }


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Pi0.5 dual-camera KYC screen")
    parser.add_argument("--evaluation", action="append", required=True)
    parser.add_argument("--wrist-intervention", action="append", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    evaluation_pairs = [
        _parse_named_path(value, allowed=METHODS, kind="evaluation")
        for value in args.evaluation
    ]
    intervention_pairs = [
        _parse_named_path(
            value,
            allowed=WRIST_INTERVENTIONS,
            kind="wrist intervention",
        )
        for value in args.wrist_intervention
    ]
    evaluation_paths = dict(evaluation_pairs)
    intervention_paths = dict(intervention_pairs)
    if len(evaluation_pairs) != len(METHODS) or set(evaluation_paths) != set(METHODS):
        raise ValueError("exactly one evaluation is required for every method")
    if (
        len(intervention_pairs) != len(WRIST_INTERVENTIONS)
        or set(intervention_paths) != set(WRIST_INTERVENTIONS)
    ):
        raise ValueError("exactly one initial and lagged wrist intervention is required")
    if args.bootstrap_resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite summary: {args.output}")

    payload = summarize(
        evaluation_paths=evaluation_paths,
        intervention_paths=intervention_paths,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["gate"], sort_keys=True))


if __name__ == "__main__":
    main()
