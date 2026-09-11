from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from compare_kyc_camera_evaluations import (
    index_fov_rows,
    join_episode_rows,
    validate_paired_evaluations,
)
from summarize_kyc_scaling_stage_b1 import (
    BOOTSTRAP_SEED,
    METRICS,
    episode_key,
    in_training_support,
    method_metrics,
    paired_group_bootstrap,
)


def scene_tag(scene: str) -> str:
    return {"fixed": "fx", "cue_randomized": "cue"}[scene]


def arm_tag(arm: str) -> str:
    return {"poseaug_control": "ctrl", "kyc": "kyc"}[arm]


def evaluation_path(
    root: Path,
    *,
    budget: int,
    train_scene: str,
    wrist: str,
    arm: str,
    seed: int,
    eval_scene: str,
) -> Path:
    name = (
        f"n{budget}-tr{scene_tag(train_scene)}-w{wrist}-"
        f"m{arm_tag(arm)}-s{seed}-ev{scene_tag(eval_scene)}"
    )
    return root / f"n{budget}" / name / "camera_sweep_test.json"


def primary_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row["visibility_stratum"] == "fully_supported"
        and in_training_support(row)
    ]


def group_method_gain(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> dict[int, float]:
    indexed = {
        method: {
            episode_key(row): float(row[metric])
            for row in rows
            if row["method"] == method
        }
        for method in ("poseaug_control", "kyc")
    }
    if set(indexed["poseaug_control"]) != set(indexed["kyc"]):
        raise ValueError("factorial cell does not have paired Control/KYC episodes")
    grouped: dict[int, list[float]] = defaultdict(list)
    for key, control_value in indexed["poseaug_control"].items():
        grouped[key[1]].append(indexed["kyc"][key] - control_value)
    return {
        state_index: float(np.mean(values))
        for state_index, values in sorted(grouped.items())
    }


def bootstrap_group_contrast(
    gains_by_cell: Mapping[str, Mapping[int, float]],
    *,
    coefficients: Mapping[str, float],
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    if set(coefficients) - set(gains_by_cell):
        raise ValueError("contrast references an unknown factorial cell")
    groups = set.intersection(
        *(set(gains_by_cell[cell]) for cell in coefficients)
    )
    if not groups:
        raise ValueError("factorial contrast has no paired snapshot groups")
    values = np.asarray(
        [
            sum(
                float(coefficient) * gains_by_cell[cell][group]
                for cell, coefficient in coefficients.items()
            )
            for group in sorted(groups)
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(values),
        size=(bootstrap_resamples, len(values)),
    )
    distribution = values[indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(distribution, 0.025)),
        "ci95_high": float(np.quantile(distribution, 0.975)),
        "snapshot_group_count": len(values),
        "bootstrap_resamples": bootstrap_resamples,
        "coefficients": dict(coefficients),
    }


def summarize_factorial(
    *,
    evaluation_root: Path,
    fov_paths: Sequence[Path],
    budget: int,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    fov_rows = index_fov_rows(
        [json.loads(path.read_text()) for path in fov_paths]
    )
    cell_payloads = {}
    matched_gains: dict[str, dict[str, dict[int, float]]] = {
        metric: {} for metric in METRICS
    }
    for train_scene in ("fixed", "cue_randomized"):
        for wrist in ("on", "off"):
            for eval_scene in ("fixed", "cue_randomized"):
                evaluations = {
                    arm: json.loads(
                        evaluation_path(
                            evaluation_root,
                            budget=budget,
                            train_scene=train_scene,
                            wrist=wrist,
                            arm=arm,
                            seed=seed,
                            eval_scene=eval_scene,
                        ).read_text()
                    )
                    for arm in ("poseaug_control", "kyc")
                }
                joined = join_episode_rows(
                    validate_paired_evaluations(evaluations),
                    fov_rows,
                )
                selected = primary_rows(joined)
                key = (
                    f"train_{scene_tag(train_scene)}__wrist_{wrist}__"
                    f"eval_{scene_tag(eval_scene)}"
                )
                cell_payloads[key] = {
                    "train_scene": train_scene,
                    "wrist": wrist,
                    "eval_scene": eval_scene,
                    "matched_train_eval_scene": train_scene == eval_scene,
                    "methods": {
                        method: method_metrics(selected, method=method)
                        for method in ("poseaug_control", "kyc")
                    },
                    "kyc_minus_control": {
                        metric: paired_group_bootstrap(
                            selected,
                            method="kyc",
                            reference="poseaug_control",
                            metric=metric,
                            bootstrap_resamples=bootstrap_resamples,
                            seed=BOOTSTRAP_SEED + budget + seed,
                        )
                        for metric in METRICS
                    },
                }
                if train_scene == eval_scene:
                    matched_key = f"{scene_tag(train_scene)}_{wrist}"
                    for metric in METRICS:
                        matched_gains[metric][matched_key] = group_method_gain(
                            selected,
                            metric=metric,
                        )

    contrasts = {
        "wrist_interaction_fixed": {"fx_off": 1.0, "fx_on": -1.0},
        "wrist_interaction_cue": {"cue_off": 1.0, "cue_on": -1.0},
        "scene_interaction_wrist_on": {"cue_on": 1.0, "fx_on": -1.0},
        "scene_interaction_wrist_off": {"cue_off": 1.0, "fx_off": -1.0},
        "three_way_interaction": {
            "cue_off": 1.0,
            "cue_on": -1.0,
            "fx_off": -1.0,
            "fx_on": 1.0,
        },
    }
    interaction_payload = {
        name: {
            metric: bootstrap_group_contrast(
                matched_gains[metric],
                coefficients=coefficients,
                bootstrap_resamples=bootstrap_resamples,
                seed=BOOTSTRAP_SEED + budget + seed + index,
            )
            for metric in METRICS
        }
        for index, (name, coefficients) in enumerate(contrasts.items())
    }
    primary_interactions = (
        "wrist_interaction_fixed",
        "wrist_interaction_cue",
        "scene_interaction_wrist_on",
        "scene_interaction_wrist_off",
    )
    maximum_success_interaction = max(
        abs(interaction_payload[name]["success"]["estimate"])
        for name in primary_interactions
    )
    confirmation = {
        "maximum_absolute_primary_success_interaction": (
            maximum_success_interaction
        ),
        "threshold": 0.05,
        "seeds": [42, 43],
        "scope": (
            "complete_factorial"
            if maximum_success_interaction >= 0.05
            else "wrist_off_cells"
        ),
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "study": "kyc_scene_cue_by_wrist_factorial",
        "budget": budget,
        "seed": seed,
        "primary_stratum": (
            "fully_supported_and_inside_training_camera_support"
        ),
        "inference_unit": "canonical_state_index",
        "cells": cell_payloads,
        "interactions": interaction_payload,
        "confirmation_rule": confirmation,
    }


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize KYC scene-cue by wrist factorial evaluations"
    )
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--fov-json", type=Path, nargs="+", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(args)


def main(args: Iterable[str] | None = None) -> None:
    parsed = parse_args(args)
    if parsed.output.exists():
        raise FileExistsError(f"refusing to overwrite summary: {parsed.output}")
    payload = summarize_factorial(
        evaluation_root=parsed.evaluation_root,
        fov_paths=parsed.fov_json,
        budget=parsed.budget,
        seed=parsed.seed,
        bootstrap_resamples=parsed.bootstrap_resamples,
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["confirmation_rule"], sort_keys=True))


if __name__ == "__main__":
    main()
