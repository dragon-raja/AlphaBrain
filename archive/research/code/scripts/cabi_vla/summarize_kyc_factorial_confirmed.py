from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from compare_kyc_camera_evaluations import (
    index_fov_rows,
    join_episode_rows,
    validate_paired_evaluations,
)
from summarize_kyc_factorial import (
    evaluation_path,
    group_method_gain,
    primary_rows,
)
from summarize_kyc_scaling_stage_b1 import METRICS, method_metrics
from summarize_kyc_scaling_stage_b2 import hierarchical_group_bootstrap


def scaling_evaluation_path(
    root: Path,
    *,
    budget: int,
    arm: str,
    seed: int,
) -> Path:
    return (
        root
        / f"n{budget}"
        / f"n{budget}-{arm}-s{seed}-fixed-wrist-on"
        / "camera_sweep_test.json"
    )


def load_matched_cell(
    *,
    factor_eval_root: Path,
    scaling_eval_root: Path,
    fov_rows: Mapping,
    budget: int,
    seed: int,
    scene: str,
    wrist: str,
) -> list[Mapping[str, Any]]:
    paths = {}
    for arm in ("poseaug_control", "kyc"):
        if scene == "fixed" and wrist == "on":
            path = scaling_evaluation_path(
                scaling_eval_root,
                budget=budget,
                arm=arm,
                seed=seed,
            )
        else:
            path = evaluation_path(
                factor_eval_root,
                budget=budget,
                train_scene=scene,
                wrist=wrist,
                arm=arm,
                seed=seed,
                eval_scene=scene,
            )
        paths[arm] = path
    payloads = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
    return primary_rows(
        join_episode_rows(
            validate_paired_evaluations(payloads),
            fov_rows,
        )
    )


def equal_seed_method_mean(
    per_seed: Mapping[int, Mapping[str, Mapping[str, float]]],
    *,
    method: str,
) -> dict[str, float]:
    return {
        metric: float(
            np.mean(
                [
                    per_seed[seed][method][metric]
                    for seed in sorted(per_seed)
                ]
            )
        )
        for metric in METRICS
    }


def summarize_confirmed_factorial(
    *,
    seed41_summary: Mapping[str, Any],
    factor_eval_root: Path,
    scaling_eval_root: Path,
    fov_paths: Sequence[Path],
    bootstrap_resamples: int,
) -> dict[str, Any]:
    budget = int(seed41_summary["budget"])
    seeds = [41, *map(int, seed41_summary["confirmation_rule"]["seeds"])]
    scope = str(seed41_summary["confirmation_rule"]["scope"])
    if scope == "complete_factorial":
        cells = {
            "fx_on": ("fixed", "on"),
            "fx_off": ("fixed", "off"),
            "cue_on": ("cue_randomized", "on"),
            "cue_off": ("cue_randomized", "off"),
        }
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
    elif scope == "wrist_off_cells":
        cells = {
            "fx_off": ("fixed", "off"),
            "cue_off": ("cue_randomized", "off"),
        }
        contrasts = {
            "scene_interaction_wrist_off": {"cue_off": 1.0, "fx_off": -1.0},
        }
    else:
        raise ValueError(f"unknown confirmation scope: {scope}")

    fov_rows = index_fov_rows(
        [json.loads(path.read_text()) for path in fov_paths]
    )
    cell_payloads = {}
    gains: dict[str, dict[str, dict[int, float]]] = {
        metric: {} for metric in METRICS
    }
    for cell, (scene, wrist) in cells.items():
        per_seed_methods = {}
        for seed in seeds:
            rows = load_matched_cell(
                factor_eval_root=factor_eval_root,
                scaling_eval_root=scaling_eval_root,
                fov_rows=fov_rows,
                budget=budget,
                seed=seed,
                scene=scene,
                wrist=wrist,
            )
            per_seed_methods[seed] = {
                method: method_metrics(rows, method=method)
                for method in ("poseaug_control", "kyc")
            }
            for metric in METRICS:
                gains[metric].setdefault(cell, {})[seed] = group_method_gain(
                    rows,
                    metric=metric,
                )
        cell_payloads[cell] = {
            "scene": scene,
            "wrist": wrist,
            "per_seed": {
                str(seed): per_seed_methods[seed] for seed in seeds
            },
            "equal_seed_mean": {
                method: equal_seed_method_mean(
                    per_seed_methods,
                    method=method,
                )
                for method in ("poseaug_control", "kyc")
            },
            "hierarchical_kyc_minus_control": {
                metric: hierarchical_group_bootstrap(
                    gains[metric][cell],
                    resamples=bootstrap_resamples,
                    seed=20260728 + budget + metric_index,
                )
                for metric_index, metric in enumerate(METRICS)
            },
        }

    interaction_payload = {}
    for contrast_index, (name, coefficients) in enumerate(contrasts.items()):
        interaction_payload[name] = {}
        for metric_index, metric in enumerate(METRICS):
            contrast_by_seed = {}
            for seed in seeds:
                groups = set.intersection(
                    *(
                        set(gains[metric][cell][seed])
                        for cell in coefficients
                    )
                )
                contrast_by_seed[seed] = {
                    group: sum(
                        coefficient * gains[metric][cell][seed][group]
                        for cell, coefficient in coefficients.items()
                    )
                    for group in groups
                }
            interaction_payload[name][metric] = hierarchical_group_bootstrap(
                contrast_by_seed,
                resamples=bootstrap_resamples,
                seed=(
                    20260728
                    + budget
                    + 100 * contrast_index
                    + metric_index
                ),
            )
            interaction_payload[name][metric]["coefficients"] = dict(
                coefficients
            )

    return {
        "schema_version": 1,
        "status": "complete",
        "study": "kyc_scene_cue_by_wrist_factorial_confirmed",
        "budget": budget,
        "training_seeds": seeds,
        "confirmation_scope": scope,
        "primary_stratum": (
            "fully_supported_and_inside_training_camera_support"
        ),
        "inference_unit": "canonical_state_index",
        "cells": cell_payloads,
        "interactions": interaction_payload,
        "bootstrap_resamples": bootstrap_resamples,
    }


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize multi-seed KYC scene-cue by wrist confirmation"
    )
    parser.add_argument("--seed41-summary", type=Path, required=True)
    parser.add_argument("--factor-eval-root", type=Path, required=True)
    parser.add_argument("--scaling-eval-root", type=Path, required=True)
    parser.add_argument("--fov-json", type=Path, nargs="+", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(args)


def main(args: Iterable[str] | None = None) -> None:
    parsed = parse_args(args)
    if parsed.output.exists():
        raise FileExistsError(
            f"refusing to overwrite confirmed factorial summary: {parsed.output}"
        )
    payload = summarize_confirmed_factorial(
        seed41_summary=json.loads(parsed.seed41_summary.read_text()),
        factor_eval_root=parsed.factor_eval_root,
        scaling_eval_root=parsed.scaling_eval_root,
        fov_paths=parsed.fov_json,
        bootstrap_resamples=parsed.bootstrap_resamples,
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "scope": payload["confirmation_scope"],
                "cells": list(payload["cells"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
