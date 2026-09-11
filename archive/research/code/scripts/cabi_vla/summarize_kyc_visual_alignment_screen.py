from __future__ import annotations

import argparse
import json
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


METHODS = ("poseaug_rgb_fla", "poseaug_control_fla", "kyc_fla")


def _parse_evaluation(value: str) -> tuple[str, Path]:
    method, separator, path = value.partition("=")
    if not separator or method not in METHODS or not path:
        raise ValueError(f"evaluation must be METHOD=PATH with METHOD in {METHODS}")
    return method, Path(path)


def _load_rows(paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    rows = []
    keys_by_method = {}
    for method in METHODS:
        payload = json.loads(paths[method].read_text())
        if payload.get("status") != "complete":
            raise ValueError(f"incomplete evaluation for {method}")
        method_rows = [{**row, "method": method} for row in payload["rows"]]
        keys_by_method[method] = {episode_key(row) for row in method_rows}
        rows.extend(method_rows)
    if len({frozenset(keys) for keys in keys_by_method.values()}) != 1:
        raise ValueError("visual alignment evaluations are not episode-paired")
    return rows


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


def _ray_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    rows = payload["rows"]
    if not rows:
        raise ValueError("ray diagnostic has no rows")
    result: dict[str, Any] = {"row_count": len(rows)}
    for comparison in ("canonical_vs_correct", "mismatched_vs_correct"):
        result[comparison] = {
            metric: float(np.mean([row[comparison][metric] for row in rows]))
            for metric in (
                "chunk_rms",
                "first_action_rms",
                "max_abs",
                "cosine_similarity",
            )
        }
    return result


def summarize(
    *,
    evaluation_paths: Mapping[str, Path],
    ray_diagnostic: Path,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    rows = _load_rows(evaluation_paths)
    strata = {}
    for stratum in (
        "inside_training_support",
        "objects_visible",
        "fully_visible",
    ):
        selected = _stratum_rows(rows, stratum)
        strata[stratum] = {
            "methods": {
                method: method_metrics(selected, method=method)
                for method in METHODS
            },
            "paired_differences": {
                f"{method}_minus_{reference}": {
                    metric: paired_group_bootstrap(
                        selected,
                        method=method,
                        reference=reference,
                        metric=metric,
                        bootstrap_resamples=bootstrap_resamples,
                        seed=BOOTSTRAP_SEED,
                    )
                    for metric in METRICS
                }
                for method, reference in (
                    ("kyc_fla", "poseaug_control_fla"),
                    ("kyc_fla", "poseaug_rgb_fla"),
                    ("poseaug_control_fla", "poseaug_rgb_fla"),
                )
            },
        }

    ray = _ray_metrics(ray_diagnostic)
    primary = strata["fully_visible"]
    control_success = float(primary["methods"]["poseaug_control_fla"]["success"])
    rgb_success = float(primary["methods"]["poseaug_rgb_fla"]["success"])
    success_gain = float(
        primary["paired_differences"][
            "kyc_fla_minus_poseaug_control_fla"
        ]["success"]["delta"]
    )
    wrong_ray_rms = float(ray["mismatched_vs_correct"]["chunk_rms"])
    baseline_valid = max(control_success, rgb_success) >= 0.20
    geometry_gain = success_gain >= 0.05
    causal_ray_use = wrong_ray_rms >= 0.005
    if not baseline_valid:
        decision = "BASELINE_INVALID"
    elif geometry_gain and causal_ray_use:
        decision = "ADVANCE_TO_FULL_CONFIRMATION"
    else:
        decision = "DO_NOT_ADVANCE_FROM_SCREEN"
    return {
        "schema_version": 1,
        "status": "complete",
        "study": "kyc_pi05_visual_alignment_screen",
        "inference_unit": "canonical_state_index",
        "strata": strata,
        "ray_diagnostic": ray,
        "gate": {
            "baseline_valid": baseline_valid,
            "minimum_baseline_success": 0.20,
            "kyc_minus_control_success": success_gain,
            "minimum_geometry_gain": 0.05,
            "mismatched_ray_chunk_rms": wrong_ray_rms,
            "minimum_causal_ray_rms": 0.005,
            "decision": decision,
        },
    }


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", action="append", required=True)
    parser.add_argument("--ray-diagnostic", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(args)


def main(args: Iterable[str] | None = None) -> None:
    parsed = parse_args(args)
    pairs = [_parse_evaluation(value) for value in parsed.evaluation]
    paths = dict(pairs)
    if set(paths) != set(METHODS) or len(pairs) != len(METHODS):
        raise ValueError(f"exactly one evaluation is required for each of {METHODS}")
    if parsed.output.exists():
        raise FileExistsError(f"refusing to overwrite summary: {parsed.output}")
    payload = summarize(
        evaluation_paths=paths,
        ray_diagnostic=parsed.ray_diagnostic,
        bootstrap_resamples=parsed.bootstrap_resamples,
    )
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["gate"], sort_keys=True))


if __name__ == "__main__":
    main()
