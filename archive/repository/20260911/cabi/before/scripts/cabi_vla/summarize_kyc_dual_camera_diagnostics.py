from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from summarize_kyc_dual_camera_screen import (
    METHODS,
    WRIST_INTERVENTIONS,
    grouped_linear_contrast,
)
from summarize_kyc_scaling_stage_b1 import episode_key


POSE_ORDER = (
    "baseline",
    "az_m60",
    "az_p60",
    "el_m25",
    "el_p25",
    "rad_0900",
    "rad_1250",
)
BEHAVIOR_METRICS = (
    "source_selection_success",
    "lift_success",
    "transport_success",
    "target_placement_success",
    "success",
    "progress",
    "completion_steps",
)
PAIR_FIELDS = (
    "scene_cue_seed",
    "initial_agent_sha256",
    "initial_wrist_sha256",
    "camera_position",
    "camera_quaternion_wxyz",
    "camera_intrinsics",
    "camera_to_world_opencv",
    "policy_camera_intrinsics",
    "policy_camera_to_world_opencv",
    "floor_texture_xy",
    "floor_texture_yaw_deg",
    "visual_table_xy",
    "visual_table_yaw_deg",
    "fixed_room_visuals_hidden",
    "robot_base_visual_hidden",
)


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


def _load_rows(path: Path, *, condition: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"incomplete evaluation for {condition}: {path}")
    rows = [{**row, "method": condition} for row in payload.get("rows", [])]
    if not rows:
        raise ValueError(f"evaluation has no rows for {condition}: {path}")
    return rows


def _paired_index(
    rows_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[tuple[str, int, int, str], Mapping[str, Any]]]:
    indexed = {
        condition: {episode_key(row): row for row in rows}
        for condition, rows in rows_by_condition.items()
    }
    if any(len(values) != len(rows_by_condition[condition]) for condition, values in indexed.items()):
        raise ValueError("evaluation contains duplicate episode keys")
    if len({frozenset(values) for values in indexed.values()}) != 1:
        raise ValueError(
            "evaluations are not episode-paired: "
            + repr({condition: len(values) for condition, values in indexed.items()})
        )
    return indexed


def _pairing_audit(
    rows_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    indexed = _paired_index(rows_by_condition)
    reference_name = next(iter(indexed))
    checked = 0
    for condition, rows in indexed.items():
        if condition == reference_name:
            continue
        for key, reference in indexed[reference_name].items():
            candidate = rows[key]
            for field in PAIR_FIELDS:
                if field not in reference or field not in candidate:
                    raise KeyError(f"paired evaluation is missing field: {field}")
                if json.dumps(reference[field], sort_keys=True) != json.dumps(
                    candidate[field], sort_keys=True
                ):
                    raise ValueError(
                        f"pairing mismatch for {condition}, key={key}, field={field}"
                    )
                checked += 1
    return {
        "status": "passed",
        "reference": reference_name,
        "condition_count": len(indexed),
        "episode_count_per_condition": len(indexed[reference_name]),
        "field_comparisons": checked,
        "fields": list(PAIR_FIELDS),
    }


def _means(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot summarize empty rows")
    result: dict[str, float | int] = {"episode_count": len(rows)}
    for metric in BEHAVIOR_METRICS:
        result[metric] = float(np.mean([float(row[metric]) for row in rows]))
    return result


def _visibility(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot summarize empty visibility rows")
    return {
        "episode_count": len(rows),
        "task_objects_visible": float(
            np.mean([bool(row["task_objects_visible"]) for row in rows])
        ),
        "task_objects_fully_visible": float(
            np.mean([bool(row["task_objects_fully_visible"]) for row in rows])
        ),
        "task_centers_in_frame": float(
            np.mean([bool(row["task_centers_in_frame"]) for row in rows])
        ),
        "source_touches_border": float(
            np.mean([bool(row["source_touches_border"]) for row in rows])
        ),
        "target_touches_border": float(
            np.mean([bool(row["target_touches_border"]) for row in rows])
        ),
        "source_visible_fraction": float(
            np.mean([float(row["source_visible_fraction"]) for row in rows])
        ),
        "target_visible_fraction": float(
            np.mean([float(row["target_visible_fraction"]) for row in rows])
        ),
    }


def _factorial_effects(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    contrasts = {
        "external_at_canonical_wrist": {
            "external_fla": 1.0,
            "dual_control_fla": -1.0,
        },
        "external_at_real_wrist": {
            "dual_fla": 1.0,
            "wrist_fla": -1.0,
        },
        "wrist_at_canonical_external": {
            "wrist_fla": 1.0,
            "dual_control_fla": -1.0,
        },
        "wrist_at_real_external": {
            "dual_fla": 1.0,
            "external_fla": -1.0,
        },
        "external_average_main_effect": {
            "external_fla": 0.5,
            "dual_control_fla": -0.5,
            "dual_fla": 0.5,
            "wrist_fla": -0.5,
        },
        "wrist_average_main_effect": {
            "wrist_fla": 0.5,
            "dual_control_fla": -0.5,
            "dual_fla": 0.5,
            "external_fla": -0.5,
        },
        "interaction": {
            "dual_fla": 1.0,
            "external_fla": -1.0,
            "wrist_fla": -1.0,
            "dual_control_fla": 1.0,
        },
    }
    return {
        name: {
            metric: grouped_linear_contrast(
                rows,
                coefficients=coefficients,
                metric=metric,
                bootstrap_resamples=bootstrap_resamples,
                seed=20260803 + effect_index * 10 + metric_index,
            )
            for metric_index, metric in enumerate(
                ("success", "transport_success", "progress")
            )
        }
        for effect_index, (name, coefficients) in enumerate(contrasts.items())
    }


def _training_summary(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"training metrics are empty: {path}")
    steps = [int(row["step"]) for row in rows]
    if steps != sorted(steps) or len(set(steps)) != len(steps):
        raise ValueError(f"training steps are not strictly ordered: {path}")
    losses = np.asarray([float(row["flow_matching_loss"]) for row in rows])
    tail = losses[-min(20, len(losses)) :]
    return {
        "path": str(path),
        "logged_point_count": len(rows),
        "final_step": steps[-1],
        "examples_seen": int(rows[-1]["examples_seen"]),
        "final_logged_flow_matching_loss": float(losses[-1]),
        "last_20_logs_mean_flow_matching_loss": float(tail.mean()),
        "last_20_logs_median_flow_matching_loss": float(np.median(tail)),
        "minimum_logged_flow_matching_loss": float(losses.min()),
    }


def summarize(
    *,
    evaluation_paths: Mapping[str, Path],
    intervention_paths: Mapping[str, Path],
    training_metric_paths: Mapping[str, Path],
    bootstrap_resamples: int,
    seed: int = 41,
    training_updates: int = 2000,
    execution_horizon: int = 3,
) -> dict[str, Any]:
    rows_by_method = {
        method: _load_rows(evaluation_paths[method], condition=method)
        for method in METHODS
    }
    pairing = _pairing_audit(rows_by_method)
    all_rows = [row for method in METHODS for row in rows_by_method[method]]
    reference_rows = rows_by_method["dual_rgb_fla"]

    present_poses = {str(row["camera_pose"]) for row in reference_rows}
    poses = [pose for pose in POSE_ORDER if pose in present_poses]
    poses += sorted(present_poses - set(poses))
    pose_diagnostics = {}
    for pose in poses:
        pose_diagnostics[pose] = {
            "visibility": _visibility(
                [row for row in reference_rows if row["camera_pose"] == pose]
            ),
            "methods": {
                method: _means(
                    [row for row in rows_by_method[method] if row["camera_pose"] == pose]
                )
                for method in METHODS
            },
        }

    interventions = {
        name: _load_rows(intervention_paths[name], condition=name)
        for name in WRIST_INTERVENTIONS
    }
    correct_rows = [{**row, "method": "correct"} for row in rows_by_method["dual_fla"]]
    causal_rows = {"correct": correct_rows, **interventions}
    causal_pairing = _pairing_audit(causal_rows)
    for pose in poses:
        pose_diagnostics[pose]["wrist_ray_intervention"] = {
            condition: _means(
                [row for row in rows if row["camera_pose"] == pose]
            )
            for condition, rows in causal_rows.items()
        }

    return {
        "schema_version": 1,
        "status": "complete",
        "study": "kyc_pi05_dual_camera_diagnostics",
        "seed": seed,
        "training_updates": training_updates,
        "execution_horizon": execution_horizon,
        "inference_unit": "canonical_state_index",
        "pairing_audit": pairing,
        "causal_pairing_audit": causal_pairing,
        "overall": {
            method: _means(rows_by_method[method]) for method in METHODS
        },
        "factorial_effects": _factorial_effects(
            all_rows,
            bootstrap_resamples=bootstrap_resamples,
        ),
        "pose_order": poses,
        "pose_diagnostics": pose_diagnostics,
        "edge_diagnostics": {
            edge: {
                method: _means(
                    [row for row in rows_by_method[method] if row["edge_id"] == edge]
                )
                for method in METHODS
            }
            for edge in sorted({str(row["edge_id"]) for row in reference_rows})
        },
        "training": {
            method: _training_summary(training_metric_paths[method])
            for method in METHODS
        },
    }


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate diagnostic details for the Pi0.5 dual-camera KYC screen"
    )
    parser.add_argument("--evaluation", action="append", required=True)
    parser.add_argument("--wrist-intervention", action="append", required=True)
    parser.add_argument("--training-metrics", action="append", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--training-updates", type=int, default=2000)
    parser.add_argument("--execution-horizon", type=int, default=3)
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
    metric_pairs = [
        _parse_named_path(value, allowed=METHODS, kind="training metrics")
        for value in args.training_metrics
    ]
    evaluation_paths = dict(evaluation_pairs)
    intervention_paths = dict(intervention_pairs)
    metric_paths = dict(metric_pairs)
    if len(evaluation_pairs) != len(METHODS) or set(evaluation_paths) != set(METHODS):
        raise ValueError("exactly one evaluation is required for every method")
    if (
        len(intervention_pairs) != len(WRIST_INTERVENTIONS)
        or set(intervention_paths) != set(WRIST_INTERVENTIONS)
    ):
        raise ValueError("exactly one initial and lagged intervention is required")
    if len(metric_pairs) != len(METHODS) or set(metric_paths) != set(METHODS):
        raise ValueError("exactly one training metric file is required for every method")
    if args.bootstrap_resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    if args.seed < 0 or args.training_updates <= 0 or args.execution_horizon <= 0:
        raise ValueError("seed must be non-negative and budgets must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostics: {args.output}")

    payload = summarize(
        evaluation_paths=evaluation_paths,
        intervention_paths=intervention_paths,
        training_metric_paths=metric_paths,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
        training_updates=args.training_updates,
        execution_horizon=args.execution_horizon,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "episode_count": payload["pairing_audit"]["episode_count_per_condition"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
