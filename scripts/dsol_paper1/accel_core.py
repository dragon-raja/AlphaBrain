from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


ACCEL_PREFIXES = tuple(range(2, 11))
PRIMARY_ACCEL_PREFIX = 3
REQUIRED_RELATION_SETS = frozenset(
    {"canonical", "train", "strong_info", "reveal", "oracle"}
)


def shared_flow_noise(
    *,
    seed: int,
    candidate_count: int,
    action_horizon: int,
    action_dim: int,
) -> np.ndarray:
    """Return one float32 Gaussian x0 repeated exactly across candidates."""

    if candidate_count < 1 or action_horizon < 1 or action_dim < 1:
        raise ValueError(
            "candidate_count, action_horizon, and action_dim must be positive"
        )
    generator = np.random.default_rng(int(seed))
    one = generator.standard_normal(
        (1, action_horizon, action_dim), dtype=np.float32
    )
    return np.repeat(one, candidate_count, axis=0)


def audit_shared_flow_noise(initial_noise: np.ndarray) -> dict[str, Any]:
    noise = np.asarray(initial_noise)
    if noise.ndim < 2 or noise.shape[0] < 1:
        raise ValueError("initial_noise must have candidate as its first dimension")
    if not np.all(np.isfinite(noise)):
        raise ValueError("initial_noise contains non-finite values")
    first = noise[0:1]
    max_abs_difference = float(np.max(np.abs(noise - first)))
    return {
        "candidate_count": int(noise.shape[0]),
        "exactly_shared": bool(
            np.array_equal(noise, np.repeat(first, len(noise), axis=0))
        ),
        "max_abs_difference": max_abs_difference,
    }


def compute_accel_scores(
    velocity_trace: np.ndarray,
    *,
    prefixes: Sequence[int] = ACCEL_PREFIXES,
    epsilon: float = 1e-12,
) -> dict[int, dict[str, np.ndarray]]:
    """Compute prefix acceleration scores for a candidate-batched velocity trace.

    The input shape is ``[candidate, denoise_step, ...]``. For prefix ``p``:

        accel_p = p * sum_{t=1}^{p-1} ||v_t - v_{t-1}||_2
                    / sum_{t=0}^{p-1} ||v_t||_2

    A zero denominator is marked degenerate and assigned a finite score of zero;
    ranking code excludes degenerate candidates rather than treating zero as best.
    """

    trace = np.asarray(velocity_trace, dtype=np.float64)
    if trace.ndim < 3:
        raise ValueError("velocity_trace must have shape [candidate, step, ...]")
    if trace.shape[0] < 1 or trace.shape[1] < 2:
        raise ValueError("velocity_trace requires at least one candidate and two steps")
    if not np.all(np.isfinite(trace)):
        raise ValueError("velocity_trace contains non-finite values")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    requested = tuple(int(prefix) for prefix in prefixes)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("prefixes must be non-empty and unique")
    if min(requested) < 2 or max(requested) > trace.shape[1]:
        raise ValueError(
            f"prefixes must be in [2, {trace.shape[1]}], got {requested}"
        )

    flattened = trace.reshape(trace.shape[0], trace.shape[1], -1)
    velocity_norm = np.linalg.norm(flattened, axis=2)
    delta_norm = np.linalg.norm(np.diff(flattened, axis=1), axis=2)
    result: dict[int, dict[str, np.ndarray]] = {}
    for prefix in requested:
        denominator = velocity_norm[:, :prefix].sum(axis=1)
        numerator = float(prefix) * delta_norm[:, : prefix - 1].sum(axis=1)
        degenerate = denominator <= epsilon
        score = np.zeros_like(denominator)
        np.divide(numerator, denominator, out=score, where=~degenerate)
        result[prefix] = {
            "score": score,
            "degenerate": degenerate,
            "numerator": numerator,
            "denominator": denominator,
        }
    return result


def rank_accel_candidates(
    candidate_ids: Sequence[str],
    velocity_trace: np.ndarray,
    *,
    initial_noise: np.ndarray | None = None,
    primary_prefix: int = PRIMARY_ACCEL_PREFIX,
    prefixes: Sequence[int] = ACCEL_PREFIXES,
) -> dict[str, Any]:
    ids = [str(candidate_id) for candidate_id in candidate_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_ids must be unique")
    trace = np.asarray(velocity_trace)
    if trace.shape[0] != len(ids):
        raise ValueError("candidate_ids and velocity_trace candidate counts differ")
    if primary_prefix not in prefixes:
        raise ValueError("primary_prefix must be included in prefixes")

    metrics = compute_accel_scores(trace, prefixes=prefixes)
    rows = []
    for index, candidate_id in enumerate(ids):
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "candidate_index": index,
        }
        for prefix in prefixes:
            values = metrics[int(prefix)]
            row[f"accel_{prefix}"] = float(values["score"][index])
            row[f"accel_{prefix}_degenerate"] = bool(
                values["degenerate"][index]
            )
            row[f"accel_{prefix}_denominator"] = float(
                values["denominator"][index]
            )
            row[f"accel_{prefix}_numerator"] = float(
                values["numerator"][index]
            )
        rows.append(row)

    primary_key = f"accel_{primary_prefix}"
    degenerate_key = f"accel_{primary_prefix}_degenerate"
    valid = [row for row in rows if not row[degenerate_key]]
    valid.sort(key=lambda row: (row[primary_key], row["candidate_id"]))
    invalid = [row for row in rows if row[degenerate_key]]
    invalid.sort(key=lambda row: row["candidate_id"])
    ranking = valid + invalid
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
        row["rank_valid"] = not row[degenerate_key]

    payload: dict[str, Any] = {
        "schema": "dsol_accel_fixed_state_ranking_v1",
        "primary_metric": primary_key,
        "score_direction": "lower_is_better",
        "trace_coordinate_system": "normalized_action",
        "candidate_count": len(ids),
        "valid_candidate_count": len(valid),
        "status": "OK" if valid else "NO_VALID_CANDIDATE",
        "selected_candidate_id": valid[0]["candidate_id"] if valid else None,
        "ranking": ranking,
    }
    if initial_noise is not None:
        noise_audit = audit_shared_flow_noise(initial_noise)
        payload["shared_flow_noise_audit"] = noise_audit
        if not noise_audit["exactly_shared"]:
            payload["status"] = "INVALID_UNSHARED_FLOW_NOISE"
            payload["selected_candidate_id"] = None
    return payload


def _wrapped_angle_delta(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def pose_distance(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, float] | None:
    spherical_keys = {"azimuth_deg", "elevation_deg", "radius_scale"}
    if spherical_keys.issubset(left) and spherical_keys.issubset(right):
        left_radius = float(left["radius_scale"])
        right_radius = float(right["radius_scale"])
        if left_radius <= 0 or right_radius <= 0:
            raise ValueError("radius_scale must be positive")
        azimuth = _wrapped_angle_delta(
            float(left["azimuth_deg"]), float(right["azimuth_deg"])
        )
        elevation = abs(
            float(left["elevation_deg"]) - float(right["elevation_deg"])
        )
        log_radius = abs(math.log(left_radius / right_radius))
        normalized = math.sqrt(
            (azimuth / 180.0) ** 2
            + (elevation / 90.0) ** 2
            + log_radius**2
        )
        return {
            "normalized_pose_distance": normalized,
            "azimuth_delta_deg": azimuth,
            "elevation_delta_deg": elevation,
            "log_radius_delta": log_radius,
        }

    matrix_keys = {"camera_position", "camera_rotation_matrix"}
    if matrix_keys.issubset(left) and matrix_keys.issubset(right):
        left_position = np.asarray(left["camera_position"], dtype=np.float64)
        right_position = np.asarray(right["camera_position"], dtype=np.float64)
        left_rotation = np.asarray(left["camera_rotation_matrix"], dtype=np.float64)
        right_rotation = np.asarray(right["camera_rotation_matrix"], dtype=np.float64)
        if left_position.shape != (3,) or right_position.shape != (3,):
            raise ValueError("camera_position must be length 3")
        if left_rotation.shape != (3, 3) or right_rotation.shape != (3, 3):
            raise ValueError("camera_rotation_matrix must be 3x3")
        relative = left_rotation.T @ right_rotation
        cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        rotation_deg = math.degrees(math.acos(cosine))
        translation = float(np.linalg.norm(left_position - right_position))
        return {
            "normalized_pose_distance": math.sqrt(
                translation**2 + (rotation_deg / 180.0) ** 2
            ),
            "translation_distance": translation,
            "rotation_distance_deg": rotation_deg,
        }
    return None


def analyze_selected_relations(
    ranking: Mapping[str, Any],
    candidate_metadata: Sequence[Mapping[str, Any]],
    references: Mapping[str, str | Sequence[str]],
) -> dict[str, Any]:
    missing_relations = sorted(REQUIRED_RELATION_SETS - set(references))
    if missing_relations:
        raise ValueError(
            f"references are missing required relations: {missing_relations}"
        )
    metadata = {str(row["candidate_id"]): dict(row) for row in candidate_metadata}
    if len(metadata) != len(candidate_metadata):
        raise ValueError("candidate metadata IDs must be unique")
    selected_id = ranking.get("selected_candidate_id")
    ranked_rows = {
        str(row["candidate_id"]): row for row in ranking.get("ranking", [])
    }
    if selected_id is not None and str(selected_id) not in metadata:
        raise ValueError("selected candidate is absent from candidate metadata")

    relation_rows: dict[str, Any] = {}
    for relation, raw_ids in references.items():
        ids = (
            [str(raw_ids)]
            if isinstance(raw_ids, str)
            else [str(value) for value in raw_ids]
        )
        if not ids:
            raise ValueError(f"reference relation {relation!r} is empty")
        missing = sorted(set(ids) - set(metadata))
        available_ranked = [
            ranked_rows[value] for value in ids if value in ranked_rows
        ]
        best_ranked = (
            min(available_ranked, key=lambda row: int(row["rank"]))
            if available_ranked
            else None
        )
        row: dict[str, Any] = {
            "reference_candidate_ids": ids,
            "missing_candidate_ids": missing,
            "selected_exact_match": (
                selected_id is not None and str(selected_id) in ids
            ),
            "best_reference_accel_candidate_id": (
                None if best_ranked is None else best_ranked["candidate_id"]
            ),
            "best_reference_accel_rank": (
                None if best_ranked is None else int(best_ranked["rank"])
            ),
        }
        if selected_id is not None:
            distances = []
            for reference_id in ids:
                if reference_id not in metadata:
                    continue
                distance = pose_distance(
                    metadata[str(selected_id)], metadata[reference_id]
                )
                if distance is not None:
                    distances.append(
                        (
                            distance["normalized_pose_distance"],
                            reference_id,
                            distance,
                        )
                    )
            if distances:
                _, nearest_id, nearest_distance = min(
                    distances, key=lambda item: (item[0], item[1])
                )
                row["nearest_reference_candidate_id"] = nearest_id
                row["nearest_reference_pose_distance"] = nearest_distance
        relation_rows[str(relation)] = row

    return {
        "schema": "dsol_accel_relation_analysis_v1",
        "ranking_status": ranking.get("status"),
        "selected_candidate_id": selected_id,
        "relations": relation_rows,
    }
