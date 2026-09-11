from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np


POLICY_INPUT_FIELDS = frozenset({"observation", "robot_state", "language_instruction"})
FUTURE_ONLY_FIELDS = frozenset(
    {
        "pair_id",
        "branch_id",
        "branch_outcome",
        "action_chunk",
        "event_time",
        "feedback_reveal_time",
        "action_divergence_time",
        "gripper_transition_horizon",
        "oracle_feedback_horizon",
        "per_step_branch_divergence",
        "is_deterministic_control",
        "future_observation",
        "future_contact",
        "attachment_result",
    }
)


@dataclass(frozen=True)
class CounterfactualRecord:
    pair_id: str
    branch_id: str
    branch_outcome: str
    observation: Any
    robot_state: list[float]
    language_instruction: str
    action_chunk: list[list[float]]
    event_time: int
    feedback_reveal_time: int
    action_divergence_time: int
    gripper_transition_horizon: int
    oracle_feedback_horizon: int
    per_step_branch_divergence: list[float]
    is_deterministic_control: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DivergenceEstimate:
    within_branch_threshold: float
    action_divergence_time: int
    oracle_feedback_horizon: int
    per_step_branch_divergence: list[float]


def _as_rollouts(value: np.ndarray | Sequence[Any]) -> np.ndarray:
    rollouts = np.asarray(value, dtype=np.float64)
    if rollouts.ndim != 3:
        raise ValueError(f"expected repeated actions shaped [R, H, D], got {rollouts.shape}")
    if not np.isfinite(rollouts).all():
        raise ValueError("action rollouts must contain only finite values")
    return rollouts


def normalized_action_distances(first: np.ndarray, second: np.ndarray, scale: np.ndarray | None = None) -> np.ndarray:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError(f"expected matching [H, D] actions, got {first.shape} and {second.shape}")
    if scale is None:
        scale = np.ones(first.shape[-1], dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if scale.shape != (first.shape[-1],) or np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError(f"action scale must be finite, positive, and shaped [{first.shape[-1]}]")
    return np.sqrt(np.mean(np.square((first - second) / scale[None, :]), axis=-1))


def estimate_within_branch_threshold(
    branch_rollouts: Mapping[str, np.ndarray | Sequence[Any]],
    *,
    quantile: float = 0.95,
    action_scale: np.ndarray | None = None,
    minimum_threshold: float = 1e-6,
) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")

    distances = []
    for rollouts_value in branch_rollouts.values():
        rollouts = _as_rollouts(rollouts_value)
        for left, right in combinations(range(rollouts.shape[0]), 2):
            distances.extend(normalized_action_distances(rollouts[left], rollouts[right], action_scale))
    if not distances:
        raise ValueError("at least one branch must contain two repeated rollouts")
    return max(float(np.quantile(np.asarray(distances), quantile)), minimum_threshold)


def first_persistent_divergence(per_step_distance: np.ndarray, threshold: float, persistence: int = 2) -> int:
    distances = np.asarray(per_step_distance, dtype=np.float64)
    if distances.ndim != 1 or not np.isfinite(distances).all():
        raise ValueError("per_step_distance must be a finite one-dimensional array")
    if persistence <= 0:
        raise ValueError(f"persistence must be positive, got {persistence}")
    above = distances > threshold
    for index in range(max(0, len(distances) - persistence + 1)):
        if bool(np.all(above[index : index + persistence])):
            return index
    return len(distances)


def estimate_branch_divergence(
    branch_rollouts: Mapping[str, np.ndarray | Sequence[Any]],
    *,
    quantile: float = 0.95,
    threshold_multiplier: float = 1.0,
    persistence: int = 2,
    action_scale: np.ndarray | None = None,
) -> DivergenceEstimate:
    if len(branch_rollouts) < 2:
        raise ValueError("counterfactual divergence requires at least two branches")
    rollouts = {name: _as_rollouts(value) for name, value in branch_rollouts.items()}
    shapes = {value.shape[1:] for value in rollouts.values()}
    if len(shapes) != 1:
        raise ValueError(f"all branches must share [H, D], got {sorted(shapes)}")
    if threshold_multiplier <= 0:
        raise ValueError(f"threshold_multiplier must be positive, got {threshold_multiplier}")

    threshold = estimate_within_branch_threshold(
        rollouts,
        quantile=quantile,
        action_scale=action_scale,
    ) * threshold_multiplier
    branch_means = {name: value.mean(axis=0) for name, value in rollouts.items()}
    between = [
        normalized_action_distances(branch_means[left], branch_means[right], action_scale)
        for left, right in combinations(branch_means, 2)
    ]
    per_step = np.max(np.stack(between), axis=0)
    divergence = first_persistent_divergence(per_step, threshold, persistence)
    return DivergenceEstimate(
        within_branch_threshold=threshold,
        action_divergence_time=divergence,
        oracle_feedback_horizon=divergence,
        per_step_branch_divergence=per_step.tolist(),
    )


def threshold_sensitivity(
    branch_rollouts: Mapping[str, np.ndarray | Sequence[Any]],
    multipliers: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
    **kwargs: Any,
) -> dict[str, DivergenceEstimate]:
    return {
        str(multiplier): estimate_branch_divergence(
            branch_rollouts,
            threshold_multiplier=float(multiplier),
            **kwargs,
        )
        for multiplier in multipliers
    }


def validate_record(record: CounterfactualRecord) -> None:
    actions = np.asarray(record.action_chunk, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] == 0:
        raise ValueError(f"action_chunk must be non-empty [H, D], got {actions.shape}")
    horizon = actions.shape[0]
    for name in (
        "event_time",
        "feedback_reveal_time",
        "action_divergence_time",
        "gripper_transition_horizon",
        "oracle_feedback_horizon",
    ):
        value = getattr(record, name)
        if not 0 <= value <= horizon:
            raise ValueError(f"{name} must be in [0, {horizon}], got {value}")
    if record.oracle_feedback_horizon != record.action_divergence_time:
        raise ValueError("oracle_feedback_horizon must equal action_divergence_time")
    if len(record.per_step_branch_divergence) != horizon:
        raise ValueError("per_step_branch_divergence must contain one value per action step")


def build_policy_inputs(record: CounterfactualRecord) -> dict[str, Any]:
    inputs = {
        "observation": record.observation,
        "robot_state": record.robot_state,
        "language_instruction": record.language_instruction,
    }
    assert_no_future_information(inputs)
    return inputs


def assert_no_future_information(policy_inputs: Mapping[str, Any]) -> None:
    keys = set(policy_inputs)
    leaked = sorted(keys & FUTURE_ONLY_FIELDS)
    unknown = sorted(keys - POLICY_INPUT_FIELDS)
    if leaked or unknown:
        details = []
        if leaked:
            details.append(f"future-only fields: {leaked}")
        if unknown:
            details.append(f"non-whitelisted fields: {unknown}")
        raise ValueError("policy input leakage check failed: " + "; ".join(details))
