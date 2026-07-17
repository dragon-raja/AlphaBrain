from __future__ import annotations

import hashlib
from typing import Mapping, Sequence

import numpy as np


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.square(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)).mean()))


def continuation_compatibility(
    candidate_actions: np.ndarray,
    candidate_effect: np.ndarray,
    correct_actions: np.ndarray,
    swapped_actions: np.ndarray,
    correct_effect: np.ndarray,
    swapped_effect: np.ndarray,
    *,
    action_margin: float,
    effect_margin: float,
) -> dict[str, float | bool]:
    correct_action_distance = rmse(candidate_actions, correct_actions)
    swapped_action_distance = rmse(candidate_actions, swapped_actions)
    correct_effect_distance = rmse(candidate_effect, correct_effect)
    swapped_effect_distance = rmse(candidate_effect, swapped_effect)
    action_compatible = correct_action_distance + action_margin < swapped_action_distance
    effect_compatible = correct_effect_distance + effect_margin < swapped_effect_distance
    return {
        "correct_action_rmse": correct_action_distance,
        "swapped_action_rmse": swapped_action_distance,
        "action_signed_margin": swapped_action_distance - correct_action_distance,
        "action_compatible": action_compatible,
        "correct_effect_rmse": correct_effect_distance,
        "swapped_effect_rmse": swapped_effect_distance,
        "effect_signed_margin": swapped_effect_distance - correct_effect_distance,
        "effect_compatible": effect_compatible,
        "joint_compatible": action_compatible and effect_compatible,
    }


def physical_compatibility(
    outcome: str,
    *,
    teacher_success: bool,
    grasp_trace: Sequence[bool],
    empty_lift: bool,
    recovery_action_seen: bool,
    initial_object_distance: float,
    final_object_distance: float,
    minimum_distance_reduction: float = 0.001,
) -> bool:
    if not teacher_success:
        return False
    if outcome == "attached":
        return bool(grasp_trace) and all(grasp_trace)
    if outcome == "slipped":
        distance_reduced = final_object_distance <= initial_object_distance - minimum_distance_reduction
        return not empty_lift and (recovery_action_seen or distance_reduced)
    raise ValueError(f"unknown outcome: {outcome}")


def recall_prefix(values: Sequence[bool], sizes: Sequence[int]) -> dict[str, bool]:
    array = np.asarray(values, dtype=bool)
    if not sizes or max(sizes) > len(array):
        raise ValueError("prefix sizes exceed candidate count")
    return {str(size): bool(np.any(array[:size])) for size in sizes}


def summarize_group_rows(
    rows: Sequence[Mapping[str, object]], sizes: Sequence[int]
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for outcome in ("attached", "slipped"):
        selected = [row for row in rows if row["outcome"] == outcome]
        if not selected:
            continue
        outcome_summary: dict[str, float] = {}
        for field in ("joint_recall", "action_recall", "effect_recall", "physical_recall"):
            for size in sizes:
                outcome_summary[f"{field}@{size}"] = float(
                    np.mean([bool(row[field][str(size)]) for row in selected])
                )
        summary[outcome] = outcome_summary
    return summary
