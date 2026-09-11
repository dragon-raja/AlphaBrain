from __future__ import annotations

from typing import Sequence

import numpy as np


STAGES = (
    "feedback_reveal",
    "failure_continuation",
    "recovery_start",
    "reapproach",
    "preclose",
    "post_regrasp",
    "final_failure",
)


def immediate_correct_mode(
    *,
    starts_grasped: bool,
    grasp_trace: Sequence[bool],
    actions: np.ndarray,
    failure_continuation_seen: bool,
    premature_commitment_seen: bool,
    recovery_action_seen: bool,
) -> bool:
    actions = np.asarray(actions, dtype=np.float64)
    if starts_grasped:
        opens_gripper = bool(np.any(actions[:, -1] < -0.2))
        return bool(grasp_trace) and all(grasp_trace) and not opens_gripper
    acquired_grasp = any(grasp_trace)
    return bool(
        not failure_continuation_seen
        and not premature_commitment_seen
        and (recovery_action_seen or acquired_grasp)
    )


def recall_at_n(values: Sequence[bool], sizes: Sequence[int] = (1, 4, 8, 16)) -> dict[str, bool]:
    array = np.asarray(values, dtype=bool)
    return {str(size): bool(np.any(array[:size])) for size in sizes}


def classify_boundary_stage(
    *,
    replan_index: int,
    grasped: bool,
    previous_failure_continuation: bool,
    recovery_started: bool,
    eef_object_distance: float,
    initial_eef_object_distance: float,
    candidate0_closes: bool,
) -> str:
    if replan_index == 0:
        return "feedback_reveal"
    if grasped:
        return "post_regrasp"
    if previous_failure_continuation:
        return "failure_continuation"
    if candidate0_closes and eef_object_distance <= 0.04:
        return "preclose"
    if recovery_started and eef_object_distance < initial_eef_object_distance - 0.005:
        return "reapproach"
    return "recovery_start"
