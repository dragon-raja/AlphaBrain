from __future__ import annotations

import hashlib
from typing import Iterable, Mapping, Sequence

import numpy as np


MILESTONE_NAMES = ("stable_grasp", "lift", "transport", "success")
PROFILE_NAMES = (*MILESTONE_NAMES, "no_regress", "progress_auc")
UTILITY_BASE = 8.0
UTILITY_MAX = sum(UTILITY_BASE**power for power in range(6))
DEPLOYABLE_ARRAY_KEYS = {
    "agentview_image",
    "wrist_image",
    "robot_state",
    "vla_feature",
    "candidates",
    "candidate_seeds",
}
PRIVILEGED_TOKENS = (
    "sim_state",
    "object_pose",
    "object_position",
    "contact",
    "grasped",
    "branch_outcome",
    "future_image",
    "continuation",
)


def frozen_source_split(
    source_ids: Iterable[int],
    *,
    holdout_count: int = 6,
    salt: str = "ccv-vla-gate0-v1",
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    unique = sorted({int(value) for value in source_ids})
    if not 0 < holdout_count < len(unique):
        raise ValueError("holdout_count must leave non-empty fit and holdout sets")

    def digest(source_id: int) -> bytes:
        return hashlib.sha256(f"{salt}::{source_id}".encode("ascii")).digest()

    ranked = sorted(unique, key=lambda source_id: (digest(source_id), source_id))
    holdout = tuple(sorted(ranked[:holdout_count]))
    fit = tuple(sorted(set(unique) - set(holdout)))
    return fit, holdout


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def close_milestones(values: Sequence[bool | int | float]) -> np.ndarray:
    """Return [grasp, lift, transport, success] with cumulative prerequisite closure."""
    array = np.asarray(values, dtype=bool)
    if array.shape != (4,):
        raise ValueError(f"expected four milestones, got {array.shape}")
    return np.maximum.accumulate(array[::-1])[::-1].astype(np.float32)


def summary_signature(summary: Mapping[str, object]) -> np.ndarray:
    milestones = close_milestones(
        [
            bool(summary.get("regrasp_reached", summary.get("stable_grasp_at_end", False))),
            bool(summary["lift_reached"]),
            bool(summary["transport_reached"]),
            bool(summary["success"]),
        ]
    )
    progress = float(summary["progress_auc"])
    if not np.isfinite(progress):
        raise ValueError("progress_auc must be finite")
    return np.concatenate(
        [milestones, np.asarray([not bool(summary["regress"]), progress], dtype=np.float32)]
    )


def profile_from_signatures(signatures: np.ndarray) -> np.ndarray:
    array = np.asarray(signatures, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(PROFILE_NAMES):
        raise ValueError(f"expected [R, {len(PROFILE_NAMES)}] signatures, got {array.shape}")
    profile = array.mean(axis=0)
    if np.any(np.diff(profile[:4]) > 1e-8):
        raise ValueError(f"milestone survival profile is not monotone: {profile[:4]}")
    return profile.astype(np.float32)


def viability_key(profile: Sequence[float]) -> tuple[float, ...]:
    value = np.asarray(profile, dtype=np.float64)
    if value.shape != (len(PROFILE_NAMES),):
        raise ValueError(f"expected {len(PROFILE_NAMES)} profile entries, got {value.shape}")
    grasp, lift, transport, success, no_regress, progress = value
    return (float(success), float(transport), float(lift), float(grasp), float(no_regress), float(progress))


def best_candidate_index(profiles: np.ndarray) -> int:
    array = np.asarray(profiles, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(PROFILE_NAMES):
        raise ValueError(f"expected [N, {len(PROFILE_NAMES)}] profiles, got {array.shape}")
    return max(range(len(array)), key=lambda index: viability_key(array[index]))


def scalar_viability_utility(profile: Sequence[float]) -> float:
    grasp, lift, transport, success, no_regress, progress = np.asarray(
        profile, dtype=np.float64
    )
    raw = (
        success * UTILITY_BASE**5
        + transport * UTILITY_BASE**4
        + lift * UTILITY_BASE**3
        + grasp * UTILITY_BASE**2
        + no_regress * UTILITY_BASE
        + progress
    )
    return float(raw / UTILITY_MAX)


def assert_deployable_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    unexpected = set(arrays) - DEPLOYABLE_ARRAY_KEYS
    if unexpected:
        raise ValueError(f"unexpected deployable arrays: {sorted(unexpected)}")
    missing = DEPLOYABLE_ARRAY_KEYS - set(arrays)
    if missing:
        raise ValueError(f"missing deployable arrays: {sorted(missing)}")
    lowered = [key.lower() for key in arrays]
    leaked = sorted(
        key for key in lowered if any(token in key for token in PRIVILEGED_TOKENS)
    )
    if leaked:
        raise ValueError(f"privileged arrays leaked into deployable view: {leaked}")


def milestone_violation_count(raw_signatures: np.ndarray) -> int:
    array = np.asarray(raw_signatures, dtype=bool)
    if array.shape[-1] != 4:
        raise ValueError("raw signatures must end in four milestone entries")
    return int(np.count_nonzero(np.any(array[..., 1:] > array[..., :-1], axis=-1)))


def coupled_policy_continuations(
    branch_pool,
    policy,
    *,
    endpoint_count: int,
    seed: int,
    pair_id: str,
    state_id: str,
    execution_horizon: int,
    lookahead_actions: int,
    repeats: int,
) -> tuple[list[list[Mapping[str, object]]], dict[str, int]]:
    """Roll one shared flow-noise schedule from every candidate endpoint."""
    if endpoint_count < 1 or repeats < 1:
        raise ValueError("endpoint_count and repeats must be positive")
    if execution_horizon < 1 or lookahead_actions < 1:
        raise ValueError("execution and lookahead horizons must be positive")
    all_results: list[list[Mapping[str, object]]] = [[] for _ in range(endpoint_count)]
    batch_calls = 0
    simulator_actions = 0
    continuation_replans = int(np.ceil(lookahead_actions / execution_horizon))
    for repeat in range(repeats):
        actions_done = [0] * endpoint_count
        observations = branch_pool.reset_continuation(endpoint_count)
        for continuation_replan in range(continuation_replans):
            chunks, _ = policy.predict_observation_batch_coupled(
                observations,
                seed=stable_seed(
                    "ccv-depth-coupled",
                    seed,
                    pair_id,
                    state_id,
                    repeat,
                    continuation_replan,
                ),
            )
            batch_calls += 1
            advanced = branch_pool.advance(
                chunks,
                actions_done,
                execution_horizon,
                lookahead_actions,
            )
            observations = [row["observation"] for row in advanced]
            for index, row in enumerate(advanced):
                actions_done[index] += int(row["executed"])
                simulator_actions += int(row["executed"])
        for index, summary in enumerate(branch_pool.summaries(endpoint_count)):
            all_results[index].append(summary)
    return all_results, {
        "policy_batch_calls": batch_calls,
        "simulator_actions": simulator_actions,
    }
