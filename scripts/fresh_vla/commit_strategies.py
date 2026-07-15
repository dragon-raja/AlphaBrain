from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from libero_snapshot_collector import gripper_transition_horizon

COMMIT_METHODS = (
    "fixed_k1",
    "fixed_k2",
    "fixed_k3",
    "oracle_branch_safe_commit",
    "oracle_feedback_reveal_commit",
    "gripper_commit",
    "random_matched_commit",
    "self_consistency_commit",
)
DEFAULT_MAX_COMMIT = 3


@dataclass(frozen=True)
class CommitDecision:
    length: int
    diagnostics: Mapping[str, Any]


def boundary_commit_length(current_step: int, boundary_step: int, *, max_commit: int = DEFAULT_MAX_COMMIT) -> int:
    """Stop exactly at a pre-registered boundary, then return to the default commit."""
    if current_step < 0 or boundary_step < 0:
        raise ValueError("steps must be non-negative")
    if max_commit <= 0:
        raise ValueError("max_commit must be positive")
    if current_step >= boundary_step:
        return max_commit
    return max(1, min(max_commit, boundary_step - current_step))


def gripper_commit_length(
    actions: np.ndarray,
    *,
    current_gripper_action: float,
    max_commit: int = DEFAULT_MAX_COMMIT,
) -> int:
    value = np.asarray(actions, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] == 0:
        raise ValueError("actions must be non-empty [H, D]")
    limited = value[:max_commit]
    current = np.asarray([[current_gripper_action]], dtype=np.float64)
    gripper_actions = np.concatenate((current, limited[:, -1, None]), axis=0)
    return min(max_commit, gripper_transition_horizon(gripper_actions) - 1)


def action_disagreement(sampled_chunks: np.ndarray) -> np.ndarray:
    chunks = np.asarray(sampled_chunks, dtype=np.float64)
    if chunks.ndim != 3 or chunks.shape[0] < 2 or chunks.shape[1] == 0 or chunks.shape[2] == 0:
        raise ValueError("sampled_chunks must have shape [N>=2, H, D]")
    return np.sqrt(np.mean(np.var(chunks, axis=0, ddof=1), axis=-1))


def self_consistency_commit_length(
    sampled_chunks: np.ndarray,
    *,
    max_commit: int = DEFAULT_MAX_COMMIT,
    threshold: float = 0.15,
) -> CommitDecision:
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    scores = action_disagreement(sampled_chunks)
    limit = min(max_commit, len(scores))
    length = 1
    for index in range(limit):
        if scores[index] > threshold:
            break
        length = index + 1
    return CommitDecision(
        length=length,
        diagnostics={
            "disagreement": scores[:limit].tolist(),
            "threshold": threshold,
            "sample_count": int(np.asarray(sampled_chunks).shape[0]),
        },
    )


def build_random_matched_boundaries(
    groups: Sequence[Mapping[str, Any]],
    *,
    source_label: str = "action_divergence_time",
    seed: int,
) -> dict[str, int]:
    """Permute oracle boundaries across snapshot groups while preserving their multiset."""
    ordered = sorted(groups, key=lambda row: str(row["pair_id"]))
    pair_ids = [str(group["pair_id"]) for group in ordered]
    boundaries = np.asarray([int(group[source_label]) for group in ordered], dtype=np.int64)
    if len(boundaries) > 1:
        rng = np.random.default_rng(seed)
        original = boundaries.copy()
        for _ in range(32):
            rng.shuffle(boundaries)
            if not np.array_equal(boundaries, original):
                break
    return {pair_id: int(boundaries[index]) for index, pair_id in enumerate(pair_ids)}


class CommitController:
    """Select execution length without changing or conditioning the policy output."""

    def __init__(
        self,
        method: str,
        groups: Sequence[Mapping[str, Any]],
        *,
        seed: int,
        max_commit: int = DEFAULT_MAX_COMMIT,
        self_consistency_samples: int = 8,
        self_consistency_threshold: float = 0.15,
        random_boundary_overrides: Mapping[str, int | None] | None = None,
    ):
        if method not in COMMIT_METHODS:
            raise ValueError(f"unknown commit method: {method}")
        if max_commit <= 0:
            raise ValueError("max_commit must be positive")
        if self_consistency_samples < 2:
            raise ValueError("self_consistency_samples must be at least 2")
        self.method = method
        self.max_commit = max_commit
        self.self_consistency_samples = self_consistency_samples
        self.self_consistency_threshold = self_consistency_threshold
        self.groups = {str(group["pair_id"]): group for group in groups}
        self.random_boundaries = (
            {str(pair_id): None if value is None else int(value) for pair_id, value in random_boundary_overrides.items()}
            if random_boundary_overrides is not None
            else build_random_matched_boundaries(groups, seed=seed)
        )
        if self.random_boundaries.keys() != self.groups.keys():
            raise ValueError("random boundary map must contain exactly the evaluated snapshot groups")

    @property
    def policy_samples_per_invocation(self) -> int:
        return self.self_consistency_samples if self.method == "self_consistency_commit" else 1

    def decide(
        self,
        pair_id: str,
        *,
        global_step: int,
        sampled_chunks: np.ndarray,
        current_gripper_action: float | None = None,
    ) -> CommitDecision:
        chunks = np.asarray(sampled_chunks, dtype=np.float64)
        if chunks.ndim != 3 or chunks.shape[0] != self.policy_samples_per_invocation:
            raise ValueError(f"expected {self.policy_samples_per_invocation} sampled chunks, got {chunks.shape}")
        group = self.groups[pair_id]
        if self.method.startswith("fixed_k"):
            length = int(self.method[-1])
            return CommitDecision(length, {"source": "fixed"})
        if self.method == "oracle_branch_safe_commit":
            boundary = int(group["action_divergence_time"])
            return CommitDecision(
                self.max_commit,
                {
                    "source": "action_divergence_time",
                    "boundary_step": boundary,
                    "alignment": "runtime_event_interrupt",
                },
            )
        if self.method == "oracle_feedback_reveal_commit":
            boundary = int(group["feedback_reveal_time"])
            return CommitDecision(
                self.max_commit,
                {
                    "source": "feedback_reveal_time",
                    "boundary_step": boundary,
                    "alignment": "runtime_event_interrupt",
                },
            )
        if self.method == "gripper_commit":
            if current_gripper_action is None:
                raise ValueError("gripper_commit requires the current gripper command")
            return CommitDecision(
                gripper_commit_length(
                    chunks[0],
                    current_gripper_action=current_gripper_action,
                    max_commit=self.max_commit,
                ),
                {"source": "current_command_to_candidate_chunk_gripper_transition"},
            )
        if self.method == "random_matched_commit":
            boundary = self.random_boundaries[pair_id]
            if boundary is None:
                return CommitDecision(
                    self.max_commit,
                    {"source": "permuted_full_h_runtime_event_time", "boundary_step": None},
                )
            return CommitDecision(
                boundary_commit_length(global_step, boundary, max_commit=self.max_commit),
                {"source": "permuted_full_h_runtime_event_time", "boundary_step": boundary},
            )
        if self.method == "self_consistency_commit":
            return self_consistency_commit_length(
                chunks,
                max_commit=self.max_commit,
                threshold=self.self_consistency_threshold,
            )
        raise AssertionError(self.method)
