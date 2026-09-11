from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

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
TARGET_POLICIES = (41, 42, 43)
SELECTION_SALT = "basin-vla-relativity-v1"


def parse_cache_name(path: Path) -> tuple[str, str, int]:
    parts = path.stem.split("--")
    if len(parts) != 3 or not parts[2].startswith("r"):
        raise ValueError(f"unexpected on-policy cache name: {path.name}")
    pair_id, stage, replan = parts
    if stage not in STAGES:
        raise ValueError(f"unknown on-policy stage in {path.name}: {stage}")
    return pair_id, stage, int(replan[1:])


def select_cache_files(paths: Sequence[Path], per_stage: int = 3) -> list[Path]:
    if per_stage < 1:
        raise ValueError("per_stage must be positive")
    by_stage: dict[str, list[Path]] = defaultdict(list)
    seen = set()
    for path in paths:
        pair_id, stage, replan = parse_cache_name(path)
        identity = (pair_id, stage, replan)
        if identity in seen:
            raise ValueError(f"duplicate formal cache state: {identity}")
        seen.add(identity)
        by_stage[stage].append(path)

    selected = []
    for stage in STAGES:
        ranked = sorted(
            by_stage[stage],
            key=lambda path: hashlib.sha256(
                f"{SELECTION_SALT}:{path.name}".encode()
            ).hexdigest(),
        )
        selected.extend(ranked[:per_stage])
    return selected


def lexicographic_percentiles(keys: Sequence[Sequence[float]]) -> np.ndarray:
    tuples = [tuple(float(value) for value in row) for row in keys]
    if not tuples:
        raise ValueError("continuation keys must be non-empty")
    unique = sorted(set(tuples))
    if len(unique) == 1:
        return np.full(len(tuples), 0.5, dtype=np.float64)
    ranks = {value: index / (len(unique) - 1) for index, value in enumerate(unique)}
    return np.asarray([ranks[value] for value in tuples], dtype=np.float64)


def pairwise_policy_metrics(
    first: Sequence[float],
    second: Sequence[float],
) -> dict[str, float | int]:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        raise ValueError("policy ranks must be equal one-dimensional arrays with at least two candidates")
    comparable = 0
    flips = 0
    total = 0
    for first_index in range(len(left)):
        for second_index in range(first_index + 1, len(left)):
            total += 1
            left_delta = left[first_index] - left[second_index]
            right_delta = right[first_index] - right[second_index]
            if left_delta == 0.0 or right_delta == 0.0:
                continue
            comparable += 1
            flips += int(np.sign(left_delta) != np.sign(right_delta))
    first_top = set(np.flatnonzero(left == np.max(left)).tolist())
    second_top = set(np.flatnonzero(right == np.max(right)).tolist())
    union = first_top | second_top
    return {
        "candidate_pair_count": total,
        "comparable_pair_count": comparable,
        "comparable_fraction": comparable / total,
        "preference_flip_rate": float(flips / comparable) if comparable else 0.0,
        "pair_agreement": float(1.0 - flips / comparable) if comparable else 1.0,
        "top_tier_jaccard": len(first_top & second_top) / len(union),
    }


def leave_one_policy_out_choice(
    ranks_by_policy: Mapping[int, Sequence[float]],
    target_policy: int,
) -> dict[str, float | int]:
    if target_policy not in ranks_by_policy:
        raise ValueError(f"missing target policy {target_policy}")
    others = [
        np.asarray(values, dtype=np.float64)
        for policy, values in sorted(ranks_by_policy.items())
        if policy != target_policy
    ]
    if len(others) < 2:
        raise ValueError("leave-one-policy-out selection requires at least two other policies")
    target = np.asarray(ranks_by_policy[target_policy], dtype=np.float64)
    if any(value.shape != target.shape for value in others):
        raise ValueError("policy rank arrays have different candidate counts")
    average = np.mean(others, axis=0)
    selected = int(np.argmax(average))
    oracle_utility = float(np.max(target))
    selected_utility = float(target[selected])
    return {
        "selected_index": selected,
        "target_utility": selected_utility,
        "target_oracle_utility": oracle_utility,
        "oracle_minus_loo": oracle_utility - selected_utility,
        "candidate0_utility": float(target[0]),
        "oracle_minus_candidate0": oracle_utility - float(target[0]),
    }


def bootstrap_mean(values: Sequence[float], *, samples: int = 20_000, seed: int = 0) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("bootstrap requires non-empty one-dimensional values")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    chunk = max(1, min(samples, 1_000_000 // len(array)))
    for start in range(0, samples, chunk):
        count = min(chunk, samples - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        estimates[start : start + count] = array[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "mean": float(array.mean()),
        "bootstrap_95_low": float(low),
        "bootstrap_95_high": float(high),
        "count": int(len(array)),
    }
