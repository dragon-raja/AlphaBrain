from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np


OUTCOMES = ("attached", "detached")
LABELS = {"attached": 1.0, "detached": -1.0}
PROBE_NAMES = ("hold_closed", "micro_lift", "full_lift", "micro_lateral", "release")
NON_RELEASE_PROBES = tuple(name for name in PROBE_NAMES if name != "release")
MODALITIES = ("vision", "robot_state", "vision_state")


def probe_actions(name: str, steps: int = 4) -> np.ndarray:
    if steps != 4:
        raise ValueError("Gate 0 probes are frozen at four control steps")
    actions = np.zeros((steps, 7), dtype=np.float32)
    if name == "hold_closed":
        actions[:, -1] = 1.0
    elif name == "micro_lift":
        actions[:, 2] = 0.25
        actions[:, -1] = 1.0
    elif name == "full_lift":
        actions[:, 2] = 1.0
        actions[:, -1] = 1.0
    elif name == "micro_lateral":
        actions[:, 0] = np.asarray([0.25, -0.25, 0.25, -0.25], dtype=np.float32)
        actions[:, -1] = 1.0
    elif name == "release":
        actions[:, -1] = -1.0
    else:
        raise ValueError(f"unknown probe: {name}")
    return actions


def detach_offsets(direction_xy: Sequence[float]) -> tuple[np.ndarray, ...]:
    direction = np.asarray(direction_xy, dtype=np.float64)
    if direction.shape != (2,):
        raise ValueError("detach direction must be an XY vector")
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        direction = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        direction = direction / norm
    return tuple(
        np.asarray([direction[0] * millimeters / 1000.0, direction[1] * millimeters / 1000.0, 0.0])
        for millimeters in np.arange(0.25, 3.0 + 0.125, 0.25)
    )


def pixel_mae(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=np.float32)
    second = np.asarray(right, dtype=np.float32)
    if first.shape != second.shape:
        raise ValueError(f"pixel arrays have different shapes: {first.shape} versus {second.shape}")
    return float(np.mean(np.abs(first - second)))


def block_average(image: np.ndarray, size: int = 16) -> np.ndarray:
    value = np.asarray(image, dtype=np.float32)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image, got {value.shape}")
    if size <= 0 or value.shape[0] % size or value.shape[1] % size:
        raise ValueError(f"image shape {value.shape[:2]} is not divisible by output size {size}")
    row_block = value.shape[0] // size
    column_block = value.shape[1] // size
    pooled = value.reshape(size, row_block, size, column_block, 3).mean(axis=(1, 3))
    return (pooled / 255.0).reshape(-1)


def feature_vector(
    agentview: np.ndarray,
    wrist: np.ndarray,
    robot_state: np.ndarray,
    modality: str,
    image_size: int = 16,
) -> np.ndarray:
    state = np.asarray(robot_state, dtype=np.float32).reshape(-1)
    if modality == "robot_state":
        return state
    vision = np.concatenate((block_average(agentview, image_size), block_average(wrist, image_size)))
    if modality == "vision":
        return vision
    if modality == "vision_state":
        return np.concatenate((vision, state))
    raise ValueError(f"unknown modality: {modality}")


def ridge_scores(
    train: np.ndarray,
    labels: np.ndarray,
    query: np.ndarray,
    regularization: float = 1.0,
) -> np.ndarray:
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    train_value = np.asarray(train, dtype=np.float64)
    query_value = np.asarray(query, dtype=np.float64)
    labels_value = np.asarray(labels, dtype=np.float64)
    if train_value.ndim != 2 or query_value.ndim != 2:
        raise ValueError("ridge features must be matrices")
    if train_value.shape[1] != query_value.shape[1] or len(train_value) != len(labels_value):
        raise ValueError("ridge feature and label shapes are inconsistent")
    mean = train_value.mean(axis=0)
    scale = train_value.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train_value = (train_value - mean) / scale
    query_value = (query_value - mean) / scale
    train_value = np.concatenate((train_value, np.ones((len(train_value), 1))), axis=1)
    query_value = np.concatenate((query_value, np.ones((len(query_value), 1))), axis=1)
    kernel = train_value @ train_value.T
    alpha = np.linalg.solve(kernel + regularization * np.eye(len(kernel)), labels_value)
    return query_value @ train_value.T @ alpha


def bootstrap_mean(
    values: Sequence[float],
    *,
    samples: int = 20_000,
    seed: int = 0,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("bootstrap requires a non-empty one-dimensional sequence")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    chunk_size = max(1, min(samples, 1_000_000 // len(array)))
    for start in range(0, samples, chunk_size):
        count = min(chunk_size, samples - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        estimates[start : start + count] = array[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "mean": float(array.mean()),
        "bootstrap_95_low": float(low),
        "bootstrap_95_high": float(high),
        "count": int(len(array)),
    }


def summarize_predictions(
    metadata: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    bootstrap_samples: int = 20_000,
    seed: int = 0,
) -> dict[str, object]:
    labels_value = np.asarray(labels, dtype=np.float64)
    scores_value = np.asarray(scores, dtype=np.float64)
    predictions = np.where(scores_value >= 0, 1.0, -1.0)
    if len(metadata) != len(labels_value) or len(labels_value) != len(scores_value):
        raise ValueError("prediction metadata, labels, and scores have different lengths")
    group_correct: dict[str, list[float]] = defaultdict(list)
    branch_correct: dict[str, list[float]] = defaultdict(list)
    paired_scores: dict[str, dict[str, float]] = defaultdict(dict)
    for row, label, prediction, score in zip(metadata, labels_value, predictions, scores_value, strict=True):
        pair_id = str(row["pair_id"])
        outcome = str(row["outcome"])
        correct = float(label == prediction)
        group_correct[pair_id].append(correct)
        branch_correct[outcome].append(correct)
        paired_scores[pair_id][outcome] = float(score)
    group_values = [float(np.mean(values)) for values in group_correct.values()]
    ranking = [
        float(values["attached"] > values["detached"])
        for values in paired_scores.values()
        if values.keys() >= {"attached", "detached"}
    ]
    return {
        "sample_accuracy": float(np.mean(predictions == labels_value)),
        "attached_accuracy": float(np.mean(branch_correct["attached"])),
        "detached_accuracy": float(np.mean(branch_correct["detached"])),
        "pair_ranking_accuracy": float(np.mean(ranking)),
        "group_bootstrap_95": bootstrap_mean(
            group_values, samples=bootstrap_samples, seed=seed
        ),
        "pair_ranking_bootstrap_95": bootstrap_mean(
            ranking, samples=bootstrap_samples, seed=seed + 1
        ),
        "group_count": len(group_values),
    }
