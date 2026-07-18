from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
LEADS = (1, 2, 3, 4, 5)
HORIZON = 16


@dataclass(frozen=True)
class BranchExample:
    pair_id: str
    source_id: int
    split: str
    lead: int
    feature: np.ndarray
    attached: np.ndarray
    slipped: np.ndarray


def block_average(image: np.ndarray, size: int = 8) -> np.ndarray:
    value = np.asarray(image, dtype=np.float32)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 image, got {value.shape}")
    if size <= 0 or value.shape[0] % size or value.shape[1] % size:
        raise ValueError(f"image shape {value.shape[:2]} is not divisible by {size}")
    row = value.shape[0] // size
    column = value.shape[1] // size
    return (value.reshape(size, row, size, column, 3).mean(axis=(1, 3)) / 255.0).reshape(-1)


def observation_feature(
    agentview: np.ndarray,
    wrist: np.ndarray,
    robot_state: np.ndarray,
    *,
    lead: int | None = None,
    image_size: int = 8,
) -> np.ndarray:
    values = [
        block_average(agentview, image_size),
        block_average(wrist, image_size),
        np.asarray(robot_state, dtype=np.float32).reshape(-1),
    ]
    if lead is not None:
        if lead not in LEADS:
            raise ValueError(f"lead must be one of {LEADS}, got {lead}")
        one_hot = np.zeros(len(LEADS), dtype=np.float32)
        one_hot[LEADS.index(lead)] = 1.0
        values.append(one_hot)
    return np.concatenate(values).astype(np.float64)


def grouped_folds(source_ids: Sequence[int], folds: int = 5) -> list[np.ndarray]:
    unique = sorted(set(int(value) for value in source_ids))
    if len(unique) < folds:
        raise ValueError(f"need at least {folds} source groups, got {len(unique)}")
    buckets: list[list[int]] = [[] for _ in range(folds)]
    for source in unique:
        digest = hashlib.sha256(f"branch-vla-gate0:{source}".encode()).digest()
        buckets[int.from_bytes(digest[:8], "big") % folds].append(source)
    if any(not bucket for bucket in buckets):
        buckets = [unique[index::folds] for index in range(folds)]
    source_array = np.asarray(source_ids, dtype=np.int64)
    return [np.isin(source_array, bucket) for bucket in buckets]


@dataclass(frozen=True)
class RidgeModel:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        value = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        value = np.concatenate((value, np.ones((len(value), 1), dtype=np.float64)), axis=1)
        return value @ self.weight


def fit_ridge(features: np.ndarray, targets: np.ndarray, alpha: float) -> RidgeModel:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = (x - mean) / scale
    x = np.concatenate((x, np.ones((len(x), 1), dtype=np.float64)), axis=1)
    if len(x) < x.shape[1]:
        weight = x.T @ np.linalg.solve(
            x @ x.T + alpha * np.eye(len(x), dtype=np.float64),
            y,
        )
    else:
        gram = x.T @ x
        weight = np.linalg.solve(
            gram + alpha * np.eye(gram.shape[0], dtype=np.float64),
            x.T @ y,
        )
    return RidgeModel(mean=mean, scale=scale, weight=weight)


def select_alpha_regression(
    features: np.ndarray,
    targets: np.ndarray,
    source_ids: Sequence[int],
    *,
    alphas: Sequence[float] = ALPHAS,
) -> tuple[float, dict[str, float]]:
    fold_masks = grouped_folds(source_ids)
    scores: dict[str, float] = {}
    for alpha in alphas:
        fold_scores = []
        for validation in fold_masks:
            train = ~validation
            model = fit_ridge(features[train], targets[train], alpha)
            prediction = model.predict(features[validation])
            fold_scores.append(float(np.mean(np.square(prediction - targets[validation]))))
        scores[str(alpha)] = float(np.mean(fold_scores))
    selected = min(alphas, key=lambda value: (scores[str(value)], value))
    return float(selected), scores


def select_alpha_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    source_ids: Sequence[int],
    *,
    alphas: Sequence[float] = ALPHAS,
) -> tuple[float, dict[str, float]]:
    fold_masks = grouped_folds(source_ids)
    scores: dict[str, float] = {}
    for alpha in alphas:
        fold_accuracies = []
        for validation in fold_masks:
            train = ~validation
            model = fit_ridge(features[train], labels[train, None], alpha)
            prediction = np.where(model.predict(features[validation])[:, 0] >= 0, 1.0, -1.0)
            fold_accuracies.append(float(np.mean(prediction == labels[validation])))
        scores[str(alpha)] = float(np.mean(fold_accuracies))
    selected = max(alphas, key=lambda value: (scores[str(value)], -value))
    return float(selected), scores


def masked_mse(prediction: np.ndarray, target: np.ndarray, lead: int) -> float:
    return float(np.mean(np.square(prediction[lead:] - target[lead:])))


def prefix_mse(prediction: np.ndarray, target: np.ndarray, lead: int) -> float:
    return float(np.mean(np.square(prediction[:lead] - target[:lead])))


def mean_by_source(rows: Iterable[Mapping[str, float | int]], key: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["source_id"])].append(float(row[key]))
    return {source: float(np.mean(values)) for source, values in grouped.items()}


def bootstrap_mean(values: Sequence[float], *, samples: int = 20_000, seed: int = 260718) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("cannot bootstrap empty values")
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "bootstrap_95_low": float(np.quantile(draws, 0.025)),
        "bootstrap_95_high": float(np.quantile(draws, 0.975)),
    }
