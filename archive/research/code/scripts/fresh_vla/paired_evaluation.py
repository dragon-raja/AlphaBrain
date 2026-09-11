from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class EvaluationIdentity:
    sample_ids: tuple[str, ...]
    flow_times: np.ndarray
    noise: np.ndarray
    action_normalization: Mapping[str, Any]
    sample_content: np.ndarray | None = None

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps(self.sample_ids, separators=(",", ":")).encode())
        arrays = [self.flow_times, self.noise]
        if self.sample_content is not None:
            arrays.append(self.sample_content)
        for array in arrays:
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode())
            digest.update(str(contiguous.shape).encode())
            digest.update(contiguous.tobytes())
        digest.update(json.dumps(self.action_normalization, sort_keys=True, separators=(",", ":")).encode())
        return digest.hexdigest()


def per_sample_flow_metrics(per_step_loss: np.ndarray, horizons: Sequence[int], fixed_k: int = 3) -> list[dict[str, float | None]]:
    losses = np.asarray(per_step_loss, dtype=np.float64)
    horizon_values = np.asarray(horizons, dtype=np.int64)
    if losses.ndim != 2:
        raise ValueError(f"per_step_loss must be [B, H], got {losses.shape}")
    if horizon_values.shape != (losses.shape[0],):
        raise ValueError(f"expected {losses.shape[0]} horizons, got {horizon_values.shape}")
    if np.any((horizon_values < 0) | (horizon_values > losses.shape[1])):
        raise ValueError(f"horizons must be in [0, {losses.shape[1]}]")
    if not 1 <= fixed_k <= losses.shape[1]:
        raise ValueError(f"fixed_k must be in [1, {losses.shape[1]}], got {fixed_k}")

    rows = []
    for sample_loss, horizon in zip(losses, horizon_values, strict=True):
        rows.append(
            {
                "fixed_k": float(sample_loss[:fixed_k].mean()),
                "oracle_prefix": float(sample_loss[:horizon].mean()) if horizon > 0 else None,
                "suffix": float(sample_loss[horizon:].mean()) if horizon < len(sample_loss) else None,
                "full": float(sample_loss.mean()),
            }
        )
    return rows


def bootstrap_summary(values: Sequence[float], *, bootstrap_samples: int = 10_000, seed: int = 0) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("values must be a non-empty finite one-dimensional sequence")
    standard_error = float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0
    rng = np.random.default_rng(seed)
    means = []
    chunk_size = max(1, min(bootstrap_samples, 1_000_000 // array.size))
    for start in range(0, bootstrap_samples, chunk_size):
        count = min(chunk_size, bootstrap_samples - start)
        means.append(rng.choice(array, size=(count, array.size), replace=True).mean(axis=1))
    samples = np.concatenate(means)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "standard_error": standard_error,
        "bootstrap_95_low": float(low),
        "bootstrap_95_high": float(high),
    }


def paired_delta_summary(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    expected_fingerprint: str,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    def indexed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        result = {}
        for row in rows:
            if row.get("evaluation_fingerprint") != expected_fingerprint:
                raise ValueError("paired evaluation identity mismatch")
            sample_id = str(row["sample_id"])
            if sample_id in result:
                raise ValueError(f"duplicate sample_id: {sample_id}")
            result[sample_id] = row
        return result

    baseline = indexed(baseline_rows)
    candidate = indexed(candidate_rows)
    if baseline.keys() != candidate.keys():
        raise ValueError("paired evaluations must contain exactly the same sample IDs")
    sample_ids = sorted(baseline)
    deltas = [float(candidate[key][metric]) - float(baseline[key][metric]) for key in sample_ids]
    return {
        "metric": metric,
        "candidate_minus_baseline": bootstrap_summary(
            deltas,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "candidate_better": sum(delta < 0 for delta in deltas),
        "baseline_better": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "paired_deltas": dict(zip(sample_ids, deltas, strict=True)),
    }
