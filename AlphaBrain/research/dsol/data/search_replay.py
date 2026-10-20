"""Read-only, provenance-checked LIBERO search inputs and separate outcome labels.

This adapter reads the audited INITIAL32 cache, not raw simulator snapshots.
No encoder, policy, training, simulator, or environment dependency is imported.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import read, sha


def frozen_array(value, dtype=float):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class QueryResult:
    visual: np.ndarray
    accel_mean: float
    accel_std: float
    score_samples: int


class CachedQueries:
    """Only query() exposes candidate observations; labels are not accepted."""

    def __init__(self, visual, accel_members):
        visual, scores = np.asarray(visual), np.asarray(accel_members)
        if (visual.ndim != 2 or scores.ndim != 2 or len(visual) != len(scores)
                or min(visual.shape) < 1 or scores.shape[1] < 1
                or not np.isfinite(visual).all() or not np.isfinite(scores).all()
                or (scores < 0).any()):
            raise ValueError("Finite candidate features and nonnegative scoring members required")
        self._visual = frozen_array(visual)
        self._scores = frozen_array(scores)
        self.size, self.feature_dim = visual.shape
        self.score_samples = scores.shape[1]

    def query(self, candidate):
        if not isinstance(candidate, (int, np.integer)) or not 0 <= candidate < self.size:
            raise ValueError("Invalid candidate")
        scores = self._scores[candidate]
        return QueryResult(self._visual[candidate].copy(), float(scores.mean()),
                           float(scores.std()), self.score_samples)


class OutcomeTable:
    """Evaluation sidecar. NaN is missing, never a failure or a zero reward."""

    def __init__(self, labels):
        values = np.asarray(labels, dtype=float)
        if values.ndim != 4 or not np.all(np.isnan(values) | np.isin(values, [0, 1])):
            raise ValueError("Expected model x initial x candidate x repeat binary/NaN labels")
        self._labels = frozen_array(values)

    def paired(self, model, initial, start, selected):
        before = self._labels[model, initial, start]
        after = self._labels[model, initial, selected]
        valid = np.isfinite(before) & np.isfinite(after)
        count = int(valid.sum())
        return {
            "paired_repeats": count,
            "total_repeats": len(before),
            "stay": float(before[valid].mean()) if count else None,
            "selected": float(after[valid].mean()) if count else None,
            "gain": float((after[valid] - before[valid]).mean()) if count else None,
            "rescued_fraction": float(((before[valid] == 0) & (after[valid] == 1)).mean()) if count else None,
            "harmed_fraction": float(((before[valid] == 1) & (after[valid] == 0)).mean()) if count else None,
        }


@dataclass(frozen=True)
class ReplayInputs:
    candidate_ids: tuple
    states: tuple
    poses: np.ndarray
    visual: np.ndarray
    wrist: np.ndarray
    accel_members: np.ndarray
    development: tuple
    evaluation: tuple
    score_scales: tuple
    source_hashes: dict

    def queries(self, model, initial):
        return CachedQueries(self.visual[initial], self.accel_members[model, initial])


def load_initial32(source):
    """Require the historical schema/identity; no implicit path repair or fitting."""
    source = Path(source)
    provenance = read(source / "provenance.json")
    hashes = {str(source / "provenance.json"): sha(source / "provenance.json")}
    for name in ("matrix.npz", "states.json", "embeddings.npz"):
        digest = sha(source / name)
        if digest != provenance["artifacts"][name]:
            raise ValueError("Audited source changed: " + name)
        hashes[str(source / name)] = digest
    states = read(source / "states.json")
    with np.load(source / "matrix.npz", allow_pickle=False) as z:
        labels = z["success"].copy()
        accel = z["accel_members"].copy()
        poses = z["camera"].copy()
        ids = tuple(z["candidate_ids"].tolist())
    with np.load(source / "embeddings.npz", allow_pickle=False) as z:
        visual, context = z["external"].copy(), z["context"].copy()
    if (labels.shape != (2, 32, 97, 32) or not np.isin(labels, [0, 1]).all()
            or accel.shape != (2, 32, 97, 8) or poses.shape != (32, 97, 4, 4)
            or visual.shape != (32, 97, 512) or context.shape != (32, 1024)
            or len(states) != 32):
        raise ValueError("INITIAL32 cache has unexpected dimensions or missing labels")
    expected_ids = (("canonical",) + tuple(f"broad_train_{i:03d}" for i in range(64))
                    + tuple(f"broad_heldout_{i:03d}" for i in range(32)))
    if ids != expected_ids:
        raise ValueError("Candidate identities/order changed")
    if not all(np.isfinite(x).all() for x in (accel, poses, visual, context)) or (accel < 0).any():
        raise ValueError("Nonfinite inputs or negative Accel")
    if not np.allclose(poses[..., 3, :], [0, 0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError("Invalid camera transforms")
    rotation = poses[..., :3, :3]
    if (not np.allclose(np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-5, rtol=0)):
        raise ValueError("Invalid camera rotation")
    pairs = [s["pair_key"] for s in states]
    groups = [(s["task_id"], s["initialization"]["init_state_index"]) for s in states]
    if len(set(pairs)) != 32 or len(set(groups)) != 32:
        raise ValueError("Duplicate physical initial identity")
    ordinals = np.array([s["task_ordinal"] for s in states])
    initials = np.array([s["initialization"]["init_state_index"] for s in states])
    if (not np.array_equal(ordinals, np.repeat(np.arange(8), 4))
            or not np.array_equal(initials, np.tile(np.arange(4), 8))
            or any(s["initialization"]["kind"] != "official_libero_initial_state_v1" for s in states)
            or any(len({states[i]["task_id"] for i in range(t * 4, t * 4 + 4)}) != 1 for t in range(8))
            or len({s["task_id"] for s in states}) != 8):
        raise ValueError("Unexpected task/initial partition")
    dev, evaluation = tuple(np.flatnonzero(initials < 2)), tuple(np.flatnonzero(initials >= 2))
    # Only development scores set the scale. Never outcomes or evaluation scores.
    scales = tuple(float(np.median(accel[m, list(dev)].mean(-1))) for m in range(2))
    if min(scales) <= 0:
        raise ValueError("Degenerate development score scale")
    return ReplayInputs(
        ids, tuple(states), frozen_array(poses), frozen_array(visual),
        frozen_array(context[:, 512:]), frozen_array(accel), dev, evaluation, scales, hashes,
    ), OutcomeTable(labels)
