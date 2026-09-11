"""Extracted from the audited INITIAL32 analysis; preserve scientific semantics."""

import numpy as np


def hierarchy(q):
    """q = task x initial x candidate noise-averaged success."""
    if q.ndim != 3 or not np.isfinite(q).all():
        raise ValueError("Expected a finite task x initial x candidate matrix")
    return dict(
        canonical=float(q[:, :, 0].mean()),
        uniform=float(q.mean()),
        global_best=float(q.mean(axis=(0, 1)).max()),
        task_best=float(q.mean(axis=1).max(axis=1).mean()),
        initial_oracle=float(q.max(axis=2).mean()),
    )


def paired_interval(values, subset, seed=20260911):
    """Mean/CI of fixed per-initial outcomes or paired differences (not max bias correction)."""
    v = np.asarray(values)[subset].reshape(8, -1)
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 8, (10000, 8))
    i = rng.integers(0, v.shape[1], (10000, 8, v.shape[1]))
    samples = v[t[:, :, None], i].mean(axis=(1, 2))
    return {"mean": float(v.mean()), "ci95": np.quantile(samples, [0.025, 0.975]).tolist()}
