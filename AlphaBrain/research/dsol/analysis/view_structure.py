"""Descriptive view structure on fixed-view, noise-averaged outcomes.

Canonical is kept as a baseline, not inserted into the noncanonical spatial
ranking. These post-hoc diagnostics neither fit a selector nor establish a
causal effect or a physically realizable camera search policy.
"""

import numpy as np
from scipy.stats import spearmanr


def correlation(x, y):
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def mean_field_rank_correlation(success, accel):
    """Correlation of the two plotted mean fields, not median initial rho.

    Inputs are initial x candidate noise means, with the caller explicitly
    selecting the same population and candidates as its spatial plot.
    """
    q, a = np.asarray(success, float), np.asarray(accel, float)
    if q.shape != a.shape or q.ndim != 2 or not q.size:
        raise ValueError('Expected matching nonempty initial x candidate means')
    if not np.isfinite(q).all() or not np.isfinite(a).all():
        raise ValueError('Non-finite spatial fields')
    return correlation(-a.mean(axis=0), q.mean(axis=0))


def score_quartiles(success, scores):
    """Per-initial success in four equally sized ascending-score groups."""
    q, a = np.asarray(success), np.asarray(scores)
    if q.shape != a.shape or q.ndim != 2 or q.shape[1] % 4:
        raise ValueError("Expected matching initial x candidate arrays divisible by four")
    if not np.isfinite(q).all() or not np.isfinite(a).all():
        raise ValueError("Non-finite input")
    order = np.argsort(a, axis=1, kind="stable")
    return np.take_along_axis(q, order, axis=1).reshape(len(q), 4, -1).mean(axis=-1)


def camera_distance(camera):
    """Translation in metres and SO(3) geodesic degrees relative to canonical."""
    c = np.asarray(camera)
    if c.ndim != 4 or c.shape[-2:] != (4, 4):
        raise ValueError("Expected initial x candidate x 4 x 4 camera transforms")
    translation = np.linalg.norm(c[:, :, :3, 3] - c[:, :1, :3, 3], axis=-1)
    rotation = np.einsum("sji,scjk->scik", c[:, 0, :3, :3], c[:, :, :3, :3])
    angle = np.rad2deg(np.arccos(np.clip((np.trace(rotation, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
    return translation, angle


def summarize_structure(success, accel, camera, tasks, initials, candidate_ids):
    q, a = np.asarray(success), np.asarray(accel)
    tasks, initials = np.asarray(tasks), np.asarray(initials)
    if q.shape != a.shape or q.ndim != 3 or q.shape[0] != 2:
        raise ValueError("Expected two model x initial x candidate arrays")
    if not np.isfinite(q).all() or not np.isfinite(a).all():
        raise ValueError("Non-finite input")
    if len(tasks) != q.shape[1] or len(initials) != len(tasks) or len(candidate_ids) != q.shape[2]:
        raise ValueError("Metadata mismatch")
    d, r = camera_distance(camera)
    if d.shape != q.shape[1:]:
        raise ValueError("Camera identity mismatch")
    rho = [correlation(a[0, i, 1:], a[1, i, 1:]) for i in range(len(tasks))]
    order = np.argsort(a[:, :, 1:], axis=-1, kind="stable")
    overlap = [len(set(order[0, i, :10]) & set(order[1, i, :10])) / 10 for i in range(len(tasks))]
    result = {
        "scope": "Exploratory; same measured noise means; 96 noncanonical views, no interpolation, no acquisition simulation",
        "cross_model": {"accel_rank_rho": rho, "low10_overlap_fraction": overlap},
        "geometry": {"minimum_noncanonical_rotation_deg": float(r[:, 1:].min()),
                     "minimum_noncanonical_translation_m": float(d[:, 1:].min())},
        "models": [],
    }
    for m in range(2):
        bands = score_quartiles(q[m, :, 1:], a[m, :, 1:])
        by_task = []
        for task in np.unique(tasks):
            mask = tasks == task
            qs = q[m, mask, 1:].mean(axis=0)
            top = np.argsort(-qs, kind="stable")[:5] + 1
            by_task.append({
                "task": int(task), "quartile_success": bands[mask].mean(axis=0).tolist(),
                "canonical_success": float(q[m, mask, 0].mean()),
                "top5_noncanonical": [{"candidate_id": str(candidate_ids[c]), "success": float(q[m, mask, c].mean())}
                                       for c in top],
            })
        result["models"].append({
            "quartiles_per_initial": bands.tolist(),
            "quartiles_all": bands.mean(axis=0).tolist(),
            "quartiles_heldout": bands[initials >= 2].mean(axis=0).tolist(),
            "per_initial_accel_success_rho": [correlation(-a[m, i, 1:], q[m, i, 1:]) for i in range(len(tasks))],
            "spatial_average_accel_success_rho": correlation(-a[m, :, 1:].mean(axis=0), q[m, :, 1:].mean(axis=0)),
            "per_initial_closeness_success_rho": {
                "translation": [correlation(-d[i, 1:], q[m, i, 1:]) for i in range(len(tasks))],
                "rotation": [correlation(-r[i, 1:], q[m, i, 1:]) for i in range(len(tasks))],
            },
            "by_task": by_task,
        })
    return result
