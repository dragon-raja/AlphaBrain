"""CPU-only descriptive Accel fields; interpolation is never a measured rollout.

Geometry is scaled by the supplied catalog bounds, not outcome-dependent scales.
Validation withholds entire camera locations (all noise members together).
"""

import numpy as np
from scipy.interpolate import LinearNDInterpolator, RBFInterpolator
from scipy.spatial import Delaunay, cKDTree
from scipy.stats import spearmanr


def normalize_coordinates(coordinates):
    x = np.asarray(coordinates, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("Expected finite view x 3 coordinates")
    lo, span = x.min(0), np.ptp(x, axis=0)
    if (span <= 0).any() or len(np.unique(x, axis=0)) != len(x):
        raise ValueError("Expected distinct, three-dimensional coordinates")
    return (x - lo) / span, lo, span


def quadratic_design(x):
    x = np.asarray(x)
    return np.column_stack([np.ones(len(x)), x, x * x,
                            x[:, 0] * x[:, 1], x[:, 0] * x[:, 2], x[:, 1] * x[:, 2]])


def predict_field(x, values, query, method):
    """Fit only supplied measurements; do not clamp or hide overshoot."""
    x, y, query = np.asarray(x), np.asarray(values), np.asarray(query)
    if len(x) != len(y) or not np.isfinite(y).all():
        raise ValueError("Invalid measured values")
    if method == "nearest":
        return y[cKDTree(x).query(query)[1]]
    if method == "constant":
        return np.broadcast_to(y.mean(axis=0), (len(query),) + y.shape[1:]).copy()
    if method == "quadratic":
        return quadratic_design(query) @ np.linalg.lstsq(quadratic_design(x), y, rcond=None)[0]
    if method == "linear":
        return LinearNDInterpolator(x, y)(query)
    if method in {"rbf_exact", "rbf_smooth"}:
        return RBFInterpolator(x, y, kernel="thin_plate_spline", degree=1,
                               smoothing=0.0 if method == "rbf_exact" else 0.01)(query)
    raise ValueError("Unknown interpolation method: " + method)


def camera_folds(x, mode, seed=20260915):
    """Random 6-fold versus contiguous octants; uses geometry only."""
    if mode == "random":
        ids = np.empty(len(x), dtype=int)
        ids[np.random.default_rng(seed).permutation(len(x))] = np.arange(len(x)) % 6
    elif mode == "spatial":
        ids = (np.asarray(x) >= .5).astype(int) @ np.array([1, 2, 4])
    else:
        raise ValueError("Unknown fold mode")
    return [(ids != k, ids == k) for k in np.unique(ids)]


def field_metrics(truth, prediction):
    y, p = np.asarray(truth), np.asarray(prediction)
    mask = np.isfinite(p)
    y, p = y[mask], p[mask]
    result = {"coverage": float(mask.mean()), "n": len(y)}
    if len(y) < 4:
        return result
    error = np.mean((y - p) ** 2)
    variance = np.var(y)
    k = max(1, len(y) // 4)
    good = np.argsort(y, kind="stable")[:k]
    selected = np.argsort(p, kind="stable")[:k]
    result.update(rmse=float(np.sqrt(error)),
                  r2=None if variance == 0 else float(1 - error / variance),
                  rho=None if np.ptp(y) == 0 or np.ptp(p) == 0 else float(spearmanr(y, p).statistic),
                  low_quartile_recall=float(len(np.intersect1d(good, selected)) / k),
                  selected_accel=float(y[selected[0]]),
                  best_accel=float(y.min()),
                  selected_range_regret=None if np.ptp(y) == 0 else float((y[selected[0]] - y.min()) / np.ptp(y)))
    return result


def validate_interpolation(x, fields, methods, mode, target_fields=None):
    """fields: field x camera. Return OOF predictions and per-field metrics."""
    target_fields = fields if target_fields is None else np.asarray(target_fields)
    if target_fields.shape != fields.shape:
        raise ValueError("Training/target field shape mismatch")
    predictions = {}
    rows = []
    for method in methods:
        p = np.full_like(fields, np.nan)
        interior = np.zeros(len(x), dtype=bool)
        for train, test in camera_folds(x, mode):
            p[:, test] = predict_field(x[train], fields[:, train].T, x[test], method).T
            interior[test] = Delaunay(x[train]).find_simplex(x[test]) >= 0
        predictions[method] = p
        for i, (y, yp) in enumerate(zip(target_fields, p)):
            rows.append({"field": i, "method": method, "split": mode,
                         **field_metrics(y, yp),
                         "interior": field_metrics(y[interior], yp[interior])})
    return predictions, rows


def shape_metrics(a, b):
    a, b = np.asarray(a), np.asarray(b)
    result = field_metrics(a, b)
    return {k: result.get(k) for k in ("rho", "low_quartile_recall", "selected_range_regret")}


def structure_diagnostics(members, tasks, initials):
    """Separate repeated-noise, within-task, across-task and model variation."""
    a = np.asarray(members)
    if a.ndim != 4 or a.shape[0] != 2 or a.shape[-1] != 8 or not np.isfinite(a).all():
        raise ValueError("Expected finite 2 x initial x camera x 8 scores")
    tasks, initials = np.asarray(tasks), np.asarray(initials)
    if len(tasks) != a.shape[1] or len(initials) != len(tasks):
        raise ValueError("Metadata mismatch")
    mean = a.mean(-1)
    out = {"noise": [], "within_task": [], "other_task_template": [], "cross_model": []}
    for m in range(2):
        for s, y in enumerate(mean[m]):
            half = shape_metrics(a[m, s, :, :4].mean(-1), a[m, s, :, 4:].mean(-1))
            # Ratio is descriptive, not corrected reliability or a confidence interval.
            ratio = np.mean(a[m, s].var(-1, ddof=1) / 8) / max(np.var(y), 1e-30)
            centered = a[m, s] - a[m, s].mean(axis=0, keepdims=True)
            shape_ratio = np.mean(centered.var(-1, ddof=1) / 8) / max(np.var(y), 1e-30)
            out["noise"].append({"model": m, "initial": s, **half,
                                 "mean_noise_variance_to_spatial_variance": float(ratio),
                                 "offset_removed_noise_variance_to_spatial_variance": float(shape_ratio)})
            others = (tasks == tasks[s]) & (np.arange(len(tasks)) != s)
            out["within_task"].append({"model": m, "initial": s,
                                       **shape_metrics(y, mean[m, others].mean(0))})
            # Standardize only reference fields. Target values never enter the template.
            reference = mean[m, tasks != tasks[s]]
            standardized = (reference - reference.mean(1, keepdims=True)) / np.maximum(reference.std(1, keepdims=True), 1e-12)
            out["other_task_template"].append({"model": m, "initial": s,
                                               **shape_metrics(y, standardized.mean(0))})
    for s in range(len(tasks)):
        out["cross_model"].append({"initial": s, **shape_metrics(mean[0, s], mean[1, s])})
    return out


def quadratic_curvature(x, fields):
    coef = np.linalg.lstsq(quadratic_design(x), fields.T, rcond=None)[0].T
    rows = []
    for i, c in enumerate(coef):
        h = np.diag(2 * c[4:7])
        h[0, 1] = h[1, 0] = c[7]
        h[0, 2] = h[2, 0] = c[8]
        h[1, 2] = h[2, 1] = c[9]
        eigen = np.linalg.eigvalsh(h)
        positive = bool(eigen.min() > 1e-10)
        stationary = np.linalg.solve(h, -c[1:4]) if positive else None
        rows.append({"field": i, "hessian_eigenvalues": eigen.tolist(), "positive_definite": positive,
                     "stationary_in_box": bool(stationary is not None and ((stationary >= 0) & (stationary <= 1)).all()),
                     "stationary": None if stationary is None else stationary.tolist()})
    return rows


def dense_field(x, fields, resolution=24):
    """Interpolated estimates plus support metadata; outside convex hull is NaN."""
    axis = np.linspace(0, 1, resolution)
    query = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    supported = Delaunay(x).find_simplex(query) >= 0
    nearest = cKDTree(x).query(query)[0]
    prediction = predict_field(x, fields.T, query, "rbf_smooth").T
    prediction[:, ~supported] = np.nan
    return {"normalized_query": query, "inside_hull": supported,
            "nearest_measured_distance": nearest, "estimated_accel": prediction}
