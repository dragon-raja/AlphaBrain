"""Cross-initial/task Accel templates: coverage and shape diagnostics only.

No success labels, candidate acquisition policy, or hidden target-template fit.
Target values are used for calibration only in the explicitly labelled folds.
"""

import numpy as np

from AlphaBrain.research.dsol.analysis.accel_field import camera_folds, field_metrics, quadratic_design


def template_for(fields, tasks, target, kind):
    fields, tasks = np.asarray(fields), np.asarray(tasks)
    if fields.ndim != 2 or len(fields) != len(tasks) or not np.isfinite(fields).all():
        raise ValueError("Expected finite initial x view fields and matching tasks")
    if kind == "same_task":
        keep = (tasks == tasks[target]) & (np.arange(len(tasks)) != target)
        ref = fields[keep]
    elif kind == "other_tasks":
        ref = fields[tasks != tasks[target]]
        ref = (ref - ref.mean(1, keepdims=True)) / np.maximum(ref.std(1, keepdims=True), 1e-12)
    else:
        raise ValueError("Unknown template kind")
    if not len(ref):
        raise ValueError("No reference fields")
    return ref.mean(0)


def coverage_curve(truth, template, low_count):
    """Fraction of true low-score sites captured by retaining 1..N template sites.

    Site fraction is NOT continuous volume, probability of task success, or
    fraction of camera moves. Equal-score ties are broken by catalog order.
    """
    y, t = np.asarray(truth), np.asarray(template)
    if y.ndim != 1 or y.shape != t.shape or not np.isfinite(y).all() or not np.isfinite(t).all():
        raise ValueError("Expected matching finite vectors")
    if not 1 <= low_count <= len(y):
        raise ValueError("Invalid low-region size")
    low = np.zeros(len(y), dtype=bool)
    low[np.argsort(y, kind="stable")[:low_count]] = True
    order = np.argsort(t, kind="stable")
    capture = np.cumsum(low[order]) / low_count
    return {"capture": capture,
            "min_sites_capture80": int(np.flatnonzero(capture >= .8)[0] + 1),
            "min_sites_capture90": int(np.flatnonzero(capture >= .9)[0] + 1),
            "min_sites_capture100": int(np.flatnonzero(capture >= 1)[0] + 1)}


def calibrated_prediction(template, x, values, train, method):
    """Positive affine plus optional spatial residual; target test values unused.

    Solve y = Z beta + b template, b >= 0 by residualizing against Z.
    Full-map fits describe shape; OOF fits diagnose predictability. Neither is
    advertised as a zero-query or low-query deployable method.
    """
    t, y, x, train = np.asarray(template), np.asarray(values), np.asarray(x), np.asarray(train, dtype=bool)
    if t.shape != y.shape or len(x) != len(y) or len(train) != len(y) or not np.isfinite(t).all() or not np.isfinite(y).all():
        raise ValueError("Invalid calibration inputs")
    if method == "constant":
        return np.full_like(y, y[train].mean(), dtype=float), {"scale": 0.0}
    if method == "affine":
        z = np.ones((len(y), 1))
    elif method == "affine_linear":
        z = np.column_stack([np.ones(len(x)), x - .5])
    elif method == "affine_quadratic":
        z = quadratic_design(x - .5)
    else:
        raise ValueError("Unknown calibration method")
    zt, tt, yt = z[train], t[train], y[train]
    tr = tt - zt @ np.linalg.lstsq(zt, tt, rcond=None)[0]
    yr = yt - zt @ np.linalg.lstsq(zt, yt, rcond=None)[0]
    denominator = float(tr @ tr)
    b = 0.0 if denominator < 1e-20 else max(0.0, float(tr @ yr) / denominator)
    beta = np.linalg.lstsq(zt, yt - b * tt, rcond=None)[0]
    return z @ beta + b * t, {"scale": b, "spatial_coefficients": beta.tolist()}


def analyze_templates(members, tasks, x, low_fractions=(.1, .25, .4)):
    members = np.asarray(members)
    if members.ndim != 4 or members.shape[0] != 2 or members.shape[-1] != 8 or not np.isfinite(members).all():
        raise ValueError("Expected two model x initial x view x eight noise scores")
    means = members.mean(-1)
    n = means.shape[-1]
    if len(x) != n:
        raise ValueError("Geometry mismatch")
    coverage, decomposition, curves = [], [], {}
    methods = ["constant", "affine", "affine_linear", "affine_quadratic"]
    for m in range(2):
        for kind in ["same_task", "other_tasks"]:
            templates = np.array([template_for(means[m], tasks, s, kind) for s in range(len(tasks))])
            for fraction in low_fractions:
                count = max(1, int(np.ceil(n * fraction)))
                group = []
                for s, y in enumerate(means[m]):
                    row = coverage_curve(y, templates[s], count)
                    group.append(row.pop("capture"))
                    coverage.append({"model": m, "initial": s, "template": kind,
                                     "low_count": count, **row,
                                     "capture_at_24": float(group[-1][23]),
                                     "capture_at_32": float(group[-1][31]),
                                     "capture_at_48": float(group[-1][47])})
                curves[f"m{m}_{kind}_low{count}"] = np.array(group)
            # Rank structure check with disjoint scoring noise halves AND held-out task/initial.
            for direction in range(2):
                src = members[m, ..., direction*4:direction*4+4].mean(-1)
                dst = members[m, ..., (1-direction)*4:(1-direction)*4+4].mean(-1)
                group = []
                for s, y in enumerate(dst):
                    template = template_for(src, tasks, s, kind)
                    group.append(coverage_curve(y, template, n//4)["capture"])
                curves[f"m{m}_{kind}_noise{direction}"] = np.array(group)
            for s, y in enumerate(means[m]):
                template = templates[s]
                for mode in ["full_map_descriptive", "random", "spatial"]:
                    folds = [(np.ones(n, dtype=bool), np.ones(n, dtype=bool))] if mode == "full_map_descriptive" else camera_folds(x, mode)
                    for method in methods:
                        prediction = np.empty(n)
                        scales = []
                        for train, test in folds:
                            p, fit = calibrated_prediction(template, x, y, train, method)
                            prediction[test] = p[test]
                            scales.append(fit["scale"])
                        decomposition.append({"model": m, "initial": s, "template": kind,
                                              "split": mode, "method": method,
                                              "scales": scales, **field_metrics(y, prediction)})
    return coverage, decomposition, curves
