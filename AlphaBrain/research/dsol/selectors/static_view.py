"""Small static-view diagnostics, not a physical camera-search controller.

No outcome labels enter metric rules. Calibration consumes seen rows only and
fits relative view value, not an absolute probability of task success.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


def metric_choices(repeats, distance, *, neighborhood=7, se_penalty=1.):
    a, d = np.asarray(repeats, float), np.asarray(distance, float)
    if a.ndim != 3 or a.shape[-1] < 2 or d.shape != a.shape[:2] + (a.shape[1],):
        raise ValueError('Expected initial x view x score-repeat and matching distances')
    if not np.isfinite(a).all() or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError('Invalid score or distance')
    if not 1 <= neighborhood <= a.shape[1] or se_penalty < 0:
        raise ValueError('Invalid frozen rule parameters')
    mean = a.mean(-1)
    nearest = np.argsort(d, axis=-1, kind='stable')[..., :neighborhood]
    local = np.take_along_axis(mean[:, None, :], nearest, axis=-1).mean(-1)
    return dict(min_mean=mean.argmin(1), min_median=np.median(a, axis=-1).argmin(1),
                min_mean_plus_se=(mean + se_penalty*a.std(-1, ddof=1)/np.sqrt(a.shape[-1])).argmin(1),
                min_neighbor_mean=local.argmin(1))


def designs(repeats, parameters, tasks, *, task_interaction=False, family='accel_pose'):
    """All queried scores + known rig controls; no canonical image or labels.

    Controls are relative to a known calibrated mount, so pose-based variants
    are calibration-assisted, not canonical-free recovery in arbitrary scenes.
    """
    a, p, tasks = np.asarray(repeats, float), np.asarray(parameters, float), np.asarray(tasks)
    if a.ndim != 3 or p.shape != (a.shape[1], 3) or tasks.shape != (len(a),):
        raise ValueError('Invalid design axes')
    if family not in ('accel', 'pose', 'accel_pose') or not np.isfinite(a).all():
        raise ValueError('Invalid feature family or scores')
    mean = a.mean(-1)
    rank = (rankdata(mean, axis=1)-1)/max(a.shape[1]-1, 1)
    relative = (mean-mean.mean(1, keepdims=True))/np.maximum(mean.std(1, keepdims=True), 1e-12)
    variability = a.std(-1, ddof=1)/np.maximum(np.abs(mean), 1e-12)
    score_x = np.stack([rank, rank*rank, relative, variability], axis=-1)
    control = (p-np.array([0., 0., 1.]))/np.array([30., 15., .2])
    pose_x = np.column_stack([control, control*control,
                             control[:, 0]*control[:, 1], control[:, 0]*control[:, 2],
                             control[:, 1]*control[:, 2]])
    pose_x = np.broadcast_to(pose_x, (*a.shape[:2], pose_x.shape[-1]))
    x = score_x if family == 'accel' else pose_x if family == 'pose' else np.concatenate([score_x, pose_x], -1)
    if task_interaction:
        one_hot = (tasks[:, None] == np.unique(tasks)[None, :]).astype(float)
        interaction = (x[..., :, None]*one_hot[:, None, None, :]).reshape(*x.shape[:2], -1)
        x = np.concatenate([x, interaction], -1)
    return x


@dataclass
class RelativeRidge:
    scale: np.ndarray
    weights: np.ndarray

    def predict(self, x):
        return (np.asarray(x)/self.scale) @ self.weights


def fit_relative(x, labels, *, alpha):
    x, q = np.asarray(x, float), np.asarray(labels, float)
    if x.ndim != 3 or q.shape != x.shape[:2] or alpha <= 0:
        raise ValueError('Expected calibration initials x views, positive ridge penalty')
    if not np.isfinite(x).all() or not np.isfinite(q).all():
        raise ValueError('Nonfinite calibration data')
    centered = (x-x.mean(1, keepdims=True)).reshape(-1, x.shape[-1])
    target = (q-q.mean(1, keepdims=True)).reshape(-1)
    scale = centered.std(0)
    scale[scale < 1e-8] = 1.
    design = centered/scale
    weights = np.linalg.solve(design.T@design + alpha*np.eye(design.shape[-1]), design.T@target)
    return RelativeRidge(scale, weights)


def calibrate_seen(x, labels, train, layouts, tasks, *, alphas=(1., 10., 100.)):
    """Preprocessing and alpha selection never receive heldout labels.

    Caller must pass labels ONLY for the explicit seen rows. Layout-group CV
    keeps shared scene layouts out of their validation folds.
    """
    x, q, train = np.asarray(x), np.asarray(labels), np.asarray(train, int)
    layouts, tasks = np.asarray(layouts), np.asarray(tasks)
    if q.shape != x[train].shape[:2] or len(np.unique(train)) != len(train):
        raise ValueError('Only explicit seen labels may enter calibration')
    if len(np.unique(layouts[train])) < 2:
        raise ValueError('Need at least two seen layout groups')
    rows = []
    for alpha in alphas:
        values, local_tasks = [], []
        for layout in np.unique(layouts[train]):
            tr, va = layouts[train] != layout, layouts[train] == layout
            model = fit_relative(x[train[tr]], q[tr], alpha=alpha)
            choice = model.predict(x[train[va]]).argmax(1)
            values.extend(q[va, choice].tolist()); local_tasks.extend(tasks[train[va]].tolist())
        values, local_tasks = np.asarray(values), np.asarray(local_tasks)
        utility = np.mean([values[local_tasks == t].mean() for t in np.unique(local_tasks)])
        rows.append(dict(alpha=float(alpha), seen_layout_oof_success=float(utility)))
    selected = max(rows, key=lambda row: (row['seen_layout_oof_success'], row['alpha']))
    model = fit_relative(x[train], q, alpha=selected['alpha'])
    return model, dict(alpha=selected['alpha'], cv=rows)
