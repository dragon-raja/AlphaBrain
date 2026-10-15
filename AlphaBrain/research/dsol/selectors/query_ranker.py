"""Candidate-local ranker with no canonical image, flag, ID or target pose.

An acquired external image, current wrist image, known task ID and optionally
that acquired candidate's eight Accel measurements are sufficient at inference.
Within-initial centering fits view differences rather than task difficulty.
"""

from dataclasses import dataclass
import numpy as np


def _pca(x, dimension):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    return mean, vt[:min(dimension, len(x) - 1)]


@dataclass
class QueryRanker:
    image_mean: np.ndarray
    image_components: np.ndarray
    wrist_mean: np.ndarray
    wrist_components: np.ndarray
    use_accel: bool
    scale: np.ndarray | None = None
    weights: np.ndarray | None = None
    num_tasks: int = 8

    def features(self, image, wrist, task, accel=None):
        z = (np.asarray(image, dtype=np.float64) - self.image_mean) @ self.image_components.T
        w = (np.asarray(wrist, dtype=np.float64) - self.wrist_mean) @ self.wrist_components.T
        if self.use_accel:
            a = np.asarray(accel)
            if a.shape[-1] != 8 or not np.isfinite(a).all():
                raise ValueError("Eight queried Accel scores required")
            z = np.concatenate([z, np.stack([a.mean(-1), a.std(-1)], axis=-1)], axis=-1)
        t = np.eye(self.num_tasks)[np.asarray(task)]
        return np.concatenate([z, (z[..., :, None] * t[..., None, :]).reshape(*z.shape[:-1], -1),
                               (z[..., :, None] * w[..., None, :]).reshape(*z.shape[:-1], -1)], axis=-1)

    def score(self, image, wrist, task, accel=None):
        # Lower is better, matching the search API. This is a relative score,
        # not a calibrated probability and not a full-candidate normalization.
        return -(self.features(image, wrist, task, accel) / self.scale) @ self.weights


def fit_ranker(images, wrists, tasks, accel, labels, *, use_accel, alpha, image_dim=8, wrist_dim=4,
               num_tasks=8):
    """All arrays here must contain development initials only."""
    im, ic = _pca(images.reshape(-1, images.shape[-1]), image_dim)
    wm, wc = _pca(wrists.reshape(-1, wrists.shape[-1]), wrist_dim)
    model = QueryRanker(im, ic, wm, wc, use_accel, num_tasks=num_tasks)
    n, k = labels.shape
    query_wrists = wrists if wrists.ndim == 3 else np.broadcast_to(wrists[:, None], (n, k, wrists.shape[-1]))
    x = model.features(images, query_wrists,
                       np.broadcast_to(tasks[:, None], (n, k)), accel)
    x = (x - x.mean(axis=1, keepdims=True)).reshape(n * k, -1)
    y = (labels - labels.mean(axis=1, keepdims=True)).reshape(-1)
    model.scale = x.std(axis=0)
    model.scale[model.scale < 1e-8] = 1.
    x = x / model.scale
    model.weights = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)
    return model


def development_fit(images, wrists, tasks, accel, labels, initials, config, *, use_accel):
    """Reject heldout rows at the fit boundary, including preprocessing/CV."""
    if not set(np.unique(initials)).issubset({0, 1}):
        raise ValueError("Only initial 0/1 may enter development_fit")
    trials = []
    kwargs = dict(use_accel=use_accel, image_dim=config['image_pca_dim'], wrist_dim=config['wrist_pca_dim'])
    for alpha in config['alpha_grid']:
        values = []
        for valid in [0, 1]:
            tr, va = initials != valid, initials == valid
            model = fit_ranker(images[tr], wrists[tr], tasks[tr], accel[tr], labels[tr], alpha=alpha, **kwargs)
            for i in np.flatnonzero(va):
                scores = model.score(images[i], wrists[i], tasks[i], accel[i])
                values.append(float(labels[i, np.argmin(scores)]))
        trials.append(dict(alpha=alpha, development_oof_success=float(np.mean(values))))
    chosen = max(trials, key=lambda v: (v['development_oof_success'], v['alpha']))
    model = fit_ranker(images, wrists, tasks, accel, labels, alpha=chosen['alpha'], **kwargs)
    return model, dict(cv=trials, alpha=chosen['alpha'], objective='within-initial relative view value')
