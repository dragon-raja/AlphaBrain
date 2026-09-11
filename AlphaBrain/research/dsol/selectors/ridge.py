"""Extracted from the audited INITIAL32 analysis; preserve scientific semantics."""

import numpy as np
from .specification import CONTRACT


def pca_fit(x, train, dimension):
    flat = x[train].reshape(-1, x.shape[-1])
    mean = flat.mean(axis=0)
    _, _, vt = np.linalg.svd(flat - mean, full_matrices=False)
    components = vt[: min(dimension, len(flat) - 1)]
    return (x - mean) @ components.T, mean, components


def feature_matrix(geometry, context, images, tasks, train, family):
    c, cm, cv = pca_fit(context, train, CONTRACT["context_pca_dim"])
    n, k, _ = geometry.shape
    task = np.eye(8)[tasks]
    broadcast = lambda x: np.broadcast_to(x[:, None, :], (n, k, x.shape[-1]))
    pieces = [
        geometry,
        broadcast(task),
        broadcast(c),
        (geometry[:, :, :, None] * task[:, None, None, :]).reshape(n, k, -1),
        (geometry[:, :, :, None] * c[:, None, None, :]).reshape(n, k, -1),
    ]
    transforms = {"context_mean": cm, "context_components": cv}
    if family == "candidate_image_ridge":
        z, zm, zv = pca_fit(images, train, CONTRACT["candidate_pca_dim"])
        pieces.extend([z, z - z[:, :1], (z[:, :, :, None] * task[:, None, None, :]).reshape(n, k, -1)])
        transforms.update(image_mean=zm, image_components=zv)
    elif family != "geometry_context_ridge":
        raise ValueError(family)
    return np.concatenate(pieces, axis=-1), transforms


def ridge_predict(features, labels, train, alpha):
    x = features[train].reshape(-1, features.shape[-1])
    y = labels[train].reshape(-1)
    mu = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1
    x = (x - mu) / scale
    ym = y.mean()
    w = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (y - ym))
    pred = ((features - mu) / scale) @ w + ym
    return pred, dict(feature_mean=mu, feature_scale=scale, weights=w, intercept=np.array(ym))


def fit_ranker(geometry, context, images, tasks, initials, q, family):
    dev = np.flatnonzero(initials < 2)
    trials = []
    for alpha in CONTRACT["alpha_grid"]:
        scores = []
        for vi in [0, 1]:
            train = np.flatnonzero(initials == 1 - vi)
            valid = np.flatnonzero(initials == vi)
            x, _ = feature_matrix(geometry, context, images, tasks, train, family)
            pred, _ = ridge_predict(x, q, train, alpha)
            choices = pred[valid].argmax(axis=1)
            scores.extend(q[valid, choices].tolist())
        trials.append({"alpha": alpha, "development_oof_success": float(np.mean(scores))})
    chosen = max(trials, key=lambda a: (a["development_oof_success"], a["alpha"]))
    x, transform = feature_matrix(geometry, context, images, tasks, dev, family)
    pred, weights = ridge_predict(x, q, dev, chosen["alpha"])
    return pred, {**transform, **weights}, {"cv": trials, "alpha": chosen["alpha"], "feature_dimension": x.shape[-1]}
