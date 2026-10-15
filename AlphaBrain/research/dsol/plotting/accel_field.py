"""Explicitly labelled estimated Accel slices and measured structure diagnostics."""

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from AlphaBrain.research.dsol.analysis.accel_field import predict_field


def field_slices(coordinates, x, values, title):
    import matplotlib.pyplot as plt

    lo, span = coordinates.min(0), np.ptp(coordinates, axis=0)
    axis = np.linspace(0, 1, 65)
    gx, gy = np.meshgrid(axis, axis)
    radii = [.2, .5, .8]
    queries = [np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, r)]) for r in radii]
    estimates = [predict_field(x, values.T, q, "rbf_smooth").T for q in queries]
    support = [Delaunay(x).find_simplex(q) >= 0 for q in queries]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for m, label in enumerate(["Canonical-trained", "Multiview-trained"]):
        pooled = np.concatenate([values[m]] + [z[m, mask] for z, mask in zip(estimates, support)])
        vmin, vmax = pooled.min(), pooled.max()
        for j, (q, z, mask) in enumerate(zip(queries, estimates, support)):
            ax = axes[m, j]
            field = np.where(mask, z[m], np.nan).reshape(gx.shape)
            im = ax.pcolormesh(gx * span[0] + lo[0], gy * span[1] + lo[1], field,
                               shading="nearest", cmap="viridis_r", vmin=vmin, vmax=vmax)
            near = abs(x[:, 2] - radii[j]) < .06
            ax.scatter(coordinates[near, 0], coordinates[near, 1], s=23, facecolors="none", edgecolors="black", linewidths=.6)
            ax.set(title=f"{label} | radius={lo[2] + radii[j] * span[2]:.3f}", xlabel="Azimuth (deg)", ylabel="Elevation (deg)")
            ax.set_facecolor("#eeeeee")
        fig.colorbar(im, ax=axes[m].tolist(), label="Estimated mean Accel (row-specific scale)", shrink=.85)
    fig.suptitle(title + "\nRBF estimate, NOT measured intermediate views | gray: outside measured convex hull\n"
                 "Hollow circles: nearby measured sites (radius +/- 6% of range); low Accel is bright", fontsize=11)
    return fig


def diagnostic_heatmaps(diagnostics, task_labels):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for m in range(2):
        for j, (key, title) in enumerate([
            ("noise", "Noise halves: 4 vs 4"),
            ("within_task", "Other initials, same task"),
            ("other_task_template", "Template from other tasks"),
        ]):
            vals = np.array([r["low_quartile_recall"] for r in diagnostics[key] if r["model"] == m]).reshape(8, 4)
            ax = axes[m, j]
            im = ax.imshow(vals, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax.set(title=title, xlabel="Official initial index", xticks=range(4), yticks=range(8), yticklabels=task_labels if j == 0 else [])
            for r in range(8):
                for c in range(4):
                    ax.text(c, r, f"{vals[r,c]:.2f}", ha="center", va="center", fontsize=7, color="black" if vals[r,c] > .55 else "white")
        axes[m, 0].set_ylabel(["Canonical-trained", "Multiview-trained"][m])
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Overlap of lowest 24 / 96 views (chance: 0.25)", shrink=.85)
    fig.suptitle("Do low-Accel regions survive changing noise / initial / task?\nMeasured locations only; no interpolation", fontsize=12)
    return fig


def sampling_support(coordinates, x):
    import matplotlib.pyplot as plt

    lo, span = coordinates.min(0), np.ptp(coordinates, axis=0)
    fig = plt.figure(figsize=(12, 4.2), layout="constrained")
    ax = fig.add_subplot(131, projection="3d")
    ax.scatter(*coordinates.T, s=20)
    ax.set(xlabel="Azimuth (deg)", ylabel="Elevation (deg)", zlabel="Radius scale", title="96 measured positions")
    xx, yy = np.meshgrid(np.linspace(0, 1, 70), np.linspace(0, 1, 70))
    q = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, .5)])
    inside = Delaunay(x).find_simplex(q) >= 0
    distance = cKDTree(x).query(q)[0]
    ax = fig.add_subplot(132)
    im = ax.pcolormesh(xx * span[0] + lo[0], yy * span[1] + lo[1], np.where(inside, distance, np.nan).reshape(xx.shape), shading="nearest")
    fig.colorbar(im, ax=ax, label="Nearest-site distance (scaled coordinates)")
    ax.set(title="Mid-radius geometric support", xlabel="Azimuth (deg)", ylabel="Elevation (deg)")
    ax = fig.add_subplot(133)
    ids = (x >= .5).astype(int) @ np.array([1, 2, 4])
    ax.scatter(*coordinates[:, :2].T, c=ids, cmap="tab10", s=25 + 45*x[:,2])
    ax.set(title="Spatial holdout octants (projected)", xlabel="Azimuth (deg)", ylabel="Elevation (deg)")
    fig.suptitle("Support is a sampled 3-parameter camera family, not full SE(3). Distance is not prediction uncertainty.", fontsize=11)
    return fig
