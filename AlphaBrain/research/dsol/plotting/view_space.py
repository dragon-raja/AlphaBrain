"""Plot measured camera coordinates; never interpolate unmeasured outcomes."""

import numpy as np


def candidate_coordinates(candidates):
    """Canonical has no task-pivot parameterization; return noncanonical coordinates."""
    if candidates[0]["pose"] is not None:
        raise ValueError("Expected canonical camera first")
    return np.array(
        [[c["pose"][key] for key in ["azimuth_deg", "elevation_deg", "radius_scale"]] for c in candidates[1:]],
        dtype=float,
    )


def projected_field(ax, coordinates, values):
    if len(coordinates) != len(values):
        raise ValueError("Coordinate/value mismatch")
    size = 12 + 55 * (coordinates[:, 2] - 0.85) / 0.4
    return ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=values,
        cmap="viridis",
        vmin=0,
        vmax=100,
        s=size,
        edgecolors="white",
        linewidths=0.3,
    )


def success_matrix(ax, values):
    if values.ndim != 2:
        raise ValueError("Expected initial x candidate matrix")
    return ax.imshow(values * 100, vmin=0, vmax=100, aspect="auto", cmap="viridis", interpolation="nearest")


def adaptive_limits(values, *, minimum_span=1.0, bounds=(0.0, 100.0)):
    """Full measured range with padding, never quantile clipping or saturation.

    Pass pooled arrays for a shared comparative scale. Separate local scales
    must be labelled explicitly; colors then cannot compare absolute values.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0 or not np.isfinite(v).all() or minimum_span <= 0:
        raise ValueError("Expected finite nonempty values and positive minimum span")
    lo, hi = float(v.min()), float(v.max())
    if lo < bounds[0] or hi > bounds[1]:
        raise ValueError("Values outside scale bounds")
    pad = max((hi - lo) * 0.03, (minimum_span - (hi - lo)) / 2)
    return max(bounds[0], lo - pad), min(bounds[1], hi + pad)


def absolute_space_fields(success, accel):
    """Comparable model x candidate fields; input has model x initial x view.

    Inputs are already noise-averaged. Candidate zero is canonical and excluded
    from this spatial parameterization. No ranking or model-wise normalization.
    """
    q, a = np.asarray(success), np.asarray(accel)
    if q.shape != a.shape or q.ndim != 3 or q.shape[0] != 2 or q.shape[2] < 2:
        raise ValueError("Expected matching two-model x initial x candidate arrays")
    if not np.isfinite(q).all() or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("Expected finite, nonnegative scores")
    if (q < 0).any() or (q > 1).any():
        raise ValueError("Success must be in [0, 1]")
    raw = a[:, :, 1:].mean(axis=1)
    upper = max(.05, float(np.ceil(raw.max() / .05) * .05))
    return dict(success=q[:, :, 1:].mean(axis=1) * 100,
                accel=raw, success_limits=(0., 100.), accel_limits=(0., upper))


def spatial_field(ax, coordinates, values, *, limits=(0, 100), cmap="viridis"):
    if len(coordinates) != len(values):
        raise ValueError("Coordinate/value mismatch")
    return ax.scatter(*coordinates.T, c=values, cmap=cmap, vmin=limits[0], vmax=limits[1], s=34, depthshade=False)
