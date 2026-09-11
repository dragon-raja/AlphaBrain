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


def spatial_field(ax, coordinates, values):
    if len(coordinates) != len(values):
        raise ValueError("Coordinate/value mismatch")
    return ax.scatter(*coordinates.T, c=values, cmap="viridis", vmin=0, vmax=100, s=34, depthshade=False)
