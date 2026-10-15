"""Rigid two-camera orbit geometry, independent of simulator and policy."""

import numpy as np


def orbit_rig(positions, rotations, *, azimuth_deg=0., elevation_deg=0., radius_scale=1., focus_distance=1.):
    """Orbit camera midpoint around a camera-derived reference, preserving rig.

    Rotations map camera-local vectors to world; optical forward is local -Z.
    Reference is midpoint plus mean optical direction times focus_distance.
    It does not use object ground truth. This is a three-parameter family, not
    arbitrary SE(3). All camera offsets rotate rigidly; radius never scales
    the left/right baseline. Values are absolute relative to the passed rig.
    """
    positions = np.asarray(positions, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    values = np.asarray([azimuth_deg, elevation_deg, radius_scale, focus_distance])
    if positions.shape != (2, 3) or rotations.shape != (2, 3, 3):
        raise ValueError('Expected exactly two camera positions and rotations')
    if not all(np.isfinite(x).all() for x in (positions, rotations, values)):
        raise ValueError('All geometry must be finite')
    if radius_scale <= 0 or focus_distance <= 0:
        raise ValueError('Positive radius scale and focus distance required')
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-6) or not np.allclose(np.linalg.det(rotations), 1., atol=1e-6):
        raise ValueError('Camera rotations must be proper orthonormal matrices')
    forward = -rotations[:, :, 2].mean(axis=0)
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        raise ValueError('Opposite camera directions have no shared forward')
    forward /= norm
    right = np.cross(forward, [0., 0., 1.])
    norm = np.linalg.norm(right)
    if norm < 1e-8:
        raise ValueError('Vertical rig requires an explicitly specified orbit frame')
    right /= norm

    def rotate(axis, degrees):
        x, y, z = axis
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        angle = np.deg2rad(degrees)
        return np.eye(3) + np.sin(angle) * skew + (1. - np.cos(angle)) * (skew @ skew)

    rotation = rotate(np.array([0., 0., 1.]), azimuth_deg) @ rotate(right, elevation_deg)
    midpoint = positions.mean(axis=0)
    pivot = midpoint + focus_distance * forward
    target_midpoint = pivot + radius_scale * (rotation @ (midpoint - pivot))
    target_positions = (positions - midpoint) @ rotation.T + target_midpoint
    target_rotations = rotation[None] @ rotations
    return dict(positions=target_positions, rotations=target_rotations, pivot=pivot,
                world_rotation=rotation, midpoint=target_midpoint)
