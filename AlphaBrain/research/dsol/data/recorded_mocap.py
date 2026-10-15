"""Recover mocap inputs from recorded world body poses without simulation.

The compiled model's body-to-mocap mapping, not body order or names guessed
from a particular robot, determines the mocap slot. No quaternion fitting,
normalization, or initial-state substitution is performed.
"""
import numpy as np


def mocap_from_body_poses(body_mocapid, nmocap, model_names, recorded_names,
                          body_xpos, body_xquat):
    mapping = np.asarray(body_mocapid)
    if mapping.ndim != 1 or not np.issubdtype(mapping.dtype, np.integer):
        raise ValueError('body_mocapid must be a one-dimensional integer array')
    if tuple(model_names) != tuple(recorded_names) or len(model_names) != len(mapping):
        raise ValueError('Recorded body identity/order differs from compiled model')
    if nmocap < 0 or np.any(mapping < -1):
        raise ValueError('Invalid mocap mapping')
    bodies = np.flatnonzero(mapping >= 0)
    slots = mapping[bodies]
    if not np.array_equal(np.sort(slots), np.arange(nmocap)):
        raise ValueError('Mocap mapping must cover every slot exactly once')
    positions, quaternions = np.asarray(body_xpos), np.asarray(body_xquat)
    if positions.shape != (len(mapping), 3) or quaternions.shape != (len(mapping), 4):
        raise ValueError('Recorded body pose shape differs from compiled model')
    ordered_bodies = bodies[np.argsort(slots)]
    pos, quat = positions[ordered_bodies].copy(), quaternions[ordered_bodies].copy()
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(quat)):
        raise ValueError('Nonfinite recorded mocap body pose')
    if np.any(np.linalg.norm(quat, axis=1) == 0):
        raise ValueError('Zero recorded mocap body quaternion')
    return pos, quat, ordered_bodies
