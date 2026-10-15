"""CPU-only audit of saved linked-rig mounts; no simulator or outcome access."""
import numpy as np
from scipy.spatial.transform import Rotation


def rotations_wxyz(quaternions):
    q = np.asarray(quaternions, dtype=float)
    return Rotation.from_quat(q.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(q.shape[:-1] + (3, 3))


def audit_mounts(positions, quaternions, exterior_ids, parameters, focus_distance=1.):
    """All exterior positions must share one parent coordinate frame.

    Relative-pose checks are frame invariant. The pivot check uses the saved
    initial mount frame; it does not reconstruct a moving world trajectory.
    """
    positions, quaternions, parameters = map(np.asarray, (positions, quaternions, parameters))
    if positions.ndim != 3 or positions.shape[-1] != 3 or quaternions.shape != positions.shape[:2] + (4,):
        raise ValueError('Expected candidate x camera x position/quaternion')
    if parameters.shape != (len(positions), 3) or len(set(exterior_ids)) != 2:
        raise ValueError('Expected two exterior cameras and three candidate parameters')
    if not all(np.isfinite(x).all() for x in (positions, quaternions, parameters)):
        raise ValueError('Nonfinite geometry')
    p = positions[:, exterior_ids]; r = rotations_wxyz(quaternions[:, exterior_ids])
    midpoint = p.mean(axis=1)
    forward = -r[:, :, :, 2].mean(axis=1)
    forward /= np.linalg.norm(forward, axis=1, keepdims=True)
    pivot = midpoint[0] + focus_distance * forward[0]
    baseline = np.linalg.norm(p[:, 1]-p[:, 0], axis=1)
    relative_p = np.einsum('vji,vj->vi', r[:, 0], p[:, 1]-p[:, 0])
    relative_r = np.einsum('vji,vjk->vik', r[:, 0], r[:, 1])
    untouched = [i for i in range(positions.shape[1]) if i not in exterior_ids]
    checks = dict(
        baseline_error_m=float(np.max(np.abs(baseline-baseline[0]))),
        relative_translation_error_m=float(np.max(np.abs(relative_p-relative_p[0]))),
        relative_rotation_component_error=float(np.max(np.abs(relative_r-relative_r[0]))),
        central_ray_pivot_error_m=float(np.max(np.abs(midpoint + focus_distance*parameters[:, 2, None]*forward-pivot))),
        untouched_mount_positions=bool(np.array_equal(positions[:, untouched], np.broadcast_to(positions[0, untouched], positions[:, untouched].shape))),
        untouched_mount_quaternions=bool(np.array_equal(quaternions[:, untouched], np.broadcast_to(quaternions[0, untouched], quaternions[:, untouched].shape))))
    if any(checks[k] > 1e-8 for k in checks if k.endswith(('_m', '_error'))) or not all(checks[k] for k in checks if k.startswith('untouched')):
        raise ValueError(f'Invalid linked rig: {checks}')
    return dict(positions=p, rotations=r, midpoint=midpoint, forward=forward, pivot=pivot,
                baseline=baseline, checks=checks,
                camera_displacement=np.linalg.norm(p[1:]-p[0], axis=-1),
                midpoint_displacement=np.linalg.norm(midpoint[1:]-midpoint[0], axis=-1),
                optical_angle_deg=np.rad2deg(np.arccos(np.clip(np.einsum('vci,ci->vc', -r[1:, :, :, 2], -r[0, :, :, 2]), -1, 1))))


def camera_opencv(position, native_rotation):
    """Convert MuJoCo local -Z optical forward to OpenCV local +Z."""
    result = np.eye(4)
    result[:3, 3] = position
    result[:3, :3] = native_rotation @ np.diag([1., -1., -1.])
    return result
