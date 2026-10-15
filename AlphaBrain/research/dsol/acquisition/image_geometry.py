"""Local pinhole image derivatives, independent of policy and score definition.

Diagnostic building blocks only: no renderer, search loop or robot command.
Optical frame: x right, y down, z forward. Camera body twist is ordered
(vx, vy, vz, wx, wy, wz), translation in metres, rotation in radians.
For a static world point, dX/dt = -v - w cross X. Images and metric axial
depth must be aligned in this frame, after any resize/flip and K adjustment.
Brightness constancy is local; newly revealed surfaces are NOT predicted.
"""
from __future__ import annotations

import numpy as np


def pixel_motion_jacobian(depth_m, intrinsics, *, valid_mask=None):
    """Return (H,W,2,6) point-motion Jacobian and its valid-depth mask.

    ``depth_m`` is optical z-depth, not Euclidean ray distance or a GL buffer.
    Only standard zero-skew pinhole intrinsics are supported. Invalid depth
    contributes zero, but is returned explicitly so callers cannot confuse
    missing geometry with a measured zero derivative.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    if depth.ndim != 2 or min(depth.shape) < 1:
        raise ValueError("depth_m must be a nonempty HxW array")
    if (k.shape != (3, 3) or not np.isfinite(k).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1], atol=1e-12, rtol=0)
            or k[0, 1] != 0 or k[1, 0] != 0):
        raise ValueError("intrinsics must be a finite zero-skew pinhole matrix")
    valid = np.isfinite(depth) & (depth > 0)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != depth.shape:
            raise ValueError("valid_mask shape differs from depth")
        valid &= mask
    v, u = np.indices(depth.shape, dtype=np.float64)
    x, y = (u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1]
    inv_z = np.zeros_like(depth)
    np.divide(1.0, depth, out=inv_z, where=valid)
    zeros = np.zeros_like(depth)
    lu = k[0, 0] * np.stack(
        [-inv_z, zeros, x * inv_z, x * y, -(1 + x*x), y], axis=-1)
    lv = k[1, 1] * np.stack(
        [zeros, -inv_z, y * inv_z, 1 + y*y, -x*y, -x], axis=-1)
    jacobian = np.stack([lu, lv], axis=-2)
    return np.where(valid[..., None, None], jacobian, 0.0), valid


def image_motion_jacobian(image, depth_m, intrinsics, *, valid_mask=None):
    """Approximate fixed-pixel dI/d(camera twist), shape (H,W,C,6).

    Uses dI/dt = -grad_uv(I) dot pixel_velocity. This minus sign differs
    from the motion of a tracked surface point. Channels and intensity scale
    are preserved. Depth edges/occlusions require an explicit caller mask.
    """
    pixels = np.asarray(image, dtype=np.float64)
    if (pixels.ndim != 3 or min(pixels.shape[:2]) < 3
            or pixels.shape[2] < 1 or not np.isfinite(pixels).all()):
        raise ValueError("image must be finite HxWxC, H and W at least 3")
    if np.shape(depth_m) != pixels.shape[:2]:
        raise ValueError("depth and image resolution differ")
    motion, valid = pixel_motion_jacobian(
        depth_m, intrinsics, valid_mask=valid_mask)
    gy, gx = np.gradient(pixels, axis=(0, 1), edge_order=2)
    image_jacobian = -(gx[..., None] * motion[:, :, None, 0, :]
                       + gy[..., None] * motion[:, :, None, 1, :])
    return image_jacobian, valid


def contract_image_gradient(image_gradient, image_jacobian, *, minimize=True):
    """Compose ANY scalar metric's image derivative with the geometric one.

    The caller supplies d(mean score)/d(image), not a saliency visualization.
    Return the six-dimensional gradient of a lower-is-better objective;
    set minimize=False for a higher-is-better score. This does not normalize
    metres/radians, enforce reachability, or issue a descent command.
    """
    grad = np.asarray(image_gradient, dtype=np.float64)
    jac = np.asarray(image_jacobian, dtype=np.float64)
    if (grad.ndim != 3 or jac.shape != (*grad.shape, 6)
            or not np.isfinite(grad).all() or not np.isfinite(jac).all()):
        raise ValueError("finite HxWxC gradient and HxWxCx6 Jacobian required")
    result = np.einsum("hwc,hwck->k", grad, jac)
    return result if minimize else -result
