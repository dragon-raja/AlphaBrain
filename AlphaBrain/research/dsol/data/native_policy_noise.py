"""Explicit seven-to-32 coordinate noise binding for native LIBERO inference."""
import hashlib
import numpy as np

NOISE_RULE = 'libero-native32-tensor-sha256-extra25-v1'


def extend_noise(noise, seed, dimension=32):
    noise = np.asarray(noise, dtype=np.float32)
    if noise.shape != (10, 7) or not np.isfinite(noise).all():
        raise ValueError('Expected finite legacy noise of shape (10, 7)')
    if dimension == 7:
        return noise.copy()
    if dimension != 32:
        raise ValueError('Unsupported diagnostic action dimension')
    extra_seed = int.from_bytes(hashlib.sha256(
        f'libero-native32-extra25-v1::{int(seed)}'.encode()).digest()[:16], 'little')
    result = np.empty((10, 32), dtype=np.float32)
    result[:, :7] = noise
    result[:, 7:] = np.random.Generator(np.random.PCG64(extra_seed)).standard_normal((10, 25), dtype=np.float32)
    return result


def bound_noise(noise):
    """Same explicit input tensor => same extension at every pose and worker.

    The extra coordinates use the historical domain-separated generator. Their
    seed is now bound to canonical little-endian tensor bytes, not call order,
    worker identity, task outcome, or a mutable global random generator.
    """
    value = np.ascontiguousarray(noise, dtype='<f4')
    if value.shape != (10, 7) or not np.isfinite(value).all():
        raise ValueError('Expected finite scoring/execution noise (10, 7)')
    seed = int.from_bytes(hashlib.sha256(
        NOISE_RULE.encode() + b'\0' + value.tobytes()).digest()[:16], 'little')
    return extend_noise(value, seed), seed
