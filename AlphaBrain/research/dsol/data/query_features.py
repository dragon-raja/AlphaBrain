"""Ordered learned-query pooling, deliberately distinct from an image grid."""
import numpy as np


def pool_four_query_groups(tokens, *, queries, channels):
    x = np.asarray(tokens)
    if (type(queries) is not int or queries < 4 or queries % 4 or type(channels) is not int
            or channels < 1 or x.shape != (1, queries, channels) or x.dtype != np.float32
            or not np.isfinite(x).all()):
        raise ValueError('Explicit finite float32 ordered learned queries required')
    return x.reshape(4, queries//4, channels).mean(axis=1, dtype=np.float32).reshape(-1).copy()
