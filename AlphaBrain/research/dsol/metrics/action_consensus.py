"""Label-free same-state, paired-noise action agreement (not correctness)."""
import numpy as np
from scipy.spatial.distance import cdist


def disagreement(actions, horizon, channels):
    """Mean pairwise RMS command distance, averaging paired noise distances.

    actions: candidate, noise, time, channel. No cross-noise comparisons,
    channel rescaling, outcome-dependent weights, or clustering thresholds.
    The catalog is uniformly weighted; duplicated views change its density.
    """
    a = np.asarray(actions, dtype=np.float64)
    if a.ndim != 4 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError('Finite candidate/noise/time/channel actions required')
    if not 1 <= horizon <= a.shape[2]:
        raise ValueError('Invalid command horizon')
    a = a[:, :, :horizon, channels]
    if not a.shape[-1]:
        raise ValueError('No channels')
    distances = []
    for repeat in range(a.shape[1]):
        x = a[:, repeat].reshape(len(a), -1)
        d = cdist(x, x, metric='euclidean') / np.sqrt(x.shape[1])
        distances.append(d.sum(1) / (len(a)-1))
    return np.mean(distances, axis=0)
