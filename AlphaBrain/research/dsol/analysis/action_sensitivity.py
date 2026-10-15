"""Descriptive action differences; never turns them into a success-rate claim."""

import numpy as np


def action_difference(reference, actual):
    reference, actual = np.asarray(reference), np.asarray(actual)
    if reference.shape != actual.shape or reference.ndim != 2 or reference.shape[1] < 12:
        raise ValueError('Matching action chunks with at least 12 channels required')
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError('Finite action chunks required')
    delta = actual - reference
    return dict(max_abs=float(np.max(np.abs(delta))), rms=float(np.sqrt(np.mean(delta**2))),
        per_channel_max_abs=np.max(np.abs(delta), axis=0).tolist(),
        relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-12)),
        gripper_threshold_flips=int(np.sum((reference[:, 6] >= .5) != (actual[:, 6] >= .5))),
        control_mode_threshold_flips=int(np.sum((reference[:, 11] >= .5) != (actual[:, 11] >= .5))),
        scope='One generated chunk, not executed trajectory or task success equivalence')
