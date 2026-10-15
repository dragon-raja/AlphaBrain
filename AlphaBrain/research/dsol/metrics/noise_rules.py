"""Score-noise rule comparisons, extending the preserved historical rules."""

import numpy as np
from .view_rules import metric_choices


def noise_rule_choices(accel_members, visibility):
    """Per-initial choices; single-noise decisions are NOT an ensemble decision.

    The single-noise rule returns one candidate per scoring seed. Evaluation
    averages their complete-task outcomes, never chooses a seed by success.
    """
    a, v = np.asarray(accel_members), np.asarray(visibility)
    if a.ndim != 3 or a.shape[-1] != 8 or v.shape != a.shape[:2]:
        raise ValueError('Expected initial x candidate x 8 scores and matching visibility')
    if not np.isfinite(a).all() or not np.isfinite(v).all():
        raise ValueError('Nonfinite metric inputs')
    return {'single_accel': a.argmin(axis=1), **metric_choices(a.mean(axis=-1), v)}
