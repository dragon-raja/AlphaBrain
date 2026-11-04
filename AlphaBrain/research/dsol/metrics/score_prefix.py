"""Compute a prefix score without modifying native solver or action axes."""
from ..data.score_prefix import validate
from .accel import compute_accel_scores


def score(trace, spec):
    import numpy as np
    validate(spec); array = np.asarray(trace)
    if array.shape != (spec['denoising_steps'], spec['scored_actions'], spec['valid_dimensions']):
        raise ValueError('Native trace does not match the explicit scoring window')
    value = compute_accel_scores(array[None], prefixes=[spec['prefix']])[spec['prefix']]
    if bool(value['degenerate'][0]): raise ValueError('Degenerate prefix score')
    return float(value['score'][0])
