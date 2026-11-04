"""Training-only fixed reward units; never fit observation units or use outcomes."""
import math
import statistics
from ..execution.manifest import digest


def fixed_model_scale(records, training_addresses, *, model_sha256):
    expected = set(training_addresses)
    if not expected or len(expected) != len(training_addresses) or not model_sha256:
        raise ValueError('Explicit unique training addresses and model identity required')
    seen = set(); values = []
    for row in records:
        address = row['address']
        if (row['split'] != 'train' or address not in expected or address in seen
                or row['model_sha256'] != model_sha256 or row['query_kind'] != 'initial'):
            raise ValueError('Calibration must cover training initial queries exactly once')
        scores = row['score_members']
        if (len(scores) != 2 or any(isinstance(x, bool) or not math.isfinite(x) or x < 0 for x in scores)):
            raise ValueError('Two finite nonnegative paired scores required')
        seen.add(address); values.append(statistics.mean(scores))
    if seen != expected:
        raise ValueError('Missing training calibration queries')
    scale = statistics.median(values)
    if scale <= 0:
        raise ValueError('Degenerate training reward scale')
    return dict(schema='FIXED_MODEL_REWARD_SCALE_V1', scale=scale,
                estimator='median_of_initial_two_member_means', count=len(values),
                model_sha256=model_sha256, training_addresses_sha256=digest(sorted(expected)),
                records_sha256=digest(records), fixed_during_training=True,
                observation_scale_changed=False, query_cost=.01)
