"""Evidence bounds for recorded execution, without model or simulator calls."""
import numpy as np


def first_row_difference(left, right):
    """Exact row comparison on shared support; length mismatch is separate.

    A matching prefix is not a claim that the complete trajectories agree.
    Steps are zero-based array rows; callers must state the sampling cadence.
    """
    left, right = np.asarray(left), np.asarray(right)
    if (left.ndim < 2 or right.ndim != left.ndim
            or left.shape[1:] != right.shape[1:]
            or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError('Finite arrays with matching row schemas required')
    count = min(len(left), len(right))
    unequal = np.flatnonzero(np.any(left[:count] != right[:count], axis=tuple(range(1, left.ndim))))
    return dict(shared_rows=count, first_different_row=int(unequal[0]) if len(unequal) else None,
                lengths_equal=len(left) == len(right), dtypes_equal=left.dtype == right.dtype,
                complete_equal=bool(not len(unequal) and left.shape == right.shape and left.dtype == right.dtype))


def compare_recorded_noise(historical_calls, current_hashes):
    """Compare addressed noise on common support, not unequal episode tails."""
    if [r['replan_index'] for r in historical_calls] != list(range(len(historical_calls))):
        raise ValueError('Historical replanning addresses are not contiguous')
    old = [r['noise_sha256'] for r in historical_calls]
    count = min(len(old), len(current_hashes))
    if not count:
        raise ValueError('No shared noise records')
    bad = [i for i in range(count) if old[i] != current_hashes[i]]
    return dict(compared_replans=count, first_mismatch=bad[0] if bad else None,
                historical_replans=len(old), current_replans=len(current_hashes))
