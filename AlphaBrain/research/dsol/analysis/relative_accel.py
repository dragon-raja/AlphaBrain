"""Paired-noise label diagnostics; no policy or success labels required."""

import numpy as np


def paired_label_audit(members):
    a = np.asarray(members, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 4 or a.shape[1] % 2 or not np.isfinite(a).all():
        raise ValueError('Expected finite view x even-number-of-noises array')
    i, j = np.triu_indices(len(a), 1)
    d = a[i] - a[j]
    paired = d.var(-1, ddof=1)
    independent = a.var(-1, ddof=1)[i] + a.var(-1, ddof=1)[j]
    valid = independent > 1e-16
    split = a.shape[1] // 2
    directions = []
    for train, test in [(d[:, :split], d[:, split:]), (d[:, split:], d[:, :split])]:
        mean = train.mean(-1)
        se = train.std(-1, ddof=1) / np.sqrt(split)
        for multiplier in [0., 1., 2.]:
            # Heuristic signal/noise filters, not multiple-testing significance.
            keep = (abs(mean) > multiplier * se) & (abs(mean) > 1e-14)
            directions.append(dict(threshold_se=multiplier, coverage=float(keep.mean()),
                sign_agreement=float(np.mean(mean[keep] * test.mean(-1)[keep] > 0)) if keep.any() else None))
    centered = a - a.mean(axis=0, keepdims=True)
    return dict(pair_count=len(i), paired_to_independent_variance_median=float(np.median(paired[valid] / independent[valid])) if valid.any() else None,
                centered_to_raw_noise_variance=float(centered.var(-1, ddof=1).mean() / max(a.var(-1, ddof=1).mean(), 1e-16)),
                split_checks=directions)


def selection_metrics(predictions, target):
    """Aggregate starts within an initial; no rollout interpretation."""
    p, y = np.asarray(predictions), np.asarray(target)
    if p.ndim != 2 or y.shape != (p.shape[1],):
        raise ValueError('Expected start x view predictions and one target field')
    selected = p.argmin(-1)
    low = np.argsort(y, kind='stable')[:max(1, len(y) // 4)]
    return dict(low_quartile_hit=float(np.isin(selected, low).mean()),
                normalized_regret=float(((y[selected] - y.min()) / max(np.ptp(y), 1e-12)).mean()))
