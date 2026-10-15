"""Descriptive score stability and fixed-vs-instance diagnostics.

No simulator, fitting on confirmation outcomes, or choice of a winning window.
Noise-split privileged selection is not an unbiased estimate of a true oracle.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from ..data.artifacts import read, sha, write


def score_stability(scores):
    """Two disjoint four-repeat means; all candidates and initials retained."""
    scores = np.asarray(scores, float)
    if scores.ndim != 4 or scores.shape[2] != 8 or not np.isfinite(scores).all():
        raise ValueError('Expected finite initial x candidate x 8 repeats x window')
    result = []
    for window in range(1, scores.shape[-1]+1):
        left = scores[:, :, :4, :window].mean((2, 3))
        right = scores[:, :, 4:, :window].mean((2, 3))
        correlations = []
        for a, b in zip(left, right):
            if np.ptp(a) > 0 and np.ptp(b) > 0:
                correlations.append(float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]))
        k = min(10, scores.shape[1])
        overlap = [len(set(np.argsort(a, kind='stable')[:k]) & set(np.argsort(b, kind='stable')[:k]))/k
                   for a, b in zip(left, right)]
        result.append(dict(window=window, defined_initials=len(correlations),
            mean_spearman=float(np.mean(correlations)) if correlations else None,
            top1_agreement=float(np.mean(left.argmin(1) == right.argmin(1))),
            median_top10_overlap=float(np.median(overlap))))
    return result


def fixed_comparisons(success, accel, tasks, development, allowed):
    """Develop fixed choices on other initials; join heldout observed outcomes."""
    y = np.asarray(success); accel = np.asarray(accel); tasks = np.asarray(tasks)
    development = np.asarray(development, bool); allowed = np.asarray(allowed, int)
    if (y.ndim != 3 or y.shape[-1] < 2 or y.shape[-1] % 2 or not np.isin(y, [0, 1]).all()
            or accel.shape[:2] != y.shape[:2] or accel.ndim != 3 or not np.isfinite(accel).all()
            or tasks.shape != (len(y),) or development.shape != (len(y),)
            or not development.any() or development.all() or not len(allowed)
            or len(set(allowed)) != len(allowed) or allowed.min() < 0 or allowed.max() >= y.shape[1]):
        raise ValueError('Invalid complete matrix or development/candidate partition')
    q = y.mean(-1); row = np.arange(len(y)); test = ~development
    global_view = allowed[q[development][:, allowed].mean(0).argmax()]
    task_views = []
    for task in tasks:
        mask = development & (tasks == task)
        if not mask.any():
            raise ValueError('Task has no development initial')
        task_views.append(allowed[q[mask][:, allowed].mean(0).argmax()])
    task_views = np.asarray(task_views)
    metric_views = allowed[accel.mean(-1)[:, allowed].argmin(1)]
    half = y.shape[-1]//2; privileged = []
    for choose, evaluate in [(slice(None, half), slice(half, None)), (slice(half, None), slice(None, half))]:
        selected = allowed[y[:, :, choose].mean(-1)[:, allowed].argmax(1)]
        privileged.append(y[:, :, evaluate].mean(-1)[row, selected])
    return dict(evaluation_initials=int(test.sum()), global_fixed_view=int(global_view),
        task_fixed_views=task_views[test].tolist(),
        success={
            'global_fixed': float(q[test, global_view].mean()),
            'task_fixed': float(q[row, task_views][test].mean()),
            'accel_min': float(q[row, metric_views][test].mean()),
            'split_noise_privileged_selection': float(np.mean(privileged, axis=0)[test].mean())},
        caveat='Exploratory known-task initials. Success-supervised fixed choices have different supervision from Accel-only templates. Split-noise selection is a noisy privileged selector, not a true oracle upper bound.')


def run(config_file):
    config = read(config_file); output = Path(config['output'])
    if output.exists():
        raise FileExistsError(output)
    result = dict(sources={str(config_file): sha(config_file), str(__file__): sha(__file__)},
        interpretation='Descriptive only; no new rule frozen or independent confirmation claimed.', models={})
    for name, item in config['panels'].items():
        result['sources'][item['arrays']] = sha(item['arrays'])
        with np.load(item['arrays'], allow_pickle=False) as data:
            if item['kind'] == 'confirmation':
                for model in ['official', 'xr1']:
                    result['models'][name+'_'+model] = dict(score_stability=score_stability(data[model+'__scores']))
            elif item['kind'] == 'libero':
                states = read(item['states']); result['sources'][item['states']] = sha(item['states'])
                tasks = [s['task_id'] for s in states]
                development = [s['split'] == 'development' for s in states]
                for index, model in enumerate(['canonical', 'broad']):
                    scores = data['accel_members'][index]
                    result['models'][name+'_'+model] = dict(score_stability=score_stability(scores[..., None]),
                        fixed_comparisons={scope: fixed_comparisons(data['success'][index], scores, tasks, development, ix)
                            for scope, ix in [('all97', np.arange(97)), ('noncanonical96', np.arange(1, 97))]})
            else:
                raise ValueError('Unknown panel kind')
    write(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', required=True)
    run(parser.parse_args().config)
