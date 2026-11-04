"""Task-equal measured response maps; distinguish marginal and within-state ranks."""
from collections import defaultdict
import numpy as np

from .spatial_score_validity import correlation, ranks


def pearson(x, y):
    """Pearson correlation of measured values, not their ranks."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 2 or not np.isfinite([x, y]).all():
        raise ValueError('Paired finite vectors required')
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def linear_comparison(rows, *, score='p3', expected_views=27):
    """Average spatial-field Pearson plus explicit pooled/within-state diagnostics.

    Primary positive orientation is corr(-Accel, success), consistent with the
    historical favorable-direction plots. Raw signed correlation is also saved.
    No per-state correlation median is substituted for the spatial-field r.
    """
    converted = [dict(row, p3=row[score]) for row in rows]
    averaged = summarize(converted, expected_views=expected_views)
    scopes = {}
    per_state = []
    for exclude in (False, True):
        start = int(exclude)
        x = np.asarray(averaged['mean']['score'][start:])
        y = np.asarray(averaged['mean']['success'][start:])
        raw_r = pearson(x, y)
        slope, intercept = np.polyfit(x, y, 1) if np.ptp(x) else (0., y.mean())
        xx, yy, cx, cy = [], [], [], []
        for state in sorted({r['state_key'] for r in converted}):
            values = sorted([r for r in converted if r['state_key'] == state and r['view_id'] >= start], key=lambda r: r['view_id'])
            a = np.asarray([np.mean(r['p3']) for r in values])
            b = np.asarray([np.mean(r['success']) for r in values])
            xx.extend(a); yy.extend(b); cx.extend(a-a.mean()); cy.extend(b-b.mean())
            if exclude:
                per_state.append(dict(state_key=state, task=values[0]['task'], raw_r=pearson(a, b),
                                      favorable_r=pearson(-a, b)))
        scopes['nonreference' if exclude else 'all'] = dict(n_views=len(x),
            raw_r=raw_r, favorable_r=None if raw_r is None else -raw_r,
            r_squared=None if raw_r is None else raw_r**2,
            raw_slope=float(slope), raw_intercept=float(intercept),
            pooled_raw_r=pearson(xx, yy), pooled_favorable_r=pearson(-np.asarray(xx), yy),
            state_centered_raw_r=pearson(cx, cy), state_centered_favorable_r=pearson(-np.asarray(cx), cy),
            pooled_positions=len(xx))
    per_task = [dict(task=t['task'], favorable_r=pearson(-np.asarray(t['score'][1:]), t['success'][1:]))
                for t in averaged['per_task']]
    return dict(score=score, tasks=averaged['tasks'], states=averaged['states'],
                angles_deg=averaged['angles_deg'], mean=averaged['mean'], scopes=scopes,
                per_state=per_state, per_task=per_task,
                sign='favorable_r = Pearson(-Accel, empirical success); positive supports lower Accel / higher success',
                scope='Measured views only. Average-field association is not an individual-state selector or a causal training diagnosis.')


def summarize(rows, *, expected_views=27, bootstrap_seed=20261029):
    groups = defaultdict(list)
    for row in rows:
        groups[row['state_key']].append(row)
    if not groups:
        raise ValueError('Empty panel')
    tasks = defaultdict(list)
    geometry = None
    per_state = []
    for state, values in sorted(groups.items()):
        values = sorted(values, key=lambda r: r['view_id'])
        if (len(values) != expected_views or
                [r['view_id'] for r in values] != list(range(expected_views)) or
                len({r['task'] for r in values}) != 1):
            raise ValueError('Incomplete or inconsistent state')
        angles = np.asarray([r['angles_deg'] for r in values], float)
        scores = np.asarray([r['p3'] for r in values], float)
        if (angles.shape != (expected_views, 2) or scores.shape != (expected_views, 2)
                or not np.isfinite(angles).all() or not np.isfinite(scores).all()
                or len(set(map(tuple, angles))) != expected_views):
            raise ValueError('Invalid measured geometry or scores')
        if any(len(r['success']) != 2 or any(type(x) is not bool for x in r['success']) for r in values):
            raise ValueError('Two explicit Boolean outcomes required')
        if geometry is not None and not np.array_equal(geometry, angles):
            raise ValueError('View identity has different geometry across states')
        geometry = angles
        score, success = scores.mean(axis=1), np.asarray([r['success'] for r in values]).mean(axis=1)
        # Display-only rank, never a training reward or outcome-dependent selection.
        rank = ranks(score) / (expected_views - 1)
        tasks[values[0]['task']].append((score, success, rank))
        per_state.append(dict(state_key=state, task=values[0]['task'],
                              rho_nonreference=correlation(score[1:], success[1:])))
    per_task = []
    for task, arrays in sorted(tasks.items()):
        mean = np.mean(arrays, axis=0)
        per_task.append(dict(task=task, states=len(arrays), score=mean[0].tolist(),
            success=mean[1].tolist(), rank=mean[2].tolist(),
            rho_all=correlation(mean[0], mean[1]),
            rho_nonreference=correlation(mean[0, 1:], mean[1, 1:]),
            state_rhos=[x['rho_nonreference'] for x in per_state if x['task'] == task]))
    mean = {key: np.mean([t[key] for t in per_task], axis=0).tolist()
            for key in ('score', 'success', 'rank')}
    valid = [t['rho_nonreference'] for t in per_task if t['rho_nonreference'] is not None]
    interval = None
    if valid:
        draws = np.random.default_rng(bootstrap_seed).choice(valid, size=(5000, len(valid))).mean(axis=1)
        interval = np.quantile(draws, [.025, .975]).tolist()
    state_rhos = [x['rho_nonreference'] for x in per_state if x['rho_nonreference'] is not None]
    return dict(tasks=len(tasks), states=len(groups), views=expected_views, positions=len(rows),
        angles_deg=geometry.tolist(), per_task=per_task, per_state=per_state, mean=mean,
        rho_average_surface_all=correlation(mean['score'], mean['success']),
        rho_average_surface_nonreference=correlation(mean['score'][1:], mean['success'][1:]),
        rho_average_rank_surface_nonreference=correlation(mean['rank'][1:], mean['success'][1:]),
        mean_task_rho_nonreference=float(np.mean(valid)) if valid else None,
        mean_task_rho_cluster_95=interval,
        median_state_rho_nonreference=float(np.median(state_rhos)) if state_rhos else None,
        valid_task_correlations=len(valid), valid_state_correlations=len(state_rhos),
        weighting='Within state: two members/repeats; within task: equal states; across tasks: equal tasks',
        inference='Posthoc descriptive development panel; negative rho favors low Accel/high success. '
                  'No correlation computed on interpolated points. Task bootstrap is exploratory with few tasks.')
