"""Complete native-view panel diagnostics, without fitting or tuning a policy."""
from collections import defaultdict
import numpy as np
from .native_view_response import rank
from .recovery_summary import paired_summary


def analyze(panel):
    grouped = defaultdict(list)
    for row in panel:
        if (len(row['success']) != 2 or any(type(x) is not bool for x in row['success'])
                or not np.isfinite([row['accel'], row['independent_accel']]).all()):
            raise ValueError('Two Boolean outcomes and finite scores required')
        grouped[row['state_key']].append(row)
    if not grouped:
        raise ValueError('Empty panel')
    pairs = []; selected = defaultdict(list); states = []; by_view = defaultdict(list)
    for key, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: r['view_id'])
        if [r['view_id'] for r in rows] != list(range(27)):
            raise ValueError('Complete unique 27-view surface required')
        if len({(r['task'], r['suite'], r['split']) for r in rows}) != 1:
            raise ValueError('Mixed state identity')
        y = np.array([r['success'] for r in rows], dtype=bool)
        scores = np.array([r['accel'] for r in rows]); rates = y.mean(axis=1)
        a, b = rank(scores), rank(rates)
        rho = float(np.corrcoef(a,b)[0,1]) if np.std(a) and np.std(b) else None
        for row in rows:
            by_view[row['view_id']].append(row)
            if row['view_id'] == 0: continue
            for repeat in (0,1):
                pairs.append(dict(task=row['task'], state_key=key, suite=row['suite'], split=row['split'],
                    view_id=row['view_id'], repeat=repeat, method='perturbed_stay',
                    success=row['success'][repeat], stay_success=row['success'][repeat],
                    nominal_success=rows[0]['success'][repeat]))
        choices = dict(min_accel_27=int(np.argmin(scores)), min_accel_26=1+int(np.argmin(scores[1:])))
        for method, index in choices.items():
            for repeat in (0,1):
                selected[method].append(dict(task=rows[0]['task'], state_key=key, view_id=index,
                    repeat=repeat, method=method, success=bool(y[index,repeat]),
                    stay_success=bool(y[0,repeat]), nominal_success=bool(y[0,repeat])))
        order = np.argsort(scores, kind='stable')
        states.append(dict(state_key=key, task=rows[0]['task'], suite=rows[0]['suite'], split=rows[0]['split'],
            nominal=float(rates[0]), perturbed=float(rates[1:].mean()),
            nominal_accel=float(scores[0]), perturbed_accel=float(scores[1:].mean()),
            spearman=rho, low7_success=float(rates[order[:7]].mean()), high7_success=float(rates[order[-7:]].mean()),
            selected_views=choices,
            selected_independent_accel=float(rows[choices['min_accel_27']]['independent_accel']),
            nominal_independent_accel=float(rows[0]['independent_accel']),
            nominal_success_count=int(y[0].sum()), perturbed_success_count=int(y[1:].sum())))
    summary = paired_summary(pairs)
    per_task = []
    for task in sorted({r['task'] for r in states}):
        items = [s for s in states if s['task'] == task]
        per_task.append(dict(task=task, suite=items[0]['suite'], split=items[0]['split'], states=len(items),
            nominal=float(np.mean([s['nominal'] for s in items])),
            perturbed=float(np.mean([s['perturbed'] for s in items])),
            loss=float(np.mean([s['nominal']-s['perturbed'] for s in items]))))
    groups = {}
    for field in ('suite', 'split'):
        for name in sorted({r[field] for r in pairs}):
            groups[field+'/'+name] = paired_summary([r for r in pairs if r[field] == name])
    levels = {'small':range(1,9), 'medium':range(9,17), 'large':range(17,25), 'legacy_two':(25,26)}
    for name, ids in levels.items():
        groups['severity/'+name] = paired_summary([r for r in pairs if r['view_id'] in ids])
    task_correlations = []
    for task in sorted({s['task'] for s in states}):
        values = [s['spearman'] for s in states if s['task'] == task and s['spearman'] is not None]
        if values: task_correlations.append(float(np.mean(values)))
    correlation = dict(states_defined=sum(s['spearman'] is not None for s in states), states_total=len(states),
        task_mean_spearman=float(np.mean(task_correlations)) if task_correlations else None,
        state_median_spearman=float(np.median([s['spearman'] for s in states if s['spearman'] is not None])) if task_correlations else None,
        interpretation='Within-state rank correlation across views; negative supports lower score with higher success. Constant-outcome states excluded, not independent task samples.')
    nominal = [r for r in panel if r['view_id'] == 0]
    harm = sum(r['nominal_success'] and not r['success'] for r in pairs)
    rescue = sum(not r['nominal_success'] and r['success'] for r in pairs)
    return dict(status='COMPLETE_BASELINE_DIAGNOSTIC', tasks=len(per_task), states=len(states),
        nominal_episodes=2*len(nominal), nominal_successes=sum(sum(r['success']) for r in nominal),
        perturbed_episodes=len(pairs), perturbed_successes=sum(r['success'] for r in pairs),
        perturbation_harm_pairs=harm, perturbation_improvement_pairs=rescue,
        dense=summary, groups=groups, per_task=per_task, per_state=states,
        per_view=[dict(view_id=v, yaw_deg=rr[0]['yaw_deg'], pitch_deg=rr[0]['pitch_deg'],
                      success_rate=float(np.mean([np.mean(r['success']) for r in rr])),
                      mean_accel=float(np.mean([r['accel'] for r in rr]))) for v,rr in sorted(by_view.items())],
        correlation=correlation, offline_selectors={m:paired_summary(rr) for m,rr in selected.items()},
        offline_selector_scope='Each state queries all 27 or 26 candidates using score members 0/1 and chooses before reading success. Not an 8-query online recovery method; no policy fitting or evaluation tuning.')
