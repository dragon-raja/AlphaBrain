"""Within-state score validity on a complete, outcome-independent pose panel."""
from collections import defaultdict
import numpy as np


def ranks(values):
    a = np.asarray(values)
    return np.asarray([np.sum(a < v) + (np.sum(a == v) - 1) / 2 for v in a], float)


def correlation(a, b):
    a, b = ranks(a), ranks(b)
    if np.std(a) == 0 or np.std(b) == 0: return None
    return float(np.corrcoef(a, b)[0, 1])


def summarize(rows, *, expected_views=27, low_count=7, bootstrap_seed=20261029, score_names=('p3','p9')):
    if not score_names or len(set(score_names))!=len(score_names):
        raise ValueError('Distinct explicit score fields required')
    grouped = defaultdict(list)
    for row in rows: grouped[row['state_key']].append(row)
    if not grouped: raise ValueError('Empty spatial panel')
    per_state = []
    for state, values in sorted(grouped.items()):
        if len(values) != expected_views or {v['view_id'] for v in values} != set(range(expected_views)):
            raise ValueError('Incomplete or duplicated spatial grid')
        for v in values:
            if len(v['success']) != 2 or any(type(x) is not bool for x in v['success']):
                raise ValueError('Exactly two explicit operation outcomes required')
            if any(len(v[k]) != 2 or not np.isfinite(v[k]).all() for k in score_names):
                raise ValueError('Exactly two finite score members required')
        for exclude_reference in (False, True):
            selected = sorted((v for v in values if not exclude_reference or v['view_id'] != 0),
                              key=lambda v: v['view_id'])
            if len(selected) < low_count: raise ValueError('Insufficient low-score group size')
            success = np.asarray([np.mean(v['success']) for v in selected])
            for score_name in score_names:
                scores = np.asarray([v[score_name] for v in selected])
                order = np.argsort(scores.mean(axis=1), kind='stable')
                member_orders = [np.argsort(scores[:, i], kind='stable') for i in (0, 1)]
                low_success = float(success[order[:low_count]].mean())
                per_state.append(dict(state_key=state, task=values[0]['task'], score=score_name,
                    exclude_reference=exclude_reference, views=len(selected),
                    spearman=correlation(scores.mean(axis=1), success),
                    low_group_success=low_success, uniform_success=float(success.mean()),
                    low_group_gain=low_success-float(success.mean()),
                    argmin_success=float(success[order[0]]),
                    argmin_view=selected[order[0]]['view_id'],
                    low_views=[selected[i]['view_id'] for i in order[:low_count]],
                    member_argmin_agrees=bool(member_orders[0][0] == member_orders[1][0]),
                    member_rank_correlation=correlation(scores[:, 0], scores[:, 1])))
    aggregate = []
    for score in score_names:
        for exclude in (False, True):
            subset = [x for x in per_state if x['score'] == score and x['exclude_reference'] == exclude]
            tasks = defaultdict(list)
            for row in subset: tasks[row['task']].append(row['low_group_gain'])
            gains = np.asarray([np.mean(x) for _, x in sorted(tasks.items())])
            rng = np.random.default_rng(bootstrap_seed)
            draws = rng.choice(gains, size=(5000, len(gains)), replace=True).mean(axis=1)
            aggregate.append(dict(score=score, exclude_reference=exclude, tasks=len(tasks),
                states=len(subset), low_group_gain_task_mean=float(gains.mean()),
                task_cluster_95_interval=np.quantile(draws, [.025, .975]).tolist(),
                argmin_success_state_mean=float(np.mean([x['argmin_success'] for x in subset])),
                member_argmin_agreement=float(np.mean([x['member_argmin_agrees'] for x in subset]))))
    return dict(status='COMPLETE', positions=len(rows), per_state=per_state, aggregate=aggregate,
                scope='Development fixed-grid diagnostic; two operation repeats, no continuous-domain or held-out-task claim')
