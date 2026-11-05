"""Development-only comparisons for a frozen shared-core decision.

Policy seeds share task/initial/start/noise identities. Do not treat repeated
policies as new scenes or use these exploratory intervals as selection tests.
"""
from collections import defaultdict

import numpy as np

from .panel_uncertainty import paired_panel_intervals


def identity(row):
    return (row['seed'], row['suite'], row['task'], row['state_key'],
            row['view_id'], row['repeat'])


def index_panel(rows):
    panel = {identity(r): r for r in rows}
    if not panel or len(panel) != len(rows):
        raise ValueError('Empty or duplicated paired panel')
    for r in rows:
        if (r['split'] != 'validation' or type(r['success']) is not bool
                or not 1 <= r['queries'] <= r['query_cap']):
            raise ValueError('Complete development outcomes and explicit cap required')
    return panel


def summarize(rows):
    index_panel(rows)
    seeds = sorted({r['seed'] for r in rows})
    rates = [np.mean([r['success'] for r in rows if r['seed'] == s]) for s in seeds]
    return dict(operations=len(rows), successes=sum(r['success'] for r in rows),
                success_rate=float(np.mean(rates)), seed_rates=rates,
                seed_sd=float(np.std(rates, ddof=1)) if len(seeds) > 1 else None,
                mean_queries=float(np.mean([r['queries'] for r in rows])),
                initial_only_fraction=float(np.mean([r['queries'] == 1 for r in rows])),
                cap_fraction=float(np.mean([r['queries'] == r['query_cap'] for r in rows])),
                mean_initial_accel=float(np.mean([r['search_initial_accel'] for r in rows])),
                mean_best_accel=float(np.mean([r['search_best_accel'] for r in rows])))


def contrast(left, right, *, draws=20000):
    a, b = index_panel(left), index_panel(right)
    if a.keys() != b.keys():
        raise ValueError('Compared panels have different physical identities')
    for k in a:
        for field in ('horizon', 'replan_steps', 'settle_steps', 'angles_deg'):
            if a[k][field] != b[k][field]:
                raise ValueError('Changed paired operation protocol')
        x, y = a[k]['execution_noise_sha256'], b[k]['execution_noise_sha256']
        n = min(len(x), len(y))
        if not n or x[:n] != y[:n]:
            raise ValueError('Changed paired execution noise')
    seeds = sorted({k[0] for k in a})
    tasks = sorted({k[1:3] for k in a})
    arrays = []
    for panel in (a, b):
        values = []
        for seed in seeds:
            by_task = []
            for task in tasks:
                states = defaultdict(list)
                for key, r in panel.items():
                    if key[0] == seed and key[1:3] == task:
                        states[key[3]].append(float(r['success']))
                by_task.append([np.mean(states[s]) for s in sorted(states)])
            values.append(by_task)
        arrays.append(np.asarray(values, float))
    result = paired_panel_intervals(*arrays, draws=draws)
    result.update(rescues=sum(a[k]['success'] and not b[k]['success'] for k in a),
                  harms=sum(b[k]['success'] and not a[k]['success'] for k in a),
                  query_difference=float(np.mean([a[k]['queries'] - b[k]['queries'] for k in a])),
                  task_order=[list(t) for t in tasks])
    return result
