"""CPU log audit and coverage-aware paired interim statistics.

Missing operations are never failures. Completion order never chooses tasks by
their labels. Reward is reconstructed from the batch-frozen optimizer receipt,
not the provisional collector reward stored beside raw rollout metrics.
"""
from collections import defaultdict
import numpy as np
from .prefix_comparison import learning_batch


def training_health(updates, episodes, *, expected_updates=64):
    if (type(expected_updates) is not int or expected_updates < 8
            or len(updates) != expected_updates or len(episodes) != 32*expected_updates):
        raise ValueError('Complete explicitly declared update/episode endpoint required')
    points = []; reward_error = scale_error = 0.
    for index, update in enumerate(updates, 1):
        if update['update'] != index:
            raise ValueError('Repeated or missing optimizer update')
        batch = episodes[(index-1)*32:index*32]
        point = learning_batch(update, batch)
        receipt = update['reward_settlement']
        scale = float(np.median([r['initial_accel'] for r in episodes[:index*32]]))
        scale_error = max(scale_error, abs(scale-receipt['denominator']))
        expected = [(r['initial_accel']-r['best_accel'])/receipt['denominator']-.01*(r['queries']-1) for r in batch]
        reward_error = max(reward_error, float(np.max(np.abs(np.array(expected)-receipt['rewards']))))
        for row in batch:
            if (not 1 <= row['queries'] <= 8 or row['scoring_trajectories'] != 2*row['queries']
                    or not 0 <= row['best_accel'] <= row['initial_accel']):
                raise ValueError('Invalid raw score/query accounting')
        point.update(update['means'])
        point['mean_initial_accel'] = float(np.mean([r['initial_accel'] for r in batch]))
        point['mean_best_accel'] = float(np.mean([r['best_accel'] for r in batch]))
        point['at_cap_fraction'] = float(np.mean([r['queries'] == 8 for r in batch]))
        point['normalized_gain'] = float(np.mean(expected))+.01*(point['queries']-1)
        point['query_cost'] = .01*(point['queries']-1)
        if not all(np.isfinite(v) for v in point.values()):
            raise ValueError('Nonfinite optimizer log')
        points.append(point)
    if reward_error > 1e-9 or scale_error > 1e-12:
        raise ValueError('Settled G reward or cumulative initial-score scale differs')
    metrics = [k for k in points[0] if k not in ('episodes', 'update')]
    windows = {name: {k: float(np.mean([p[k] for p in subset])) for k in metrics}
               for name, subset in [('first8', points[:8]), ('last8', points[-8:]), ('last16', points[-16:])]}
    return dict(points=points, windows=windows, max_reward_error=reward_error,
        max_scale_error=scale_error, finite_optimizer_logs=True,
        maximum_kl=max(p['approximate_kl'] for p in points),
        maximum_clip_fraction=max(p['clip_fraction'] for p in points),
        maximum_gradient_norm=max(p['gradient_norm'] for p in points),
        reward_source='reward_settlement.rewards after online G assignment; collector total_reward is provisional',
        scope='JSON log-level consistency, not a replay of all saved tensors, GAE or gradients')


def physical_key(row):
    return row['state_key'], int(row['view_id']), int(row['repeat'])


def paired_diagnostic(rows):
    if not rows:
        return dict(status='PENDING_MATCHED_OPERATIONS', operations=0)
    keys = [(int(r['seed']), *physical_key(r)) for r in rows]
    if len(set(keys)) != len(keys) or any(type(r[k]) is not bool for r in rows for k in ('success','stay_success')):
        raise ValueError('Repeated or invalid paired outcome')
    tasks = defaultdict(list)
    for r in rows: tasks[r['suite'], r['task']].append(r)
    result = dict(status='DESCRIPTIVE_PARTIAL', operations=len(rows), tasks=len(tasks),
        seeds=sorted({r['seed'] for r in rows}),
        policy_success=float(np.mean([r['success'] for r in rows])),
        stay_success=float(np.mean([r['stay_success'] for r in rows])),
        rescues=sum(r['success'] and not r['stay_success'] for r in rows),
        harms=sum(not r['success'] and r['stay_success'] for r in rows),
        mean_queries=float(np.mean([r['queries'] for r in rows])),
        initial_only_fraction=float(np.mean([r['queries']==1 for r in rows])),
        query_cap_fraction=float(np.mean([r['queries']==8 for r in rows])))
    result['net_pp'] = 100*(result['policy_success']-result['stay_success'])
    for name, field in [('reference_success','nominal_success'), ('p3_success','p3_success')]:
        result[name] = float(np.mean([r[field] for r in rows])) if all(r.get(field) is not None for r in rows) else None
    result['pfull_minus_p3_pp'] = (100*(result['policy_success']-result['p3_success'])
                                  if result['p3_success'] is not None else None)
    task_values = [dict(suite=suite, task=task, operations=len(items),
        policy_success=float(np.mean([r['success'] for r in items])),
        stay_success=float(np.mean([r['stay_success'] for r in items])),
        net_pp=100*float(np.mean([int(r['success'])-int(r['stay_success']) for r in items])))
        for (suite, task),items in sorted(tasks.items())]
    result['per_task'] = task_values
    result['task_macro_net_pp'] = float(np.mean([t['net_pp'] for t in task_values]))
    if len(tasks) >= 5:
        values = np.array([t['net_pp'] for t in task_values]);rng=np.random.default_rng(20261103)
        draws = rng.integers(len(values),size=(5000,len(values)))
        result['exploratory_task_cluster_interval_pp'] = np.quantile(values[draws].mean(1),[.025,.975]).tolist()
    else:
        result['exploratory_task_cluster_interval_pp'] = None
    return result


def complete_three_seed_tasks(rows, expected):
    """Keep tasks only when every frozen outcome is present in all three seeds."""
    actual = {(r['seed'], *physical_key(r)):r for r in rows}
    if len(actual) != len(rows): raise ValueError('Duplicate observed operation')
    tasks = defaultdict(set)
    for r in expected: tasks[r['suite'],r['task']].add((r['seed'],*physical_key(r)))
    if not set(actual).issubset(set().union(*tasks.values())):
        raise ValueError('Unplanned operation')
    complete = {t for t, keys in tasks.items() if keys.issubset(actual)
                and {k[0] for k in keys} == {0,1,2}}
    return [r for r in rows if (r['suite'],r['task']) in complete], sorted(complete)


def development_progress(rows):
    result={}
    for seed in range(3):
        arms={v:{physical_key(r):r for r in rows if r['seed']==seed and r['checkpoint_version']==v} for v in (16,32)}
        if any(len(arm)!=64 for arm in arms.values()) or arms[16].keys()!=arms[32].keys():
            raise ValueError('Same complete 64-operation development panel required')
        points=[]
        for version,arm in arms.items():
            values=list(arm.values())
            points.append(dict(episodes=version*32,operations=64,successes=sum(r['success'] for r in values),
                success_rate=float(np.mean([r['success'] for r in values])),mean_queries=float(np.mean([r['queries'] for r in values]))))
        keys=arms[16].keys()
        result[str(seed)]=dict(points=points,rescues=sum(arms[32][k]['success'] and not arms[16][k]['success'] for k in keys),
            harms=sum(arms[16][k]['success'] and not arms[32][k]['success'] for k in keys),
            task_count=len({r['task'] for r in arms[16].values()}))
    return result
