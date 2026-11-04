"""Paired task-level development analysis; prefixes never choose their test set."""
from collections import defaultdict
import numpy as np


def endpoint_key(row):return (row['state_key'],row['start'],row['repeat'])


def compare_endpoint(arms, *, expected, bootstrap=10000, seed=20261102):
    """Require all three complete balanced arms before computing any contrast."""
    if set(arms)!={'stay','p3','full'}:raise ValueError('Complete Stay/P3/full endpoint required')
    indexed={}
    for arm,rows in arms.items():
        table={endpoint_key(r):r for r in rows}
        if len(table)!=len(rows) or set(table)!=set(expected):raise ValueError('Missing or duplicated endpoint')
        if any(type(r['success']) is not bool for r in rows):raise ValueError('Explicit operation success required')
        indexed[arm]=table
    keys=sorted(expected);tasks=sorted({indexed['stay'][k]['task'] for k in keys})
    for arm in arms:
        if any(indexed[arm][k]['task']!=indexed['stay'][k]['task'] for k in keys):raise ValueError('Mismatched paired task')
    result=dict(operations_per_arm=len(keys),tasks=len(tasks),rates={},contrasts={})
    for arm,table in indexed.items():
        result['rates'][arm]=dict(successes=sum(r['success'] for r in table.values()),success_rate=float(np.mean([r['success'] for r in table.values()])),
            mean_queries=float(np.mean([r['queries'] for r in table.values()])))
    rng=np.random.default_rng(seed);draws=rng.integers(0,len(tasks),size=(bootstrap,len(tasks)))
    for a,b in [('p3','stay'),('full','stay'),('full','p3')]:
        pairs=[(indexed[a][k]['success'],indexed[b][k]['success']) for k in keys]
        task_delta=np.array([np.mean([int(indexed[a][k]['success'])-int(indexed[b][k]['success']) for k in keys if indexed[a][k]['task']==task]) for task in tasks])
        result['contrasts'][a+'-minus-'+b]=dict(delta=float(task_delta.mean()),rescues=sum(x and not y for x,y in pairs),harms=sum(y and not x for x,y in pairs),
            task_bootstrap_95=np.quantile(task_delta[draws].mean(1),[.025,.975]).tolist(),task_deltas=dict(zip(tasks,task_delta.tolist())))
    return result


def learning_batch(update, episodes):
    """Interpret reward within its own score definition, not across prefixes."""
    number=update['update'];expected=set(range((number-1)*32,number*32))
    if len(episodes)!=32 or {r['episode'] for r in episodes}!=expected:raise ValueError('One complete 32-episode batch required')
    settled=update['reward_settlement']
    if (settled['kind']!='G' or settled['count']!=number*32 or not settled['frozen_during_optimizer']
            or len(settled['rewards'])!=32):raise ValueError('Original batch-frozen G reward required')
    return dict(update=number,episodes=number*32,mean_reward=float(np.mean(settled['rewards'])),scale=settled['denominator'],
        queries=float(np.mean([r['queries'] for r in episodes])),initial_stop_rate=float(np.mean([r['queries']==1 for r in episodes])),
        mean_raw_improvement=float(np.mean([r['initial_accel']-r['best_accel'] for r in episodes])),
        approximate_kl=update['means']['approximate_kl'],clip_fraction=update['means']['clip_fraction'],value_loss=update['means']['value_loss'])
