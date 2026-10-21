"""Pair bounded live tasks with the same historical endpoint/noise labels."""

import numpy as np


def summarize_live(receipt):
    if receipt.get('status') != 'COMPLETE':
        raise ValueError('Complete live receipt required; missing tasks are not failures')
    tasks=receipt['tasks'];methods=('stay','local_accel','ppo')
    expected={(i,m,s) for i in (0,4) for m in methods for s in range(4)}
    keyed={(r['initial'],r['method'],r['repeat']):r for r in tasks}
    if len(tasks)!=len(expected) or set(keyed)!=expected:
        raise ValueError('Expected unique two-case, three-method, four-seed coverage')
    for r in tasks:
        if any(type(r.get(k)) is not bool for k in ('success','historical_same_seed_success')):
            raise ValueError('Both live and historical labels must exist as booleans')
        if not 1<=r['queries']<=8:
            raise ValueError('Query count outside the frozen budget')
    rows=[]
    for method in methods:
        selected=[r for r in tasks if r['method']==method]
        for initial in (0,4):
            group=[r for r in selected if r['initial']==initial]
            if len({(r['selected'],r['queries']) for r in group})!=1:
                raise ValueError('Execution repeats must share one fixed search result')
        rows.append(dict(method=method,trials=len(selected),
            live_successes=sum(r['success'] for r in selected),
            historical_successes=sum(r['historical_same_seed_success'] for r in selected),
            agreement=sum(r['success']==r['historical_same_seed_success'] for r in selected),
            historical_fail_live_pass=sum(r['success'] and not r['historical_same_seed_success'] for r in selected),
            historical_pass_live_fail=sum(not r['success'] and r['historical_same_seed_success'] for r in selected),
            mean_queries=float(np.mean([r['queries'] for r in selected])),
            mean_steps=float(np.mean([r['steps'] for r in selected])),
            cases=[dict(initial=i,selected=keyed[i,method,0]['selected'],queries=keyed[i,method,0]['queries'],
                live_successes=sum(keyed[i,method,s]['success'] for s in range(4)),
                historical_successes=sum(keyed[i,method,s]['historical_same_seed_success'] for s in range(4)))
                for i in (0,4)]))
    return dict(rows=rows,matched_tasks=len(tasks),
                scope='Same two development cases, selected views and four execution-noise seeds; not the 16-initial evaluation population')
