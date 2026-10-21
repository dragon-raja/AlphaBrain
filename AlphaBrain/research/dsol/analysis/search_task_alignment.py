"""Paired task summaries; absent replay labels never become failures."""
from collections import defaultdict

import numpy as np

METHODS = ('stay', 'local_accel', 'ppo')


def cutoff_pair(row, limit):
    """Apply the SAME cutoff to both outcomes, preserving missing history."""
    if type(limit) is not int or limit <= 0:
        raise ValueError('Positive common horizon required')
    result = dict(row)
    result.update(success=bool(row['success'] and row['steps'] <= limit),
        replay_success=bool(row['replay_success'] and row['replay_completion_steps'] <= limit)
            if row.get('replay_success') is not None else None,common_horizon=limit)
    return result


def task_rows(rows):
    keys = [(r['cohort'],r['initial'],r['start'],r['method'],r['repeat']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate execution address')
    groups=defaultdict(list)
    for row in rows:
        if type(row['success']) is not bool:
            raise ValueError('Missing live outcome is not failure')
        groups[row['cohort'],row['task']].append(row)
    results=[]
    for (cohort,task),group in sorted(groups.items()):
        baseline={(r['initial'],r['start'],r['repeat']):r for r in group if r['method']=='stay'}
        for method in METHODS:
            selected=[r for r in group if r['method']==method]
            addresses={(r['initial'],r['start'],r['repeat']) for r in selected}
            if not selected or addresses!=set(baseline):
                raise ValueError('Incomplete paired method coverage')
            replay=[r for r in selected if r.get('replay_success') is not None]
            deltas=[int(r['success'])-int(baseline[r['initial'],r['start'],r['repeat']]['success']) for r in selected]
            harm=sum(x<0 for x in deltas);rescue=sum(x>0 for x in deltas)
            # Collapse correlated starts within each initial and execution seed.
            strata=defaultdict(list)
            for r,delta in zip(selected,deltas):strata[r['initial'],r['repeat']].append(delta)
            strata_gain=[float(np.mean(v)) for v in strata.values()]
            results.append(dict(cohort=cohort,task=task,method=method,n=len(selected),
                successes=sum(r['success'] for r in selected),success_rate=float(np.mean([r['success'] for r in selected])),
                queries=float(np.mean([r['queries'] for r in selected])),harm=harm,rescue=rescue,
                harm_rate=harm/len(selected),rescue_rate=rescue/len(selected),
                initial_seed_positive=sum(x>0 for x in strata_gain),initial_seed_negative=sum(x<0 for x in strata_gain),
                initial_seed_ties=sum(x==0 for x in strata_gain),
                replay_n=len(replay),replay_success_rate=float(np.mean([r['replay_success'] for r in replay])) if replay else None,
                replay_agreement=sum(r['success']==r['replay_success'] for r in replay),
                replay_pass_live_fail=sum(r['replay_success'] and not r['success'] for r in replay),
                replay_fail_live_pass=sum(not r['replay_success'] and r['success'] for r in replay)))
    return results


def paired_cluster_interval(rows, method='ppo', reference='local_accel', *, repeats=2000, seed=20261021):
    """Exploratory paired bootstrap: tasks then initials; retain seed/start means.

    Does not treat repeated starts, selected views, or eight execution seeds as
    independent tasks. Conditional on the measured seeds, not a population proof.
    """
    base={(r['task'],r['initial'],r['start'],r['repeat']):r for r in rows if r['method']==reference}
    strata=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r['method']!=method:continue
        key=(r['task'],r['initial'],r['start'],r['repeat'])
        if key not in base:raise ValueError('Unpaired bootstrap row')
        strata[r['task']][r['initial']].append(int(r['success'])-int(base[key]['success']))
    values=[np.array([np.mean(v) for v in initials.values()]) for initials in strata.values()]
    if not values:raise ValueError('No paired task groups')
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(repeats):
        samples.append(np.mean([rng.choice(values[i],len(values[i]),replace=True).mean()
                               for i in rng.integers(len(values),size=len(values))]))
    return dict(mean=float(np.mean([x.mean() for x in values])),
                interval95=np.quantile(samples,[.025,.975]).tolist(),tasks=len(values),
                scope='Exploratory task/initial bootstrap conditional on measured execution seeds')
