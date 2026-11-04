"""Task-weighted paired recovery, retaining improvements and additional harm."""
from collections import defaultdict
import numpy as np


def paired_method_contrast(rows, reference, *, seed=20261027, bootstrap_samples=5000):
    """Pair declared methods by policy seed and trial; never fill missing seeds."""
    def keyed(values):
        result={}
        for r in values:
            key=(int(r['method'].rsplit('-seed-',1)[1]),r['state_key'],r['view_id'],r['repeat'])
            if key in result or type(r['success']) is not bool:raise ValueError('Invalid paired comparison')
            result[key]=r
        return result
    a,b=keyed(rows),keyed(reference)
    if not a or a.keys()!=b.keys():raise ValueError('Exact seed/trial coverage required')
    tasks=sorted({r['task'] for r in a.values()});seeds=sorted({k[0] for k in a})
    differences=[]
    for s in seeds:
        values=[]
        for task in tasks:
            keys=[k for k,r in a.items() if k[0]==s and r['task']==task]
            if not keys or any(b[k]['task']!=task for k in keys):raise ValueError('Task/seed coverage differs')
            values.append(np.mean([int(a[k]['success'])-int(b[k]['success']) for k in keys]))
        differences.append(values)
    d=np.asarray(differences);rng=np.random.default_rng(seed)
    si=rng.integers(len(seeds),size=(bootstrap_samples,len(seeds)))
    ti=rng.integers(len(tasks),size=(bootstrap_samples,len(tasks)))
    draws=d[si[:,:,None],ti[:,None,:]].mean(axis=(1,2))
    return dict(pairs=len(a),tasks=len(tasks),policy_seeds=seeds,difference_pp=float(100*d.mean()),
        seed_differences_pp=(100*d.mean(axis=1)).tolist(),exploratory_task_seed_interval_pp=(100*np.quantile(draws,[.025,.975])).tolist(),
        interpretation='Matched seeds/trials; task and seed resampling. One seed does not quantify training variability; no equivalence claim.')


def paired_summary(rows, *, seed=20261024, bootstrap_samples=2000):
    """Descriptive point estimates and task-cluster bootstrap, not view IID CIs.

    Each row is a unique (task, initial, start, execution repeat, method/seed)
    pairing. The caller chooses one declared method and evaluation split.
    """
    if not rows:raise ValueError('Empty paired population')
    identities=[(r['task'],r['state_key'],r['view_id'],r['repeat'],r['method']) for r in rows]
    if len(set(identities))!=len(rows):raise ValueError('Repeated paired outcome')
    grouped=defaultdict(list)
    for r in rows:
        if any(type(r[k]) is not bool for k in ('success','stay_success','nominal_success')):
            raise ValueError('Complete Boolean outcomes required; missing is not failure')
        grouped[r['task']].append(r)
    tasks=[]
    for task,items in sorted(grouped.items()):
        y=np.array([r['success'] for r in items],dtype=float)
        stay=np.array([r['stay_success'] for r in items],dtype=float)
        nominal=np.array([r['nominal_success'] for r in items],dtype=float)
        tasks.append(dict(task=task,pairs=len(items),success=float(y.mean()),stay=float(stay.mean()),nominal=float(nominal.mean()),
            rescue=float(((y==1)&(stay==0)).mean()),harm=float(((y==0)&(stay==1)).mean()),
            net_recovery=float((y-stay).mean()),perturbation_loss=float((nominal-stay).mean())))
    fields=['success','stay','nominal','rescue','harm','net_recovery','perturbation_loss']
    point={k:float(np.mean([t[k] for t in tasks])) for k in fields}
    draws=np.random.default_rng(seed).integers(len(tasks),size=(bootstrap_samples,len(tasks)))
    intervals={k:np.quantile(np.array([t[k] for t in tasks])[draws].mean(axis=1),[.025,.975]).tolist() for k in fields}
    return dict(pairs=len(rows),task_count=len(tasks),task_macro=point,task_cluster_bootstrap_95=intervals,
        bootstrap_seed=seed,bootstrap_samples=bootstrap_samples,per_task=tasks,
        rescue_count=sum(r['success'] and not r['stay_success'] for r in rows),
        harm_count=sum(not r['success'] and r['stay_success'] for r in rows),
        interpretation='Tasks are the resampling clusters; descriptive for this fixed task/initial/angle panel.')
