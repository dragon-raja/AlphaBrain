"""Paired development selection screen; reused endpoints are not new episodes.

This consumes operations at predeclared starts/reference/selected endpoint,
not a dense success surface. It never chooses a score rule from outcome labels.
"""
from collections import defaultdict
import numpy as np


def summarize(tasks,*,expected_states,expected_tasks,seed=20261102):
    variants=('reference','stay-23','stay-24','minimum-p3');indexed={};groups=defaultdict(list)
    for row in tasks:
        key=(row['state_key'],row['variant'],row['repeat'])
        if key in indexed or row['variant'] not in variants or row['repeat'] not in (0,1) or type(row['success']) is not bool:
            raise ValueError('Duplicate or invalid selection operation')
        indexed[key]=row
        groups[row['state_key']].append(row)
    if len(groups)!=expected_states or len(tasks)!=8*expected_states:raise ValueError('Incomplete selection screen')
    states=[];task_deltas=defaultdict(list)
    for state,values in sorted(groups.items()):
        if {(v['variant'],v['repeat']) for v in values}!={(v,m) for v in variants for m in (0,1)}:
            raise ValueError('Incomplete paired variant/repeat coverage')
        methods={v:[indexed[(state,v,m)]['success'] for m in (0,1)] for v in variants}
        rescued=harmed=before=after=0
        for v in ('stay-23','stay-24'):
            for m in (0,1):
                a,b=methods[v][m],methods['minimum-p3'][m]
                before+=a;after+=b;rescued+=not a and b;harmed+=a and not b
        task='::'.join(state.split('::')[:-1]);delta=(after-before)/4
        task_deltas[task].append(delta)
        states.append(dict(state_key=state,task=task,success=methods,rescued=int(rescued),harmed=int(harmed),
                           before=int(before),after=int(after),net_pp=100*delta))
    if len(task_deltas)!=expected_tasks:raise ValueError('Unexpected task distribution')
    deltas=np.array([np.mean(x) for _,x in sorted(task_deltas.items())])
    samples=np.random.default_rng(seed).choice(deltas,size=(10000,len(deltas)),replace=True).mean(axis=1)
    rates={v:float(np.mean([x['success'][v] for x in states])) for v in variants}
    rates['pooled_stay']=(rates['stay-23']+rates['stay-24'])/2
    return dict(states=expected_states,tasks=expected_tasks,unique_operations=len(tasks),rates=rates,
        paired_comparisons=4*expected_states,selected_unique_operations=2*expected_states,
        rescued=sum(x['rescued'] for x in states),harmed=sum(x['harmed'] for x in states),
        net_pp=float(100*deltas.mean()),task_cluster_interval_pp=(100*np.quantile(samples,[.025,.975])).tolist(),
        per_state=states,endpoint_reuse='Each selected outcome paired to two starts; counted once as an operation. Bootstrap unit is task.',
        scope='Development candidate selection, no PPO; starts [-30,+20] and [-30,-20] only. No full spatial success curve.')
