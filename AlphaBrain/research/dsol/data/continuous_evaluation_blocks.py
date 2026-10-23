"""Fixed paired/nominal work units; quotas and identities do not depend on hosts."""
import math

METHODS=['stay','local_near_best_accel','continuous_ppo']
STAGES={'admission':[0,16], 'capacity':[1,2,3,4,5,6,17,18],
        'formal':list(range(7,16))+list(range(19,24))}


def blocks(plan,rows):
    if len(rows)!=48 or plan['evaluation']['methods']!=METHODS:
        raise ValueError('Frozen 48-search/three-method schedule required')
    result=[];nominal={};seen=set()
    for index in range(16):
        group=rows[index*3:index*3+3];first=group[0]
        identity=tuple(first[k] for k in ('cohort','initial','state_key','task','start'))
        if identity in seen:raise ValueError('Duplicate paired block')
        seen.add(identity)
        if [r['method'] for r in group]!=METHODS:raise ValueError('Method order differs')
        for row in group:
            if (tuple(row[k] for k in ('cohort','initial','state_key','task','start'))!=identity
                    or row['angles_deg']!=first['angles_deg'] or row['execution_repeats']!=[0,1,2,3]
                    or row['query_cap']!=(1 if row['method']=='stay' else 8)):
                raise ValueError('Broken paired identity, seed, pose or query budget')
        task=next(t for t in plan['evaluation']['tasks'] if t['task']==first['task'])
        per_episode=math.ceil(plan['evaluation']['horizons'][task['suite']]/5)
        result.append(dict(index=index,kind='paired',rows=group,tasks=12,searches=3,
                           limits=dict(score=34,execution=12*per_episode)))
        key=(first['cohort'],first['initial'])
        nominal.setdefault(key,dict(first,method='nominal',angles_deg=[0.,0.],start='nominal'))
    expected={(t['cohort'],i) for t in plan['evaluation']['tasks'] for i in t['indices']}
    if set(nominal)!=expected or len(nominal)!=8:raise ValueError('Nominal state coverage differs')
    for row in nominal.values():
        task=next(t for t in plan['evaluation']['tasks'] if t['task']==row['task'])
        result.append(dict(index=len(result),kind='nominal',rows=[row],tasks=4,searches=0,
            limits=dict(execution=4*math.ceil(plan['evaluation']['horizons'][task['suite']]/5))))
    totals=dict(score=sum(b['limits'].get('score',0) for b in result),
                execution=sum(b['limits']['execution'] for b in result if b['kind']=='paired'),
                nominal=sum(b['limits']['execution'] for b in result if b['kind']=='nominal'))
    if totals!=dict(score=544,execution=10560,nominal=1760):raise ValueError('Frozen quota changed')
    return result


def phase(index):
    return next(name for name,indices in STAGES.items() if index in indices)
