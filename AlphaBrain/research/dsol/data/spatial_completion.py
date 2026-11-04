"""Outcome-independent completion of an immutable spatial operation lattice.

Only verified prior rows may enter. Identical scientific trials can have several
historical arm names; all duplicates must agree before deterministic reuse.
"""
from copy import deepcopy


def key(row):
    return row['state_key'],row['view_id'],row['repeat']


def plan(expected,prior,*,maximum_replans):
    if type(maximum_replans) is not int or maximum_replans<1:raise ValueError('Bounded native operation horizon required')
    planned={key(r):r for r in expected}
    if not planned or len(planned)!=len(expected):raise ValueError('Empty or duplicate expected spatial trials')
    sources={};aliases={}
    for r in sorted(prior,key=lambda r:r['source_result']):
        k=key(r)
        if k not in planned or type(r['success']) is not bool or not r.get('trajectory_identity'):
            raise ValueError('Unplanned, unknown or unverified prior trial')
        if k in sources:
            a=sources[k]
            if any(a[x]!=r[x] for x in ('success','steps','replans','trajectory_identity')):
                raise ValueError('Repeated scientific trial disagrees; explicit audit required')
            aliases[k].append(r['source_result'])
        else:sources[k]=r;aliases[k]=[r['source_result']]
    reuse=[dict(deepcopy(planned[k]),prior=deepcopy(sources[k]),aliases=aliases[k]) for k in planned if k in sources]
    missing=[deepcopy(r) for k,r in planned.items() if k not in sources]
    return dict(expected=deepcopy(expected),reuse=reuse,missing=missing,
        counts=dict(expected=len(expected),prior_actual=len(prior),reused=len(reuse),duplicates=len(prior)-len(reuse),new_operations=len(missing)),
        limits=dict(episodes=len(missing),score=0,paired_execution=len(missing)*maximum_replans,nominal_execution=0))
