"""Predeclared development-only factorial-screen selection, not a test claim."""
from fractions import Fraction
from collections import defaultdict
from ..data.search_core import VARIANTS


def choose_structure(outcomes):
    if set(outcomes)!=set(VARIANTS):raise ValueError('All eight completed structures required')
    common=None;scores={};details={}
    for variant,rows in outcomes.items():
        groups=defaultdict(list);keys=set()
        for r in rows:
            if r['split']!='validation' or r['seed'] not in (0,1) or type(r['success']) is not bool:raise ValueError('Development labels only, two fixed seeds')
            key=(r['seed'],r['state_key'],r['view_id'],r['repeat'])
            if key in keys:raise ValueError('Duplicate paired outcome')
            keys.add(key);groups[r['task']].append(int(r['success']))
        if not keys or (common is not None and common!=keys):raise ValueError('Incomplete or different paired panels')
        common=keys
        if len(rows)!=128 or len(groups)!=8 or any(len(v)!=16 for v in groups.values()):raise ValueError('Expected 8 tasks x 16 paired outcomes')
        score=sum((Fraction(sum(v),len(v)) for v in groups.values()),Fraction(0))/len(groups)
        scores[variant]=score;details[variant]=dict(success=float(score),successes=sum(sum(v) for v in groups.values()),operations=len(rows))
    base=VARIANTS['U0']
    order=sorted(scores,key=lambda v:(-scores[v],sum(a!=b for a,b in zip(VARIANTS[v],base)),v))
    return dict(selected_base=order[0],ranking=order,variants=details,rule='equal_task_macro_success_two_seeds_then_fewer_changed_factors_then_lexical',split='validation',heldout_test_used=False)
