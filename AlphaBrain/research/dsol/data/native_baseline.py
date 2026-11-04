"""Full forty-task native baseline identities, independent of resource assignment."""
from collections import Counter
from .native_view_panel import views, limits, addresses

HORIZONS = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520)
SCHEMA = 'NATIVE_FULL40_BASELINE_V1'


def blocks(states):
    if len(states)!=400 or len({r['state_key'] for r in states})!=400:
        raise ValueError('Forty tasks times ten unique initial states required')
    tasks={}
    for r in states:
        if (r['suite'] not in HORIZONS or r['max_steps']!=HORIZONS[r['suite']]
                or r['cohort']!='A' or type(r['initial']) is not int or r['initial']<0):
            raise ValueError('Baseline state contract changed')
        tasks.setdefault((r['suite'],r['task']),[]).append(r)
    if len(tasks)!=40 or Counter(k[0] for k in tasks)!=dict.fromkeys(HORIZONS,10):
        raise ValueError('All forty tasks in four suites required')
    if any(len(rows)!=10 for rows in tasks.values()) or sorted(r['initial'] for r in states)!=list(range(400)):
        raise ValueError('Ten states per task and unique simulator indices required')
    result=[]
    for state in states:
        rows=[dict(state,view_id=i,angles_deg=angles,execution_repeats=[0,1],score_members=[0,1,2,3])
              for i,angles in enumerate(views())]
        for offset in range(0,len(rows),3):
            subset=rows[offset:offset+3]
            result.append(dict(index=len(result),kind='native_view_panel',rows=subset,
                               limits=limits(subset),addresses=addresses(subset)))
    total={k:sum(b['limits'][k] for b in result) for k in result[0]['limits']}
    if total!=dict(episodes=21600,score=43200,paired_execution=1425600,nominal_execution=0):
        raise ValueError('Full baseline budget changed')
    return result
