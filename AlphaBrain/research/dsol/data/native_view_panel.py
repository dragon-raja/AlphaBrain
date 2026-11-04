"""Scientific identities for the approved native fixed-state response panel."""
from collections import Counter
import math

from ..execution.manifest import digest


def views():
    result = [[0., 0.]]
    for level in (1, 2, 3):
        y, p = 10.*level, 20.*level/3
        result += [[y,0],[-y,0],[0,p],[0,-p],[y,p],[y,-p],[-y,p],[-y,-p]]
    return result + [[-17.5,9.5],[17.5,-9.5]]


def panel_rows(states):
    if len(states)!=8 or len({s['state_key'] for s in states})!=8:
        raise ValueError('Exactly eight unique frozen states required')
    result=[]
    for state in states:
        for index, angles in enumerate(views()):
            reuse = index in (0,25,26)
            result.append(dict(state,view_id=index,angles_deg=angles,
                execution_repeats=[] if reuse else [0,1,2,3],
                score_members=[2,3] if index in (25,26) else [0,1,2,3]))
    return result


def addresses(rows):
    result=[]
    for row in rows:
        base=dict(model='native_reference',task=row['task'],initial=row['state_key'],view=f"panel-{row['view_id']:02d}")
        result += [dict(base,stream='execution',repeat=s,noise=f'paired-seed-{s}') for s in row['execution_repeats']]
        result += [dict(base,stream='score',repeat=s,noise=f'panel-score-{s}') for s in row['score_members']]
    return result


def limits(rows):
    return dict(episodes=sum(len(r['execution_repeats']) for r in rows),
        score=sum(len(r['score_members']) for r in rows),
        paired_execution=sum(len(r['execution_repeats'])*math.ceil(r['max_steps']/5) for r in rows),
        nominal_execution=0)


def blocks(states):
    rows=panel_rows(states);result=[]
    for index in range(72):
        subset=rows[index*3:index*3+3]
        result.append(dict(index=index,kind='native_view_panel',rows=subset,
            limits=limits(subset),addresses=addresses(subset)))
    total={k:sum(b['limits'][k] for b in result) for k in result[0]['limits']}
    if total!=dict(episodes=768,score=832,paired_execution=42240,nominal_execution=0):
        raise ValueError('Approved panel budget changed')
    all_addresses=[digest(a) for b in result for a in b['addresses']]
    if len(set(all_addresses))!=len(all_addresses):
        raise ValueError('Repeated scientific address')
    return result


def validate_coverage(rows):
    expected={(r['state_key'],r['view_id']) for r in panel_rows(
        list({r['state_key']:r for r in rows}.values()))}
    found=Counter((r['state_key'],r['view_id']) for r in rows)
    if set(found)!=expected or set(found.values())!={1}:
        raise ValueError('Missing or duplicate panel position')
