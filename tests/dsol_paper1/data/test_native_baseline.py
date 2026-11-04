from copy import deepcopy
import pytest
from AlphaBrain.research.dsol.data.native_baseline import blocks,HORIZONS


def states():
    return [dict(cohort='A',initial=j*100+t*10+i,state_key=f'{s}/{t}/{i}',
        task=f'{s}_{t}',suite=s,max_steps=h) for j,(s,h) in enumerate(HORIZONS.items())
        for t in range(10) for i in range(10)]


def test_complete_dense_baseline_is_partitioned_without_changing_identity():
    result=blocks(states());assert len(result)==3600
    rows=[r for b in result for r in b['rows']]
    assert len(rows)==10800 and len({(r['state_key'],r['view_id']) for r in rows})==10800
    assert all(r['execution_repeats']==[0,1] and r['score_members']==[0,1,2,3] for r in rows)
    assert sum(b['limits']['score']+b['limits']['paired_execution'] for b in result)==1468800


@pytest.mark.parametrize('fault',['missing','duplicate','horizon','suite','index'])
def test_missing_changed_or_duplicated_scientific_cells_refused(fault):
    rows=deepcopy(states())
    if fault=='missing':rows.pop()
    if fault=='duplicate':rows[0]=rows[1]
    if fault=='horizon':rows[0]['max_steps']=100
    if fault=='suite':rows[0]['suite']='other'
    if fault=='index':rows[0]['initial']=1
    with pytest.raises(ValueError):blocks(rows)
