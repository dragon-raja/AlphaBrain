from AlphaBrain.research.dsol.data.coordinate_probe import planned_rows
from AlphaBrain.research.dsol.data.native_recovery import limits, addresses


def test_panel_is_balanced_and_independent_of_outcomes():
    states=[];records=[]
    for suite in ('libero_spatial','libero_object','libero_goal','libero_10'):
        for task in ('a','b'):
            for initial in (20,21,22):
                i=len(records)
                states.append(dict(initial=i,state_key=f'{suite}-{task}-{initial}',task=f'{suite}-{task}',suite=suite))
                records.append(dict(state=dict(split='validation',initialization=dict(init_state_index=initial))))
    rows=planned_rows(dict(states=states),records)
    assert len(rows)==144 and len({(r['state_key'],r['view_id'],r['method']) for r in rows})==144
    assert len({r['state_key'] for r in rows})==8
    assert len(addresses(rows))==288 and limits(rows)['score']==2592
    assert all(r['task'].endswith('-a') for r in rows)
    for r in rows:
        mechanical=r['angles_deg'][0]-r['mount_yaw_offset_deg']
        assert abs(mechanical)<=25
        assert -30<=r['mount_yaw_offset_deg']-25<r['mount_yaw_offset_deg']+25<=30
    for state in states:state['success']=False
    assert planned_rows(dict(states=[{k:v for k,v in s.items() if k!='success'} for s in states]),records)==rows
