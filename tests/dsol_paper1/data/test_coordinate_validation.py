from collections import Counter
from AlphaBrain.research.dsol.data.coordinate_validation import planned_rows,CRITERIA
from AlphaBrain.research.dsol.data.native_recovery import limits,addresses


def test_complete_unseen_panel_balance_and_physical_envelope():
    states=[];records=[]
    for task in range(12):
        for initial in range(40,50):
            index=len(records)
            states.append(dict(initial=index,state_key=f'{task}-{initial}',task=f'task-{task}',
                suite=['libero_spatial','libero_object','libero_goal','libero_10'][task//3]))
            records.append(dict(state=dict(split='test_unseen_task',initialization=dict(init_state_index=initial))))
    rows=planned_rows(dict(states=states),records)
    assert len(rows)==2268 and limits(rows)['episodes']==4536
    assert len({str(a) for a in addresses(rows)})==4536
    assert Counter(r['method_family'] for r in rows)==dict(zip(
        ['G-seed-0','G-seed-1','G-seed-2','local','stay','home'],[432]*5+[108]))
    for row in rows:
        assert abs(row['angles_deg'][0])<=30 and abs(row['angles_deg'][1])<=20
        assert abs(row['angles_deg'][0]-row['mount_yaw_offset_deg'])<=15
    assert CRITERIA['primary_offsets']==[-15.,15.]
