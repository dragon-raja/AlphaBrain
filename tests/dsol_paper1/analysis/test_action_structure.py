from AlphaBrain.research.dsol.analysis.action_structure import summarize_pairs,PANELS
from AlphaBrain.research.dsol.analysis.action_structure_evidence import rank_correlation


def test_missing_equal_and_unchanged_records_survive_pair_summary():
    rows=[]
    for panel in PANELS:
        for mode in ('include','exclude'):
            for start,end in ((1,3),(3,5)):
                for label,changed,task,value in [('improved',True,'a',2.),('degraded',True,'b',0.),('unknown',True,'c',1.),('equal',False,'d',0.)]:
                    rows.append(dict(panel=panel,mode=mode,start=start,end=end,outcome=label,changed=changed,task=task,added_delta={'feature':value}))
    g=summarize_pairs(rows)[0]
    assert g['n']==4 and g['changed']==3
    assert g['outcomes']==dict(improved=1,degraded=1,unknown=1,equal=1)
    assert g['features']['feature']['values']['unknown']==[1.]
    assert g['features']['feature']['concordance']==1.
    assert g['features']['feature']['leave_one_task_out']['a'] is None
    assert g['features']['feature']['leave_one_task_out']['b'] is None


def test_constant_or_missing_direction_is_not_fake_correlation():
    assert rank_correlation([1,1,1],[1,2,3]) is None
    assert rank_correlation([float('nan'),1],[2,3]) is None
    assert abs(rank_correlation([1,2,3],[3,2,1])+1)<1e-12
