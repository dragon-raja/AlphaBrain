import numpy as np
from AlphaBrain.research.dsol.analysis.action_consensus import describe, aggregate


def test_missing_endpoints_not_reported_as_complete_success():
    a=np.zeros((4,8,5,7));a[1]=1;a[2]=2;a[3]=10
    s=np.ones((4,8,5));y=np.array([[0,0],[np.nan,np.nan],[1,1],[0,0]])
    r=describe(a,s,y,['a','b','c','d'],dict(initial=0,task='x',split='s'),5)
    assert r['signals']['chunk_translation']['winner']=='b'
    assert r['signals']['chunk_translation']['winner_success'] is None
    agg=aggregate([r])['all']['chunk_translation']
    assert agg['winner_coverage']==0 and agg['winner_success'] is None


def test_consensus_is_not_correctness():
    a=np.zeros((4,8,5,7));a[1]=1;a[2]=2;a[3]=10
    s=np.ones((4,8,5));y=np.array([[0,0],[0,0],[0,0],[1,1]])
    r=describe(a,s,y,['a','b','c','d'],dict(initial=0,task='x',split='s'),5)
    assert r['signals']['chunk_translation']['rho_agreement_success']<0
    assert r['signals']['chunk_translation']['winner_success']==0
