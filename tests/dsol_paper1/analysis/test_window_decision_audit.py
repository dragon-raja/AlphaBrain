import numpy as np
from AlphaBrain.research.dsol.analysis.window_decision_audit import decisions, report


def test_rank_invariant_to_positive_per_stage_affine_scaling():
    s=np.random.default_rng(4).random((6,8,5))
    a,_=decisions(s,False);b,_=decisions(s*np.array([1,8,.2,5,3])+np.arange(5),False)
    assert a[2]==b[2]


def test_strong_agreement_can_update_w1_without_labels():
    s=np.ones((3,8,5));s[0]=.5;s[1,:,0]=.8;s[1,:,1:]=0
    c,d=decisions(s,False)
    assert c[0]==0 and c[2]==c[3]==1 and d['accepted']


def test_half_disagreement_falls_back_to_w1():
    s=np.ones((3,8,5));s[0,:,0]=0;s[0,:,1:]=1
    s[1,:4,1:]=0;s[1,4:,1:]=.6
    s[2,:4,1:]=.7;s[2,4:,1:]=0
    c,d=decisions(s,False)
    assert not d['accepted'] and c[3]==c[0]


def test_missing_endpoint_not_imputed_or_dropped():
    s=np.ones((3,8,5));s[0]=.5;s[1,:,0]=.8;s[1,:,1:]=0
    y=np.array([[1.,0.],[np.nan,np.nan],[0.,1.]])
    r=report([s],[y],[dict(task='x',split='all')],False,dict(horizon=3,required_noise_votes=7))
    assert r['scopes']['all']['success'][0]==.5
    assert r['scopes']['all']['success'][3] is None
