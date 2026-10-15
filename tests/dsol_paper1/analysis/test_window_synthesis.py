import numpy as np
import pytest

from AlphaBrain.research.dsol.analysis.window_synthesis import select, summarize, panel, signal_diagnostics


def test_cumulative_means_and_excluded_canonical():
    s=np.ones((3,8,5))
    s[0]=.5
    s[1,:,0]=0
    s[1,:,1:]=2
    s[2]=.8
    assert select(s).tolist()==[1,0,0,0,0]
    assert select(s,True).tolist()==[1,2,2,2,2]


def test_stable_catalog_ties_and_invalid_data():
    s=np.ones((3,8,5))
    assert select(s).tolist()==[0]*5
    assert select(s,True).tolist()==[1]*5
    s[0,0,0]=np.nan
    with pytest.raises(ValueError):select(s)


def test_identical_methods_paired_ci_exact_zero():
    base=np.array([[0.,1.,0.,1.],[1.,1.,1.,1.],[0.,0.,1.,1.],[1.,0.,1.,0.]])
    x=np.repeat(base[:,None,:],7,axis=1)
    rows=[dict(task=str(i//2),split='all') for i in range(4)]
    r=summarize(x,rows,draws=100)
    assert r['gain_ci95']==[[0.,0.]]*5
    assert r['mean']==[.625]*7


def test_missing_exclusion_is_not_complete_case_success():
    s=np.ones((3,8,5));s[0]=0;s[1]=.5
    y=np.array([[1.,1.,1.,1.],[np.nan]*4,[0.,0.,0.,0.]])
    r=panel([dict(task='a',split='all')],[s],[y],[[2]],[np.full((3,8),-1)],5)
    assert r['methods']['exclude']['coverage']==[0]*5
    assert not r['methods']['exclude']['scopes']
    assert r['methods']['exclude']['complete_window_means']==[None]*5
    assert r['methods']['include']['scopes']['all']['mean']==[0.,1.,1.,1.,1.,1.,1.]


def test_window1_noise_halves_and_success_step_boundary():
    s=np.broadcast_to(np.arange(3)[:,None,None],(3,8,5)).copy()
    first=np.full((3,8),-1);first[0]=5
    r=signal_diagnostics(s,first,5)
    assert r['half_agreement']==[True]*5
    assert r['post_success']==[0.,1.,1.,1.,1.]
