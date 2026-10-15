import numpy as np
import pytest

from AlphaBrain.research.dsol.analysis.short_view_windows import window_choices,summarize


def test_k1_exact_and_later_windows_change_using_scores_only():
    scores=np.ones((2,3,8,5));scores[:,0,:,0]=0;scores[:,0,:,1:]=10
    scores[:,1]=2
    mean,picks=window_choices(scores)
    assert (picks[:,0]==0).all()
    assert (picks[:,1:]==2).all()
    for i in range(2):
        for j in range(3):assert mean[i,j,0]==scores[i,j,:,0].copy().mean()


def test_outcomes_never_change_selector_and_anchor_mismatch_rejected():
    rng=np.random.default_rng(1);a=rng.random((4,3,8,5));y=rng.integers(2,size=(4,3,4))
    means,picks=window_choices(a);tasks=np.array(['a','a','b','b']);splits=np.array(['pretrain','target']*2)
    args=(means[...,0],tasks,splits,np.array([1,2]),np.full((4,3,8),-1),25)
    first=summarize(a,y,*args);second=summarize(a,1-y,*args)
    np.testing.assert_array_equal(first[2],second[2]);np.testing.assert_allclose(first[3]+second[3],1)
    bad=means[...,0].copy();bad[0,0]+=1e-12
    with pytest.raises(ValueError):summarize(a,y,bad,*args[1:])


def test_incomplete_probe_is_not_imputed():
    scores=np.ones((2,3,8,5));scores[0,0,0,4]=np.nan
    with pytest.raises(ValueError):window_choices(scores)
