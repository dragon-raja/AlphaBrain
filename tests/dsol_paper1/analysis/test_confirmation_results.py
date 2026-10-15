import numpy as np
import pytest

from AlphaBrain.research.dsol.analysis.confirmation_results import condition_outcomes, paired_bootstrap, summarize_population


def test_deduplication_and_disturbance_weighting():
    y=np.full((2,97,8),np.nan); y[:,0]=1; y[:,1:33]=0
    choices=np.array([[0,1,0,1,0],[1,0,1,0,1]])
    x=condition_outcomes(y,choices,list(range(1,33)))
    assert np.array_equal(x[:,2],x[:,4])
    assert np.all(x[:,0]==1) and np.all(x[:,1]==0)
    with pytest.raises(ValueError):condition_outcomes(y,choices,[40])


def test_paired_noise_and_shared_layout_bootstrap():
    noise=np.array([[0,1]*4,[1,0]*4]*4,float)
    x=np.repeat(noise[:,None],7,axis=1)
    tasks=np.array(['a']*4+['b']*4);splits=np.array(['seen','seen','out','out']*2)
    layouts=np.array([1,2,3,4]*2)
    a,b=paired_bootstrap(x,tasks,splits,layouts,draws=100)
    assert np.array_equal(a[:,0],a[:,6])
    assert np.array_equal(b[:,0],b[:,6])
    assert np.array_equal(a,paired_bootstrap(x,tasks,splits,layouts,draws=100)[0])


def test_identical_methods_have_zero_paired_interval():
    x=np.ones((4,7,8))
    rows=[dict(task=t,split='seen',layout_id=l) for t in ('a','b') for l in (1,2)]
    result=summarize_population(x,rows,draws=100)
    assert result['methods']['window3']['success']['mean']==1
    assert result['methods']['window3']['gain_vs_stay']['ci95']==[0,0]
