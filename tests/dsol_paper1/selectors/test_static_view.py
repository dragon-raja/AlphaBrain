"""Static rules and calibration tests without GPUs or private data."""
import numpy as np
import pytest

from AlphaBrain.research.dsol.selectors.static_view import metric_choices, calibrate_seen, designs
from AlphaBrain.research.dsol.analysis.static_view_decisions import score_stability, noise_split_outcome_selection


def test_neighbor_rule_avoids_isolated_minimum():
    a=np.repeat(np.array([[[0.],[10.],[2.],[2.],[2.]]]),8,-1)
    distance=np.abs(np.arange(5)[:,None]-np.arange(5))[None].astype(float)
    choices=metric_choices(a,distance,neighborhood=3)
    assert choices['min_mean'][0]==0
    assert choices['min_neighbor_mean'][0]==3


def test_scoring_bootstrap_uses_same_draw_addresses_across_views():
    a=np.broadcast_to(np.arange(8)[None,None,:]+np.arange(20)[None,:,None],(2,20,8)).copy()
    choices,info=score_stability(a,draws=30)
    assert np.all(choices==0)
    assert info['half_score_rank_correlation_mean']==pytest.approx(1.)
    assert info['half_minimum_same_fraction']==1.


def test_calibration_accepts_only_explicit_seen_label_shape():
    x=np.arange(4*3*2).reshape(4,3,2).astype(float)
    train=np.array([0,1]);layouts=np.array([0,1,0,1]);tasks=np.array(['a']*4)
    q=np.array([[0.,.5,1.],[0.,.5,1.]])
    model,info=calibrate_seen(x,q,train,layouts,tasks)
    assert model.predict(x).argmax(1).tolist()==[2]*4
    assert info['alpha'] in [1,10,100]
    with pytest.raises(ValueError,match='seen labels'):
        calibrate_seen(x,np.zeros((4,3)),train,layouts,tasks)


def test_design_pose_does_not_use_accel():
    rng=np.random.default_rng(2)
    a=rng.uniform(.1,1,size=(4,20,8));p=rng.normal(size=(20,3));tasks=np.array(['a','b','a','b'])
    np.testing.assert_array_equal(designs(a,p,tasks,family='pose'),designs(a*5,p,tasks,family='pose'))
    np.testing.assert_allclose(designs(a,p,tasks,family='accel'),designs(a*5,p,tasks,family='accel'),atol=1e-12)


def test_zero_scores_have_finite_design_and_rules():
    a=np.zeros((2,20,8));p=np.zeros((20,3));tasks=np.array(['a','a'])
    assert np.isfinite(designs(a,p,tasks)).all()
    distance=np.zeros((2,20,20))
    assert all(np.array_equal(c,[0,0]) for c in metric_choices(a,distance).values())


def test_noise_split_is_not_per_execution_winner():
    y=np.array([[[1,1,0,0],[0,0,1,1]]])
    assert y.mean(-1).max()==.5
    assert noise_split_outcome_selection(y)==0
    with pytest.raises(ValueError,match='even'):
        noise_split_outcome_selection(y[...,:3])


def test_heldout_features_do_not_change_preprocessing_or_alpha():
    rng=np.random.default_rng(3)
    x=rng.normal(size=(6,10,4));train=np.array([0,1,2,3])
    layouts=np.array([0,1,0,1,5,6]);tasks=np.array(['a','a','b','b','a','b'])
    q=rng.uniform(size=(4,10))
    first,info=calibrate_seen(x,q,train,layouts,tasks)
    changed=x.copy();changed[4:]*=10000
    second,other=calibrate_seen(changed,q,train,layouts,tasks)
    np.testing.assert_array_equal(first.scale,second.scale)
    np.testing.assert_array_equal(first.weights,second.weights)
    assert info==other
