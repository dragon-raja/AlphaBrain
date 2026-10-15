import numpy as np
import pytest
from AlphaBrain.research.dsol.analysis.observation_evidence import paired_noise
from AlphaBrain.research.dsol.data.observation_evidence import temporal_pair,modality_vector
from AlphaBrain.research.dsol.metrics.observation_response import response_metrics


def test_pair_separation_and_no_outcome_use():
    rows=[{'frame':i,'success':i%2} for i in range(101)]
    assert [r['frame'] for r in temporal_pair(rows,'frame',50)]==[25,75]
    with pytest.raises(ValueError):temporal_pair(rows[:5],'frame',10)


def test_noise_shared_and_addressed():
    a=paired_noise('task',0,(50,32),41)
    np.testing.assert_array_equal(a,paired_noise('task',0,(50,32),41))
    assert not np.array_equal(a,paired_noise('task',1,(50,32),41))
    assert not np.array_equal(a,paired_noise('other',0,(50,32),41))


def test_modality_order():
    mapping={'a':dict(original_key='x',start=0,end=2),'b':dict(original_key='x',start=2,end=4)}
    np.testing.assert_array_equal(modality_vector({'x':[[1,2,3,4]]},0,mapping,['b','a']),[3,4,1,2])


def test_correct_view_invariance_does_not_mean_state_collapse():
    y=np.stack([np.zeros((5,2)),np.ones((5,2))])
    p=np.broadcast_to(y[:,None,None],(2,3,8,5,2)).copy()
    r=response_metrics(p,y,[1,1],execute_steps=5,dims=[0,1])
    assert r['view_response_rms']==0 and r['canonical_state_response_rms']==1
    assert r['canonical_demonstration_mse']==0 and r['state_assignment_margin']==[1.0]
    r=response_metrics(np.full_like(p,.5),y,[1,1],execute_steps=5,dims=[0,1])
    assert r['view_response_rms']==0 and r['canonical_state_response_rms']==0
    assert r['state_assignment_margin']==[0.0] and r['canonical_demonstration_mse']>.0


def test_state_change_must_have_correct_direction():
    y=np.stack([np.zeros((5,2)),np.ones((5,2))])
    p=np.broadcast_to(y[::-1,None,None],(2,3,8,5,2))
    r=response_metrics(p,y,[1,1],execute_steps=5,dims=[0,1])
    assert r['state_assignment_margin']==[-1.0]
    assert np.allclose(r['state_change_cosine'],[-1])


def test_mean_prediction_error_does_not_hide_noise_error():
    y=np.zeros((2,5,2));p=np.ones((2,3,8,5,2));p[:,:,::2]*=-1
    r=response_metrics(p,y,[1,1],execute_steps=5,dims=[0,1])
    assert r['canonical_demonstration_mse']==0
    assert r['canonical_expected_noise_mse']==1
    assert r['perturbed_expected_noise_mse']==1
