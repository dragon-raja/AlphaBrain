import numpy as np
from AlphaBrain.research.dsol.analysis.channel_phase import close_mask,contact_bodies,sign_change_intervals,temporal_context


def test_native_command_thresholds_are_distinct():
    values=np.array([-.1,0.,.1,.5,1.])
    assert close_mask(values,dict(threshold=.5,inclusive=True)).tolist()==[False,False,False,True,True]
    assert close_mask(values,dict(threshold=0.,inclusive=False)).tolist()==[False,False,True,True,True]


def test_contact_requires_nonpositive_distance_and_gripper_to_object():
    names=dict(body_names=['gripper0_leftfinger','robot0_arm','obj_main'],geom_bodyid=[0,1,2])
    assert contact_bodies([[0,2,.001],[1,2,-.1]],names)=={}
    assert contact_bodies([[2,0,0]],names)=={'obj_main':['gripper0_leftfinger']}


def test_future_contact_cannot_leak_into_earlier_score():
    frames=[dict(step=t,raw_joint_separation_m=.08,observed_joint_separation_m=.08,render_tip_center_separation_m=.097,
        target_distance_m={'obj':.1-t*.01},target_displacement_m={'obj':0.},contact_bodies={'obj':['finger']} if t==3 else {},closing_command=t==3) for t in range(4)]
    early=temporal_context(frames,2,2)
    assert early['target_contact_seen']=={'obj':False}
    assert early['closing_commands_previous_interval']==0
    assert temporal_context(frames,3,2)['first_target_contact_so_far']=={'obj':3}


def test_score_crossing_is_an_interval_not_an_interpolated_event():
    assert sign_change_intervals([16,32,48],[-.1,.2,np.nan])==[dict(after=16,by=32,from_sign=-1,to_sign=1)]
    assert sign_change_intervals([16,32],[-.1,0.])==[]


def test_recovery_baseline_uses_frozen_perturbation_support_not_canonical():
    from AlphaBrain.research.dsol.analysis.channel_phase import baseline_indices
    views=['canonical','broad_train_000']+[f'broad_heldout_{i:03d}' for i in range(32)]
    assert baseline_indices('libero_canonical',views,'disturbance_uniform')==list(range(2,34))
    assert baseline_indices('plus_broad',['canonical','a','b'],'disturbance_uniform')==[1,2]
    assert baseline_indices('plus_broad',['canonical','a','b'],'canonical')==[0]


def test_missing_frozen_libero_candidates_fail_closed():
    import pytest
    from AlphaBrain.research.dsol.analysis.channel_phase import baseline_indices
    with pytest.raises(ValueError,match='support'):
        baseline_indices('libero_broad',['canonical','broad_heldout_000'],'disturbance_uniform')
