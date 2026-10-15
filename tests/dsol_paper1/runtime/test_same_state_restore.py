from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.Robocasa365.same_state_restore import RAW, GEOMETRY, CAMERAS, restore_frame, forbid_step, validate_restoration_model
from benchmarks.Robocasa365.snapshot_bundle import compiled_model_signature


def test_restoration_recomputes_geometry_without_integrating_or_copying_it():
    arrays={f'frame/026/sim/{k}':np.array([2.]) for k in RAW+GEOMETRY}
    arrays['frame/026/sim/time']=np.array(1.)
    data=SimpleNamespace(**{k:np.array([0.]) for k in RAW+GEOMETRY},time=0.)
    model=SimpleNamespace(na=0,nmocap=0,cam_pos=np.zeros(1),cam_quat=np.zeros(1))
    def forward():
        # Only kinematic computation; deliberately disagree with recorded body position.
        for k in GEOMETRY:getattr(data,k)[:]=3.
        data.qacc_warmstart[:]=99.
    sim=SimpleNamespace(data=data,model=model,set_state_from_flattened=lambda x:None,forward=forward,step=forbid_step)
    snapshot=SimpleNamespace(model_runtime={},sim_state=np.zeros(1))
    checks=restore_frame(sim,snapshot,arrays,26,dict(positions=np.ones(1),quaternions=np.ones(1)))
    assert all(checks[k]['exact'] for k in RAW)
    assert all(not checks[k]['exact'] for k in GEOMETRY)
    assert data.time==1. and data.body_xpos[0]==3.


def test_missing_mocap_state_cannot_be_assumed_static():
    sim=SimpleNamespace(model=SimpleNamespace(na=0,nmocap=1))
    with pytest.raises(ValueError,match='not recorded'):
        restore_frame(sim,None,{},26,{})


def test_step_is_forbidden():
    with pytest.raises(RuntimeError,match='No simulator steps'):
        forbid_step()


def test_runtime_camera_schema_is_checked_before_installation():
    model=SimpleNamespace(nq=1,nv=1,nu=1,nbody=1,ncam=3,ngeom=1,
        body_names=('world',),camera_names=CAMERAS,joint_names=('joint',),geom_names=('geom',),
        cam_pos=np.zeros((3,3)),cam_quat=np.zeros((3,4)),body_pos=np.zeros((1,3)))
    bundle=SimpleNamespace(manifest={'model_signature':compiled_model_signature(model)},
                           snapshot=SimpleNamespace(model_runtime={'body_pos':np.zeros((1,3))}))
    sim=SimpleNamespace(model=model)
    candidate=dict(positions=np.zeros((3,3)),quaternions=np.zeros((3,4)))
    assert validate_restoration_model(sim,bundle,{'view':candidate})['status']=='PASS_MODEL_SCHEMA'
    with pytest.raises(ValueError,match='camera installation'):
        validate_restoration_model(sim,bundle,{'view':dict(candidate,positions=np.zeros((2,3)))})
    model.ncam=2
    with pytest.raises(ValueError,match='signature'):
        validate_restoration_model(sim,bundle,{'view':candidate})


def test_moving_mocap_is_restored_before_forward_and_act_is_not_invented():
    names=('world', 'second_slot', 'first_slot')
    model=SimpleNamespace(na=1,nmocap=2,body_names=names,body_mocapid=np.array([-1,1,0]),
                          cam_pos=np.zeros(1),cam_quat=np.zeros(1))
    data=SimpleNamespace(**{k:np.zeros(1) for k in RAW},time=0.,act=np.zeros(1),
                         mocap_pos=np.zeros((2,3)),mocap_quat=np.zeros((2,4)),
                         body_xpos=np.zeros((3,3)),body_xquat=np.zeros((3,4)),
                         site_xpos=np.zeros(1),site_xmat=np.zeros(1))
    arrays={f'frame/026/sim/{k}':np.ones(1) for k in RAW+('act','site_xpos','site_xmat')}
    arrays.update({'frame/026/sim/time':np.array(3.),
                   'frame/026/sim/body_xpos':np.array([[0.,0,0],[1,2,3],[4,5,6]]),
                   'frame/026/sim/body_xquat':np.tile([1.,0,0,0],(3,1))})
    def forward():
        data.body_xpos[1:]=data.mocap_pos[[1,0]]
        data.body_xquat[0]=[1,0,0,0]
        data.body_xquat[1:]=data.mocap_quat[[1,0]]
        data.site_xpos[:]=data.act
        data.site_xmat[:]=1
    sim=SimpleNamespace(model=model,data=data,forward=forward,step=forbid_step,set_state_from_flattened=lambda _:None)
    snapshot=SimpleNamespace(model_runtime={},sim_state=np.zeros(1))
    for displacement in (0., 10.):
        arrays['frame/026/sim/body_xpos'][1:]+=displacement
        checks=restore_frame(sim,snapshot,arrays,26,dict(positions=np.zeros(1),quaternions=np.zeros(1)),names)
        assert all(c['exact'] for c in checks.values())
    del arrays['frame/026/sim/act']
    with pytest.raises(ValueError, match='actuator activation was not recorded'):
        restore_frame(sim,snapshot,arrays,26,{},names)
