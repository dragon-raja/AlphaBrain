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


def test_render_pose_rejects_unsupported_camera_before_mutation():
    from benchmarks.Robocasa365.recorded_render_pose import install_recorded_render_pose
    model=SimpleNamespace(cam_mode=np.array([1]), light_mode=np.array([0]))
    data=SimpleNamespace(xpos=np.zeros((1,3)))
    sim=SimpleNamespace(model=SimpleNamespace(_model=model),data=SimpleNamespace(_data=data))
    with pytest.raises(ValueError, match='fixed cameras'):
        install_recorded_render_pose(sim,np.ones((1,3)),np.ones((1,4)))
    assert not data.xpos.any()


def test_render_pose_derives_sites_without_changing_raw_state():
    from benchmarks.Robocasa365.recorded_render_pose import install_recorded_render_pose
    model=SimpleNamespace(cam_mode=np.array([0]), light_mode=np.array([0]), nbody=1,
        ngeom=0,nsite=1,body_ipos=np.zeros((1,3)),body_iquat=np.array([[1.,0,0,0]]),
        site_pos=np.array([[2.,0,0]]),site_quat=np.array([[1.,0,0,0]]),
        site_bodyid=np.array([0]),site_sameframe=np.array([0]))
    data=SimpleNamespace(xpos=np.zeros((1,3)),xquat=np.zeros((1,4)),xmat=np.zeros((1,9)),
        xipos=np.zeros((1,3)),ximat=np.zeros((1,9)),site_xpos=np.zeros((1,3)),
        site_xmat=np.zeros((1,9)),qpos=np.array([99.]),time=12.)
    calls=[]
    def local(d,pos,mat,p,q,b,s):
        pos[:]=d.xpos[b]+p
        mat[:]=np.eye(3).ravel()
    native=SimpleNamespace(mju_quat2Mat=lambda res,q:res.__setitem__(slice(None),np.eye(3).ravel()),
        mj_local2Global=local,mj_camlight=lambda m,d:calls.append('cameras'))
    sim=SimpleNamespace(model=SimpleNamespace(_model=model),data=SimpleNamespace(_data=data))
    install_recorded_render_pose(sim,np.array([[3.,0,0]]),np.array([[1.,0,0,0]]),native=native)
    np.testing.assert_array_equal(data.site_xpos,[[5.,0,0]])
    assert data.qpos[0]==99. and data.time==12. and calls==['cameras']


def test_scene_resources_are_released_before_next_context_exists():
    from benchmarks.Robocasa365.same_state_restore import initialize_renderer, close_render_sim
    events=[]
    current=[None]
    class Context:
        def __init__(self,sim,device_id):
            self.label=sim.label
            self.vopt=SimpleNamespace(geomgroup=np.zeros(2))
            self.con=SimpleNamespace(offSamples=4)
            self.gl_ctx=SimpleNamespace(make_current=self.make_current)
            sim._render_context_offscreen=self
            self.make_current()
        def make_current(self):
            current[0]=self.label
        def __del__(self):
            events.append((self.label,current[0]))
    # Avoid self-reference in fake gl_ctx; real GL context is separate too.
    class GL:
        def __init__(self,label):self.label=label
        def make_current(self):current[0]=self.label
    original=Context.__init__
    def init(self,sim,device_id):
        original(self,sim,device_id)
        self.gl_ctx=GL(self.label)
    Context.__init__=init
    class Sim:
        def __init__(self,label):self.label=label
        def free(self):del self._render_context_offscreen
    for label in ('cabinet','sink'):
        sim=Sim(label)
        initialize_renderer(sim,Context,0,4)
        close_render_sim(sim)
        assert events[-1]==(label,label)
    assert len(events)==2
