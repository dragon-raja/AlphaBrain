from types import SimpleNamespace
import numpy as np
import pytest
from benchmarks.Robocasa_tabletop.head_control import SubstepHeadHold,ProjectedSceneAdvance
from benchmarks.Robocasa_tabletop.head_servo import ServoSettings


def test_substep_hold_uses_measured_state_native_limits_and_explicit_target():
    class Layout:
        actuator_names=('yaw','roll','pitch')
        def named_targets(self,angles,*,roll_hold):
            if max(map(abs,angles))>1:raise ValueError('limit')
    class IO:
        layout=Layout();efforts=None
        def read_head_state(self):return dict(positions=(0.,0.,0.),velocities=(0.,0.,0.))
        def write_head_efforts(self,named):self.efforts=named
    io=IO();settings=ServoSettings((200.,)*3,(10.,)*3,(5.,)*3,.002,.002,3,2000,20.)
    control=SubstepHeadHold(io,settings,target_rad=[.1,0.,-.1],guard=lambda:None)
    control.apply();assert io.efforts==dict(yaw=5.,roll=0.,pitch=-5.)
    control.check([.1,0.,-.1])
    with pytest.raises(ValueError):control.check([0.,0.,0.])
    with pytest.raises(ValueError):control.set_target([2.,0.,0.])


def test_projection_preserves_nonhead_integration_but_not_head_motion():
    data=SimpleNamespace(qpos=np.array([0.,2.]),qvel=np.array([0.,3.]),
        qacc_warmstart=np.array([0.,4.]),ctrl=np.array([1.,5.]),time=0.)
    class API:
        mjtState=SimpleNamespace(mjSTATE_INTEGRATION=1)
        def mj_stateSize(self,*args):return 9
        def mj_getState(self,m,d,x,s):x[:]=np.r_[d.qpos,d.qvel,d.qacc_warmstart,d.ctrl,d.time]
        def mj_setState(self,m,d,x,s):
            d.qpos[:]=x[:2];d.qvel[:]=x[2:4];d.qacc_warmstart[:]=x[4:6];d.ctrl[:]=x[6:8];d.time=x[8]
        def mj_step(self,m,d):
            d.qpos+=1;d.qvel+=2;d.qacc_warmstart+=3;d.time+=.002
        def mj_forward(self,m,d):d.qacc_warmstart[:]=0
    class IO:
        qpos=np.array([0]);qvel=np.array([0]);actuators=np.array([0])
        def non_head_state_identity(self):return tuple([data.qpos[1],data.qvel[1]])
    api=API();io=IO();scene=io.non_head_state_identity()
    with pytest.raises(ValueError):ProjectedSceneAdvance(api,None,data,io,protocol='implicit',guard=lambda:None)
    step=ProjectedSceneAdvance(api,None,data,io,protocol=ProjectedSceneAdvance.protocol,guard=lambda:None)
    step();assert io.non_head_state_identity()==scene
    assert data.qpos[0]==1 and data.qvel[0]==2 and data.qacc_warmstart[0]==3
    assert data.qacc_warmstart[1]==4 and data.ctrl[1]==5 and data.time==.002
    data.qpos[1]+=1
    with pytest.raises(ValueError,match='Scene changed'):step()
