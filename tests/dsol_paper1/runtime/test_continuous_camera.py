"""CPU geometry and actual-source protocol tests; fake physics, no VLA."""
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.LIBERO.continuous_camera import ContinuousCamera
from benchmarks.LIBERO.local_view_runtime import RenderedViewSource


def rotation(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


class Sim:
    def __init__(self):
        self.model=SimpleNamespace(cam_mode=np.array([0]),cam_bodyid=np.array([0]),
            cam_pos=np.array([[.1,.2,.3]]),cam_quat=np.array([[0.,1.,0.,0.]]))
        parent=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        self.data=SimpleNamespace(body_xmat=parent[None],body_xpos=np.array([[1.,2.,3.]]),
            mocap_pos=np.zeros((0,3)),mocap_quat=np.zeros((0,4)),geom_xpos=np.ones((2,3)),
            geom_xmat=np.tile(np.eye(3),(2,1,1)),cam_xmat=np.zeros((1,3,3)),cam_xpos=np.zeros((1,3)))
        self.state=np.zeros(10)
        self.forward()

    def get_state(self):return self.state

    def forward(self):
        self.data.cam_xmat[0]=self.data.body_xmat[0]@rotation(self.model.cam_quat[0])
        self.data.cam_xpos[0]=self.data.body_xpos[0]+self.data.body_xmat[0]@self.model.cam_pos[0]


def camera():
    sim=Sim();p=np.eye(4)
    p[:3,:3]=sim.data.cam_xmat[0]@np.diag([1.,-1.,-1.]);p[:3,3]=sim.data.cam_xpos[0]
    return sim,ContinuousCamera(sim,0,p)


def test_parent_transform_and_native_zero_roundtrip():
    sim,cam=camera();q=sim.model.cam_quat.copy();p=sim.model.cam_pos.copy()
    target=cam.mount.pose([7.125,-4.375]);receipt=cam.move(target)
    np.testing.assert_allclose(receipt['camera_to_world'],target,rtol=0,atol=1e-14)
    cam.move(cam.mount.pose([0,0]))
    np.testing.assert_array_equal(sim.model.cam_quat,q)
    np.testing.assert_array_equal(sim.model.cam_pos,p)


@pytest.mark.parametrize('kind',['too_far','translated','roll','geometry_changed'])
def test_reject_invalid_move(kind):
    sim,cam=camera();p=cam.mount.pose([3.,0.])
    if kind=='too_far':p=cam.mount.pose([11.,0.])
    elif kind=='translated':p[0,3]+=.01
    elif kind=='roll':p[:3,:3]=p[:3,:3]@np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    else:sim.data.geom_xpos[0,0]+=.001
    with pytest.raises(ValueError):cam.move(p)


def test_source_roundtrip_accounts_actual_queries_and_blocks_retry(tmp_path):
    sim,cam=camera();changed=[False];calls=[]
    def request(cmd):
        kind=cmd['kind'];calls.append(kind)
        if kind=='continuous_info':return dict(reference=cam.mount.reference,receipt=cam.receipt())
        if kind=='move_pose':return dict(receipt=cam.move(cmd['camera_to_world']))
        assert kind=='query_pose'
        item=dict(lang='task',image=[np.zeros((2,2,3),np.uint8)]*2,state=np.zeros(7),
            camera_intrinsics=np.eye(3),camera_to_world_opencv=cam.actual_pose())
        if changed[0]:item['state'][0]=1
        return dict(example=item,receipt=cam.receipt())
    class Runtime:
        calls=0
        def infer(self,*args,**kwargs):
            self.calls+=1
            return dict(score=1.,actions=np.zeros((10,7)),trace=np.zeros((2,10,7)),
                external=np.ones(4),wrist=np.ones(4))
    runtime=Runtime();source=RenderedViewSource(request,runtime,np.zeros((2,10,7),np.float32),tmp_path)
    source.query();source.move_angles([7.125,-4.375]);source.query()
    source.move_angles([0.,0.]);source.query()
    assert source.queries==3 and runtime.calls==6
    changed[0]=True
    with pytest.raises(ValueError,match='input mismatch'):source.query()
    assert runtime.calls==6 and source.queries==4 and source.failed
    before=len(calls)
    with pytest.raises(ValueError):source.query()
    assert len(calls)==before
