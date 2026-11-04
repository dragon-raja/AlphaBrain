"""CPU geometry and actual-source protocol tests; fake physics, no VLA."""
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.LIBERO.continuous_camera import ContinuousCamera
from benchmarks.LIBERO.local_view_runtime import RenderedViewSource
from AlphaBrain.research.dsol.acquisition.local_view_search import LocalViewConfig, PanTiltMount
from AlphaBrain.research.dsol.data.native_view_panel import views


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


@pytest.mark.parametrize('seed', range(12))
def test_full_panel_inverse_with_rotated_reference(seed):
    rng = np.random.default_rng(seed)
    quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    reference = np.eye(4)
    reference[:3, :3] = rotation(quaternion)
    reference[:3, 3] = rng.normal(size=3)
    cam = ContinuousCamera.__new__(ContinuousCamera)
    cam.mount = PanTiltMount(reference)
    for angles in views():
        pose = cam.mount.pose(angles)
        decoded = cam.angles_for_pose(pose)
        np.testing.assert_allclose(decoded, angles, rtol=0, atol=1e-12)
        # Interior points keep exactly the previous inverse computation.
        if abs(angles[0]) < 30 and abs(angles[1]) < 20:
            r = reference[:3, :3].T @ pose[:3, :3]
            old = np.rad2deg([np.arctan2(-r[2, 0], r[0, 0]),
                             np.arctan2(-r[1, 2], r[1, 1])])
            np.testing.assert_array_equal(decoded, old)


@pytest.mark.parametrize('axis,limit', [(0, 30.), (1, 20.)])
@pytest.mark.parametrize('sign', [-1, 1])
def test_inverse_does_not_admit_outside_angles(axis, limit, sign):
    _, cam = camera()
    outside = np.zeros(2)
    outside[axis] = sign * (limit + 1e-8)
    wider = PanTiltMount(cam.mount.reference, LocalViewConfig(yaw_limit_deg=31, pitch_limit_deg=21))
    with pytest.raises(ValueError, match='outside mount limits'):
        cam.angles_for_pose(wider.pose(outside))
    outside[axis] = sign * np.nextafter(limit, np.inf)
    with pytest.raises(ValueError, match='outside mount limits'):
        cam.mount.pose(outside)


def test_camera_reaches_every_panel_boundary_via_legal_steps():
    _, cam = camera()
    for target in views():
        for waypoint in cam.mount.route(cam.current, target):
            cam.move(cam.mount.pose(waypoint))
        np.testing.assert_allclose(cam.current, target, rtol=0, atol=1e-12)


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


def test_source_reattach_uses_verified_current_position(tmp_path):
    _,cam=camera()
    def request(cmd):
        if cmd['kind']=='continuous_info':return dict(reference=cam.mount.reference,receipt=cam.receipt())
        assert cmd['kind']=='move_pose'
        return dict(receipt=cam.move(cmd['camera_to_world']))
    source=RenderedViewSource(request,None,np.zeros((2,10,7)),tmp_path)
    source.move_angles([-20.,13.333333333333334])
    with pytest.raises(ValueError,match='Resumed source position'):
        RenderedViewSource(request,None,np.zeros((2,10,7)),tmp_path,current_angles=[0.,0.])
    resumed=RenderedViewSource(request,None,np.zeros((2,10,7)),tmp_path,
        current_angles=[-20.,13.333333333333334])
    resumed.move_angles([20.,-13.333333333333334])
    np.testing.assert_allclose(cam.current,[20.,-13.333333333333334],atol=1e-12)
    assert resumed.moves==4 and resumed.queries==0
