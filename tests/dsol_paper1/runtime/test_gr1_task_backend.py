"""Fresh frame/actual head coupling without constructing a simulator or renderer."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from benchmarks.Robocasa_tabletop.task_backend import HeadTaskBackend, EgoFrame


def backend():
    class IO:
        q=np.zeros(3); v=np.zeros(3); stamp=0.; scene='fixed'
        def read_head_state(self):return dict(positions=self.q.copy(),velocities=self.v.copy(),observed_at=self.stamp)
        def non_head_state_identity(self):return self.scene
    io=IO()
    class Servo:
        roll_hold=0.
        settings=SimpleNamespace(position_tolerance=.001,velocity_tolerance=.001)
        def move_head(self,target):
            io.q=np.array([target[0]+.0001,0.,target[1]+.0001]);io.stamp+=1.
            return dict(measured_joint_rad=tuple(io.q[[0,2]]),settled_at=io.stamp,settled=True)
    class Geometry:
        def measured_pose(self,angles,roll):
            p=np.eye(4);p[:2,3]=angles/100.;return p
    geo=Geometry()
    def capture(epoch,sequence):
        return EgoFrame(np.zeros((2,2,3),np.uint8),epoch,sequence,io.stamp,
            geo.measured_pose(np.rad2deg(io.q[[0,2]]),0.))
    b=HeadTaskBackend(None,io,Servo(),geo,capture,guard=lambda:None,
        input_builder=lambda env,image:dict(image=image))
    return b


def test_actual_measured_arrival_and_monotonic_capture():
    b=backend();receipt=b.move_head_degrees([3.,2.])
    assert receipt.joint_angles_deg!= (3.,2.)
    b.reset_observation_cache(1)
    x,epoch=b.capture_input(1)
    assert epoch==1 and x['image'].shape==(2,2,3)
    b.capture_input(1)
    assert b.sequence==2
    with pytest.raises(ValueError,match='advance'):b.reset_observation_cache(1)
    assert b.status=='HOLD'


@pytest.mark.parametrize('fault',['epoch','sequence','time','pose','scene','joint','nan_time'])
def test_stale_or_mutating_capture_holds_backend(fault):
    b=backend();b.move_head_degrees([0.,0.]);original=b.capture_frame
    def broken(epoch,sequence):
        f=original(epoch,sequence)
        if fault=='epoch':return replace(f,epoch=epoch+1)
        if fault=='sequence':return replace(f,sequence=sequence-1)
        if fault=='time':return replace(f,observed_at=f.observed_at-1.)
        if fault=='pose':return replace(f,camera_to_world=np.eye(4)*2)
        if fault=='scene':b.io.scene='moved'
        if fault=='joint':b.io.q[0]+=.1
        if fault=='nan_time':b.io.stamp=float('nan')
        return f
    b.capture_frame=broken
    with pytest.raises(ValueError):b.capture_input(0)
    assert b.status=='HOLD'
    with pytest.raises(RuntimeError):b.capture_input(0)


def test_initial_pose_requires_explicit_settlement():
    b=backend()
    with pytest.raises(ValueError,match='initial'):b.pose_receipt()
    assert b.status=='HOLD'
