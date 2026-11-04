"""Injected CPU traces only; no robot dynamics, rendering, or policy calls."""
import pytest
from benchmarks.Robocasa_tabletop.head_layout import HeadLayout
from benchmarks.Robocasa_tabletop.head_servo import HeadServo, ServoSettings


class TraceIO:
    def __init__(self, trace):
        self.trace, self.index, self.commands, self.stops = trace, 0, [], 0
        self.scene = 'fixed'

    def read_head_state(self):
        p, v = self.trace[min(self.index, len(self.trace)-1)]
        return dict(positions=p, velocities=v, observed_at=float(self.index))

    def non_head_state_identity(self): return self.scene
    def write_head_efforts(self, command): self.commands.append(command)
    def advance(self): self.index += 1
    def stop_head_motion(self): self.stops += 1


def servo(io, guard=lambda: None, clock=lambda: 0.):
    layout = HeadLayout(((-1., 1.), (-.2, .2), (-.2, .2)),
                        ((0,0,1),(1,0,0),(0,1,0)), ('yaw_motor','roll_motor','pitch_motor'), 'motor')
    cfg = ServoSettings((10.,)*3, (1.,)*3, (.5,)*3, .001, .001, 2, 5, 1.)
    return HeadServo(io, layout, cfg, roll_hold=0., guard=guard, clock=clock)


def test_three_axis_measured_settling_and_bounded_motor_effort():
    io=TraceIO([((0.,.1,0.),(0.,0.,0.)), ((.2,0.,.1),(0.,0.,0.)),
                ((.2,0.,.1),(0.,0.,0.))])
    result=servo(io).move_head((.2,.1))
    assert result['measured_joint_rad']==(.2,.1) and result['steps']==2
    assert io.commands[0]=={'yaw_motor':.5,'roll_motor':-.5,'pitch_motor':.5}
    assert result['measured_roll_rad']==0 and io.stops==0


@pytest.mark.parametrize('failure',['roll','velocity','stale','scene','authority','deadline'])
def test_failures_hold_and_stop_without_automatic_replay(failure):
    p=(.2,.1,.1) if failure=='roll' else (.2,0.,.1)
    v=(.1,0.,0.) if failure=='velocity' else (0.,0.,0.)
    io=TraceIO([(p,v)]*8)
    guard=lambda: None
    clock=lambda: 0.
    if failure=='stale':io.advance=lambda: None
    if failure=='scene':
        def advance():io.index+=1;io.scene='changed'
        io.advance=advance
    if failure=='authority':
        def guard():
            if io.index>0:raise ConnectionError('lost authority')
    if failure=='deadline':clock=lambda: float(io.index)*2
    s=servo(io,guard,clock)
    with pytest.raises((ValueError,TimeoutError,ConnectionError)):s.move_head((.2,.1))
    assert s.status=='HOLD' and io.stops==1 and len(io.commands)<=5
    n=len(io.commands)
    with pytest.raises(RuntimeError):s.move_head((.2,.1))
    assert len(io.commands)==n


def test_invalid_target_never_commands_motor():
    io=TraceIO([((0.,0.,0.),(0.,0.,0.))]);s=servo(io)
    with pytest.raises(ValueError):s.move_head((0.,.8))
    assert not io.commands


def test_failed_local_stop_is_recorded_without_claiming_physical_stop():
    io=TraceIO([((0.,0.,0.),(0.,0.,0.))]);s=servo(io)
    def failed_stop():raise ConnectionError('stop channel lost')
    io.stop_head_motion=failed_stop
    with pytest.raises(TimeoutError):s.move_head((.2,.1))
    assert s.status=='HOLD' and s.stop_failure=='ConnectionError'
