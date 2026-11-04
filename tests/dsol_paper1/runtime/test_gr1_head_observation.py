"""Mock-only boundary checks. No GR1 simulator/model/GPU admission implied."""
import json
import numpy as np
import pytest
from benchmarks.Robocasa_tabletop.head_observation import HeadObservationPort
from AlphaBrain.research.dsol.data.native_observation import ObservationBundle,CameraFrame
from AlphaBrain.research.dsol.execution.phase_budget import PhaseBudget
from AlphaBrain.research.dsol.execution.manifest import file_digest

class MockHead:
    def __init__(self):self.epoch=0;self.moves=0;self.scene='fixed';self.stale=False;self.disconnect=False
    def non_head_state_identity(self):return self.scene
    def move_head(self,target):
        self.moves+=1
        return dict(measured_joint_rad=target,settled=True,settled_at=10.)
    def reset_observation_cache(self,epoch):self.epoch=epoch
    def capture(self,q):
        if self.disconnect:raise ConnectionError('mock loss')
        bundle=ObservationBundle(q,'sim',10.,11.,1.,0.,'a'*64,'b'*64,
            (CameraFrame('ego','movable',True,9. if self.stale else 11.,'c'*64),))
        return bundle,self.epoch

def port(backend):
    return HeadObservationPort(backend,joint_limits_rad=[[-1.,1.],[-.3,.4]],tolerance_rad=.001,guard=lambda:None)

def test_move_stop_freshness_no_fake_auxiliary():
    b=MockHead();p=port(b);p.move([.2,.1]);obs=p.capture('q0')
    assert len(obs.cameras)==1 and obs.cameras[0].valid
    with pytest.raises(ValueError):p.capture('q0')
    with pytest.raises(ValueError):p.move([0.,.5])
    p.stop();assert b.moves==1
    with pytest.raises(RuntimeError):p.move([0.,0.])

@pytest.mark.parametrize('failure',['stale','disconnect','changed_scene'])
def test_failure_holds_port_and_keeps_budget_until_explicit_audit(tmp_path,failure):
    a=tmp_path/'auth.json';a.write_text(json.dumps(dict(status='EXPLICIT_USER_APPROVAL',total_model_call_limit=2)))
    c=tmp_path/'campaign.json';c.write_text(json.dumps(dict(limits=dict(score=2,paired_execution=0,nominal_execution=0))))
    ledger=PhaseBudget(tmp_path/'local',tmp_path/'mirror',a,authorization_sha256=file_digest(a));ledger.reserve('query',c)
    b=MockHead();p=port(b)
    if failure=='changed_scene':
        original=b.move_head
        def move(target):
            result=original(target);b.scene='drift';return result
        b.move_head=move
        with pytest.raises(ValueError):p.move([0.,0.])
    else:
        p.move([0.,0.]);setattr(b,failure,True)
        with pytest.raises((ValueError,ConnectionError)):p.capture('q0')
    assert p.status=='HOLD' and ledger.status()['held_unsettled']==2
    with pytest.raises(RuntimeError):p.capture('q1')
    # Reopening a ledger does not refund or authorize replay after lost contact.
    recovered=PhaseBudget(tmp_path/'local',tmp_path/'mirror',a,authorization_sha256=file_digest(a))
    with pytest.raises(ValueError):recovered.reserve('query',c)
    assert recovered.status()['held_unsettled']==2


def test_authority_loss_prevents_motion_and_requires_audit():
    b=MockHead()
    def denied():raise ConnectionError('authority unavailable')
    p=HeadObservationPort(b,joint_limits_rad=[[-1,1],[-1,1]],tolerance_rad=.001,guard=denied)
    with pytest.raises(ConnectionError):p.move([0,0])
    assert b.moves==0 and p.status=='HOLD'
