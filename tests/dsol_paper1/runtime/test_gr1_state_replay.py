"""Mock integration API; no simulator or model dependency in default tests."""
from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import pytest
from benchmarks.Robocasa_tabletop.state_replay import IntegrationReplay, EXTERNAL_FIELDS


class API:
    __version__='mock'
    mjtState=SimpleNamespace(mjSTATE_INTEGRATION=3)
    @staticmethod
    def mj_stateSize(m,s):return 4
    @staticmethod
    def mj_getState(m,d,out,s):out[:]=d.state
    @staticmethod
    def mj_setState(m,d,x,s):d.state=x.copy()
    @staticmethod
    def mj_forward(m,d):d.forwarded+=1;d.state[-1]=99


class External:
    def __init__(self):self.value={k:{'marker':1} for k in EXTERNAL_FIELDS};self.restores=0
    def capture(self):return deepcopy(self.value)
    def restore(self,value):self.restores+=1;self.value=deepcopy(value)


def setup():
    model=SimpleNamespace(**dict.fromkeys(('nq','nv','na','nu','nbody','nmocap','nuserdata'),1))
    data=SimpleNamespace(state=np.array([1.,2.,3.,4.]),forwarded=0);ext=External()
    return IntegrationReplay(API,model,data,ext,model_sha256='a'*64,guard=lambda:None),data,ext


def test_roundtrip_recomputes_kinematics_preserves_integration_and_external_state():
    replay,data,ext=setup();snapshot=replay.capture()
    data.state[:]=7;ext.value['task']['marker']=8
    receipt=replay.restore(snapshot)
    assert np.array_equal(data.state,[1,2,3,4]) and ext.value['task']['marker']==1
    assert data.forwarded==1 and not receipt['dispatch_resumed']
    data.state[:]=9
    assert np.array_equal(snapshot['integration'],[1,2,3,4])


@pytest.mark.parametrize('failure',['model','topology','nonfinite','external'])
def test_wrong_identity_or_partial_state_never_mutates_backend(failure):
    replay,data,ext=setup();snapshot=replay.capture()
    if failure=='model':snapshot['identity']['model_sha256']='b'*64
    elif failure=='topology':snapshot['identity']['topology']['nq']=3
    elif failure=='nonfinite':snapshot['integration'][0]=float('nan')
    else:del snapshot['external']['native_cache']
    with pytest.raises(ValueError):replay.restore(snapshot)
    assert ext.restores==0 and data.forwarded==0 and replay.status=='HOLD'
    with pytest.raises(RuntimeError):replay.restore(snapshot)


def test_external_restore_failure_holds_and_preserves_original_failure():
    replay,data,ext=setup();snapshot=replay.capture()
    def fail(x):raise ConnectionError('external state unavailable')
    ext.restore=fail
    with pytest.raises(ConnectionError):replay.restore(snapshot)
    assert replay.status=='HOLD' and data.forwarded==0
