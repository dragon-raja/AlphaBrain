"""The new native search API preserves per-action history and unknown holds."""
from pathlib import Path
import numpy as np
import pytest
from benchmarks.LIBERO import rldx_search_runtime as r
from benchmarks.LIBERO.rldx_runtime import accel_scores
from benchmarks.LIBERO.local_search_rollout import input_identity
from AlphaBrain.research.dsol.data.native_features import from_wire
from tests.dsol_paper1.helpers.dispatch_policy import allow_synthetic_boundaries


def example(i=0):
    image=np.full((256,256,3),i,np.uint8)
    return dict(lang='task',image=[image,image.copy()],state=np.zeros(7,np.float32),camera_intrinsics=np.eye(3),camera_to_world_opencv=np.eye(4),
        fastwam_raw=dict(schema='LIBERO_RAW_RGB256_PROPRIO_V1',agentview_image=image,robot0_eye_in_hand_image=image.copy(),
         robot0_eef_pos=np.array([i,0,0]),robot0_eef_quat=np.array([0.,0.,0.,1.]),robot0_gripper_qpos=np.zeros(2)))


def runtime(tmp_path,monkeypatch,unknown=False):
    allow_synthetic_boundaries(monkeypatch,'benchmarks/LIBERO/rldx_search_runtime.py:RLDXSearchRuntime.infer')
    value=object.__new__(r.RLDXSearchRuntime);value.spec=r.feature_spec();value.output=tmp_path;value.limit=20;value.calls=0;value.failed=False;value.model_sha256='a'*64;value.history=None
    class Native:
        output=tmp_path/'native-rldx'
        calls=0
        observed=[]
        def infer(self,history,seed,identity,**kwargs):
            self.calls+=1;self.observed.append(history.observation())
            if unknown and self.calls==2:raise RuntimeError('unknown native outcome')
            v=np.arange(4*16*64,dtype=np.float32).reshape(4,1,16,64)+1
            return dict(actions=np.zeros((16,7)),normalized_actions=np.zeros((1,16,64)),velocities=v,scores=accel_scores(v),backbone_features=np.ones((1,64,4096),np.float32))
    value.native=Native();return value


def test_capture_identity_noise_and_unknown_are_not_replayed(tmp_path,monkeypatch):
    v=runtime(tmp_path,monkeypatch,unknown=True);e=example();noise=np.zeros((10,7),np.float32)
    request=dict(query_id='one',input_sha256=input_identity(e),history_sha256='b'*64)
    got=v.infer(e,noise,{},capture=True,native_request=request);packet=from_wire(got['native_features'])
    assert packet.layout==r.feature_spec()['layout'] and packet.packed().shape==(16384,)
    assert got['trace'].shape==(4,8,7)
    with pytest.raises(RuntimeError,match='unknown'):v.infer(e,noise,{},capture=True,native_request=request)
    assert v.failed and v.calls==2
    with pytest.raises(ValueError,match='Unknown'):v.infer(e,noise,{},capture=True,native_request=request)
    assert v.calls==2 and len((tmp_path/'calls.jsonl').read_text().splitlines())==1


def test_rollout_updates_every_executed_action_and_closes_history(tmp_path,monkeypatch):
    v=runtime(tmp_path,monkeypatch);steps=[]
    class Bank:
        def get(self,*args):return dict(noise=np.zeros((10,7),np.float32),noise_sha256='c'*64)
    def query(payload):
        assert payload['kind']=='step' and payload['actions'].shape==(1,7)
        steps.append(1);return dict(executed=1,success=len(steps)==19,example=example(len(steps)),physics=str(len(steps)))
    result,arrays=r.rollout_rldx_episode(query,v.infer,example(),Bank(),'state',0,{},runtime=v,record=lambda e:None)
    assert result['steps']==19 and result['replans']==3 and v.history is None
    assert arrays['actions'].shape==(19,7) and arrays['normalized_actions'].shape==(3,1,16,64)
    assert [obs['state']['eef_pos_absolute'][0,0,0] for obs in v.native.observed]==[0,8,16]
    # Real frames at step16 are 10,12,14,16, not four copies of step16.
    frames=v.native.observed[-1]['video']['front_view'][0]
    assert [int(x[0,0,0]) for x in frames]==[10,12,14,16]


def test_operation_never_fabricates_a_reset_history(tmp_path,monkeypatch):
    v=runtime(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='explicit native temporal history'):v.infer(example(),np.zeros((10,7)),{},capture=False)
    assert v.calls==0
    with pytest.raises(ValueError):r.noise_seed(np.zeros((16,64)))
