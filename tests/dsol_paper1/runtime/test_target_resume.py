import json
from pathlib import Path
import numpy as np
import pytest
from benchmarks.LIBERO.target_recovery_resume import RecoveredCalls
from benchmarks.LIBERO.local_search_rollout import input_identity
from AlphaBrain.research.dsol.data.flow_noise import tensor_sha256


def fixture(tmp_path):
    case=tmp_path/'old/case-0000';case.mkdir(parents=True);(case/'episode-0').mkdir()
    example=dict(state=np.zeros(7),image=[np.zeros((2,2,3),np.uint8)],lang='task')
    noise=np.zeros((10,7),np.float32)
    identity=dict(state_key='s',start=15,method='local',repeat=0,replan=0)
    actions=np.arange(224,dtype=np.float32).reshape(32,7)/100
    prediction=dict(kind='prediction',identity=identity,input_sha256=input_identity(example),full_decoded_actions=actions.tolist())
    executed=dict(kind='executed',identity=identity,actions=actions[:10].tolist(),success=False)
    (case/'episode-0/events.jsonl').write_text(json.dumps(prediction)+'\n'+json.dumps(executed)+'\n')
    attempt=dict(identity=dict(identity,stage='execution'),input_sha256=input_identity(example),noise_sha256=tensor_sha256(noise))
    (case.parent/'model-attempts.jsonl').write_text(json.dumps(attempt)+'\n')
    class Calls:
        stage='execution';calls=0
        def infer(self,*args,**kwargs):self.calls+=1;return dict(live=True)
    live=Calls();output=tmp_path/'new';output.mkdir()
    return RecoveredCalls(live,case,output,lambda:None),live,example,noise,identity,actions


def test_durable_prediction_reuses_zero_model_calls_and_verifies_prefix(tmp_path):
    replay,live,ex,noise,identity,actions=fixture(tmp_path)
    result=replay.infer(ex,noise,identity,capture=False)
    assert live.calls==0 and result['full_actions'].dtype==np.float32
    np.testing.assert_array_equal(result['full_actions'],actions)
    with pytest.raises(ValueError,match='consumed twice'):replay.infer(ex,noise,identity,capture=False)
    replay.verify_trajectory(0,dict(replans=2,success=True),dict(replan_action_offsets=np.array([0,10]),actions=actions[:20]))
    with pytest.raises(ValueError,match='termination differs'):
        replay.verify_trajectory(0,dict(replans=1,success=True),dict(replan_action_offsets=np.array([0]),actions=actions[:10]))
    # An absent prediction is an explicitly audited missing call, never a
    # fabricated result or a refund of the prior attempt.
    assert replay.infer(ex,noise,dict(identity,replan=1),capture=False)==dict(live=True)
    assert live.calls==1


@pytest.mark.parametrize('change',['input','noise','identity'])
def test_recovery_refuses_mismatched_known_prefix(tmp_path,change):
    replay,live,ex,noise,identity,_=fixture(tmp_path)
    if change=='input':ex['state'][0]=1
    elif change=='noise':noise[0,0]=1
    else:identity=dict(identity,start=14)
    with pytest.raises(ValueError):replay.infer(ex,noise,identity,capture=False)
    assert live.calls==0


def test_cached_scoring_reuses_exact_npz_with_input_and_noise_binding(tmp_path):
    replay,live,ex,noise,_,_=fixture(tmp_path);replay.stage='score'
    identity=dict(case='case-0000',query=1,member=0,stage='continuous_admission')
    replay.attempts=[dict(identity=dict(identity,stage='score'),input_sha256=input_identity(ex),noise_sha256=tensor_sha256(noise))]
    np.savez(replay.case/'query-01.npz',actions_0=np.zeros((10,7)),trace_0=np.ones((10,32,7)),
        external_0=np.ones(4),wrist_0=np.ones(4),score_0=np.array(.1))
    assert float(replay.infer(ex,noise,identity)['score'])==.1 and live.calls==0
