"""Shared-core binding and real entry tests with synthetic GR1 model ports."""
from copy import deepcopy
from dataclasses import asdict
import json
import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('gymnasium')

from AlphaBrain.research.dsol.data.search_core import core_spec, VARIANTS
from AlphaBrain.research.dsol.data.task_condition import SCHEMA, instruction_key
from AlphaBrain.research.dsol.execution.manifest import file_digest
from AlphaBrain.research.dsol.acquisition.local_view_search import LocalViewConfig, PoseReceipt
from AlphaBrain.research.dsol.selectors.continuous_ppo import batch_observations
from benchmarks.Robocasa_tabletop.policy_binding import (
    make_binding, build_policy, checkpoint_envelope, load_policy, bind_collector,
)
from benchmarks.Robocasa_tabletop.model_runtime import ScoreContract
from benchmarks.Robocasa_tabletop.search_runtime import ArticulatedHeadMount, GR1SearchSource
from benchmarks.Robocasa_tabletop.episode_runtime import run_search_episode


def fixture(tmp_path, variant='U12', text=False):
    torch.set_num_threads(1)
    provenance = {k:'a'*64 for k in ('model_sha256', 'producer_sha256', 'history_sha256',
                                    'processor_sha256', 'normalization_sha256', 'source_sha256')}
    task = None
    if text:
        spec = dict(schema=SCHEMA, family='SYNTHETIC_CPU_ONLY', model_sha256='b'*64,
            tokenizer_sha256='c'*64, encoder_source_sha256='d'*64, tap='language_backbone_last_hidden',
            pooling='valid_instruction_tokens_mean_float32', dimension=8, inputs=['instruction'],
            attention='bidirectional_text_only')
        cache = tmp_path/'task.json'
        cache.write_text(json.dumps(dict(spec=spec, instructions={instruction_key('fixture task'):
            dict(text='fixture task', vector=[1.]*8)})))
        task = dict(spec=spec, cache=str(cache), cache_sha256=file_digest(cache))
    contract = ScoreContract(10, 10, 16, 16)
    binding = make_binding(core_spec(variant, width=16), provenance, contract, task=task)

    def fk(angles):
        p = np.eye(4); p[:2,3] = np.asarray(angles)/100; return p

    class Backend:
        angles = np.zeros(2); epoch = 0; prompt = 'fixture task'
        def pose_receipt(self): return PoseReceipt(fk(self.angles), 'fixed-scene')
        def move_head_degrees(self, angles): self.angles = angles.copy(); return self.pose_receipt()
        def reset_observation_cache(self, epoch): self.epoch = epoch
        def capture_input(self, epoch):
            return {'observation/image':np.zeros((2,2,3),np.uint8),
                    'observation/state':np.zeros(29,np.float32), 'prompt':self.prompt}, self.epoch

    class Model:
        calls = 0; fail = False
        def infer(self, example, noise, *, request):
            self.calls += 1
            if self.fail: raise ConnectionError('unknown synthetic call')
            return dict(feature=np.ones(4608,np.float32), score=1.)
    model = Model(); model.contract = contract
    source = GR1SearchSource(ArticulatedHeadMount(LocalViewConfig(), fk,
        joint_limits_deg=[[-40,40],[-25,25]]), Backend(), model,
        noise_bank=np.zeros((2,16,32),np.float32), episode_id='synthetic-binding',
        model_sha256='a'*64, producer_sha256='a'*64, history_sha256='a'*64, guard=lambda:None)
    options = dict(initial_angles=[0,0], physical_state_id='fixed-scene', score_scale=1., seed=4)
    return binding, source, options


@pytest.mark.parametrize('variant', list(VARIANTS))
@pytest.mark.parametrize('text', [False, True])
def test_all_core_choices_and_text_roundtrip_real_entry(tmp_path, variant, text):
    b,s,options = fixture(tmp_path, variant, text)
    # File/IPC JSON roundtrip must not silently change tuple/list identity.
    b = json.loads(json.dumps(b))
    policy = build_policy(b)
    env,opts = bind_collector(s,b,instruction='fixture task',**options)
    obs,_ = env.reset(options=opts)
    assert ('task_features' in obs) == text
    assert obs['feature_mask'][0].tolist() == [True] and 'wrist' not in obs
    tensors = batch_observations([obs])
    (policy.actor_history(tensors).square().sum()+policy.critic_history(tensors).square().sum()).backward()
    for branch in (policy.actor_history, policy.critic_history):
        assert branch.adapter.projections['ego'][0].weight.grad is not None
        assert branch.core.joints[0].weight.grad is not None
        if text: assert branch.core.task_projection.weight.grad is not None
    assert not ({id(p) for p in policy.actor_history.parameters()} &
                {id(p) for p in policy.critic_history.parameters()})
    with torch.no_grad(): policy.stop_head.bias.fill_(8.)
    restored = load_policy(checkpoint_envelope(policy,b),b)
    torch.testing.assert_close(policy.value(tensors), restored.value(tensors),rtol=0,atol=0)
    _,fresh,options = fixture(tmp_path,variant,text)
    log=[]
    result=run_search_episode(fresh,restored,b,instruction='fixture task',session_options=options,
        max_decisions=8,max_seconds=30.,guard=lambda:None,clock=lambda:0.,persist=log.append)
    assert result['status']=='COMPLETE' and result['completed_model_calls']==2
    assert log[0]['status']=='INTENT' and log[-1]['status']=='COMPLETE'


def test_same_shape_wrong_provenance_rejected_before_weight_load(tmp_path):
    b,_,_=fixture(tmp_path)
    packet=checkpoint_envelope(build_policy(b),b)
    for key in b['provenance']:
        changed=deepcopy(b);changed['provenance'][key]='e'*64
        with pytest.raises(ValueError,match='provenance'): load_policy(packet,changed)


def test_missing_task_cache_and_changed_instruction_fail_before_model(tmp_path):
    b,s,options=fixture(tmp_path,text=True)
    with pytest.raises(ValueError,match='Unregistered instruction'):
        bind_collector(s,b,instruction='another task',**options)
    assert s.queries==0 and s.runtime.calls==0
    env,opts=bind_collector(s,b,instruction='fixture task',**options)
    s.backend.prompt='different actual observation task'
    with pytest.raises(ValueError,match='instruction changed'):env.reset(options=opts)
    assert s.runtime.calls==0 and s.status=='HOLD'


def test_actual_search_entry_holds_unknown_call_and_denied_guard(tmp_path):
    b,s,options=fixture(tmp_path);policy=build_policy(b);s.runtime.fail=True;log=[]
    with pytest.raises(ConnectionError):
        run_search_episode(s,policy,b,instruction='fixture task',session_options=options,
            max_decisions=8,max_seconds=30.,guard=lambda:None,clock=lambda:0.,persist=log.append)
    assert log[-1]['status']=='QUERY_UNKNOWN' and log[-1]['outcome']=='UNKNOWN_REQUIRES_AUDIT'
    assert s.status=='HOLD' and s.runtime.calls==1 and log[-1]['queries']==1
    with pytest.raises(RuntimeError):s.query()
    _,s,options=fixture(tmp_path)
    def denied():raise ConnectionError('authority disconnected')
    with pytest.raises(ConnectionError):
        run_search_episode(s,policy,b,instruction='fixture task',session_options=options,
            max_decisions=8,max_seconds=30.,guard=denied,clock=lambda:0.,persist=log.append)
    assert s.runtime.calls==0 and s.status=='HOLD'
