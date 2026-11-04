"""Actual LiveEpisodes -> native history -> unchanged PPO -> durable resume."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('gymnasium')
from AlphaBrain.research.dsol.data.native_features import layout_spec
from AlphaBrain.research.dsol.data.native_observation import NativeStream
from AlphaBrain.research.dsol.selectors.search_policy_factory import create_search_policy
from AlphaBrain.research.dsol.selectors.continuous_ppo import ContinuousPPO,ContinuousPPOConfig
from AlphaBrain.research.dsol.data.native_training import reward_config
from AlphaBrain.research.dsol.execution.manifest import digest
from benchmarks.LIBERO.continuous_training import LiveEpisodes
from tests.dsol_paper1.helpers.native_pipeline import Runtime,renderer
from tests.dsol_paper1.helpers.dispatch_policy import allow_synthetic_boundaries


def setup(tmp_path,monkeypatch,streams,*,warmup=False,limit=1026,sequence=False,core_variant=None):
    from benchmarks.LIBERO import search_live
    monkeypatch.setattr(search_live,'simulator',renderer)
    allow_synthetic_boundaries(monkeypatch,'benchmarks/LIBERO/continuous_training.py:LiveEpisodes.__init__')
    layout=layout_spec(tuple(streams))
    if sequence:layout.update(schema='NATIVE_SEQUENCE_FEATURE_LAYOUT_V1',pooling='explicit_ordered_query_groups_4')
    producer=dict(layout=layout)
    spec=dict(schema='NATIVE_CONTINUOUS_POLICY_V1',layout=layout,width=16,heads=4,layers=1,producer_sha256=digest(producer))
    if core_variant:
        from AlphaBrain.research.dsol.data.search_core import core_spec
        spec=dict(schema='UNIFIED_NATIVE_CONTINUOUS_POLICY_V1',layout=layout,
                  core=core_spec(core_variant,width=16),projection_groups=['visual']*len(streams),
                  producer_sha256=digest(producer))
    anchors=tmp_path/'noise.npz';np.savez(anchors,first_score_noise=np.zeros((1,2,10,7),np.float32))
    sim=tmp_path/'sim.json';sim.write_text(json.dumps(dict(anchors=str(anchors),operation_model='synthetic',models={'synthetic':dict(weights_sha256='e'*64)},native_features=producer)))
    cfg=dict(sim_config=str(sim),trajectory_cap=limit,native_policy=spec);plan=dict(actor=dict(score_scale=.16535300100378683))
    runtime=Runtime(layout,limit,warmup=warmup)
    return spec,cfg,plan,runtime


@pytest.mark.parametrize('core_variant',[None,'U0','U1','U2','U3'])
@pytest.mark.parametrize('streams',[
    (NativeStream('joint',7,'joint',('external','wrist'),('movable','context')),),
    (NativeStream('external',5,'per_view',('external',),('movable',)),NativeStream('wrist',5,'per_view',('wrist',),('context',))),
    (NativeStream('head',3,'per_view',('head',),('movable',)),NativeStream('left',4,'per_view',('left',),('context',)),NativeStream('right',9,'per_view',('right',),('context',)))])
def test_complete_native_batch_and_explicit_resume(tmp_path,monkeypatch,streams,core_variant):
    if core_variant and len(streams)>2:
        pytest.skip('Multiple context streams explicitly unsupported by the current shared core')
    torch.set_num_threads(1);torch.manual_seed(4)
    spec,cfg,plan,runtime=setup(tmp_path,monkeypatch,streams,warmup=True,core_variant=core_variant)
    schedule=[dict(episode=i,initial=0,angles_deg=[-17.5,9.5],proposal_seed=i) for i in range(64)]
    policy=create_search_policy(spec);initial=deepcopy(policy.state_dict());root=tmp_path/'run'
    settings=ContinuousPPOConfig(updates=2,n_epochs=1)
    binding=dict(source_kind='SYNTHETIC_NATIVE_CPU_ONLY',feature_contract=policy.feature_contract_sha256)
    live=LiveEpisodes(cfg,plan,root,runtime=runtime)
    with ContinuousPPO(policy,settings,schedule,live,root,binding=binding,reward_config=reward_config()) as trainer:
        trainer.run(through_update=1)
        assert trainer.cursor==32 and trainer.reward_state['batches']==1
        assert live.scoring.calls==2*trainer.queries+1
        obs=trainer.pending
        assert not obs
    assert any(not torch.equal(initial[k],policy.state_dict()[k]) for k in initial if isinstance(initial[k],torch.Tensor))
    query=next(root.glob('episodes/*-queries/query-01.npz'))
    with np.load(query,allow_pickle=False) as z:
        assert 'native_metadata_0' in z and 'wrist_0' not in z and 'external_0' not in z
    calls=live.scoring.calls
    second=Runtime(spec['layout'],cfg['trajectory_cap'],warmup=True)
    live2=LiveEpisodes(cfg,plan,root,runtime=second,expected_calls=calls)
    resumed=create_search_policy(spec)
    with ContinuousPPO(resumed,settings,schedule,live2,root,binding=binding,reward_config=reward_config(),resume=True) as trainer:
        trainer.run();assert trainer.cursor==64 and trainer.update==2
        assert trainer.reward_state['batches']==2 and live2.scoring.calls==2*trainer.queries+2


def test_native_query_failure_is_charged_and_stale_feature_is_not_retried(tmp_path,monkeypatch):
    streams=(NativeStream('joint',3,'joint',('external','wrist'),('movable','context')),)
    spec,cfg,plan,runtime=setup(tmp_path,monkeypatch,streams);runtime.stale=True
    root=tmp_path/'run';root.mkdir();live=LiveEpisodes(cfg,plan,root,runtime=runtime)
    with pytest.raises(ValueError,match='Stale'):
        with live(dict(episode=0,initial=0,angles_deg=[0,0],proposal_seed=0),root/'query'):pass
    assert runtime.calls==1
    assert len((root/'model-attempts.jsonl').read_text().splitlines())==1


def test_cpu_proof_cannot_admit_native_production(tmp_path):
    from AlphaBrain.research.dsol.data.native_training import validate_admission
    from AlphaBrain.research.dsol.execution.manifest import file_digest
    proof=tmp_path/'proof.json';proof.write_text(json.dumps(dict(status='PASS_NATIVE_FEATURE_TRAJECTORY',scope='CPU_SIMULATION')))
    with pytest.raises(ValueError,match='CPU simulation'):
        validate_admission(dict(native_admission=str(proof),input_hashes={str(proof):file_digest(proof)}),{},{} )


def test_actual_controlled_collection_and_cpu_learner_entry(tmp_path,monkeypatch):
    """Synthetic admission bypass is explicit; exercise the real run/step bodies.

    This never manufactures a REAL_NATIVE_MODEL proof. Production validation is
    separately tested to reject this synthetic setup.
    """
    from benchmarks.LIBERO import controlled_training as collection,search_runtime
    from benchmarks.LIBERO.controlled_learner import step
    from AlphaBrain.research.dsol.data import native_training
    from AlphaBrain.research.dsol.execution.manifest import file_digest,digest
    streams=(NativeStream('joint',3,'joint',('external','wrist'),('movable','context')),)
    spec,cfg,plan,_=setup(tmp_path,monkeypatch,streams,limit=32)
    spec.update(width=128,heads=4,layers=2)
    torch.manual_seed(6);initial=create_search_policy(spec);init=tmp_path/'initial.pt'
    torch.save(dict(optimizer_updates=0,feature_contract_sha256=initial.feature_contract_sha256,state_dict=initial.state_dict()),init)
    schedule=[dict(episode=i,initial=0,suite='libero_spatial',angles_deg=[-17.5,9.5],proposal_seed=i) for i in range(2048)]
    schedule_path=tmp_path/'schedule.json';schedule_path.write_text(json.dumps(schedule))
    protocol=dict(schema=native_training.PROTOCOL_SCHEMA,arms=['G'],seeds=[0,1,2],updates=64,episodes_per_update=32,
        query_cap=8,query_cost=.01,observation_scale=.16535300100378683,family='fastwam',native_policy=spec,
        initial_checkpoint=str(init),schedules={str(i):str(schedule_path) for i in range(3)},
        model_hashes={s:'e'*64 for s in ('libero_spatial','libero_object','libero_goal','libero_10')},
        runtime_contract_hashes={s:'f'*64 for s in ('libero_spatial','libero_object','libero_goal','libero_10')})
    proto=tmp_path/'protocol.json';proto.write_text(json.dumps(protocol))
    learner=dict(schema=native_training.LEARNER_SCHEMA,protocol=str(proto),native_policy=spec,arm='G',seed=0,updates=64,
        episodes_per_update=32,run='G-seed-0',schedule=str(schedule_path),initial_checkpoint=str(init),output=str(tmp_path/'learner'),
        input_hashes={str(p):file_digest(p) for p in (init,proto,schedule_path)})
    learner_path=tmp_path/'learner.json';learner_path.write_text(json.dumps(learner))
    result=step(learner_path,execute_cpu=True);assert result['version']==0
    publication=Path(learner['output'])/'publications/policy-0000.pt'
    allow_synthetic_boundaries(monkeypatch,'benchmarks/LIBERO/controlled_training.py:run')
    monkeypatch.setattr(search_runtime,'create_policy_runtime',lambda sim,output,limit:Runtime(spec['layout'],limit))
    records={}
    for index in range(16):
        block=dict(index=0,rows=schedule[index*2:index*2+2],host='synthetic',policy=str(publication),policy_sha256=file_digest(publication),
            version=0,seed=0,run='G-seed-0',reward_scale=native_training.reward_config()['fixed_scale'],
            limits=dict(score=32),sim_config=cfg['sim_config'])
        campaign=dict(schema=native_training.COLLECTION_SCHEMA,protocol=str(proto),native_policy=spec,kind='rollout',sim_config=cfg['sim_config'],
            observation_scale=plan['actor']['score_scale'],blocks=[block])
        path=tmp_path/f'campaign-{index}.json';path.write_text(json.dumps(campaign))
        monkeypatch.setattr(collection,'validate_campaign',lambda path,c=campaign:c)
        output=tmp_path/f'block-{index}'
        collection.run(collection.request_for(path,0),output,guard=lambda:None)
        value=json.loads((output/'collection.json').read_text());records.update(value['records'])
        assert value['training_updates']==0 and value['manipulation_actions']==0
        assert value['calls']==len((output/'model-attempts.jsonl').read_text().splitlines())
    batch=tmp_path/'records.json';batch.write_text(json.dumps(dict(run='G-seed-0',version=0,policy_sha256=file_digest(publication),records=records)))
    result=step(learner_path,records_path=batch,execute_cpu=True)
    assert result['version']==1 and result['episodes']==32 and result['operation_model_updates']==0
    saved=torch.load(Path(learner['output'])/'publications/policy-0001.pt',weights_only=True)
    assert saved['feature_contract_sha256']==digest(spec)
    with pytest.raises(ValueError,match='Stale/mixed'):step(learner_path,records_path=batch,execute_cpu=True)


def test_sequence_layout_reaches_real_collector_and_cpu_ppo(tmp_path,monkeypatch):
    torch.set_num_threads(1);torch.manual_seed(0)
    streams=(NativeStream('joint_cognition',7,'joint',('external','wrist'),('movable','context')),)
    spec,cfg,plan,runtime=setup(tmp_path,monkeypatch,streams,sequence=True)
    policy=create_search_policy(spec);schedule=[dict(episode=i,initial=0,angles_deg=[-17.,9.],proposal_seed=i) for i in range(32)]
    root=tmp_path/'sequence-run';live=LiveEpisodes(cfg,plan,root,runtime=runtime)
    with ContinuousPPO(policy,ContinuousPPOConfig(updates=1,n_epochs=1),schedule,live,root,
                       binding=dict(source_kind='SYNTHETIC_SEQUENCE_CPU_ONLY'),reward_config=reward_config()) as trainer:
        trainer.run();assert trainer.cursor==32 and trainer.update==1
    assert live.scoring.calls>64
    with np.load(next(root.glob('episodes/*-queries/query-01.npz')),allow_pickle=False) as z:
        meta=json.loads(str(z['native_metadata_0']));assert meta['layout']['schema']=='NATIVE_SEQUENCE_FEATURE_LAYOUT_V1'
