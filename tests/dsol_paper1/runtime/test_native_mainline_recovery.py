"""Native mainline controlled evaluation with synthetic sensor/runtime/operation."""
import json
from pathlib import Path
import numpy as np
import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('gymnasium')
from AlphaBrain.research.dsol.data.native_features import layout_spec
from AlphaBrain.research.dsol.data.native_observation import NativeStream
from AlphaBrain.research.dsol.selectors.search_policy_factory import create_search_policy
from AlphaBrain.research.dsol.execution.manifest import file_digest,digest
from tests.dsol_paper1.helpers.native_pipeline import Runtime,renderer
from tests.dsol_paper1.helpers.dispatch_policy import allow_synthetic_boundaries


@pytest.mark.parametrize('family',['fastwam','groot_n15'])
@pytest.mark.parametrize('method',['stay','reference','G-seed-0','local','random_from_local'])
def test_real_evaluation_entry_keeps_accounting_and_native_layout(tmp_path,monkeypatch,family,method):
    from benchmarks.LIBERO import controlled_target_recovery as entry,native_recovery,search_runtime,search_live,local_search_rollout
    from AlphaBrain.research.dsol.data import flow_noise
    streams=((NativeStream('joint',3,'joint',('external','wrist'),('movable','context')),) if family=='fastwam' else
             (NativeStream('external',3,'per_view',('external',),('movable',)),NativeStream('wrist',3,'per_view',('wrist',),('context',))))
    producer=dict(layout=layout_spec(streams))
    policy=dict(schema='NATIVE_CONTINUOUS_POLICY_V1',layout=layout_spec(streams),width=16,heads=4,layers=1,producer_sha256=digest(producer))
    torch.set_num_threads(1);actor=create_search_policy(policy);publication=tmp_path/'actor.pt'
    torch.save(dict(version=64,state_dict=actor.state_dict(),feature_contract_sha256=actor.feature_contract_sha256),publication)
    sim=tmp_path/'sim.json';sim.write_text(json.dumps(dict(operation_contract=dict(horizons={'libero_spatial':220}),operation_model='synthetic',models={'synthetic':dict(weights_sha256='e'*64)},native_features=producer)))
    from AlphaBrain.research.dsol.data.native_training import publication_origin
    initial=tmp_path/'initial.txt';initial.write_text('synthetic initialization identity')
    protocol=dict(family=family,seeds=[0,1,2],native_policy=policy,initial_checkpoint=str(initial),query_cap=8,query_cost=.01,
        model_hashes={s:'e'*64 for s in ('libero_spatial','libero_object','libero_goal','libero_10')},
        runtime_contract_hashes={s:'f'*64 for s in ('libero_spatial','libero_object','libero_goal','libero_10')})
    proto=tmp_path/'proto.json';proto.write_text(json.dumps(protocol))
    torch.save(dict(version=64,state_dict=actor.state_dict(),feature_contract_sha256=actor.feature_contract_sha256,
        training_protocol_sha256=file_digest(proto),origin=publication_origin(protocol,file_digest(proto),0)),publication)
    noise=tmp_path/'noise.npz';np.savez(noise,noise=np.zeros((1,2,10,7),np.float32))
    row=dict(trial_id='trial',state_key='state',initial=0,start=0,view_id=1,task='task',suite='libero_spatial',
        method=method,angles_deg=[-17.5,9.5],proposal_seed=7,random_seed=8,query_cap=8,max_steps=220)
    block=dict(index=0,rows=[row],host='synthetic',sim_config=str(sim),limits=dict(episodes=2,score=(0 if method in ('stay','reference') else 20)+int(family=='groot_n15'),paired_execution=44,nominal_execution=0))
    c=dict(schema=native_recovery.SCHEMA,purpose='native_mainline_evaluation',native_policy=policy,protocol=str(proto),sim_config=str(sim),score_noise=str(noise),
        execution_noise='unused-synthetic',score_scale=.16535300100378683,blocks=[block],actors={'G-seed-0':dict(path=str(publication),version=64,seed=0)})
    path=tmp_path/'campaign.json';path.write_text(json.dumps(c))
    # Explicitly bypass only asset/GPU admission for synthetic wiring, never
    # construct or persist an admission proof accepted by production.
    monkeypatch.setattr(entry,'validate_campaign',lambda path:c)
    monkeypatch.setattr(search_live,'simulator',renderer)
    monkeypatch.setattr(flow_noise,'ExplicitFlowNoiseBank',lambda path:object())
    monkeypatch.setattr(search_runtime,'create_policy_runtime',lambda sim,output,limit:Runtime(policy['layout'],limit,warmup=family=='groot_n15',output=output))
    def rollout(query,infer,example,bank,key,repeat,row,**kwargs):
        infer(example,np.zeros((10,7),np.float32),dict(repeat=repeat,purpose='synthetic_operation'),capture=False)
        return dict(success=True,replans=1,input_sha256=[local_search_rollout.input_identity(example)]),dict(actions=np.zeros((1,7)))
    monkeypatch.setattr(local_search_rollout,'rollout_fastwam_episode',rollout)
    monkeypatch.setattr(local_search_rollout,'rollout_groot_episode',rollout)
    allow_synthetic_boundaries(monkeypatch,'benchmarks/LIBERO/controlled_target_recovery.py:run')
    request=entry.request_for(path,0);out=tmp_path/'output';entry.run(request,out,guard=lambda:None)
    files={str(p.relative_to(out)):file_digest(p) for p in out.rglob('*') if p.is_file()}
    (out/'complete.json').write_text(json.dumps(dict(request_sha256=digest(request),artifacts=files)))
    result=entry.validate_block(request,out)
    assert result['calls']['execution']==2
    attempts=[json.loads(line) for line in (out/'model-attempts.jsonl').read_text().splitlines()]
    diagnostics=sum(r['identity'].get('purpose')=='native_diagnostic' for r in attempts)
    assert diagnostics==(0 if method in ('stay','reference') else 4)
    assert all('external_0' not in np.load(p,allow_pickle=False).files for p in out.glob('case-*/query-*.npz'))
    if method=='G-seed-0':
        # Same real tensors and shapes, different preprocessing: actual loader rejects.
        payload=torch.load(publication,map_location='cpu',weights_only=True)
        payload['origin']['runtime_contract_hashes']['libero_spatial']='a'*64
        torch.save(payload,publication)
        with pytest.raises(ValueError,match='origin'):entry.load_actors(c,[row])
