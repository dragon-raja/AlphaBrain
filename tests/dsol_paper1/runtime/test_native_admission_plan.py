"""CPU-only bounded concurrent admission planning; no model/simulator launch."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from benchmarks.LIBERO import native_feature_admission as entry
from AlphaBrain.research.dsol.execution.manifest import digest,file_digest


def test_two_workers_keep_separate_budgets_and_unique_training_states(tmp_path,monkeypatch):
    from benchmarks.LIBERO import model_adapters,search_runtime
    def save(name,obj):
        p=tmp_path/name;p.write_text(json.dumps(obj));return str(p)
    states=save('states.json',dict(records=[dict(state=dict(split='train',pair_key=f's{i}',suite='libero_spatial')) for i in (0,1)]))
    small=save('asset.json',{})
    from AlphaBrain.research.dsol.data.flow_noise import materialize_bank
    bank=materialize_bank(output_dir=tmp_path/'noise',bank_id='cpu',state_keys=['s0','s1'],repeat_count=1,max_replans=40,action_horizon=10,action_dim=7,root_seed=17)
    producer=dict(layout=['synthetic'])
    sim=save('sim.json',dict(inputs=states,anchors=small,continuous_camera=True,continuous_execution='PAIRED_EVALUATION_ONLY',native_features=producer,operation_contract=dict(executed_actions_per_replan=10,horizons={'libero_spatial':400})))
    monkeypatch.setattr(model_adapters,'model_adapter',lambda family:SimpleNamespace(implementation='synthetic',warmup_calls=0,suite_specific=False,validate_runtime=lambda *args:dict(input_hashes={})))
    monkeypatch.setattr(search_runtime,'selected_checkpoint',lambda *args,**kw:dict(implementation='synthetic',runtime_contract=small,runtime_contract_sha256=file_digest(small)))
    rows=[dict(initial=i,state_key=f's{i}',suite='libero_spatial',task='synthetic') for i in (0,1)]
    cfg=dict(schema=entry.SCHEMA,family='synthetic',sim_config=sim,execution_noise=small,native_policy=dict(layout=producer['layout'],producer_sha256=digest(producer)),execution_hosts=['synthetic'],input_hashes={p:file_digest(p) for p in (states,small,sim)},blocks=[dict(index=i,host='synthetic',rows=[r],limits=dict(episodes=1,score=18,paired_execution=120,nominal_execution=0)) for i,r in enumerate(rows)],limits=dict(episodes=2,score=36,paired_execution=240,nominal_execution=0))
    cfg['execution_noise']=bank['manifest_path'];cfg['input_hashes'].update({p:file_digest(p) for p in (bank['manifest_path'],bank['noise_file'])})
    path=save('campaign.json',cfg)
    assert entry.validate_campaign(path)==cfg
    assert entry.request_for(path,1)['block']['rows']==[rows[1]]
    cfg['blocks'][1]['rows']=[rows[0]];save('campaign.json',cfg)
    with pytest.raises(ValueError,match='unique'):entry.validate_campaign(path)


@pytest.mark.parametrize('execution_calls',[0,24])
def test_actual_authority_accounts_admission_trajectory_calls(tmp_path,monkeypatch,execution_calls):
    from benchmarks.LIBERO import controlled_admission,continuous_authority
    out=tmp_path/'result';limits=dict(episodes=1,score=19,paired_execution=66,nominal_execution=0)
    request=dict(block=dict(limits=limits))
    binding=dict(mode='libero',scientific_adapter='controlled_admission_v1',tasks={'key':dict(request=request,output=str(out))})
    launch=dict(key='key',slot={'gpu_id':'0'},operation='LIBERO_BLOCK',permit=dict(authorization='EXPLICIT_MODEL_STAGE_V1'))
    def run(req,output,**kwargs):
        output.mkdir();(output/'synthetic.json').write_text('{}')
    monkeypatch.setattr(controlled_admission,'run',run)
    monkeypatch.setattr(controlled_admission,'validate_block',lambda *args:dict(calls=dict(score=19,execution=execution_calls),receipt_sha256='synthetic'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.setenv('WANDB_MODE','disabled')
    counts,_=continuous_authority.execute_block(binding,launch,lambda:None)
    assert counts==dict(episodes=1,score=19,paired_execution=execution_calls,nominal_execution=0)


@pytest.mark.parametrize('fault',['state','horizon','shape','hash'])
def test_execution_noise_coverage_rejected_on_cpu(tmp_path,fault):
    from AlphaBrain.research.dsol.data.flow_noise import materialize_bank
    bank=materialize_bank(output_dir=tmp_path/'noise',bank_id='cpu',state_keys=['s0'],repeat_count=1,max_replans=2,action_horizon=10,action_dim=8 if fault=='shape' else 7,root_seed=17)
    c=dict(execution_noise=bank['manifest_path'],input_hashes={bank['noise_file']:('wrong' if fault=='hash' else bank['noise_file_sha256'])})
    sim=dict(operation_contract=dict(executed_actions_per_replan=10,horizons={'synthetic':30 if fault=='horizon' else 20}))
    rows=[dict(state_key='absent' if fault=='state' else 's0',suite='synthetic')]
    with pytest.raises((KeyError,IndexError,ValueError)):entry.validate_execution_noise(c,sim,rows)
@pytest.mark.parametrize('suite_specific,expected_calls',[(True,4),(False,1)])
def test_native_collection_runtime_verification_is_invocation_local(monkeypatch,suite_specific,expected_calls):
    """Many blocks verify each runtime once, but later invocations recheck bytes."""
    from types import SimpleNamespace
    from collections import defaultdict
    from benchmarks.LIBERO import native_collection as entry,model_adapters,search_runtime
    from AlphaBrain.research.dsol.data import native_training
    from AlphaBrain.research.dsol.selectors import search_policy_factory
    suites=['libero_spatial','libero_object','libero_goal','libero_10']
    h='a'*64;calls=[];changed=False
    model=lambda s:dict(implementation='synthetic',weights_sha256=h,
        runtime_contract=s if suite_specific else 'shared',runtime_contract_sha256=h)
    def validate(path,stamp):
        calls.append(path)
        if changed:raise ValueError('Changed runtime bytes')
        return dict(suite=path,input_hashes={path+'-weight':h})
    adapter=SimpleNamespace(implementation='synthetic',suite_specific=suite_specific,warmup_calls=0,validate_runtime=validate)
    monkeypatch.setattr(model_adapters,'model_adapter',lambda family:adapter)
    monkeypatch.setattr(search_runtime,'selected_checkpoint',lambda sim,**kw:model(sim['suite']))
    monkeypatch.setattr(search_policy_factory,'validate_policy_spec',lambda p:None)
    monkeypatch.setattr(native_training,'validate_admission',lambda *a:None)
    policy=dict(width=128,heads=4,layers=2)
    protocol=dict(family='synthetic',model_hashes=dict.fromkeys(suites,h),runtime_contract_hashes=dict.fromkeys(suites,h),schedules={'0':'schedule'},observation_scale=.16535300100378683)
    monkeypatch.setattr(native_training,'from_campaign',lambda c:protocol)
    states=[dict(state=dict(split='train',pair_key=str(i),task_id='task')) for i in range(400)]
    rows=[dict(episode=i,initial=i,state_key=str(i),task='task',suite=suites[i%4]) for i in range(32)]
    docs={'states':dict(records=states),'schedule':rows}
    docs.update({s+'-sim':dict(suite=s,operation_model='synthetic',inputs='states',anchors='noise') for s in suites})
    c=dict(schema=native_training.COLLECTION_SCHEMA,kind='rollout',schedules=protocol['schedules'],query_cap=8,query_cost=.01,
        observation_scale=.16535300100378683,settle_reward_at_learner=True,native_policy=policy,model_hashes=protocol['model_hashes'],
        training_inputs='states',training_noise='noise',protocol='protocol',operation_model='synthetic',
        sim_configs={s:s+'-sim' for s in suites},native_admissions={s:s+'-proof' for s in suites},blocks=[])
    assets={'states','noise','protocol','schedule','actor',*c['sim_configs'].values(),*c['native_admissions'].values(),*[model(s)['runtime_contract'] for s in suites]}
    c['input_hashes']=dict.fromkeys(assets,h)
    for s in suites:
        selected=[r for r in rows if r['suite']==s]
        for i in range(0,8,2):
            c['blocks'].append(dict(index=len(c['blocks']),rows=selected[i:i+2],seed=0,version=0,arm='G',run='G-seed-0',policy='actor',policy_sha256=h,
                reward_scale=native_training.reward_config()['fixed_scale'],sim_config=s+'-sim',limits=dict(episodes=2,score=32,paired_execution=0,nominal_execution=0)))
    c['limits']=dict(episodes=32,score=512,paired_execution=0,nominal_execution=0);docs['campaign']=c
    monkeypatch.setattr(entry,'read',lambda path:docs[path])
    monkeypatch.setattr(entry,'file_digest',lambda path:h)
    assert entry.validate_campaign('campaign') is c
    assert len(calls)==expected_calls
    changed=True
    with pytest.raises(ValueError,match='Changed runtime bytes'):entry.validate_campaign('campaign')
    assert len(calls)==expected_calls+1


def test_native_recovery_requires_stopped_audit_same_actor_and_bound_records(tmp_path):
    import json
    import copy
    import pytest
    from benchmarks.LIBERO.native_collection import validate_recovery
    from AlphaBrain.research.dsol.execution.manifest import file_digest
    def save(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return str(p)
    protocol=save('protocol.json',{});actor='a'*64
    block=dict(run='G-seed-0',version=0,policy_sha256=actor,rows=[dict(episode=0,initial=7)])
    c=dict(protocol=protocol,native_policy={'layout':'native'},blocks=[block],input_hashes={})
    old=save('old.json',c);audit=save('audit.json',dict(all_workers_stopped=True));record=save('record.pt',{})
    proof=dict(schema='NATIVE_COLLECTION_RECOVERY_V1',status='VERIFIED_STOPPED_PARTIAL_COLLECTION',all_workers_stopped=True,
        automatic_retry=False,protocol_sha256=file_digest(protocol),held_score_cap=33,source_campaign=old,process_audit=audit,
        resume_version=0,records={'G-seed-0:0':dict(path=record,sha256=file_digest(record))})
    recovery=save('recovery.json',proof);c['collection_recovery']=recovery
    c['input_hashes']={p:file_digest(p) for p in [recovery,old,audit,record]};validate_recovery(c)
    for field,value in [('all_workers_stopped',False),('automatic_retry',True),('protocol_sha256','b'*64),('resume_version',1)]:
        changed=dict(proof,**{field:value});save('recovery.json',changed);c['input_hashes'][recovery]=file_digest(recovery)
        with pytest.raises(ValueError):validate_recovery(c)
    save('recovery.json',proof);c['input_hashes'][recovery]=file_digest(recovery)
    changed=copy.deepcopy(c);changed['blocks'][0]['policy_sha256']='b'*64
    with pytest.raises(ValueError,match='actor'):validate_recovery(changed)
    save('record.pt',{'changed':True})
    with pytest.raises(ValueError,match='artifact'):validate_recovery(c)
