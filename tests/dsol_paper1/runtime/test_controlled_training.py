"""CPU manifests and synthetic real-entry collection; no GPU or real simulator."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from benchmarks.LIBERO import controlled_training as adapter
from AlphaBrain.research.dsol.execution.manifest import file_digest,digest


def save(path,value):path.write_text(json.dumps(value));return str(path)


@pytest.fixture
def campaign(tmp_path):
    model=tmp_path/'model';model.mkdir();(model/'model.safetensors').write_bytes(b'synthetic')
    weights=file_digest(model/'model.safetensors')
    inputs=save(tmp_path/'inputs.json',dict(records=[dict(state=dict(pair_key=f'state-{i}',task_id='task',split='train')) for i in range(400)]))
    noise=tmp_path/'noise.npz';np.savez(noise,first_score_noise=np.zeros((400,2,10,7),dtype=np.float32))
    sim=save(tmp_path/'sim.json',dict(inputs=inputs,anchors=str(noise),operation_model='native_reference',
        models=dict(native_reference=dict(path=str(model),weights_sha256=weights,implementation='openpi_native'))))
    protocol=save(tmp_path/'protocol.json',dict(synthetic=True))
    requests=[dict(address=f'initial-{i}',initial=i//2,state_key=f'state-{i//2}',task='task',split='train',angles_deg=[-15.,5.]) for i in range(800)]
    schedule=save(tmp_path/'requests.json',requests)
    blocks=[dict(index=i,host='node',rows=requests[i*4:i*4+4],limits=dict(episodes=0,score=8,paired_execution=0,nominal_execution=0)) for i in range(200)]
    c=dict(schema=adapter.SCHEMA,kind='calibration',operation_model='native_reference',model_sha256=weights,
        query_cost=.01,query_cap=8,observation_scale=.16535300100378683,sim_config=sim,training_inputs=inputs,
        training_noise=str(noise),protocol=protocol,calibration_requests=schedule,blocks=blocks,
        limits=dict(episodes=0,score=1600,paired_execution=0,nominal_execution=0))
    c['input_hashes']={p:file_digest(p) for p in [sim,inputs,str(noise),protocol,schedule,str(model/'model.safetensors')]}
    return Path(save(tmp_path/'campaign.json',c)),c


def test_exact_calibration_contract(campaign):
    path,c=campaign;assert adapter.validate_campaign(path)==c


@pytest.mark.parametrize('fault',[None,'old_schema','raw','asset','weights','model'])
def test_target_model_requires_explicit_bound_domain_contract(monkeypatch,fault):
    from benchmarks.LIBERO.legacy import fastwam_runtime
    contract=dict(checkpoint='target.pt',input_hashes={'target.pt':'weights','visual.pt':'visual'})
    monkeypatch.setattr(fastwam_runtime,'validate_contract',lambda p,h:contract)
    c=dict(operation_model='fastwam',model_sha256='weights',
        input_hashes={'runtime.json':'runtime','target.pt':'weights','visual.pt':'visual'})
    sim=dict(operation_model='fastwam',raw_observation=fastwam_runtime.RAW_OBSERVATION,
        models=dict(fastwam=dict(path='target.pt',weights_sha256='weights',implementation='fastwam_native',
            runtime_contract='runtime.json',runtime_contract_sha256='runtime')))
    if fault=='raw':sim.pop('raw_observation')
    if fault=='asset':c['input_hashes'].pop('visual.pt')
    if fault=='weights':c['model_sha256']='wrong'
    if fault=='model':c['operation_model']='native_reference'
    if fault:
        with pytest.raises(ValueError):adapter.validate_operation_binding(c,sim,domain=fault!='old_schema')
    else:adapter.validate_operation_binding(c,sim,domain=True)


@pytest.mark.parametrize('fault',['missing','duplicate','heldout','cost','cap','unbound','wrongstate','wrongbudget'])
def test_collection_contract_fails_closed(campaign,fault):
    path,c=campaign;c=deepcopy(c)
    if fault=='missing':c['blocks'].pop()
    if fault=='duplicate':c['blocks'][1]['rows'][0]=c['blocks'][0]['rows'][0]
    if fault=='heldout':c['blocks'][0]['rows'][0]['split']='test'
    if fault=='cost':c['query_cost']=.001
    if fault=='cap':c['query_cap']=9
    if fault=='unbound':del c['input_hashes'][c['training_inputs']]
    if fault=='wrongstate':c['blocks'][0]['rows'][0]['state_key']='different'
    if fault=='wrongbudget':c['limits']['score']+=1
    save(path,c)
    with pytest.raises(ValueError):adapter.validate_campaign(path)


def test_direct_entry_has_no_production_permission(tmp_path):
    from AlphaBrain.research.dsol.execution.dispatch_policy import ProductionDispatchDisabled
    with pytest.raises(ProductionDispatchDisabled):adapter.run({},tmp_path/'out',guard=lambda:None)
    assert not (tmp_path/'out').exists()


@pytest.fixture
def synthetic_runtime(monkeypatch):
    from contextlib import contextmanager
    from benchmarks.LIBERO import search_runtime,search_live
    from tests.dsol_paper1.helpers.dispatch_policy import allow_synthetic_boundaries
    allow_synthetic_boundaries(monkeypatch,'benchmarks/LIBERO/controlled_training.py:run',
                              'benchmarks/LIBERO/continuous_training.py:LiveEpisodes.__init__')
    class Runtime:
        def __init__(self,sim,output,limit):self.calls=0;self.limit=limit
        def infer(self,example,noise,identity,capture=True):
            self.calls+=1
            return dict(actions=np.zeros((10,7)),trace=np.zeros((10,10,7)),score=.01,
                        external=np.ones(4608),wrist=np.ones(4608))
    @contextmanager
    def simulator(path,config,initial,output):
        current=np.eye(4)
        def receipt():return dict(camera_to_world=current.copy(),physical_state_id='synthetic',settled=True)
        def query(message):
            nonlocal current
            if message['kind']=='continuous_info':return dict(reference=np.eye(4),receipt=receipt())
            if message['kind']=='move_pose':
                current=np.array(message['camera_to_world']);return dict(receipt=receipt())
            assert message['kind']=='query_pose'
            return dict(receipt=receipt(),example=dict(image=[current.copy(),np.ones((2,2))],state=np.zeros(7),
                camera_intrinsics=np.eye(3),camera_to_world_opencv=current.copy(),instruction='synthetic',lang='synthetic'))
        yield query
    monkeypatch.setattr(search_runtime,'create_policy_runtime',Runtime)
    monkeypatch.setattr(search_live,'simulator',simulator)


def test_actual_calibration_entry_with_fake_vla_and_renderer(campaign,tmp_path,synthetic_runtime):
    path,_=campaign;request=adapter.request_for(path,0);out=tmp_path/'out';grants=[]
    adapter.run(request,out,guard=lambda:grants.append(True))
    result=json.loads((out/'collection.json').read_text())
    assert result['calls']==8 and len(result['records'])==4 and len(grants)==8
    assert all(r['score_members']==[.01,.01] for r in result['records'])
    artifacts={str(p.relative_to(out)):file_digest(p) for p in out.rglob('*') if p.is_file()}
    save(out/'complete.json',dict(request_sha256=digest(request),artifacts=artifacts))
    assert adapter.validate_block(request,out)['calls']==dict(score=8,execution=0)
    (out/'collection.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact'):adapter.validate_block(request,out)


@pytest.mark.parametrize('fault',['visual','heldout','schedule','hash','batch_size'])
def test_domain_protocol_rejects_unbound_or_incompatible_inputs(campaign,tmp_path,fault):
    from AlphaBrain.research.dsol.data import domain_adaptation
    from tests.dsol_paper1.helpers.domain_adaptation import contract
    _,c=campaign
    initial=tmp_path/'initial.pt';initial.write_bytes(b'not loaded during validation')
    rows=[dict(episode=i,initial=i,state_key=f'state-{i}',task='task') for i in range(64)]
    schedule=save(tmp_path/'schedule.json',rows)
    protocol=contract(tmp_path,c,schedule,initial)
    cfg=dict(protocol=protocol,adaptation_protocol=protocol,input_hashes={protocol:file_digest(protocol)})
    assert domain_adaptation.from_campaign(cfg)['episodes_per_update']==8
    p=json.loads(Path(protocol).read_text())
    if fault=='visual':p['source_checkpoints']['0']['visual_contract_sha256']='0'*64
    if fault=='batch_size':p['episodes_per_update']=32
    if fault=='hash':initial.write_bytes(b'changed checkpoint')
    if fault=='heldout':
        states=json.loads(Path(c['training_inputs']).read_text());states['records'][0]['state']['split']='test'
        save(Path(c['training_inputs']),states);p['input_hashes'][c['training_inputs']]=file_digest(c['training_inputs'])
    if fault=='schedule':
        rows[0]['state_key']='different';save(Path(schedule),rows);p['input_hashes'][schedule]=file_digest(schedule)
    save(Path(protocol),p);cfg['input_hashes'][protocol]=file_digest(protocol)
    with pytest.raises(ValueError):domain_adaptation.from_campaign(cfg)


@pytest.mark.parametrize('arm',[None,'B','G','L','source-window','source-first_batch_fixed',
                              'random-window','random-first_batch_fixed'])
def test_actual_rollout_and_cpu_optimizer_boundary(campaign,tmp_path,synthetic_runtime,arm):
    torch=pytest.importorskip('torch');pytest.importorskip('gymnasium')
    from benchmarks.LIBERO.controlled_learner import step
    from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
    torch.set_num_threads(2);torch.manual_seed(0)
    path,c=campaign
    from AlphaBrain.research.dsol.data import reward_pilot,domain_adaptation
    domain=arm in domain_adaptation.ARMS
    batch_size=8 if domain else 32
    initial=tmp_path/'initial.pt';policy=ContinuousActorCritic(4608)
    if arm is None:
        torch.save(dict(optimizer_updates=0,visual_dim=4608,state_dict=policy.state_dict()),initial)
    else:
        from AlphaBrain.research.dsol.data.continuous_training_store import rng_state
        torch.save(dict(policy=policy.state_dict(),optimizer=torch.optim.Adam(policy.parameters(),lr=3e-4,eps=1e-5).state_dict(),
            update=64,cursor=2048,queries=5000,rng=rng_state()),initial)
    rows=[dict(episode=i,source_episode=i,initial=i%400,state_key=f'state-{i%400}',task='task',
               angles_deg=[-15.,5.],proposal_seed=i) for i in range(8192 if arm is None else 64 if domain else 256)]
    schedule=save(tmp_path/'schedule.json',rows)
    cfg=dict(schema='CONTROLLED_PPO_LEARNER_V1',arm='B',seed=0,run='B-seed-0',
        updates=256,episodes_per_update=32,schedule=schedule,initial_checkpoint=str(initial),
        protocol=c['protocol'],output=str(tmp_path/'learner'))
    if domain:
        from tests.dsol_paper1.helpers.domain_adaptation import contract
        protocol=contract(tmp_path,c,schedule,initial)
        cfg.update(schema=domain_adaptation.LEARNER_SCHEMA,arm=arm,run=f'{arm}-seed-0',updates=8,
            episodes_per_update=8,protocol=protocol,adaptation_protocol=protocol)
        c.update(schema=domain_adaptation.COLLECTION_SCHEMA,protocol=protocol,adaptation_protocol=protocol,
            settle_reward_at_learner=True)
    elif arm is not None:
        from tests.dsol_paper1.helpers.reward_pilot import contract
        protocol=contract(tmp_path,schedule,initial)
        cfg.update(schema=reward_pilot.LEARNER_SCHEMA,arm=arm,run=f'{arm}-seed-0',updates=8,
            protocol=protocol,adaptation_protocol=protocol)
        c.update(schema=reward_pilot.COLLECTION_SCHEMA,protocol=protocol,adaptation_protocol=protocol,
            settle_reward_at_learner=True)
    cfg['input_hashes']={p:file_digest(p) for p in [schedule,str(initial),c['protocol']]}
    config=save(tmp_path/'learner.json',cfg)
    publication=step(config,execute_cpu=True)
    assert publication['version']==0 and publication['episodes']==0
    calibration=save(tmp_path/'scale.json',dict(schema='FIXED_MODEL_REWARD_SCALE_V1',count=800,
        model_sha256=c['model_sha256'],fixed_during_training=True,scale=.01))
    c.update(kind='rollout',calibration_receipt=calibration,schedules={'0':schedule} if arm is None else {str(s):schedule for s in (0,1,2)},blocks=[],
             limits=dict(episodes=batch_size,score=batch_size*16,paired_execution=0,nominal_execution=0))
    for i in range(batch_size//4):
        c['blocks'].append(dict(index=i,host='node',rows=rows[i*4:i*4+4],run=cfg['run'],arm=cfg['arm'],seed=0,version=0,
            policy=publication['policy'],policy_sha256=publication['policy_sha256'],
            reward_scale=c['observation_scale'] if domain else .01 if arm is None else reward_pilot.FIXED_SCALE,
            limits=dict(episodes=4,score=64,paired_execution=0,nominal_execution=0)))
    c['input_hashes'].update({p:file_digest(p) for p in [schedule,calibration,publication['policy'],c['protocol']]});save(path,c)
    assert adapter.validate_campaign(path)==c
    records={}
    for i in range(batch_size//4):
        output=tmp_path/f'block-{i}'
        adapter.run(adapter.request_for(path,i),output,guard=lambda:None)
        result=json.loads((output/'collection.json').read_text());records.update(result['records'])
    if arm=='L':
        reused={f"{cfg['run']}:{i}":records[str(i)] for i in range(4)}
        proof=dict(status='VERIFIED_STOPPED_PARTIAL_COLLECTION',all_workers_stopped=True,
            automatic_retry=False,held_failed_phase_cap=4608,resume_update=0,records=reused)
        recovery=save(tmp_path/'recovery.json',proof);c['collection_recovery']=recovery
        c['input_hashes'].update({ref['path']:ref['sha256'] for ref in reused.values()})
        for missing in (None,3):
            if missing is not None:proof['records'].pop(f"{cfg['run']}:{missing}")
            save(Path(recovery),proof);c['input_hashes'][recovery]=file_digest(recovery);save(path,c)
            adapter.validate_campaign(path)
            output=tmp_path/f'recovered-{missing}'
            adapter.run(adapter.request_for(path,0),output,guard=lambda:None)
            recovered=json.loads((output/'collection.json').read_text())
            expected=0 if missing is None else 2*torch.load(records[str(missing)]['path'],weights_only=True)['metrics']['queries']
            assert recovered['calls']==expected
            for i in range(4):
                if i!=missing:assert recovered['records'][str(i)]==records[str(i)]
        proof['all_workers_stopped']=False;save(Path(recovery),proof)
        c['input_hashes'][recovery]=file_digest(recovery);save(path,c)
        with pytest.raises(ValueError,match='stopped-worker'):adapter.validate_campaign(path)
    batch=dict(run=cfg['run'],version=0,policy_sha256=publication['policy_sha256'],records=records)
    batch_path=save(tmp_path/'batch.json',batch)
    published=step(config,records_path=batch_path,execute_cpu=True)
    assert published['version']==1 and published['episodes']==batch_size and published['operation_model_updates']==0
    assert file_digest(published['policy'])==published['policy_sha256']
    if arm is not None:
        receipt=json.loads((tmp_path/'learner/updates/001.json').read_text())['reward_settlement']
        if domain:
            assert receipt['mode']==domain_adaptation.ARMS[arm][1]
            assert receipt['count']==8 and receipt['denominator']==.01
            assert not published['initialization']['critic_copied']
            # Records are version-bound: a partial or duplicate set cannot
            # enter another optimizer update and must not advance publication.
            batch['version']=1;batch['policy_sha256']=published['policy_sha256']
            batch['records'].pop('7');save(Path(batch_path),batch)
            with pytest.raises(ValueError):step(config,records_path=batch_path,execute_cpu=True)
            batch['version']=0;save(Path(batch_path),batch)
        else:
            assert receipt['kind']==arm and receipt['count']==32 and receipt['scale']==.01
        assert receipt['additional_model_calls']==0
    with pytest.raises(ValueError,match='Stale'):step(config,records_path=batch_path,execute_cpu=True)
    assert json.loads((tmp_path/'learner/publication.json').read_text())==published
