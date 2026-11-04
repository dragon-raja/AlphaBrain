"""Composition regression: a node binding cannot inject forbidden env keys."""
from benchmarks.LIBERO.controlled_campaign import bind_node,adapter_for,campaign_nodes


def test_canonical_evidence_path_and_reject_nonlocal_phase_components(tmp_path):
    import pytest
    from benchmarks.LIBERO.controlled_campaign import phase_evidence
    cfg=dict(output=str(tmp_path))
    assert phase_evidence(cfg,'train-0001')==tmp_path/'execution/train-0001-real'
    assert phase_evidence(cfg,'probe',cpu=True)==tmp_path/'execution/probe-cpu'
    for phase in ('','..','../phase','/phase'):
        with pytest.raises(ValueError):phase_evidence(cfg,phase)


def test_collection_receipts_cannot_cross_requests_or_hide_missing_rollouts(tmp_path):
    import json
    import pytest
    from benchmarks.LIBERO.controlled_campaign import collection_records,phase_evidence
    from benchmarks.LIBERO.controlled_training import request_for
    from AlphaBrain.research.dsol.execution.manifest import digest,file_digest
    def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    cfg=dict(output=str(tmp_path/'output'));campaign=tmp_path/'campaign.json'
    write(campaign,dict(blocks=[dict(index=0,host='h',rows=[dict(episode=0)])]))
    root=phase_evidence(cfg,'train-0000');block=root/'results/block-0000';sample=tmp_path/'synthetic-record.pt';sample.write_bytes(b'synthetic')
    write(root/'completion.json',dict(status='COMPLETE',dispatch_enabled=False,counts=dict(episodes=1)))
    write(block/'collection.json',dict(status='COMPLETE',records={'0':dict(path=str(sample),sha256=file_digest(sample))}))
    receipt=dict(request_sha256=digest(request_for(campaign,0)),artifacts={'collection.json':file_digest(block/'collection.json')});write(block/'complete.json',receipt)
    assert collection_records(cfg,'train-0000',campaign)['0']['path']==str(sample)
    receipt['request_sha256']='0'*64;write(block/'complete.json',receipt)
    with pytest.raises(ValueError,match='request/receipt'):collection_records(cfg,'train-0000',campaign)


def test_native_resident_slots_separate_runtime_identities_without_host_migration(tmp_path):
    import json
    import pytest
    from benchmarks.LIBERO.controlled_campaign import resident_runtime_slots,campaign_capacity
    paths=[]
    for i,value in enumerate([{'model':'a'},{'model':'a'},{'model':'b'}]):
        p=tmp_path/f'{i}.json';p.write_text(json.dumps(value));paths.append(str(p))
    _,slots=campaign_capacity({'capacity':{h:dict(gpu_ids=[0,1],workers_per_gpu=2) for h in ['one','two']}},['one','two'])
    blocks=[dict(index=i,host='one' if i<6 else 'two',sim_config=paths[i%3]) for i in range(12)]
    c=dict(native_policy={'explicit':True},blocks=blocks);out=resident_runtime_slots(c,slots)
    for h in slots:
        a={s['id'] for b in blocks if b['host']==h and b['index']%3!=2 for s in out[b['index']]}
        b={s['id'] for b in blocks if b['host']==h and b['index']%3==2 for s in out[b['index']]}
        assert a and b and not a&b and a|b=={s['id'] for s in slots[h]}
        assert all(s['host']==h for block in blocks if block['host']==h for s in out[block['index']])
    assert out[0]==out[1]  # Same content at two paths is one runtime.
    with pytest.raises(ValueError,match='Not enough resident slots'):
        resident_runtime_slots(c,{h:values[:1] for h,values in slots.items()})
    legacy=dict(blocks=blocks)
    assert resident_runtime_slots(legacy,slots)[0]==slots['one']


def test_explicit_fresh_capacity_does_not_inherit_native_dual_worker():
    import pytest
    from benchmarks.LIBERO.controlled_campaign import campaign_capacity
    c={'capacity':{'fresh':{'gpu_ids':[2,3],'workers_per_gpu':1}}}
    cap,slots=campaign_capacity(c,{'fresh'})
    assert [x['id'] for x in slots['fresh']]==['2','3']
    node=bind_node(dict(source='/old',environment={'PYTHONPATH':'/old'}),
        dict(release='/new',artifacts=[]),workers_per_gpu=cap['fresh']['workers_per_gpu'])
    assert node['workers_per_gpu']==1
    with pytest.raises(ValueError):campaign_capacity(c,{'fresh','gnu'})
    c['capacity']['fresh']['gpu_ids']=[2,2]
    with pytest.raises(ValueError):campaign_capacity(c,{'fresh'})


def test_tail_wave_selects_capacity_without_moving_tasks_or_mutating_fleet():
    from copy import deepcopy
    from benchmarks.LIBERO.controlled_campaign import campaign_capacity
    fleet={h:dict(gpu_ids=list(range(8)),workers_per_gpu=1)
           for h in ('fresh','cy','gnu','video')}
    original=deepcopy(fleet)
    tasks=[dict(host='cy',trial_id='last-block')]
    nodes=campaign_nodes({h:{} for h in fleet},tasks,{h:h for h in fleet})
    capacity,slots=campaign_capacity(dict(capacity=fleet),nodes)
    assert set(capacity)==set(slots)=={'cy'}
    assert len(slots['cy'])==8 and all(s['host']=='cy' for s in slots['cy'])
    assert tasks==[dict(host='cy',trial_id='last-block')] and fleet==original
    capacity['cy']['gpu_ids'].append(8)
    assert fleet==original


def test_capacity_rejects_empty_or_unadmitted_tail_hosts():
    import pytest
    from benchmarks.LIBERO.controlled_campaign import campaign_capacity
    config=dict(capacity={'cy':dict(gpu_ids=[0],workers_per_gpu=1)})
    for hosts in (set(),{'cy','unknown'},{'unknown'}):
        with pytest.raises(ValueError,match='cover the selected hosts'):
            campaign_capacity(config,hosts)


def test_partial_fleet_waves_preserve_each_declared_worker_limit():
    from benchmarks.LIBERO.controlled_campaign import campaign_capacity
    fleet={'one':dict(gpu_ids=[0,2],workers_per_gpu=1),
           'two':dict(gpu_ids=[1],workers_per_gpu=2),
           'three':dict(gpu_ids=[3],workers_per_gpu=1)}
    for hosts in ({'one'},{'one','two'},set(fleet)):
        capacity,slots=campaign_capacity(dict(capacity=fleet),hosts)
        assert set(capacity)==set(slots)==hosts
        assert all(len(slots[h])==len(fleet[h]['gpu_ids'])*fleet[h]['workers_per_gpu']
                   for h in hosts)


def test_node_binding_preserves_channel_allowlist_and_does_not_mutate_template():
    original=dict(source='/frozen/old',environment=dict(PYTHONPATH='/frozen/old:/dependencies',HF_HUB_OFFLINE='1'))
    result=bind_node(original,dict(release='/frozen/new',artifacts=[]))
    assert result['environment']==dict(PYTHONPATH='/frozen/new:/dependencies',HF_HUB_OFFLINE='1')
    assert original['source']=='/frozen/old' and original['environment']['PYTHONPATH'].startswith('/frozen/old:')
    assert result['workers_per_gpu']==2
    assert 'WANDB_MODE' not in result['environment'] and 'CUDA_VISIBLE_DEVICES' not in result['environment']


def test_only_registered_collection_and_recovery_adapters():
    import pytest
    for kind in ('training','recovery'):
        adapter,_=adapter_for(kind)
        assert callable(adapter.validate_campaign) and callable(adapter.request_for)
    with pytest.raises(ValueError):adapter_for('direct_worker')


def test_two_node_phase_does_not_probe_unused_video_or_reassign_its_tasks():
    import pytest
    template={h:dict(source='/frozen',environment={}) for h in ('fresh','gnu','video')}
    aliases=dict(fresh='ai2r-fresh',gnu='gnu-a',video='ai2r-video')
    selected=campaign_nodes(template,[dict(host='fresh'),dict(host='gnu')],aliases,['video'])
    assert set(selected)=={'fresh','gnu'} and 'video' in template
    selected['fresh']['environment']['test']='independent'
    assert template['fresh']['environment']=={}
    for tasks in ([],[dict(host='video')],[dict(host='unbound')]):
        with pytest.raises(ValueError,match='no automatic reassignment'):
            campaign_nodes(template,tasks,aliases,['video'])


def test_recovery_composition_binds_exact_model_alias_paths(tmp_path):
    import json
    from benchmarks.LIBERO.controlled_recovery import common_asset_hashes
    from AlphaBrain.research.dsol.execution.manifest import file_digest
    def save(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return str(p)
    weights=save('model/model.safetensors',{'dummy':True})
    norm=save('model/assets/physical-intelligence/libero/norm_stats.json',{})
    inputs=save('inputs.json',{});noise=save('noise.json',{});design=save('design.json',{})
    sim=save('sim.json',dict(inputs=inputs,operation_model='native',models={'native':dict(path=str(tmp_path/'model'),
        weights_sha256=file_digest(weights),implementation='openpi_native')}))
    paired=save('paired.json',dict(design=design,cohorts={'A':dict(sim_config=sim,execution_noise=noise)}))
    baseline=save('baseline.json',dict(paired_config=paired,score_noise=noise,input_hashes={}))
    hashes=common_asset_hashes(baseline)
    assert set(hashes)=={baseline,paired,noise,sim,inputs,design,weights,norm}
    assert hashes[weights]==file_digest(weights) and hashes[norm]==file_digest(norm)


def test_new_model_explicit_phase_window_does_not_change_cpu_or_extend_deadline():
    import pytest
    from benchmarks.LIBERO.controlled_campaign import phase_duration
    c=dict(dispatch_deadline_utc_seconds=50000,phase_timeout_seconds=21600)
    assert phase_duration(c,132,cpu=False,now=10000)==21600
    assert phase_duration(c,132,cpu=False,now=49000)==1000
    assert phase_duration(c,132,cpu=True,now=10000)==1800
    assert phase_duration(dict(dispatch_deadline_utc_seconds=50000),132,cpu=False,now=10000)==7200
    for bad in (True,0,59,86401,1.5):
        with pytest.raises(ValueError):phase_duration(dict(c,phase_timeout_seconds=bad),132,cpu=False,now=10000)
    with pytest.raises(ValueError):phase_duration(c,132,cpu=False,now=50000)


def test_shared_node_control_root_keeps_experiment_and_phase_namespace(tmp_path):
    from benchmarks.LIBERO.controlled_campaign import remote_control_base
    node=dict(control_root=str(tmp_path/'owned-node-root'))
    a=remote_control_base(dict(experiment_id='run-a'),node,tmp_path/'center-a/paired-evaluation-real','cy')
    b=remote_control_base(dict(experiment_id='run-b'),node,tmp_path/'center-b/paired-evaluation-real','cy')
    assert a!=b
    assert a==tmp_path/'owned-node-root/run-a/paired-evaluation-real/cy'
    assert b==tmp_path/'owned-node-root/run-b/paired-evaluation-real/cy'
    assert a!=remote_control_base(dict(experiment_id='run-a'),node,tmp_path/'center-a/train-0004-real','cy')
    assert not a.exists() and not b.exists()  # Pure path binding, no remote write.
    assert remote_control_base(dict(experiment_id='run-a'),{},tmp_path/'center-a/phase','fresh')==tmp_path/'center-a/phase/fresh'


def test_control_root_rejects_escaping_names_without_creating_files(tmp_path):
    import pytest
    from benchmarks.LIBERO.controlled_campaign import remote_control_base
    node=dict(control_root=str(tmp_path/'owned'))
    for experiment in ('','..','../other','nested/run'):
        with pytest.raises(ValueError):remote_control_base(dict(experiment_id=experiment),node,tmp_path/'phase','cy')
    with pytest.raises(ValueError):remote_control_base(dict(experiment_id='run'),dict(control_root='relative'),tmp_path/'phase','cy')
    with pytest.raises(ValueError):remote_control_base(dict(experiment_id='run'),node,tmp_path/'phase','../cy')
    assert not (tmp_path/'owned').exists()
