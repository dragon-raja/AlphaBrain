"""CPU-only cold-start scope and full-panel contracts."""
import json
from copy import deepcopy
from pathlib import Path
import pytest
from AlphaBrain.research.dsol.data import reward_cold_start as cold,reward_pilot,native_recovery
from AlphaBrain.research.dsol.execution.manifest import file_digest

@pytest.fixture
def protocol(tmp_path):
    def save(name,x):
        p=tmp_path/name;p.write_text(json.dumps(x));return str(p)
    initial=tmp_path/'initial.pt';initial.write_bytes(b'synthetic')
    rows=[dict(state_key=f's{i}',view_id=j,method='B-seed-0',suite='libero_spatial') for i in range(400) for j in range(8)]
    original=save('recovery.json',rows);limits=native_recovery.limits(cold.evaluation_rows(rows),paired_diagnostic=True)
    caps=dict(training=98304,evaluation=limits['score']+limits['paired_execution'],engineering=1024);caps['total']=sum(caps.values())
    auth=save('auth.json',dict(schema='USER_APPROVED_G_FULL_V1',call_caps=caps,user_approved_full_first_step=True))
    audit=save('audit.json',dict(status='PASS_B_REFERENCE_IDENTITY_AUDIT'))
    configs=[];schedules={};refs={}
    for seed in range(3):
        schedule=save(f'schedule-{seed}.json',[dict(episode=i,initial=i%400) for i in range(2048)])
        cfg=dict(run=f'B-seed-{seed}',schedule=schedule,initial_checkpoint=str(initial),input_hashes={schedule:file_digest(schedule),str(initial):file_digest(initial)})
        configs.append(dict(seed=seed,path=save(f'learner-{seed}.json',cfg)));schedules[str(seed)]=schedule
        refs[str(seed)]=save(f'first-{seed}.json',dict(records={}))
    p=dict(schema=cold.SCHEMA,arms=['G'],seeds=[0,1,2],updates=64,episodes_per_update=32,initial_update=0,initial_cursor=0,
        fixed_scale=reward_pilot.FIXED_SCALE,query_cap=8,query_cost=.01,floor_fraction=.1,diagnostic_calls=4,episodes_per_block=2,
        schedules=schedules,initial_checkpoints=[dict(seed=s,path=str(initial)) for s in range(3)],original_learners=configs,
        first_batch_references=refs,authorization=auth,reuse_audit=audit,original_recovery_schedule=original,call_caps=caps)
    paths=[auth,audit,original,str(initial),*schedules.values(),*refs.values(),*[c['path'] for c in configs]]
    p['input_hashes']={f:file_digest(f) for f in paths};path=save('protocol.json',p)
    return Path(path),p

def test_full_cold_contract_and_panel(protocol):
    path,p=protocol
    assert reward_pilot.load_protocol(path,file_digest(path))==p
    rows=reward_pilot.evaluation_rows(p,json.loads(Path(p['original_recovery_schedule']).read_text()))
    assert len(rows)==9600 and {r['method'] for r in rows}=={f'G-seed-{s}' for s in range(3)}

@pytest.mark.parametrize('key,value',[('arms',['B','G']),('updates',8),('initial_update',64),('query_cost',.001),('episodes_per_block',4)])
def test_scope_cannot_silently_change(protocol,key,value):
    path,p=protocol;p=deepcopy(p);p[key]=value;path.write_text(json.dumps(p))
    with pytest.raises(ValueError):reward_pilot.load_protocol(path,file_digest(path))

def test_budget_cannot_silently_increase(protocol):
    path,p=protocol;p['call_caps']['training']+=1;path.write_text(json.dumps(p))
    with pytest.raises(ValueError):reward_pilot.load_protocol(path,file_digest(path))

def test_schedule_must_equal_original_prefix(protocol):
    path,p=protocol
    target=path.parent/'reordered.json';rows=json.loads(Path(p['schedules']['0']).read_text());rows[0],rows[1]=rows[1],rows[0];target.write_text(json.dumps(rows))
    p['schedules']['0']=str(target);p['input_hashes'][str(target)]=file_digest(target);path.write_text(json.dumps(p))
    with pytest.raises(ValueError):reward_pilot.load_protocol(path,file_digest(path))

def test_cold_orchestrator_names_and_full_barrier(protocol):
    from benchmarks.LIBERO.reward_adaptation import Adaptation,run_names
    path,p=protocol;assert len(run_names())==9 and len(run_names(p))==3
    runner=Adaptation(dict(output=str(path.parent/'out'),protocol=str(path),input_hashes={str(path):file_digest(path)},local_root=str(path.parent/'local')))
    assert runner.updates==64 and runner.names==['G-seed-0','G-seed-1','G-seed-2']

def test_actual_one_shot_runs_64_barriers_and_full_evaluation(protocol,monkeypatch):
    from benchmarks.LIBERO import reward_adaptation as entry
    path,p=protocol;root=path.parent/'run';root.mkdir()
    cfg=dict(output=str(root),protocol=str(path),input_hashes={str(path):file_digest(path)},local_root=str(path.parent/'local'),baseline_references=str(path))
    runner=entry.Adaptation(cfg);collected=[];updated=[];stages=[]
    monkeypatch.setattr(runner,'initialize',lambda:None)
    monkeypatch.setattr(runner,'batch',lambda version:collected.append(version) or path)
    monkeypatch.setattr(runner,'update',lambda version:updated.append(version))
    monkeypatch.setattr(runner,'stage',lambda phase,campaign,**kwargs:stages.append((phase,kwargs)))
    monkeypatch.setattr(runner,'prepare_evaluation',lambda:[dict(phase='eval-000',path=str(path),searches=3),dict(phase='eval-001',path=str(path),searches=9597)])
    monkeypatch.setattr(entry,'recovery_gate',lambda *args:None)
    runner.run();state=json.loads((root/'current.json').read_text())
    assert collected==updated==list(range(64))
    assert len([x for x in stages if x[0].startswith('train-')])==64
    assert state['completed_searches']==9600 and state['completed_operation_episodes']==19200 and state['dispatch_enabled'] is False
