"""CPU proofs for two bound workers sharing one physical GPU."""
from copy import deepcopy
import json
import pytest
from AlphaBrain.research.dsol.execution import worker_gpu, gpu_guard
from AlphaBrain.research.dsol.execution.authority_wire import identity
from AlphaBrain.research.dsol.execution.manifest import digest
from AlphaBrain.research.dsol.execution.resources import worker_slots

@pytest.fixture
def case(tmp_path):
    (tmp_path/'tasks').mkdir()
    slots=worker_slots('fresh',[0],workers_per_gpu=2)
    binding=dict(tasks={'old':dict(eligible_slots=slots),'new':dict(eligible_slots=slots)})
    here=identity();store=dict(identity=here,binding_sha256=digest(binding),store_id='store')
    (tmp_path/'store.json').write_text(json.dumps(store))
    owner=dict(host='center',session='current',boot_id='boot')
    launch=dict(key='old',binding_sha256=digest(binding),operation='LIBERO_BLOCK',owner=owner,epoch=4,
                permit=dict(stage='formal'),slot=slots[0])
    row=dict(status='RUNNING',identity=here,store_id='store',starts=1,launch=launch,
             launch_sha256=digest(launch),process=dict(pid=100,pgid=100,start_ticks='120'))
    path=tmp_path/'tasks/old.json';path.write_text(json.dumps(row))
    new=deepcopy(launch);new.update(key='new',slot=slots[1])
    node=dict(workers_per_gpu=2,python_resolved='/python',python_sha256='python-hash',
              gpu_guard=dict(devices=[dict(index=0,uuid='GPU-0')]))
    proc=dict(pid=100,pgid=100,sid=100,start_ticks=120,state='S',executable='/python',executable_sha256='python-hash',
              argv=['/python','-B','-m','benchmarks.LIBERO.continuous_authority','--root',str(tmp_path)])
    return tmp_path,binding,new,node,row,path,proc

def test_bound_sibling_uses_existing_peak_reservation_guard(case,monkeypatch):
    root,binding,launch,node,row,path,proc=case
    monkeypatch.setattr(gpu_guard,'process',lambda pid:deepcopy(proc))
    owners=worker_gpu.owned_siblings(root,binding,launch,node)
    assert owners==[dict(pid=100,start=120,boot=identity()['boot'],group=100,gpu_uuid='GPU-0')]
    seen=[]
    monkeypatch.setattr(gpu_guard,'preflight',lambda config,*,owned,target_gpu_uuid:seen.append(owned) or dict(status='PASS'))
    assert worker_gpu.preflight(root,binding,launch,node)['status']=='PASS'
    assert seen==[owners]

@pytest.mark.parametrize('field,value', [('start_ticks',121),('pgid',101),('sid',101),('state','Z'),('executable_sha256','changed')])
def test_pid_reuse_or_changed_process_cannot_share_gpu(case,field,value):
    root,binding,launch,node,row,path,proc=case;proc[field]=value
    with pytest.raises(RuntimeError,match='identity'):
        worker_gpu.owned_siblings(root,binding,launch,node,process=lambda pid:proc)

@pytest.mark.parametrize('change', ['epoch','owner','unknown','slot','store'])
def test_foreign_or_unsettled_worker_is_not_whitelisted(case,change):
    root,binding,launch,node,row,path,proc=case
    if change=='epoch':row['launch']['epoch']=3
    if change=='owner':row['launch']['owner']['session']='old'
    if change=='unknown':row['status']='UNKNOWN'
    if change=='slot':row['launch']['slot']=launch['slot']
    if change=='store':row['store_id']='other'
    row['launch_sha256']=digest(row['launch']);path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError):worker_gpu.owned_siblings(root,binding,launch,node,process=lambda pid:proc)

def test_single_worker_binding_does_not_silently_become_dual(case):
    root,binding,launch,node,row,path,proc=case;node['workers_per_gpu']=1
    with pytest.raises(RuntimeError,match='capacity'):worker_gpu.owned_siblings(root,binding,launch,node,process=lambda pid:proc)

def test_physical_gpu_lock_serializes_both_slots(tmp_path):
    slots=worker_slots('fresh',[0],workers_per_gpu=2)
    with worker_gpu.launch_lock(tmp_path,slots[0]):
        with pytest.raises(BlockingIOError):
            with worker_gpu.launch_lock(tmp_path,slots[1]):pass
    with worker_gpu.launch_lock(tmp_path,slots[1]):pass

def test_dual_slot_order_fills_distinct_gpus_first():
    slots=worker_slots('node',range(8),workers_per_gpu=2)
    assert len(slots)==16 and len({s['id'] for s in slots})==16
    assert [s['gpu_id'] for s in slots]==list(map(str,range(8)))*2
    triple=worker_slots('node',range(8),workers_per_gpu=3)
    assert len(triple)==24 and len({s['id'] for s in triple})==24
    assert [s['gpu_id'] for s in triple]==list(map(str,range(8)))*3
    with pytest.raises(ValueError):worker_slots('node',[0],workers_per_gpu=4)


@pytest.mark.parametrize('own',[False,True])
def test_idle_resident_keeps_gpu_ownership_between_blocks(case,monkeypatch,own):
    root,binding,launch,node,row,path,proc=case
    binding['resident_workers']=True
    store=json.loads((root/'store.json').read_text());store['binding_sha256']=digest(binding)
    (root/'store.json').write_text(json.dumps(store))
    row['launch']['binding_sha256']=launch['binding_sha256']=digest(binding)
    row.update(status='IDLE',resident=True,launch_sha256=digest(row['launch']))
    (root/'residents').mkdir();(root/'residents/slot.json').write_text(json.dumps(row))
    if own:
        launch['slot']=row['launch']['slot'];monkeypatch.setattr(worker_gpu.os,'getpid',lambda:100)
    owners=worker_gpu.owned_siblings(root,binding,launch,node,process=lambda pid:proc)
    assert [p['pid'] for p in owners]==[100]
    row['status']='UNKNOWN';(root/'residents/slot.json').write_text(json.dumps(row))
    with pytest.raises(RuntimeError,match='Unsettled|idle'):
        worker_gpu.owned_siblings(root,binding,launch,node,process=lambda pid:proc)


def test_only_proven_idle_current_process_gets_continuation_accounting(case,monkeypatch):
    root,binding,launch,node,row,path,proc=case
    binding['resident_workers']=True
    store=json.loads((root/'store.json').read_text());store['binding_sha256']=digest(binding)
    (root/'store.json').write_text(json.dumps(store))
    row['launch']['binding_sha256']=launch['binding_sha256']=digest(binding)
    launch['slot']=row['launch']['slot'];row.update(status='IDLE',resident=True,launch_sha256=digest(row['launch']))
    (root/'residents').mkdir();record=root/'residents/slot.json';record.write_text(json.dumps(row))
    monkeypatch.setattr(worker_gpu.os,'getpid',lambda:100)
    monkeypatch.setattr(gpu_guard,'process',lambda pid:deepcopy(proc))
    calls=[]
    monkeypatch.setattr(gpu_guard,'preflight',lambda config,**kw:calls.append(kw) or dict(status='PASS'))
    assert worker_gpu.preflight(root,binding,launch,node)['status']=='PASS'
    assert calls[0]['continuing_pid']==100 and calls[0]['owned'][0]['pid']==100
    assert len(list((root/'preflights').glob('*.json')))==1
    row['status']='RUNNING';record.write_text(json.dumps(row))
    with pytest.raises(RuntimeError,match='idle'):worker_gpu.preflight(root,binding,launch,node)
