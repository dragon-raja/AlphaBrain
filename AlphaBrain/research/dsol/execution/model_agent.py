"""Shared bound-worker wire lifecycle; benchmark behavior is explicitly injected.

No benchmark imports, model construction or dispatch authority is supplied here.
Adapters own node validation, scientific execution and result verification.
"""
from contextlib import redirect_stderr, nullcontext
from copy import deepcopy
import fcntl
import json
import math
import os
import sys
from pathlib import Path
import time
import uuid
from .authority_wire import identity, require_local
from .manifest import digest, file_digest
from .queue import _durable_write


def execute_cpu_contract(binding,launch,guard,*,cache=None):
    checks=binding.get('cpu_contract_checks',2)
    if type(checks) is not int or not 2<=checks<=128:raise ValueError('Invalid CPU stress contract')
    for _ in range(checks-1):guard()
    if cache is not None:
        with cache.lease(digest(binding),object,lambda resource:None):pass
    counts=deepcopy(launch['limits'])
    return counts,digest(dict(synthetic=launch['key'],counts=counts))


def group_alive(owner):
    if not owner:
        return False
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
            if int(stat[2])==owner['pgid'] and stat[0] not in ('Z','X'):
                return True
        except FileNotFoundError:pass
    return False


def serve(root, binding, wire, adapter):
    require_local(root);host,node=adapter.verify_node(binding)
    store=json.loads((root/'store.json').read_text())
    if store.get('schema')!='MODEL_WORKER_STORE_V1' or store['identity']!=identity() or store['binding_sha256']!=digest(binding):
        raise RuntimeError('Worker store changed; explicit reconciliation required')
    session=uuid.uuid4().hex
    wire.send(dict(kind='HELLO',identity=identity(),store_id=store['store_id'],source=adapter.source_identity(),session=session))
    request=wire.receive()
    if request==dict(kind='INSPECT_SESSION',session=session):
        wire.send(dict(kind='INSPECTION_READY',session=session))
        while True:
            try:request=wire.receive()
            except ConnectionError:return
            if request==dict(kind='INSPECTION_PROBE',session=session):
                wire.send(dict(kind='INSPECTION_ALIVE',session=session))
                continue
            if (set(request)!={'kind','session','key'} or request.get('kind')!='INSPECT'
                    or request.get('session')!=session or request.get('key') not in binding['tasks']):
                raise RuntimeError('Inspection connection cannot launch a worker or change identity')
            # Retain interpreter/source checks and fresh local lock/result
            # validation on EVERY request. Only process/SSH setup is amortized.
            adapter.verify_node(binding)
            serve_task(root,binding,wire,request,session,store,adapter=adapter)
    if request==dict(kind='PROBE',session=session):
        while True:
            wire.send(dict(kind='HEALTHY',session=session))
            try:request=wire.receive()
            except ConnectionError:return
            if request!=dict(kind='PROBE',session=session):
                raise RuntimeError('Heartbeat connection cannot launch a worker')
    from AlphaBrain.research.dsol.execution.resident_session import ResidentSession
    from AlphaBrain.research.dsol.execution.resident_runtime import ResidentRuntimeCache
    resident = ResidentSession(root) if binding.get('resident_workers') and request.get('kind')=='PREPARE' else None
    cache = ResidentRuntimeCache() if resident else None
    simulator_pool=None
    if resident and binding.get('resident_simulator') and binding['mode']=='libero':
        simulator_pool=adapter.make_simulator_pool(root/'simulator-logs'/session)
    try:
        while True:
            row = serve_task(root,binding,wire,request,session,store,adapter=adapter,resident=resident,cache=cache,simulator_pool=simulator_pool)
            if row is None:return
            # Completion is emitted only after the task lock has been released.
            wire.send(dict(kind='DONE',session=session,row=row))
            if resident is None:return
            try:request=wire.receive()
            except ConnectionError:return
            while request==dict(kind='IDLE_PROBE',session=session):
                if time.time()>=row['launch']['permit']['finish_until']:
                    raise RuntimeError('Resident finish deadline expired')
                wire.send(dict(kind='IDLE',session=session))
                try:request=wire.receive()
                except ConnectionError:return
            if request.get('kind')!='PREPARE':
                raise RuntimeError('Resident channel only accepts the next fresh block')
    finally:
        if simulator_pool is not None:simulator_pool.close()
        if resident is not None:resident.close()


def serve_task(root,binding,wire,request,session,store,*,adapter,resident=None,cache=None,simulator_pool=None):
    key=request.get('key','')
    if key not in binding['tasks'] or request.get('session')!=session:
        raise ValueError('Unknown task or old session')
    path=root/'tasks'/(key+'.json')
    with (root/'locks'/(key+'.lock')).open('a') as lock:
        active=False
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:active=True
        if request.get('kind')=='INSPECT':
            row=json.loads(path.read_text()) if path.exists() else None
            resident_complete = (binding.get('resident_workers') and row is not None
                                 and row.get('resident') is True and row['status']=='COMPLETE')
            active = active or (row is not None and not resident_complete and group_alive(row.get('process')))
            if row and not active:adapter.verify_completed(binding,row)
            wire.send(dict(kind='STATE',session=session,key=key,active=active,row=row));return
        if active or request.get('kind')!='PREPARE' or path.exists():
            raise RuntimeError('Duplicate/active/unknown task cannot be replayed')
        from AlphaBrain.research.dsol.execution.worker_gpu import launch_lock
        launch=request['launch']
        if resident is not None:resident.check(launch)
        with (launch_lock(root,launch['slot']) if binding['mode']=='libero' or 'gpu_id' in launch['slot'] else nullcontext()):
            spec,_=adapter.validate_launch(binding,launch,root=root)
            stamp=digest(launch)
            row=dict(status='PREPARED',launch=deepcopy(launch),launch_sha256=stamp,session=session,
                     store_id=store['store_id'],identity=identity(),starts=0,resident=resident is not None)
            _durable_write(path,row)
            wire.send(dict(kind='READY',session=session,launch_sha256=stamp))
            if wire.receive()!=dict(kind='GO',session=session,launch_sha256=stamp):raise RuntimeError('Fresh GO required')
            adapter.validate_launch(binding,launch,root=root)
            if os.getpgrp()!=os.getpid():os.setsid()
            birth=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
            row.update(status='RUNNING',starts=1,process=dict(pid=os.getpid(),pgid=os.getpgrp(),start_ticks=birth))
            _durable_write(path,row)
            if resident is not None:resident.update(row,'RUNNING')
            wire.send(dict(kind='ACK',session=session,launch_sha256=stamp,row=row))
        sequence=0
        def guard():
            nonlocal sequence
            if time.time()>=launch['permit']['finish_until']:raise RuntimeError('Worker finish deadline expired')
            sequence+=1;wire.send(dict(kind='CHECK',session=session,sequence=sequence))
            if wire.receive()!=dict(kind='CONTINUE',session=session,sequence=sequence):
                raise RuntimeError('Live center continuation required')
        try:
            guard()
            if binding['mode']=='cpu_contract':
                executor=getattr(adapter,'execute_cpu_contract',execute_cpu_contract)
                counts,result_sha=executor(binding,launch,guard,cache=cache)
                if cache is not None:
                    row['resident_cache']=dict(loads=cache.loads,reuses=cache.reuses)
            else:
                with (root/'tasks'/(key+'.stderr.log')).open('x') as log, redirect_stderr(log):
                    counts,result_sha=adapter.execute_block(binding,launch,guard,runtime_cache=cache,simulator_pool=simulator_pool)
            row.update(status='COMPLETE',receipt=dict(simulation_only=binding['mode']=='cpu_contract',
                task_sha256=launch['task_sha256'],binding_sha256=digest(binding),budget_consumed=counts,
                scientific_success=None,result_sha256=result_sha),finished_at=time.time())
            _durable_write(path,row)
            if resident is not None:resident.update(row,'IDLE')
            return row
        except BaseException as error:
            row.update(status='UNKNOWN',error=type(error).__name__,automatic_retry=False)
            _durable_write(path,row)
            if resident is not None:resident.update(row,'UNKNOWN')
            raise


def node_for(binding):
    matches = [(host,node) for host,node in binding['nodes'].items() if node['identity'] == identity()]
    if len(matches) != 1:
        raise RuntimeError('Node identity differs from immutable binding')
    return matches[0]


def verify_node(binding, *, sources=True):
    host,node = node_for(binding)
    if str(Path(sys.executable).resolve()) != node['python_resolved'] or file_digest(Path(sys.executable).resolve()) != node['python_sha256']:
        raise RuntimeError('Interpreter identity changed')
    if sources:
        for entry in node['source_artifacts']:
            if file_digest(entry['path']) != entry['sha256']:
                raise RuntimeError('Worker execution source changed')
    return host,node


def prepare(root, binding):
    require_local(root.parent)
    from AlphaBrain.research.dsol.execution.local_ipc import probe_unix_listener
    # This runs for CPU contracts too, before a model permit or worker exists.
    probe_unix_listener()
    root.mkdir(exist_ok=False)
    _durable_write(root/'store.json',dict(schema='MODEL_WORKER_STORE_V1',store_id=uuid.uuid4().hex,
                                        identity=identity(),binding_sha256=digest(binding)))
    (root/'tasks').mkdir();(root/'locks').mkdir();(root/'residents').mkdir()


def validate_cpu_launch(binding, launch, *, root=None):
    host,node = verify_node(binding)
    if binding['mode'] != 'cpu_contract':raise ValueError('CPU binding required')
    real = False
    if (launch.get('operation') != ('LIBERO_BLOCK' if real else 'CPU_BLOCK_CONTRACT')
            or launch.get('binding_sha256') != digest(binding)
            or launch.get('resident_workers', False) is not binding.get('resident_workers', False)):
        raise ValueError('Worker mode/binding differs from live grant')
    task = launch['task']; key = launch['key']; spec = binding['tasks'][key]
    expected = dict(key=key, addresses=spec['addresses'], output=spec['output'],
                    identity_experiment_id=spec['identity_experiment_id'], eligible_slots=spec['eligible_slots'],
                    payload=dict(binding_sha256=digest(binding),request=spec['request']))
    if (task != expected or digest(task) != launch['task_sha256'] or spec['host'] != host
            or launch['slot'] not in spec['eligible_slots'] or launch['limits'] != spec['limits']):
        raise ValueError('Unbound block, host, slot, output or budget')
    permit=launch['permit'];now=time.time()
    times=[permit.get(k) for k in ('issued_at','dispatch_until','finish_until')]
    if any(type(t) not in (int,float) or not math.isfinite(t) for t in times):raise ValueError('Invalid grant clock')
    if (permit.get('authorization') != ('EXPLICIT_MODEL_STAGE_V1' if real else 'CPU_SIMULATION_ONLY')
            or permit.get('owner') != launch['owner'] or permit.get('epoch') != launch['epoch']
            or type(launch['epoch']) is not int or launch['epoch']<1 or launch['attempt']!=1
            or permit.get('stage') != spec['stage'] or not times[0] <= now < times[1] < times[2]
            or (real and permit.get('binding_sha256') != digest(binding))):
        raise RuntimeError('Current stage grant missing/expired')
    if Path(task['output']).exists():
        raise RuntimeError('Existing output is never overwritten or automatically replayed')
    return spec,node

