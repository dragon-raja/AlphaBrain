#!/usr/bin/env python3
"""Execution-only extension: frozen episodes, dynamic local work queue."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import queue
import signal
import socket
import sys
import time

ROOT=Path('/share/longjunyu/alphabrain/experiments/dsol-standard-initial-aa0-v1-20260908')
sys.path.insert(0,str(ROOT/'repo/scripts/dsol_paper1'))
import standard_initial_aa0_v1 as standard
core=standard.core
EXT=ROOT/'scheduling/dynamic-v1'
FROZEN_ENTRY=ROOT/'repo/scripts/dsol_paper1/standard_initial_aa0_v1.py'


def consume(q, callback):
    while True:
        try:index=q.get_nowait()
        except queue.Empty:return
        try:callback(index)
        finally:q.task_done()


def signature(row):
    return (row['aa0']['trace_sha256'],row['success'],row['completion_steps'],
            row['aa0']['counts']['calls'],row['aa0']['counts']['steps'])


def verify_extension():
    m=core.read(EXT/'manifest.json')
    core.require(core.sha(Path(__file__))==m['scheduler_sha256'],'Scheduler source changed')
    core.require(core.sha(ROOT/'release.json')==m['scientific_release_sha256'],'Scientific release changed')
    return m


def matrix(host,r,phase,indices):
    core.require(len(indices)==len(set(indices)),'Duplicate requested index')
    for model in ['canonical','broad']:
        todo=[i for i in indices if not core.valid(core.output_path(host,phase,model,i),r,model,i)]
        if not todo:continue
        servers=[]; q=queue.Queue()
        for i in todo:q.put_nowait(i)
        done=len(indices)-len(todo)
        audit=EXT/host/f'{phase}-{model}-assignments.jsonl';audit.parent.mkdir(parents=True,exist_ok=True)
        try:
            for j in range(16):
                port=r['base_port']+j
                with socket.socket() as s:core.require(s.connect_ex(('127.0.0.1',port))!=0,'Port occupied')
                servers.append(core.spawn([core.python(),core.REPO/'scripts/cabi_vla/serve_alphabrain_pi05_websocket.py',
                    '--checkpoint',r['models'][model]['path'],'--port',port,'--device','cuda:0','--cpu-threads',2],
                    EXT/host/'logs'/f'{phase}-{model}-server-{j}.log',core.env(gpu=j//2)))
            started=time.time()
            while True:
                core.require(all(p.poll() is None for p in servers),'Policy startup failed')
                ready=[]
                for j in range(16):
                    with socket.socket() as s:ready.append(s.connect_ex(('127.0.0.1',r['base_port']+j))==0)
                if all(ready):break
                core.require(time.time()-started<360,'Policy startup timeout');time.sleep(2)
            def worker(w):
                nonlocal done
                # Shift GPU mapping during gates to test migration, not only replay original affinity.
                server=(w+8)%16 if phase.startswith('schedule-gate') else w%16
                def execute(index):
                    nonlocal done
                    core.require(not core.STOP.is_set(),'Stopped')
                    path=core.output_path(host,phase,model,index)
                    stamp=time.time()
                    core.finish(core.spawn([core.python(True),FROZEN_ENTRY,'episode','--host',host,'--model',model,
                        '--index',index,'--gpu',server//2,'--port',r['base_port']+server,'--output',path],
                        EXT/host/'logs'/f'{phase}-{model}-worker-{w}.log',core.sim_env()),1800)
                    core.require(core.valid(path,r,model,index),'Missing completed episode')
                    row=core.read(path)
                    if phase.startswith('schedule-gate'):
                        old=core.read(core.output_path(host,'gate-a',model,index))
                        core.require(signature(row)==signature(old),f'Dynamic/static trace mismatch: {host}/{model}/{index}')
                    with core.LOCK:
                        import json
                        with audit.open('a') as f:
                            f.write(json.dumps(dict(index=index,worker=w,gpu=server//2,start_utc=stamp,end_utc=time.time(),
                                episode_sha256=core.sha(path),scheduler_sha256=core.sha(Path(__file__))))+'\n')
                        done+=1
                        if done%32==0:core.status(host,phase,model=model,model_completed=done,model_budget=len(indices),scheduler='dynamic-v1',pending=q.qsize())
                consume(q,execute)
            with ThreadPoolExecutor(max_workers=32) as pool:
                fs=[pool.submit(worker,w) for w in range(32)]
                try:
                    for f in as_completed(fs):f.result()
                except BaseException:
                    core.STOP.set()
                    for p in list(core.ACTIVE):core.clean(p)
                    raise
        finally:
            for p in servers:core.clean(p)
    # Existing successful phase receipts are left byte-for-byte intact on resume.
    receipt=ROOT/'hosts'/host/(phase+'-complete.json')
    if not receipt.exists():core.write(receipt,dict(release_sha256=core.sha(ROOT/'release.json'),episodes=len(indices)*2,
                                    utc=time.time(),scheduler='dynamic-v1',execution_manifest_sha256=core.sha(EXT/'manifest.json')))


def check_gates(r):
    for h in ['fresh','gnu']:
        receipt=EXT/h/'gate-pass.json'
        if not receipt.exists():return False
        a=core.read(receipt)
        core.require(a['execution_manifest_sha256']==core.sha(EXT/'manifest.json'),'Gate manifest differs')
    for h in ['fresh','gnu']:
        for phase in ['schedule-gate-dynamic-v1-a','schedule-gate-dynamic-v1-b']:
            for model in ['canonical','broad']:
                for i in r['gate_indices']:
                    p=core.output_path(h,phase,model,i)
                    core.require(core.valid(p,r,model,i),'Gate record missing')
                    core.require(signature(core.read(p))==signature(core.read(core.output_path(h,'gate-a',model,i))),
                                 f'Gate mismatch {h}/{phase}/{model}/{i}')
    return True


def run(host):
    import fcntl
    verify_extension();r=core.release()
    with (ROOT/'hosts'/host/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        core.status(host,'VERIFYING_DYNAMIC_SCHEDULER')
        core.require(core.check_gate(r),'Original scientific gate invalid')
        for p,d in r['initial_source_hashes'].items():core.require(core.sha(p)==d,'Initial source changed')
        for m in r['models'].values():core.require(core.sha(Path(m['path'])/'model.safetensors')==m['weights_sha256'],'Model changed')
        bank=core.read(r['noise_bank']['path']);core.require(core.sha(bank['noise_file'])==r['noise_bank']['file_sha256'],'Noise changed')
        for phase in ['schedule-gate-dynamic-v1-a','schedule-gate-dynamic-v1-b']:
            core.status(host,phase,scheduler='dynamic-v1');matrix(host,r,phase,r['gate_indices'])
        core.write(EXT/host/'gate-pass.json',dict(status='PASS_LOCAL_DYNAMIC_VS_STATIC_FULL_TRACE',episodes=192,
            execution_manifest_sha256=core.sha(EXT/'manifest.json'),utc=time.time()))
        core.status(host,'WAITING_PEER_DYNAMIC_GATE');start=time.time()
        while not check_gates(r):
            core.require(time.time()-start<3600,'Peer dynamic gate timed out');time.sleep(5)
        core.write(EXT/host/'cross-host-gate-pass.json',dict(status='PASS_DYNAMIC_REPEAT_CROSS_HOST_AND_STATIC',
             execution_manifest_sha256=core.sha(EXT/'manifest.json'),utc=time.time()))
        owned=r['hosts'][host]['state_indices']
        for wave in range(4):
            subset=[i for i in owned if r['states'][i]['initialization']['init_state_index']==wave]
            indices=[s*3104+offset for offset in range(3104) for s in subset]
            phase=f'dense-init-{wave}';core.status(host,phase,scheduler='dynamic-v1');matrix(host,r,phase,indices)
        core.status(host,'COMPLETE',dense_episodes=len(owned)*3104*2,scheduler='dynamic-v1')
        core.write(ROOT/'hosts'/host/'dense-complete.json',dict(episodes=len(owned)*3104*2,utc=time.time(),
            release_sha256=core.sha(ROOT/'release.json'),execution_manifest_sha256=core.sha(EXT/'manifest.json')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--host',choices=['fresh','gnu'],required=True);a=p.parse_args()
    def interrupted(*_):core.STOP.set();raise InterruptedError('Owned dynamic scheduler interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:run(a.host)
    except BaseException as e:core.status(a.host,'FAILED_DYNAMIC_SCHEDULER_REVIEW',error=str(e));raise
    finally:
        core.STOP.set()
        for child in list(core.ACTIVE):core.clean(child)


if __name__=='__main__':main()
