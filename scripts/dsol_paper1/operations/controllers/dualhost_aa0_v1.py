#!/usr/bin/env python3
"""Isolated AA0 full64 evaluation. Never signals keepalive or foreign jobs."""
from __future__ import annotations

from pathlib import Path as _RepoPath
import sys as _sys
_REPOSITORY_ROOT = _RepoPath(__file__).resolve().parents[4]
if str(_REPOSITORY_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPOSITORY_ROOT))

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import tempfile
import shutil
import numpy as np
from scripts.dsol_paper1.runtime.shared_runtime_paths import shared_scripts

ROOT=Path('/share/longjunyu/alphabrain/experiments/dsol-aa0-full64-dualhost-v1-20260908')
REPO=ROOT/'repo'
RUNTIME=Path('/share/longjunyu/alphabrain/datasets/libero-plus/runtime/LIBERO-plus')
SOURCE=Path('/workspace/projects/alphabrain-dsol-paper1')
# Import implementation from this checkout. REPO identifies frozen experiment
# assets / replay subprocesses; it must not silently replace local modules.
sys.path.insert(0,str(shared_scripts(_REPOSITORY_ROOT)))
from scripts.dsol_paper1.runtime.render_bridge_common_v1 import read, sha, array_identity, render_protocol
from scripts.dsol_paper1.runtime.evaluate_dsol_libero_hdf5_views import protocol_spec_at, protocol_spec_count
LOCK=threading.RLock(); ACTIVE=[]; STOP=threading.Event()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
    temp=path.with_name(path.name+'.tmp-'+str(os.getpid())+'-'+str(threading.get_ident()))
    with temp.open('x') as f:f.write(payload)
    os.replace(temp,path)

def require(ok,message):
    if not ok:raise ValueError(message)

def release():
    r=read(ROOT/'release.json')
    for p,digest in r['source_hashes'].items():require(sha(p)==digest,'Frozen source changed: '+p)
    return r

def env(sim=False,gpu=None):
    e=os.environ.copy()
    e.update(OMP_NUM_THREADS='1' if sim else '2',OPENBLAS_NUM_THREADS='1' if sim else '2',MKL_NUM_THREADS='1' if sim else '2',
             TOKENIZERS_PARALLELISM='false',PRETRAINED_MODELS_DIR='/share/longjunyu/alphabrain/pretrained_models',ALPHABRAIN_DISABLE_AUTO_DOWNLOAD='1',
             PYTHONPATH=':'.join(map(str,([ROOT/'runtime/policy/lib/python3.12/site-packages'] if not sim else [])+[REPO,shared_scripts(REPO),ROOT/'runtime/openpi-src',ROOT/'runtime/openpi-client-src'])),
             IMAGEIO_FFMPEG_EXE='/usr/bin/ffmpeg')
    if gpu is None:e.pop('CUDA_VISIBLE_DEVICES',None)
    else:e['CUDA_VISIBLE_DEVICES']=str(gpu)
    return e

def python(sim=False):return ROOT/'runtime/python38/bin/python3.8' if sim else Path('/alphabrain/.venv/bin/python')

def sim_env(gpu=None):
    e=env(True,gpu);e['PYTHONPATH']=str(ROOT/'runtime/sim/lib/python3.8/site-packages')+':'+e['PYTHONPATH'];return e

def spawn(argv,log,environ):
    require(not STOP.is_set(),'Controller stopped')
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:
        p=subprocess.Popen(list(map(str,argv)),stdout=f,stderr=subprocess.STDOUT,env=environ,start_new_session=True)
    with LOCK:ACTIVE.append(p)
    return p

def clean(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    with LOCK:
        if p in ACTIVE:ACTIVE.remove(p)

def finish(p,seconds=900):
    try:require(p.wait(timeout=seconds)==0,'Owned subprocess failed; inspect log')
    finally:clean(p)

def status(host,phase,**kw):write(ROOT/'hosts'/host/'status.json',dict(phase=phase,utc=time.time(),pid=os.getpid(),**kw))

def build():
    require(not (ROOT/'release.json').exists(),'Release already exists')
    old=Path('/share/longjunyu/alphabrain/experiments/dsol-statewise-view-oracle-v2')
    ps=[old/'protocols'/('dense-O-'+s+'-wave-00.json') for s in ['development','test']]
    data=[read(p) for p in ps];blocks=[b for p in data for b in p['state_blocks']]
    order=read('/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908/broad-existing/states.json')['states']
    indexed={b['state']['pair_key']:b for b in blocks};blocks=[indexed[s['pair_key']] for s in order]
    require(len(blocks)==64 and len(indexed)==64,'Wrong state population')
    protocol={**data[0],'state_blocks':blocks,'policy_repeat_ids':list(range(32)),
              'episode_identity_prefix':'dualhost-aa0-v1','diagnostic_role':'full64_aa0_reassessment'}
    require(protocol_spec_count(protocol)==198656,'Wrong matrix size')
    bridge=read('/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908/render-protocol-bridge-v1/release.json')
    catalog_path=REPO/'configs/dsol_paper1/libero_view_catalog_v2_m1.json'
    require(sha(catalog_path)==bridge['catalog']['sha256'],'Relocated catalog bytes changed')
    catalog={**bridge['catalog'],'path':str(catalog_path)}
    protocol['catalog']=str(catalog_path)
    gatekeys={s['pair_key'] for s in bridge['states']};gate=[]
    for si,b in enumerate(blocks):
        if b['state']['pair_key'] in gatekeys:
            for ci,c in enumerate(b['candidates']):
                if c['selected_candidate_id'] in ['canonical','broad_train_000','broad_heldout_000']:
                    for n in [0,1]:gate.append(si*97*32+ci*32+n)
    require(len(gate)==48,'Gate must cover eight tasks, three views, two noises')
    paths=[p for folder in ['AlphaBrain','scripts','configs'] for p in (REPO/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.suffix in ['.py','.sh','.yaml','.json']]
    r=dict(schema='dualhost_aa0_full64_v1',authorized='User requested relaunch using gnu; never stop or pause keepalive',
           protocol=protocol,states=[b['state'] for b in blocks],models=bridge['models'],catalog=catalog,noise_bank=bridge['noise_bank'],
           hosts={'fresh':{'state_indices':list(range(0,64,2))},'gnu':{'state_indices':list(range(1,64,2))}},
           gate_indices=gate,source_hashes={str(p):sha(p) for p in paths},source_protocol_hashes={str(p):sha(p) for p in ps},
           candidate_count=97,noise_repeats=32,total_episodes=397312,offsamples=0,replan_steps=5,denoising_steps=10,
           workers=32,copies=2,base_port=30700,asset_cells=['aa0-a'],score_ensemble_size=8,
           requirements=['all 64 sources retained','no old/new ledger union','same key cross-host full trace equality before dense','keepalive untouched'])
    write(ROOT/'release.json',r);print('FROZEN',ROOT,flush=True)

def episode(args,r):
    import scripts.dsol_paper1.runtime.evaluate_dsol_libero_hdf5_views as evaluator
    evaluator.configure_imports()
    from scripts.dsol_paper1.runtime.audit_libero_hdf5_restore import _configure_runtime
    spec=protocol_spec_at(r['protocol'],args.index)
    output=Path(args.output);config=Path(tempfile.mkdtemp(prefix='dsol-aa0-sim-'))
    _configure_runtime(RUNTIME,Path(spec['hdf5']).parent.parent,config)
    from AlphaBrain.research.dsol.data.flow_noise import ExplicitFlowNoiseBank
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from scripts.dsol_paper1.diagnostics.trace_view_repeatability_v1 import physical_arrays
    import libero.libero.envs as module
    original=module.OffScreenRenderEnv;digest=hashlib.sha256();counts={'calls':0,'steps':0}
    def update(x):digest.update((json.dumps(x,sort_keys=True,separators=(',',':'))+'\n').encode())
    def factory(*a,**kw):
        e=original(*a,**kw);step0=e.step
        def step(action):
            before={k:array_identity(v) for k,v in physical_arrays(e).items()};res=step0(action)
            update({'step':counts['steps'],'action':array_identity(action),'pre':before,'post':{k:array_identity(v) for k,v in physical_arrays(e).items()},'success':bool(res[2])})
            counts['steps']+=1;return res
        e.step=step;return e
    module.OffScreenRenderEnv=factory;client=WebsocketClientPolicy(host='127.0.0.1',port=args.port)
    class Client:
        def infer(self,request):
            response=client.infer(request)
            update({'call':counts['calls'],'inputs':{k:array_identity(v) if isinstance(v,np.ndarray) else v for k,v in request.items()},'actions':array_identity(response['actions'])})
            counts['calls']+=1;return response
    environment=None
    try:
        with render_protocol(0) as contexts:
            row,environment=evaluator.run_episode(spec,runtime=RUNTIME,config_root=config,client=Client(),
                replan_steps=5,wait_steps=0,resize_size=224,seed=20260818,save_video=False,video_dir=output.parent/'unused',render_gpu=args.gpu,
                noise_bank=ExplicitFlowNoiseBank(Path(r['noise_bank']['path']),verify_file=False),require_explicit_noise=True)
            require(contexts and set(contexts)=={0},'AA0 intervention missing')
            require(getattr(environment,'noise',None)==0,'Wrapper noise changed')
            row['legacy_candidate_features_not_used']=row.pop('candidate_features',{})
            row['aa0']={'release_sha256':sha(ROOT/'release.json'),'model':args.model,'checkpoint_sha256':r['models'][args.model]['weights_sha256'],
                        'host':args.host,'index':args.index,'trace_sha256':digest.hexdigest(),'counts':counts,'render_samples':contexts}
            require(not output.exists(),'Refusing duplicate episode write');write(output,row)
    finally:
        if environment is not None:environment.close()
        module.OffScreenRenderEnv=original
        shutil.rmtree(config)

def assets(args,r):
    import scripts.dsol_paper1.operations.controllers.run_render_bridge_worker_v1 as worker
    worker.ROOT=ROOT;worker.REPO=REPO
    # Reuse the audited renderer/scorer without changing the historical module.
    if args.mode=='render':worker.render(argparse.Namespace(index=args.index,cell='aa0-a',offsamples=0,gpu=args.gpu),r)
    else:
        # Score one state with exactly the audited ensemble implementation.
        import torch
        from AlphaBrain.model.framework.base_framework import BaseFramework
        from scripts.dsol_paper1.runtime.run_view_value_expectation_accel_ensemble import rank_state
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        m=BaseFramework.from_pretrained(r['models'][args.model]['path'],strict_checkpoint=True).to(torch.bfloat16).to('cuda:0').eval();m.gripper_remap=False
        for p in m.parameters():p.requires_grad_(False)
        for i in r['hosts'][args.host]['state_indices'][args.index::8]:
            output=ROOT/'scores'/args.model/'aa0-a'/f'{i:02d}'
            if (output/'completion.json').exists():continue
            state=dict(r['states'][i],pair_key=r['states'][i]['asset_source_pair_key'])
            rank_state(state,model=m,render_dir=ROOT/'assets/aa0-a'/f'{i:02d}',ensemble_size=8,root_seed=20260921,batch_size=16,output_dir=output)
            write(output/'completion.json',dict(release_sha256=sha(ROOT/'release.json'),ranking_sha256=sha(output/'ranking.json')))

def output_path(host,phase,model,index):return ROOT/'hosts'/host/phase/model/f'{index//3104:02d}'/f'{index:06d}.json'

def valid(path,r,model,index):
    if not path.exists():return False
    a=read(path);spec=protocol_spec_at(r['protocol'],index)
    require(a['aa0']['release_sha256']==sha(ROOT/'release.json') and a['aa0']['model']==model and a['aa0']['index']==index,'Episode identity mismatch')
    require(a['aa0']['checkpoint_sha256']==r['models'][model]['weights_sha256'],'Checkpoint mismatch')
    require(a['pair_key']==spec['pair_key'] and a['selected_candidate_id']==spec['selected_candidate_id'] and a['policy_repeat_id']==spec['policy_repeat_id'],'Wrong paired key')
    return True

def matrix(host,r,phase,indices):
    for model in ['canonical','broad']:
        servers=[]
        todo=[i for i in indices if not valid(output_path(host,phase,model,i),r,model,i)]
        if not todo:continue
        try:
            for j in range(16):
                port=r['base_port']+j
                with socket.socket() as s:require(s.connect_ex(('127.0.0.1',port))!=0,'Unowned occupied port')
                servers.append(spawn([python(),shared_scripts(REPO)/'serve_alphabrain_pi05_websocket.py','--checkpoint',r['models'][model]['path'],'--port',port,'--device','cuda:0','--cpu-threads',2],
                    ROOT/'hosts'/host/'logs'/f'{phase}-{model}-server-{j}.log',env(gpu=j//2)))
            start=time.time()
            while True:
                require(all(p.poll() is None for p in servers),'Policy server failed')
                ready=[]
                for j in range(16):
                    with socket.socket() as s:ready.append(s.connect_ex(('127.0.0.1',r['base_port']+j))==0)
                if all(ready):break
                require(time.time()-start<360,'Policy startup timeout');time.sleep(2)
            done=len(indices)-len(todo)
            def worker(w):
                nonlocal done
                for index in todo[w::32]:
                    require(not STOP.is_set(),'Stopped');path=output_path(host,phase,model,index)
                    finish(spawn([python(True),Path(__file__),'episode','--host',host,'--model',model,'--index',index,'--gpu',(w%16)//2,'--port',r['base_port']+w%16,'--output',path],
                                 ROOT/'hosts'/host/'logs'/f'{phase}-{model}-worker-{w}.log',sim_env()),1800)
                    valid(path,r,model,index)
                    with LOCK:
                        done+=1
                        if done%32==0:status(host,phase,model=model,model_completed=done,model_budget=len(indices))
            with ThreadPoolExecutor(max_workers=32) as pool:
                fs=[pool.submit(worker,w) for w in range(32)]
                try:
                    for f in as_completed(fs):f.result()
                except BaseException:
                    STOP.set()
                    for child in list(ACTIVE):clean(child)
                    raise
        finally:
            for p in servers:clean(p)
    write(ROOT/'hosts'/host/(phase+'-complete.json'),dict(release_sha256=sha(ROOT/'release.json'),episodes=len(indices)*2,utc=time.time()))

def check_gate(r):
    if not all((ROOT/'hosts'/h/'gate-b-complete.json').exists() for h in ['fresh','gnu']):return False
    for model in ['canonical','broad']:
        for i in r['gate_indices']:
            signatures=[]
            for h in ['fresh','gnu']:
                for phase in ['gate-a','gate-b']:
                    p=output_path(h,phase,model,i);valid(p,r,model,i);a=read(p)
                    signatures.append((a['aa0']['trace_sha256'],a['success'],a['completion_steps'],a['aa0']['counts']['calls'],a['aa0']['counts']['steps']))
            require(len(set(signatures))==1,f'Cross-host/repeat mismatch: {model}/{i}')
    return True

def run(args,r):
    import fcntl
    (ROOT/'hosts'/args.host).mkdir(parents=True,exist_ok=True)
    with (ROOT/'hosts'/args.host/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status(args.host,'VERIFYING')
        for m in r['models'].values():require(sha(Path(m['path'])/'model.safetensors')==m['weights_sha256'],'Model bytes changed')
        bank=read(r['noise_bank']['path']);require(sha(bank['noise_file'])==r['noise_bank']['file_sha256'],'Noise bytes changed')
        write(ROOT/'hosts'/args.host/'hardware.json',dict(hostname=socket.gethostname(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version','--format=csv'],text=True),keepalive_policy='untouched'))
        for phase in ['gate-a','gate-b']:status(args.host,phase);matrix(args.host,r,phase,r['gate_indices'])
        status(args.host,'WAITING_CROSS_HOST_GATE')
        start=time.time()
        while not check_gate(r):
            require(time.time()-start<24*3600,'Cross-host gate wait expired');time.sleep(10)
        write(ROOT/'hosts'/args.host/'gate-pass.json',dict(status='PASS_FULL_TRACE_CROSS_HOST_AND_REPEAT',episodes_per_host=192,utc=time.time()))
        owned=r['hosts'][args.host]['state_indices'];status(args.host,'RENDERING_OWNED_32_STATES')
        def render_one(i):
            p=ROOT/'assets/aa0-a'/f'{i:02d}'/'render.json'
            if p.exists():require(read(p)['release_sha256']==sha(ROOT/'release.json'),'Asset identity mismatch');return
            finish(spawn([python(True),Path(__file__),'render','--index',i,'--gpu',(i//2)%8],ROOT/'hosts'/args.host/'logs'/f'render-{i}.log',sim_env()),900)
        with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(render_one,owned))
        for model in ['canonical','broad']:
            status(args.host,'SCORING_OWNED_STATES',model=model)
            ps=[spawn([python(),Path(__file__),'score','--host',args.host,'--model',model,'--index',g],ROOT/'hosts'/args.host/'logs'/f'score-{model}-{g}.log',env(gpu=g)) for g in range(8)]
            for p in ps:finish(p,7200)
        indices=[i for s in owned for i in range(s*3104,(s+1)*3104)]
        status(args.host,'DENSE');matrix(args.host,r,'dense',indices)
        status(args.host,'COMPLETE',dense_episodes=len(indices)*2)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['build','run','episode','render','score']);p.add_argument('--host',choices=['fresh','gnu']);p.add_argument('--model',choices=['canonical','broad']);p.add_argument('--index',type=int,default=0);p.add_argument('--gpu',type=int,default=0);p.add_argument('--port',type=int,default=30700);p.add_argument('--output');a=p.parse_args()
    if a.mode=='build':build();return
    r=release() if a.mode=='run' else read(ROOT/'release.json')
    if a.mode=='run':
        def interrupted(*_):STOP.set();raise InterruptedError('Owned controller interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        try:run(a,r)
        except BaseException as e:status(a.host,'FAILED_REQUIRES_REVIEW',error=str(e));raise
        finally:
            STOP.set()
            for child in list(ACTIVE):clean(child)
    elif a.mode=='episode':episode(a,r)
    else:assets(a,r)

if __name__=='__main__':main()
