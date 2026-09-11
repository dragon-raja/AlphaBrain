#!/usr/bin/env python3
"""Own an isolated GPU window, execute the preregistered bridge, clean up."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from render_bridge_common_v1 import ROOT,REPO,read,sha,write_new,verify_release
import run_full64_view_landscape_v2 as control

ACTIVE=[]; LOCK=threading.RLock(); STOP=threading.Event()

def status(name,**values): control.atomic_json(ROOT/'status.json',{'status':name,'utc':control.stamp(),**values})

def environment(gpu=None,sim=False):
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1' if sim else '2',OPENBLAS_NUM_THREADS='1' if sim else '2',MKL_NUM_THREADS='1' if sim else '2',
        NUMEXPR_NUM_THREADS='1' if sim else '2',TOKENIZERS_PARALLELISM='false',IMAGEIO_FFMPEG_EXE='/usr/bin/ffmpeg',
        PRETRAINED_MODELS_DIR='/share/longjunyu/alphabrain/pretrained_models',ALPHABRAIN_DISABLE_AUTO_DOWNLOAD='1',
        PYTHONPATH=str(REPO)+':/projects/openpi/src:/projects/openpi/packages/openpi-client/src:'+str(REPO/'scripts/cabi_vla'))
    if gpu is not None: env['CUDA_VISIBLE_DEVICES']=str(gpu)
    else: env.pop('CUDA_VISIBLE_DEVICES',None)
    return env

def start(argv,log,env):
    if STOP.is_set(): raise InterruptedError('Bridge stopped')
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        with LOCK:
            if STOP.is_set(): raise InterruptedError('Bridge stopped')
            p=subprocess.Popen(list(map(str,argv)),stdout=stream,stderr=subprocess.STDOUT,env=env)
            ACTIVE.append(p)
    return p

def finish(p,seconds):
    try:
        if p.wait(timeout=seconds)!=0: raise RuntimeError('Owned bridge subprocess failed; inspect logs')
    finally:
        if p.poll() is None:
            p.terminate()
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        with LOCK:
            if p in ACTIVE: ACTIVE.remove(p)

def stop_children():
    STOP.set()
    with LOCK: children=list(ACTIVE)
    for p in children:
        if p.poll() is None: p.terminate()
    for p in children:
        try:p.wait(timeout=15)
        except subprocess.TimeoutExpired:p.kill();p.wait()

def interrupted(signum,frame):
    STOP.set()
    with LOCK:
        for p in ACTIVE:
            if p.poll() is None:p.terminate()
    raise InterruptedError('Bridge interrupted')

def batch(function,items,workers):
    pool=ThreadPoolExecutor(max_workers=workers)
    futures=[pool.submit(function,x) for x in items]
    try:
        for f in futures:f.result()
    except BaseException:
        stop_children(); raise
    finally:pool.shutdown(wait=True,cancel_futures=True)

def run_assets(release):
    jobs=[(cell,i) for cell in release['asset_cells'] for i in range(8)]
    def render(job):
        cell,i=job
        p=start(['/workspace/envs/fresh-libero/bin/python',REPO/'scripts/dsol_paper1/run_render_bridge_worker_v1.py',
            'render','--index',i,'--gpu',i,'--offsamples',0 if cell.startswith('aa0') else 4,'--cell',cell],
            ROOT/'logs'/f'asset-{cell}-{i}.log',environment(sim=True))
        finish(p,600)
    # One source index per physical device in each render pass.
    for cell in release['asset_cells']:
        status('RENDERING_CANDIDATES',cell=cell); batch(render,[(cell,i) for i in range(8)],8)
    status('SCORING_CANDIDATES')
    def score(job):
        model,index,gpu=job
        p=start(['/alphabrain/.venv/bin/python',REPO/'scripts/dsol_paper1/run_render_bridge_worker_v1.py',
            'score','--index',index,'--model',model,'--shards',4],ROOT/'logs'/f'score-{model}-{index}.log',environment(gpu))
        finish(p,1800)
    batch(score,[(model,index,index+4*j) for j,model in enumerate(('canonical','broad')) for index in range(4)],8)

def run_cell(release,cell):
    output=ROOT/'rollouts'/cell['name']
    if output.exists():raise FileExistsError('No implicit cell rerun: '+str(output))
    output.mkdir(parents=True); servers=[]; begin=time.monotonic()
    try:
        for index in range(8*cell['copies']):
            servers.append(start(['/alphabrain/.venv/bin/python',REPO/'scripts/cabi_vla/serve_alphabrain_pi05_websocket.py',
                '--checkpoint',release['models'][cell['model']]['path'],'--port',release['base_port']+index,'--device','cuda:0','--cpu-threads',2],
                output/'logs'/f'policy-{index}.log',environment(index//cell['copies'])))
        deadline=time.time()+240
        while True:
            if any(p.poll() is not None for p in servers):raise RuntimeError('Policy service failed to start')
            ready=[]
            for index in range(len(servers)):
                with socket.socket() as sock:ready.append(sock.connect_ex(('127.0.0.1',release['base_port']+index))==0)
            if all(ready):break
            if STOP.is_set() or time.time()>deadline:raise RuntimeError('Policy startup timeout')
            time.sleep(2)
        rollout_start=time.monotonic()
        def worker(worker_index):
            server_index=worker_index%len(servers); gpu=server_index//cell['copies']
            for index in range(worker_index,len(release['specs']),cell['workers']):
                p=start(['/workspace/envs/fresh-libero/bin/python',REPO/'scripts/dsol_paper1/run_render_bridge_worker_v1.py',
                    'episode','--index',index,'--gpu',gpu,'--port',release['base_port']+server_index,
                    '--offsamples',cell['offsamples'],'--cell',cell['name']],output/'logs'/f'episode-{index:04d}.log',environment(sim=True))
                finish(p,600)
        batch(worker,range(cell['workers']),cell['workers'])
        ledger={str(p):sha(p) for p in sorted(output.glob('[0-9][0-9][0-9][0-9].json'))}
        if len(ledger)!=192:raise ValueError('Incomplete bridge cell')
        write_new(output/'completion.json',{'status':'PASS_COMPLETE','cell':cell,'release_sha256':sha(ROOT/'release.json'),
            'wall_seconds_including_startup':time.monotonic()-begin,'rollout_seconds':time.monotonic()-rollout_start,'rows':ledger})
    finally:
        for p in servers:
            if p.poll() is None:p.terminate()
        for p in servers:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            with LOCK:
                if p in ACTIVE:ACTIVE.remove(p)

def main():
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    release=verify_release(); control.preflight()
    if (ROOT/'execution-entry.json').exists():raise FileExistsError('No implicit bridge restart')
    for model in release['models'].values():
        if sha(Path(model['path'])/'model.safetensors')!=model['weights_sha256']:raise ValueError('Checkpoint changed')
    bank=read(release['noise_bank']['path'])
    if sha(bank['noise_file'])!=release['noise_bank']['file_sha256']:raise ValueError('Noise file changed')
    control.resources_available(release['base_port'],24)
    write_new(ROOT/'execution-entry.json',{'utc':control.stamp(),'pid':os.getpid(),'release_sha256':sha(ROOT/'release.json'),
        'hardware':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total','--format=csv'],text=True)})
    stopped=[]
    try:
        for gpu in range(8):
            name=f'gpu-keepalive-{gpu}'
            if subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0:
                subprocess.run(['tmux','kill-session','-t',name],check=True);stopped.append(gpu)
        run_assets(release)
        for cell in release['cells']:
            verify_release(); status('RUNNING_PAIRED_CELL',cell=cell['name']); run_cell(release,cell)
        status('ANALYZING')
        finish(start([sys.executable,REPO/'scripts/dsol_paper1/analyze_render_protocol_bridge_v1.py'],ROOT/'logs/analysis.log',environment()),600)
        status('PASS_BRIDGE_COMPLETE_REQUIRES_MIGRATION_REVIEW',analysis=str(ROOT/'analysis.json'))
    except BaseException as error:
        status('STOPPED_REQUIRES_REVIEW',error=str(error));raise
    finally:
        stop_children()
        for gpu in stopped:
            if subprocess.run(['tmux','has-session','-t',f'gpu-keepalive-{gpu}'],capture_output=True).returncode:
                subprocess.run(['bash','/workspace/ai2r/gpu_compute_keepalive/start.sh','1','8192',f'gpu-keepalive-{gpu}',str(gpu)],
                    check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

if __name__=='__main__':main()
