#!/usr/bin/env python3
"""Queue behind the first-eight job; benchmark concurrency and complete 56 states."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

from build_matched_view_landscape_protocol_v1 import read_json, write_new_json, source_identity
from explicit_flow_noise import sha256_file
from audit_statewise_view_oracle_v2_run import atomic_json
from run_matched_view_accel_v1 import validate_result

REPO=Path(__file__).resolve().parents[2]
PARENT=Path('/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908')
ROOT=PARENT/'full64-extension-v2'
RUNNER=REPO/'scripts/dsol_paper1/run_dsol_libero_hdf5_closed_loop_eval.sh'
AUDITOR=REPO/'scripts/dsol_paper1/audit_statewise_view_oracle_v2_run.py'
ACTIVE=[]
STOPPED=False


def require(value,message):
    if not value: raise ValueError(message)


def stamp(): return datetime.now(timezone.utc).isoformat()


def status(name,**values):
    atomic_json(ROOT/'controller-status.json',dict(status=name,updated_at_utc=stamp(),pid=os.getpid(),**values))


def stop(signum,frame):
    global STOPPED
    STOPPED=True
    for proc in ACTIVE:
        if proc.poll() is None: proc.terminate()  # Only our own timeout supervisors.


def verify_identity(item):
    require(sha256_file(Path(item['path']))==item['sha256'],'Frozen file changed: '+item['path'])


def preflight():
    release=read_json(ROOT/'release.json')
    require(release['status']=='FROZEN_USER_AUTHORIZED_EXTENSION' and release['root']==str(ROOT),'Wrong release')
    require((release['total_state_count'],release['new_state_count'],release['reused_state_count'])==(64,56,8),'Wrong state budget')
    require(sum(p['episode_count'] for p in release['route'])==173824,'Wrong episode budget')
    require(not any(release[k] for k in ('training','active_camera','robocasa')),'Out-of-scope action')
    require(release['performance_modes']==[{'label':'baseline32','copies':2,'workers':32},{'label':'copies3-workers48','copies':3,'workers':48},{'label':'copies3-workers64','copies':3,'workers':64}], 'Unfrozen concurrency choices')
    require(release['unchanged_scientific_controls']=={'replan_steps':5,'wait_steps':0,'flow_denoising_steps':10,'score_noise_members':8,'rollout_repeat_count':32,'candidate_count':97},'Scientific controls changed')
    for item in [release['old_release'],release['selection'],release['accel_manifest'],release['benchmark_protocol'],*release['source_identities'],*release['critical_files'],*release['route']]: verify_identity(item)
    old=release['old_release_content']
    runner_files=[REPO/'scripts/cabi_vla/serve_alphabrain_pi05_websocket.py',REPO/'scripts/cabi_vla/serve_openpi_deterministic.py',
        REPO/'scripts/dsol_paper1/evaluate_dsol_libero_hdf5_views.py',REPO/'scripts/dsol_paper1/summarize_dsol_libero_hdf5_closed_loop.py',RUNNER]
    combined=hashlib.sha256(subprocess.check_output(['sha256sum',*map(str,runner_files)])).hexdigest()
    require(combined==old['runner_code_sha256'],'Frozen runner/server/evaluator changed')
    for key in ('catalog','checkpoint_receipt'): verify_identity(old[key])
    bank=read_json(Path(old['noise_bank']['path']))
    require(sha256_file(Path(old['noise_bank']['path']))==old['noise_bank']['manifest_sha256'],'Noise manifest changed')
    require(bank['noise_file_sha256']==old['noise_bank']['file_sha256'],'Noise file declaration differs')
    return release


def environment(release,part,output,mode):
    old=release['old_release_content']
    env=os.environ.copy()
    env.update(CHECKPOINT=old['canonical_checkpoint']['path'],OUTPUT_DIR=str(output),PROTOCOL=part['path'],
        NOISE_BANK_MANIFEST=old['noise_bank']['path'],REQUIRE_EXPLICIT_NOISE='1',GPU_COUNT='8',
        EVAL_WORKER_COUNT=str(mode['workers']),POLICY_SERVER_COPIES_PER_GPU=str(mode['copies']),
        POLICY_CPU_THREADS='2',SIM_CPU_THREADS='1',DSOL_GPU_DEVICES='0,1,2,3,4,5,6,7',BASE_PORT=str(release['base_port']),
        REPLAN_STEPS='5',WAIT_STEPS='0',EVAL_SEED='20260818',VIDEO_EPISODES='0',RUN_ANALYSIS='0',KEEPALIVE_MODE='managed',
        POLICY_BACKEND='alphabrain',MAX_EPISODES_PER_SHARD='',POLICY_PYTHON='/alphabrain/.venv/bin/python',
        SIM_PYTHON='/workspace/envs/fresh-libero/bin/python',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        PYTHONPATH=str(REPO)+':/projects/openpi/src:/projects/openpi/packages/openpi-client/src')
    return env


def resources_available(port,count=24):
    # Never terminate allocations that are not ours.
    for attempt in range(30):
        pids=[]
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        for value in set(rows.split()):
            if not value.isdigit(): continue
            try: cmd=(Path('/proc')/value/'cmdline').read_bytes()
            except FileNotFoundError: continue
            if b'/workspace/ai2r/gpu_compute_keepalive/gpu_compute_keepalive.py' not in cmd: pids.append(value)
        ports=[]
        for p in range(port,port+count):
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1',p))==0: ports.append(p)
        if not pids and not ports: return
        if STOPPED: raise InterruptedError('Stopped')
        time.sleep(.5)
    raise RuntimeError(f'Unowned GPU/port allocation; no process changed: {pids}, {ports}')


def command(argv,env,log,deadline):
    remaining=int(deadline-time.time())
    require(remaining>0 and not STOPPED,'Execution budget exhausted or interrupted')
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        proc=subprocess.Popen(['timeout','--signal=TERM','--kill-after=120s',str(remaining)+'s',*map(str,argv)],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        ACTIVE.append(proc)
        try: code=proc.wait()
        finally: ACTIVE.remove(proc)
    require(code==0 and not STOPPED,f'Command failed ({code}); no implicit retry: {log}')


def validate_run(manifest,release,part,mode):
    old=release['old_release_content']
    expected=dict(checkpoint=old['canonical_checkpoint']['path'],checkpoint_sha256=old['canonical_checkpoint']['weights_sha256'],
        protocol_sha256=part['sha256'],noise_bank_manifest_sha256=old['noise_bank']['manifest_sha256'],
        code_sha256=old['runner_code_sha256'],policy_backend='alphabrain',require_explicit_noise=True,
        gpu_count=8,eval_worker_count=mode['workers'],policy_server_copies_per_gpu=mode['copies'],
        replan_steps=5,wait_steps=0,eval_seed=20260818,video_episodes_per_worker=0,max_episodes_per_shard=None,run_analysis=False)
    for key,value in expected.items(): require(manifest.get(key)==value,'Run contract mismatch: '+key)


def completed(output,release,part,mode):
    path=output/'segment-receipt.json'
    if not path.exists():
        require(not output.exists() or not any(output.iterdir()),'Partial output requires explicit review; no retry: '+str(output))
        return None
    receipt=read_json(path)
    require(receipt['status']=='PASS_COMPLETE' and receipt['release_sha256']==sha256_file(ROOT/'release.json'),'Receipt identity changed')
    require(receipt['protocol_sha256']==part['sha256'] and receipt['mode']==mode,'Receipt protocol/mode changed')
    require({str(p) for p in output.glob('episodes-shard-*.jsonl')}==set(receipt['ledger_sha256']),'Ledger set differs')
    for p,digest in receipt['ledger_sha256'].items(): require(sha256_file(Path(p))==digest,'Ledger changed')
    for name in ('run_manifest.json','audit.json'): require(sha256_file(output/name)==receipt['artifact_sha256'][name],'Manifest/audit changed')
    validate_run(read_json(output/'run_manifest.json'),release,part,mode)
    return receipt


def run_segment(release,part,output,mode,deadline):
    prior=completed(output,release,part,mode)
    if prior: return prior
    preflight(); resources_available(release['base_port'])
    output.mkdir(parents=True,exist_ok=True)
    env=environment(release,part,output,mode)
    started=time.monotonic()
    command(['bash',RUNNER],env,ROOT/'logs'/(output.name+'.log'),deadline)
    seconds=time.monotonic()-started
    validate_run(read_json(output/'run_manifest.json'),release,part,mode)
    command([sys.executable,AUDITOR,'--protocol',part['path'],'--noise-bank-manifest',release['old_release_content']['noise_bank']['path'],
        '--run-manifest',output/'run_manifest.json','--episode-ledgers',str(output/'episodes-shard-*.jsonl'),'--output',output/'audit.json'],env,ROOT/'logs'/(output.name+'.log'),deadline)
    audit=read_json(output/'audit.json')
    require(audit['status']=='PASS_COMPLETE' and audit['episode_count']==part['episode_count'],'Incomplete audit')
    receipt=dict(status='PASS_COMPLETE',created_at_utc=stamp(),release_sha256=sha256_file(ROOT/'release.json'),
        protocol_sha256=part['sha256'],episode_count=part['episode_count'],mode=mode,runner_wall_seconds=seconds,
        artifact_sha256={name:sha256_file(output/name) for name in ('run_manifest.json','audit.json')},
        ledger_sha256={str(p):sha256_file(p) for p in output.glob('episodes-shard-*.jsonl')})
    write_new_json(output/'segment-receipt.json',receipt)
    return receipt


def action_signatures(output):
    rows={}
    for p in output.glob('episodes-shard-*.jsonl'):
        for line in p.read_text().splitlines():
            r=json.loads(line); key=r['episode_id']
            require(key not in rows,'Duplicate benchmark episode')
            rows[key]=(r['success'],r['completion_steps'],r['initial_metrics']['physics_state_sha256'],
                       tuple((c['replan_index'],c['noise_seed'],c['noise_sha256'],c['action_chunk_sha256']) for c in r['policy_calls']))
    return rows


def select_mode(results):
    baseline=results[0]
    eligible=[r for r in results if r['exact_action_equivalence'] and r['runner_wall_seconds'] <= baseline['runner_wall_seconds']/1.05]
    return min(eligible,key=lambda r:r['runner_wall_seconds'])['mode'] if eligible else baseline['mode']


def score_extension(release,deadline):
    manifest_path=Path(release['accel_manifest']['path']); root=manifest_path.parent
    manifest=read_json(manifest_path); completion=root/'completion.json'
    if completion.exists():
        receipt=read_json(completion)
        require(receipt['status']=='PASS_COMPLETE' and receipt['manifest_sha256']==sha256_file(manifest_path),'Accel completion mismatch')
    else:
        resources_available(release['base_port'])
        stopped=[]; handles=[]; procs=[]
        try:
            for gpu in range(8):
                name=f'gpu-keepalive-{gpu}'
                if subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0:
                    subprocess.run(['tmux','kill-session','-t',name],check=True); stopped.append(gpu)
            for gpu in range(8):
                env=environment(release,release['route'][0],root,release['performance_modes'][0]); env['CUDA_VISIBLE_DEVICES']=str(gpu)
                (root/'logs').mkdir(exist_ok=True); stream=(root/'logs'/f'shard-{gpu}.log').open('a'); handles.append(stream)
                proc=subprocess.Popen(['timeout','--signal=TERM','--kill-after=120s','7200s',sys.executable,
                    str(REPO/'scripts/dsol_paper1/run_matched_view_accel_v1.py'),'--manifest',str(manifest_path),'--num-shards','8','--shard-index',str(gpu),'--device','cuda:0'],
                    env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                procs.append(proc); ACTIVE.append(proc)
            codes=[p.wait() for p in procs]
            require(all(c==0 for c in codes) and not STOPPED,'Extension Accel failed; no rollout launched')
        finally:
            for p in procs:
                if p.poll() is None: p.terminate(); p.wait()
                if p in ACTIVE: ACTIVE.remove(p)
            for stream in handles: stream.close()
            for gpu in stopped:
                if subprocess.run(['tmux','has-session','-t',f'gpu-keepalive-{gpu}'],capture_output=True).returncode:
                    subprocess.run(['bash','/workspace/ai2r/gpu_compute_keepalive/start.sh','1','8192',f'gpu-keepalive-{gpu}',str(gpu)],check=True)
    rows=[json.loads(l) for p in root.glob('rank-shard-*.jsonl') for l in p.read_text().splitlines() if l.strip()]
    states={s['asset_source_pair_key']:s for s in manifest['states']}
    require(len(rows)==56 and {r['pair_key'] for r in rows}==set(states),'Accel state set incomplete')
    for row in rows: validate_result(row,states[row['pair_key']],sha256_file(manifest_path))
    if not completion.exists(): write_new_json(completion,dict(status='PASS_COMPLETE',state_count=56,candidate_count=97,ensemble_size=8,
        checkpoint_sha256=manifest['checkpoint_sha256'],manifest_sha256=sha256_file(manifest_path),created_at_utc=stamp()))


def verify_first_eight():
    value=read_json(PARENT/'controller-status.json')
    require(value['status']=='PASS_COMPLETE_STATIC_DIAGNOSTIC' and value['completed_unique_episodes']==24832,'First-eight matrix not complete; extension not started')
    total=0
    for output in sorted((PARENT/'closed-loop/canonical/O').iterdir()):
        r=read_json(output/'segment-receipt.json')
        require(r['status']=='PASS_COMPLETE','First-eight receipt not PASS')
        require(r['identity']==value['identity'],'First-eight segment identity differs')
        for p,digest in r['ledger_sha256'].items(): require(sha256_file(Path(p))==digest,'First-eight ledger changed')
        for name,key in (('run_manifest.json','run_manifest_sha256'),('audit.json','audit_sha256')):
            require(sha256_file(output/name)==r[key],'First-eight audit/manifest changed')
        total+=r['episode_count']
    require(total==24832,'First-eight reuse count differs')


def execute(release):
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'controller.lock').open('a') as local_lock, (PARENT/'logs/pipeline-gpu.lock').open('a') as gpu_lock:
        fcntl.flock(local_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        queued=time.time(); status('QUEUED_BEHIND_FIRST_EIGHT',new_dense_episodes=173824,total_dense_episodes=198656)
        while True:
            require(not STOPPED and time.time()-queued<release['max_queue_seconds'],'Queue interrupted or timed out')
            try: fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB); break
            except BlockingIOError: time.sleep(20)
        verify_first_eight(); preflight()
        started=time.time(); deadline=started+release['max_execution_seconds']
        old=release['old_release_content']
        require(sha256_file(Path(old['canonical_checkpoint']['path'])/'model.safetensors')==old['canonical_checkpoint']['weights_sha256'],'Checkpoint bytes changed')
        bank=read_json(Path(old['noise_bank']['path']))
        require(sha256_file(Path(bank['noise_file']))==old['noise_bank']['file_sha256'],'Noise bytes changed')
        status('SCORING_56_ADDITIONAL_STATES',reused_dense_episodes=24832)
        score_extension(release,deadline)
        choice_path=ROOT/'performance-choice.json'
        if choice_path.exists():
            choice=read_json(choice_path)
            require(choice['release_sha256']==sha256_file(ROOT/'release.json'),'Performance choice release changed')
            mode=choice['selected_mode']
            require(mode in release['performance_modes'],'Unknown chosen mode')
        else:
            results=[]; baseline=None
            for mode in release['performance_modes']:
                status('BENCHMARKING_CONCURRENCY',mode=mode)
                out=ROOT/'benchmarks'/mode['label']
                r=run_segment(release,release['benchmark_protocol'],out,mode,deadline)
                signatures=action_signatures(out)
                if baseline is None: baseline=signatures
                results.append(dict(mode=mode,runner_wall_seconds=r['runner_wall_seconds'],exact_action_equivalence=signatures==baseline,
                                    receipt=source_identity(out/'segment-receipt.json')))
            mode=select_mode(results)
            write_new_json(choice_path,dict(status='PASS_PERFORMANCE_SELECTION',release_sha256=sha256_file(ROOT/'release.json'),
                selected_mode=mode,results=results,selection_uses_success_values=False,
                rule='Exact full action/noise trajectory equality, then >=5% end-to-end speedup; benchmark outcomes excluded from scientific matrix'))
        done=0
        for part in release['route']:
            status('RUNNING_FULL64_EXTENSION',current_segment=part['label'],new_dense_completed=done,total_dense_completed=24832+done,
                   total_dense_budget=198656,mode=mode,execution_started_epoch=started)
            run_segment(release,part,ROOT/'closed-loop/O'/part['label'],mode,deadline)
            done+=part['episode_count']
            print(json.dumps(dict(event='segment_pass',segment=part['label'],new_completed=done,utc=stamp())),flush=True)
        require(done==173824,'Extension incomplete')
        status('PASS_COMPLETE_DENSE_FULL64',new_dense_completed=done,total_dense_completed=198656,mode=mode)
    # GPU lock is released before CPU plots and statistics.
    command([sys.executable,REPO/'scripts/dsol_paper1/finalize_full64_view_landscape_v2.py'],
        environment(release,release['route'][0],ROOT,mode),ROOT/'logs/finalize.log',deadline)
    status('PASS_COMPLETE_FULL64_WITH_REPORTS',new_dense_completed=173824,total_dense_completed=198656,mode=mode)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--check-only',action='store_true'); args=parser.parse_args()
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    release=preflight()
    if args.check_only:
        print(json.dumps(dict(status='PASS_READY_FULL64_EXTENSION',new_states=56,new_episodes=173824,benchmark_extra=576,gpu_used=False)))
        return
    try: execute(release)
    except BaseException as error:
        status('STOPPED_REQUIRES_REVIEW',error=str(error),automatic_retry=False)
        raise


if __name__=='__main__': main()
