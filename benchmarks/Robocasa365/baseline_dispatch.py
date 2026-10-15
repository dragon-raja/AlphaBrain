"""Explicit bounded dispatch for the 800-rollout starting-checkpoint panel.

Reuses frozen worker code and per-host environments. Never stops other jobs,
retries failed trials, or dispatches the cancelled full attribution matrix.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import shutil

from AlphaBrain.research.dsol.data.artifacts import read, sha, write, path_mapping
from AlphaBrain.research.dsol.data.diagnostic_panel import validate_diagnostic


def relocated_environment(environment, mappings):
    """Expand every path-list segment, not only a string's leading prefix."""
    result = {}
    for key, value in environment.items():
        for old, new in mappings:
            value = value.replace(old.rstrip('/')+'/', new.rstrip('/')+'/')
        result[key] = value
    return result


def launch(root, selected, ssh, *, retry_startup=False):
    root = Path(root); plan = read(root/'plan.json'); validate_diagnostic(plan)
    templates = read(Path(plan['p0_plan']).parent/'formal-launch-plan.json')['jobs']
    record_file = root/'launch.json'
    record = read(record_file) if record_file.exists() else dict(
        plan_sha256=sha(root/'plan.json'), dispatcher_sha256=sha(__file__), jobs=[])
    if record['plan_sha256'] != sha(root/'plan.json'):
        raise ValueError('Launch plan identity changed')
    raw_mappings = json.loads((root/'plan.json').read_text()).get('path_mappings', [])
    if retry_startup:
        if not selected:
            raise ValueError('Startup retry requires explicit worker names')
        for name in selected:
            status = read(root/'supervisors'/name/'status.json')
            progress = list((root/'workers'/name).glob('*/progress.json'))
            if status['status'] != 'ERROR' or not progress or any(read(p)['completed_rollouts'] != 0 for p in progress):
                raise ValueError('Only zero-rollout startup failures may be explicitly retried')
            if list((root/'workers'/name).glob('*/cells/*/*/execution_*.json')):
                raise ValueError('Never retry measured outcomes')
            destination = root/'startup-failures'/name/str(time.time_ns())
            destination.mkdir(parents=True, exist_ok=False)
            shutil.copy2(record_file, destination/'launch-before.json')
            for category in ('supervisors','workers'):
                (root/category/name).rename(destination/category)
            for job in record['jobs']:
                if job['worker'] == name:
                    job['retired_zero_rollout_startup'] = str(destination)
            write(record_file, record)
    existing = {j['worker'] for j in record['jobs'] if not j.get('retired_zero_rollout_startup')}
    workers = [w for w in plan['workers'] if not selected or w['worker_id'] in selected]
    if selected and set(selected) != {w['worker_id'] for w in workers}:
        raise ValueError('Unknown requested worker')
    for worker in workers:
        name = worker['worker_id']
        if name in existing:
            continue
        if (root/'supervisors'/name).exists() or (root/'workers'/name).exists():
            raise FileExistsError('Unrecorded worker output; explicit audit required')
        template = next(t for t in templates if t['host'] == worker['host'])
        python = template['command'][0]
        env = relocated_environment(template['environment'], raw_mappings)
        env['CUDA_VISIBLE_DEVICES'] = str(worker['gpu'])
        env['PYTHONPATH'] = env['PYTHONPATH'].replace(str(Path(plan['p0_plan']).parent/'source'), plan['source'])
        source = Path(plan['source'])
        command = [python,str(source/'scripts/dsol_paper1/operations/supervise_robocasa_job.py'),
            '--output',str(root/'supervisors'/name),'--execute','--',python,
            str(source/'scripts/dsol_paper1/runtime/robocasa_attribution_worker.py'),
            '--plan',str(root/'plan.json'),'--worker',name,
            '--output',str(root/'workers'/name),'--execute']
        # Check this GPU immediately before launching. Preserve known keepalive.
        probe = """import json,os,subprocess
from pathlib import Path
raw=subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-compute-apps=pid','--format=csv,noheader'],text=True)
foreign=[]
for pid in set(raw.split()):
 if pid.isdigit():
  process_path=Path('/proc')/pid/'cmdline'
  if not process_path.exists() or b'gpu_compute_keepalive.py' not in process_path.read_bytes(): foreign.append(pid)
if foreign: raise RuntimeError('Existing GPU consumers: '+str(foreign))
"""
        payload = dict(command=command, environment=env, cwd=str(source))
        code = ('import json,os,subprocess\np=json.loads('+repr(json.dumps(payload))+')\n'
                'os.environ.update(p["environment"])\n'+probe+
                '\nchild=subprocess.Popen(p["command"],env=os.environ.copy(),cwd=p["cwd"],'
                'stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)\nprint(child.pid)')
        cmd = [python,'-c',code] if worker['host']=='fresh' else [*ssh,shlex.join([python,'-c',code])]
        pid = int(subprocess.check_output(cmd,text=True,timeout=60).strip())
        record['jobs'].append(dict(worker=name,host=worker['host'],gpu=worker['gpu'],pid=pid,
            launched_unix=time.time(),dispatcher_sha256=sha(__file__),**payload))
        write(record_file,record)
        print(name,pid,flush=True)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--workers',nargs='*')
    parser.add_argument('--ssh-host',required=True)
    parser.add_argument('--ssh-port',required=True)
    parser.add_argument('--ssh-key',required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--monitor',action='store_true')
    parser.add_argument('--retry-startup',action='store_true',help='Explicit selected zero-rollout startup failures only; preserve all evidence')
    args=parser.parse_args()
    if not args.execute: parser.error('Explicit --execute required')
    ssh=['ssh','-i',args.ssh_key,'-p',args.ssh_port,'-o','BatchMode=yes','-o','ConnectTimeout=10',args.ssh_host]
    with path_mapping(read(args.root/'plan.json').get('path_mappings', [])):
        launch(args.root,args.workers,ssh,retry_startup=args.retry_startup)
        if args.monitor:
            monitor(args.root)


def monitor(root):
    """Observe bounded jobs and audit/summarize on completion; no auto retries."""
    root=Path(root);plan=read(root/'plan.json')
    from AlphaBrain.research.dsol.analysis.baseline_panel import summarize
    from AlphaBrain.research.dsol.plotting.baseline_panel import plot
    import inspect
    code_files=[Path(__file__),Path(inspect.getfile(summarize)),Path(inspect.getfile(plot))]
    identities={str(p):sha(p) for p in code_files}
    state=dict(status='RUNNING',plan_sha256=sha(root/'plan.json'),total_rollouts=plan['rollout_count'],
        started_unix=time.time(),postprocessing_sources=identities)
    write(root/'postprocessing-source-identity.json',identities)
    try:
        deadline=time.time()+plan.get('monitor_hours', 6)*3600
        while True:
            workers={};done=0
            for w in plan['workers']:
                name=w['worker_id'];status_file=root/'supervisors'/name/'status.json'
                try: super_status=read(status_file)['status']
                except FileNotFoundError: super_status='NOT_YET_VISIBLE'
                for phase in w['phases']:
                    progress_file=root/'workers'/name/phase/'progress.json'
                    try: progress=read(progress_file)
                    except FileNotFoundError: progress={}
                    done+=progress.get('completed_rollouts',0)
                    workers[name+'/'+phase]=dict(supervisor=super_status,phase=progress.get('status'),
                        completed=progress.get('completed_rollouts',0),error=progress.get('error'))
            state.update(workers=workers,completed_rollouts=done,heartbeat_unix=time.time())
            write(root/'status.json',state)
            if any(v['supervisor']=='ERROR' or v['phase']=='ERROR' for v in workers.values()):
                raise RuntimeError('A worker failed; no retry or protocol relaxation. Other bounded jobs preserved.')
            if all(v['supervisor']=='COMPLETE' for v in workers.values()):break
            if time.time()>deadline:raise TimeoutError('Monitor timeout; bounded workers preserved')
            time.sleep(30)
        if done!=plan['rollout_count']:raise ValueError('Completion count differs from frozen plan')
        if any(sha(p)!=digest for p,digest in identities.items()):
            raise ValueError('Postprocessing source changed during execution')
        state['status']='AUDITING';write(root/'status.json',state)
        summary=summarize(root);plot(summary,root/'analysis')
        state.update(status='COMPLETE_AUDITED',finished_unix=time.time(),
            summary=str(root/'analysis/summary.json'),report=str(root/'analysis/report.md'))
    except Exception as exc:
        state.update(status='ERROR',error=f'{type(exc).__name__}: {exc}',finished_unix=time.time())
        raise
    finally:
        write(root/'status.json',state)


if __name__=='__main__':
    main()
