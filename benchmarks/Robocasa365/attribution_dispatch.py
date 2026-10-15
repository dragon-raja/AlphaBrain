"""Bounded dual-host batch dispatch after all pre-launched pilots pass.

No job retries, destructive operations, credential changes or unrelated process
management. A new GPU consumer or any failed pilot stops before dispatch.
"""
from pathlib import Path
import argparse
import json
import math
import os
import shlex
import subprocess
import time

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.data.evaluation_matrix import flow_noise
from AlphaBrain.research.dsol.data.rollout_replay import array_identity


def remote(python, code, ssh):
    return subprocess.check_output([*ssh,shlex.join([python,'-c',code])],text=True,timeout=60).strip()


def read_status(path):
    """A cross-host atomic-replace visibility gap is not job completion.

    Only absent metadata is treated as unavailable; malformed content and an
    actual ERROR remain failures. The enclosing bounded loop checks again.
    """
    try:
        return read(path)['status']
    except FileNotFoundError:
        return 'STATUS_UNAVAILABLE'


def audit_pilots(root, plan):
    p0=read(plan['p0_plan']);counts={};timings={}
    for worker in [w for w in plan['workers'] if w['gpu'] in (0,1)]:
        identifier=worker['worker_id'];timings[identifier]=dict(seconds=0.,steps=0,rollouts=0)
        for phase in worker['phases']:
            output=root/'pilot-workers'/identifier/phase
            progress=read(output/'progress.json')
            if progress['status']!='COMPLETE':raise ValueError('Pilot incomplete')
            initial=worker['pilot_initials'][0]
            model=plan['models'][phase];arm=model['parent_arm']
            views=([plan['panel_views'][0],plan['panel_views'][5]] if worker['lane']=='checkpoints'
                   else list(dict.fromkeys(plan['choices'][arm][initial].values())))
            repeats=plan['checkpoint_repeats'] if worker['lane']=='checkpoints' else plan['confirmation_repeats']
            expected=len(views)*len(repeats)
            if progress['completed_rollouts']!=expected:raise ValueError('Pilot count mismatch')
            old_worker=next(w for w in p0['workers'] if w['arm']==arm and initial in w['initials'])
            for view in views:
                cell=output/'cells'/initial/view
                complete=read(cell/'complete.json')
                if not complete['complete'] or len(complete['trials'])!=len(repeats):raise ValueError('Uncommitted pilot cell')
                for checks in ('canonical_pixels','candidate_pixels'):
                    if not all(x['pass'] for x in complete[checks].values()):raise ValueError('Pilot pixels failed')
                old_cell=Path(plan['p0_plan']).parent/'workers'/old_worker['worker_id']/'cells'/initial/view
                old_trial=read(old_cell/'execution_0.json')
                scores=read((cell if worker['lane']=='checkpoints' else old_cell)/'scores.json')
                for repeat,record in zip(repeats,complete['trials']):
                    file=cell/f'execution_{repeat}.json';trial=read(file)
                    if sha(file)!=record['sha256'] or trial['repeat']!=repeat or not trial['complete_task_attempt']:
                        raise ValueError('Pilot trial identity mismatch')
                    if not (0<trial['steps']<=trial['horizon'] and (trial['steps']==trial['horizon'] or
                        trial['success'] or trial['terminated'] or trial['truncated'])):raise ValueError('Truncated task attempt')
                    if trial['initial_physical_sha256']!=old_trial['initial_physical_sha256']:
                        raise ValueError('Initial physical identity differs from P0')
                    if trial['plans'][0]['input_arrays']!=scores['input_arrays']:
                        raise ValueError('First scored/selected observation differs')
                    if len(trial['plans'])!=math.ceil(trial['steps']/25):raise ValueError('Replan count changed')
                    for replan,query in enumerate(trial['plans']):
                        if query['noise']!=array_identity(flow_noise(initial,'execution',repeat,replan)):
                            raise ValueError('Noise address mismatch')
                    timings[identifier]['seconds']+=trial['seconds'];timings[identifier]['steps']+=trial['steps']
                    timings[identifier]['rollouts']+=1
            counts[identifier+'/'+phase]=expected
    receipt=dict(status='PASS',plan_sha256=sha(root/'plan.json'),rollouts=sum(counts.values()),counts=counts,timings=timings)
    write(root/'pilot-audit.json',receipt)
    return receipt


def run(root, ssh):
    root=Path(root);plan=read(root/'plan.json')
    state=dict(status='WAITING_FOR_PILOTS',started_unix=time.time(),plan_sha256=sha(root/'plan.json'))
    try:
        authorization=read(root/'confirmation-authorization.json')
        if not authorization['approved'] or authorization['plan_sha256']!=state['plan_sha256']:
            raise ValueError('New-noise authorization missing')
        pilots=[w['worker_id'] for w in plan['workers'] if w['gpu'] in (0,1)]
        deadline=time.time()+3*3600
        while True:
            statuses={w:read_status(root/'pilot-supervisors'/w/'status.json') for w in pilots}
            state.update(pilots=statuses,heartbeat_unix=time.time());write(root/'dispatch-status.json',state)
            if any(s=='ERROR' for s in statuses.values()):raise ValueError('Pilot failed; no automatic retry')
            if all(s=='COMPLETE' for s in statuses.values()):break
            if time.time()>deadline:raise TimeoutError('Pilot gate timed out')
            time.sleep(20)
        audit_pilots(root,plan)
        templates=read(Path(plan['p0_plan']).parent/'formal-launch-plan.json')['jobs']
        # Check GPU process identities without stopping any process or exposing
        # unrelated full command lines. Known keepalive is intentionally kept.
        probe='''import subprocess,json
from pathlib import Path
raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
foreign=[]
for item in set(raw.split()):
 if not item.isdigit(): continue
 path=Path('/proc')/item/'cmdline'
 if path.exists() and b'gpu_compute_keepalive.py' not in path.read_bytes(): foreign.append(int(item))
print(json.dumps(foreign))'''
        for host in ('fresh','gnu'):
            template=next(t for t in templates if t['host']==host);python=template['command'][0]
            output=subprocess.check_output([python,'-c',probe],text=True) if host=='fresh' else remote(python,probe,ssh)
            if json.loads(output):raise RuntimeError('Other GPU consumers detected on '+host+'; no jobs stopped')
        jobs=[]
        state['status']='DISPATCHING';write(root/'dispatch-status.json',state)
        for worker in plan['workers']:
            template=next(t for t in templates if t['host']==worker['host'])
            env=template['environment'].copy();env['CUDA_VISIBLE_DEVICES']=str(worker['gpu'])
            env['PYTHONPATH']=env['PYTHONPATH'].replace(str(Path(plan['p0_plan']).parent/'source'),plan['source'])
            python=template['command'][0];source=Path(plan['source']);identifier=worker['worker_id']
            if (root/'supervisors'/identifier).exists() or (root/'workers'/identifier).exists():
                raise FileExistsError('Refuse duplicate launch '+identifier)
            command=[python,str(source/'scripts/dsol_paper1/operations/supervise_robocasa_job.py'),
                '--output',str(root/'supervisors'/identifier),'--execute','--',python,
                str(source/'scripts/dsol_paper1/runtime/robocasa_attribution_worker.py'),
                '--plan',str(root/'plan.json'),'--worker',identifier,'--output',str(root/'workers'/identifier),'--execute']
            payload=dict(command=command,environment=env)
            if worker['host']=='fresh':
                pid=subprocess.Popen(command,env={**os.environ,**env},stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                                     start_new_session=True,cwd=source).pid
            else:
                code='import json,os,subprocess;p=json.loads('+repr(json.dumps(payload))+');c=subprocess.Popen(p["command"],env={**os.environ,**p["environment"]},stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True,cwd='+repr(str(source))+');print(c.pid)'
                pid=int(remote(python,code,ssh))
            jobs.append(dict(worker=identifier,host=worker['host'],pid=pid,**payload))
            write(root/'launch.json',dict(plan_sha256=state['plan_sha256'],jobs=jobs))
        state.update(status='RUNNING',workers=len(jobs),launched_unix=time.time());write(root/'dispatch-status.json',state)
        deadline=time.time()+5*24*3600
        while True:
            statuses={j['worker']:read_status(root/'supervisors'/j['worker']/'status.json') for j in jobs}
            state.update(worker_status=statuses,heartbeat_unix=time.time());write(root/'dispatch-status.json',state)
            if any(s=='ERROR' for s in statuses.values()):raise RuntimeError('Formal worker failed; remaining bounded jobs preserved')
            if all(s=='COMPLETE' for s in statuses.values()):break
            if time.time()>deadline:raise TimeoutError('Dispatch observation deadline reached; bounded workers preserved')
            time.sleep(30)
        state.update(status='COMPLETE_PENDING_RESULT_AUDIT',finished_unix=time.time())
    except Exception as exc:
        state.update(status='ERROR',error=f'{type(exc).__name__}: {exc}',finished_unix=time.time())
        raise
    finally:
        write(root/'dispatch-status.json',state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--ssh-host',required=True);parser.add_argument('--ssh-port',required=True)
    parser.add_argument('--ssh-key',required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:parser.error('Explicit --execute required')
    ssh=['ssh','-i',args.ssh_key,'-p',args.ssh_port,'-o','BatchMode=yes','-o','ConnectTimeout=10',args.ssh_host]
    run(args.root,ssh)


if __name__=='__main__':
    main()
