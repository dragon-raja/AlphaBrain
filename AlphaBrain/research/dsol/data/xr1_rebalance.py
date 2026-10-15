"""Explicit missing-trial work orders and a bounded dual-host drain queue."""
import os
from pathlib import Path
import shlex
import subprocess
import time

from .artifacts import read, sha, write


def validate_order(order, initials, views, panel_sha):
    if order['panel_sha256'] != panel_sha or order['initial'] not in initials:
        raise ValueError('Work order panel/initial identity mismatch')
    selected = order['views']
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(views):
        raise ValueError('Invalid work-order candidate subset')
    if set(order['repeats']) != set(selected):
        raise ValueError('Incomplete repeat assignments')
    for repeats in order['repeats'].values():
        if not repeats or repeats != sorted(set(repeats)) or not set(repeats) <= set(range(4)):
            raise ValueError('Invalid explicit execution repeats')


def validate_trial(path, initial, repeat):
    from .xr1_continuation import _seed
    path = Path(path); row = read(path)
    if (row['repeat'] != repeat or row['complete_task_attempt'] is not True
            or type(row['success']) is not bool or not 0 < row['steps'] <= row['horizon']
            or sha(path.with_name(f'trace_{repeat}.npz')) != row['trace_sha256']):
        raise ValueError(f'Invalid retained/completed trial: {path}')
    if not row['plans'] or any(p['seed'] != _seed(initial, 'execution', repeat, k)
                              or p['step'] != k*16 for k,p in enumerate(row['plans'])):
        raise ValueError('Execution address changed')
    return row


def prepare(remaining, destination, source, roots, pi05_subspaces):
    """Call ONLY after the old workers/controllers have exited."""
    import shutil
    from .xr1_continuation import process_running
    remaining, destination, source = map(Path, (remaining, destination, source))
    plan = read(remaining / 'plan.json'); launch = read(remaining / 'launch.json')
    if any(process_running(w['pid']) for w in launch['workers']):
        raise RuntimeError('Old worker still alive; cannot assign missing trials')
    destination.mkdir(parents=True, exist_ok=False)
    frozen = destination / 'source'
    for folder in ('AlphaBrain', 'benchmarks', 'scripts', 'configs'):
        shutil.copytree(source / folder, frozen / folder, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    write(destination/'source-manifest.json', {str(p.relative_to(frozen)):sha(p) for p in frozen.rglob('*') if p.is_file()})
    workers = {i: Path(w['output']) for w in launch['workers'] for i in w['initials']}
    orders=[]; retained=0; mapping=[]
    for initial in plan['initial_ids']:
        pending=[]
        for view in plan['views']:
            cell=workers[initial]/'cells'/initial/view
            executions=[]; missing=[]
            for repeat in range(4):
                path=cell/f'execution_{repeat}.json'
                if path.exists():
                    validate_trial(path,initial,repeat);retained+=1;executions.append(str(path))
                else:
                    missing.append(repeat);executions.append(None)
            record=dict(initial=initial,view=view,scores=str(cell/'scores.json') if (cell/'scores.json').exists() else None,
                        executions=executions)
            mapping.append(record)
            if missing:
                pending.append((view,missing,record))
        for start in range(0,len(pending),8):
            chunk=pending[start:start+8];key=f'block-{len(orders):03d}'
            order=dict(initial=initial,views=[v for v,_,_ in chunk],repeats={v:r for v,r,_ in chunk},
                       panel_sha256=plan['panel_sha256'],phase='remaining',output=str(destination/'workers'/key))
            order_path=destination/'orders'/(key+'.json');write(order_path,order)
            orders.append(dict(key=key,path=str(order_path),sha256=sha(order_path),rollouts=sum(len(r) for _,r,_ in chunk)))
            for view,repeats,record in chunk:
                cell=Path(order['output'])/'cells'/initial/view
                if record['scores'] is None:record['scores']=str(cell/'scores.json')
                for r in repeats:record['executions'][r]=str(cell/f'execution_{r}.json')
    write(destination/'cell-manifest.json',dict(format='xr1_explicit_trial_sources_v1',records=mapping,
          source_rule='Existing atomic-complete trials retained; missing addresses assigned before outcomes; no fastest/successful winner'))
    cfg=dict(plan=plan,source=str(frozen),orders=orders,roots=list(map(str,roots)),pi05_subspaces=str(pi05_subspaces),
             retained_remaining_trials=retained,new_trials=sum(o['rollouts'] for o in orders),
             old_root=str(remaining),manifest_sha256=sha(destination/'cell-manifest.json'))
    if retained+cfg['new_trials']!=14720:raise ValueError('Scope changed during reassignment')
    write(destination/'plan.json',cfg)
    return cfg


def drain(root, remote_config):
    """All idle local slots run immediately; GNU joins only after explicit admission."""
    root=Path(root);cfg=read(root/'plan.json');plan=cfg['plan'];source=Path(cfg['source'])
    state=dict(status='RUNNING',pid=os.getpid(),completed_blocks=0,total_blocks=len(cfg['orders']),
               retained_trials=cfg['retained_remaining_trials'],new_trials=cfg['new_trials'],started_unix=time.time())
    if (root/'status.json').exists():raise ValueError('Queue already started; audit before any recovery')
    write(root/'status.json',state)
    pending=list(cfg['orders']);active={};done=[];launches=[];deadline=time.monotonic()+12*3600
    try:
        while pending or active:
            if time.monotonic()>deadline:raise TimeoutError('Drain deadline exceeded')
            slots=[('fresh',gpu) for gpu in range(8)]
            admission=Path(remote_config)
            remote=read(admission) if admission.exists() else {}
            if remote.get('status')=='PASS':slots += [('gnu',gpu) for gpu in range(8)]
            for slot,(proc,entry,log) in list(active.items()):
                code=proc.poll()
                if code is None:continue
                log.close();work=read(entry['path']);out=Path(work['output']);status=read(out/'status.json')
                if code!=0 or status['status']!='COMPLETE' or status['completed_rollouts']!=entry['rollouts']:
                    raise RuntimeError(f'Block failed: {entry["key"]}; no implicit retry')
                identity=read(out/'identity.json')
                if identity['work_order_sha256']!=entry['sha256'] or identity['checkpoint_receipt_sha256']!=plan['checkpoint_receipt_sha256']:
                    raise ValueError('Worker identity mismatch')
                for view in work['views']:
                    for r in work['repeats'][view]:validate_trial(out/'cells'/work['initial']/view/f'execution_{r}.json',work['initial'],r)
                done.append(entry['key']);del active[slot]
            for host,gpu in slots:
                if not pending or (host,gpu) in active:continue
                entry=pending.pop(0)
                if sha(entry['path'])!=entry['sha256']:raise ValueError('Frozen work order changed')
                work=read(entry['path'])
                env=dict(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
                         HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                         PYTHONPATH=str(source),HF_MODULES_CACHE=str(root/'hf-modules'/host))
                sim_python=plan['sim_python'];model_python=plan['model_python'];upstream=plan['upstream']
                if host=='gnu':
                    sim_python=remote['sim_python'];model_python=remote['model_python'];upstream=remote['upstream']
                    env.update(remote['environment'])
                    env['PYTHONPATH']=str(source)+':'+remote['environment']['PYTHONPATH']
                    env['XR1_MODEL_PYTHONPATH']=str(source)+':'+remote['model_site_packages']
                command=[sim_python,str(source/'scripts/dsol_paper1/runtime/robocasa_xr1_worker.py'),
                         '--panel',plan['panel'],'--checkpoint',plan['checkpoint'],'--upstream',upstream,
                         '--model-python',model_python,'--output',work['output'],'--formal','--phase','remaining',
                         '--work-order',entry['path'],'--execute']
                if host=='gnu':command=remote['ssh']+['env '+shlex.join([f'{k}={v}' for k,v in env.items()]+command)]
                log=(root/(entry['key']+'.log')).open('xb')
                proc=subprocess.Popen(command,cwd=source,env={**os.environ,**env},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[host,gpu]=(proc,entry,log)
                launches.append(dict(host=host,gpu=gpu,pid=proc.pid,order=entry['path'],command=command,environment=env))
                write(root/'launches.json',launches)
            state.update(completed_blocks=len(done),pending_blocks=len(pending),active_blocks=len(active),
                         hosts=sorted({host for host,gpu in active}),checked_unix=time.time())
            write(root/'status.json',state);time.sleep(5)
        if sha(root/'cell-manifest.json')!=cfg['manifest_sha256']:raise ValueError('Result ownership changed')
        state.update(status='ANALYZING');write(root/'status.json',state)
        command=[plan['sim_python'],'-m','AlphaBrain.research.dsol.analysis.xr1_fullview','--panel',plan['panel'],
                 '--roots',*cfg['roots'],'--output',str(root/'analysis'),'--pi05-subspaces',cfg['pi05_subspaces'],
                 '--cell-manifest',str(root/'cell-manifest.json')]
        with (root/'analysis.log').open('xb') as log:
            subprocess.run(command,cwd=source,env={**os.environ,'PYTHONPATH':str(source),'OPENBLAS_NUM_THREADS':'1'},
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
        state.update(status='COMPLETE',completed_unix=time.time());write(root/'status.json',state)
    except BaseException as exc:
        state.update(status='FAILED',error=repr(exc));write(root/'status.json',state);raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--remote-config',type=Path,required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if not args.execute:parser.error('Explicit --execute required')
    drain(args.root,args.remote_config)
