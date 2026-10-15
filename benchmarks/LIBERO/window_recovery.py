"""Retain verified window blocks and resume only missing blocks on new outputs."""
import argparse
from pathlib import Path
import time
import shutil

from AlphaBrain.research.dsol.data.artifacts import read,sha,write
from AlphaBrain.research.dsol.execution.manifest import task_key
from AlphaBrain.research.dsol.execution.queue import prepare as prepare_queue
from .window_queue import validate_block


def publish(config_file):
    cfg=read(config_file); worker=read(cfg['worker_config']); root=Path(worker['root'])
    ownership=[]; retained=[]; missing=[]
    for i in range(32):
        for model in ('canonical','broad'):
            old=root/'probes'/model/f'{i:02d}'; status=old/'status.json'
            complete=status.exists() and read(status)['status']=='COMPLETE'
            if complete:
                validate_block(cfg['original_worker_config'],model,i,old); retained.append((model,i))
            else:missing.append((model,i))
            ownership.append(dict(model=model,initial=i,retained=complete,
                output=str(old if complete else root/'probes-recovery-v1'/model/f'{i:02d}'),
                config=cfg['original_worker_config'] if complete else cfg['worker_config']))
    pilots=[('canonical',16),('broad',16)]
    if not set(pilots)<=set(missing):raise ValueError('Repair pilot ownership changed')
    capacities={h:len(g) for h,g in cfg['gpus'].items()}; assigned={h:[] for h in capacities}; costs={h:0 for h in capacities}
    for block in missing:
        if block in pilots:continue
        h=min(capacities,key=lambda x:(costs[x]/capacities[x],x)); assigned[h].append(block);costs[h]+=1
    path=Path(cfg['ownership'])
    if path.exists():raise FileExistsError(path)
    write(path,dict(status='PUBLISHED',retained=retained,pilots=pilots,partition=assigned,records=ownership,
        config_sha256=sha(config_file),old_results_unchanged=True,scientific_blocks=64))


def prepare(config_file,host,phase):
    cfg=read(config_file); worker=read(cfg['worker_config']); owned=read(cfg['ownership'])
    if owned['config_sha256']!=sha(config_file):raise ValueError('Recovery configuration changed')
    blocks=owned['pilots'] if phase=='pilot' else owned['partition'][host]
    if phase=='pilot' and host!='gnu':raise ValueError('Pilots assigned only to GNU')
    records=read(worker['inputs'])['records']; tasks=[];fingerprint=sha(cfg['worker_config'])
    for model,i in blocks:
        record=records[i]; item=next(r for r in owned['records'] if r['model']==model and r['initial']==i)
        if item['retained']:raise ValueError('Never resubmit a completed block')
        addresses=[dict(model=model,task=record['state']['task_id'],initial=record['state']['pair_key'],
            view=v,stream='score',repeat=n,noise=f'libero-window-20261012::{n}::windows1-5')
            for v in record['candidate_ids'] for n in range(8)]
        tasks.append(dict(key=task_key(cfg['experiment_id'],fingerprint,addresses),addresses=addresses,
            payload=dict(model=model,initial=i),output=item['output']))
    source=Path(worker['source']);paths={Path(config_file),Path(cfg['worker_config']),Path(cfg['ownership']),Path(worker['inputs']),Path(worker['anchors']),Path(worker['release'])}
    paths.update(p for d in ('AlphaBrain','scripts','benchmarks') for p in (source/d).rglob('*.py'))
    control=Path(cfg['control_roots'][host])/phase
    plan=dict(format='dsol_work_queue_v1',experiment_id=cfg['experiment_id'],protocol_sha256=fingerprint,
        tasks=tasks,slots=[dict(host=host,id=str(g)) for g in (cfg['gpus'][host][:2] if phase=='pilot' else cfg['gpus'][host])],
        artifacts=[dict(path=str(p),sha256=sha(p)) for p in sorted(paths)],
        backend=dict(module='benchmarks.LIBERO.window_queue',factory='factory'),
        runtime=dict(host=host,source=str(source),config=cfg['worker_config'],python=cfg['python'],
            workers_per_gpu=1,environment=cfg['environments'][host]))
    prepare_queue(plan,control);return control


def run(config_file,host):
    from AlphaBrain.research.dsol.execution.cli import main
    from benchmarks.Robocasa365.probe_queue import ProbeBackend
    cfg=read(config_file); root=Path(read(cfg['worker_config'])['root']); deadline=time.monotonic()+cfg['timeout_seconds']
    status=root/f'window-recovery-{host}.json'
    oldroot=Path(cfg['old_controls'][host]); plan=read(oldroot/'manifest.json'); state=read(oldroot/'state.json')
    if state['status']!='FAILED':raise ValueError('Expected stopped original queue')
    backend=ProbeBackend(plan,oldroot)
    for row in state['tasks'].values():
        if row.get('handle') and backend.alive(row['handle']):raise RuntimeError('Old worker still alive')
    def execute(phase):
        control=prepare(config_file,host,phase);write(status,dict(status='RUNNING',phase=phase,control=str(control)))
        try:
            main(['run','--execute','--root',str(control),'--timeout-seconds',str(max(1,int(deadline-time.monotonic()))),'--poll-seconds','2'])
            if read(control/'state.json')['status']!='COMPLETE':raise RuntimeError('Recovery queue incomplete')
        finally:
            shutil.copytree(control,root/'controls'/f'window-recovery-{host}-{phase}')
    try:
        if host=='gnu':execute('pilot')
        owned=read(cfg['ownership'])
        while True:
            ready=True
            for model,i in owned['pilots']:
                item=next(r for r in owned['records'] if r['model']==model and r['initial']==i)
                p=Path(item['output'])/'status.json'
                if p.exists() and read(p)['status']=='FAILED':raise RuntimeError('Repair pilot failed; equality remains strict')
                ready &= p.exists() and read(p)['status']=='COMPLETE'
            if ready:break
            if time.monotonic()>deadline:raise TimeoutError('Repair pilot deadline')
            write(status,dict(status='WAITING',phase='two_model_repair_pilot'));time.sleep(20)
        for model,i in owned['pilots']:
            item=next(r for r in owned['records'] if r['model']==model and r['initial']==i)
            validate_block(item['config'],model,i,item['output'])
        execute('production');write(status,dict(status='COMPLETE'))
    except BaseException as error:
        write(status,dict(status='FAILED',error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--host',choices=('fresh','gnu'));p.add_argument('--publish',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.publish:publish(a.config)
    elif a.execute and a.host:run(a.config,a.host)
    else:p.error('Explicit --publish or --host and --execute required')
