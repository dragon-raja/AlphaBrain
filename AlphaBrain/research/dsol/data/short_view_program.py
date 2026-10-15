"""Prepare per-host dynamic queues for already admitted five-window probes."""
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .artifacts import read,write,sha
from ..execution.manifest import task_key
from ..execution.queue import prepare


def partition_missing(initials, views, reused, chunk_size=24):
    """Deterministic missing-cell chunks, independent of scores and outcomes."""
    if chunk_size<1 or len(set(initials))!=len(initials) or len(set(views))!=len(views):
        raise ValueError('Unique axes and positive chunk size required')
    expected={(i,v) for i in initials for v in views}
    if not set(reused)<=expected:raise ValueError('Reuse outside frozen population')
    chunks=[]
    for initial in initials:
        missing=[v for v in views if (initial,v) not in reused]
        for start in range(0,len(missing),chunk_size):
            chunks.append(dict(initial=initial,views=missing[start:start+chunk_size]))
    return chunks


def audit_reuse(old_root, panel, model, workers=8):
    """Accept only committed, hashed cells with exact original window-one scores."""
    old_root,panel=Path(old_root),Path(panel);plan=read(panel/'plan.json')
    old_protocol=read(old_root/'protocol.json')
    if old_protocol['model']!=model or old_protocol['initials']!=plan['initial_ids'] or old_protocol['views']!=plan['all_views']:
        raise ValueError('Old population mismatch')
    with np.load(panel/'analysis/panel_arrays.npz',allow_pickle=False) as z:
        if z['initials'].tolist()!=plan['initial_ids'] or z['views'].tolist()!=plan['all_views']:
            raise ValueError('Original panel axes mismatch')
        anchor=z['accel'][z['models'].tolist().index(model)].copy()
    source=Path(old_protocol['source']);probe_sha=sha(source/'benchmarks/Robocasa365/short_view_probe.py')
    config_sha=sha(source/'configs/dsol_paper1/short_view_probe_20261008.json')
    for initial in plan['initial_ids']:
        identity_path=old_root/'workers'/initial/'identity.json'
        if not identity_path.exists():continue
        identity=read(identity_path)
        if (identity['model']!=model or identity['initials']!=[initial] or identity['views']!=plan['all_views']
                or identity['source_sha256']!=probe_sha or identity['config_sha256']!=config_sha):
            raise ValueError('Old worker identity mismatch')
    def inspect(item):
        ii,vi,initial,view=item;cell=old_root/'workers'/initial/'cells'/initial/view
        receipt=cell/'complete.json'
        if not receipt.exists():return None
        done=read(receipt);payload=cell/'probes.npz'
        if done['status']!='COMPLETE' or not done['window1_exact'] or done['arrays_sha256']!=sha(payload):
            raise ValueError(f'Invalid committed receipt: {cell}')
        with np.load(payload,allow_pickle=False) as z:
            scores=z['scores'];means=z['prefix_means']
            if scores.shape!=(8,5) or not np.isfinite(scores).all():raise ValueError('Incomplete probe')
            expected=np.cumsum([scores[:,k].copy().mean() for k in range(5)])/np.arange(1,6)
            if not np.array_equal(expected,means) or means[0]!=anchor[ii,vi]:raise ValueError('Original anchor mismatch')
        records=done['records']
        if [(r['repeat'],r['replan']) for r in records]!=[(r,k) for r in range(8) for k in range(5)]:
            raise ValueError('Incomplete record coverage')
        if any('early_terminal' in r for r in records):raise ValueError('Terminal cell cannot be reused')
        return dict(initial=initial,view=view,cell=str(cell),receipt_sha256=sha(receipt),arrays_sha256=done['arrays_sha256'])
    items=[(ii,vi,i,v) for ii,i in enumerate(plan['initial_ids']) for vi,v in enumerate(plan['all_views'])]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [r for r in pool.map(inspect,items) if r is not None]


def build_missing(runtime_path,queue_root,old_root,chunk_size=24):
    """New immutable run; preserve old failures and explicitly map reused cells."""
    runtime_path=Path(runtime_path).resolve()
    runtime=read(runtime_path);source=Path(runtime['source']);cfg=read(runtime['config'])
    panel=Path(cfg['panel']);plan=read(panel/'plan.json');selection=read(panel/'selection.json')
    initials,views=plan['initial_ids'],plan['all_views'];jobs={j['initial_id']:j for j in selection['jobs']}
    if len(initials)!=40 or len(views)!=97:raise ValueError('Original panel required')
    admission=read(runtime['admission']);admission_identity=read(Path(runtime['admission']).with_name('identity.json'))
    if (admission['status']!='COMPLETE' or admission['completed_cells']!=2 or admission['model']!=runtime['model']
            or admission_identity['source_sha256']!=sha(source/'benchmarks/Robocasa365/short_view_probe.py')
            or admission_identity['config_sha256']!=sha(runtime['config'])):
        raise ValueError('Repaired source requires real admission')
    reused=audit_reuse(old_root,panel,runtime['model'])
    chunks=partition_missing(initials,views,{(r['initial'],r['view']) for r in reused},chunk_size)
    result=Path(runtime['result_root']);result.mkdir(parents=True,exist_ok=False)
    write(result/'reuse.json',dict(old_root=str(old_root),records=reused,
        rule='Committed complete cells only; old partial files retained and never imputed'))
    protocol=dict(format='short_view_five_window_fixed_horizon_v2',model=runtime['model'],initials=initials,views=views,
        repeats=list(range(8)),windows=[1,2,3,4,5],old_protocol=str(Path(old_root)/'protocol.json'),
        old_protocol_sha256=sha(Path(old_root)/'protocol.json'),reuse_sha256=sha(result/'reuse.json'),
        change='Success alone no longer stops fixed-horizon probes; log success steps; actual termination fails closed',
        unchanged='Original score noise/input/metric, all candidates, prefix averaging and independent endpoint matrix',
        post_success='Scores may include post-completion observations; completion flags never used for scoring',
        source=str(source),admission=runtime['admission'],admission_sha256=sha(runtime['admission']),
        reused_cells=len(reused),new_cells=sum(len(c['views']) for c in chunks),chunk_size=chunk_size)
    write(result/'protocol.json',protocol);fingerprint=sha(result/'protocol.json')
    name='short-view-'+runtime['model']+'-fixed-horizon-repair-20261009';tasks=[];owners=list(reused)
    for n,chunk in enumerate(chunks):
        initial=chunk['initial'];chosen=chunk['views'];output=result/'workers'/f'{initial}-{n:03d}'
        addresses=[dict(model=runtime['model'],task=jobs[initial]['task'],initial=initial,view=v,
            stream='score',repeat=r,noise=f'original_score/{initial}/{r}/replans0-4') for v in chosen for r in range(8)]
        tasks.append(dict(key=task_key(name,fingerprint,addresses),addresses=addresses,output=str(output),payload=chunk))
        owners.extend(dict(initial=initial,view=v,cell=str(output/'cells'/initial/v)) for v in chosen)
    write(result/'ownership.json',dict(records=owners,rule='One predetermined owner per original cell; no result-based merging'))
    inputs=[result/'protocol.json',result/'reuse.json',result/'ownership.json',Path(runtime_path),
        Path(runtime['config']),panel/'plan.json',panel/'selection.json',Path(runtime['admission']),Path(__file__)]
    inputs += [p for p in source.rglob('*.py') if '__pycache__' not in p.parts]
    manifest=dict(format='dsol_work_queue_v1',experiment_id=name,protocol_sha256=fingerprint,
        slots=[dict(host=runtime['host'],id=str(i)) for i in range(8)],tasks=tasks,
        artifacts=[dict(path=str(p),sha256=sha(p)) for p in inputs],runtime=runtime,
        backend=dict(module='benchmarks.Robocasa365.probe_queue',factory='factory'))
    write(result/'manifest.json',manifest);prepare(manifest,queue_root)
    return dict(status='PREPARED',root=str(queue_root),blocks=len(tasks),reused_cells=len(reused),
        missing_cells=protocol['new_cells'],short_probes=protocol['new_cells']*8)


def build(runtime_path,queue_root):
    runtime_path=Path(runtime_path).resolve()
    runtime=read(runtime_path);source=Path(runtime['source']);cfg=read(runtime['config'])
    panel=Path(cfg['panel']);plan=read(panel/'plan.json');selection=read(panel/'selection.json')
    jobs={j['initial_id']:j for j in selection['jobs']};initials=plan['initial_ids'];views=plan['all_views']
    if len(initials)!=40 or len(views)!=97:raise ValueError('Original full small panel required')
    admission=read(runtime['admission'])
    if admission['status']!='COMPLETE' or admission['completed_cells']!=2 or admission['model']!=runtime['model']:
        raise ValueError('Real probe admission required before queue preparation')
    result=Path(runtime['result_root']);result.mkdir(parents=True,exist_ok=False)
    protocol=dict(format='short_view_five_window_v1',model=runtime['model'],initials=initials,views=views,
        repeats=list(range(8)),windows=[1,2,3,4,5],score_definition='Original native Accel p3; mean across 8 paired probes and prefix',
        first_replan='Exact original score-stream noise, cached input and raw velocity comparison on every cell',
        later_noise='Same score stream indexed by replan1..4; execution stream never used',
        final_evaluation='Reuse original independent four-noise endpoint matrix; no new complete task rollout',
        short_probe_count=40*97*8,early_termination='Fail closed; keep partial evidence and stop queue for protocol review',
        physical_camera_motion=False,step_stride=16 if runtime['model']=='xr1' else 25,
        source=str(source),admission=runtime['admission'],admission_sha256=sha(runtime['admission']),
        output_scope='All original tasks; seen and heldout reported separately; no test-based window selection')
    write(result/'protocol.json',protocol);fingerprint=sha(result/'protocol.json');name='short-view-'+runtime['model']+'-20261008'
    tasks=[]
    for initial in initials:
        addresses=[dict(model=runtime['model'],task=jobs[initial]['task'],initial=initial,view=view,
            stream='score',repeat=r,noise=f'original_score/{initial}/{r}/replans0-4') for view in views for r in range(8)]
        tasks.append(dict(key=task_key(name,fingerprint,addresses),addresses=addresses,
            output=str(result/'workers'/initial),payload=dict(initial=initial,views=views)))
    inputs=[result/'protocol.json',Path(runtime_path),Path(runtime['config']),panel/'plan.json',panel/'selection.json',
            Path(runtime['admission'])]
    inputs += [p for p in source.rglob('*.py') if '__pycache__' not in p.parts]
    manifest=dict(format='dsol_work_queue_v1',experiment_id=name,protocol_sha256=fingerprint,
        slots=[dict(host=runtime['host'],id=str(i)) for i in range(8)],tasks=tasks,
        artifacts=[dict(path=str(p),sha256=sha(p)) for p in inputs],runtime=runtime,
        backend=dict(module='benchmarks.Robocasa365.probe_queue',factory='factory'))
    write(result/'manifest.json',manifest);prepare(manifest,queue_root)
    return dict(status='PREPARED',root=str(queue_root),blocks=len(tasks),short_probes=40*97*8)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',required=True);parser.add_argument('--queue',required=True)
    parser.add_argument('--reuse-root');parser.add_argument('--chunk-size',type=int,default=24)
    args=parser.parse_args()
    print(build_missing(args.runtime,args.queue,args.reuse_root,args.chunk_size) if args.reuse_root else build(args.runtime,args.queue))
