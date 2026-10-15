"""Matched five-window scoring, followed by reuse of independent task outcomes.

Window one must reproduce the historical selector exactly. No window is chosen
using heldout outcomes, and probe completion flags never enter the selector.
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..data.artifacts import read, write, sha
from .view_landscape import load_panel, association
from .robocasa_statistics import interval


def window_choices(scores):
    scores=np.asarray(scores,float)
    if scores.ndim!=4 or scores.shape[-2:]!=(8,5) or not np.isfinite(scores).all():
        raise ValueError('Expected finite initial x view x 8 repeats x 5 replans')
    # Preserve the original contiguous eight-draw summation, including its
    # floating-point order. Endpoint success is not available to this function.
    n,v=scores.shape[:2]
    means=np.empty((n,v,5))
    for i in range(n):
        for j in range(v):
            means[i,j]=np.cumsum([scores[i,j,:,k].copy().mean() for k in range(5)])/np.arange(1,6)
    return means,means.argmin(axis=1)


def read_cells(records,initials,views):
    expected={(i,v) for i in initials for v in views}
    keys=[(r['initial'],r['view']) for r in records]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('Exactly one owner per cell required')
    def inspect(row):
        cell=Path(row['cell']);receipt=cell/'complete.json';done=read(receipt);payload=cell/'probes.npz'
        digest=sha(payload)
        if (done['status']!='COMPLETE' or not done['window1_exact'] or done['arrays_sha256']!=digest
                or row.get('receipt_sha256',sha(receipt))!=sha(receipt)):
            raise ValueError(f'Invalid completion receipt: {cell}')
        with np.load(payload,allow_pickle=False) as z:
            scores=z['scores'].copy();saved=z['prefix_means'].copy()
        means,_=window_choices(scores[None,None])
        if not np.array_equal(saved,means[0,0]):raise ValueError('Saved prefix arithmetic mismatch')
        events=done['records']
        if [(r['repeat'],r['replan']) for r in events]!=[(r,k) for r in range(8) for k in range(5)]:
            raise ValueError('Incomplete score records')
        if any('early_terminal' in r for r in events):raise ValueError('True terminal cell was committed')
        # First success action step per scoring repeat; -1 means none observed.
        first=np.full(8,-1,int)
        for r in events:
            if r.get('success_steps') and first[r['repeat']]<0:first[r['repeat']]=min(r['success_steps'])
        return row,scores,first,done['seconds'],digest,sha(receipt)
    a=np.empty((len(initials),len(views),8,5));first=np.empty((*a.shape[:3],),int)
    times=np.empty(a.shape[:2]);sources=[];ii={x:i for i,x in enumerate(initials)};vv={x:v for v,x in enumerate(views)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for row,scores,event,seconds,digest,receipt_digest in pool.map(inspect,records):
            i,j=ii[row['initial']],vv[row['view']];a[i,j]=scores;first[i,j]=event;times[i,j]=seconds
            sources.append(dict(**row,payload_verified_sha256=digest,receipt_verified_sha256=receipt_digest))
    return a,first,times,sources


def summarize(scores,y,original,tasks,splits,extra,first_success,stride):
    means,chosen=window_choices(scores);q=np.asarray(y).mean(-1);tasks=np.asarray(tasks);splits=np.asarray(splits)
    if q.shape!=means.shape[:2] or not np.isin(y,[0,1]).all():raise ValueError('Independent binary endpoint matrix required')
    if not np.array_equal(means[...,0],original):raise ValueError('Window one did not reproduce all original scores')
    if not np.array_equal(chosen[:,0],original.argmin(1)):raise ValueError('Window one changed original choices')
    value=q[np.arange(len(q))[:,None],chosen];stay=q[:,extra].mean(1)
    populations={}
    for name,mask in [('all',np.ones(len(q),bool)),('seen',splits=='pretrain'),('heldout',splits=='target')]:
        if not mask.any():raise ValueError(f'Missing population: {name}')
        rows=[]
        for k in range(5):
            selected_first=first_success[np.arange(len(q)),chosen[:,k]]
            post=(selected_first>=0)&(selected_first<=k*stride)
            rows.append(dict(window=k+1,success=float(value[mask,k].mean()),
                gain_vs_window1=interval(value[mask,k]-value[mask,0],tasks[mask]),
                gain_vs_stay=interval(value[mask,k]-stay[mask],tasks[mask]),
                changed_vs_window1=int((chosen[mask,k]!=chosen[mask,0]).sum()),
                association=association(means[mask,:,k],q[mask]),
                selected_probe_post_success_fraction=float(post[mask].mean()),
                per_task=[dict(task=str(t),success=float(value[mask&(tasks==t),k].mean()),
                    window1=float(value[mask&(tasks==t),0].mean()),stay=float(stay[mask&(tasks==t)].mean())) for t in np.unique(tasks[mask])]))
        populations[name]=dict(initials=int(mask.sum()),stay=float(stay[mask].mean()),canonical=float(q[mask,0].mean()),
            historical_initial_accel=float(q[np.arange(len(q)),original.argmin(1)][mask].mean()),
            empirical_oracle=float(q[mask].max(1).mean()),windows=rows)
    return dict(window1_exact=True,populations=populations),means,chosen,value


def run(config_path):
    cfg=read(config_path);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    panel=load_panel(cfg['panel'])
    with np.load(cfg['endpoint_arrays'],allow_pickle=False) as z:
        if z['initials'].tolist()!=panel['initials'] or z['views'].tolist()!=panel['views']:raise ValueError('Endpoint axes changed')
        y=z['success'].copy();old=z['accel'].copy();models=z['models'].tolist()
    result=dict(protocol='matched_five_window_probes',sources={'config':sha(config_path),'endpoint_arrays':sha(cfg['endpoint_arrays'])},
        interpretation='Probe each candidate then reset to initial state; reuse four independent complete-task execution labels. Not online correction or new endpoint evaluation.',
        uncertainty='Hierarchical paired bootstrap over observed tasks/initials; four execution repeats held fixed. Not full execution-noise uncertainty.',
        post_success='Fixed-horizon probes include actual observations after success when environment has not terminated; no score imputation.',models={})
    arrays={}
    for model,entry in cfg['models'].items():
        if 'ownership' in entry:records=read(entry['ownership'])['records']
        else:
            root=Path(entry['root']);protocol=read(root/'protocol.json')
            if protocol['initials']!=panel['initials'] or protocol['views']!=panel['views'] or protocol['model']!=model:
                raise ValueError('Original probe protocol mismatch')
            records=[dict(initial=i,view=v,cell=str(root/'workers'/i/'cells'/i/v)) for i in panel['initials'] for v in panel['views']]
        scores,first,times,sources=read_cells(records,panel['initials'],panel['views'])
        m=models.index(model);summary,means,choices,values=summarize(scores,y[m],old[m],panel['tasks'],panel['splits'],panel['extra'],first,entry['stride'])
        summary['probe_cell_count']=len(records);summary['probe_seconds_sum']=float(times.sum())
        summary['success_seen_probe_count']=int((first>=0).sum());summary['stride']=entry['stride']
        result['models'][model]=summary
        write(out/(model+'-source-audit.json'),dict(records=sources))
        arrays.update({model+'__scores':scores,model+'__means':means,model+'__choices':choices,
            model+'__values':values,model+'__first_success_step':first,model+'__cell_seconds':times})
    np.savez_compressed(out/'windows.npz',**arrays,initials=panel['initials'],views=panel['views'])
    result['arrays_sha256']=sha(out/'windows.npz');write(out/'summary.json',result)
    from ..plotting.short_view_windows import plot
    plot(result,out)
    return result


def after_queue(config_path,queue_root,timeout_seconds=10800):
    """CPU-only completion hook. Never starts, cancels or reissues GPU work."""
    from ..execution.queue import summary
    cfg=read(config_path);status_path=Path(cfg['output']).with_suffix('.status.json')
    if status_path.exists():raise ValueError('Existing analysis hook receipt; inspect before restarting')
    config_digest=sha(config_path);started=time.time()
    write(status_path,dict(status='WAITING_FOR_QUEUE',queue=str(queue_root),started_unix=started,config_sha256=config_digest))
    try:
        while True:
            state=summary(queue_root)
            if state['status']=='COMPLETE':break
            if state['status'] not in ('PREPARED','RUNNING'):raise RuntimeError(f'Queue is not healthy: {state}')
            if time.time()-started>timeout_seconds:raise TimeoutError('Queue did not complete within analysis wait budget')
            time.sleep(20)
        if sha(config_path)!=config_digest:raise ValueError('Analysis configuration changed while waiting')
        write(status_path,dict(status='ANALYZING',queue=state,started_unix=started))
        result=run(config_path)
        write(status_path,dict(status='COMPLETE',started_unix=started,finished_unix=time.time(),
            models=list(result['models']),summary=str(Path(cfg['output'])/'summary.json')))
        return result
    except BaseException as exc:
        write(status_path,dict(status='FAILED',error=repr(exc),started_unix=started,finished_unix=time.time()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--after-queue');parser.add_argument('--timeout-seconds',type=float,default=10800)
    args=parser.parse_args()
    if args.after_queue:after_queue(args.config,args.after_queue,args.timeout_seconds)
    else:run(args.config)
