"""Prespecified recovery confirmation statistics from accepted endpoint data.

No candidate selection or outcome-based task filtering happens in this module.
The 32 disturbance positions are a fixed finite population, not 32 independent
initial states. Execution addresses are resampled jointly across all methods.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..data.artifacts import read, write, sha
from ..metrics.accel import compute_accel_scores

CONDITIONS = ['canonical', 'disturbance_uniform', 'window1', 'window2', 'window3', 'window4', 'window5']


def verify_frozen_artifacts(root, audit, groups_file):
    expected={}
    for path,digest in audit['source_files'].items():
        if sha(path)!=digest:raise ValueError('Audit input changed')
        if Path(path).name in ('manifest.json','pilot-manifest.json'):
            for a in read(path).get('artifacts',[]):
                if a['path'] in expected and expected[a['path']]!=a['sha256']:
                    raise ValueError('Conflicting frozen source identities')
                expected[a['path']]=a['sha256']
        if Path(path).name=='authorization.json':
            if read(path)['development_strata_sha256']!=sha(groups_file):
                raise ValueError('Development grouping changed after authorization')
    for info in audit['models'].values():
        for source in info['source_versions']:
            manifest=Path(source)/'source-manifest.json'
            expected[str(manifest)]=sha(manifest)
            for relative,digest in read(manifest)['files'].items():expected[str(Path(source)/relative)]=digest
    def check(item):
        path,digest=item
        if sha(path)!=digest:raise ValueError('Frozen artifact changed: '+path)
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(check,expected.items()))
    write(root/'artifact-audit.json',dict(status='PASS',files=expected,grouping_frozen=True))


def verify_accel_traces(root):
    """Recompute all stored Accel values, not only the five-window arithmetic."""
    result={}
    for model in ('official','xr1'):
        rows=read(root/(model+'-probe-audit.json'))['records']
        def check(row):
            cell=Path(row['cell'])
            if sha(cell/'probes.npz')!=row['payload_verified_sha256']:
                raise ValueError('Probe payload changed since acceptance')
            with np.load(cell/'probes.npz',allow_pickle=False) as z:
                trace=z['velocity_trace']; scores=z['scores']
                metric_trace=trace if model=='official' else trace[...,:16,:12]
                metric=compute_accel_scores(metric_trace.reshape(40,*metric_trace.shape[2:]),prefixes=(3,))[3]
                delta=float(np.max(np.abs(metric['score'].reshape(8,5)-scores)))
                if not np.allclose(metric['score'].reshape(8,5),scores,rtol=1e-12,atol=1e-12):
                    raise ValueError('Accel differs from saved velocities')
                with np.load(cell/'initial-anchor.npz',allow_pickle=False) as a:
                    if not np.array_equal(trace[:,0],a['velocity_trace']):
                        raise ValueError('Window-one velocity differs from anchor')
            return delta,int(metric['degenerate'].sum())
        with ThreadPoolExecutor(max_workers=8) as pool:checks=list(pool.map(check,rows))
        result[model]=dict(cells=len(rows),scores=len(rows)*40,max_absolute_difference=max(d for d,_ in checks),
                           degenerate_scores=sum(n for _,n in checks),initial_velocities_exact=True)
        if result[model]['degenerate_scores']:
            raise ValueError('Degenerate scores require explicit scientific review')
    write(root/'metric-audit.json',dict(status='PASS',models=result,metric_sha256=sha(compute_accel_scores.__code__.co_filename)))
    return result


def condition_outcomes(success, choices, extra):
    y, choices = np.asarray(success), np.asarray(choices)
    if y.ndim != 3 or y.shape[2] != 8 or choices.shape != (len(y),5):
        raise ValueError('Expected eight repeats and five frozen choices')
    selected = y[np.arange(len(y))[:,None], choices]
    out = np.concatenate([y[:,0:1], y[:,extra].mean(1)[:,None], selected],axis=1)
    if not np.isfinite(out).all() or not ((out>=0)&(out<=1)).all():
        raise ValueError('Missing endpoint or invalid probability')
    return out


def paired_bootstrap(values, tasks, splits, layouts, *, draws=10000, seed=20261012):
    """Task -> split-stratified initial -> paired execution resampling.

Also return a crossed Bayesian cluster sensitivity with a shared weight for the
same official split/layout across tasks. It is not an additional independent
experiment. Neither interval accounts for retraining or a new task population.
"""
    x=np.asarray(values,float); tasks=np.asarray(tasks); splits=np.asarray(splits); layouts=np.asarray(layouts)
    if x.ndim!=3 or x.shape[-1]!=8 or not np.isfinite(x).all() or len(tasks)!=len(x):
        raise ValueError('Finite initial x method x eight execution outcomes required')
    rng=np.random.default_rng(seed); unique=np.unique(tasks); nmethod=x.shape[1]
    result=np.zeros((draws,nmethod)); denominator=np.zeros(draws)
    task_draw=rng.integers(len(unique),size=(draws,len(unique)))
    for slot in range(len(unique)):
        for ti,task in enumerate(unique):
            mask=np.flatnonzero(task_draw[:,slot]==ti)
            if not len(mask): continue
            sub_splits=np.unique(splits[tasks==task]); total=np.zeros((len(mask),nmethod))
            for split in sub_splits:
                ids=np.flatnonzero((tasks==task)&(splits==split))
                chosen=ids[rng.integers(len(ids),size=(len(mask),len(ids)))]
                repeats=rng.integers(8,size=(*chosen.shape,8))
                # Preserve common random numbers across candidate methods.
                data=x[chosen].transpose(0,1,3,2)
                total+=np.take_along_axis(data,repeats[...,None],axis=2).mean((1,2))/len(sub_splits)
            result[mask]+=total*len(sub_splits)
            denominator[mask]+=len(sub_splits)
    result/=denominator[:,None]
    # Shared scene layouts are correlated design units. Retain this dependency
    # rather than pretending every physical initial is an unrelated kitchen.
    layout_keys=np.array([str(s)+'/'+str(l) for s,l in zip(splits,layouts)])
    _,li=np.unique(layout_keys,return_inverse=True)
    crossed=np.empty_like(result)
    for start in range(0,draws,200):
        b=min(200,draws-start)
        lw=rng.exponential(size=(b,li.max()+1))[:,li]
        iw=rng.exponential(size=(b,len(x)))*lw
        rw=rng.integers(8,size=(b,len(x),8))
        base=np.broadcast_to(x.transpose(0,2,1),(b,len(x),8,nmethod))
        sampled=np.take_along_axis(base,rw[...,None],axis=2).mean(2)
        unit_counts=np.array([len(np.unique(splits[tasks==t])) for t in unique])
        tw=rng.exponential(size=(b,len(unique)))*unit_counts; means=[]
        for task in unique:
            sm=[]
            for split in np.unique(splits[tasks==task]):
                ids=np.flatnonzero((tasks==task)&(splits==split)); w=iw[:,ids]
                sm.append((sampled[:,ids]*w[...,None]).sum(1)/w.sum(1)[:,None])
            means.append(np.mean(sm,axis=0))
        crossed[start:start+b]=(np.stack(means,axis=1)*tw[...,None]).sum(1)/tw.sum(1)[:,None]
    return result,crossed


def interval(point, samples, sensitivity):
    return dict(mean=float(point),ci95=np.quantile(samples,[.025,.975]).tolist(),
                family2_ci975=np.quantile(samples,[.0125,.9875]).tolist(),
                shared_layout_ci95=np.quantile(sensitivity,[.025,.975]).tolist())


def summarize_population(x, rows, *, draws=10000):
    tasks=np.array([r['task'] for r in rows]); splits=np.array([r['split'] for r in rows])
    layouts=np.array([r['layout_id'] for r in rows])
    # Equal frozen task/split units. Main panels are exactly balanced. Preserve
    # that unit in descriptive strata, where some tasks have only one split.
    avg=np.mean([x[(tasks==t)&(splits==s)].mean((0,2))
                 for t in np.unique(tasks) for s in np.unique(splits[tasks==t])],axis=0)
    boot,cross=paired_bootstrap(x,tasks,splits,layouts,draws=draws)
    methods={}
    for k,name in enumerate(CONDITIONS):
        methods[name]=dict(success=interval(avg[k],boot[:,k],cross[:,k]),
            gain_vs_stay=interval(avg[k]-avg[1],boot[:,k]-boot[:,1],cross[:,k]-cross[:,1]),
            gain_vs_window1=interval(avg[k]-avg[2],boot[:,k]-boot[:,2],cross[:,k]-cross[:,2]),
            gap_vs_canonical=interval(avg[k]-avg[0],boot[:,k]-boot[:,0],cross[:,k]-cross[:,0]))
    per_task=[]
    for task in np.unique(tasks):
        v=x[tasks==task].mean((0,2))
        per_task.append(dict(task=str(task),initials=int((tasks==task).sum()),
            methods=dict(zip(CONDITIONS,map(float,v))),window3_gain=float(v[4]-v[1]),
            window3_vs_window1=float(v[4]-v[2]),view_loss=float(v[0]-v[1])))
    return dict(initials=len(x),tasks=len(np.unique(tasks)),methods=methods,per_task=per_task)


def run(accepted, groups_file):
    root=Path(accepted); audit=read(root/'audit.json')
    if (audit['status']!='PASS' or sha(root/'accepted-arrays.npz')!=audit['arrays_sha256']
            or sha(root/'population.json')!=audit['population_sha256']):
        raise ValueError('Accepted arrays changed')
    population=read(root/'population.json'); groups=read(groups_file)
    verify_frozen_artifacts(root,audit,groups_file)
    verify_accel_traces(root)
    group_map={(r['model'],r['task'],r['split']):r for r in groups['rows']}
    result=dict(status='COMPLETE',primary_window=3,conditions=CONDITIONS,models={},
        sources={str(root/'audit.json'):sha(root/'audit.json'),str(root/'metric-audit.json'):sha(root/'metric-audit.json'),
                 str(root/'artifact-audit.json'):sha(root/'artifact-audit.json'),str(groups_file):sha(groups_file)},
        analysis_sha256=sha(__file__),new_rollouts=0,
        uncertainty='10000 paired task/initial/execution bootstrap draws, splits stratified. Shared-layout crossed Bayesian cluster sensitivity reported separately. Primary overall two-model family uses 97.5% marginal intervals (Bonferroni). Other windows, tasks, and strata are descriptive; no best-window confirmation claim.',
        scope='Fresh initials of ten known tasks; probe then reset, fixed installation during complete task. No online camera movement; no new full-97-view oracle.')
    records=[]; array_out={}
    with np.load(root/'accepted-arrays.npz',allow_pickle=False) as z:
        views=z['views'].tolist(); extra=[views.index(v) for v in z['extra_views'].tolist()]
        for model,rows in population.items():
            y=z[model+'__success'];choices=z[model+'__choices'];first=z[model+'__first_success']
            x=condition_outcomes(y,choices,extra);array_out[model]=x
            reports={}
            for label,split in [('all',None),('seen','pretrain'),('heldout','target')]:
                ids=[i for i,r in enumerate(rows) if split is None or r['split']==split]
                report=summarize_population(x[ids],[rows[i] for i in ids])
                report['changed_vs_window1']=(choices[ids]!=choices[ids,0,None]).sum(0).tolist()
                report['canonical_choices']=(choices[ids]==0).sum(0).tolist()
                stride=25 if model=='official' else 16
                # Success before obtaining the last score is not a selection input.
                report['selected_probe_post_success_fraction']=[float(((first[np.array(ids),choices[ids,k]]>=0)&
                    (first[np.array(ids),choices[ids,k]]<=k*stride)).mean()) for k in range(5)]
                reports[label]=report
            strata={}
            for group in sorted({r['group'] for r in groups['rows'] if r['model']==model}):
                ids=[i for i,r in enumerate(rows) if group_map[model,r['task'],r['split']]['group']==group]
                if ids:
                    strata[group]=summarize_population(x[ids],[rows[i] for i in ids],draws=4000)
                    strata[group]['frozen_groups']=[r for r in groups['rows'] if r['model']==model and r['group']==group]
            result['models'][model]=dict(populations=reports,frozen_strata=strata)
            for i,r in enumerate(rows):
                records.append(dict(model=model,**r,**dict(zip(CONDITIONS,x[i].mean(-1).tolist()))))
    np.savez_compressed(root/'condition-outcomes.npz',**array_out)
    result['condition_outcomes_sha256']=sha(root/'condition-outcomes.npz')
    with (root/'per-initial.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write(root/'summary.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--accepted',required=True);p.add_argument('--groups',required=True)
    a=p.parse_args();run(a.accepted,a.groups)
