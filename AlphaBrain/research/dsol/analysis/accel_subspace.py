"""Exploratory action-subspace audit of stored traces; no new VLA inference.

The registered Accel uses all 50x32 internal coordinates. RoboCasa returns
the first 12 action coordinates and executes the first 25 chunk steps. This
audit tests sensitivity, not an outcome-driven replacement of that metric.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse

import numpy as np

from ..data.artifacts import read, sha, write
from ..metrics.accel import compute_accel_scores
from .robocasa_statistics import correlation, interval


VARIANTS = ('full32_h50', 'active12_h50', 'active12_h25', 'unused20_h50')


def trace_subspaces(trace, prefix=3):
    trace = np.asarray(trace, dtype=np.float64)
    if trace.shape != (8,10,50,32):
        raise ValueError('Expected frozen RoboCasa score trace shape')
    pieces=(trace,trace[...,:12],trace[:,:,:25,:12],trace[...,12:])
    scores=[]
    for piece in pieces:
        result=compute_accel_scores(piece,prefixes=(prefix,))[prefix]
        if result['degenerate'].any():
            raise ValueError('Degenerate exploratory subspace; do not treat as best')
        scores.append(result['score'])
    v=trace[:,:3]; dv=np.diff(v,axis=1)
    energy=[float(np.square(x[...,12:]).sum()/np.square(x).sum()) for x in (v,dv)]
    return np.stack(scores),np.array(energy)


def panel_cell(root, plan, baseline, initial, view):
    """Locate a score cell without changing the frozen split/reuse convention."""
    reuse = view in plan['reuse_views']
    local = baseline if reuse else plan
    base = Path(plan['baseline_plan']).parent if reuse else Path(root)
    found = [w for w in local['workers'] if initial in w['initials']]
    if len(found) != 1:
        raise ValueError('Expected exactly one official worker per initial')
    return base/'workers'/found[0]['worker_id']/'official'/'cells'/initial/view


def run_official_panel(root, output, posttraining_scores, population, workers=4):
    """Rescore official saved traces; reuse prior posttraining subspace analysis.

    No inference, rollout, new selection training, or alteration of old scores.
    Prefix 8 is an explicitly exploratory original-paper-inspired comparison.
    """
    from .view_landscape import load_panel, association

    root, output = Path(root), Path(output)
    plan = read(root/'plan.json'); baseline = read(plan['baseline_plan'])
    if sha(plan['baseline_plan']) != plan['baseline_plan_sha256']:
        raise ValueError('Baseline plan identity changed')
    panel = load_panel(root)
    if panel['models'] != ['official', 'canonical', 'broad']:
        raise ValueError('Unexpected model order')
    output.mkdir(parents=True, exist_ok=False)
    source_files = [root/'plan.json', root/'selection.json', root/'analysis/panel_arrays.npz',
                    Path(posttraining_scores), Path(population), Path(__file__)]
    write(output/'provenance.json', dict(inputs={str(p):sha(p) for p in source_files},
        protocol='Saved initial traces only; 40 matched initials, 97 views, 8 score and 4 execution repeats',
        original_metric_unchanged=True, new_rollouts=0))
    write(output/'progress.json', dict(status='RUNNING', completed=0, total=len(panel['initials'])))

    def initial_scores(initial):
        scored, later = [], []
        for view in panel['views']:
            cell = panel_cell(root, plan, baseline, initial, view)
            meta = read(cell/'scores.json')
            if sha(cell/'scores.npz') != meta['arrays_sha256']:
                raise ValueError('Official trace bytes changed')
            with np.load(cell/'scores.npz', allow_pickle=False) as data:
                trace = data['velocity_trace']
                value, _ = trace_subspaces(trace)
                later_value, _ = trace_subspaces(trace, prefix=8)
            old = np.array([r['metrics']['3']['score'] for r in meta['rows']])
            if not np.allclose(value[0], old, atol=1e-10, rtol=1e-8):
                raise ValueError('Official legacy score not reproduced')
            scored.append(value); later.append(later_value)
        return np.stack(scored), np.stack(later)

    n, v = len(panel['initials']), len(panel['views'])
    values = np.empty((3,n,v,len(VARIANTS),8))
    prefix8 = np.empty((n,v,len(VARIANTS),8))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(initial_scores, key):i for i,key in enumerate(panel['initials'])}
        for done, future in enumerate(as_completed(pending), start=1):
            i = pending[future]
            values[0,i], prefix8[i] = future.result()
            write(output/'progress.json', dict(status='RUNNING', completed=done, total=n))
    pop = read(population)
    indices = [next(i for i,j in enumerate(pop['jobs']) if j['initial_id']==key) for key in panel['initials']]
    if [x['view_id'] for x in pop['views']] != panel['views']:
        raise ValueError('Posttraining candidate axis changed')
    with np.load(posttraining_scores, allow_pickle=False) as old:
        if old['variants'].tolist() != list(VARIANTS):
            raise ValueError('Posttraining metric variants changed')
        values[1:] = old['scores'][:,indices]
    if not np.allclose(values[:,:,:,0].mean(-1), panel['accel'], rtol=1e-8, atol=1e-10):
        raise ValueError('Matched panel score identity failed')
    np.savez_compressed(output/'scores.npz', scores=values, official_prefix8=prefix8,
                        variants=np.array(VARIANTS), initials=np.array(panel['initials']),
                        views=np.array(panel['views']), models=np.array(panel['models']))
    q = panel['y'].mean(-1)
    result = dict(scope='Exploratory score-window diagnosis, not independent method confirmation',
                  source_rollouts_reused=True, original_score_reproduced=True, models={})
    for m, name in enumerate(panel['models']):
        result['models'][name] = {}
        variants = [(key, values[m,:,:,k].mean(-1)) for k,key in enumerate(VARIANTS)]
        if m == 0:
            variants += [(key+'_prefix8', prefix8[:,:,k].mean(-1)) for k,key in enumerate(VARIANTS)]
        for split in ('pretrain','target','all'):
            mask = np.ones(n,dtype=bool) if split=='all' else panel['splits']==split
            local_q = q[m,mask]; stay=local_q[:,panel['extra']].mean(1)
            rows = np.arange(mask.sum()); detail={}
            for key, score in variants:
                local_a=score[mask]; chosen=local_a.argmin(1)
                picked=local_q[rows,chosen]
                detail[key]=dict(reference=float(local_q[:,0].mean()),stay=float(stay.mean()),
                    selected=float(picked.mean()),gain=float((picked-stay).mean()),
                    gain_interval=interval(picked-stay,panel['tasks'][mask]),
                    same_choice_as_original=float(np.mean(chosen==panel['accel'][m,mask].argmin(1))),
                    association=association(local_a[:,1:],local_q[:,1:]))
            result['models'][name][split]=detail
    write(output/'summary.json',result)
    write(output/'progress.json',dict(status='COMPLETE',completed=n,total=n,new_rollouts=0))
    return result


def _initial(root, worker, initial, views):
    scores=[];energy=[]
    for view in views:
        cell=Path(root)/'workers'/worker/'cells'/initial/view
        meta=read(cell/'scores.json')
        if sha(cell/'scores.npz')!=meta['arrays_sha256']:
            raise ValueError('Stored trace changed')
        with np.load(cell/'scores.npz',allow_pickle=False) as arrays:
            computed,energies=trace_subspaces(arrays['velocity_trace'])
        original=np.array([r['metrics']['3']['score'] for r in meta['rows']])
        if not np.allclose(computed[0],original,atol=1e-10,rtol=1e-8):
            raise ValueError('Original metric not reproduced')
        scores.append(computed);energy.append(energies)
    return np.stack(scores),np.stack(energy)


def run(plan_file, analysis, output, workers=4):
    plan, analysis, output=read(plan_file),Path(analysis),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    pop=read(analysis/'population.json'); jobs=pop['jobs']; views=plan['view_ids']
    with np.load(analysis/'matrix.npz') as z:
        q=z['success'].mean(-1)
    values=np.empty((2,len(jobs),len(views),len(VARIANTS),8)); energies=np.empty((2,len(jobs),len(views),2))
    index={(w['arm'],i):w['worker_id'] for w in plan['workers'] for i in w['initials']}
    completed=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future={pool.submit(_initial,Path(plan_file).parent,index[arm,j['initial_id']],j['initial_id'],views):(m,i)
                for m,arm in enumerate(('canonical','broad')) for i,j in enumerate(jobs)}
        for f in as_completed(future):
            m,i=future[f]; values[m,i],energies[m,i]=f.result();completed+=1
            if completed%20==0:
                write(output/'progress.json',dict(status='RUNNING',completed_model_initials=completed,total=800))
    np.savez_compressed(output/'subspaces.npz',scores=values,unused_energy_fraction=energies,variants=np.array(VARIANTS))
    result=dict(scope='Exploratory score-definition sensitivity; original primary metric unchanged',models={})
    for m,arm in enumerate(('canonical','broad')):
        result['models'][arm]={}
        for split in ('pretrain','target'):
            ids=np.array([i for i,j in enumerate(jobs) if j['split']==split]);tasks=np.array([jobs[i]['task'] for i in ids]);rows=np.arange(len(ids))
            local_q=q[m,ids];r={}
            original=values[m,ids,:,0].mean(-1)
            for k,name in enumerate(VARIANTS):
                cost=values[m,ids,:,k].mean(-1)
                rho=[correlation(-a,b) for a,b in zip(cost,local_q)];rho=[x for x in rho if x is not None]
                r[name]=dict(selected_success=interval(local_q[rows,cost.argmin(1)],tasks),
                            median_success_rank_correlation=float(np.median(rho)),
                            mean_cost=float(cost.mean()),same_choice_as_original=float(np.mean(cost.argmin(1)==original.argmin(1))))
            r['unused_coordinate_squared_energy_fraction']=dict(velocity=float(energies[m,ids,:,0].mean()),velocity_change=float(energies[m,ids,:,1].mean()))
            result['models'][arm][split]=r
    write(output/'summary.json',result)
    write(output/'progress.json',dict(status='COMPLETE',completed_model_initials=completed,total=800))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path);parser.add_argument('--analysis',type=Path)
    parser.add_argument('--official-panel',type=Path)
    parser.add_argument('--posttraining-scores',type=Path);parser.add_argument('--population',type=Path)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.official_panel:
        if not args.posttraining_scores or not args.population:
            parser.error('Official audit requires posttraining scores and population identities')
        run_official_panel(args.official_panel,args.output,args.posttraining_scores,args.population,args.workers)
    else:
        if not args.plan or not args.analysis:
            parser.error('Legacy audit requires --plan and --analysis')
        run(args.plan,args.analysis,args.output,args.workers)


if __name__=='__main__':
    main()
