"""Offline cross-benchmark noise/geometry diagnostics, not causal attribution.

Uses existing paired repeats only. No renderer, new rollout, fitting or change
to the registered primary metric. Negative variance estimates are retained.
"""
from itertools import combinations
from pathlib import Path
import argparse

import numpy as np
from scipy.stats import rankdata

from ..data.artifacts import read, write, sha
from .robocasa_statistics import correlation, interval


def view_components(y):
    """Cross-repeat covariance estimates stable across-view variance.

    With independent repeat streams, E[z_r(v) z_s(v)] for r!=s equals
    Var_v p(v), after centering each repeat over candidate views. CRN within
    one repeat is allowed. Renderer dependence across repeats is a caveat.
    """
    y=np.asarray(y,dtype=float)
    if y.ndim!=3 or y.shape[-1]<2 or not np.isfinite(y).all():
        raise ValueError('Expected finite initial x view x repeated outcomes')
    r=y.shape[-1]
    centered=y-y.mean(1,keepdims=True)
    observed=np.mean(centered.mean(-1)**2,axis=1)
    signal=np.mean(centered.sum(-1)**2-(centered**2).sum(-1),axis=1)/(r*(r-1))
    # Fraction of ordered pairs of distinct repeats with different binary results.
    if not np.all((y==0)|(y==1)):
        flip=None  # Aggregate task/global fields are continuous, not binary.
    else:
        count=y.sum(-1)
        flip=(2*count*(r-count)/(r*(r-1))).mean(1)
    return dict(signal=signal,observed=observed,noise_of_mean=observed-signal,flip=flip)


def scalar_metrics(y, accel, extra):
    q=y.mean(-1);rows=np.arange(len(q));cost=accel.mean(-1)
    rho=[correlation(-a,b) for a,b in zip(cost,q)];rho=[r for r in rho if r is not None]
    return dict(reference=float(q[:,0].mean()),stay=float(q[:,extra].mean()),
                min_accel=float(q[rows,cost.argmin(1)].mean()),empirical_oracle=float(q.max(1).mean()),
                median_accel_rank_correlation=float(np.median(rho)),defined_rank_initials=len(rho))


def noise_harmonization(y,accel,extra,*,draws=256,seed=20260923):
    """4-of-32 reanalysis; ranges describe subsets of the stored bank, not CIs."""
    y=np.asarray(y,float);rng=np.random.default_rng(seed);records=[]
    if y.shape[-1]<=4:raise ValueError('Need more than four existing repeats')
    for _ in range(draws):
        selected=np.sort(rng.choice(y.shape[-1],4,replace=False))
        remaining=np.setdiff1d(np.arange(y.shape[-1]),selected)
        sample=y[...,selected];q=sample.mean(-1);winner=q.argmax(1)
        metrics=scalar_metrics(sample,accel,extra)
        metrics['selected4_evaluated_remaining']=float(y[...,remaining].mean(-1)[np.arange(len(y)),winner].mean())
        metrics['min_accel_gain']=metrics['min_accel']-metrics['stay']
        records.append(metrics)
    return {key:dict(mean=float(np.mean([d[key] for d in records])),
                    subset_q025_q975=np.quantile([d[key] for d in records],[.025,.975]).tolist())
            for key in records[0]}


def noise_splits(y):
    """Simple finite-repeat diagnostic; selection always precedes evaluation."""
    y=np.asarray(y,float);r=y.shape[-1];rows=np.arange(len(y));results=[]
    partitions=list(combinations(range(4),2)) if r==4 else [tuple(range(r//2)),tuple(range(r//2,r))]
    for selection in partitions:
        evaluation=[i for i in range(r) if i not in selection]
        a,b=y[...,selection].mean(-1),y[...,evaluation].mean(-1)
        ranks=[correlation(x,z) for x,z in zip(a,b)];ranks=[v for v in ranks if v is not None]
        results.append(dict(selected_success=float(b[rows,a.argmax(1)].mean()),
                            global_view_rank_correlation=correlation(a.mean(0),b.mean(0)),
                            median_view_rank_correlation=float(np.median(ranks)),defined_initials=len(ranks)))
    return results


def common_parameter_box(y, accel, extra, parameters):
    """Sensitivity check only: equal angle bounds do not equalize rig geometry."""
    parameters = np.asarray(parameters, float)
    if parameters.shape != (y.shape[1], 3):
        raise ValueError('Expected one azimuth/elevation/radius tuple per view')
    if accel.ndim != 3 or accel.shape[:2] != y.shape[:2]:
        raise ValueError('Score and outcome initial/view axes differ')
    keep = ((np.abs(parameters[:, 0]) <= 30) & (np.abs(parameters[:, 1]) <= 15)
            & (parameters[:, 2] >= .9) & (parameters[:, 2] <= 1.2))
    extra_keep = np.intersect1d(np.flatnonzero(keep), extra)
    if not len(extra_keep):
        raise ValueError('No extra views in common parameter box')
    cost = np.where(keep[None, :], accel.mean(-1), np.inf)
    q = y.mean(-1)
    return dict(candidate_count=int(keep.sum()), extra_count=len(extra_keep),
                extra_success=float(q[:, extra_keep].mean()),
                min_accel=float(q[np.arange(len(q)), cost.argmin(1)].mean()))


def balanced_initial_components(y, tasks, *, draws=256, seed=20260923):
    """Two initials/task sensitivity; subset ranges are not confidence intervals."""
    groups = [np.flatnonzero(tasks == task) for task in np.unique(tasks)]
    if any(len(g) < 2 for g in groups):
        raise ValueError('Need at least two initials per task')
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(draws):
        chosen = np.concatenate([rng.choice(g, 2, replace=False) for g in groups])
        selected = y[chosen]
        signal = view_components(selected)['signal'].mean()
        task = view_components(selected.reshape(len(groups), 2, *y.shape[1:]).mean(1))['signal'].mean()
        rows.append([signal, task, signal - task])
    values = np.asarray(rows)
    return {key: dict(mean=float(values[:, i].mean()),
                      subset_q025_q975=np.quantile(values[:, i], [.025, .975]).tolist())
            for i, key in enumerate(('total', 'task', 'initial_specific'))}


def rank_field_association(y, accel):
    """Pooled within-initial score-rank / success-field association.

    Not Spearman between two ranked variables: only the score is ranked.
    Cross-repeat covariance removes estimated execution measurement variance
    from the denominator. Score-bank uncertainty is not corrected. Values can
    exceed one in finite samples; do not clip them into apparent validity.
    """
    y, accel = np.asarray(y, float), np.asarray(accel, float)
    if accel.ndim != 3 or accel.shape[:2] != y.shape[:2]:
        raise ValueError('Score and outcome initial/view axes differ')
    x = rankdata(-accel.mean(-1), axis=1)
    x -= x.mean(1, keepdims=True)
    q = y.mean(-1)
    q -= q.mean(1, keepdims=True)
    covariance, score_variance = np.mean(x * q), np.mean(x * x)
    components = view_components(y)
    observed, stable = components['observed'].mean(), components['signal'].mean()
    return dict(observed=float(covariance / np.sqrt(score_variance * observed))
                if score_variance > 0 and observed > 0 else None,
                execution_noise_adjusted=float(covariance / np.sqrt(score_variance * stable))
                if score_variance > 0 and stable > 0 else None,
                caveat='Exploratory association, not causal, not per-initial Spearman; '
                       'independent execution repeats assumed; score-bank noise not corrected')


def model_summary(y,accel,tasks,extra):
    pieces=view_components(y);signal=pieces['signal'];observed=pieces['observed']
    task_means=np.stack([y[tasks==t].mean(0) for t in np.unique(tasks)])
    task_signal=float(view_components(task_means)['signal'].mean())
    global_signal=float(view_components(y.mean(0,keepdims=True))['signal'][0])
    metrics=scalar_metrics(y,accel,extra)
    half=accel.shape[-1]//2
    ar=[correlation(a,b) for a,b in zip(accel[...,:half].mean(-1),accel[...,half:].mean(-1))]
    ar=[x for x in ar if x is not None]
    q=y.mean(-1);fixed=accel.mean(-1).argmin(1);rows=np.arange(len(y))
    return dict(**metrics,execution_repeats=y.shape[-1],initials=len(y),tasks=len(np.unique(tasks)),
        stable_view_variance=interval(signal,tasks), observed_view_variance=float(observed.mean()),
        estimated_noise_of_mean=float(pieces['noise_of_mean'].mean()),
        signal_fraction_of_observed_variance=float(signal.mean()/observed.mean()),
        estimated_stable_view_sd=float(np.sqrt(max(0,signal.mean()))),
        negative_initial_signal_estimate_fraction=float(np.mean(signal<0)),
        repeat_outcome_disagreement=float(pieces['flip'].mean()),
        stable_task_view_variance=task_signal,stable_global_view_variance=global_signal,
        stable_initial_specific_variance=float(signal.mean()-task_signal),
        score_half_rank_median=float(np.median(ar)),
        rank_field_association=rank_field_association(y,accel),
        cross_repeat_splits=noise_splits(y),
        accel_gain_vs_stay=interval(q[rows,fixed]-q[:,extra].mean(1),tasks),
        no_success_in_any_observed_cell=int(np.sum(q.max(1)==0)),
        by_execution_repeat=dict(reference=y[:,0].mean(0).tolist(),stay=y[:,extra].mean((0,1)).tolist(),
                                 accel=y[rows,fixed].mean(0).tolist()))


def run(libero,robocasa,output,*,libero_catalog=None):
    libero,robocasa,output=map(Path,(libero,robocasa,output));output.mkdir(parents=True,exist_ok=False)
    source={}
    with np.load(libero/'matrix.npz') as z:
        s=read(libero/'states.json')
        source['libero']=dict(y=z['success'].astype(float),accel=z['accel_members'],
            tasks=np.array([j['task_id'] for j in s]),test=np.array([j['split']=='heldout_initial' for j in s]),
            extra=np.flatnonzero(np.char.startswith(z['candidate_ids'],'broad_heldout')))
    with np.load(robocasa/'matrix.npz') as z:
        p=read(robocasa/'population.json')
        source['robocasa']=dict(y=z['success'].astype(float),accel=z['accel_prefixes'][...,1],
            tasks=np.array([j['task'] for j in p['jobs']]),test=np.array([j['split']=='target' for j in p['jobs']]),
            extra=np.array([i for i,v in enumerate(p['views']) if v['role']=='extra_eval']),
            parameters=[v['parameters'] for v in p['views']])
    if libero_catalog is not None:
        catalog=read(libero_catalog)
        poses={v['pose_id']:[v['azimuth_deg'],v['elevation_deg'],v['radius_scale']]
               for v in catalog['broad_training_64']+catalog['broad_heldout_32']}
        poses['canonical']=[0,0,1]
        with np.load(libero/'matrix.npz') as z:
            source['libero']['parameters']=[poses[v] for v in z['candidate_ids']]
    result=dict(scope='Exploratory read-only cross-benchmark comparison; not a single-factor causal experiment',
        noise_assumption='Independent repeat streams conditional on initial/view; renderer temporal dependence not separately identified',
        subset_ranges='256 subsets of existing 32-noise LIBERO bank; not 256 independent experiments',
        input_hashes={str(root/'matrix.npz'):sha(root/'matrix.npz') for root in (libero,robocasa)},benchmarks={})
    result['source_sha256']=sha(__file__)
    if libero_catalog is not None:
        result['input_hashes'][str(libero_catalog)]=sha(libero_catalog)
    for name,s in source.items():
        result['benchmarks'][name]={}
        for scope,mask in (('all',np.ones(len(s['tasks']),bool)),('evaluation',s['test'])):
            row={}
            for m,arm in enumerate(('canonical','broad')):
                row[arm]=model_summary(s['y'][m,mask],s['accel'][m,mask],s['tasks'][mask],s['extra'])
                if scope=='evaluation' and libero_catalog is not None:
                    row[arm]['common_parameter_box']=common_parameter_box(
                        s['y'][m,mask],s['accel'][m,mask],s['extra'],s['parameters'])
                    row[arm]['two_initials_per_task']=balanced_initial_components(s['y'][m,mask],s['tasks'][mask])
                if name=='libero':row[arm]['downsample4']=noise_harmonization(s['y'][m,mask],s['accel'][m,mask],s['extra'])
            paired=(s['y'][1,mask][:,s['extra']]-s['y'][0,mask][:,s['extra']]).mean((1,2))
            row['multiview_gain_extra']=interval(paired,s['tasks'][mask])
            row['multiview_gain_extra_by_noise']=(s['y'][1,mask][:,s['extra']]-s['y'][0,mask][:,s['extra']]).mean((0,1)).tolist()
            result['benchmarks'][name][scope]=row
    write(output/'summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--libero',required=True,type=Path);p.add_argument('--robocasa',required=True,type=Path)
    p.add_argument('--libero-catalog',type=Path,help='Enable common-parameter and initial-count sensitivity checks')
    p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    result=run(args.libero,args.robocasa,args.output,libero_catalog=args.libero_catalog)
    from ..plotting.cross_benchmark_signal import plot
    plot(result,args.output/'comparison.png')


if __name__=='__main__':main()
