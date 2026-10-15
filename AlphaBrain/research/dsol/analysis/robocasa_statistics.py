"""Finite-matrix statistics and deployment-permission offline P0 selectors.

No heldout outcome is passed to preprocessing or hyperparameter selection.
All method outcomes average repeated execution before aggregating initials.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import csv
import gzip
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.stats import rankdata

from ..acquisition.discrete_search import camera_graph, search
from ..selectors.query_ranker import QueryRanker
from ..data.artifacts import read, write, sha
from .rig_geometry import audit_mounts, camera_opencv


def interval(values, tasks, *, repeats=4000, seed=20260923):
    """Equal-task paired hierarchical bootstrap; finite noise averages fixed."""
    values, tasks = np.asarray(values, float), np.asarray(tasks)
    if values.ndim != 1 or values.shape != tasks.shape or not np.isfinite(values).all():
        raise ValueError('Expected finite per-initial scalar outcomes with task labels')
    unique = np.unique(tasks)
    groups = [np.flatnonzero(tasks == task) for task in unique]
    rng = np.random.default_rng(seed)
    selected = rng.integers(len(groups), size=(repeats, len(groups)))
    samples = np.zeros(repeats)
    for position in range(len(groups)):
        for t, rows in enumerate(groups):
            mask = selected[:, position] == t
            draws = rng.choice(rows, size=(int(mask.sum()), len(rows)))
            samples[mask] += values[draws].mean(1) / len(groups)
    return dict(mean=float(np.mean([values[g].mean() for g in groups])),
                ci95=np.quantile(samples, [.025, .975]).tolist(), initials=len(values), tasks=len(unique))


def empirical_hierarchy(q, tasks):
    """Noise must already be averaged: no per-noise choice of winner."""
    q = np.asarray(q)
    if q.ndim != 2:
        raise ValueError('Expected initial x view mean outcomes')
    return dict(canonical=float(q[:, 0].mean()), uniform=float(q.mean()),
                global_fixed=float(q.mean(0).max()),
                task_fixed=float(np.mean([q[np.asarray(tasks)==t].mean(0).max() for t in np.unique(tasks)])),
                initial_oracle=float(q.max(1).mean()))


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def layout_interval(values, tasks, layouts, *, repeats=4000, seed=20260923):
    """Sensitivity: resample shared layout clusters, keep the task set fixed.

    This complements (does not replace) the predeclared task/initial interval.
    A layout can occur in several tasks. Its multiplicity is shared across
    those tasks; draws omitting a task are excluded and counted explicitly.
    """
    values, tasks, layouts = map(np.asarray, (values, tasks, layouts))
    unique, inverse = np.unique(layouts, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.choice(unique, size=(repeats, len(unique)))
    counts = (draws[:, :, None] == unique[None, None, :]).sum(1)
    weights = counts[:, inverse]
    means = []
    valid = np.ones(repeats, dtype=bool)
    for task in np.unique(tasks):
        mask = tasks == task
        denominator = weights[:, mask].sum(1)
        valid &= denominator > 0
        means.append((weights[:, mask] @ values[mask]) / np.maximum(denominator, 1))
    samples = np.mean(means, axis=0)[valid]
    if not len(samples):
        raise ValueError('No complete-task layout bootstrap draw')
    return dict(mean=float(np.mean([values[tasks == t].mean() for t in np.unique(tasks)])),
                ci95=np.quantile(samples, [.025, .975]).tolist(),
                layouts=len(unique), valid_draws=len(samples), requested_draws=repeats,
                scope='Shared layout clusters; fixed observed task set; supplementary')


def pca_eigh(x, dimension):
    """Full training-only covariance PCA, same subspace as SVD (sign arbitrary)."""
    x = np.asarray(x, float).reshape(-1, x.shape[-1])
    mean = x.mean(0)
    centered = x-mean
    _, vectors = np.linalg.eigh(centered.T @ centered)
    return mean, vectors[:, -dimension:][:, ::-1].T


def ranker_design(image, wrist, tasks, acc, train, config, *, use_accel):
    im, ic = pca_eigh(image[train], config['image_pca_dim'])
    wm, wc = pca_eigh(wrist[train], config['wrist_pca_dim'])
    model = QueryRanker(im, ic, wm, wc, use_accel, num_tasks=len(np.unique(tasks)))
    x = model.features(image[train], wrist[train], np.broadcast_to(tasks[train, None], acc[train].shape[:2]), acc[train])
    x = (x-x.mean(1, keepdims=True)).reshape(-1, x.shape[-1])
    model.scale = x.std(0)
    model.scale[model.scale < 1e-8] = 1.
    return model, x/model.scale


def fit_heldout_ranker(image, wrist, tasks, acc, q, splits, layouts, cfg, *, use_accel):
    train = np.flatnonzero(splits == 'pretrain')
    if len(train) == 0 or np.any(splits[train] != 'pretrain'):
        raise ValueError('Only pretrain scenes may enter fitting')
    alphas = cfg['alpha_grid']
    cv_scores = {a: [] for a in alphas}
    # Leave-one-layout-out avoids putting another initial of the validation
    # layout into its fitting fold. All target scenes remain untouched.
    for layout in np.unique(layouts[train]):
        tr = train[layouts[train] != layout]
        va = train[layouts[train] == layout]
        model, x = ranker_design(image, wrist, tasks, acc, tr, cfg, use_accel=use_accel)
        y = (q[tr]-q[tr].mean(1, keepdims=True)).reshape(-1)
        xtx, xty = x.T @ x, x.T @ y
        feature = model.features(image[va], wrist[va], np.broadcast_to(tasks[va, None], q[va].shape), acc[va]) / model.scale
        for alpha in alphas:
            model.weights = np.linalg.solve(xtx+alpha*np.eye(len(xty)), xty)
            choice = np.argmax(feature @ model.weights, axis=1)
            cv_scores[alpha].extend(zip(tasks[va].tolist(), q[va, choice].tolist()))
    records = []
    for alpha in alphas:
        rows = cv_scores[alpha]
        value = np.mean([np.mean([v for t,v in rows if t == task]) for task in np.unique(tasks)])
        records.append(dict(alpha=alpha, development_oof_success=float(value)))
    selected = max(records, key=lambda r: (r['development_oof_success'], r['alpha']))['alpha']
    model, x = ranker_design(image, wrist, tasks, acc, train, cfg, use_accel=use_accel)
    y = (q[train]-q[train].mean(1, keepdims=True)).reshape(-1)
    model.weights = np.linalg.solve(x.T @ x+selected*np.eye(x.shape[1]), x.T @ y)
    scores = model.score(image, wrist, np.broadcast_to(tasks[:, None], q.shape), acc)
    return scores, model, dict(alpha=selected, cv=records, fitting_initials=len(train),
                               validation='pretrain leave-one-layout-out; no target labels/features in fitting',
                               test='target layout/style holdout; tasks known; catalog available')


def _search_initial(args):
    i, poses, scores, cfg, starts = args
    edges, distance, translation, rotation = camera_graph(poses, neighbors=cfg['neighbors'],
        translation_scale=cfg['translation_scale_m'], rotation_scale=cfg['rotation_scale_deg'])
    results = {}
    for name, value in scores.items():
        for mode in ('local', 'random'):
            repeats = 1 if mode == 'local' else cfg['random_repeats']
            choices = np.zeros((len(starts), repeats, len(cfg['budgets'])), dtype=np.int16)
            travel = np.zeros_like(choices, dtype=float)
            for s, start in enumerate(starts):
                for r in range(repeats):
                    route = search(edges, distance, int(start), cfg['budgets'], lambda v: value[v],
                                   random_seed=None if mode=='local' else cfg['random_seed']+r)
                    for b, budget in enumerate(cfg['budgets']):
                        choices[s, r, b] = route[budget]['selected']
                        path = route[budget]['path']
                        travel[s, r, b] = sum(translation[a,c] for a,c in zip(path[:-1],path[1:]))
            results[mode+'_'+name] = (choices, travel)
    return i, results


def summarize(config):
    cfg = read(config)
    out = Path(cfg['output'])
    audit = read(out / 'audit.json')
    if audit['status'] != 'PASS' or any(sha(out/name) != h for name,h in audit['artifacts'].items()):
        raise ValueError('An unchanged admitted matrix is required')
    population = read(out / 'population.json')
    jobs, views = population['jobs'], population['views']
    splits = np.array([j['split'] for j in jobs])
    task_names = sorted({j['task'] for j in jobs})
    tasks = np.array([task_names.index(j['task']) for j in jobs])
    layouts = np.array([j['layout_id'] for j in jobs])
    with np.load(out/'matrix.npz') as z:
        success = z['success']
        all_acc = z['accel_prefixes']
    q, acc = success.mean(-1), all_acc[..., cfg['primary_accel_prefix']-2]
    params = np.array([v['parameters'] for v in views])
    train_views = np.array([i for i,v in enumerate(views) if v['role']=='train'])
    extra = np.array([i for i,v in enumerate(views) if v['role']=='extra_eval'])
    features = out/'features'
    if not (features/'complete.json').exists():
        raise ValueError('Frozen RGB encoding not complete')
    image, wrist = [], []
    for j in jobs:
        file = features/(j['initial_id']+'.npz')
        if sha(file) != read(file.with_suffix('.json'))['sha256']:
            raise ValueError('Feature content changed')
        with np.load(file) as z:
            image.append(z['exterior']); wrist.append(z['wrist'])
    image, wrist = np.stack(image, axis=1), np.stack(wrist, axis=1)
    models, method_arrays, score_arrays = {}, {}, {}
    learner_dir = out/'learners'
    learner_dir.mkdir(exist_ok=True)
    for m, name in enumerate(('canonical', 'broad')):
        scores = {'accel':acc[m].mean(-1)}
        learners = {}
        for use_accel, method in [(False,'rgb'), (True,'rgb_accel')]:
            print(json.dumps(dict(stage='fit',model=name,method=method)), flush=True)
            prediction, model, details = fit_heldout_ranker(image[m],wrist[m],tasks,acc[m],q[m],splits,layouts,cfg,use_accel=use_accel)
            scores[method] = prediction
            learners[method] = details
            np.savez_compressed(learner_dir/(name+'_'+method+'.npz'),
                                **{k:v for k,v in vars(model).items() if isinstance(v,np.ndarray)})
        score_arrays[name] = scores
        rows = np.arange(len(jobs))
        methods = {'canonical':q[m,:,0], 'stay_extra32':q[m][:,extra].mean(1),
                   'stay_noncanonical96':q[m,:,1:].mean(1), 'uniform97':q[m].mean(1),
                   'min_accel':q[m,rows,np.argmin(scores['accel'],1)],
                   'single_accel':q[m,rows[:,None],np.argmin(acc[m],axis=1)].mean(1),
                   'top10_uniform':np.take_along_axis(q[m],np.argsort(scores['accel'],axis=1,kind='stable')[:,:10],axis=1).mean(1),
                   'rgb':q[m,rows,np.argmin(scores['rgb'],1)],
                   'rgb_accel':q[m,rows,np.argmin(scores['rgb_accel'],1)],
                   'empirical_oracle':q[m].max(1)}
        dev = splits == 'pretrain'
        global_choice = int(np.argmax(q[m,dev].mean(0)))
        task_choices = {t:int(np.argmax(q[m,dev & (tasks==t)].mean(0))) for t in np.unique(tasks)}
        methods['seen_global_fixed'] = q[m,:,global_choice]
        methods['seen_task_fixed'] = q[m,rows,[task_choices[t] for t in tasks]]
        method_arrays[name] = methods
        model_result = dict(learners=learners, global_choice=views[global_choice]['view_id'],
                            task_choices={task_names[t]:views[v]['view_id'] for t,v in task_choices.items()},scopes={})
        for split in ('pretrain','target'):
            ix = np.flatnonzero(splits==split)
            t = tasks[ix]
            coverage = {label: interval(q[m,ix][:,selection].mean(1),t)
                        for label,selection in [('canonical',[0]),('train64',train_views),('extra32',extra),('all97',np.arange(97))]}
            methods_summary = {key:dict(success=interval(value[ix],t),
                                       gain_vs_stay=interval(value[ix]-methods['stay_extra32'][ix],t),
                                       gain_vs_canonical=interval(value[ix]-methods['canonical'][ix],t))
                               for key,value in methods.items()}
            rho, bins, residual_rho, good_view_rows = [], [], [], []
            mean_acc = scores['accel'][ix,1:]
            for index in ix:
                rho.append(correlation(-scores['accel'][index,1:],q[m,index,1:]))
                order = np.argsort(scores['accel'][index,1:],kind='stable')+1
                bins.append([q[m,index,g].mean() for g in np.array_split(order,4)])
                # A descriptive check within each initial, controlling only
                # known geometric offsets. This is not a causal adjustment.
                design=np.column_stack([np.ones(96),np.abs(params[1:,0]),np.abs(params[1:,1]),
                                        np.abs(np.log(params[1:,2]))])
                rx=rankdata(-scores['accel'][index,1:]);ry=rankdata(q[m,index,1:])
                ex=rx-design@np.linalg.lstsq(design,rx,rcond=None)[0]
                ey=ry-design@np.linalg.lstsq(design,ry,rcond=None)[0]
                residual_rho.append(None if np.std(ex)<1e-9 or np.std(ey)<1e-9 else float(np.corrcoef(ex,ey)[0,1]))
                better=np.flatnonzero(q[m,index,1:]>q[m,index,0])+1
                percentile=rankdata(scores['accel'][index,1:])/96
                good_view_rows.append(dict(initial_id=jobs[index]['initial_id'],task=jobs[index]['task'],
                    better_than_reference_count=len(better),
                    mean_accel_percentile=float(percentile[better-1].mean()) if len(better) else None,
                    low_accel_quartile_fraction=float(np.mean(percentile[better-1]<=.25)) if len(better) else None,
                    mean_abs_azimuth=float(np.abs(params[better,0]).mean()) if len(better) else None,
                    mean_abs_elevation=float(np.abs(params[better,1]).mean()) if len(better) else None,
                    mean_radius=float(params[better,2].mean()) if len(better) else None))
            finite = [r for r in rho if r is not None]
            winners = np.argmin(acc[m,ix],axis=1)
            vote = np.mean(winners==np.argmin(scores['accel'][ix],axis=1)[:,None],axis=1)
            # Diagnostic only: choose on two execution draws and evaluate on
            # the other two; no refitting of the actual metric/learned method.
            crossfit = []
            for left,right in [([0,1],[2,3]),([2,3],[0,1])]:
                pick = success[m,ix][:,:,left].mean(-1).argmax(1)
                crossfit.append(success[m,ix][:,:,right].mean(-1)[np.arange(len(ix)),pick])
            task_rows = []
            for task in np.unique(t):
                sub = ix[t==task]
                task_rows.append(dict(task=task_names[task],initials=len(sub),
                    methods={k:float(v[sub].mean()) for k,v in methods.items()},
                    accel_mean=float(scores['accel'][sub,1:].mean()),
                    noncanonical_variance=float(q[m,sub,1:].var(axis=1).mean())))
            model_result['scopes'][split] = dict(coverage=coverage,methods=methods_summary,
                hierarchy=empirical_hierarchy(q[m,ix],t),tasks=task_rows,
                accel_mean=float(mean_acc.mean()),accel_score_std_mean=float(acc[m,ix,1:].std(-1).mean()),
                quartiles=[interval(np.asarray(bins)[:,b],t) for b in range(4)],
                spearman_median=float(np.median(finite)) if finite else None,
                spearman_defined=len(finite),spearman_undefined=len(rho)-len(finite),
                spearman_per_initial=rho,noise_winner_agreement=float(vote.mean()),
                geometry_residual_correlation_median=float(np.median([r for r in residual_rho if r is not None])) if any(r is not None for r in residual_rho) else None,
                better_view_diagnostics=good_view_rows,
                twofold_execution_selection=interval(np.mean(crossfit,axis=0),t),
                oracle_caveat='Empirical maximum of four-noise means; optimistic selection; twofold diagnostic is not deployed method or new rollout',
                scoring_caveat='Initial-observation normalized action-flow Accel prefix3; not whole-episode physical acceleration')
            keep = np.array([i for i in ix if jobs[i]['task'] != 'NavigateKitchen'])
            model_result['scopes'][split]['without_navigation'] = {
                k:dict(success=interval(v[keep], tasks[keep]),
                       gain_vs_stay=interval(v[keep]-methods['stay_extra32'][keep], tasks[keep]))
                for k,v in methods.items()}
            model_result['scopes'][split]['layout_cluster_sensitivity'] = {
                k:layout_interval(methods[k][ix]-methods['stay_extra32'][ix],tasks[ix],layouts[ix])
                for k in ('min_accel','rgb','rgb_accel')}
            # Exploratory diagnostics only: the primary prefix remains 3.
            # Do not choose a new method by the best target entry in this table.
            model_result['scopes'][split]['prefix_sensitivity'] = [
                dict(prefix=p+2, success=float(q[m,ix,np.argmin(all_acc[m,ix,:,:,p].mean(-1),axis=1)].mean()))
                for p in range(all_acc.shape[-1])]
        models[name]=model_result
    np.savez_compressed(out/'method-arrays.npz',**{m+'__'+k:v for m,r in method_arrays.items() for k,v in r.items()},
                        **{m+'__score_'+k:v for m,r in score_arrays.items() for k,v in r.items()})
    result = dict(models=models,task_names=task_names,config=cfg,scopes={},
                  limitations=['Finite execution noise: four per cell', 'No target segmentation visibility data',
                    'Known tasks with heldout layout AND style, not unseen tasks', 'No physical camera movement during acquisition',
                    'One posttraining seed per arm', 'All97 catalog is not a physically certified reachable space',
                    'Seen learned-method scores are in-sample; heldout target is evaluation',
                    'Bootstrap fixes observed noise averages; not a correction of maximum selection bias'])
    for split in ('pretrain','target'):
        ix=np.flatnonzero(splits==split)
        result['scopes'][split]=dict(training_gain={k:interval(method_arrays['broad'][k][ix]-method_arrays['canonical'][k][ix],tasks[ix])
                                                   for k in ('canonical','stay_extra32','stay_noncanonical96')},
            accel_change=interval(acc[1,ix,1:].mean((1,2))-acc[0,ix,1:].mean((1,2)),tasks[ix]))
        result['scopes'][split]['training_gain_layout_sensitivity'] = {
            k:layout_interval(method_arrays['broad'][k][ix]-method_arrays['canonical'][k][ix],tasks[ix],layouts[ix])
            for k in ('canonical','stay_extra32')}
    write(out/'summary.json',result)
    with (out/'per-task.csv').open('w',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(['model','split','task',*method_arrays['canonical'].keys()])
        for model,item in models.items():
            for split,scope in item['scopes'].items():
                for row in scope['tasks']:
                    writer.writerow([model,split,row['task'],*[row['methods'][k] for k in method_arrays['canonical']]])
    with (out/'per-view.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['model','split','task','view_id','role','azimuth_deg','elevation_deg','radius_scale','success','accel_mean','score_noise_std'])
        for m,model in enumerate(('canonical','broad')):
            for split in ('pretrain','target'):
                for task in [None,*range(len(task_names))]:
                    ix=np.flatnonzero((splits==split) & (True if task is None else tasks==task))
                    for v,view in enumerate(views):
                        writer.writerow([model,split,'ALL' if task is None else task_names[task],view['view_id'],view['role'],
                            *view['parameters'],q[m,ix,v].mean(),acc[m,ix,v].mean(),acc[m,ix,v].std(-1).mean()])
    with (out/'per-initial-methods.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['model','split','task','initial_id','layout_id',*method_arrays['canonical'].keys()])
        for model,methods in method_arrays.items():
            for i,j in enumerate(jobs):
                writer.writerow([model,j['split'],j['task'],j['initial_id'],j['layout_id'],*[value[i] for value in methods.values()]])
    print(json.dumps(dict(stage='summary_complete',output=str(out))),flush=True)


def replay(config):
    cfg=read(config);out=Path(cfg['output']);population=read(out/'population.json')
    jobs,views=population['jobs'],population['views']
    plan=read(cfg['matrix_plan']); release=read(plan['reference_release'])
    released={r['initial_id']:r for r in release['records']}
    poses=[];params=np.array([v['parameters'] for v in views])
    for job in jobs:
        record=released[job['initial_id']];root=Path(record['candidate_root'])
        summary=read(root/'summary.json')
        if sha(root/'summary.json')!=record['candidate_summary_sha256'] or sha(root/'geometry.npz')!=summary['files']['geometry.npz']['sha256']:
            raise ValueError('Geometry identity changed')
        xml=ET.fromstring(gzip.decompress((Path(job['path'])/'compiled_model.xml.gz').read_bytes()))
        names=[c.attrib['name'] for c in xml.find('worldbody').iter('camera')]
        ids=[names.index(n) for n in ('robot0_agentview_left','robot0_agentview_right')]
        with np.load(root/'geometry.npz') as g:
            rig=audit_mounts(g['camera_positions'],g['camera_quaternions'],ids,params)
        poses.append(np.stack([camera_opencv(p,r) for p,r in zip(rig['midpoint'],rig['rotations'][:,0])]))
    starts=np.array([i for i,v in enumerate(views) if v['role']=='extra_eval'])
    with np.load(out/'matrix.npz') as z:q=z['success'].mean(-1)
    with np.load(out/'method-arrays.npz') as z:score_data={k:z[k] for k in z.files if '__score_' in k}
    target=np.array([j['split']=='target' for j in jobs]);indices=np.flatnonzero(target)
    task_names=sorted({j['task'] for j in jobs});tasks=np.array([task_names.index(j['task']) for j in jobs])
    output=out/'search';output.mkdir(exist_ok=True)
    report={}
    for m,model in enumerate(('canonical','broad')):
        arguments=[(int(i),poses[i],{name:score_data[model+'__score_'+name][i] for name in ('accel','rgb','rgb_accel')},cfg,starts) for i in indices]
        values,travel={},{}
        with ProcessPoolExecutor(max_workers=12) as pool:
            futures={pool.submit(_search_initial,arg):arg[0] for arg in arguments}
            done=0
            for future in as_completed(futures):
                i,res=future.result();position=int(np.flatnonzero(indices==i)[0])
                for name,(choices,distances) in res.items():
                    if name not in values:
                        values[name]=np.zeros((len(indices),len(cfg['budgets'])))
                        travel[name]=np.zeros_like(values[name])
                    values[name][position]=q[m,i,choices].mean((0,1))
                    travel[name][position]=distances.mean((0,1))
                np.savez_compressed(output/(model+'_'+jobs[i]['initial_id']+'.npz'),**{k:v[0] for k,v in res.items()})
                done+=1
                if done%20==0:print(json.dumps(dict(stage='replay',model=model,done=done,total=len(indices))),flush=True)
        report[model]={}
        for name,value in values.items():
            report[model][name]={}
            baseline=q[m,indices][:,starts].mean(1)
            for b,budget in enumerate(cfg['budgets']):
                entry=dict(success=interval(value[:,b],tasks[indices]),gain_vs_stay=interval(value[:,b]-baseline,tasks[indices]),
                           mean_catalog_path_m=float(travel[name][:,b].mean()))
                if name.startswith('local_'):
                    entry['gain_vs_random']=interval(value[:,b]-values[name.replace('local_','random_',1)][:,b],tasks[indices])
                report[model][name][str(budget)]=entry
        np.savez_compressed(output/(model+'_outcomes.npz'),**values)
    write(out/'search-summary.json',dict(models=report,initials=len(indices),starts_per_initial=len(starts),
        scope='target heldout layouts/styles only; frozen within-episode world, cached endpoint outcomes; not physical active acquisition',
        random_repeats=cfg['random_repeats'],budgets=cfg['budgets'],same_score_budget_per_family=True,
        costs='Unique observed nodes; Accel families require eight scored flows/node, RGB requires no policy scoring. Catalog path length is not verified physical travel.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--stage',choices=['summarize','replay'],required=True)
    a=p.parse_args();(summarize if a.stage=='summarize' else replay)(a.config)
