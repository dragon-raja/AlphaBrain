"""Offline attribution of frozen selectors; no encoder, fit, or policy calls."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json
import numpy as np
from ..data.artifacts import read, write, sha
from .statistics import paired_interval


def select_visited(scores, visited):
    """Retain the first visited node on score ties, matching historical search."""
    scores, visited = np.asarray(scores), np.asarray(visited)
    if not np.isfinite(scores).all() or visited.ndim < 1:
        raise ValueError('Finite scores and ordered visits required')
    return np.take_along_axis(visited, np.argmin(scores[visited], axis=-1)[..., None], -1)[..., 0]


def gap_decomposition(labels, visited, selected):
    """Empirical max gaps and two-way half-bank cross evaluation.

    Empirical gaps are optimistic diagnostics, not attainable policy returns.
    Cross-evaluated gaps may be negative and must never be clipped.
    """
    y = np.asarray(labels, dtype=float)
    visited, selected = np.asarray(visited), np.asarray(selected)
    if y.ndim != 2 or not np.isfinite(y).all() or y.shape[-1] % 2:
        raise ValueError('Complete candidate x even-repeat label bank required')
    q = y.mean(-1)
    current = q[selected]
    best_visited = q[visited].max(-1)
    best_all = float(q.max())
    result = dict(actual=current, empirical_best_visited=best_visited,
                  empirical_search_gap=best_all-best_visited,
                  empirical_ranking_gap=best_visited-current)
    half = y.shape[-1]//2
    cv = defaultdict(list)
    for train, test in ((y[:, :half].mean(-1), y[:, half:].mean(-1)),
                        (y[:, half:].mean(-1), y[:, :half].mean(-1))):
        global_id = int(train.argmax())
        local_id = select_visited(-train, visited)
        cv['cross_best_all'].append(np.full(selected.shape, test[global_id]))
        cv['cross_best_visited'].append(test[local_id])
        cv['cross_search_gap'].append(test[global_id]-test[local_id])
        cv['cross_ranking_gap'].append(test[local_id]-test[selected])
    result.update({k: np.mean(v, axis=0) for k,v in cv.items()})
    if not np.allclose(result['cross_search_gap']+result['cross_ranking_gap'],
                       result['cross_best_all']-current):
        raise AssertionError('Cross evaluation identity failed')
    return result


def split_signs(labels, selected, baseline):
    y = np.asarray(labels, dtype=float)
    if not np.isfinite(y).all() or y.shape[-1] % 2:
        raise ValueError('Missing or odd-size label bank')
    h = y.shape[-1]//2
    d = y[selected]-y[baseline]
    a,b = d[..., :h].mean(-1), d[..., h:].mean(-1)
    return dict(positive_both=(a>0)&(b>0), negative_both=(a<0)&(b<0),
                strict_reversal=(a*b<0), equal_both=(a==0)&(b==0), first=a, second=b)


def run(config_path):
    cfg = read(config_path)
    out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=False)
    write(out/'frozen-plan.json', cfg)
    sources = {}
    def checked(path, digest=None):
        p=Path(path); h=sha(p)
        if digest is not None and digest != h: raise ValueError('Input identity: '+str(p))
        sources[str(p)]=h
        return p
    def restored(path):
        value=str(path)
        for old,new in cfg['path_rewrites'].items():
            if value.startswith(old): return Path(new+value[len(old):])
        return Path(value)
    root,src,rec = (Path(cfg[k]) for k in ('v2','matrix','recovery'))
    old=read(checked(root/'final-analysis/analysis.json'))
    state_rows=old['checkpoint_results'][0]['state_metrics']
    methods=('geometry_context_ridge','image_queryable_ridge','statewise_P_top1')
    wanted={(r['pair_key'],r['candidate_id']) for r in state_rows if r['method'] in methods or r['method']=='canonical'}
    banks={}; coverage={}
    for bank,dirs in [('O',sorted((root/'test/O').glob('wave-*'))),('P',[root/'test/P']),('Q',[root/'test/Q-seed41'])]:
        vals=defaultdict(dict);n=0;seen=set()
        for directory in dirs:
            for p in sorted(directory.glob('episodes-shard-*.jsonl')):
                checked(p)
                for line in p.open():
                    row=json.loads(line); eid=row['episode_id']
                    if eid in seen: raise ValueError('Duplicate episode '+eid)
                    seen.add(eid);n+=1
                    key=(row['pair_key'],row['condition'].removeprefix('candidate__'))
                    if key not in wanted: continue
                    repeat=int(row['policy_repeat_id'])
                    if repeat in vals[key]: raise ValueError('Duplicate label')
                    vals[key][repeat]=int(row['success'])
        repeats=64 if bank=='Q' else 32
        for key,v in vals.items():
            if sorted(v)!=list(range(repeats)): raise ValueError('Incomplete bank '+str(key))
        banks[bank]={key:np.array([v[i] for i in range(repeats)]) for key,v in vals.items()}
        coverage[bank]=dict(raw_episodes=n,selected_candidate_state_pairs=len(vals),repeats=repeats)
        print('Loaded historical bank',bank,n,flush=True)
    v2rows=[]
    for r in state_rows:
        if r['method'] not in methods: continue
        record={k:r[k] for k in ('method','pair_key','candidate_id','task_id')}
        for bank in banks:
            a=banks[bank].get((r['pair_key'],r['candidate_id']))
            b=banks[bank].get((r['pair_key'],'canonical'))
            record[bank]=None if a is None or b is None else dict(success=float(a.mean()),baseline=float(b.mean()),gain=float((a-b).mean()))
        if record['Q'] is None or record['Q']['success']!=r['success']: raise ValueError('Q golden mismatch')
        v2rows.append(record)
    write(out/'v2-labels.json',dict(coverage=coverage,rows=v2rows))

    provenance=read(checked(src/'provenance.json'))
    matrix=checked(src/'matrix.npz',provenance['artifacts']['matrix.npz'])
    states=read(checked(src/'states.json',provenance['artifacts']['states.json']))
    summary=read(checked(rec/'summary.json'));receipt=read(checked(rec/'provenance.json'))
    with np.load(matrix,allow_pickle=False) as z:
        labels=z['success'].copy();ids=z['candidate_ids'].tolist()
    if labels.shape!=(2,32,97,32) or not np.isin(labels,[0,1]).all():raise ValueError('Unexpected outcome matrix')
    ev=[i for i,s in enumerate(states) if s['initialization']['init_state_index']>=2]
    if [states[i]['task_ordinal'] for i in ev]!=list(np.repeat(np.arange(8),2)):raise ValueError('Unexpected test ordering')
    visibility=[];assets=[]
    for i in ev:
        p=checked(restored(states[i]['initial_asset_record']))
        asset=read(p);byid={x['candidate_id']:x for x in asset['visibility']}
        vis={}
        for camera in ('agentview','robot0_eye_in_hand'):
            vis[camera]=np.array([min(e['visible_pixels'] for e in byid[c]['per_camera'][camera]['entities'].values()) for c in ids])
        visibility.append(vis);assets.append(dict(record=str(p),images=str(p.parent/'policy_inputs.npz')))
    rows=[];cross=[];events=[];validation=[]
    methods2=('local_accel','random_accel','local_rgb','local_rgb_accel','random_rgb_accel')
    budgets=cfg['budgets']
    for mi,model in enumerate(('canonical','broad')):
        scorepath=checked(rec/f'{model}-candidate-scores.npz',receipt['artifacts'][f'{model}-candidate-scores.npz'])
        with np.load(scorepath,allow_pickle=False) as z:scores={k:z[k].copy() for k in z.files}
        for method in methods2:
            p=checked(rec/f'{model}-{method}-replay.npz',receipt['artifacts'][f'{model}-{method}-replay.npz'])
            with np.load(p,allow_pickle=False) as z:
                visited=z['visited'][:,64:96].copy();selected=z['selected'][:,64:96].copy();recorded=z['success'][:,64:96].copy()
            family=method.split('_',1)[1]
            for bi,budget in enumerate(budgets):
                metrics=defaultdict(list);swaps=defaultdict(list)
                for ti,i in enumerate(ev):
                    y=labels[mi,i];v=visited[ti,...,:budget];s=selected[ti,...,bi];q=y.mean(-1)
                    if not np.array_equal(select_visited(scores[family][ti],v),s):raise ValueError('Selection golden mismatch')
                    if not np.array_equal(q[s],recorded[ti,...,bi]):raise ValueError('Outcome golden mismatch')
                    if any(len(set(x))!=budget for x in v.reshape(-1,budget)):raise ValueError('Duplicate visit')
                    d=gap_decomposition(y,v,s)
                    starts=np.broadcast_to(np.arange(65,97)[:,None],s.shape)
                    signs=split_signs(y,s,starts)
                    for k,value in d.items():metrics[k].append(float(np.mean(value)))
                    metrics['stay'].append(float(q[starts].mean()))
                    for k in ('positive_both','negative_both','strict_reversal','equal_both'):metrics[k].append(float(signs[k].mean()))
                    metrics['harmed_full'].append(float((q[s]<q[starts]).mean()))
                    minpix=visibility[ti]['agentview'];wrist=visibility[ti]['robot0_eye_in_hand']
                    metrics['selected_all_entities_nonzero'].append(float((minpix[s]>0).mean()))
                    stablebad=signs['negative_both'];stablegood=signs['positive_both']
                    metrics['stable_harm_both_views_nonzero'].append(float((stablebad&(minpix[s]>0)&(wrist[s]>0)).mean()))
                    metrics['stable_gain_external_missing'].append(float((stablegood&(minpix[s]==0)).mean()))
                    for f in ('accel','rgb','rgb_accel'):
                        picked=select_visited(scores[f][ti],v)
                        swaps[f].append(float(q[picked].mean()))
                    if budget==cfg['primary_budget'] and method=='local_rgb_accel':
                        best=select_visited(-q,v)
                        for sj in range(32):
                            c=int(s[sj,0]);start=int(starts[sj,0]);bv=int(best[sj,0])
                            events.append(dict(model=model,initial=i,task=states[i]['task_id'],start=start,selected=c,best_visited=bv,
                                start_id=ids[start],selected_id=ids[c],best_visited_id=ids[bv],
                                gain=float(q[c]-q[start]),half_gain=[float(signs[k][sj,0]) for k in ('first','second')],
                                success=[float(q[j]) for j in (start,c,bv)],
                                external_min_pixels=[int(minpix[j]) for j in (start,c,bv)],wrist_min_pixels=[int(wrist[j]) for j in (start,c,bv)],
                                stable_harm=bool(stablebad[sj,0]),stable_gain=bool(stablegood[sj,0]),asset=assets[ti]))
                aggregate={k:paired_interval(np.array(v),np.arange(16)) for k,v in metrics.items()}
                golden=summary['models'][model]['methods'][method][str(budget)]['heldout32']['success']['mean']
                if not np.isclose(aggregate['actual']['mean'],golden,rtol=0,atol=1e-14):raise ValueError('Aggregate golden mismatch')
                rows.append(dict(model=model,method=method,budget=budget,metrics=aggregate))
                for f,v in swaps.items():cross.append(dict(model=model,search=method,scorer=f,budget=budget,success=paired_interval(np.array(v),np.arange(16))))
                validation.append([model,method,budget,True])
        print('Replayed frozen selections',model,flush=True)
    write(out/'recovery-attribution.json',dict(rows=rows,cross_scoring=cross,events=events,validation=validation))
    write(out/'sources.json',sources)
    for p,h in sources.items():
        if sha(p)!=h: raise ValueError('Historical source mutated: '+p)
    write(out/'receipt.json',dict(status='COMPLETE_OFFLINE_ATTRIBUTION',new_model_calls=0,new_training=0,new_actions=0,
        source_count=len(sources),original_sources_unchanged=True,exact_replay_groups=len(validation),
        caveats=cfg['caveats']))
    return out


def run_robocasa(config_path):
    """Reconstruct historical query paths from cached scores, verify endpoints."""
    import gzip
    import xml.etree.ElementTree as ET
    from ..acquisition.discrete_search import camera_graph, search
    from .rig_geometry import audit_mounts, camera_opencv
    from .robocasa_statistics import interval
    cfg=read(config_path);root=Path(cfg['robocasa']);out=Path(cfg['output'])/'robocasa'
    out.mkdir(exist_ok=False);write(out/'frozen-plan.json',cfg)
    sources={}
    def checked(p,digest=None):
        p=Path(p);h=sha(p)
        if digest is not None and h!=digest:raise ValueError('Source hash '+str(p))
        sources[str(p)]=h;return p
    def restored(p):
        p=str(p)
        for old,new in cfg['path_rewrites'].items():
            if p.startswith(old):return Path(new+p[len(old):])
        return Path(p)
    summary=read(checked(root/'summary.json'));population=read(checked(root/'population.json'))
    jobs,views=population['jobs'],population['views'];audit=read(checked(root/'audit.json'))
    with np.load(checked(root/'matrix.npz',audit['artifacts']['matrix.npz'])) as z:y=z['success'].copy()
    with np.load(checked(root/'method-arrays.npz')) as z:scores={k:z[k].copy() for k in z.files if '__score_' in k}
    for p in sorted((root/'learners').glob('*.npz')):checked(p)
    if y.shape!=(2,400,97,4) or not np.isin(y,[0,1]).all():raise ValueError('Incomplete RoboCasa labels')
    search_summary=read(checked(root/'search-summary.json'))
    plan=read(checked(restored(cfg['robocasa_matrix_plan'])))
    release=read(checked(restored(plan['reference_release'])))
    released={r['initial_id']:r for r in release['records']}
    indices=[i for i,j in enumerate(jobs) if j['split']=='target']
    task_names=sorted({j['task'] for j in jobs});tasks=np.array([task_names.index(jobs[i]['task']) for i in indices])
    params=np.array([v['parameters'] for v in views]);starts=np.array([i for i,v in enumerate(views) if v['role']=='extra_eval'])
    results=defaultdict(lambda:defaultdict(list));cross=defaultdict(list);exact=0;full=[]
    for ti,i in enumerate(indices):
        job=jobs[i];record=released[job['initial_id']];candidate=restored(record['candidate_root'])
        meta=read(checked(candidate/'summary.json',record['candidate_summary_sha256']))
        geometry=checked(candidate/'geometry.npz',meta['files']['geometry.npz']['sha256'])
        xmlpath=checked(restored(job['path'])/'compiled_model.xml.gz')
        xml=ET.fromstring(gzip.decompress(xmlpath.read_bytes()))
        names=[c.attrib['name'] for c in xml.find('worldbody').iter('camera')]
        camera_ids=[names.index(n) for n in ('robot0_agentview_left','robot0_agentview_right')]
        with np.load(geometry) as g:rig=audit_mounts(g['camera_positions'],g['camera_quaternions'],camera_ids,params)
        poses=np.stack([camera_opencv(p,r) for p,r in zip(rig['midpoint'],rig['rotations'][:,0])])
        edges,distance,_,_=camera_graph(poses,6,.1,15.)
        for mi,model in enumerate(('canonical','broad')):
            label=y[mi,i].astype(float);q=label.mean(-1);score={f:scores[model+'__score_'+f][i] for f in ('accel','rgb','rgb_accel')}
            oldpath=checked(root/'search'/f'{model}_{job["initial_id"]}.npz')
            with np.load(oldpath) as z:old={k:z[k].copy() for k in z.files}
            for family in ('accel','rgb','rgb_accel'):
                full.append(dict(model=model,initial=job['initial_id'],task=job['task'],family=family,
                    selected=int(score[family].argmin()),success=float(q[score[family].argmin()]),canonical=float(q[0]),
                    half_gain=[float((label[score[family].argmin(),h:h+2]-label[0,h:h+2]).mean()) for h in (0,2)]))
                traces=[search(edges,distance,int(s),cfg['budgets'],lambda node:score[family][node]) for s in starts]
                for bi,budget in enumerate(cfg['budgets']):
                    v=np.array([t[budget]['visited'] for t in traces]);s=np.array([t[budget]['selected'] for t in traces])
                    if not np.array_equal(s,old['local_'+family][:,0,bi]):raise ValueError('RoboCasa query reconstruction mismatch')
                    exact+=1;key=(model,family,budget);d=gap_decomposition(label,v,s);signs=split_signs(label,s,starts)
                    for k,value in d.items():results[key][k].append(float(np.mean(value)))
                    results[key]['stay'].append(float(q[starts].mean()))
                    for k in ('positive_both','negative_both','strict_reversal','equal_both'):results[key][k].append(float(signs[k].mean()))
                    for f in score:cross[(model,family,budget,f)].append(float(q[select_visited(score[f],v)].mean()))
        if (ti+1)%20==0:print('RoboCasa cached query reconstruction',ti+1,'/200',flush=True)
    rows=[]
    for (model,family,budget),metrics in results.items():
        agg={k:interval(np.array(v),tasks) for k,v in metrics.items()}
        golden=search_summary['models'][model]['local_'+family][str(budget)]['success']['mean']
        if not np.isclose(agg['actual']['mean'],golden,atol=1e-14,rtol=0):raise ValueError('RoboCasa aggregate mismatch')
        rows.append(dict(model=model,method='local_'+family,budget=budget,metrics=agg))
    write(out/'attribution.json',dict(rows=rows,cross_scoring=[dict(model=m,search='local_'+f,budget=b,scorer=s,success=interval(np.array(v),tasks)) for (m,f,b,s),v in cross.items()],full_query=full,
        visibility='Not collected in original released cache; no segmentation/semantic labels manufactured',
        scope='Historical pi0.5 canonical/broad P0, not XR-1; 200 target initials, known tasks, four execution repeats; 2/2 split is very coarse'))
    write(out/'sources.json',sources)
    for p,h in sources.items():
        if sha(p)!=h:raise ValueError('Source mutated')
    write(out/'receipt.json',dict(status='COMPLETE',new_model_calls=0,new_training=0,new_actions=0,exact_groups=exact,source_count=len(sources)))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--robocasa',action='store_true')
    a=p.parse_args();print((run_robocasa if a.robocasa else run)(a.config))
