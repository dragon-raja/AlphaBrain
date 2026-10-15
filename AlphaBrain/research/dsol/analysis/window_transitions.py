"""Paired, descriptive window-transition audit. Never fits or executes policies."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from ..data.artifacts import read, sha, write
from .window_synthesis import select


def transition(scores, outcomes, views, start, end, exclude=False):
    """Score-only frozen choices, followed by paired endpoint diagnostics.

    start/end are one-based cumulative windows. Execution halves describe
    repeatability, not significance or an independent confirmation.
    """
    s = np.asarray(scores, float)
    y = np.asarray(outcomes, float)
    choices = select(s, exclude)
    a, b = int(choices[start-1]), int(choices[end-1])
    raw_delta = (s[a]-s[b]).mean(0)
    cumulative = np.cumsum(s, axis=-1)/np.arange(1, 6)
    ranks = (rankdata(s[int(exclude):], axis=0)-1)/max(1, len(s)-int(exclude)-1)
    per_noise = cumulative[a, :, end-1]-cumulative[b, :, end-1]
    half_winners = [int(cumulative[int(exclude):, part, end-1].mean(1).argmin())+int(exclude)
                    for part in (slice(0, 4), slice(4, 8))]
    margin = float(per_noise.mean())
    old_disadvantage = float(-raw_delta[:start].sum())
    new_advantage = float(raw_delta[start:end].sum())
    complete = bool(np.isfinite(y[[a, b]]).all())
    delta = y[b]-y[a] if complete else None
    half = y.shape[-1]//2
    halves = [float(delta[:half].mean()), float(delta[half:].mean())] if complete else None
    effect = float(delta.mean()) if complete else None
    return dict(start=start, end=end, old_view=str(views[a]), new_view=str(views[b]),
                switched=a != b, canonical_in=a != 0 and b == 0,
                canonical_out=a == 0 and b != 0, complete=complete,
                old_success=float(y[a].mean()) if np.isfinite(y[a]).all() else None,
                new_success=float(y[b].mean()) if np.isfinite(y[b]).all() else None,
                delta=effect, outcome='missing' if not complete else
                ('improved' if effect > 1e-12 else 'degraded' if effect < -1e-12 else 'equal'),
                execution_half_deltas=halves,
                both_execution_halves_same_direction=bool(halves[0]*halves[1] > 0) if complete else None,
                execution_rescued=int((delta > 0).sum()) if complete else None,
                execution_harmed=int((delta < 0).sum()) if complete else None,
                scoring_noise_support=int((per_noise > 0).sum()),
                scoring_halves_same_winner=half_winners[0] == half_winners[1] == b,
                normalized_margin=margin/(float(cumulative[int(exclude):, :, end-1].mean(1).std())+1e-30),
                stage_old_minus_new=raw_delta.tolist(),
                stage_rank_old_minus_new=(ranks[a-int(exclude)]-ranks[b-int(exclude)]).mean(0).tolist(),
                added_stages_outweigh_initial_disadvantage=new_advantage >= old_disadvantage-1e-12)


def aggregate(rows):
    complete = [r for r in rows if r['complete']]
    tasks = sorted({r['task'] for r in rows})
    per_task = {}
    for task in tasks:
        rs = [r for r in rows if r['task'] == task]
        ok = all(r['complete'] for r in rs)
        per_task[task] = dict(n=len(rs), improved=sum(r['outcome']=='improved' for r in rs),
            degraded=sum(r['outcome']=='degraded' for r in rs),
            delta=float(np.mean([r['delta'] for r in rs])) if ok else None)
    groups = {}
    for label in ('improved', 'degraded', 'equal'):
        rs = [r for r in rows if r['switched'] and r['outcome'] == label]
        groups[label] = dict(n=len(rs),
            score_support_mean=float(np.mean([r['scoring_noise_support'] for r in rs])) if rs else None,
            score_halves_same_winner=sum(r['scoring_halves_same_winner'] for r in rs),
            strong_score_support=sum(r['scoring_noise_support'] >= 7 for r in rs),
            execution_halves_same_direction=sum(r['both_execution_halves_same_direction'] for r in rs),
            normalized_margin_median=float(np.median([r['normalized_margin'] for r in rs])) if rs else None)
    return dict(n=len(rows), coverage=len(complete), switched=sum(r['switched'] for r in rows),
                counts={k:sum(r['outcome']==k for r in rows) for k in ('improved','degraded','equal','missing')},
                delta=float(np.mean([v['delta'] for v in per_task.values()])) if len(complete)==len(rows) else None,
                per_task=per_task, switched_groups=groups)


def run(config_file):
    cfg=read(config_file); base=read(cfg['synthesis_config']); root=Path(base['libero_plus_root'])
    out=Path(cfg['output'])
    if out.exists(): raise ValueError('Refuse to overwrite an existing audit')
    accepted=read(Path(base['output'])/'summary.json')
    for p,h in accepted['sources'].items():
        if sha(p)!=h: raise ValueError('Changed input: '+p)
    with np.load(base['libero_matrix']) as z:
        old=z['success'].copy(); views=z['candidate_ids'].copy()
    heldout=np.flatnonzero(np.char.startswith(views, 'broad_heldout_'))
    own=read(root/'window-recovery-ownership.json')['records']
    prod=read(base['plus_production_config']); panels={}; baseline={}
    def probe(path):
        with np.load(Path(path)/'probes.npz') as z: return z['scores'].copy(), z['views'].copy()
    for mi,model in enumerate(('canonical','broad')):
        scores=[probe(next(r for r in own if r['model']==model and r['initial']==i)['output'])[0] for i in range(32)]
        records=accepted['panels']['libero_'+model]['records']
        panels['libero97_'+model]=(scores,old[mi],[views]*32,records,32)
        panels['libero97_matched4_'+model]=(scores,old[mi,:,:,:4],[views]*32,records,4)
        ss=[];ys=[];vs=[]
        for i in range(32):
            s,v=probe(root/'plus-probes'/model/f'{i:02d}');ss.append(s);vs.append(v)
            y=np.full((len(s),4),np.nan);y[0]=old[mi,i,0,:4]
            with np.load(root/'plus-evaluation'/model/f'{i:02d}'/'probes.npz') as z: y[z['execution_view_indices']]=z['success']
            if i==0:
                with np.load(Path(prod['admission_root'])/model/'00'/'probes.npz') as z:
                    for j,vv in enumerate(z['execution_view_indices']):
                        if vv in prod['reused_rollout_indices']['0']: y[vv]=z['success'][j]
            if not np.isfinite(y).all(): raise ValueError('Incomplete Plus')
            ys.append(y)
        panels['plus_native_'+model]=(ss,ys,vs,records,4)
        native=np.array([y[1:].mean() for y in ys])
        custom4=old[mi][:,heldout,:4].mean((1,2)); custom32=old[mi][:,heldout].mean((1,2))
        baseline[model]=dict(custom32=float(custom32.mean()), custom4=float(custom4.mean()),
            plus4=float(native.mean()),
            per_task={t:dict(custom32=float(custom32[[r['task']==t for r in records]].mean()),
                            custom4=float(custom4[[r['task']==t for r in records]].mean()),
                            plus4=float(native[[r['task']==t for r in records]].mean())) for t in sorted({r['task'] for r in records})},
            plus_candidates_per_initial=[len(y)-1 for y in ys])
    with np.load(Path(base['robocasa_accepted'])/'accepted-arrays.npz') as z:
        for model in ('official','xr1'):
            panels['robocasa_'+model]=(z[model+'__scores'].copy(),z[model+'__success'].copy(),
                [z['views'].copy()]*80,accepted['panels']['robocasa_'+model]['records'],8)
    result={}; all_rows=[]
    for name,(scores,ys,vs,records,repeats) in panels.items():
        result[name]={}
        original_key=name.replace('libero97_matched4_','libero_').replace('libero97_','libero_').replace('plus_native_','plus_')
        golden=accepted['panels'][original_key]['methods']['include']['scopes']['all']['mean'][2:]
        actual=np.mean([y[select(s)].mean(-1) for s,y in zip(scores,ys)],axis=0)
        if 'matched4' in name:
            golden=accepted['bridge'][name.rsplit('_',1)[-1]]['include']['mean'][2:]
        if not np.allclose(actual,golden,rtol=0,atol=1e-14): raise ValueError('Published choices/outcomes changed')
        for exclude in (False,True):
            mode='exclude' if exclude else 'include'; result[name][mode]={}
            for start,end in ((1,3),(3,4),(3,5),(1,5)):
                rows=[dict(panel=name,mode=mode,execution_repeats=repeats,**r,
                      **transition(s,y,v,start,end,exclude)) for r,s,y,v in zip(records,scores,ys,vs)]
                all_rows.extend(rows); scopes={}
                for split in ('all',*sorted({r['split'] for r in rows})):
                    scopes[split]=aggregate([r for r in rows if split=='all' or r['split']==split])
                result[name][mode][f'W{start}_W{end}']=scopes
    write(out/'summary.json',dict(status='PASS',panels=result,baseline_matching=baseline,
        config=cfg,config_sha256=sha(config_file),source_sha256=sha(__file__),sources=accepted['sources'],
        note='Descriptive paired endpoint audit, not fitted rules, causal attribution or new independent tests. Plus-to97 selections/endpoints equal matched4 LIBERO97; only disturbance reference differs.'))
    write(out/'initial-transitions.json',all_rows)
    with (out/'initial-transitions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in all_rows for k in r)))
        writer.writeheader();writer.writerows(all_rows)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    print('PASS',len(run(p.parse_args().config)))
