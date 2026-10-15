"""Existing-data structural audit; no model, renderer, or simulator imports."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path

import numpy as np

from ..data.artifacts import read, sha, write
from ..metrics.flow_structure import structure
from .window_synthesis import select, summarize


RULES = {'original': 'baseline__score', 'near': 'near__score',
         'near_radial': 'near__radial', 'near_angular': 'near__angular'}
WINDOWS = (1, 3, 5)


def checked_npz(path, expected):
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected:
        raise ValueError('Changed input: '+str(path))
    return np.load(io.BytesIO(payload), allow_pickle=False)


def choices(scores, exclude=False):
    """Exclude undefined candidates, with no fallback based on labels."""
    output = []
    for w in WINDOWS:
        valid = np.isfinite(scores[:, :, :w]).all((1, 2))
        if exclude:
            valid[0] = False
        ids = np.flatnonzero(valid)
        # Preserve original mean-over-noise then cumulative-window arithmetic.
        cost = np.cumsum(scores[ids].mean(1), axis=-1)[:, w-1] / w
        output.append(int(ids[np.argmin(cost)]) if len(ids) else -1)
    return output


def weights(records):
    result = np.zeros(len(records))
    tasks = sorted({r['task'] for r in records})
    for task in tasks:
        splits = sorted({r['split'] for r in records if r['task'] == task})
        for split in splits:
            ids = [i for i, r in enumerate(records) if r['task'] == task and r['split'] == split]
            result[ids] = 1 / (len(tasks)*len(splits)*len(ids))
    return result


def aggregate(rows):
    """Incomplete labels yield bounds, never an observed-winner mean."""
    result = {}
    for scope in ['all', *sorted({r['split'] for r in rows})]:
        rr = [r for r in rows if scope == 'all' or r['split'] == scope]
        wt = weights(rr)
        result[scope] = {}
        for mode in ('include', 'exclude'):
            result[scope][mode] = {}
            for rule in RULES:
                entries = []
                for wi, w in enumerate(WINDOWS):
                    old = [r['selection'][mode]['original'][wi] for r in rr]
                    new = [r['selection'][mode][rule][wi] for r in rr]
                    known = np.array([a['success'] is not None for a in new])
                    complete = known.all()
                    lo = np.array([a['success'] if a['success'] is not None else 0. for a in new])
                    hi = np.where(known, lo, 1.)
                    oldlo = np.array([a['success'] if a['success'] is not None else 0. for a in old])
                    oldhi = np.array([a['success'] if a['success'] is not None else 1. for a in old])
                    # If both rules pick the same unknown candidate, paired difference is exactly zero.
                    same = np.array([a['view'] == b['view'] and a['view'] is not None for a, b in zip(old, new)])
                    dlo, dhi = lo-oldhi, hi-oldlo
                    dlo[same] = 0.; dhi[same] = 0.
                    paired = [(a['success'], b['success']) for a,b in zip(old,new)
                              if a['success'] is not None and b['success'] is not None]
                    delta = np.array([b-a for a,b in paired])
                    entry = dict(window=w, n=len(rr), label_coverage=int(known.sum()),
                        success=float(wt@lo) if complete else None,
                        success_bounds=[float(wt@lo), float(wt@hi)],
                        paired_delta_bounds=[float(wt@dlo), float(wt@dhi)],
                        same_choice=int(same.sum()), paired_known=len(paired),
                        improved=int((delta>1e-12).sum()), degraded=int((delta < -1e-12).sum()),
                        equal=int((np.abs(delta)<=1e-12).sum()),
                        undefined_winners=sum(a['view'] is None for a in new))
                    if complete and all(a['success'] is not None for a in old):
                        x = np.array([[a['outcomes'], a['outcomes'], b['outcomes']] for a,b in zip(old,new)])
                        stats = summarize(x, rr)
                        entry.update(success=stats['mean'][2], delta=stats['gain'][0],
                            delta_ci95=stats['gain_ci95'][0],
                            per_task_delta={k:v[2]-v[0] for k,v in stats['per_task'].items()})
                        ds = list(entry['per_task_delta'].values())
                        entry['leave_one_task_out_delta'] = [(sum(ds)-d)/(len(ds)-1) for d in ds] if len(ds)>1 else []
                    entries.append(entry)
                result[scope][mode][rule] = entries
    return result


def initial_record(meta, views, y, features, original, golden):
    if views != golden['views']:
        raise ValueError('Candidate order differs from previous audit')
    observed = np.isfinite(y).all(1)
    labels = [float(a.mean()) if valid else None for a,valid in zip(y,observed)]
    if labels != golden['success']:
        raise ValueError('Endpoint labels differ from previous audit')
    if not np.array_equal(features['baseline__score'], original):
        raise ValueError('Original score is not reproduced exactly')
    row = {k:meta[k] for k in ('initial', 'task', 'split')}
    row.update(views=views, candidate_count=len(views), labeled_candidates=int(observed.sum()),
        score_count=int(original.size), original_exact=True,
        max_identity_error=float(np.nanmax(features['baseline__reconstruction_error'])), selection={})
    for mode, exclude in [('include', False), ('exclude', True)]:
        row['selection'][mode] = {}
        for name, key in RULES.items():
            ids = choices(features[key], exclude)
            if name == 'original' and ids != select(original,exclude)[[0,2,4]].tolist():
                raise ValueError('Original selections differ')
            row['selection'][mode][name] = [dict(window=w, view=views[j] if j>=0 else None,
                success=labels[j] if j>=0 else None,
                outcomes=y[j].tolist() if j>=0 and observed[j] else None) for w,j in zip(WINDOWS,ids)]
    row['invalid_directions'] = {k:int((~a).sum()) for k,a in features.items() if k.endswith('__direction_valid')}
    return row


def run(config_file):
    cfg = read(config_file); out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=False)
    write(out/'frozen-plan.json', cfg)
    base = read(cfg['synthesis_config']); root = Path(base['libero_plus_root'])
    frozen_path = Path(base['output'])/'summary.json'; frozen = read(frozen_path)
    consensus = Path(cfg['consensus_root']); consensus_summary = read(consensus/'summary.json')
    sources = {str(frozen_path):sha(frozen_path), str(consensus/'summary.json'):sha(consensus/'summary.json')}
    def verify(path, expected=None):
        path = str(path); digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('Changed source: '+path)
        sources[path] = digest
        return Path(path)
    verify(cfg['synthesis_config']); verify(cfg['prefix_analysis']+'/summary.json')
    for p in cfg['coordinate_sources']:
        verify(p)
    ownership = read(verify(root/'window-recovery-ownership.json'))['records']
    production = read(verify(base['plus_production_config']))
    path = base['libero_matrix'];verify(path,frozen['sources'][path])
    with np.load(path) as z:
        old_y = z['success'].copy()
    panels = {}; coverage = {}
    def save_initial(panel, index, meta, views, y, features, original, golden):
        row = initial_record(meta,views,y,features,original,golden)
        p = out/panel; p.mkdir(exist_ok=True)
        np.savez_compressed(p/f'{index:03d}.npz', **features, views=np.array(views), outcomes=y)
        write(p/f'{index:03d}.json',row)
        return row
    def finish(panel, rows):
        panels[panel] = aggregate(rows)
        coverage[panel] = dict(initials=len(rows), candidates=sum(r['candidate_count'] for r in rows),
            labeled_candidates=sum(r['labeled_candidates'] for r in rows),
            scores=sum(r['score_count'] for r in rows), original_exact=all(r['original_exact'] for r in rows),
            max_identity_error=max(r['max_identity_error'] for r in rows),
            invalid_directions={k:sum(r['invalid_directions'][k] for r in rows) for k in rows[0]['invalid_directions']})
        write(out/panel/'summary.json',panels[panel]);write(out/'coverage.json',coverage)
        print(panel,'COMPLETE',coverage[panel]['candidates'],flush=True)
    for family in ('libero', 'plus'):
        for mi, model in enumerate(('canonical','broad')):
            panel=family+'_'+model; old_panel=('libero97_' if family=='libero' else 'plus_native_')+model
            golden=read(verify(consensus/(old_panel+'.json'))); rows=[]
            for i,meta in enumerate(frozen['panels'][panel]['records']):
                p=Path(next(r for r in ownership if r['model']==model and r['initial']==i)['output']) if family=='libero' else root/'plus-probes'/model/f'{i:02d}'
                path=str(p/'probes.npz');sources[path]=frozen['sources'][path]
                with checked_npz(path,sources[path]) as z:
                    features=structure(z['velocities'],'libero');original=z['scores'];views=z['views'].tolist()
                y=old_y[mi,i] if family=='libero' else np.full((len(views),4),np.nan)
                if family=='plus':
                    y[0]=old_y[mi,i,0,:4];path=str(root/'plus-evaluation'/model/f'{i:02d}'/'probes.npz')
                    sources[path]=frozen['sources'][path]
                    with checked_npz(path,sources[path]) as z:y[z['execution_view_indices']]=z['success']
                    if i==0:
                        path=str(Path(production['admission_root'])/model/'00'/'probes.npz');sources[path]=frozen['sources'][path]
                        with checked_npz(path,sources[path]) as z:
                            for j,v in enumerate(z['execution_view_indices']):
                                if v in production['reused_rollout_indices']['0']:y[v]=z['success'][j]
                rows.append(save_initial(panel,i,meta,views,y,features,original,golden[i]))
                if i%8==7:print(panel,i+1,flush=True)
            finish(panel,rows)
    robo=Path(base['robocasa_accepted']);path=str(robo/'accepted-arrays.npz')
    sources[path]=frozen['sources'][path]
    with checked_npz(path,sources[path]) as accepted:
        views=accepted['views'].tolist()
        for model in ('official','xr1'):
            panel='robocasa_'+model; golden=read(verify(consensus/(panel+'.json')))
            audit=read(verify(robo/f'{model}-probe-audit.json'));index={(r['initial'],r['view']):r for r in audit['records']}
            rows=[]
            def load(receipt):
                cell=Path(receipt['cell']);path=str(cell/'probes.npz')
                if sha(cell/'complete.json')!=receipt['receipt_verified_sha256']:
                    raise ValueError('Receipt changed')
                with checked_npz(path,receipt['payload_verified_sha256']) as z:
                    features=structure(z['velocity_trace'],model);original=z['scores'].copy()
                    if not np.array_equal(features['baseline__score'],original):raise ValueError('Cell score differs')
                return features, original
            for i,meta in enumerate(frozen['panels'][panel]['records']):
                if accepted[model+'__initials'][i]!=meta['initial']:raise ValueError('Initial order differs')
                receipts=[index[meta['initial'],v] for v in views]
                with ThreadPoolExecutor(max_workers=cfg['cpu_workers']) as pool:loaded=list(pool.map(load,receipts))
                features={k:np.stack([r[0][k] for r in loaded]) for k in loaded[0][0]}
                original=np.stack([r[1] for r in loaded])
                if not np.array_equal(original,accepted[model+'__scores'][i]):raise ValueError('Accepted score differs')
                for r in receipts:
                    sources[str(Path(r['cell'])/'probes.npz')]=r['payload_verified_sha256']
                    sources[str(Path(r['cell'])/'complete.json')]=r['receipt_verified_sha256']
                rows.append(save_initial(panel,i,meta,views,accepted[model+'__success'][i],features,original,golden[i]))
                if i%5==4:print(panel,i+1,flush=True)
            finish(panel,rows)
    write(out/'sources.json',sources)
    write(out/'summary.json',dict(status='PASS',coverage=coverage,panels=panels,config=cfg,
        source_sha256=sha(__file__),metric_sha256=sha(Path(__file__).parents[1]/'metrics/flow_structure.py'),
        sources_sha256=sha(out/'sources.json'),scope='Development evidence on previously inspected panels; original scores and labels unchanged. No new inference or simulation. Fixed four rules, W1/W3/W5, include/exclude canonical. Unknown outcomes remain unknown; bounds use [0,1].'))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    print(run(p.parse_args().config))
