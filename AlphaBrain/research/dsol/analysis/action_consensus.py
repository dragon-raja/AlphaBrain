"""Existing-data action-consensus audit. Never invokes a policy or simulator."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from ..data.artifacts import read, sha, write
from ..metrics.action_consensus import disagreement
from .window_synthesis import select, rank_corr


def describe(actions, scores, outcomes, views, record, horizon):
    signals = dict(chunk_translation=disagreement(actions, horizon, slice(0, 3)),
                   first_translation=disagreement(actions, 1, slice(0, 3)),
                   chunk_rotation=disagreement(actions, horizon, slice(3, 6)))
    observed = np.isfinite(outcomes).all(1)
    y = np.full(len(views), np.nan); y[observed] = outcomes[observed].mean(1)
    low = np.argsort(scores[:, :, 0].mean(1), kind='stable')[:int(np.ceil(len(views)/4))]
    results = {}
    for name, d in signals.items():
        ranks = (rankdata(d)-1)/(len(d)-1)
        def rate(ids):
            ids = np.asarray(ids); ids = ids[observed[ids]]
            return dict(n=len(ids), success=float(y[ids].mean()) if len(ids) else None)
        # Fixed rank halves within the low-Accel quarter. Descriptive, not a selector.
        ordered = low[np.argsort(d[low], kind='stable')]; cut = (len(low)+1)//2
        quartiles = [rate(ids) for ids in np.array_split(np.argsort(d, kind='stable'), 4)]
        winner = int(np.argmin(d))
        results[name] = dict(disagreement=d.tolist(), rank=ranks.tolist(),
            rho_agreement_success=rank_corr(-d[observed], y[observed]),
            quartiles=quartiles, low_accel_consensus=rate(ordered[:cut]),
            low_accel_outlying=rate(ordered[cut:]), winner=views[winner],
            winner_success=float(y[winner]) if observed[winner] else None)
    return dict(**record, views=views, observed=observed.tolist(),
                success=[float(v) if np.isfinite(v) else None for v in y],
                low_accel_indices=low.tolist(), choices=select(scores).tolist(), signals=results)


def aggregate(records):
    def finite_mean(values):
        values=[v for v in values if v is not None]
        return float(np.mean(values)) if values else None
    out={}
    for scope in ['all', *sorted({r['split'] for r in records})]:
        rr=[r for r in records if scope=='all' or r['split']==scope]; ss={}
        for signal in rr[0]['signals']:
            vs=[r['signals'][signal] for r in rr]
            rho=[v['rho_agreement_success'] for v in vs if v['rho_agreement_success'] is not None]
            wins=[v['winner_success'] for v in vs]
            low=[(v['low_accel_consensus']['success'],v['low_accel_outlying']['success']) for v in vs]
            low=[p for p in low if None not in p]
            ss[signal]=dict(n=len(rr),rho_mean=float(np.mean(rho)) if rho else None,
                rho_positive=sum(v>0 for v in rho),rho_negative=sum(v<0 for v in rho),rho_defined=len(rho),
                quartile_success=[finite_mean([v['quartiles'][k]['success'] for v in vs]) for k in range(4)],
                low_accel_paired_n=len(low),low_accel_consensus_success=float(np.mean([p[0] for p in low])) if low else None,
                low_accel_outlying_success=float(np.mean([p[1] for p in low])) if low else None,
                winner_coverage=sum(v is not None for v in wins),
                winner_success=float(np.mean(wins)) if all(v is not None for v in wins) else None)
        out[scope]=ss
    return out


def fixed_support(config_file):
    """Use predeclared extra32 support, not adaptively selected window endpoints."""
    cfg=read(config_file);base=read(cfg['synthesis_config']);out=Path(cfg['output'])
    target=out/'fixed-support.json'
    if target.exists():raise ValueError('Refuse to overwrite fixed-support audit')
    path=Path(base['robocasa_accepted'])/'accepted-arrays.npz'
    summary=read(out/'summary.json')
    if sha(path)!=summary['sources'][str(path)]:raise ValueError('Changed accepted arrays')
    with np.load(path) as z:extra=z['extra_views'].tolist()
    result={}
    for model in ('official','xr1'):
        records=read(out/('robocasa_'+model+'.json'));scopes={}
        for scope in ['all',*sorted({r['split'] for r in records})]:
            rr=[r for r in records if scope=='all' or r['split']==scope];signals={}
            for name in rr[0]['signals']:
                details=[]
                for r in rr:
                    ids=np.array([r['views'].index(v) for v in extra]);d=np.array(r['signals'][name]['disagreement']);y=r['success']
                    if any(y[j] is None for j in ids):raise ValueError('Missing predeclared support')
                    yy=np.array([y[j] for j in ids]);dd=d[ids]
                    q=[float(yy[ix].mean()) for ix in np.array_split(np.argsort(dd,kind='stable'),4)]
                    low=[j for j in r['low_accel_indices'] if j in ids]
                    low=sorted(low,key=lambda j:d[j]);cut=(len(low)+1)//2
                    low_pair=[float(np.mean([y[j] for j in group])) for group in (low[:cut],low[cut:])] if len(low)>=2 else None
                    details.append(dict(initial=r['initial'],task=r['task'],rho=rank_corr(-dd,yy),quartiles=q,low_pair=low_pair))
                rho=[r['rho'] for r in details if r['rho'] is not None];low=[r['low_pair'] for r in details if r['low_pair'] is not None]
                signals[name]=dict(n=len(rr),rho_mean=float(np.mean(rho)),rho_defined=len(rho),
                    quartile_success=np.mean([r['quartiles'] for r in details],axis=0).tolist(),
                    low_accel_paired_n=len(low),low_accel_consensus_success=float(np.mean([v[0] for v in low])),
                    low_accel_outlying_success=float(np.mean([v[1] for v in low])),details=details)
            scopes[scope]=signals
        result['robocasa_'+model]=scopes
    write(target,dict(status='PASS',panels=result,views=extra,source_sha256=sha(__file__),
        scope='Associations on the same predeclared extra32 camera endpoints for every initial. Agreement itself uses all97 initial action chunks. Not all97 success labels; low-Accel subset uses full97 W1 ranks intersected with extra32.'))
    return result


def finalize(config_file):
    """Revalidate cached numeric aggregation and summarize all transition cases."""
    cfg=read(config_file);out=Path(cfg['output']);summary=read(out/'summary.json')
    target=out/'validation-and-replay-budget.json'
    if target.exists():raise ValueError('Refuse to overwrite validation')
    for panel,expected in summary['panels'].items():
        if aggregate(read(out/(panel+'.json')))!=expected:raise ValueError('Aggregation golden mismatch')
    transitions=read(out/'transitions.json');counts={}
    for panel in summary['panels']:
        counts[panel]={}
        for start,end in ((1,3),(3,5)):
            counts[panel][f'W{start}_W{end}']={}
            for outcome in ('improved','degraded','equal'):
                rr=[r for r in transitions if r['panel']==panel and r['start']==start and r['end']==end and r['outcome']==outcome and r['switched']]
                ds=[r['signals']['chunk_translation']['new_minus_old'] for r in rr]
                counts[panel][f'W{start}_W{end}'][outcome]=dict(n=len(ds),more_consensus=sum(d<0 for d in ds),less_consensus=sum(d>0 for d in ds),same=sum(d==0 for d in ds))
    rt=read(out/'runtime-evidence.json')
    libero=np.array([r['seconds']/97 for r in rt['libero97_canonical']]);xr=np.array([r['metadata']['seconds'] for r in rt['robocasa_xr1']]);pi=np.array([r['metadata']['seconds'] for r in rt['robocasa_official']])
    write(target,dict(status='PASS',cached_aggregation_exact=True,transition_counts=counts,
        code_sha256=sha(__file__),metric_sha256=sha(Path(__file__).parents[1]/'metrics/action_consensus.py'),
        replay=dict(executed=False,scope='Proposed only: LIBERO initials5,6 and XR1 initials39dd9788d2b8a223cfc09850,a90105e397160a8fa8d737b8; two frozen good/bad views each; eight original scoring repeats; five replans. No full task replay or new success labels.',
            branches=64,scoring_calls_if_all_rerun=320,environment_steps=2688,
            historical_cell_seconds_p10_p50_p90=dict(libero=np.quantile(libero,[.1,.5,.9]).tolist(),xr1=np.quantile(xr,[.1,.5,.9]).tolist(),official=np.quantile(pi,[.1,.5,.9]).tolist()),
            hot_worker_baseline_seconds_p50=float(4*np.median(libero)+4*np.median(xr)),
            assumptions='Historical one cell includes8 repeats/five scores. LIBERO timings amortize whole97 batch. New image/state logging, startup/compilation and replay identity checks excluded; not a throughput guarantee.',
            provisional_execution_budget_minutes=[15,60],instrumentation_and_validation_hours=[2,4],
            storage_estimate='Five checkpoints only: approximately0.1–0.5GB. All environment-step RGB streams: approximately1–3GB raw at224/256 resolution, before lossless compression; excludes full-resolution renderer dumps.')))


def run(config_file):
    cfg=read(config_file); base=read(cfg['synthesis_config']); root=Path(base['libero_plus_root']);out=Path(cfg['output'])
    if out.exists(): raise ValueError('Refuse to overwrite analysis')
    frozen=read(Path(base['output'])/'summary.json');sources={}
    def verify(path, expected=None):
        path=Path(path); h=sha(path)
        if expected is not None and expected!=h: raise ValueError('Changed evidence: '+str(path))
        sources[str(path)]=h; return path
    verify(Path(base['output'])/'summary.json')
    matrix=verify(base['libero_matrix'], frozen['sources'][base['libero_matrix']])
    with np.load(matrix) as z: old=z['success'].copy()
    ownership=read(verify(root/'window-recovery-ownership.json'))['records']
    production=read(verify(base['plus_production_config']));panels={};runtime={}
    for family in ('libero97','plus_native'):
        for mi,model in enumerate(('canonical','broad')):
            panel=family+'_'+model; rows=[]; times=[]
            for i in range(32):
                p=Path(next(r for r in ownership if r['model']==model and r['initial']==i)['output']) if family=='libero97' else root/'plus-probes'/model/f'{i:02d}'
                path=p/'probes.npz'
                with np.load(verify(path,frozen['sources'][str(path)])) as z:
                    actions=np.clip(z['actions'][:,:,0],-1,1);scores=z['scores'];views=z['views'].tolist()
                status=read(p/'status.json');times.append({k:v for k,v in status.items() if 'second' in k or 'elapsed' in k})
                y=old[mi,i] if family=='libero97' else np.full((len(views),4),np.nan)
                if family=='plus_native':
                    y[0]=old[mi,i,0,:4];path=root/'plus-evaluation'/model/f'{i:02d}'/'probes.npz'
                    with np.load(verify(path,frozen['sources'][str(path)])) as z:y[z['execution_view_indices']]=z['success']
                    if i==0:
                        path=Path(production['admission_root'])/model/'00'/'probes.npz'
                        with np.load(verify(path,frozen['sources'][str(path)])) as z:
                            for j,v in enumerate(z['execution_view_indices']):
                                if v in production['reused_rollout_indices']['0']:y[v]=z['success'][j]
                if not np.isfinite(y).all():raise ValueError('Missing full-panel outcomes')
                key=('libero_' if family=='libero97' else 'plus_')+model
                rows.append(describe(actions,scores,y,views,frozen['panels'][key]['records'][i],5))
            panels[panel]=rows;runtime[panel]=times
            write(out/(panel+'.json'),rows);print(panel,'DONE',flush=True)
    robo=Path(base['robocasa_accepted'])
    path=robo/'accepted-arrays.npz'
    verify(path,frozen['sources'][str(path)])
    with np.load(path) as z:
        for model,horizon in [('official',25),('xr1',16)]:
            panel='robocasa_'+model;audit=read(verify(robo/f'{model}-probe-audit.json'))
            index={(r['initial'],r['view']):r for r in audit['records']};views=z['views'].tolist();rows=[];times=[]
            def anchor(receipt):
                cell=Path(receipt['cell']);meta=read(verify(cell/'initial-anchor.json'))
                with np.load(verify(cell/'initial-anchor.npz',meta['arrays_sha256'])) as a:
                    return a['actions'].copy(),a['scores'].copy()
            for i,record in enumerate(frozen['panels'][panel]['records']):
                receipts=[index[record['initial'],v] for v in views]
                with ThreadPoolExecutor(max_workers=8) as pool:loaded=list(pool.map(anchor,receipts))
                actions=np.stack([a for a,s in loaded]);initial_scores=np.stack([s for a,s in loaded])
                scores=z[model+'__scores'][i]
                if not np.array_equal(initial_scores,scores[:,:,0]):raise ValueError('Anchor scoring identity differs')
                rows.append(describe(actions,scores,z[model+'__success'][i],views,record,horizon))
                # Fixed canonical receipt from each initial, no outcome-based timing sample.
                receipt=receipts[0];c=read(verify(Path(receipt['cell'])/'complete.json',receipt['receipt_verified_sha256']))
                times.append(dict(metadata={k:v for k,v in c.items() if k!='records' and ('second' in k or 'elapsed' in k)},
                                  inference_seconds=sum(r.get('infer_seconds',0) for r in c['records'])))
                if i%10==9:print(panel,i+1,flush=True)
            panels[panel]=rows;runtime[panel]=times;write(out/(panel+'.json'),rows)
    transition_path=Path(cfg['transition_audit'])/'initial-transitions.json';transitions=read(verify(transition_path));pairs=[]
    for t in transitions:
        if t['panel'] not in panels or t['mode']!='include' or (t['start'],t['end']) not in ((1,3),(3,5)):continue
        r=next(r for r in panels[t['panel']] if r['initial']==t['initial']);a=r['views'].index(t['old_view']);b=r['views'].index(t['new_view'])
        pairs.append({k:t[k] for k in ('panel','initial','task','split','start','end','old_view','new_view','delta','outcome','switched')}|
            dict(signals={s:dict(old_rank=v['rank'][a],new_rank=v['rank'][b],new_minus_old=v['rank'][b]-v['rank'][a]) for s,v in r['signals'].items()}))
    summary=dict(status='PASS',panels={p:aggregate(r) for p,r in panels.items()},sources=sources,
                 config=cfg,config_sha256=sha(config_file),source_sha256=sha(__file__),
                 metric_sha256=sha(Path(__file__).parents[1]/'metrics/action_consensus.py'),
                 scope='Existing-data descriptive audit. Equal initial weights within each panel/split. RoboCasa associations conditional on observed endpoints, NOT full97. No new inference, replay, learned rule, or threshold tuning. Initial action agreement does not measure later-state agreement or task progress.',
                 primary='Mean paired-noise RMS translation command distance to all other views, first executed chunk. Lower is more consensus. First action and rotation are predefined sensitivity analyses; not selected by outcomes.')
    write(out/'transitions.json',pairs);write(out/'runtime-evidence.json',runtime);write(out/'summary.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--fixed-support',action='store_true');parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    if args.finalize:finalize(args.config);print('PASS')
    elif args.fixed_support:fixed_support(args.config);print('PASS')
    else:print(run(args.config)['status'])
