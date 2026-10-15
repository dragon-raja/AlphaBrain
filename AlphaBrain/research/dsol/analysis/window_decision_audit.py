"""Bounded offline audit of ordinal aggregation and W1-anchored updates."""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from ..data.artifacts import read, sha, write
from .window_synthesis import select


def decisions(scores, exclude, horizon=3, votes=7):
    """Only scoring observations enter decisions. W1 is not a stay action."""
    s=np.asarray(scores,float)[int(exclude):]
    if s.ndim!=3 or s.shape[1:]!=(8,5) or not np.isfinite(s).all():raise ValueError('Invalid scores')
    base=int(s[:,:,0].mean(1).argmin())
    raw=int(s[:,:,:horizon].mean((1,2)).argmin())
    ranks=(rankdata(s,axis=0)-1)/max(1,len(s)-1)
    per_noise=ranks[:,:,:horizon].mean(-1)
    proposal=int(per_noise.mean(1).argmin())
    halves=[int(per_noise[:,p].mean(1).argmin()) for p in (slice(0,4),slice(4,8))]
    support=int((per_noise[proposal]<per_noise[base]).sum())
    accepted=proposal!=base and halves[0]==halves[1]==proposal and support>=votes
    chosen=np.array([base,raw,proposal,proposal if accepted else base])+int(exclude)
    spread=np.subtract(*np.percentile(s,[75,25],axis=0)).mean(0)
    return chosen,dict(accepted=bool(accepted),proposed_change=proposal!=base,score_votes=support,
                       spread_ratio=(spread/spread[0]).tolist() if spread[0]>0 else None)


def report(scores,outcomes,records,exclude,cfg):
    choices=[];info=[];values=[];paired=[]
    for s,y in zip(scores,outcomes):
        c,d=decisions(s,exclude,cfg['horizon'],cfg['required_noise_votes'])
        choices.append(c.tolist());info.append(d);values.append(np.asarray(y)[c].mean(-1))
        pair=np.asarray(y)[c[:2]]
        if np.isfinite(pair).all():
            half=pair.shape[-1]//2;a=pair[:,:half].mean(-1);b=pair[:,half:].mean(-1)
            # Privileged diagnostic: labels select an expert in one execution
            # half only; the other half evaluates it. Not a deployable policy.
            value=(b[int(a[1]>a[0])]+a[int(b[1]>b[0])])/2
            paired.append(dict(cross_half=value,w1=float(pair[0].mean()),w3=float(pair[1].mean()),
                               repeat_halves_agree=bool(np.sign(a[1]-a[0])==np.sign(b[1]-b[0]))))
    values=np.asarray(values);scopes={}
    for scope in ['all',*sorted({r['split'] for r in records})]:
        ids=np.array([i for i,r in enumerate(records) if scope=='all' or r['split']==scope])
        tasks=np.array([records[i]['task'] for i in ids]);v=values[ids]
        means=[float(np.mean([v[tasks==t,j].mean() for t in np.unique(tasks)]))
               if np.isfinite(v[:,j]).all() else None for j in range(4)]
        scopes[scope]=dict(initials=len(ids),coverage=np.isfinite(v).sum(0).tolist(),success=means,
                           gate_accepts=sum(info[i]['accepted'] for i in ids))
    valid=[x['spread_ratio'] for x in info if x['spread_ratio'] is not None]
    paired_summary=None
    if len(paired)==len(records):
        paired_summary={key:float(np.mean([v[key] for v in paired])) for key in paired[0]}
    return dict(scopes=scopes,choices=choices,decisions=info,two_expert_noise_diagnostic=paired_summary,
                median_candidate_IQR_ratio=np.median(valid,axis=0).tolist(),
                note='All means conditional on existing data. Gate votes are an engineering stability criterion, not calibrated success confidence.')


def run(config_file):
    cfg=read(config_file);base=read(cfg['synthesis_config']);root=Path(base['libero_plus_root'])
    accepted=read(Path(base['output'])/'summary.json');sources={}
    # Bind to the already-audited files without modifying their old receipts.
    for p,h in accepted['sources'].items():
        if sha(p)!=h:raise ValueError('Source changed: '+p)
    sources.update(accepted['sources'])
    with np.load(base['libero_matrix']) as z:yold=z['success'].copy()
    own=read(root/'window-recovery-ownership.json')['records']
    prod=read(base['plus_production_config']);panels={}
    def score(path):
        with np.load(path/'probes.npz') as z:return z['scores'].copy()
    for mi,model in enumerate(('canonical','broad')):
        s=[score(Path(next(r for r in own if r['model']==model and r['initial']==i)['output'])) for i in range(32)]
        panels['libero_'+model]=(s,yold[mi],accepted['panels']['libero_'+model]['records'])
        ss=[];yy=[]
        for i in range(32):
            a=score(root/'plus-probes'/model/f'{i:02d}');ss.append(a);y=np.full((len(a),4),np.nan);y[0]=yold[mi,i,0,:4]
            with np.load(root/'plus-evaluation'/model/f'{i:02d}'/'probes.npz') as z:y[z['execution_view_indices']]=z['success']
            if i==0:
                with np.load(Path(prod['admission_root'])/model/'00'/'probes.npz') as z:
                    for j,v in enumerate(z['execution_view_indices']):
                        if v in prod['reused_rollout_indices']['0']:y[v]=z['success'][j]
            yy.append(y)
        panels['plus_'+model]=(ss,yy,accepted['panels']['plus_'+model]['records'])
    with np.load(Path(base['robocasa_accepted'])/'accepted-arrays.npz') as z:
        for model in ('official','xr1'):
            panels['robocasa_'+model]=(z[model+'__scores'].copy(),z[model+'__success'].copy(),accepted['panels']['robocasa_'+model]['records'])
    result={key:{'exclude' if e else 'include':report(s,y,meta,e,cfg) for e in (False,True)} for key,(s,y,meta) in panels.items()}
    # The two baselines must reproduce the published point estimates exactly.
    for key,m in result.items():
        v=m['include']['scopes']['all']['success'];a=accepted['panels'][key]['methods']['include']['scopes']['all']['mean']
        if not np.allclose([v[0],v[1]],[a[2],a[4]],rtol=0,atol=1e-15):raise ValueError('Original methods changed')
    out=Path(cfg['output'])
    if out.exists():raise ValueError('Do not overwrite exploratory audit')
    result=dict(status='PASS',panels=result,config=cfg,config_sha256=sha(config_file),source_sha256=sha(__file__),
                sources=sources,scoring_times={'libero_plus':[0,5,10,15,20],'robocasa_pi05':[0,25,50,75,100],'robocasa_xr1':[0,16,32,48,64]},
                caveat='No cross-domain time-matched causal claim. Only t=0 common. Gated fallback is W1 selected view, not physical disturbed start. This is not an independently validated method.')
    write(out/'summary.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    print(run(p.parse_args().config)['status'])
