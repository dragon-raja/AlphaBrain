"""Fixed-support and transition descriptions from the structural audit cache."""
import argparse
from pathlib import Path

import numpy as np

from ..data.artifacts import read, sha, write
from .flow_structure import RULES, WINDOWS, aggregate, choices


def run(config_file):
    cfg=read(config_file);root=Path(cfg['output']);out=root/'diagnostics'
    out.mkdir(exist_ok=False)
    write(out/'scope.json',dict(rules=RULES,windows=WINDOWS,
        fixed_support='The previously declared extra32 views from accepted-arrays.npz, identical for every initial. Separate 32-view estimand; not an estimate for full97 selections. No label-based subset.',
        transitions='All original W1->W3 and W3->W5 switches in six panels; channel attribution is descriptive, not a channel selector.',
        low_score_failures='Count observed zero-success original winners changed to positive-success candidates, and the converse. Missing labels stay unknown.'))
    summary=read(root/'summary.json');base=read(cfg['synthesis_config'])
    path=Path(base['robocasa_accepted'])/'accepted-arrays.npz'
    sources=read(root/'sources.json')
    if sha(path)!=sources[str(path)]:raise ValueError('Accepted endpoint data changed')
    with np.load(path) as z:extra=z['extra_views'].tolist()
    support={};transitions=[];failure_counts={};derived={}
    for panel in summary['coverage']:
        rr=[];rescue={mode:{name:{str(w):dict(rescued_zero_success=0,lost_to_zero_success=0,paired_known=0,unknown=0)
                               for w in WINDOWS} for name in RULES} for mode in ('include','exclude')}
        for i in range(summary['coverage'][panel]['initials']):
            jp=root/panel/f'{i:03d}.json';npz=root/panel/f'{i:03d}.npz';r=read(jp)
            derived[str(jp)]=sha(jp);derived[str(npz)]=sha(npz)
            with np.load(npz) as z:
                views=z['views'].tolist();y=z['outcomes'];s=z['baseline__score']
                for mode in rescue:
                    for rule in RULES:
                        for wi,w in enumerate(WINDOWS):
                            a=r['selection'][mode]['original'][wi]['success'];b=r['selection'][mode][rule][wi]['success'];c=rescue[mode][rule][str(w)]
                            if a is None or b is None:c['unknown']+=1
                            else:
                                c['paired_known']+=1;c['rescued_zero_success']+=int(a==0 and b>0);c['lost_to_zero_success']+=int(a>0 and b==0)
                if panel.startswith('robocasa_'):
                    ids=np.array([views.index(v) for v in extra]);yy=y[ids]
                    if not np.isfinite(yy).all():raise ValueError('Missing fixed32 endpoint')
                    row={k:r[k] for k in ('initial','task','split')};row['selection']={}
                    for mode in ('include','exclude'):
                        row['selection'][mode]={}
                        for rule,key in RULES.items():
                            chosen=choices(z[key][ids])
                            row['selection'][mode][rule]=[dict(window=w,view=extra[j] if j>=0 else None,
                                success=float(yy[j].mean()) if j>=0 else None,
                                outcomes=yy[j].tolist() if j>=0 else None) for w,j in zip(WINDOWS,chosen)]
                    rr.append(row)
                keys=[k for k in z.files if k.startswith('baseline_') and k.endswith('__accel_contribution')]
                for (si,ei,start,end) in ((0,1,1,3),(1,2,3,5)):
                    a=r['selection']['include']['original'][si];b=r['selection']['include']['original'][ei]
                    if a['view']==b['view']:continue
                    ia,ib=views.index(a['view']),views.index(b['view']);delta={};final={}
                    for key in keys:
                        diff=(z[key][ib]-z[key][ia]).mean(0)
                        cumulative=np.cumsum(diff)/np.arange(1,6)
                        delta[key]=float(cumulative[end-1]-cumulative[start-1]);final[key]=float(cumulative[end-1])
                    original=np.cumsum((s[ib]-s[ia]).mean(0))/np.arange(1,6)
                    if not np.isclose(sum(delta.values()),original[end-1]-original[start-1],atol=1e-12):raise ValueError('Transition reconstruction differs')
                    outcome='unknown' if a['success'] is None or b['success'] is None else ('improved' if b['success']>a['success'] else 'degraded' if b['success']<a['success'] else 'equal')
                    transitions.append(dict(panel=panel,initial=r['initial'],task=r['task'],start=start,end=end,
                        old_view=a['view'],new_view=b['view'],old_success=a['success'],new_success=b['success'],outcome=outcome,
                        cumulative_channel_shift=delta,end_channel_delta=final,
                        strongest_channel_towards_new=min(delta,key=delta.get)))
        failure_counts[panel]=rescue
        if rr:
            # Canonical is absent from this fixed support. Only one support comparison exists.
            support[panel]={scope:methods['include'] for scope,methods in aggregate(rr).items()}
            write(out/(panel+'-fixed32-records.json'),rr)
    groups={}
    for panel in summary['coverage']:
        groups[panel]={}
        for start,end in ((1,3),(3,5)):
            groups[panel][f'W{start}_W{end}']={}
            for outcome in ('improved','degraded','equal','unknown'):
                rr=[r for r in transitions if r['panel']==panel and r['start']==start and r['end']==end and r['outcome']==outcome]
                channels={}
                for r in rr:
                    key=r['strongest_channel_towards_new'];channels[key]=channels.get(key,0)+1
                groups[panel][f'W{start}_W{end}'][outcome]=dict(n=len(rr),strongest_channel_counts=channels)
    write(out/'transitions.json',transitions)
    write(out/'summary.json',dict(status='PASS',fixed32=support,fixed32_views=extra,transition_groups=groups,
        zero_success_changes=failure_counts,source_sha256=sha(__file__),derived_sources=derived,
        main_summary_sha256=sha(root/'summary.json'),accepted_arrays_sha256=sha(path),
        scope='Descriptive reuse of the fixed structural audit; no new score variants, inference, outcomes, or fitted weights. Fixed32 results must not be reported as full97 success rates.'))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    print(run(p.parse_args().config))
