"""CPU-only audit of saved command/flow evidence for frozen window decisions."""
import argparse
from pathlib import Path

import numpy as np

from ..data.artifacts import read,sha,write
from ..metrics.action_evidence import flow_components,libero_commands


def run(config_file):
    cfg=read(config_file);base=read(cfg['synthesis_config']);root=Path(base['libero_plus_root'])
    out=Path(cfg['output'])
    if out.exists():raise ValueError('Refuse to overwrite audit')
    transition_root=Path(cfg['transition_audit']);transitions=read(transition_root/'initial-transitions.json')
    synthesis=read(Path(base['output'])/'summary.json');sources={}
    def verify(path,expected=None):
        path=Path(path);h=sha(path)
        if expected is not None and h!=expected:raise ValueError('Evidence changed: '+str(path))
        sources[str(path)]=h;return path
    verify(transition_root/'initial-transitions.json');verify(Path(base['output'])/'summary.json')
    rows=[r for r in transitions if r['mode']=='include' and (r['start'],r['end']) in ((1,3),(3,5))
          and 'matched4' not in r['panel']]
    ownership=read(root/'window-recovery-ownership.json')['records'];profiles={}
    for family in ('libero97','plus_native'):
        for model in ('canonical','broad'):
            panel=family+'_'+model;subset=[r for r in rows if r['panel']==panel];profiles[panel]={}
            for i in sorted({r['initial'] for r in subset}):
                path=Path(next(r for r in ownership if r['model']==model and r['initial']==i)['output']) if family=='libero97' else root/'plus-probes'/model/f'{i:02d}'
                path=path/'probes.npz';verify(path,synthesis['sources'][str(path)])
                needed={r[k] for r in subset if r['initial']==i for k in ('old_view','new_view')}
                with np.load(path) as z:
                    ids=z['views'].tolist();actions=z['actions'];vel=z['velocities'];scores=z['scores'];first=z['first_success']
                    for view in needed:
                        j=ids.index(view);raw,flow=flow_components(vel[j],'libero')
                        if not np.array_equal(raw['score'],scores[j]):raise ValueError('Metric recomputation differs')
                        profiles[panel][f'{i}/{view}']=dict(flow=flow,commands=libero_commands(actions[j]),
                            probe_success_repeats=int((first[j]>=0).sum()),full_five_stage_actions=True,
                            realized_state_trajectory=False,per_stage_images=False)
    robo=Path(base['robocasa_accepted'])
    for model in ('official','xr1'):
        panel='robocasa_'+model;subset=[r for r in rows if r['panel']==panel];profiles[panel]={}
        audit_path=verify(robo/f'{model}-probe-audit.json')
        audit=read(audit_path);index={(r['initial'],r['view']):r for r in audit['records']}
        needed={(r['initial'],r[k]) for r in subset for k in ('old_view','new_view')}
        for initial,view in sorted(needed):
            receipt=index[initial,view];cell=Path(receipt['cell'])
            path=verify(cell/'probes.npz',receipt['payload_verified_sha256'])
            complete=read(verify(cell/'complete.json',receipt['receipt_verified_sha256']))
            with np.load(path) as z:
                raw,flow=flow_components(z['velocity_trace'],model)
                if not np.array_equal(raw['score'],z['scores']):raise ValueError('Robo metric recomputation differs')
            profiles[panel][f'{initial}/{view}']=dict(flow=flow,commands=None,
                probe_success_repeats=len({r['repeat'] for r in complete['records'] if r.get('success_steps')}),
                full_five_stage_actions=False,realized_state_trajectory=False,per_stage_images=False)
    paired=[]
    for r in rows:
        a=profiles[r['panel']][f"{r['initial']}/{r['old_view']}"]
        b=profiles[r['panel']][f"{r['initial']}/{r['new_view']}"]
        features={}
        if a['commands'] is not None:
            for feature in a['commands']:
                # New evidence stages only; no threshold tuning or rule fitting.
                aa=np.asarray(a['commands'][feature][r['start']:r['end']],float)
                bb=np.asarray(b['commands'][feature][r['start']:r['end']],float)
                features[feature]=dict(old=float(aa.mean()),new=float(bb.mean()),difference=float((bb-aa).mean()))
        paired.append({k:r[k] for k in ('panel','initial','task','split','start','end','old_view','new_view','outcome','delta','switched')}|
                      dict(added_stage_command_features=features))
    groups={}
    for panel in profiles:
        groups[panel]={}
        for start,end in ((1,3),(3,5)):
            groups[panel][f'W{start}_W{end}']={}
            for outcome in ('improved','degraded','equal'):
                rr=[r for r in paired if r['panel']==panel and r['start']==start and r['end']==end and r['switched'] and r['outcome']==outcome]
                features={}
                names=rr[0]['added_stage_command_features'] if rr else {}
                for f in names:
                    changes=[r['added_stage_command_features'][f]['difference'] for r in rr]
                    features[f]=dict(median_new_minus_old=float(np.median(changes)),
                                     new_lower=int(sum(v<0 for v in changes)),new_higher=int(sum(v>0 for v in changes)))
                groups[panel][f'W{start}_W{end}'][outcome]=dict(n=len(rr),features=features)
    write(out/'profiles.json',profiles);write(out/'paired-evidence.json',paired)
    write(out/'summary.json',dict(status='PASS',groups=groups,sources=sources,profile_counts={k:len(v) for k,v in profiles.items()},
        source_sha256=sha(__file__),metric_source_sha256=sha(Path(__file__).parents[1]/'metrics/action_evidence.py'),
        config=cfg,config_sha256=sha(config_file),scope='Descriptive existing-data evidence only. Commands are not realized motion or task progress. No learned rules, threshold search, new rollouts or image inference.'))
    return groups


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    print('PASS',len(run(p.parse_args().config)))
