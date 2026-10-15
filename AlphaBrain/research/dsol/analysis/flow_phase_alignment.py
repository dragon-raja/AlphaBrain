"""Align existing short-prefix physical evidence with structural flow scores."""
import argparse
from pathlib import Path

import numpy as np

from ..data.artifacts import read, sha, write
from ..metrics.flow_structure import structure
from .flow_structure import checked_npz, RULES


def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    if isinstance(value,np.generic):return value.item()
    return value


def run(config_file):
    cfg=read(config_file);root=Path(cfg['prefix_analysis']);out=Path(cfg['output'])/'phase-alignment'
    out.mkdir(parents=True,exist_ok=False)
    old=read(root/'summary.json');branches=read(root/'branches.json');contacts=read(root/'contacts.json')
    sources={str(root/name):sha(root/name) for name in ('summary.json','branches.json','contacts.json','geometry.json')}
    for path,digest in old['config']['historical_success_sources'].items():
        if sha(path)!=digest:raise ValueError('Historical outcomes changed')
        sources[path]=digest
    cases={};allrows=[]
    for case in old['config']['cases']:
        candidates=[];steps=np.arange(5)*case['stride']
        for candidate in case['candidates']:
            rr=[];fs=[]
            for repeat in range(8):
                p=Path(candidate['path'])/f'{repeat:02d}';receipt=read(p/'receipt.json')
                if sha(p/'receipt.json')!=old['sources'][str(p/'receipt.json')]:raise ValueError('Receipt changed')
                digest=old['sources'][str(p/'telemetry.npz')]
                if digest!=receipt['arrays_sha256']:raise ValueError('Telemetry identity differs')
                with checked_npz(p/'telemetry.npz',digest) as z:
                    velocity=np.stack([z[f'plan/{w}/velocity'] for w in range(5)])
                features=structure(velocity,case['model']);fs.append(features)
                plans=[r for r in receipt['records'] if r['kind']=='plan']
                if not np.array_equal(features['baseline__score'],[r['accel'] for r in plans]):raise ValueError('Original phase score differs')
                branch=next(r for r in branches if r['case']==case['id'] and r['view']==candidate['view'] and r['repeat']==repeat)
                contact=contacts['cases'][case['id']][candidate['label']]['repeats'][repeat]
                if contact['repeat']!=repeat:raise ValueError('Repeat order differs')
                body='obj_main' if case['model']=='xr1' else 'wooden_cabinet_1_cabinet_top'
                positions=np.asarray(branch['body_stage_positions'][body]);eef=np.asarray(branch['eef_by_step'])[::case['stride']]
                row=dict(case=case['id'],view=candidate['view'],candidate=candidate['label'],repeat=repeat,
                    steps=steps.tolist(),score=features['baseline__score'].tolist(),
                    target_body=body,target_distance_m=np.linalg.norm(eef-positions,axis=-1).tolist(),
                    target_displacement_m=np.linalg.norm(positions-positions[:1],axis=-1).tolist(),
                    first_contacts=contact['first_nonrobot_gripper_contacts'],
                    historical_endpoint_success=candidate['historical_endpoint_success'],
                    endpoint_labels_are_independent_of_score_noise=True,
                    final_plan_executed=False)
                rr.append(row);allrows.append(row)
                sources[str(p/'receipt.json')]=old['sources'][str(p/'receipt.json')];sources[str(p/'telemetry.npz')]=digest
            features={k:np.stack([r[k] for r in fs]) for k in fs[0]}
            np.savez_compressed(out/(case['id']+'-'+candidate['view']+'.npz'),**features)
            candidates.append(dict(label=candidate['label'],view=candidate['view'],rows=rr,features=features))
        good,bad=candidates
        comparisons={}
        for rule,key in RULES.items():
            a,b=good['features'][key],bad['features'][key]
            cumulative=np.cumsum(a-b,axis=-1)/np.arange(1,6)
            flips=[]
            for repeat in range(8):
                d=cumulative[repeat]
                changes=[int(steps[k]) for k in range(1,5) if np.isfinite(d[k-1:k+1]).all() and d[k-1]*d[k]<0]
                flips.append(dict(repeat=repeat,delta_good_minus_bad=d.tolist(),flip_steps=changes))
            comparisons[rule]=dict(stage_delta_mean=np.mean(a-b,axis=0).tolist(),
                cumulative_delta_mean=np.mean(cumulative,axis=0).tolist(),
                good_preferred_repeats=(cumulative<0).sum(0).tolist(),
                bad_preferred_repeats=(cumulative>0).sum(0).tolist(),
                invalid_repeats=(~np.isfinite(cumulative)).sum(0).tolist(),repeats=flips)
        channel_delta={}
        for key in good['features']:
            if key.startswith('baseline_') and key.endswith('__accel_contribution'):
                channel_delta[key]=np.cumsum(np.mean(good['features'][key]-bad['features'][key],axis=0))/np.arange(1,6)
        if not np.allclose(sum(channel_delta.values()),comparisons['original']['cumulative_delta_mean'],atol=1e-12,rtol=1e-11):raise ValueError('Paired channel identity failed')
        cases[case['id']]=dict(title=case['title'],steps=steps.tolist(),comparisons=comparisons,
            cumulative_channel_delta=channel_delta,
            candidates=[dict(label=c['label'],view=c['view'],rows=c['rows'],
                mean_features={k:np.mean(a,axis=0) for k,a in c['features'].items()}) for c in candidates])
    write(out/'alignment.json',clean(dict(status='PASS',cases=cases,branches=len(allrows),sources=sources,
        code_sha256=sha(__file__),metric_sha256=sha(Path(__file__).parents[1]/'metrics/flow_structure.py'),
        scope='Four previously selected cases, eight scoring-noise trajectories per candidate. Same candidate/noise identities; no same-state cross-view data, no new rollout. Historical endpoint labels are independent execution repeats, not outcomes of these prefixes. Final generated chunk is not executed.')))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    print(run(p.parse_args().config))
