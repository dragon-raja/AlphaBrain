"""Additional coverage and association checks; no new selection or fitting."""
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from ..data.artifacts import read,write,sha


def rank_correlation(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float);valid=np.isfinite(a)&np.isfinite(b)
    a,b=a[valid],b[valid]
    if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])


def audit(root):
    root=Path(root);dest=root/'evidence.json'
    if dest.exists():raise ValueError('Refuse to overwrite evidence')
    cfg=read(root/'frozen-plan.json');sources=read(root/'sources.json');pairs=read(root/'transitions.json');index=read(root/'cases.json')
    support=[];association=[];case_lookup={}
    for r in index:
        case=read(root/'cases'/r['file']);case_lookup[(r['panel'],r['initial'])]={c['view']:c for c in case['candidates']}
        feature_path=root/'cases'/r['file'].replace('.json','-features.npz')
        old_path=Path(cfg['structure_root'])/r['panel']/f"{r['initial']:03d}.npz"
        if sha(old_path)!=sources[str(old_path)]:raise ValueError('Changed structure')
        with np.load(feature_path) as f,np.load(old_path) as old:
            for stage in range(5):
                for feature,channel in [('near_rotation_variation','rotation'),('near_rotation_magnitude','rotation'),('near_gripper_variation','gripper')]:
                    a=f[feature][:,:,stage].mean(1);b=old['baseline_'+channel+'__accel_contribution'][:,:,stage].mean(1)
                    association.append(dict(panel=r['panel'],initial=r['initial'],task=r['task'],stage=stage+1,feature=feature,channel=channel,
                        rho=rank_correlation(a,b),scope='Within initial, across all candidates, after averaging eight scoring noises'))
    for p in pairs:
        if not p['changed']:continue
        candidates=case_lookup[(p['panel'],p['initial'])]
        for role in ('old','new'):
            c=candidates[p[role+'_view']];start,end=p['start'],p['end']
            support.append(dict(panel=p['panel'],initial=p['initial'],mode=p['mode'],start=start,end=end,role=role,
                any_predicted_close=bool(np.any(np.asarray(c['features']['full_close_fraction'])[:,start:end]>0)),
                any_gripper_switch=bool(np.any(np.asarray(c['features']['full_gripper_switch'])[:,start:end]>0))))
    selected_cases=[]
    for panel in ('libero_canonical','libero_broad','plus_canonical','plus_broad'):
        for label in ('improved','degraded'):
            matches=[p for p in pairs if p['panel']==panel and p['mode']=='include' and p['start']==3 and p['changed'] and p['outcome']==label]
            if matches:selected_cases.append(min(matches,key=lambda p:p['initial']))
    summary=dict(changed_transition_records=sum(p['changed'] for p in pairs),candidate_role_records=len(support),
        roles_with_predicted_close=sum(s['any_predicted_close'] for s in support),roles_with_gripper_switch=sum(s['any_gripper_switch'] for s in support),
        note='Modes reuse cases; role/window records are not independent trials. No predicted closure in added stages does not imply no closure later in full tasks.',
        selected_case_rule=cfg['case_display'],selected_cases=selected_cases,support=support,associations=association)
    write(dest,summary);return summary
