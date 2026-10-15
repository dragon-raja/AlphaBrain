"""Audit short-prefix telemetry without turning it into new task outcomes."""
from pathlib import Path
import argparse
import numpy as np
from ..data.artifacts import read,sha,write
from ..metrics.action_evidence import flow_components


def path_metrics(points):
    points=np.asarray(points,float)
    return dict(path_length=float(np.linalg.norm(np.diff(points,axis=0),axis=-1).sum()),
                displacement=float(np.linalg.norm(points[-1]-points[0])),
                start=points[0].tolist(),end=points[-1].tolist())


def analyze(config_file):
    cfg=read(config_file);out=Path(cfg['output'])
    if out.exists():raise ValueError('Refuse to overwrite analysis')
    for path,digest in cfg['historical_success_sources'].items():
        if sha(path)!=digest:raise ValueError('Historical endpoint labels changed')
    branches=[];sources={};cases={}
    for case in cfg['cases']:
        case_rows=[]
        for candidate in case['candidates']:
            meta=read(candidate['names']);sources[candidate['names']]=sha(candidate['names'])
            for repeat in range(8):
                root=Path(candidate['path'])/f'{repeat:02d}';receipt=read(root/'receipt.json')
                if not receipt['complete'] or sha(root/'telemetry.npz')!=receipt['arrays_sha256']:raise ValueError('Incomplete or changed replay')
                identity=receipt['identity']
                if identity['view']!=candidate['view'] or identity['repeat']!=repeat or identity['stream']!='score':
                    raise ValueError('Replay identity mismatch')
                plans=[r for r in receipt['records'] if r['kind']=='plan']
                if len(plans)!=5 or not all(g['exact'] for r in plans for g in r['gate'].values()):raise ValueError('Prefix not exact')
                sources[str(root/'receipt.json')]=sha(root/'receipt.json');sources[str(root/'telemetry.npz')]=receipt['arrays_sha256']
                with np.load(root/'telemetry.npz') as z:
                    steps=sorted({int(k.split('/')[1]) for k in z.files if k.startswith('frame/')})
                    if steps!=list(range(case['stride']*4+1)):raise ValueError('Missing physical timeline')
                    velocities=np.stack([z[f'plan/{w}/velocity'] for w in range(5)])
                    raw,flow=flow_components(velocities[None],case['model'])
                    if not np.array_equal(raw['score'][0],[r['accel'] for r in plans]):raise ValueError('Accel telemetry mismatch')
                    if 'frame/000/obs/robot0_eef_pos' in z:
                        eef=np.stack([z[f'frame/{s:03d}/obs/robot0_eef_pos'] for s in steps]);eef_source='world robot0_eef_pos'
                    else:
                        ids=[i for i,n in enumerate(meta['site_names']) if n.endswith('grip_site')]
                        if len(ids)!=1:raise ValueError('Ambiguous world end-effector site: '+str(ids))
                        eef=np.stack([z[f'frame/{s:03d}/sim/site_xpos'][ids[0]] for s in steps]);eef_source=meta['site_names'][ids[0]]
                    bodies=np.stack([z[f'frame/{s:03d}/sim/body_xpos'] for s in steps]);qpos=np.stack([z[f'frame/{s:03d}/sim/qpos'] for s in steps])
                    body_motion={n:path_metrics(bodies[:,i]) for i,n in enumerate(meta['body_names']) if n and n!='None'}
                    stages=[]
                    for w in range(4):
                        lo=w*case['stride'];hi=(w+1)*case['stride'];commands=np.stack([z[f'command/{s:03d}'] for s in range(lo+1,hi+1)])
                        stages.append(dict(window=w+1,eef=path_metrics(eef[lo:hi+1]),command_translation_norm=float(np.linalg.norm(commands[:,:3],axis=-1).mean()),
                            gripper_command_mean=float(commands[:,6].mean()),
                            contacts_mean=float(np.mean([len(z[f'frame/{s:03d}/sim/contacts']) for s in range(lo,hi+1)]))))
                    row=dict(case=case['id'],candidate=candidate['label'],view=candidate['view'],repeat=repeat,
                        historical_endpoint_success=candidate['historical_endpoint_success'],
                        flow=flow,eef_source=eef_source,eef=path_metrics(eef),eef_by_step=eef.tolist(),stages=stages,
                        bodies=body_motion,body_stage_positions={n:bodies[::case['stride'],i].tolist() for i,n in enumerate(meta['body_names']) if n and n!='None'},
                        qpos_start=qpos[0].tolist(),qpos_end=qpos[-1].tolist(),
                        probe_success_steps=[r['step'] for r in receipt['records'] if r['kind']=='step' and r['success']],
                        telemetry=str(root/'telemetry.npz'))
                    branches.append(row);case_rows.append(row)
        cases[case['id']]={}
        for candidate in case['candidates']:
            rr=[r for r in case_rows if r['view']==candidate['view']]
            cases[case['id']][candidate['label']]=dict(view=candidate['view'],n=len(rr),historical_endpoint_success=candidate['historical_endpoint_success'],
                accel_mean=np.mean([r['flow']['score'] for r in rr],axis=0).tolist(),
                window_mean=(np.cumsum(np.mean([r['flow']['score'] for r in rr],axis=0))/np.arange(1,6)).tolist(),
                accel_min=np.min([r['flow']['score'] for r in rr],axis=0).tolist(),accel_max=np.max([r['flow']['score'] for r in rr],axis=0).tolist(),
                eef_path_mean=float(np.mean([r['eef']['path_length'] for r in rr])),
                eef_displacement_mean=float(np.mean([r['eef']['displacement'] for r in rr])),
                stage_path_mean=np.mean([[v['eef']['path_length'] for v in r['stages']] for r in rr],axis=0).tolist(),
                stage_command_norm=np.mean([[v['command_translation_norm'] for v in r['stages']] for r in rr],axis=0).tolist(),
                prefix_success_repeats=sum(bool(r['probe_success_steps']) for r in rr))
    write(out/'branches.json',branches);write(out/'summary.json',dict(status='PASS',branches=len(branches),cases=cases,sources=sources,config=cfg,source_sha256=sha(__file__),scope='All original saved flow/score prefixes reproduced exactly. Actual physical telemetry newly recorded; old full physical trajectories were not saved. Historical full-task success labels use different execution noise. Prefix telemetry is attribution evidence, NOT a new success estimate or proof of eventual failure cause.'))


def geometry(config_file):
    """Diagnostic ground-truth distances; never supplied to a view selector."""
    cfg=read(config_file);out=Path(cfg['output']);target=out/'geometry.json'
    if target.exists():raise ValueError('Refuse to overwrite geometry audit')
    rows=read(out/'branches.json');result={}
    for case in cfg['cases']:
        selected=[r for r in rows if r['case']==case['id']];groups={}
        tracked=['akita_black_bowl_1_main','wooden_cabinet_1_cabinet_top'] if case['model']=='libero' else ['obj_main']
        for candidate in case['candidates']:
            rr=[r for r in selected if r['view']==candidate['view']];bodies={}
            for body in tracked:
                distances=[];displacements=[]
                for r in rr:
                    b=np.asarray(r['body_stage_positions'][body]);eef=np.asarray(r['eef_by_step'])[::case['stride']]
                    distances.append(np.linalg.norm(eef-b,axis=-1));displacements.append(np.linalg.norm(b-b[:1],axis=-1))
                bodies[body]=dict(distance_m_by_repeat=np.array(distances).tolist(),distance_m_mean=np.mean(distances,0).tolist(),
                    distance_m_min=np.min(distances,0).tolist(),distance_m_max=np.max(distances,0).tolist(),
                    body_displacement_m_mean=np.mean(displacements,0).tolist())
            names=read(candidate['names']);drawer=None
            if 'wooden_cabinet_1_top_level' in names['joint_names']:
                j=names['jnt_qposadr'][names['joint_names'].index('wooden_cabinet_1_top_level')]
                drawer=dict(start=[r['qpos_start'][j] for r in rr],end=[r['qpos_end'][j] for r in rr])
            groups[candidate['label']]=dict(view=candidate['view'],bodies=bodies,drawer=drawer)
        result[case['id']]=groups
    write(target,dict(status='PASS',cases=result,branches_sha256=sha(out/'branches.json'),source_sha256=sha(__file__),scope='Scene body-center to world end-effector distances for explanation only. Not exact contact-surface distances, not a general task-progress metric, not deployment-available ground truth.'))


def contact_audit(config_file):
    cfg=read(config_file);out=Path(cfg['output']);target=out/'contacts.json'
    if target.exists():raise ValueError('Refuse to overwrite contact audit')
    result={}
    for case in cfg['cases']:
        groups={}
        for candidate in case['candidates']:
            meta=read(candidate['names']);repeats=[]
            for repeat in range(8):
                root=Path(candidate['path'])/f'{repeat:02d}';receipt=read(root/'receipt.json')
                if sha(root/'telemetry.npz')!=receipt['arrays_sha256']:raise ValueError('Telemetry changed')
                hits={}
                with np.load(root/'telemetry.npz') as z:
                    for s in range(case['stride']*4+1):
                        for a,b,d in z[f'frame/{s:03d}/sim/contacts']:
                            if d>0:continue
                            a=meta['body_names'][meta['geom_bodyid'][int(a)]];b=meta['body_names'][meta['geom_bodyid'][int(b)]]
                            if a.startswith('gripper') and not b.startswith(('gripper','robot')):hits.setdefault(b,s)
                            if b.startswith('gripper') and not a.startswith(('gripper','robot')):hits.setdefault(a,s)
                    commands=[float(z[f'plan/{w}/actions'][:,6].mean()) for w in range(5)]
                repeats.append(dict(repeat=repeat,first_nonrobot_gripper_contacts=hits,gripper_command_mean=commands))
            groups[candidate['label']]=dict(view=candidate['view'],repeats=repeats)
        result[case['id']]=groups
    write(target,dict(status='PASS',cases=result,source_sha256=sha(__file__),scope='MuJoCo contact pairs with distance<=0, gripper versus nonrobot bodies. Contact does not establish successful grasp. Stage5 commands are predicted but unexecuted.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--geometry',action='store_true');p.add_argument('--contacts',action='store_true');a=p.parse_args()
    if a.contacts:contact_audit(a.config)
    elif a.geometry:geometry(a.config)
    else:analyze(a.config)
