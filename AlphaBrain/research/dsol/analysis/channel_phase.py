"""Offline temporal alignment of recorded robot state and existing flow evidence.

No renderer, simulator, model, fitted threshold or new selector. Raw state and
render-time geometry remain separate; an action prediction is never execution.
"""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from ..data.artifacts import read,sha,write
from ..metrics.flow_structure import structure
from .flow_structure import checked_npz,weights
from .flow_phase_alignment import clean


CHANNELS=('translation','rotation','gripper','base','control_mode')


def close_mask(command,semantics):
    a=np.asarray(command)
    return a>=semantics['threshold'] if semantics['inclusive'] else a>semantics['threshold']


def predicted_gripper(actions,semantics):
    values=np.asarray(actions)[:,6];closing=close_mask(values,semantics)
    return dict(mean=float(values.mean()),min=float(values.min()),max=float(values.max()),
        closing_count=int(closing.sum()),horizon=len(values),
        first_closing_offset=int(np.flatnonzero(closing)[0]) if closing.any() else None)


def contact_bodies(contacts,names):
    """Native nonpositive contact distance; include palm versus fingertip identity."""
    result={}
    for ga,gb,distance in np.asarray(contacts).reshape(-1,3):
        if distance>0:continue
        a,b=(names['body_names'][names['geom_bodyid'][int(i)]] for i in (ga,gb))
        for gripper,other in ((a,b),(b,a)):
            if gripper.startswith('gripper') and not other.startswith(('gripper','robot')):
                result.setdefault(other,set()).add(gripper)
    return {k:sorted(v) for k,v in result.items()}


def sign_change_intervals(steps,values):
    """Only identify bracketed changes; never invent unobserved crossing times."""
    result=[]
    for lo,hi,a,b in zip(steps[:-1],steps[1:],values[:-1],values[1:]):
        if np.isfinite([a,b]).all() and a*b<0:
            result.append(dict(after=int(lo),by=int(hi),from_sign=int(np.sign(a)),to_sign=int(np.sign(b))))
    return result


def temporal_context(frames,step,stride):
    """Features from observations at/before scoring; no future contact leakage."""
    current=frames[step];past=frames[max(0,step-stride):step+1]
    targets=current['target_distance_m']
    return dict(step=step,raw_joint_separation_m=current['raw_joint_separation_m'],
        observed_joint_separation_m=current['observed_joint_separation_m'],
        render_tip_center_separation_m=current['render_tip_center_separation_m'],
        separation_change_previous_interval_m=current['raw_joint_separation_m']-past[0]['raw_joint_separation_m'],
        target_distance_m=current['target_distance_m'],
        distance_change_previous_interval_m={k:current['target_distance_m'][k]-past[0]['target_distance_m'][k] for k in targets},
        target_displacement_m=current['target_displacement_m'],
        current_target_contact={k:k in current['contact_bodies'] for k in targets},
        target_contact_seen={k:any(k in f['contact_bodies'] for f in frames[:step+1]) for k in targets},
        first_target_contact_so_far={k:next((f['step'] for f in frames[:step+1] if k in f['contact_bodies']),None) for k in targets},
        closing_commands_previous_interval=sum(f['closing_command'] is True for f in past[1:]),
        observed_commands_previous_interval=len(past)-1)


def physical_timeline(z,names,model,stride,targets,semantics):
    ids=[i for i,n in enumerate(names['site_names']) if n.endswith('grip_site')]
    if len(ids)!=1:raise ValueError('Ambiguous grip site')
    fingers=[next(i for i,n in enumerate(names['joint_names']) if n.startswith('gripper') and n.endswith('finger_joint'+str(j))) for j in (1,2)]
    adr=[names['jnt_qposadr'][i] for i in fingers]
    tipids=[next(i for i,n in enumerate(names['body_names']) if n.startswith('gripper') and n.endswith('finger_joint'+str(j)+'_tip')) for j in (1,2)]
    targetids={t:names['body_names'].index(t) for t in targets}
    initial=z['frame/000/sim/body_xpos'];frames=[]
    for step in range(stride*4+1):
        prefix=f'frame/{step:03d}/';bodies=z[prefix+'sim/body_xpos'];eef=z[prefix+'sim/site_xpos'][ids[0]]
        qpos=z[prefix+'sim/qpos'][adr]
        obs=z[prefix+'obs/'+('state.gripper_qpos' if model=='xr1' else 'robot0_gripper_qpos')]
        command=float(z[f'command/{step:03d}'][6]) if step else None
        drawer=[(n,z[prefix+'sim/qpos'][names['jnt_qposadr'][i]]) for i,n in enumerate(names['joint_names']) if n=='wooden_cabinet_1_top_level']
        frames.append(dict(step=step,time=float(z[prefix+'sim/time']),raw_joint_separation_m=float(qpos[0]-qpos[1]),
            observed_joint_separation_m=float(obs[0]-obs[1]),render_tip_center_separation_m=float(np.linalg.norm(bodies[tipids[0]]-bodies[tipids[1]])),
            target_distance_m={t:float(np.linalg.norm(eef-bodies[i])) for t,i in targetids.items()},
            target_displacement_m={t:float(np.linalg.norm(bodies[i]-initial[i])) for t,i in targetids.items()},
            contact_bodies=contact_bodies(z[prefix+'sim/contacts'],names),gripper_command=command,
            closing_command=bool(close_mask(command,semantics)) if step else None,drawer_qpos=dict(drawer)))
    return frames


def baseline_indices(panel,views,name):
    if name=='canonical':return [views.index('canonical')]
    if name!='disturbance_uniform':raise ValueError('Unknown frozen baseline')
    ids=[i for i,v in enumerate(views) if v.startswith('broad_heldout_')] if panel.startswith('libero_') else [i for i,v in enumerate(views) if v!='canonical']
    if not ids or (panel.startswith('libero_') and len(ids)!=32):raise ValueError('Frozen disturbance support differs')
    return ids


def positive_audit(cfg,verify):
    root=Path(cfg['structure_summary']).parent;summary=read(cfg['structure_summary']);diag=read(cfg['structure_diagnostics'])
    original=read(verify(cfg['window_summary'],cfg['input_sha256'][cfg['window_summary']]))
    sources=diag['derived_sources'];rows=[]
    for panel in ('libero_canonical','libero_broad','plus_canonical','plus_broad'):
        for index in range(summary['coverage'][panel]['initials']):
            jp=root/panel/f'{index:03d}.json';npz=jp.with_suffix('.npz');verify(jp,sources[str(jp)]);verify(npz,sources[str(npz)])
            meta=read(jp)
            with np.load(npz,allow_pickle=False) as z:
                views=z['views'].tolist();y=z['outcomes']
                for baseline in ('disturbance_uniform','canonical'):
                    ids=baseline_indices(panel,views,baseline)
                    for mode in ('include','exclude'):
                        for chosen in meta['selection'][mode]['original']:
                            w=chosen['window'];idx=views.index(chosen['view']);cy=float(y[ids].mean()) if np.isfinite(y[ids]).all() else None
                            yy=chosen['success'];label='unknown' if yy is None or cy is None else 'improved' if yy>cy else 'degraded' if yy<cy else 'equal'
                            delta={c:float(z['baseline_'+c+'__accel_contribution'][idx,:,:w].mean()-z['baseline_'+c+'__accel_contribution'][ids,:,:w].mean()) for c in CHANNELS if 'baseline_'+c+'__accel_contribution' in z}
                            score_delta=float(z['baseline__score'][idx,:,:w].mean()-z['baseline__score'][ids,:,:w].mean())
                            if not np.isclose(sum(delta.values()),score_delta,atol=1e-12,rtol=1e-11):raise ValueError('Baseline margin attribution differs')
                            rows.append(dict(panel=panel,initial=meta['initial'],task=meta['task'],split=meta['split'],mode=mode,window=w,
                                baseline=baseline,baseline_support=len(ids),selected_view=chosen['view'],baseline_success=cy,selected_success=yy,outcome=label,score_delta=score_delta,
                                channel_delta=delta,strongest_channel_towards_selected=min(delta,key=delta.get),physical_phase_not_inferred_from_panel_record=True))
    groups=[]
    for panel in summary['coverage']:
        if panel.startswith('robocasa'):continue
        for baseline in ('disturbance_uniform','canonical'):
            for mode in ('include','exclude'):
                for w in (1,3,5):
                    group=[r for r in rows if r['panel']==panel and r['baseline']==baseline and r['mode']==mode and r['window']==w];wt=weights(group)
                    known=all(r['baseline_success'] is not None and r['selected_success'] is not None for r in group)
                    gain=float(wt@np.array([r['selected_success']-r['baseline_success'] for r in group])) if known else None
                    golden=original['panels'][panel]['methods'][mode]['scopes']['all']['mean']
                    if gain is None or not np.isclose(gain,golden[w+1]-golden[0 if baseline=='disturbance_uniform' else 1],atol=1e-14,rtol=0):raise ValueError('Original recovery baseline differs')
                    groups.append(dict(panel=panel,baseline=baseline,mode=mode,window=w,n=len(group),outcomes=dict(Counter(r['outcome'] for r in group)),gain=gain,
                        channels_by_outcome={label:dict(Counter(r['strongest_channel_towards_selected'] for r in group if r['outcome']==label and not (baseline=='canonical' and r['selected_view']=='canonical'))) for label in ('improved','degraded','equal','unknown')}))
    return dict(records=rows,groups=groups,existing_window_transitions=diag['transition_groups'],
        physical_scope='Only the two preselected LIBERO cases have frame telemetry here. LIBERO/Plus panel positive outcomes alone do not identify physical phases. No Plus same-state intervention was collected.')


def run(config_file):
    cfg=read(config_file);out=Path(cfg['output']);sources={}
    if (out/'analysis').exists():raise ValueError('Refuse to overwrite existing analysis')
    def verify(path,digest=None):
        path=str(path);actual=sha(path)
        if digest is not None and actual!=digest:raise ValueError('Changed source: '+path)
        sources[path]=actual
        return path
    for path,digest in cfg['input_sha256'].items():verify(path,digest)
    old=read(cfg['prefix_summary']);same=read(cfg['same_state_results']);ledger=read(cfg['same_state_ledger'])
    if ledger['reserved']!=ledger['completed'] or ledger['reserved']!=172:raise ValueError('Incomplete existing same-state calls')
    call_root=Path(cfg['same_state_ledger']).parent/'calls';calls={}
    for call in ledger['calls']:
        identity=call['identity'];path=call_root/f"{call['index']:03d}.npz";verify(path,call['artifact_sha256'])
        if identity['kind'] in ('diagnostic','diagnostic_cached_control'):
            key=tuple(identity[k] for k in ('initial','origin','view','step','repeat'));calls[key]=path
    bundles=[];original_nodes=[];aligned=[];physical_lookup={}
    for case in old['config']['cases']:
        for candidate in case['candidates']:
            names=read(verify(candidate['names']));model=case['model'];semantics=cfg['gripper_command_semantics'][model]
            for repeat in range(8):
                root=Path(candidate['path'])/f'{repeat:02d}';receipt=read(verify(root/'receipt.json',old['sources'][str(root/'receipt.json')]))
                path=root/'telemetry.npz';verify(path,receipt['arrays_sha256'])
                with np.load(path,allow_pickle=False) as z:
                    frames=physical_timeline(z,names,model,case['stride'],cfg['targets'][model],semantics)
                    branch=dict(case=case['id'],model=model,view=candidate['view'],candidate=candidate['label'],repeat=repeat,
                        historical_endpoint_success=candidate['historical_endpoint_success'],frames=frames,
                        first_closing_execution=next((f['step'] for f in frames if f['closing_command']),None),
                        first_target_contacts={t:next((f['step'] for f in frames if t in f['contact_bodies']),None) for t in cfg['targets'][model]})
                    bundles.append(branch)
                    for window in range(5):
                        step=window*case['stride'];metrics={k:np.asarray(v).item() for k,v in structure(z[f'plan/{window}/velocity'],model).items()}
                        recorded=next(r['accel'] for r in receipt['records'] if r['kind']=='plan' and r['step']==step)
                        if metrics['baseline__score']!=recorded:raise ValueError('Original score differs')
                        original_nodes.append(dict(case=case['id'],view=candidate['view'],repeat=repeat,step=step,
                            metrics=metrics,predicted_gripper=predicted_gripper(z[f'plan/{window}/actions'],semantics),
                            executed_prefix_length=case['stride'] if window<4 else 0,physical=temporal_context(frames,step,case['stride'])))
                    if model=='xr1':
                        initial=receipt['identity']['initial'];physical_lookup[(initial,candidate['view'],repeat)]=frames
                        for row in same:
                            if row['initial']!=initial or row['origin']!=candidate['view'] or row['repeat']!=repeat:continue
                            key=tuple(row[k] for k in ('initial','origin','view','step','repeat'));step=row['step']
                            if row['provenance']=='historical_reuse':actions=z[f'plan/{step//16}/actions']
                            else:
                                with np.load(calls[key],allow_pickle=False) as cached:actions=cached['actions'].copy()
                            aligned.append(dict(**row,predicted_gripper=predicted_gripper(actions,semantics),
                                physical=temporal_context(frames,step,case['stride']),newly_executed=False))
    contrasts=[];flips=[]
    for case in old['config']['cases']:
        if case['model']!='xr1':continue
        initial=next(r['initial'] for r in same if r['initial'].startswith(case['id'][4:]));good,bad=[c['view'] for c in case['candidates']]
        for origin in (good,bad):
            for repeat in range(8):
                rows=[r for r in aligned if r['initial']==initial and r['origin']==origin and r['repeat']==repeat]
                steps=sorted({r['step'] for r in rows});deltas=[]
                for step in steps:
                    a=next(r for r in rows if r['view']==good and r['step']==step);b=next(r for r in rows if r['view']==bad and r['step']==step)
                    diff={k:a['metrics'][k]-b['metrics'][k] for k in a['metrics'] if a['metrics'][k] is not None and b['metrics'][k] is not None and (k=='baseline__score' or k.endswith('__accel_contribution'))}
                    deltas.append(diff['baseline__score']);contrasts.append(dict(initial=initial,origin=origin,repeat=repeat,step=step,good_minus_bad=diff,
                        physical=a['physical'],good_view_prediction=a['predicted_gripper'],bad_view_prediction=b['predicted_gripper']))
                for change in sign_change_intervals(steps,deltas):
                    frames=physical_lookup[(initial,origin,repeat)];lo,hi=change['after'],change['by']
                    flips.append(dict(initial=initial,origin=origin,repeat=repeat,**change,
                        closing_executions_in_interval=sum(f['closing_command'] is True for f in frames[lo+1:hi+1]),
                        first_contact_by_interval_end=temporal_context(frames,hi,16)['first_target_contact_so_far'],
                        width_at_endpoints_m=[frames[s]['raw_joint_separation_m'] for s in (lo,hi)],
                        distance_at_endpoints_m=[frames[s]['target_distance_m'] for s in (lo,hi)]))
    positive=positive_audit(cfg,verify)
    destination=out/'analysis';destination.mkdir()
    write(destination/'physical-timelines.json',clean(bundles));write(destination/'original-nodes.json',clean(original_nodes));write(destination/'same-state-alignment.json',clean(aligned))
    write(destination/'view-contrasts.json',clean(contrasts));write(destination/'flip-intervals.json',clean(flips));write(destination/'positive-controls.json',clean(positive))
    write(destination/'sources.json',sources)
    write(destination/'summary.json',dict(status='COMPLETE',branches=len(bundles),physical_frames=sum(len(b['frames']) for b in bundles),original_nodes=len(original_nodes),same_state_cells=len(aligned),view_contrasts=len(contrasts),sign_change_intervals=len(flips),libero_plus_initial_records=128,new_model_calls=0,new_simulation_steps=0,new_actions=0,source_sha256=sha(__file__),sources_sha256=sha(destination/'sources.json'),scope=cfg['scope'],temporal_rules=cfg['temporal_rules']))
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);print(run(p.parse_args().config))
