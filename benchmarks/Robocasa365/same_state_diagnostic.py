"""Bounded no-action XR-1 cross-view audit, after the separate restoration gate."""
import argparse
import hashlib
from contextlib import contextmanager
from pathlib import Path
import time
import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, write, sha, path_mapping
from AlphaBrain.research.dsol.data.prefix_telemetry import exact
from AlphaBrain.research.dsol.data.evaluation_admission import resolve_initial, load_candidate
from AlphaBrain.research.dsol.execution.model_calls import ModelCallLedger
from AlphaBrain.research.dsol.metrics.flow_structure import structure
from AlphaBrain.research.dsol.analysis.same_state import summarize
from .same_state_restore import (restore_frame, validate_restoration_model, initialize_renderer,
                                close_render_sim, forbid_step, RAW, GEOMETRY, CAMERAS)
from .recorded_render_pose import install_recorded_render_pose
from .snapshot_bundle import load_snapshot_bundle, direct_scene_xml
from .evaluation_render import pixel_check
from .xr1_bridge import XR1Bridge
from .xr1_evaluation import noise_seed


def history_from_record(arrays, step):
    """Full seven-frame deque: upstream samples offsets -6,-4,-2,0 itself."""
    history=[]
    for frame in range(step-6,step+1):
        prefix=f'frame/{frame:03d}/obs/'
        history.append({k[len(prefix):]:arrays[k].copy() for k in arrays.files
                        if k.startswith((prefix+'state.',prefix+'video.'))})
    if any(len(o)!=len(history[0]) or not o for o in history):
        raise ValueError('Incomplete recorded policy history')
    return history


@contextmanager
def scene(bundle,candidates,cfg):
    from robosuite.utils.binding_utils import MjSim, MjRenderContextOffscreen
    sim=MjSim.from_xml_string(direct_scene_xml(bundle))
    try:
        sim.step=sim.step1=sim.step2=forbid_step
        validate_restoration_model(sim,bundle,candidates)
        initialize_renderer(sim,MjRenderContextOffscreen,cfg['gpu'],cfg['renderer_off_samples'])
        yield sim
    finally:
        close_render_sim(sim)


def render_history(sim,bundle,arrays,step,candidate,names,cfg,own_view):
    history=history_from_record(arrays,step);checks=[]
    for offset in cfg['history_offsets']:
        frame=step+offset;prefix=f'frame/{frame:03d}/sim/'
        raw=restore_frame(sim,bundle.snapshot,arrays,frame,candidate,names)
        if not all(v['exact'] for k,v in raw.items() if k not in GEOMETRY):
            raise ValueError('Recorded independent state differs')
        install_recorded_render_pose(sim,arrays[prefix+'body_xpos'],arrays[prefix+'body_xquat'])
        if not all(exact(getattr(sim.data,k),arrays[prefix+k])['exact'] for k in RAW+GEOMETRY):
            raise ValueError('Render-pose validation differs')
        before={k:np.asarray(getattr(sim.data,k)).copy() for k in RAW+('mocap_pos','mocap_quat','time')}
        for camera in CAMERAS:
            image=np.ascontiguousarray(sim.render(height=256,width=256,camera_name=camera)[::-1])
            original=history[offset+6]['video.'+camera]
            if own_view or camera==CAMERAS[-1]:
                check=pixel_check(image,original,cfg['pixel_gate']['mean_abs'])
                checks.append(dict(frame=frame,camera=camera,**check))
                if not check['pass']:raise ValueError('Repeated render or wrist pixel gate failed')
            # Cross-view intervention changes only the sampled external images.
            if camera!=CAMERAS[-1]:history[offset+6]['video.'+camera]=image
        if any(not exact(getattr(sim.data,k),v)['exact'] for k,v in before.items()):
            raise ValueError('Rendering changed raw state')
    return history,checks


def output_metrics(result):
    velocity=np.asarray(result['velocity_trace'])
    if velocity.shape!=(1,5,16,60) or not np.isfinite(velocity).all():
        raise ValueError('Invalid complete native velocity trace')
    if result['actions'].shape!=(16,12) or not np.isfinite(result['actions']).all():
        raise ValueError('Invalid native action output')
    return scalar_metrics(velocity[0])


def scalar_metrics(velocity):
    # Undefined directions remain missing, never zero-valued evidence.
    return {k:np.asarray(v).item() if np.isfinite(v) else None
            for k,v in structure(velocity,'xr1').items()}


def run(config_file,restoration,output):
    cfg=read(config_file);restoration=Path(restoration);out=Path(output)
    if read(restoration/'status.json')['status']!='PASS_STATE_RESTORATION':
        raise ValueError('Restoration gate has not passed')
    old_cfg=read(restoration/'frozen-plan.json')
    if cfg!=old_cfg:raise ValueError('Scientific config differs from admitted restoration')
    if (len(cfg['cases'])!=2 or any(len(c['views'])!=2 or len(c['steps'])!=3 for c in cfg['cases'])
        or cfg['repeats']!=list(range(8)) or cfg['history_offsets']!=[-6,-4,-2,0]
        or cfg['calls_cap']!=172):raise ValueError('Frozen diagnostic design differs')
    out.mkdir(exist_ok=False)
    inherited=read(restoration/'call-ledger.json')
    prior=inherited['inherited_reserved']+inherited['reserved']
    write(out/'call-ledger.json',dict(limit=cfg['calls_cap'],inherited_reserved=prior,
        previous_attempt=str(restoration),reserved=0,completed=0,calls=[]))
    if prior+172>cfg['calls_cap']:raise ValueError('Insufficient budget for the frozen diagnostic')
    ledger=ModelCallLedger(out/'call-ledger.json');bridge=None;results={};controls=[];sources={}
    status=dict(status='RUNNING_CONTROLS',started_unix=time.time(),new_actions=0,simulation_steps=0)
    write(out/'status.json',status)
    def call(history,prompt,seed,identity):
        digest=hashlib.sha256()
        for observation in history:
            for key,value in sorted(observation.items()):
                digest.update(key.encode());digest.update(str((value.shape,value.dtype)).encode());digest.update(value.tobytes())
        identity=dict(identity,input_sha256=digest.hexdigest())
        index=ledger.reserve(identity)
        result=bridge.infer(history,prompt,seed,capture=True)
        path=out/'calls'/f'{index:03d}.npz';path.parent.mkdir(exist_ok=True)
        with path.open('xb') as f:np.savez_compressed(f,**result)
        ledger.complete(index,sha(path));metrics=output_metrics(result)
        write(path.with_suffix('.json'),dict(identity=identity,seed=seed,metrics=metrics))
        print('CALL',index+1,identity,flush=True)
        return result,metrics
    try:
        with path_mapping([('/share/longjunyu','/share/personal/longjunyu')]):
            old=read(cfg['original_config']);base=read(old['original_config']);plan=read(base['panel_plan'])
            spec=base['xr1_model'];prior_sources=read(restoration/'restoration/sources.json')
            for path,digest in prior_sources.items():
                if sha(path)!=digest:raise ValueError('Restoration source changed: '+path)
            sources.update(prior_sources)
            if sha(Path(spec['checkpoint'])/'IMPORT_RECEIPT.json')!=spec['checkpoint_receipt_sha256']:
                raise ValueError('Checkpoint changed')
            bridge=XR1Bridge(spec['model_python'],spec['checkpoint'],spec['upstream'],out/'model.log')
            delta={};baseline_cache={}
            # Controls for BOTH cases finish before any cross-view diagnostic call.
            for case in cfg['cases']:
                initial=case['initial'];admitted=resolve_initial(plan['reference_plan'],plan['reference_release'],initial)
                bundle=load_snapshot_bundle(Path(admitted['job']['path']))
                candidates={v:load_candidate(admitted,v) for v in case['views']}
                names=read(Path(cfg['prefix_root'])/'xr1'/f'{initial}-names.json')['body_names']
                with scene(bundle,candidates,cfg) as sim:
                    for view in case['views']:
                        source=next(v for c in old['cases'] if c['initial']==initial for v in c['views'] if v['view']==view)
                        anchor=Path(source['cell'])/'initial-anchor.json'
                        if sha(anchor)!=source['sha256']['initial-anchor.json']:raise ValueError('Prompt source changed')
                        prompt=read(anchor)['prompt'];sources[str(anchor)]=sha(anchor)
                        path=Path(cfg['prefix_root'])/'xr1'/initial/view/'00/telemetry.npz'
                        with np.load(path,allow_pickle=False) as arrays:
                            first=case['steps'][0];seed=noise_seed(initial,'score',0,first//16)
                            result,metrics=call(history_from_record(arrays,first),prompt,seed,
                                dict(kind='cached_reference',initial=initial,view=view,step=first,repeat=0))
                            checks=dict(velocity=exact(result['velocity_trace'][0],arrays[f'plan/{first//16}/velocity']),
                                        actions=exact(result['actions'],arrays[f'plan/{first//16}/actions']))
                            expected=structure(arrays[f'plan/{first//16}/velocity'],'xr1')
                            checks['score']=exact(metrics['baseline__score'],expected['baseline__score'])
                            controls.append(dict(kind='cached_reference',initial=initial,view=view,checks=checks))
                            write(out/'controls.json',controls)
                            if not all(v['exact'] for v in checks.values()):raise ValueError('Fixed-input model reference differs')
                            middle=case['steps'][1]
                            baseline,bmetrics=call(history_from_record(arrays,middle),prompt,seed,
                                dict(kind='diagnostic_cached_control',initial=initial,origin=view,view=view,step=middle,repeat=0))
                            baseline_cache[(initial,view,middle,0)]=bmetrics
                            rendered_results=[]
                            for repeat_render in range(2):
                                history,pixels=render_history(sim,bundle,arrays,middle,candidates[view],names,cfg,True)
                                rendered,m=call(history,prompt,seed,dict(kind='rerender_control',initial=initial,view=view,step=middle,repeat=0,render_repeat=repeat_render))
                                for key,value in m.items():
                                    if value is not None and bmetrics[key] is not None:delta[key]=max(delta.get(key,0.),abs(float(value)-float(bmetrics[key])))
                                checks={k:exact(rendered[k],baseline[k]) for k in ('velocity_trace','actions')}
                                controls.append(dict(kind='rerender_control',initial=initial,view=view,render_repeat=repeat_render,pixels=pixels,checks=checks))
                                write(out/'controls.json',controls)
                                # Original pixels are exact in the admitted replay; enforce fixed-input equality.
                                if all(p['max_abs']==0 for p in pixels) and not all(c['exact'] for c in checks.values()):
                                    raise ValueError('Identical rerender input changed model output')
                                rendered_results.append(rendered)
            write(out/'control-delta.json',delta)
            status.update(status='PASS_CONTROLS_RUNNING_DIAGNOSTIC');write(out/'status.json',status)
            for case in cfg['cases']:
                initial=case['initial'];admitted=resolve_initial(plan['reference_plan'],plan['reference_release'],initial)
                bundle=load_snapshot_bundle(Path(admitted['job']['path']));candidates={v:load_candidate(admitted,v) for v in case['views']}
                names=read(Path(cfg['prefix_root'])/'xr1'/f'{initial}-names.json')['body_names']
                with scene(bundle,candidates,cfg) as sim:
                    for origin in case['views']:
                        source=next(v for c in old['cases'] if c['initial']==initial for v in c['views'] if v['view']==origin)
                        prompt=read(Path(source['cell'])/'initial-anchor.json')['prompt']
                        for repeat in cfg['repeats']:
                            path=Path(cfg['prefix_root'])/'xr1'/initial/origin/f'{repeat:02d}'/'telemetry.npz'
                            with np.load(path,allow_pickle=False) as arrays:
                                seed=noise_seed(initial,'score',repeat,case['steps'][0]//16)
                                for step in case['steps']:
                                    for view in case['views']:
                                        identity=dict(initial=initial,origin=origin,view=view,step=step,repeat=repeat)
                                        if view==origin and step==case['steps'][0]:
                                            metrics=scalar_metrics(arrays[f'plan/{step//16}/velocity']);provenance='historical_reuse'
                                        elif view==origin and (initial,view,step,repeat) in baseline_cache:
                                            metrics=baseline_cache[(initial,view,step,repeat)];provenance='control_call_reuse'
                                        else:
                                            history=history_from_record(arrays,step)
                                            if view!=origin:history,_=render_history(sim,bundle,arrays,step,candidates[view],names,cfg,False)
                                            _,metrics=call(history,prompt,seed,dict(kind='diagnostic',**identity));provenance='new_call'
                                        results[(initial,origin,view,step,repeat)]=dict(**identity,metrics=metrics,provenance=provenance)
                            write(out/'results.json',list(results.values()))
            summaries=[]
            metric_keys=['baseline__score']+['baseline_'+c+'__accel_contribution' for c in ('translation','rotation','gripper','base','control_mode')]
            for case in cfg['cases']:
                for step in case['steps']:
                    for metric in metric_keys:
                        values=np.array([[[results[(case['initial'],origin,view,step,r)]['metrics'][metric] for view in case['views']] for origin in case['views']] for r in cfg['repeats']])
                        summaries.append(dict(initial=case['initial'],step=step,metric=metric,cells=values.tolist(),effects=summarize(values,delta.get(metric,0.))))
            write(out/'summary.json',dict(contrasts=summaries,scope='Two previously selected cases; state includes branch history. No new task outcomes or selector validation.',missing_outcome_labels='Not imputed',noise_rule=cfg['fixed_noise']))
            if len(results)!=192 or ledger.data['reserved']!=172:raise ValueError('Frozen coverage/call accounting differs')
            status.update(status='COMPLETE',cells=len(results),historical_reused=32,controls_passed=True)
    except BaseException as exc:
        status.update(status='STOPPED_GATE_FAILURE',error=repr(exc),automatic_retry=False)
        raise
    finally:
        if bridge is not None:bridge.close()
        status.update(model_calls=ledger.data['reserved'],cumulative_model_calls=prior+ledger.data['reserved'],finished_unix=time.time())
        write(out/'sources.json',sources);write(out/'status.json',status)
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--restoration',required=True);p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:p.error('Explicit --execute required')
    print(run(a.config,a.restoration,a.output))
