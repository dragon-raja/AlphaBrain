"""No-step restoration gate for the explicitly bounded same-state XR-1 audit.

Only builds a MuJoCo rendering model from an admitted scene. No Gym reset,
environment actions, policy imports, queue, or implicit recovery is used.
"""
import argparse
import os
from pathlib import Path
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, sha, write, path_mapping
from AlphaBrain.research.dsol.data.evaluation_admission import resolve_initial, load_candidate
from AlphaBrain.research.dsol.data.prefix_telemetry import exact
from AlphaBrain.research.dsol.data.recorded_mocap import mocap_from_body_poses
from .snapshot_bundle import (load_snapshot_bundle, load_compiled_model_xml,
                              compiled_model_signature, COMPILED_MODEL_FILE)
from .evaluation_render import pixel_check
from .render_preflight import validate_render_device


RAW = ('qpos', 'qvel', 'ctrl', 'qacc_warmstart')
GEOMETRY = ('body_xpos', 'body_xquat', 'site_xpos', 'site_xmat')
CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')


def forbid_step(*args, **kwargs):
    raise RuntimeError('No simulator steps are authorized in this diagnostic')


def validate_restoration_model(sim, bundle, candidates):
    """Reject topology/array mismatches before state writes or rendering."""
    signature = compiled_model_signature(sim.model)
    if signature != bundle.manifest['model_signature']:
        raise ValueError('Direct compiled model signature differs from archived runtime')
    if not set(CAMERAS).issubset(signature['camera_names']):
        raise ValueError('Archived runtime lacks required policy cameras')
    for name, value in bundle.snapshot.model_runtime.items():
        if np.asarray(value).shape != np.asarray(getattr(sim.model, name)).shape:
            raise ValueError('Archived runtime model array shape differs: '+name)
    for view, candidate in candidates.items():
        for key, attr in (('positions', 'cam_pos'), ('quaternions', 'cam_quat')):
            value = np.asarray(candidate[key])
            if value.shape != getattr(sim.model, attr).shape or not np.isfinite(value).all():
                raise ValueError('Candidate camera installation is incompatible: '+view+'/'+key)
    return dict(signature=signature, candidate_views=list(candidates),
                xml_payload=COMPILED_MODEL_FILE, status='PASS_MODEL_SCHEMA')


def restore_frame(sim, snapshot, arrays, step, candidate, recorded_body_names=None):
    """Restore recorded independent state; recompute, never copy, geometry."""
    prefix=f'frame/{step:03d}/sim/'
    supplemental = {}
    if int(sim.model.na):
        if prefix+'act' not in arrays:
            raise ValueError('Intermediate actuator activation was not recorded; cannot assume it')
        supplemental['act'] = arrays[prefix+'act']
    if int(sim.model.nmocap):
        required = ('body_xpos', 'body_xquat')
        if recorded_body_names is None or any(prefix+k not in arrays for k in required):
            raise ValueError('Intermediate mocap body poses/identities were not recorded')
        pos, quat, _ = mocap_from_body_poses(
            sim.model.body_mocapid, int(sim.model.nmocap), sim.model.body_names,
            recorded_body_names, arrays[prefix+'body_xpos'], arrays[prefix+'body_xquat'])
        supplemental.update(mocap_pos=pos, mocap_quat=quat)
    for name, value in supplemental.items():
        if np.asarray(value).shape != getattr(sim.data, name).shape or not np.all(np.isfinite(value)):
            raise ValueError('Invalid recorded independent state: '+name)
    for name,value in snapshot.model_runtime.items():
        getattr(sim.model,name)[:]=value
    sim.set_state_from_flattened(snapshot.sim_state)
    sim.model.cam_pos[:]=candidate['positions'];sim.model.cam_quat[:]=candidate['quaternions']
    for name in RAW:getattr(sim.data,name)[:]=arrays[prefix+name]
    for name, value in supplemental.items():getattr(sim.data,name)[:]=value
    sim.data.time=float(arrays[prefix+'time'])
    sim.forward()
    # Forward may modify solver warm-start; restore its recorded value, as in
    # the existing frozen snapshot reader. This does not integrate any action.
    sim.data.qacc_warmstart[:]=arrays[prefix+'qacc_warmstart']
    checks={name:exact(getattr(sim.data,name),arrays[prefix+name]) for name in RAW+GEOMETRY}
    checks['time']=exact(sim.data.time,arrays[prefix+'time'])
    checks.update({name:exact(getattr(sim.data,name),value) for name,value in supplemental.items()})
    return checks


def run(config_file, output=None, previous_attempt=None):
    cfg=read(config_file);out=Path(output or cfg['output']);gate=out/'restoration'
    inherited = 0
    if previous_attempt is not None:
        previous = read(Path(previous_attempt)/'call-ledger.json')
        inherited = int(previous.get('inherited_reserved', 0))+int(previous['reserved'])
        if inherited < 0 or inherited > cfg['calls_cap']:
            raise ValueError('Invalid inherited model call budget')
    gate.mkdir(exist_ok=False)
    status=dict(status='RUNNING',stage='STATE_RESTORATION',started_unix=time.time(),
        model_calls=0,cumulative_model_calls=inherited,model_call_cap=cfg['calls_cap'],simulation_steps=0,new_actions=0,
        verified_frames=0,checked_frames=0,stop_on_failure=True)
    write(out/'status.json',status)
    write(out/'call-ledger.json',dict(limit=cfg['calls_cap'],inherited_reserved=inherited,
        previous_attempt=str(previous_attempt) if previous_attempt else None,reserved=0,completed=0,calls=[]))
    sources={str(config_file):sha(config_file)};records=[];sim=None
    try:
        write(gate/'render-preflight.json',validate_render_device(cfg['gpu'],os.environ))
        # Explicit execution-only import. The model is a rendering/forward model,
        # not a VLA, and no env constructor or action loop is called.
        from robosuite.utils.binding_utils import MjSim, MjRenderContextOffscreen
        with path_mapping([('/share/longjunyu','/share/personal/longjunyu')]):
            old=read(cfg['original_config']);base=read(old['original_config']);plan=read(base['panel_plan'])
            prior=read(Path(cfg['prefix_root'])/'analysis/summary.json')
            for key in ('reference_plan','reference_release','authorization'):
                if sha(plan[key])!=plan[key+'_sha256']:raise ValueError('Original plan identity changed')
                sources[plan[key]]=plan[key+'_sha256']
            for case in cfg['cases']:
                initial=case['initial'];admitted=resolve_initial(plan['reference_plan'],plan['reference_release'],initial)
                bundle=load_snapshot_bundle(Path(admitted['job']['path']))
                sources[str(bundle.path/'manifest.json')]=sha(bundle.path/'manifest.json')
                candidates={v:load_candidate(admitted,v) for v in case['views']}
                xml=load_compiled_model_xml(bundle)
                sources[str(bundle.path/COMPILED_MODEL_FILE)]=sha(bundle.path/COMPILED_MODEL_FILE)
                sim=MjSim.from_xml_string(xml)
                sim.step=forbid_step
                write(gate/f'{initial}-model-schema.json',validate_restoration_model(sim,bundle,candidates))
                names_path=Path(cfg['prefix_root'])/'xr1'/f'{initial}-names.json'
                names=read(names_path)['body_names'];sources[str(names_path)]=sha(names_path)
                if sources[str(names_path)]!=prior['sources'][str(names_path)]:
                    raise ValueError('Historical body identity record changed')
                if tuple(names)!=tuple(sim.model.body_names):
                    raise ValueError('Recorded body identities differ from compiled model')
                mapping=np.asarray(sim.model.body_mocapid)
                write(gate/f'{initial}-state-schema.json',dict(na=int(sim.model.na),nmocap=int(sim.model.nmocap),
                    mocap_bodies=[dict(body_id=int(i),name=names[i],slot=int(mapping[i])) for i in np.flatnonzero(mapping>=0)],
                    source='Recorded body world poses; no action replay or static-pose assumption'))
                context=MjRenderContextOffscreen(sim,device_id=cfg['gpu'])
                context.vopt.geomgroup[0]=0;context.vopt.geomgroup[1]=1
                if int(context.con.offSamples)!=cfg['renderer_off_samples']:
                    raise ValueError('Native renderer sample count differs')
                for view in case['views']:
                    for repeat in cfg['repeats']:
                        path=Path(cfg['prefix_root'])/'xr1'/initial/view/f'{repeat:02d}'/'telemetry.npz'
                        digest=sha(path)
                        if digest!=prior['sources'][str(path)]:raise ValueError('Historical telemetry changed')
                        sources[str(path)]=digest
                        receipt=read(path.with_name('receipt.json'))
                        if not receipt['complete'] or receipt['arrays_sha256']!=digest:raise ValueError('Incomplete branch')
                        with np.load(path,allow_pickle=False) as z:
                            for score_step in case['steps']:
                                for offset in cfg['history_offsets']:
                                    step=score_step+offset
                                    identity=dict(initial=initial,source_view=view,repeat=repeat,score_step=score_step,frame_step=step)
                                    checks=restore_frame(sim,bundle.snapshot,z,step,candidates[view],names)
                                    row=dict(**identity,geometry_checks=checks,pixel_checks={})
                                    records.append(row);status['checked_frames']+=1
                                    if not all(c['exact'] for c in checks.values()):
                                        np.savez_compressed(gate/'failed-state.npz',
                                            **{f'actual_{k}':np.asarray(getattr(sim.data,k)) for k in RAW+GEOMETRY},
                                            **{f'expected_{k}':z[f'frame/{step:03d}/sim/{k}'] for k in RAW+GEOMETRY})
                                        status['failure_identity']=identity
                                        status['failed_fields']=[k for k,v in checks.items() if not v['exact']]
                                        raise ValueError('Recorded intermediate geometry cannot be reconstructed exactly from saved independent state')
                                    independent=RAW+tuple(k for k in ('act','mocap_pos','mocap_quat') if k in checks)
                                    before={k:np.asarray(getattr(sim.data,k)).copy() for k in independent};before_time=sim.data.time
                                    for camera in CAMERAS:
                                        image=np.ascontiguousarray(sim.render(height=256,width=256,camera_name=camera)[::-1])
                                        old_image=z[f'frame/{step:03d}/obs/video.{camera}']
                                        row['pixel_checks'][camera]=pixel_check(image,old_image,cfg['pixel_gate']['mean_abs'])
                                        if not row['pixel_checks'][camera]['pass']:
                                            np.savez_compressed(gate/'failed-image.npz',actual=image,expected=old_image)
                                            status['failure_identity']=dict(**identity,camera=camera)
                                            raise ValueError('Original-view reconstructed image fails frozen pixel gate')
                                    if sim.data.time!=before_time or any(not np.array_equal(before[k],getattr(sim.data,k)) for k in independent):
                                        raise ValueError('Rendering changed recorded independent state')
                                    status['verified_frames']+=1
                            write(out/'status.json',status)
                            print(initial,view,repeat,'PASS',status['verified_frames'],flush=True)
                sim.free();sim=None
        status.update(status='PASS_STATE_RESTORATION',stage='AWAITING_ENGINEERING_MODEL_CONTROLS')
    except BaseException as exc:
        status.update(status='STOPPED_GATE_FAILURE',error=str(exc),error_type=type(exc).__name__,
            remaining_diagnostic='NOT_STARTED',automatic_retry=False)
        raise
    finally:
        if sim is not None:
            sim.free()
        status['finished_unix']=time.time()
        write(gate/'checks.json',records);write(gate/'sources.json',sources)
        status['source_sha256']=sha(__file__)
        write(out/'status.json',status)
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output');p.add_argument('--previous-attempt');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:p.error('Explicit --execute required')
    print(run(a.config,a.output,a.previous_attempt)['status'])
