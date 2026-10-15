"""Two frozen XR-1 cases, original scoring streams and read-only telemetry."""
from collections import deque
from pathlib import Path
import argparse
import os
import random
import time
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read,write,sha,path_mapping
from AlphaBrain.research.dsol.data.prefix_telemetry import frame,names,exact,save_branch
from AlphaBrain.research.dsol.data.evaluation_admission import resolve_initial,load_candidate
from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores
from .xr1_bridge import XR1Bridge
from .xr1_evaluation import compact_observation,noise_seed


def run(config_file,output):
    cfg=read(config_file);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    status=dict(status='STARTING',pid=os.getpid(),completed_branches=0,started_unix=time.time(),new_complete_rollouts=0);write(out/'status.json',status);bridge=None
    try:
        import gymnasium as gym
        import robocasa
        from robocasa.utils.env_utils import convert_action
        from .snapshot import inner_env
        from .snapshot_bundle import load_snapshot_bundle,reconstruct_and_restore
        from .evaluation_installation import restore_candidate_start,prepare_candidate_observation,check_fixed_installation
        with path_mapping([('/share/longjunyu','/share/personal/longjunyu')]):
            base=read(cfg['original_config']);plan=read(base['panel_plan']);selection=read(plan['reference_plan']);jobs={j['initial_id']:j for j in selection['jobs']}
            for key in ('reference_plan','reference_release','authorization'):
                if sha(plan[key])!=plan[key+'_sha256']:raise ValueError('Original plan identity changed')
            spec=base['xr1_model']
            if sha(Path(spec['checkpoint'])/'IMPORT_RECEIPT.json')!=spec['checkpoint_receipt_sha256']:raise ValueError('Model receipt changed')
            bridge=XR1Bridge(spec['model_python'],spec['checkpoint'],spec['upstream'],out/'model.log')
            for case in cfg['cases']:
                initial=case['initial'];job=jobs[initial];admitted=resolve_initial(plan['reference_plan'],plan['reference_release'],initial)
                bundle=load_snapshot_bundle(Path(job['path']));random.seed(job['scene_seed']);np.random.seed(job['scene_seed'])
                env=gym.make('robocasa/'+job['task'],split=job['split'],seed=job['scene_seed'],randomize_cameras=False)
                try:
                    env.reset();reconstruct_and_restore(env,bundle);sim=inner_env(env).sim
                    if int(sim._render_context_offscreen.con.offSamples)!=4:raise ValueError('Original native renderer required')
                    write(out/(initial+'-names.json'),names(sim))
                    for source in case['views']:
                        view=source['view'];old=Path(source['cell'])
                        for name,digest in source['sha256'].items():
                            if sha(old/name)!=digest:raise ValueError('Original artifact changed')
                        with np.load(old/'probes.npz') as z:expected=z['velocity_trace'].copy();scores=z['scores'].copy()
                        meta=read(old/'initial-anchor.json')
                        if sha(old/'initial-anchor.npz')!=meta['arrays_sha256']:raise ValueError('Anchor changed')
                        with np.load(old/'initial-anchor.npz') as z:anchor={k:z[k].copy() for k in z.files}
                        original_records=read(old/'complete.json')['records'];candidate=load_candidate(admitted,view)
                        restore_candidate_start(env,bundle,candidate);obs,checks=prepare_candidate_observation(env,bundle,candidate,plan.get('pixel_mean_limit',.005))
                        for camera,image in zip(('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand'),candidate['images']):obs['video.'+camera]=image
                        checks={k:exact(v,anchor[k]) for k,v in compact_observation(obs).items()}
                        write(out/initial/view/'initial-input.json',dict(render_checks=checks,anchor_sha256=meta['arrays_sha256']))
                        if not all(v['exact'] for v in checks.values()):raise ValueError('Original initial policy input differs')
                        obs['annotation.human.task_description']=meta['prompt']
                        for repeat in range(8):
                            arrays={};records=[];valid=False
                            identity=dict(model='xr1',initial=initial,view=view,repeat=repeat,original=str(old),stream='score')
                            try:
                                physical=restore_candidate_start(env,bundle,candidate);history=deque([compact_observation(obs)],maxlen=7);current=obs
                                old_hash=next(r['physical_start'] for r in original_records if r['repeat']==repeat and r['replan']==0)
                                if physical!=old_hash:raise ValueError('Original physical start differs')
                                for window in range(5):
                                    step=window*16;arrays.update({f'frame/{step:03d}/'+k:v for k,v in frame(sim,current).items()})
                                    result=bridge.infer(history,meta['prompt'],noise_seed(initial,'score',repeat,window),capture=True)
                                    velocity=result['velocity_trace'][0];actions=result['actions'];score=float(compute_accel_scores(velocity[None,:,:16,:12],prefixes=(3,))[3]['score'][0])
                                    gate=dict(velocity=exact(velocity,expected[repeat,window]),score=exact(score,scores[repeat,window]))
                                    if window==0:gate['actions']=exact(actions,anchor['actions'][repeat])
                                    arrays[f'plan/{window}/velocity']=velocity;arrays[f'plan/{window}/actions']=actions
                                    records.append(dict(kind='plan',window=window,step=step,accel=score,gate=gate))
                                    write(out/'progress.json',dict(initial=initial,view=view,repeat=repeat,window=window,gate=gate))
                                    if not all(v['exact'] for v in gate.values()):raise ValueError('Original prefix differs; no tolerance relaxation')
                                    if window<4:
                                        for k,action in enumerate(actions[:16]):
                                            current,_,terminated,truncated,info=env.step(convert_action(action));check_fixed_installation(env,candidate)
                                            current['annotation.human.task_description']=meta['prompt'];history.append(compact_observation(current));n=step+k+1
                                            arrays[f'command/{n:03d}']=action.copy();arrays.update({f'frame/{n:03d}/'+key:value for key,value in frame(sim,current).items()})
                                            records.append(dict(kind='step',step=n,success=bool(info['success']),terminated=bool(terminated),truncated=bool(truncated)))
                                            if terminated or truncated:raise ValueError('Early simulator termination')
                                valid=True;status['completed_branches']+=1;status['status']='RUNNING';write(out/'status.json',status)
                            finally:save_branch(out/initial/view/f'{repeat:02d}',arrays,records,identity,valid)
                finally:env.close()
        status.update(status='COMPLETE',all_five_windows_exact=True,finished_unix=time.time())
    except BaseException as exc:status.update(status='FAILED',error=repr(exc),finished_unix=time.time());raise
    finally:
        write(out/'status.json',status)
        if bridge is not None:bridge.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:p.error('Explicit --execute required')
    run(a.config,a.output)
