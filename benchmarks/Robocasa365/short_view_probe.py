"""Five-replan probes anchored to the original eight scoring noise addresses.

No complete-task evaluation or selection occurs here. Success is logged, not a
probe stopping rule. Actual environment termination still stops admission.
"""
from collections import deque
from pathlib import Path
import argparse
import os
import random
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, write, sha, path_mapping
from AlphaBrain.research.dsol.data.evaluation_admission import resolve_initial, load_candidate
from AlphaBrain.research.dsol.data.evaluation_matrix import flow_noise, NUMERIC_FLAGS
from AlphaBrain.research.dsol.analysis.accel_subspace import panel_cell
from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores
from .xr1_evaluation import noise_seed, compact_observation


def verify_anchor(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        difference = float(np.max(np.abs(actual-expected))) if actual.shape==expected.shape else None
        raise ValueError(f'Original scoring velocity differs: max_abs={difference}')


def prefix_means(scores):
    scores=np.asarray(scores,float)
    if scores.shape!=(8,5) or not np.isfinite(scores).all():
        raise ValueError('All eight complete five-replan probe records required')
    # Match the historical contiguous eight-draw reduction exactly. Numpy's
    # axis=0 reduction on an 8x5 matrix can use a different summation order.
    per_replan=np.array([scores[:,k].copy().mean() for k in range(5)])
    return np.cumsum(per_replan)/np.arange(1,6)


def record_step_event(record, step, success, terminated, truncated):
    """Log completion without censoring a fixed-horizon diagnostic trajectory."""
    if success:
        record.setdefault('success_steps', []).append(int(step))
    if terminated or truncated:
        record['early_terminal']=dict(step=int(step),success=bool(success),
            terminated=bool(terminated),truncated=bool(truncated))
        return True
    return False


def score_fresh_anchor(policy, model, initial, obs, candidate, cell):
    """Independent initial-only scoring pass before any candidate action.

    New initials have no historical score files. Persist this pass, then require
    the five-window path to reproduce all eight first velocities (and pi actions).
    Neither execution outcomes nor later probe observations enter this anchor.
    """
    from .evaluation_policy import policy_request
    for name,image in zip(('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand'),candidate['images']):
        obs['video.'+name]=image
    request=(dict(compact_observation(obs),prompt=obs['annotation.human.task_description'])
             if model=='xr1' else policy_request(obs))
    prompt=request['prompt']
    velocities=[];actions=[];scores=[]
    for repeat in range(8):
        if model=='xr1':
            history=deque([compact_observation(obs)],maxlen=7)
            result=policy.infer(history,prompt,noise_seed(initial,'score',repeat,0),capture=True)
            velocity=result['velocity_trace'][0];metric_v=velocity[None,:,:16,:12]
        else:
            result=policy.trace(request,flow_noise(initial,'score',repeat,0))
            velocity=result['velocity'][0];metric_v=velocity[None]
        metric=compute_accel_scores(metric_v,prefixes=(3,))[3]
        if metric['degenerate'].any():raise ValueError('Degenerate initial-only anchor')
        velocities.append(velocity);actions.append(result['actions']);scores.append(float(metric['score'][0]))
    arrays=dict(velocity_trace=np.stack(velocities),actions=np.stack(actions),scores=np.array(scores),
                **{k:v for k,v in request.items() if k!='prompt'})
    np.savez_compressed(cell/'initial-anchor.npz',**arrays)
    write(cell/'initial-anchor.json',dict(initial=initial,view=candidate['view']['view_id'],model=model,
        prompt=prompt,score_repeats=8,action_steps=0,arrays_sha256=sha(cell/'initial-anchor.npz'),
        noise='score replan0; same addressed noise as probe first inference'))
    return request,dict(prompt=prompt),arrays['velocity_trace'],arrays['actions'],arrays['scores']


def run(config_path, model, output, *, initial_ids=None, view_ids=None, model_python=None, upstream=None):
    cfg=read(config_path); output=Path(output);output.mkdir(parents=True,exist_ok=False)
    status=dict(status='STARTING',model=model,pid=os.getpid(),completed_cells=0,completed_probes=0,
                started_unix=time.time(),full_task_evaluations=0)
    write(output/'status.json',status)
    bridge=None
    try:
        import gymnasium as gym
        import robocasa  # noqa: F401
        from robocasa.utils.env_utils import convert_action
        from .snapshot import inner_env
        from .snapshot_bundle import load_snapshot_bundle,reconstruct_and_restore
        from .evaluation_installation import restore_candidate_start,prepare_candidate_observation,check_fixed_installation
        from .evaluation_policy import ResidentPolicy,policy_request
        from .xr1_bridge import XR1Bridge
        with path_mapping([('/share/longjunyu','/share/personal/longjunyu')]):
            fresh=cfg.get('population_mode')=='fresh_confirmation'
            if fresh:
                panel=None;plan=read(cfg['panel_plan']);base=None
                p0=plan;selection=read(plan['reference_plan'])
                if (sha(plan['reference_plan'])!=plan['reference_plan_sha256']
                        or sha(plan['reference_release'])!=plan['reference_release_sha256']
                        or sha(plan['authorization'])!=plan['authorization_sha256']):
                    raise ValueError('Fresh population or authorization changed')
            else:
                panel=Path(cfg['panel']); plan=read(panel/'plan.json');base=read(plan['baseline_plan'])
                p0=read(plan['p0_plan']);selection=read(panel/'selection.json')
            jobs={j['initial_id']:j for j in selection['jobs']}
            initials=list(initial_ids or plan['initial_ids']);views=list(view_ids or plan['all_views'])
            if not initials or not views or not set(initials)<=set(plan['initial_ids']) or not set(views)<=set(plan['all_views']):
                raise ValueError('Probe scope is outside original panel')
            if model=='xr1':
                previous=cfg['xr1_model'] if fresh else read(Path(cfg['xr1_canonical'])/'plan.json')
                if fresh and sha(Path(previous['checkpoint'])/'IMPORT_RECEIPT.json')!=previous['checkpoint_receipt_sha256']:
                    raise ValueError('XR-1 checkpoint receipt changed')
                bridge=XR1Bridge(model_python or previous['model_python'],previous['checkpoint'],
                    upstream or previous['upstream'],output/'model.log')
                policy=bridge;stride=16
                xr_cells={} if fresh else {(r['initial'],r['view']):Path(r['scores']) for r in read(cfg['xr1_manifest'])['records']}
                for root in (() if fresh else (cfg['xr1_canonical'],cfg['xr1_perturbed'])):
                    for worker in read(Path(root)/'launch.json')['workers']:
                        identity=read(Path(worker['output'])/'identity.json')
                        for i in identity['initials']:
                            for v in identity['views']:
                                xr_cells[i,v]=Path(worker['output'])/'cells'/i/v/'scores.json'
            elif model=='official':
                if os.environ.get('XLA_FLAGS')!=NUMERIC_FLAGS:raise ValueError('Original XLA flags required')
                spec=plan['models']['official']
                if fresh:
                    for name,expected in spec['files'].items():
                        stat=(Path(spec['checkpoint'])/name).stat()
                        if stat.st_size!=expected['bytes'] or stat.st_mtime_ns!=expected['mtime_ns']:
                            raise ValueError('Checkpoint changed since full byte audit')
                    if sha(Path(spec['checkpoint'])/'assets/norm_stats.json')!=spec['norm_sha256']:
                        raise ValueError('Normalization changed')
                policy=ResidentPolicy(Path(spec['checkpoint']),spec['config']);stride=25
            else:raise ValueError('Only XR-1 and official pi05 are admitted')
            write(output/'identity.json',dict(config_sha256=sha(config_path),source_sha256=sha(__file__),
                model=model,initials=initials,views=views,score_repeats=8,max_replans=5,stride=stride,
                population_mode='fresh_confirmation' if fresh else 'original_panel',
                noise='Original score stream, replan0 identical; score stream replan1..4, never execution stream',
                full_task_evaluations=0,probe_stopping='Fixed five replans; success logged, not used for stopping or scoring',
                early_termination='Only terminated/truncated fail closed; preserve partial arrays; no imputation'))
            for initial in initials:
                job=jobs[initial];admitted=resolve_initial(p0['reference_plan'],p0['reference_release'],initial)
                bundle=load_snapshot_bundle(Path(job['path']))
                random.seed(job['scene_seed']);np.random.seed(job['scene_seed'])
                env=gym.make('robocasa/'+job['task'],split=job['split'],seed=job['scene_seed'],randomize_cameras=False)
                try:
                    env.reset();reconstruct_and_restore(env,bundle)
                    if int(inner_env(env).sim._render_context_offscreen.con.offSamples)!=4:
                        raise ValueError('Original native renderer required')
                    for view in views:
                        cell=output/'cells'/initial/view;cell.mkdir(parents=True,exist_ok=False)
                        candidate=load_candidate(admitted,view)
                        restore_candidate_start(env,bundle,candidate)
                        obs,checks=prepare_candidate_observation(env,bundle,candidate,p0.get('pixel_mean_limit',.005))
                        old=None
                        if fresh:
                            request,meta,expected,pi_actions,original_scores=score_fresh_anchor(
                                policy,model,initial,obs,candidate,cell)
                        else:
                            old=panel_cell(panel,plan,base,initial,view);meta=read(old/'scores.json')
                            if sha(old/'scores.npz')!=meta['arrays_sha256']:raise ValueError('Original pi05 score payload changed')
                            with np.load(old/'scores.npz',allow_pickle=False) as z:
                                request={k:z[k].copy() for k in z.files if k.startswith('observation/')}
                                pi_velocity=z['velocity_trace'].copy();pi_actions=z['actions'].copy()
                            request['prompt']=meta['prompt'];obs['annotation.human.task_description']=meta['prompt']
                        if model=='xr1' and not fresh:
                            xm=read(xr_cells[initial,view]);xp=xr_cells[initial,view].with_suffix('.npz')
                            if sha(xp)!=xm['arrays_sha256']:raise ValueError('Original XR-1 score payload changed')
                            with np.load(xp,allow_pickle=False) as z:expected=z['velocity_trace'].copy()
                            for name,image in zip(('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand'),candidate['images']):
                                obs['video.'+name]=image
                            original_scores=np.array([r['accel_p3'] for r in xm['rows']])
                        elif not fresh:
                            expected=pi_velocity;original_scores=np.array([r['metrics']['3']['score'] for r in meta['rows']])
                        scores=np.full((8,5),np.nan);records=[];velocities=[];tick=time.monotonic()
                        write(cell/'source.json',dict(original_pi05_score=str(old/'scores.json') if old is not None else None,
                            original_xr1_score=str(xr_cells[initial,view]) if model=='xr1' and not fresh else None,
                            fresh_anchor=str(cell/'initial-anchor.json') if fresh else None,render_checks=checks))
                        for repeat in range(8):
                            physical=restore_candidate_start(env,bundle,candidate)
                            history=deque([compact_observation(obs)],maxlen=7);current=obs;trace=[]
                            for k in range(5):
                                start=time.monotonic()
                                if model=='xr1':
                                    seed=noise_seed(initial,'score',repeat,k)
                                    result=policy.infer(history,meta['prompt'],seed,capture=True)
                                    velocity=result['velocity_trace'][0];actions=result['actions'];metric_v=velocity[None,:,:16,:12]
                                else:
                                    current_request=request if k==0 else policy_request(current)
                                    result=policy.trace(current_request,flow_noise(initial,'score',repeat,k))
                                    velocity=result['velocity'][0];actions=result['actions'];metric_v=velocity[None]
                                if k==0:
                                    verify_anchor(velocity,expected[repeat])
                                    if model=='official' or fresh:verify_anchor(actions,pi_actions[repeat])
                                metric=compute_accel_scores(metric_v,prefixes=(3,))[3]
                                if metric['degenerate'].any():raise ValueError('Degenerate score, no automatic zero ranking')
                                score=float(metric['score'][0]);scores[repeat,k]=score
                                if k==0 and score!=original_scores[repeat]:raise ValueError('Original metric does not reproduce exactly')
                                trace.append(velocity)
                                record=dict(repeat=repeat,replan=k,action_step=k*stride,accel=score,
                                    infer_seconds=time.monotonic()-start,physical_start=physical if k==0 else None)
                                records.append(record)
                                if k==4:break
                                for aidx,action in enumerate(actions[:stride]):
                                    current,_,terminated,truncated,info=env.step(convert_action(action))
                                    check_fixed_installation(env,candidate)
                                    current['annotation.human.task_description']=meta['prompt']
                                    if model=='xr1':history.append(compact_observation(current))
                                    if record_step_event(record,k*stride+aidx+1,info['success'],terminated,truncated):
                                        np.savez_compressed(cell/'partial.npz',scores=scores)
                                        write(cell/'partial.json',dict(records=records,reason='SHORT_PROBE_TERMINATED'))
                                        raise RuntimeError('Short probe terminated before all five scores; protocol review required')
                            velocities.append(np.stack(trace));status['completed_probes']+=1
                            status.update(status='RUNNING',initial=initial,view=view,repeat=repeat,checked_unix=time.time())
                            write(output/'status.json',status)
                        means=prefix_means(scores)
                        np.savez_compressed(cell/'probes.npz',scores=scores,velocity_trace=np.stack(velocities),prefix_means=means)
                        if means[0]!=original_scores.mean():raise ValueError('Window1 mean changed')
                        write(cell/'complete.json',dict(status='COMPLETE',records=records,seconds=time.monotonic()-tick,
                            arrays_sha256=sha(cell/'probes.npz'),window1_exact=True,full_task_evaluations=0))
                        status['completed_cells']+=1;write(output/'status.json',status)
                finally:env.close()
            status.update(status='COMPLETE',finished_unix=time.time())
    except BaseException as exc:
        status.update(status='FAILED',error=repr(exc),finished_unix=time.time());raise
    finally:
        write(output/'status.json',status)
        if bridge is not None:bridge.close()
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--model',choices=['xr1','official'],required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--initials',nargs='+');parser.add_argument('--views',nargs='+')
    parser.add_argument('--model-python');parser.add_argument('--upstream');parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:parser.error('Explicit --execute required')
    run(args.config,args.model,args.output,initial_ids=args.initials,view_ids=args.views,
        model_python=args.model_python,upstream=args.upstream)
