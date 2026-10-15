"""Bounded XR-1 rollouts from the original frozen RoboCasa initial states."""
from collections import deque
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import path_mapping, read, sha, write
from AlphaBrain.research.dsol.data.evaluation_admission import resolve_initial, load_candidate
from AlphaBrain.research.dsol.data.xr1_program import phase_views
from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores
from .xr1_bridge import XR1Bridge


def noise_seed(initial, stream, repeat, replan=0):
    if stream not in ('score','execution') or min(repeat,replan)<0:
        raise ValueError('Invalid explicit noise address')
    address=['xr1_paired_v1',initial,stream,int(repeat),int(replan)]
    return int.from_bytes(hashlib.sha256(json.dumps(address,separators=(',',':')).encode()).digest()[:8],'little') % (2**63)


def compact_observation(obs):
    return {k:np.array(v,copy=True) for k,v in obs.items() if k.startswith(('state.','video.'))}


def rollout(env,bundle,candidate,policy,first,initial,prompt,repeat,horizon):
    from robocasa.utils.env_utils import convert_action
    from .evaluation_installation import restore_candidate_start, check_fixed_installation
    from .snapshot_bundle import env_numeric_fingerprint

    initial_hash=restore_candidate_start(env,bundle,candidate)
    history=deque([compact_observation(first)],maxlen=7)
    actions=deque();plans=[];traces=[];started=time.monotonic();infer_seconds=0.
    digest=hashlib.sha256()
    for step in range(horizon):
        if not actions:
            seed=noise_seed(initial,'execution',repeat,len(plans));tick=time.monotonic()
            output=policy.infer(history,prompt,seed,capture=len(plans)<5)
            infer_seconds+=time.monotonic()-tick
            chunk=output['actions']
            if chunk.shape!=(16,12) or not np.isfinite(chunk).all():
                raise ValueError('Invalid native XR-1 action chunk')
            row=dict(step=step,seed=seed,action_sha256=hashlib.sha256(chunk.tobytes()).hexdigest())
            if output['velocity_trace'] is not None:
                v=output['velocity_trace'];traces.append(v[0])
                metric=compute_accel_scores(v[:,:,:16,:12],prefixes=(3,))[3]
                row['accel_p3']=float(metric['score'][0])
                row['degenerate']=bool(metric['degenerate'][0])
            plans.append(row);actions.extend(chunk)
        action=actions.popleft();digest.update(action.tobytes())
        obs,_,terminated,truncated,info=env.step(convert_action(action))
        check_fixed_installation(env,candidate)
        history.append(compact_observation(obs))
        success=bool(info['success'])
        if success or terminated or truncated:
            break
    return dict(repeat=repeat,success=success,steps=step+1,horizon=horizon,
        complete_task_attempt=True,terminated=bool(terminated),truncated=bool(truncated),
        initial_physical_sha256=initial_hash,final_physical_sha256=env_numeric_fingerprint(env),
        executed_actions_sha256=digest.hexdigest(),plans=plans,seconds=time.monotonic()-started,
        inference_seconds=infer_seconds),np.stack(traces)


def run(panel,checkpoint,upstream,model_python,output,*,pilot=False,shard=0,shards=1,phase='canonical',work_order=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if phase not in ('canonical','perturbed','remaining'):
        raise ValueError('Unknown diagnostic phase')
    state=dict(status='STARTING',completed_rollouts=0,pilot=pilot,shard=shard,shards=shards,phase=phase)
    write(output/'status.json',state)
    bridge=None
    try:
        import gymnasium as gym
        import robocasa  # noqa: F401
        from robocasa.utils.dataset_registry_utils import get_task_horizon
        from .snapshot_bundle import load_snapshot_bundle,reconstruct_and_restore
        from .snapshot import inner_env
        from .evaluation_installation import restore_candidate_start,prepare_candidate_observation

        with path_mapping([('/share/longjunyu','/share/personal/longjunyu')]):
            plan=read(Path(panel)/'plan.json');p0=read(plan['p0_plan']);base=read(plan['baseline_plan'])
            selection=read(Path(panel)/'selection.json');jobs={j['initial_id']:j for j in selection['jobs']}
            initials=plan['initial_ids'][shard::shards]
            views=phase_views(plan,base,phase)
            order = None
            if work_order is not None:
                from AlphaBrain.research.dsol.data.xr1_rebalance import validate_order
                order = read(work_order)
                validate_order(order, plan['initial_ids'], views, sha(Path(panel)/'plan.json'))
                if pilot:
                    raise ValueError('Work orders are formal missing-cell work, not pilots')
                initials = [order['initial']]
                views = order['views']
            if pilot:
                initials=initials[:1];views=['canonical']
            if not initials or shards<1 or not 0<=shard<shards:
                raise ValueError('Invalid/empty shard')
            identity=dict(panel_sha256=sha(Path(panel)/'plan.json'),initials=initials,views=views,
                checkpoint_receipt_sha256=sha(Path(checkpoint)/'IMPORT_RECEIPT.json'),
                evaluator_sha256=sha(Path(__file__)),adapter_sha256=sha(Path(__file__).with_name('xr1_policy.py')),
                native_history=4,history_interval=2,replan_steps=16,denoise_steps=5,
                noise_version='xr1_paired_v1',execution_repeats=1 if pilot else 4,
                scope='Engineering pilot, not scientific panel' if pilot else 'Matched 40-initial '+phase+' partition',
                physical_camera_motion=False)
            write(output/'identity.json',identity)
            if order is not None:
                identity.update(work_order_sha256=sha(work_order),requested_repeats=order['repeats'])
                write(output/'identity.json',identity)
            bridge=XR1Bridge(model_python,checkpoint,upstream,output/'model.log')
            for initial in initials:
                job=jobs[initial]
                admitted=resolve_initial(p0['reference_plan'],p0['reference_release'],initial)
                bundle=load_snapshot_bundle(Path(job['path']))
                random.seed(job['scene_seed']);np.random.seed(job['scene_seed'])
                env=gym.make('robocasa/'+job['task'],split=job['split'],seed=job['scene_seed'],randomize_cameras=False)
                try:
                    env.reset();reconstruct_and_restore(env,bundle)
                    if int(inner_env(env).sim._render_context_offscreen.con.offSamples)!=4:
                        raise ValueError('Native renderer changed')
                    horizon=int(get_task_horizon(job['task']))
                    if horizon!=p0['task_horizons'][job['task']]:
                        raise ValueError('Original task horizon changed')
                    for view in views:
                        candidate=load_candidate(admitted,view)
                        restore_candidate_start(env,bundle,candidate)
                        obs,checks=prepare_candidate_observation(env,bundle,candidate,p0.get('pixel_mean_limit',.005))
                        # Persisted wording, not a newly sampled environment instruction.
                        from AlphaBrain.research.dsol.analysis.accel_subspace import panel_cell
                        original_cell=panel_cell(panel,plan,base,initial,view)
                        prompt=read(original_cell/'scores.json')['prompt']
                        # Use the checked native cache consistently; no resized Pi05 input.
                        for name,image in zip(('robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand'),candidate['images']):
                            obs['video.'+name]=image
                        cell=output/'cells'/initial/view;cell.mkdir(parents=True,exist_ok=False)
                        write(cell/'render_checks.json',checks)
                        if not pilot:
                            scoring=[];score_traces=[]
                            for repeat in range(8):
                                seed=noise_seed(initial,'score',repeat)
                                traced=bridge.infer([compact_observation(obs)],prompt,seed,capture=True)
                                velocity=traced['velocity_trace'];score_traces.append(velocity[0])
                                metric=compute_accel_scores(velocity[:,:,:16,:12],prefixes=(3,))[3]
                                if metric['degenerate'].any():
                                    raise ValueError('Degenerate initial score')
                                scoring.append(dict(seed=seed,accel_p3=float(metric['score'][0])))
                            np.savez_compressed(cell/'scores.npz',velocity_trace=np.stack(score_traces))
                            write(cell/'scores.json',dict(rows=scoring,arrays_sha256=sha(cell/'scores.npz'),
                                  prefix=3,executed_window=16,active_dims=12))
                        repeats = order['repeats'][view] if order is not None else range(identity['execution_repeats'])
                        for repeat in repeats:
                            state.update(status='RUNNING',initial=initial,view=view,repeat=repeat,task=job['task'])
                            write(output/'status.json',state)
                            trial,traces=rollout(env,bundle,candidate,bridge,obs,initial,prompt,repeat,horizon)
                            np.savez_compressed(cell/f'trace_{repeat}.npz',velocity_trace=traces)
                            trial['trace_sha256']=sha(cell/f'trace_{repeat}.npz')
                            write(cell/f'execution_{repeat}.json',trial)
                            state['completed_rollouts']+=1
                            write(output/'status.json',state)
                finally:
                    env.close()
            state.update(status='COMPLETE',new_rollouts=state['completed_rollouts'])
            write(output/'status.json',state)
    except BaseException as exc:
        state.update(status='FAILED',error=repr(exc));write(output/'status.json',state)
        raise
    finally:
        if bridge is not None:
            bridge.close()
    return state
