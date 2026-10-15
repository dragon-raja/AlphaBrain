"""Bounded same-noise P0 replay: isolate runtime variation, not score methods.

10 geometry-selected seen initials, original host/GPU, two existing views,
two existing checkpoints, two repeats of execution stream index zero = 80
rollouts. Original score input, state, weights and runtime are immutable.
"""
from pathlib import Path
import argparse
import os
import random
import subprocess
import sys
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, write, sha
from AlphaBrain.research.dsol.data.evaluation_matrix import validate_plan, NUMERIC_FLAGS
from AlphaBrain.research.dsol.data.evaluation_admission import resolve_initial, load_candidate
from benchmarks.Robocasa365.evaluation_matrix import _rollout
from benchmarks.Robocasa365.evaluation_policy import ResidentPolicy
from benchmarks.Robocasa365.evaluation_installation import prepare_candidate_observation
from benchmarks.Robocasa365.snapshot import inner_env, restore_snapshot
from benchmarks.Robocasa365.snapshot_bundle import load_snapshot_bundle, reconstruct_and_restore


def run(plan_file, worker_id, arm, output):
    plan=read(plan_file);p0=read(plan['p0_plan']);reference=validate_plan(p0)
    if sha(plan['p0_plan'])!=plan['p0_plan_sha256'] or sha(__file__)!=plan['runtime_sha256']:
        raise ValueError('Frozen plan/runtime identity differs')
    if (plan['format']!='robocasa_fixed_noise_replay_v1' or plan['execution_repeat']!=0
            or plan['replays']!=2 or plan['view_ids']!=['canonical','train_003']):
        raise ValueError('Unrecognized bounded replay protocol')
    if p0['policy_only_render']:
        raise ValueError('This diagnostic requires the original every-step render schedule')
    initials = [initial for worker in plan['workers'] for initial in worker['initials']]
    if len(initials) != 10 or len(set(initials)) != 10:
        raise ValueError('Diagnostic must remain bounded to ten unique initials')
    w=next(w for w in plan['workers'] if w['worker_id']==worker_id)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(w['gpu']) or os.environ.get('XLA_FLAGS')!=NUMERIC_FLAGS:
        raise ValueError('Runtime device/numeric flags differ')
    model=p0['models'][arm];checkpoint=Path(model['checkpoint'])
    for name,expected in model['files'].items():
        stat=(checkpoint/name).stat()
        if stat.st_size!=expected['bytes'] or stat.st_mtime_ns!=expected['mtime_ns']:
            raise ValueError('Checkpoint differs from P0')
    manifest=read(p0['source_manifest'])
    for name in ('benchmarks/Robocasa365/evaluation_matrix.py','benchmarks/Robocasa365/evaluation_policy.py',
                 'benchmarks/Robocasa365/evaluation_installation.py','AlphaBrain/research/dsol/data/evaluation_matrix.py'):
        if sha(Path(p0['source'])/name)!=manifest[name]:raise ValueError('Original runtime changed')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    status=dict(status='STARTING',completed_rollouts=0,plan_sha256=sha(plan_file),worker=worker_id,arm=arm,started_unix=time.time())
    write(output/'progress.json',status)
    try:
        import gymnasium as gym
        import robocasa  # noqa:F401
        policy=ResidentPolicy(checkpoint,model['config'])
        jobs={j['initial_id']:j for j in reference['jobs']}
        for initial in w['initials']:
            job=jobs[initial]
            old_worker=next(x for x in p0['workers'] if x['arm']==arm and initial in x['initials'])
            if old_worker['host']!=w['host'] or old_worker['gpu']!=w['gpu']:
                raise ValueError('Replay must stay on original host/GPU')
            admission=resolve_initial(p0['reference_plan'],p0['reference_release'],initial)
            bundle=load_snapshot_bundle(Path(job['path']))
            random.seed(job['scene_seed']);np.random.seed(job['scene_seed'])
            env=gym.make('robocasa/'+job['task'],split=job['split'],seed=job['scene_seed'],randomize_cameras=False)
            try:
                env.reset();reconstruct_and_restore(env,bundle)
                if int(inner_env(env).sim._render_context_offscreen.con.offSamples)!=4:
                    raise ValueError('Native rendering changed')
                for view in plan['view_ids']:
                    old_cell=Path(plan['p0_plan']).parent/'workers'/old_worker['worker_id']/'cells'/initial/view
                    scores=read(old_cell/'scores.json');old_trial=read(old_cell/'execution_0.json')
                    if sha(old_cell/'scores.npz')!=scores['arrays_sha256']:raise ValueError('Original input changed')
                    with np.load(old_cell/'scores.npz',allow_pickle=False) as arrays:
                        request={k:arrays[k].copy() for k in arrays.files if k.startswith('observation/')}
                    request['prompt']=scores['prompt']
                    candidate=load_candidate(admission,view)
                    restore_snapshot(env,bundle.snapshot)
                    _,pixels=prepare_candidate_observation(env,bundle,candidate,reference['pixel_gate']['mean_abs'])
                    for replay in range(2):
                        trial=_rollout(env,bundle,candidate,policy,request,initial,0,p0['task_horizons'][job['task']])
                        if trial['initial_physical_sha256']!=old_trial['initial_physical_sha256']:
                            raise ValueError('Physical start differs from P0')
                        if trial['plans'][0]['input_arrays']!=scores['input_arrays']:
                            raise ValueError('Frozen first request differs')
                        trial.update(provenance=dict(plan_sha256=sha(plan_file),worker=worker_id,arm=arm,
                            initial_id=initial,view_id=view,replay=replay,original_trial=str(old_cell/'execution_0.json'),
                            original_trial_sha256=sha(old_cell/'execution_0.json')),candidate_pixels=pixels)
                        write(output/'cells'/initial/view/f'replay_{replay}.json',trial)
                        status['completed_rollouts']+=1
                        status.update(status='RUNNING',task=job['task'],initial_id=initial,view_id=view,heartbeat_unix=time.time())
                        write(output/'progress.json',status)
            finally:env.close()
        status.update(status='COMPLETE',finished_unix=time.time())
    except Exception as exc:
        status.update(status='ERROR',error=f'{type(exc).__name__}: {exc}',finished_unix=time.time());raise
    finally:write(output/'progress.json',status)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True,type=Path);p.add_argument('--worker',required=True)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--arm',choices=('canonical','broad'))
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    if not args.execute:p.error('Explicit --execute required')
    if args.arm:return run(args.plan,args.worker,args.arm,args.output)
    args.output.mkdir(parents=True,exist_ok=False)
    for arm in ('canonical','broad'):
        subprocess.run([sys.executable,__file__,'--plan',str(args.plan),'--worker',args.worker,
                        '--output',str(args.output/arm),'--arm',arm,'--execute'],check=True)


if __name__=='__main__':main()
