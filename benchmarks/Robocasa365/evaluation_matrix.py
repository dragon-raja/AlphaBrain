"""Resident, checksummed RoboCasa initial/view/noise matrix execution.

Native rendering variability is recorded, not confused with Flow noise or
silently removed by modifying images. No GPU work happens at import time.
"""
from collections import deque
import hashlib
import os
from pathlib import Path
import platform
import random
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.data.evaluation_admission import resolve_initial, load_candidate
from AlphaBrain.research.dsol.data.evaluation_matrix import flow_noise, validate_plan, NUMERIC_FLAGS
from AlphaBrain.research.dsol.data.rollout_replay import array_identity
from .evaluation_installation import prepare_candidate_observation, restore_candidate_start, check_fixed_installation
from .evaluation_policy import ResidentPolicy, policy_request
from .evaluation_render import pixel_check
from .snapshot import inner_env, restore_snapshot
from .snapshot_bundle import (load_snapshot_bundle, reconstruct_and_restore,
                              refresh_bundle_observation, env_numeric_fingerprint)


def _identity(request):
    return {key: array_identity(value) for key, value in request.items() if isinstance(value, np.ndarray)}


def _score_cell(policy, request, initial, cell, provenance):
    metadata = cell / 'scores.json'
    payload = cell / 'scores.npz'
    if metadata.exists():
        scores = read(metadata)
        if scores['provenance'] != provenance or sha(payload) != scores['arrays_sha256']:
            raise ValueError('Existing score provenance or payload changed')
        with np.load(payload, allow_pickle=False) as arrays:
            frozen_request = {key: arrays[key].copy() for key in arrays.files if key.startswith('observation/')}
        frozen_request['prompt'] = scores['prompt']
        return frozen_request, scores
    if payload.exists():
        raise ValueError('Uncommitted score payload retained; explicit recovery required')
    rows, velocities, actions = [], [], []
    for repeat in range(8):
        noise = flow_noise(initial, 'score', repeat)
        trace = policy.trace(request, noise)
        if trace['steps'] != 10 or not np.all(np.isfinite(trace['actions'])):
            raise ValueError('Invalid Flow trace or actions')
        if repeat == 0:
            endpoint = policy.infer(request, noise)
            if not np.allclose(endpoint, trace['actions'], rtol=1e-5, atol=1e-5):
                raise ValueError('Traced sampler disagrees with original policy')
        rows.append(dict(repeat=repeat, noise=array_identity(noise), metrics=trace['scores'],
                         times=trace['times'].tolist()))
        velocities.append(trace['velocity'][0])
        actions.append(trace['actions'])
    arrays = {key: value for key, value in request.items() if isinstance(value, np.ndarray)}
    arrays.update(velocity_trace=np.stack(velocities), actions=np.stack(actions))
    with payload.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    scores = dict(provenance=provenance, prompt=request['prompt'], input_arrays=_identity(request),
                  rows=rows, arrays_sha256=sha(payload), primary_accel_prefix=3,
                  scope='8 initial-observation scores; not 8 environment rollouts')
    write(metadata, scores)
    return request, scores


def _rollout(env, bundle, candidate, policy, first_request, initial, repeat, horizon, schedule=None):
    from robocasa.utils.env_utils import convert_action

    initial_hash = restore_candidate_start(env, bundle, candidate)
    queue, plans = deque(), []
    obs = None
    digest = hashlib.sha256()
    started = time.monotonic()
    infer_seconds = env_seconds = 0.
    success = terminated = truncated = False
    for step in range(horizon):
        if not queue:
            request = first_request if obs is None else policy_request(obs)
            noise = flow_noise(initial, 'execution', repeat, len(plans))
            tick = time.monotonic()
            actions = policy.infer(request, noise)
            infer_seconds += time.monotonic() - tick
            if actions.shape[0] < 25 or not np.all(np.isfinite(actions)):
                raise ValueError('Invalid policy action chunk')
            plans.append(dict(step=step, noise=array_identity(noise), input_arrays=_identity(request),
                              actions=array_identity(actions)))
            queue.extend(actions[:25])
        action = queue.popleft()
        digest.update(np.ascontiguousarray(action).tobytes())
        tick = time.monotonic()
        if schedule is not None:
            schedule.before_step(step, 25)
        obs, _, terminated, truncated, info = env.step(convert_action(action))
        check_fixed_installation(env, candidate)
        env_seconds += time.monotonic() - tick
        # Preserve the persisted instruction instead of regenerated language.
        obs['annotation.human.task_description'] = first_request['prompt']
        success = bool(info['success'])
        if success or terminated or truncated:
            break
    return dict(repeat=repeat, success=success, steps=step+1, horizon=horizon,
                terminated=bool(terminated), truncated=bool(truncated), complete_task_attempt=True,
                plans=plans, initial_physical_sha256=initial_hash,
                final_physical_sha256=env_numeric_fingerprint(env), executed_actions_sha256=digest.hexdigest(),
                seconds=time.monotonic()-started, inference_seconds=infer_seconds, environment_seconds=env_seconds)


def run_worker(plan_file, worker_id, output, *, pilot=False, resume=False):
    """Run a frozen disjoint worker; any error is retained and stops that worker."""
    plan_file, output = Path(plan_file), Path(output)
    plan = read(plan_file)
    reference = validate_plan(plan)
    if sha(plan['source_manifest']) != plan['source_manifest_sha256']:
        raise ValueError('Frozen source manifest changed')
    manifest = read(plan['source_manifest'])
    for name in ('benchmarks/Robocasa365/evaluation_matrix.py',
                 'benchmarks/Robocasa365/evaluation_policy.py',
                 'benchmarks/Robocasa365/evaluation_installation.py',
                 'AlphaBrain/research/dsol/data/evaluation_matrix.py',
                 'AlphaBrain/research/dsol/metrics/pi05_jax.py'):
        if sha(Path(plan['source'])/name) != manifest[name]:
            raise ValueError('Frozen execution source changed')
    if plan['release_gate'] != 'USER_APPROVED_NATIVE_RENDER_VARIABILITY_8_SCORE_4_EXECUTION':
        raise ValueError('Formal native-render protocol not approved')
    if os.environ.get('XLA_FLAGS') != NUMERIC_FLAGS:
        raise ValueError('Numerical inference settings differ from frozen plan')
    selected = [w for w in plan['workers'] if w['worker_id'] == worker_id]
    if len(selected) != 1:
        raise ValueError('Unknown worker identity')
    worker = selected[0]
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(worker['gpu']):
        raise ValueError('GPU assignment differs from plan')
    if output.exists() and not resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'active.lock'
    lock.mkdir(exist_ok=False)
    write(lock / 'owner.json', dict(pid=os.getpid(), host=platform.node(), started_unix=time.time()))
    identity = dict(plan_sha256=sha(plan_file), worker_id=worker_id, pilot=pilot)
    if (output / 'identity.json').exists() and read(output / 'identity.json') != identity:
        raise ValueError('Cannot resume a different plan or scope')
    write(output / 'identity.json', identity)
    status = dict(**identity, status='STARTING', completed_cells=0, completed_rollouts=0,
                  started_unix=time.time(), host=platform.node(), pid=os.getpid())
    started = time.monotonic()
    try:
        import gymnasium as gym
        import robocasa  # noqa: F401
        from robocasa.utils.dataset_registry_utils import get_task_horizon

        model = plan['models'][worker['arm']]
        checkpoint = Path(model['checkpoint'])
        for name, expected in model['files'].items():
            stat = (checkpoint/name).stat()
            if stat.st_size != expected['bytes'] or stat.st_mtime_ns != expected['mtime_ns']:
                raise ValueError('Checkpoint file changed since full hash audit')
        if sha(checkpoint / 'assets/norm_stats.json') != model['norm_sha256']:
            raise ValueError('Checkpoint normalization changed')
        policy = ResidentPolicy(checkpoint, model['config'])
        ids = worker['pilot_initials'] if pilot else worker['initials']
        view_ids = plan['pilot_view_ids'] if pilot else plan['view_ids']
        jobs = {j['initial_id']: j for j in reference['jobs']}
        for initial in ids:
            job = jobs[initial]
            admission = resolve_initial(plan['reference_plan'], plan['reference_release'], initial)
            bundle = load_snapshot_bundle(Path(job['path']))
            random.seed(job['scene_seed']); np.random.seed(job['scene_seed'])
            env = gym.make('robocasa/' + job['task'], split=job['split'],
                           seed=job['scene_seed'], randomize_cameras=False)
            schedule = None
            try:
                env.reset()
                reconstruct_and_restore(env, bundle)
                if int(inner_env(env).sim._render_context_offscreen.con.offSamples) != 4:
                    raise ValueError('Native four-sample rendering setting changed')
                horizon = int(get_task_horizon(job['task']))
                if horizon != plan['task_horizons'][job['task']]:
                    raise ValueError('Native task horizon differs from frozen plan')
                obs, _ = refresh_bundle_observation(env, bundle)
                canonical_checks = {name: pixel_check(obs['video.'+name], image, reference['pixel_gate']['mean_abs'])
                                    for name, image in admission['images'].items()}
                if not all(v['pass'] for v in canonical_checks.values()):
                    raise ValueError('Initial render reference admission failed')
                if plan['policy_only_render']:
                    from .render_schedule import PolicyRenderSchedule
                    schedule = PolicyRenderSchedule(inner_env(env).sim)
                for view_id in view_ids:
                    cell = output / 'cells' / initial / view_id
                    cell.mkdir(parents=True, exist_ok=True)
                    provenance = dict(**identity, arm=worker['arm'], initial_id=initial, view_id=view_id,
                                      snapshot_manifest_sha256=job['snapshot_manifest_sha256'])
                    status.update(status='RUNNING', initial_id=initial, view_id=view_id, task=job['task'])
                    write(output / 'progress.json', status)
                    candidate = load_candidate(admission, view_id)
                    restore_snapshot(env, bundle.snapshot)
                    if schedule is not None:
                        schedule.required = True
                    obs, checks = prepare_candidate_observation(env, bundle, candidate, reference['pixel_gate']['mean_abs'])
                    request, scores = _score_cell(policy, policy_request(obs), initial, cell, provenance)
                    trials = []
                    for repeat in range(4):
                        target = cell / f'execution_{repeat}.json'
                        if target.exists():
                            trial = read(target)
                            if (trial['provenance'] != provenance or trial['repeat'] != repeat
                                    or not trial['complete_task_attempt']):
                                raise ValueError('Existing rollout identity mismatch')
                        else:
                            trial = _rollout(env, bundle, candidate, policy, request, initial, repeat, horizon, schedule)
                            trial.update(provenance=provenance, score_arrays_sha256=scores['arrays_sha256'])
                            write(target, trial)
                        trials.append(dict(file=target.name, sha256=sha(target), success=trial['success'],
                                           steps=trial['steps'], seconds=trial['seconds']))
                        status['completed_rollouts'] += 1
                        status.update(elapsed_seconds=time.monotonic()-started, heartbeat_unix=time.time())
                        write(output / 'progress.json', status)
                    summary = dict(provenance=provenance, score_metadata_sha256=sha(cell/'scores.json'),
                                   trials=trials, candidate_pixels=checks, canonical_pixels=canonical_checks,
                                   success_rate=sum(t['success'] for t in trials)/4, complete=True)
                    if not (cell / 'complete.json').exists():
                        write(cell / 'complete.json', summary)
                    status['completed_cells'] += 1
            finally:
                if schedule is not None:
                    schedule.close()
                env.close()
        status.update(status='COMPLETE', elapsed_seconds=time.monotonic()-started, finished_unix=time.time())
    except Exception as exc:
        status.update(status='ERROR', error=f'{type(exc).__name__}: {exc}', finished_unix=time.time())
        raise
    finally:
        write(output / 'progress.json', status)
        # A completed/failed attempt relinquishes only its own lock; a killed
        # process leaves the lock for explicit inspected recovery.
        (lock / 'owner.json').unlink()
        lock.rmdir()
    return status
