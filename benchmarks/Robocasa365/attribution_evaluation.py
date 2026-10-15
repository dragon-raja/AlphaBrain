"""Supplemental evaluation reusing the exact P0 rollout/noise implementation."""
from pathlib import Path
import os
import platform
import random
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.data.diagnostic_panel import validate_diagnostic
from AlphaBrain.research.dsol.data.evaluation_admission import resolve_initial, load_candidate
from .evaluation_matrix import _rollout, _score_cell
from .evaluation_policy import ResidentPolicy, policy_request
from .evaluation_installation import prepare_candidate_observation
from .evaluation_render import pixel_check
from .snapshot import inner_env, restore_snapshot
from .snapshot_bundle import load_snapshot_bundle, reconstruct_and_restore, refresh_bundle_observation


def run(plan_file, worker_id, phase, output, *, pilot=False):
    plan = read(plan_file); p0, reference = validate_diagnostic(plan)
    if pilot and plan['format'] in ('robocasa_baseline_panel_v1', 'robocasa_checkpoint_4000_panel_v1', 'robocasa_official_view_space_v1'):
        raise ValueError('Bounded baseline has no separate pilot population')
    workers = [w for w in plan['workers'] if w['worker_id'] == worker_id]
    if len(workers) != 1 or phase not in workers[0]['phases']:
        raise ValueError('Unknown worker or phase')
    worker = workers[0]
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != str(worker['gpu'])
            or os.environ.get('XLA_FLAGS') != plan['numeric_flags']):
        raise ValueError('GPU or numerical configuration changed')
    if sha(plan['source_manifest']) != plan['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    for name, digest in read(plan['source_manifest']).items():
        if sha(Path(plan['source'])/name) != digest:
            raise ValueError('Frozen source changed: '+name)
    model = plan['models'][phase]; checkpoint = Path(model['checkpoint'])
    for name, expected in model['files'].items():
        stat = (checkpoint/name).stat()
        if stat.st_size != expected['bytes'] or stat.st_mtime_ns != expected['mtime_ns']:
            raise ValueError('Checkpoint changed since hash freeze')
    if sha(checkpoint/'assets/norm_stats.json') != model['norm_sha256']:
        raise ValueError('Normalization changed')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    identity = dict(plan_sha256=sha(plan_file), worker=worker_id, phase=phase, pilot=pilot)
    status = dict(**identity, status='STARTING', host=platform.node(), pid=os.getpid(),
                  completed_rollouts=0, completed_cells=0, started_unix=time.time(), rollout_seconds=0.)
    write(output/'progress.json', status)
    try:
        import gymnasium as gym
        import robocasa  # noqa: F401
        from robocasa.utils.dataset_registry_utils import get_task_horizon
        policy = ResidentPolicy(checkpoint, model['config'])
        if plan['format'] == 'robocasa_official_view_space_v1':
            # CPU/runtime migration gate: same frozen inputs/noise must reproduce
            # all eight historical canonical scoring traces before new rollouts.
            from AlphaBrain.research.dsol.data.evaluation_matrix import flow_noise
            initial = worker['initials'][0]
            old = Path(plan['baseline_plan']).parent/'workers'/worker_id/'official/cells'/initial/'canonical'
            meta = read(old/'scores.json')
            if sha(old/'scores.npz') != meta['arrays_sha256']:
                raise ValueError('Historical scoring payload changed')
            with np.load(old/'scores.npz', allow_pickle=False) as arrays:
                request = {k: arrays[k].copy() for k in arrays.files if k.startswith('observation/')}
                request['prompt'] = meta['prompt']
                for repeat in range(8):
                    trace = policy.trace(request, flow_noise(initial, 'score', repeat))
                    if (not np.array_equal(trace['velocity'][0], arrays['velocity_trace'][repeat])
                            or not np.array_equal(trace['actions'], arrays['actions'][repeat])):
                        raise ValueError('Migrated runtime does not exactly replay historical policy')
            write(output/'policy-replay-gate.json', dict(status='PASS', repeats=8,
                initial_id=initial, source=str(old/'scores.npz'), sha256=meta['arrays_sha256']))
        jobs = {j['initial_id']: j for j in reference['jobs']}
        initials = worker['pilot_initials'] if pilot else worker['initials']
        for initial in initials:
            job = jobs[initial]
            admission = resolve_initial(p0['reference_plan'], p0['reference_release'], initial)
            bundle = load_snapshot_bundle(Path(job['path']))
            random.seed(job['scene_seed']); np.random.seed(job['scene_seed'])
            env = gym.make('robocasa/'+job['task'], split=job['split'], seed=job['scene_seed'], randomize_cameras=False)
            try:
                env.reset(); reconstruct_and_restore(env, bundle)
                if int(inner_env(env).sim._render_context_offscreen.con.offSamples) != 4 or p0['policy_only_render']:
                    raise ValueError('Native P0 renderer required')
                horizon = int(get_task_horizon(job['task']))
                if horizon != p0['task_horizons'][job['task']]:
                    raise ValueError('Task horizon changed')
                obs, _ = refresh_bundle_observation(env, bundle)
                canonical_checks = {name: pixel_check(obs['video.'+name], image, reference['pixel_gate']['mean_abs'])
                                    for name, image in admission['images'].items()}
                if not all(v['pass'] for v in canonical_checks.values()):
                    raise ValueError('Initial render reference admission failed')
                if worker['lane'] == 'checkpoints':
                    views = [plan['panel_views'][0],plan['panel_views'][5]] if pilot else plan['panel_views']
                    repeats = plan['checkpoint_repeats']
                else:
                    views = list(dict.fromkeys(plan['choices'][model['parent_arm']][initial].values()))
                    repeats = plan['confirmation_repeats']
                for view in views:
                    cell = output/'cells'/initial/view; cell.mkdir(parents=True)
                    provenance = dict(**identity, initial_id=initial, view_id=view, checkpoint=model['checkpoint'])
                    candidate = load_candidate(admission, view)
                    restore_snapshot(env, bundle.snapshot)
                    obs, checks = prepare_candidate_observation(env, bundle, candidate, reference['pixel_gate']['mean_abs'])
                    if worker['lane'] == 'checkpoints':
                        request = policy_request(obs)
                        if plan.get('first_input_source') == 'frozen_p0_canonical_scores':
                            old_worker = next(w for w in p0['workers'] if w['arm'] == 'canonical' and initial in w['initials'])
                            old_cell = Path(plan['p0_plan']).parent/'workers'/old_worker['worker_id']/'cells'/initial/view
                            old_scores = read(old_cell/'scores.json')
                            if sha(old_cell/'scores.npz') != old_scores['arrays_sha256']:
                                raise ValueError('P0 first-input payload changed')
                            with np.load(old_cell/'scores.npz', allow_pickle=False) as arrays:
                                request = {k: arrays[k].copy() for k in arrays.files if k.startswith('observation/')}
                            request['prompt'] = old_scores['prompt']
                            write(cell/'input-source.json', dict(path=str(old_cell/'scores.json'), sha256=sha(old_cell/'scores.json')))
                        request, score_meta = _score_cell(policy, request, initial, cell, provenance)
                    else:
                        arm = model['parent_arm']
                        old_worker = next(w for w in p0['workers'] if w['arm'] == arm and initial in w['initials'])
                        old_cell = Path(plan['p0_plan']).parent/'workers'/old_worker['worker_id']/'cells'/initial/view
                        score_meta = read(old_cell/'scores.json')
                        if sha(old_cell/'scores.npz') != score_meta['arrays_sha256']:
                            raise ValueError('P0 observation changed')
                        with np.load(old_cell/'scores.npz', allow_pickle=False) as arrays:
                            request = {k: arrays[k].copy() for k in arrays.files if k.startswith('observation/')}
                        request['prompt'] = score_meta['prompt']
                        write(cell/'score-source.json', dict(path=str(old_cell/'scores.json'), sha256=sha(old_cell/'scores.json')))
                    trials = []
                    for repeat in repeats:
                        trial = _rollout(env, bundle, candidate, policy, request, initial, repeat, horizon)
                        trial.update(provenance=provenance)
                        path = cell/f'execution_{repeat}.json'; write(path, trial)
                        trials.append(dict(file=path.name, sha256=sha(path), success=trial['success'], seconds=trial['seconds']))
                        status['completed_rollouts'] += 1; status['rollout_seconds'] += trial['seconds']
                        status.update(status='RUNNING', initial_id=initial, view_id=view, task=job['task'], heartbeat_unix=time.time())
                        write(output/'progress.json', status)
                    write(cell/'complete.json', dict(provenance=provenance, trials=trials, candidate_pixels=checks,
                          canonical_pixels=canonical_checks, score_arrays_sha256=score_meta['arrays_sha256'], complete=True))
                    status['completed_cells'] += 1
            finally:
                env.close()
        status.update(status='COMPLETE', finished_unix=time.time())
    except Exception as exc:
        status.update(status='ERROR', error=f'{type(exc).__name__}: {exc}', finished_unix=time.time())
        raise
    finally:
        write(output/'progress.json', status)
    return status
