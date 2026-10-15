"""Independent full-task evaluation of already frozen multi-window choices.

Native rollout implementations and explicit execution-noise addresses are reused.
This worker never searches candidates or updates choices using task outcomes.
"""
import argparse
import os
from pathlib import Path
import random
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.data.evaluation_matrix import NUMERIC_FLAGS


def validate_request(plan, selection, model, initial, views, repeats):
    if (model not in ('official', 'xr1') or selection['model'] != model
            or selection['initial'] != initial or initial not in plan['initial_ids']
            or not selection['all_97_candidates_verified'] or not selection['window1_exact']
            or selection['primary_window'] != 3 or selection['windows'] != [1, 2, 3, 4, 5]
            or selection['execution_repeats'] != list(range(8))):
        raise ValueError('Frozen choice identity or confirmation scope mismatch')
    if (not views or len(set(views)) != len(views)
            or not set(views) <= set(selection['evaluation_views'])
            or not repeats or len(set(repeats)) != len(repeats)
            or not set(repeats) <= set(range(8))):
        raise ValueError('Requested views/repeats are outside frozen confirmation')
    expected = {v for values in selection['conditions'].values() for v in values}
    if (set(selection['evaluation_views']) != expected or not 33 <= len(expected) <= 38
            or len(selection['conditions']['disturbance_uniform']) != 32
            or selection['conditions']['canonical'] != ['canonical']
            or any(len(selection['conditions'][f'window{k}']) != 1 for k in range(1, 6))):
        raise ValueError('Control coverage or selected-view deduplication changed')


def run(plan_file, model, initial, views, repeats, output, *, model_python=None, upstream=None):
    plan = read(plan_file)
    entry = plan['selections'][model][initial]
    if sha(entry['path']) != entry['sha256']:
        raise ValueError('Frozen choice file changed before execution')
    selected = read(entry['path'])
    validate_request(plan, selected, model, initial, views, repeats)
    config_file = plan['probe_config']
    if sha(config_file) != plan['probe_config_sha256']:
        raise ValueError('Model/probe configuration changed')
    cfg = read(config_file); population = read(cfg['panel_plan'])
    if sha(cfg['panel_plan']) != plan['probe_plan_sha256']:
        raise ValueError('Model or population identity changed')
    reference = read(population['reference_plan'])
    if (sha(population['reference_plan']) != population['reference_plan_sha256']
            or sha(population['reference_release']) != population['reference_release_sha256']):
        raise ValueError('Frozen observations changed')
    job = next(j for j in reference['jobs'] if j['initial_id'] == initial)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    status = dict(status='STARTING', model=model, initial=initial, pid=os.getpid(),
                  completed_rollouts=0, started_unix=time.time())
    write(output/'status.json', status)
    bridge = None
    try:
        import gymnasium as gym
        import robocasa  # noqa: F401
        from robocasa.utils.dataset_registry_utils import get_task_horizon
        from AlphaBrain.research.dsol.data.evaluation_admission import resolve_initial, load_candidate
        from .snapshot import inner_env
        from .snapshot_bundle import load_snapshot_bundle, reconstruct_and_restore
        from .evaluation_installation import restore_candidate_start, prepare_candidate_observation
        from .evaluation_policy import ResidentPolicy, policy_request
        from .evaluation_matrix import _rollout
        from .xr1_bridge import XR1Bridge
        from .xr1_evaluation import rollout as xr_rollout, compact_observation

        if model == 'official':
            if os.environ.get('XLA_FLAGS') != NUMERIC_FLAGS:
                raise ValueError('Original numerical flags required')
            spec = population['models']['official']
            for name, expected in spec['files'].items():
                stat = (Path(spec['checkpoint'])/name).stat()
                if stat.st_size != expected['bytes'] or stat.st_mtime_ns != expected['mtime_ns']:
                    raise ValueError('Official checkpoint changed since full audit')
            if sha(Path(spec['checkpoint'])/'assets/norm_stats.json') != spec['norm_sha256']:
                raise ValueError('Normalization changed')
            policy = ResidentPolicy(Path(spec['checkpoint']), spec['config'])
        else:
            spec = cfg['xr1_model']
            if sha(Path(spec['checkpoint'])/'IMPORT_RECEIPT.json') != spec['checkpoint_receipt_sha256']:
                raise ValueError('XR-1 checkpoint identity changed')
            bridge = XR1Bridge(model_python or spec['model_python'], spec['checkpoint'],
                               upstream or spec['upstream'], output/'model.log')
            policy = bridge
        write(output/'identity.json', dict(plan_sha256=sha(plan_file), model=model, initial=initial,
              selection_sha256=entry['sha256'], views=views, repeats=repeats,
              source_sha256=sha(__file__), noise_stream='execution',
              rollout_source_sha256=sha(Path(__file__).with_name(
                  'evaluation_matrix.py' if model == 'official' else 'xr1_evaluation.py'))))
        admitted = resolve_initial(population['reference_plan'], population['reference_release'], initial)
        bundle = load_snapshot_bundle(Path(job['path']))
        random.seed(job['scene_seed']); np.random.seed(job['scene_seed'])
        env = gym.make('robocasa/'+job['task'], split=job['split'], seed=job['scene_seed'], randomize_cameras=False)
        try:
            env.reset(); reconstruct_and_restore(env, bundle)
            if int(inner_env(env).sim._render_context_offscreen.con.offSamples) != 4:
                raise ValueError('Native rendering configuration changed')
            horizon = int(get_task_horizon(job['task']))
            if horizon != plan['task_horizons'][job['task']]:
                raise ValueError('Original task horizon changed')
            for view in views:
                candidate = load_candidate(admitted, view)
                restore_candidate_start(env, bundle, candidate)
                obs, checks = prepare_candidate_observation(env, bundle, candidate, population['pixel_mean_limit'])
                anchor_ref = selected['anchors'][view]
                if sha(anchor_ref['path']) != anchor_ref['sha256']:
                    raise ValueError('Initial anchor metadata changed')
                anchor = read(anchor_ref['path']); anchor_file = Path(anchor_ref['path']).with_suffix('.npz')
                if sha(anchor_file) != anchor['arrays_sha256']:
                    raise ValueError('Initial anchor arrays changed')
                obs['annotation.human.task_description'] = anchor['prompt']
                for name, image in zip(('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand'), candidate['images']):
                    obs['video.'+name] = image
                actual = (policy_request(obs) if model == 'official' else
                          dict(compact_observation(obs), prompt=anchor['prompt']))
                with np.load(anchor_file, allow_pickle=False) as arrays:
                    keys = [k for k in arrays.files if k.startswith(('observation/', 'state.', 'video.'))]
                    if set(keys) != set(actual)-{'prompt'}:
                        raise ValueError('Scoring/execution input keys differ')
                    if any(not np.array_equal(arrays[k], actual[k]) for k in keys):
                        raise ValueError('Initial scoring/execution inputs differ')
                cell = output/'cells'/initial/view; cell.mkdir(parents=True, exist_ok=False)
                write(cell/'render_checks.json', checks)
                for repeat in repeats:
                    if model == 'official':
                        trial = _rollout(env, bundle, candidate, policy, actual, initial, repeat, horizon)
                    else:
                        trial, traces = xr_rollout(env, bundle, candidate, policy, obs, initial,
                                                   anchor['prompt'], repeat, horizon)
                        np.savez_compressed(cell/f'trace_{repeat}.npz', velocity_trace=traces)
                        trial['trace_sha256'] = sha(cell/f'trace_{repeat}.npz')
                    trial.update(model=model, initial=initial, view=view, noise_stream='execution',
                                 selection_sha256=entry['sha256'])
                    write(cell/f'execution_{repeat}.json', trial)
                    status.update(status='RUNNING', view=view, repeat=repeat, checked_unix=time.time(),
                                  completed_rollouts=status['completed_rollouts']+1)
                    write(output/'status.json', status)
        finally:
            env.close()
        status.update(status='COMPLETE', finished_unix=time.time())
    except BaseException as exc:
        status.update(status='FAILED', error=repr(exc), finished_unix=time.time())
        raise
    finally:
        write(output/'status.json', status)
        if bridge is not None: bridge.close()
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'model', 'initial', 'output'): parser.add_argument('--'+name, required=True)
    parser.add_argument('--views', nargs='+', required=True)
    parser.add_argument('--repeats', nargs='+', type=int, required=True)
    parser.add_argument('--model-python'); parser.add_argument('--upstream')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute: parser.error('Explicit --execute required')
    run(args.plan, args.model, args.initial, args.views, args.repeats, args.output,
        model_python=args.model_python, upstream=args.upstream)
