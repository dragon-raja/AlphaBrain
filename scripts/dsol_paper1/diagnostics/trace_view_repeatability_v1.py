#!/usr/bin/env python3
"""Independent instrumented invocation of the unchanged frozen evaluator.

Diagnostic artifacts never enter the formal ledger. Real requests are repeated
against isolated policy services, or replayed from a captured fixed action tape.
"""
from __future__ import annotations

from pathlib import Path as _RepoPath
import sys as _sys
_REPOSITORY_ROOT = _RepoPath(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPOSITORY_ROOT))

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = (_REPOSITORY_ROOT / 'scripts/dsol_paper1')
sys.path.insert(0, str(HERE))
import scripts.dsol_paper1.runtime.evaluate_dsol_libero_hdf5_views as evaluator
evaluator.configure_imports()


def digest(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.tobytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def physical_arrays(env):
    sim = env.env.sim
    arrays = {'flat': np.asarray(env.get_sim_state()).copy()}
    for name in ('qpos', 'qvel', 'qacc', 'qacc_warmstart', 'ctrl', 'act',
                 'qfrc_applied', 'xfrc_applied', 'qfrc_bias', 'actuator_force'):
        value = getattr(sim.data, name, None)
        if value is not None:
            arrays[name] = np.asarray(value).copy()
    controller = env.env.robots[0].controller
    for name in ('goal_pos', 'goal_ori', 'ee_pos', 'ee_ori_mat', 'joint_pos', 'joint_vel', 'torques',
                 'initial_joint', 'new_update', 'mass_matrix', 'J_pos', 'J_ori', 'J_full'):
        value = getattr(controller, name, None)
        if value is not None:
            arrays['controller_' + name] = np.asarray(value).copy()
    arrays['gripper_command'] = np.asarray(env.env.robots[0].gripper.current_action).copy()
    for name in ('timestep', 'cur_time'):
        value = getattr(env.env, name, None)
        if value is not None: arrays['env_' + name] = np.asarray(value).copy()
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--ports', default='29600,29601')
    parser.add_argument('--render-gpu', type=int, default=0)
    parser.add_argument('--replay', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite diagnostic trace')
    args.output.mkdir(parents=True)
    runtime = Path('/share/longjunyu/alphabrain/datasets/libero-plus/runtime/LIBERO-plus')
    spec = json.loads(args.spec.read_text())
    replay_horizon = None
    if args.replay:
        replay_horizon = json.loads((args.replay/'episode.json').read_text())['completion_steps']
        # A diagnostic fixed-action tape ends with the captured source rollout.
        # Never fabricate additional actions if a replay has not yet succeeded.
        evaluator.MAX_STEPS_BY_SUITE = {**evaluator.MAX_STEPS_BY_SUITE, spec['suite']: replay_horizon}
    from scripts.dsol_paper1.runtime.audit_libero_hdf5_restore import _configure_runtime
    _configure_runtime(runtime, Path(spec['hdf5']).parent.parent, args.output/'libero-config')
    import libero.libero.envs as env_module
    original = env_module.OffScreenRenderEnv
    holder = {}
    step_rows = []

    def factory(*positional, **keywords):
        env = original(*positional, **keywords)
        holder['env'] = env
        original_step = env.step

        def step(action):
            index = len(step_rows)
            before = physical_arrays(env)
            result = original_step(action)
            after = physical_arrays(env)
            arrays = {'action': np.asarray(action).copy(),
                      **{'pre_' + k: v for k, v in before.items()},
                      **{'post_' + k: v for k, v in after.items()}}
            np.savez_compressed(args.output / f'step-{index:04d}.npz', **arrays)
            step_rows.append({'step': index, 'success': bool(result[2]),
                              'pre_flat_sha256': digest(before['flat']),
                              'post_flat_sha256': digest(after['flat'])})
            return result

        env.step = step
        return env

    env_module.OffScreenRenderEnv = factory
    clients = []
    if not args.replay:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        clients = [WebsocketClientPolicy(host='127.0.0.1', port=int(p)) for p in args.ports.split(',')]
    calls = []

    class Client:
        def infer(self, request):
            index = len(calls)
            arrays = {k: np.asarray(v).copy() for k, v in request.items() if isinstance(v, np.ndarray)}
            metadata = {k: v for k, v in request.items() if not isinstance(v, np.ndarray)}
            arrays.update({'physical_' + k: v for k, v in physical_arrays(holder['env']).items()})
            if args.replay:
                with np.load(args.replay / f'call-{index:04d}.npz', allow_pickle=False) as recorded:
                    if not np.array_equal(recorded['_eval_noise'], request['_eval_noise']):
                        raise ValueError('Replay noise identity changed')
                    action = recorded['actions_primary'].copy()
                response = {'actions': action, 'explicit_flow_noise': True,
                            'noise_sha256': digest(request['_eval_noise']),
                            'action_chunk_sha256': digest(action)}
                responses = [response]
            else:
                # The original evaluator uses only the first response. Repeats
                # are isolated tests on identical request bytes, not new replans.
                responses = [clients[0].infer(request), clients[0].infer(request), clients[1].infer(request)]
                response = responses[0]
            arrays['actions_primary'] = np.asarray(response['actions']).copy()
            for j, other in enumerate(responses):
                arrays[f'actions_probe_{j}'] = np.asarray(other['actions']).copy()
            np.savez_compressed(args.output / f'call-{index:04d}.npz', **arrays)
            comparisons = []
            primary = arrays['actions_primary']
            for j, other in enumerate(responses):
                current = np.asarray(other['actions'])
                comparisons.append({'probe': j, 'exact': primary.dtype == current.dtype and primary.shape == current.shape and digest(primary) == digest(current),
                                    'numeric_equal': bool(np.array_equal(primary, current)),
                                    'max_abs_error': float(np.max(np.abs(primary.astype(float)-current.astype(float))))})
            calls.append({'call': index, 'env_steps': len(step_rows), 'metadata': metadata,
                          'array_sha256': {k: digest(v) for k, v in arrays.items()},
                          'repeated_request_comparisons': comparisons})
            return response

    from AlphaBrain.research.dsol.data.flow_noise import ExplicitFlowNoiseBank
    env = None
    try:
        row, env = evaluator.run_episode(spec, runtime=runtime, config_root=args.output/'libero-config',
            client=Client(), replan_steps=5, wait_steps=0, resize_size=224,
            seed=20260818, save_video=False, video_dir=args.output/'unused-video',
            render_gpu=args.render_gpu, noise_bank=ExplicitFlowNoiseBank(args.bank, verify_file=True),
            require_explicit_noise=True)
        write_json(args.output/'episode.json', row)
        write_json(args.output/'calls.json', calls)
        write_json(args.output/'steps.json', step_rows)
        import mujoco, robosuite
        write_json(args.output/'completion.json', {'status': 'PASS_DIAGNOSTIC_TRACE_COMPLETE',
            'scientific_ledger': False, 'replay': str(args.replay) if args.replay else None,
            'replay_fixed_horizon': replay_horizon,
            'render_gpu': args.render_gpu, 'policy_calls': len(calls), 'env_steps': len(step_rows),
            'spec_sha256': hashlib.sha256(args.spec.read_bytes()).hexdigest(),
            'evaluator_sha256': hashlib.sha256(Path(evaluator.__file__).read_bytes()).hexdigest(),
            'modules': {'mujoco': mujoco.__file__, 'robosuite': robosuite.__file__, 'libero_envs': env_module.__file__},
            'server_metadata': [client.get_server_metadata() for client in clients],
            'actual_wrapper_noise': getattr(env, 'noise', None),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        print(json.dumps({'output': str(args.output), 'success': row['success'], 'calls': len(calls),
                          'duplicate_request_failures': sum(not p['exact'] for c in calls for p in c['repeated_request_comparisons'])}), flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
