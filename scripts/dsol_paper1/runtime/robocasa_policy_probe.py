"""Explicit RoboCasa real-input/Accel gold check and optional full-task debug.

This is an engineering probe, not a frozen scientific evaluation population.
It uses installed RoboCasa/OpenPI directly, with no archival script imports.
"""

import argparse
from collections import deque
import dataclasses
import hashlib
import json
from pathlib import Path
import random
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', default='pi05_pretrain_human300_all_lora_fast25')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', default='CloseBlenderLid')
    parser.add_argument('--split', default='pretrain')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--snapshot', type=Path, help='Trusted local checksummed bundle; required for any formal paired evaluation')
    parser.add_argument('--reference-plan', type=Path, help='Explicit frozen current-render reference plan')
    parser.add_argument('--reference-release', type=Path, help='Completed all-initial current-reference release')
    parser.add_argument('--initial-id', help='Initial ID within the released plan; derives task/split/seed/snapshot')
    parser.add_argument('--view-id', help='Released fixed local installation; requires the reference release')
    parser.add_argument('--confirmed-pixel-gate', type=Path, help='Explicit independently confirmed numerical admission profile')
    parser.add_argument('--full-rollout', action='store_true')
    parser.add_argument('--record-replay', action='store_true', help='Save per-replan inputs/actions and per-step physical hashes')
    parser.add_argument('--action-replay', type=Path, help='Diagnostic ONLY: replay saved actions instead of the policy')
    parser.add_argument('--policy-only-render', action='store_true', help='Diagnostic optimization: skip RGB renders not consumed by a replan')
    parser.add_argument('--diagnostic-msaa-samples', type=int, choices=(0, 2, 4),
                        help='Diagnostic ONLY: change framebuffer sampling after native reference admission')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute required')
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.record_replay and not args.full_rollout:
        parser.error('Replay recording requires --full-rollout')
    if args.action_replay and not (args.full_rollout and args.record_replay):
        parser.error('Action replay requires --full-rollout and --record-replay')
    if args.policy_only_render and not args.full_rollout:
        parser.error('Policy-only rendering requires --full-rollout')
    if args.diagnostic_msaa_samples is not None and not (args.full_rollout and args.record_replay):
        parser.error('Experimental MSAA requires full rollout and replay recording')
    admission = None
    if any((args.reference_plan, args.reference_release, args.initial_id)):
        if not all((args.reference_plan, args.reference_release, args.initial_id)):
            parser.error('Reference plan, release and initial ID are required together')
        if args.snapshot or args.confirmed_pixel_gate:
            parser.error('Released references define the snapshot and pixel gate; no overrides')
        from AlphaBrain.research.dsol.data.evaluation_admission import resolve_initial
        admission = resolve_initial(args.reference_plan, args.reference_release, args.initial_id)
        job = admission['job']
        args.snapshot, args.task, args.split, args.seed = Path(job['path']), job['task'], job['split'], job['scene_seed']
    pixel_gate = None
    if admission is not None:
        from AlphaBrain.research.dsol.data.render_reference import LIMITS
        pixel_gate = dict(LIMITS)
    if args.confirmed_pixel_gate:
        from AlphaBrain.research.dsol.data.render_admission import confirmed_pixel_gate
        pixel_gate = confirmed_pixel_gate(args.confirmed_pixel_gate)
    candidate = None
    if args.view_id:
        if admission is None:
            parser.error('Explicit candidate requires a completed reference release')
        from AlphaBrain.research.dsol.data.evaluation_admission import load_candidate
        candidate = load_candidate(admission, args.view_id)

    import gymnasium as gym
    import jax
    import jax.numpy as jnp
    import numpy as np
    import robocasa  # registers environments
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action
    from openpi.policies import policy_config
    from openpi.training import config as training_config
    from openpi.shared import normalize
    from openpi.models import model as model_api
    from openpi_client import image_tools
    from AlphaBrain.research.dsol.metrics.pi05_jax import make_velocity_sampler
    from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores
    from AlphaBrain.research.dsol.data.artifacts import write, sha
    from benchmarks.Robocasa365.evaluation_render import pixel_check

    started = time.monotonic()
    cfg = training_config.get_config(args.config)
    cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, data_dirs=[]))
    policy = policy_config.create_trained_policy(cfg, args.checkpoint,
        norm_stats=normalize.load(args.checkpoint / 'assets'), sample_kwargs={'num_steps': 10})
    random.seed(args.seed)
    np.random.seed(args.seed)
    env = gym.make(f'robocasa/{args.task}', split=args.split, seed=args.seed, randomize_cameras=False)
    result = dict(scope='Engineering diagnostic, not a success-rate estimate or historical snapshot replay',
        checkpoint=str(args.checkpoint), config=args.config, task=args.task, split=args.split,
        scene_seed=args.seed, python_numpy_seed=args.seed, flow_steps=10, replan_steps=25,
        sampler_source_sha256=sha(Path(model_api.__file__).with_name('pi0.py')),
        adapter_sha256=sha(Path(__file__)), norm_sha256=sha(args.checkpoint / 'assets/norm_stats.json'))
    result['pixel_gate'] = pixel_gate or dict(mean_abs=.002, max_abs=2, fraction_gt1=.0001, fraction_gt2=0.)
    if admission is not None:
        result['current_reference_admission'] = dict(initial_id=args.initial_id,
            plan_sha256=admission['plan_sha256'], release_sha256=admission['release_sha256'],
            reference_sha256=admission['record']['reference_sha256'],
            validation_sha256=admission['record']['validation_sha256'])
    try:
        obs, _ = env.reset()
        if args.snapshot:
            from benchmarks.Robocasa365.snapshot_bundle import load_snapshot_bundle, reconstruct_and_restore, refresh_bundle_observation, env_numeric_fingerprint
            bundle = load_snapshot_bundle(args.snapshot)
            metadata = bundle.manifest['metadata']
            if (metadata['task'], metadata['split'], metadata['scene_seed']) != (args.task, args.split, args.seed):
                raise ValueError('CLI task/split/seed differ from persistent snapshot')
            reconstruct_and_restore(env, bundle)
            result['snapshot'] = str(args.snapshot)
            result['snapshot_manifest_sha256'] = sha(args.snapshot / 'manifest.json')
            result['restored_numeric_sha256'] = env_numeric_fingerprint(env)
            obs, _ = refresh_bundle_observation(env, bundle)
            pixel_checks = {}
            for key, reference in (bundle.canonical_images or {}).items():
                check = pixel_check(obs[key], reference, result['pixel_gate']['mean_abs'])
                check['within_tolerance'] = check['pass']
                check['legacy_within_tolerance'] = pixel_check(obs[key], reference)['pass']
                pixel_checks[key] = check
            result['snapshot_pixels'] = pixel_checks
            admission_checks = pixel_checks
            if admission is not None:
                admission_checks = {'video.' + key: pixel_check(obs['video.' + key], value,
                    result['pixel_gate']['mean_abs']) for key, value in admission['images'].items()}
                result['current_reference_pixels'] = admission_checks
            if len(admission_checks) != 3 or not all(x['pass'] for x in admission_checks.values()):
                raise RuntimeError('Persistent snapshot image tolerance gate failed')
            if candidate is not None:
                from benchmarks.Robocasa365.evaluation_installation import prepare_candidate_observation
                obs, result['candidate_pixels'] = prepare_candidate_observation(
                    env, bundle, candidate, result['pixel_gate']['mean_abs'])
                result['candidate'] = dict(candidate['view'], geometry_sha256=candidate['geometry_sha256'],
                    semantics='fixed_support_local_installation_native_body_motion')
        if args.diagnostic_msaa_samples is not None:
            if admission is None:
                raise ValueError('Experimental rendering requires the original released reference for comparison')
            from benchmarks.Robocasa365.snapshot import inner_env
            original_images = {k: v.copy() for k, v in obs.items() if k.startswith('video.')}
            sim = inner_env(env).sim
            context = sim._render_context_offscreen
            context.gl_ctx.make_current()
            original_samples = int(context.con.offSamples)
            sim.model.vis.quality.offsamples = args.diagnostic_msaa_samples
            context.con.free()
            context._set_mujoco_context_and_buffers()
            obs, _ = refresh_bundle_observation(env, bundle)
            result['experimental_renderer'] = dict(original_samples=original_samples,
                actual_samples=int(context.con.offSamples), approved_for_formal_evaluation=False,
                native_image_differences={k: pixel_check(obs[k], v) for k, v in original_images.items()},
                interpretation='Changed images are diagnostic data, not admitted original-reference images')
        prompt = obs['annotation.human.task_description']

        def request(observation):
            data = {'prompt': prompt}
            for dest, source in [('image', 'robot0_agentview_left'), ('right_image', 'robot0_agentview_right'),
                                 ('wrist_image', 'robot0_eye_in_hand')]:
                data['observation/' + dest] = image_tools.convert_to_uint8(image_tools.resize_with_pad(
                    np.ascontiguousarray(observation['video.' + source]), 224, 224))
            data['observation/state'] = np.concatenate([observation['state.' + k] for k in
                ('end_effector_position_relative', 'end_effector_rotation_relative',
                 'base_position', 'base_rotation', 'gripper_qpos')])
            return data

        req = request(obs)
        result['initial_input'] = {k: dict(shape=list(v.shape), dtype=str(v.dtype),
            sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k, v in req.items() if isinstance(v, np.ndarray)}
        result['prompt'] = prompt
        if args.snapshot:
            result['scope'] = 'Engineering diagnostic with persisted physical initial state; not a success-rate estimate'
        if args.diagnostic_msaa_samples is not None:
            result['scope'] = 'Experimental MSAA full-task replay; NOT a released evaluation or success-rate estimate'
        inputs = policy._input_transform(jax.tree.map(lambda x: x, req))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        observation = model_api.Observation.from_dict(inputs)
        sampler = make_velocity_sampler(policy._model)
        checks, traces = [], []
        for seed in (916001, 916002):
            noise = np.random.default_rng(seed).standard_normal((50, 32), dtype=np.float32)
            reference = np.asarray(policy.infer(req, noise=noise)['actions'])
            traced = sampler(jax.random.key(0), observation, noise=jnp.asarray(noise)[None])
            trace = np.asarray(traced['velocity_trace'])
            transformed = policy._output_transform({'state': np.asarray(inputs['state'][0]),
                'actions': np.asarray(traced['actions'][0])})['actions']
            delta = float(np.max(np.abs(reference - transformed)))
            scores = compute_accel_scores(trace)
            check = dict(seed=seed, normalized_trace_shape=list(trace.shape),
                step_count=int(traced['step_count']), times=np.asarray(traced['times']).tolist(),
                actions_max_abs_diff=delta, actions_match=bool(np.allclose(reference, transformed, rtol=1e-5, atol=1e-5)),
                accel_3=float(scores[3]['score'][0]), degenerate=bool(scores[3]['degenerate'][0]),
                noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest())
            checks.append(check)
            traces.append(trace[0])
            print(json.dumps(check), flush=True)
        result['gold_checks'] = checks
        result['status'] = 'PASS' if all(c['actions_match'] and c['step_count'] == 10 and not c['degenerate'] for c in checks) else 'HOLD'
        if result['status'] != 'PASS':
            raise RuntimeError('Instrumented sampler failed endpoint golden check')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        trace_path = args.output.with_suffix('.npz')
        if trace_path.exists():
            raise FileExistsError(trace_path)
        with trace_path.open('xb') as stream:
            np.savez_compressed(stream, velocity_trace=np.stack(traces))
        result['trace_file'] = str(trace_path)
        result['trace_sha256'] = sha(trace_path)
        if args.full_rollout:
            from benchmarks.Robocasa365.snapshot_bundle import env_numeric_fingerprint
            recorder = None
            action_replay = None
            if args.record_replay:
                from AlphaBrain.research.dsol.data.rollout_replay import ReplayRecorder
                recorder = ReplayRecorder()
            if args.action_replay:
                from AlphaBrain.research.dsol.data.rollout_replay import ActionReplay
                action_replay = ActionReplay(args.action_replay)
                result['scope'] = 'Diagnostic fixed-action replay, NOT live policy evaluation'
                result['action_replay_source'] = str(args.action_replay)
                result['action_replay_sha256'] = sha(args.action_replay)
            if args.snapshot:
                # Sensor refresh may alter solver warm-start. Restore the exact
                # physical/controller/RNG start after diagnostics; cached obs is
                # from that same physical state and remains the first input.
                from benchmarks.Robocasa365.snapshot import restore_snapshot
                if candidate is None:
                    restore_snapshot(env, bundle.snapshot)
                    result['pre_rollout_numeric_sha256'] = env_numeric_fingerprint(env)
                    if result['pre_rollout_numeric_sha256'] != bundle.manifest['snapshot_numeric_sha256']:
                        raise RuntimeError('Pre-rollout numeric restore mismatch')
                else:
                    from benchmarks.Robocasa365.evaluation_installation import restore_candidate_start, check_fixed_installation
                    result['pre_rollout_numeric_sha256'] = restore_candidate_start(env, bundle, candidate)
            horizon = int(get_task_horizon(args.task))
            render_schedule = None
            if args.policy_only_render:
                from benchmarks.Robocasa365.render_schedule import PolicyRenderSchedule
                from benchmarks.Robocasa365.snapshot import inner_env
                render_schedule = PolicyRenderSchedule(inner_env(env).sim)
            queue, calls, success = deque(), 0, False
            inference_seconds, environment_seconds = 0., 0.
            action_digest = hashlib.sha256()
            rollout_start = time.monotonic()
            for step in range(horizon):
                if not queue:
                    # Explicit step-indexed noise; scoring seeds are disjoint.
                    noise = np.random.default_rng(917000 + calls).standard_normal((50, 32), dtype=np.float32)
                    inference_start = time.monotonic()
                    current_request = request(obs)
                    current_actions = (policy.infer(current_request, noise=noise)['actions'] if action_replay is None
                                       else action_replay.plan(calls, step, noise))
                    queue.extend(current_actions[:25])
                    inference_seconds += time.monotonic() - inference_start
                    if recorder is not None:
                        recorder.record_plan(step, current_request, noise, current_actions, env_numeric_fingerprint(env))
                    calls += 1
                action = queue.popleft()
                action_digest.update(np.ascontiguousarray(action).tobytes())
                environment_start = time.monotonic()
                if render_schedule is not None:
                    render_schedule.before_step(step, 25)
                obs, _, terminated, truncated, info = env.step(convert_action(action))
                if candidate is not None:
                    check_fixed_installation(env, candidate)
                environment_seconds += time.monotonic() - environment_start
                success = bool(info['success'])
                if recorder is not None:
                    recorder.record_step(step, action, env_numeric_fingerprint(env), success)
                if success or terminated or truncated:
                    break
            result['rollout'] = dict(success=success, steps=step + 1, horizon=horizon, policy_calls=calls,
                seconds=time.monotonic() - rollout_start, terminated=bool(terminated), truncated=bool(truncated),
                inference_seconds=inference_seconds, environment_seconds=environment_seconds,
                executed_actions_sha256=action_digest.hexdigest(),
                noise_seed_base=917000, complete_task_attempt=True,
                interpretation='One engineering trajectory; no success-rate or paired-camera claim')
            if recorder is not None:
                result['replay_file'] = recorder.save(args.output.with_suffix('.replay'))
            if render_schedule is not None:
                result['render_schedule'] = dict(rendered=render_schedule.rendered, reused=render_schedule.reused,
                    scope='only policy-consumed RGB is fresh; every physical/control/success step retained')
                render_schedule.close()
    except Exception as exc:
        result['status'] = 'ERROR'
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        env.close()
        result['seconds'] = time.monotonic() - started
        write(args.output, result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
