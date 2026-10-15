"""Native Plus initial scoring and independent-noise complete task evaluation.

Explicit pilot/production view IDs come from a frozen configuration. This uses
the private LIBERO policy bridge, never changes a camera during execution, and
validates the first observation against the completed native render cache.
"""
import argparse
from multiprocessing.connection import Client
from pathlib import Path
import tempfile
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores


def verify_example(example, cache, index):
    fields = {'external_images': 'observation/image', 'wrist_images': 'observation/wrist_image',
              'robot_states': 'observation/state', 'camera_intrinsics': 'camera_intrinsics',
              'camera_to_world_opencv': 'camera_to_world_opencv'}
    for saved, key in fields.items():
        value = np.asarray(example[key], np.float32) if saved == 'robot_states' else example[key]
        if not np.array_equal(cache[saved][index], value):
            raise ValueError('Native render/evaluation input mismatch: ' + key)


def policy_example(example, prompt):
    return dict(image=[example['observation/image'], example['observation/wrist_image']],
                lang=prompt, language=prompt, state=np.asarray(example['observation/state'], np.float32),
                camera_intrinsics=example['camera_intrinsics'], camera_to_world_opencv=example['camera_to_world_opencv'])


def simulator(config_file, initial, model, output, socket_path):
    config = read(config_file); output = Path(output)
    record = read(config['inputs'])['records'][initial]; spec = record['state']
    render_root = Path(config['render_root']) / f'{initial:02d}'
    receipt = read(render_root/'status.json')
    if (receipt['status'] != 'COMPLETE' or receipt['inputs_sha256'] != sha(render_root/'inputs.npz')
            or receipt['records_sha256'] != sha(render_root/'records.json')):
        raise ValueError('Unverified native render inputs')
    with np.load(render_root/'inputs.npz', allow_pickle=False) as z:
        cache = {k: z[k].copy() for k in z.files}
    variants = read(render_root/'records.json')['requests']
    if cache['views'].tolist() != ['canonical'] + [v['request_id'] for v in variants]:
        raise ValueError('Native Plus view IDs changed')
    with np.load(config['anchors'], allow_pickle=False) as z:
        score_noise = z['first_score_noise'][initial].copy()
    manifest = read(config['execution_noise_manifest'])
    if sha(config['execution_noise_file']) != manifest['noise_file_sha256']:
        raise ValueError('Execution noise bytes changed during path relocation')
    noise = np.load(config['execution_noise_file'], mmap_mode='r')
    if list(noise.shape) != manifest['shape'] or noise.dtype != np.float32:
        raise ValueError('Execution noise shape mismatch')
    state_index = manifest['state_keys'].index(spec['pair_key'])
    from scripts.dsol_paper1.runtime.evaluate_dsol_libero_hdf5_views import configure_imports, masked_policy_observation, MAX_STEPS_BY_SUITE
    from scripts.dsol_paper1.runtime.audit_libero_hdf5_restore import _configure_runtime
    from scripts.dsol_paper1.runtime.standard_initialization_v1 import initialize, load_initial
    from scripts.dsol_paper1.runtime.render_bridge_common_v1 import render_protocol
    configure_imports(); connection = Client(socket_path, family='AF_UNIX')
    begin = time.monotonic(); scores = np.empty((len(variants)+1, 8))
    # Same eight initial scoring noises and batch16 execution path as LIBERO.
    for start in range(0, len(scores), 16):
        stop = min(len(scores), start+16); examples = []
        for j in range(start, stop):
            example = {'observation/image': cache['external_images'][j],
                'observation/wrist_image': cache['wrist_images'][j], 'observation/state': cache['robot_states'][j],
                'camera_intrinsics': cache['camera_intrinsics'][j], 'camera_to_world_opencv': cache['camera_to_world_opencv'][j]}
            examples.append(policy_example(example, spec['prompt']))
        examples += [examples[-1]] * (16-len(examples))
        for repeat in range(8):
            connection.send(dict(kind='infer', examples=examples, noise=np.repeat(score_noise[repeat][None], 16, 0)))
            response = connection.recv()
            metric = compute_accel_scores(response['velocity'][:stop-start], prefixes=[3])[3]
            if metric['degenerate'].any():
                raise ValueError('Degenerate initial Accel denominator')
            scores[start:stop, repeat] = metric['score']
    scoring_seconds = time.monotonic()-begin
    indices = config.get('execution_view_indices', list(range(1, len(scores))))
    reused = config.get('reused_rollout_indices', {}).get(str(initial), [])
    indices = [j for j in indices if j not in reused]
    if reused:
        previous = Path(config['admission_root'])/model/f'{initial:02d}'
        prior_status = read(previous/'status.json')
        if prior_status['status'] != 'COMPLETE' or prior_status['arrays_sha256'] != sha(previous/'probes.npz'):
            raise ValueError('Reused admission is incomplete or changed')
        with np.load(previous/'probes.npz', allow_pickle=False) as prior:
            if not np.array_equal(scores, prior['scores']) or not set(reused) <= set(prior['execution_view_indices']):
                raise ValueError('Scoring or reused view differs from admission')
    if len(set(indices)) != len(indices) or any(j < 0 or j >= len(scores) for j in indices):
        raise ValueError('Invalid frozen execution view indices')
    outcomes = np.zeros((len(indices), 4), bool); rows = []
    with tempfile.TemporaryDirectory(prefix='dsol-plus-rollout-') as temporary:
        _configure_runtime(Path(config['runtime']), Path(config['dataset']), Path(temporary))
        from libero.libero.envs import OffScreenRenderEnv
        from evaluate_pi05_libero_plus_views import agentview_camera_calibration
        physical = load_initial(spec)
        with render_protocol(0) as contexts:
            for index, j in enumerate(indices):
                for repeat in range(4):
                    env = None; started = time.monotonic()
                    try:
                        bddl = spec['bddl_file'] if j == 0 else variants[j-1]['wrapper_bddl_argument']
                        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_names=('agentview', 'robot0_eye_in_hand'),
                            camera_heights=256, camera_widths=256, render_gpu_device_id=0)
                        initialize(env, spec, physical); env.env.sim.forward()
                        env.env._update_observables(force=True); obs = env.env._get_observations()
                        calibration = agentview_camera_calibration(env)
                        step = 0; call = 0; success = False; calls = []
                        while step < MAX_STEPS_BY_SUITE[spec['suite']] and not success:
                            example, _, _ = masked_policy_observation(obs, prompt=spec['prompt'], resize_size=224,
                                eval_seed=spec['environment_seed'], camera_calibration=calibration, sensor_control='both')
                            if call == 0:
                                verify_example(example, cache, j)
                            execution_noise = np.asarray(noise[state_index, repeat, call])
                            connection.send(dict(kind='infer', examples=[policy_example(example, spec['prompt'])],
                                noise=execution_noise[None]))
                            response = connection.recv(); actions = response['actions'][0]
                            from AlphaBrain.research.dsol.data.flow_noise import tensor_sha256
                            calls.append(dict(replan=call, noise_sha256=tensor_sha256(execution_noise),
                                              actions_sha256=tensor_sha256(np.asarray(actions, np.float32))))
                            call += 1
                            for action in actions[:5]:
                                obs, _, done, _ = env.step(np.clip(action, -1, 1))
                                step += 1; success = bool(done)
                                if success or step >= MAX_STEPS_BY_SUITE[spec['suite']]:
                                    break
                        final = agentview_camera_calibration(env)
                        if any(not np.array_equal(final[k], calibration[k]) for k in calibration):
                            raise ValueError('Native external camera moved during execution')
                        outcomes[index, repeat] = success
                        row = dict(initial=initial, model=model, pair_key=spec['pair_key'], view=str(cache['views'][j]),
                            repeat=repeat, success=success, steps=step, max_steps=MAX_STEPS_BY_SUITE[spec['suite']],
                            seconds=time.monotonic()-started, calls=calls, initial_input_exact=True)
                        rows.append(row)
                        write(output/f'episode-{index:03d}-{repeat}.json', row)
                        write(output/'progress.json', dict(completed_episodes=len(rows), total_episodes=len(indices)*4,
                            seconds=time.monotonic()-begin, scoring_seconds=scoring_seconds))
                    finally:
                        if env is not None:
                            env.close()
            if not contexts or set(contexts) != {0}:
                raise ValueError('Expected AA0 rendering')
    np.savez_compressed(output/'probes.npz', scores=scores, views=cache['views'], execution_view_indices=np.asarray(indices), success=outcomes)
    write(output/'episodes.json', dict(rows=rows, scoring_seconds=scoring_seconds,
        execution_noise_manifest_sha256=sha(config['execution_noise_manifest']), render_receipt_sha256=sha(render_root/'status.json')))
    connection.send(dict(kind='finish')); connection.recv(); connection.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--initial', type=int, required=True)
    parser.add_argument('--model', choices=['canonical', 'broad'], required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--socket')
    parser.add_argument('--execute', action='store_true'); args = parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute required')
    if args.socket:
        simulator(args.config, args.initial, args.model, args.output, args.socket)
    else:
        from .window_probe import run
        run(args.config, args.initial, args.model, args.output, simulator_module='benchmarks.LIBERO.plus_evaluate',
            completion_metadata=dict(scope='native_plus_initial_score_full_rollout', score_repeats=8,
                execution_repeats=4, simulator_sha256=sha(__file__)))
