"""Five native LIBERO replans, batch16 anchor, distinct scoring noise stream.

An isolated Python3.8 simulator talks to the Python3.12 policy over a private
local socket. Candidate short trajectories reset before each scoring repeat.
This is not a deployable rollback search or a new complete-task evaluation.
"""
import argparse
from multiprocessing.connection import Client, Listener
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.data.flow_noise import stable_uint64
from AlphaBrain.research.dsol.metrics.accel import shared_flow_noise, compute_accel_scores


def later_noise(pair_key, repeat, replan):
    if not 0 <= repeat < 8 or not 1 <= replan <= 4:
        raise ValueError('Only the frozen five-window eight-repeat scoring stream')
    seed = stable_uint64(f'libero-window-score::{pair_key}::{repeat}::{replan}', root_seed=20261012)
    return shared_flow_noise(seed=seed, candidate_count=1, action_horizon=10, action_dim=7)[0]


def simulator(config_file, initial, model, output, socket_path):
    config = read(config_file); inputs = read(config['inputs']); record = inputs['records'][initial]
    from scripts.dsol_paper1.runtime.audit_libero_hdf5_restore import _configure_runtime
    from scripts.dsol_paper1.runtime.evaluate_dsol_libero_hdf5_views import configure_imports, masked_policy_observation
    from scripts.dsol_paper1.runtime.standard_initialization_v1 import load_initial
    from .initialization import initialize, capture_model_reference
    from scripts.dsol_paper1.runtime.render_bridge_common_v1 import render_protocol
    from scripts.dsol_paper1.runtime.scan_libero_hdf5_views import _restore_reference
    configure_imports(); connection = Client(socket_path, family='AF_UNIX')
    with tempfile.TemporaryDirectory(prefix='dsol-libero-probe-sim-') as temporary:
        _configure_runtime(Path(config['runtime']), Path(config['dataset']), Path(temporary))
        from libero.libero.envs import OffScreenRenderEnv
        from libero_camera_pose import capture_camera_reference, install_camera_pose
        from evaluate_pi05_libero_plus_views import agentview_camera_calibration
        spec = record['state']; physical = load_initial(spec)
        native_plus = config.get('candidate_source') == 'native_plus'
        if native_plus:
            from .plus_windows import load_panel, initial_scores, verify_absolute_endpoints
            cached, variants = load_panel(config, initial)
        else:
            with np.load(record['observation_cache']['path'], allow_pickle=False) as z:
                cached = {k: z[k].copy() for k in z.files}
        with np.load(config['anchors'], allow_pickle=False) as z:
            first_noise = z['first_score_noise'][initial].copy()
            expected_scores = None if native_plus else z['member_scores'][initial, ['canonical', 'broad'].index(model)].copy()
        release = read(config['release'])
        candidates = release['protocol']['state_blocks'][initial]['candidates']
        if not native_plus and (len(candidates) != 97 or cached['candidate_ids'].tolist() != record['candidate_ids']):
            raise ValueError('Original candidate axes changed')
        count = len(cached['candidate_ids'])
        if native_plus:
            expected_scores = initial_scores(connection, cached, first_noise, spec['prompt'])
        scores = np.empty((count, 8, 5)); velocities = np.empty((count, 8, 5, 10, 10, 7), np.float32)
        actions = np.empty((count, 8, 5, 10, 7), np.float32); first_success = np.full((count, 8), -1, int)
        started = time.monotonic()
        with render_protocol(0) as contexts:
            if native_plus and model == 'canonical':
                verify_absolute_endpoints(config, dict(record, initial_index=initial), variants, output)
            for start in range(0, count, 16):
                stop = min(count, start+16); envs = []; refs = []; model_refs = []
                try:
                    for j in range(start, stop):
                        bddl = variants[j-1]['wrapper_bddl_argument'] if native_plus and j > 0 else spec['bddl_file']
                        env = OffScreenRenderEnv(bddl_file_name=bddl,
                            camera_names=('agentview', 'robot0_eye_in_hand'), camera_heights=256,
                            camera_widths=256, render_gpu_device_id=0)
                        envs.append(env); initialize(env, spec, physical)
                        model_refs.append(capture_model_reference(env))
                        refs.append(None if native_plus else capture_camera_reference(env, camera_name='agentview', table_plane_z=config['table_plane_z']))
                    for repeat in range(8):
                        observations = []; calibrations = []
                        for offset, env in enumerate(envs):
                            initialize(env, spec, physical, model_reference=model_refs[offset])
                            if native_plus:
                                env.env.sim.forward()
                            else:
                                _restore_reference(env, refs[offset])
                                pose = candidates[start+offset]['pose']
                                if pose is not None:
                                    install_camera_pose(env, refs[offset], pose)
                            env.env._update_observables(force=True)
                            observations.append(env.env._get_observations())
                            calibrations.append(agentview_camera_calibration(env))
                        for window in range(5):
                            examples = []
                            for offset, obs in enumerate(observations):
                                example, _, _ = masked_policy_observation(obs, prompt=spec['prompt'],
                                    resize_size=224, eval_seed=spec['environment_seed'],
                                    camera_calibration=calibrations[offset], sensor_control='both')
                                example['observation/state'] = np.asarray(example['observation/state'], np.float32)
                                if window == 0:
                                    for saved, key in [('external_images', 'observation/image'), ('wrist_images', 'observation/wrist_image'),
                                            ('robot_states', 'observation/state'), ('camera_intrinsics', 'camera_intrinsics'),
                                            ('camera_to_world_opencv', 'camera_to_world_opencv')]:
                                        if not np.array_equal(cached[saved][start+offset], example[key]):
                                            raise ValueError('First policy input differs from original cache: ' + key)
                                examples.append(dict(image=[example['observation/image'], example['observation/wrist_image']],
                                    lang=spec['prompt'], language=spec['prompt'], state=example['observation/state'],
                                    camera_intrinsics=example['camera_intrinsics'], camera_to_world_opencv=example['camera_to_world_opencv']))
                            examples += [examples[-1]]*(16-len(examples))
                            noise = first_noise[repeat] if window == 0 else later_noise(spec['pair_key'], repeat, window)
                            connection.send(dict(kind='infer', examples=examples, noise=np.repeat(noise[None], 16, 0)))
                            response = connection.recv()
                            if response.get('error'):
                                raise RuntimeError(response['error'])
                            trace = response['velocity'][:stop-start]; chunk = response['actions'][:stop-start]
                            score = compute_accel_scores(trace, prefixes=[3])[3]
                            if score['degenerate'].any():
                                raise ValueError('Degenerate Accel denominator')
                            if window == 0 and not np.array_equal(score['score'], expected_scores[start:stop, repeat]):
                                raise ValueError('Window1 does not exactly reproduce original batch16 scores')
                            scores[start:stop, repeat, window] = score['score']
                            velocities[start:stop, repeat, window] = trace
                            actions[start:stop, repeat, window] = chunk
                            if window < 4:
                                for offset, env in enumerate(envs):
                                    for step, action in enumerate(chunk[offset, :5]):
                                        obs, _, done, _ = env.step(np.clip(action, -1, 1))
                                        if done and first_success[start+offset, repeat] < 0:
                                            first_success[start+offset, repeat] = window*5+step+1
                                    observations[offset] = obs
                            write(Path(output)/'progress.json', dict(initial=initial, model=model,
                                completed_candidate_blocks=start//16, repeat=repeat, window=window+1,
                                seconds=time.monotonic()-started))
                finally:
                    for env in envs:
                        env.close()
            if not contexts or set(contexts) != {0}:
                raise ValueError('Original AA0 rendering was not retained')
        np.savez_compressed(Path(output)/'probes.npz', scores=scores, velocities=velocities,
            actions=actions, first_success=first_success, views=cached['candidate_ids'], initial_reference_scores=expected_scores)
        connection.send(dict(kind='finish')); connection.recv(); connection.close()


def run(config_file, initial, model, output, *, simulator_module=None, completion_metadata=None):
    """Run the private policy bridge; native Plus may supply its simulator.

    This compatibility hook shares checkpoint loading and process cleanup, not
    the two experiments' scientific protocol or result validation.
    """
    config = read(config_file); output = Path(output); output.mkdir(parents=True, exist_ok=False)
    for path, digest in config['input_hashes'].items():
        if sha(path) != digest:
            raise ValueError('Prepared probe input changed: ' + path)
    started = time.monotonic(); child = None
    write(output/'status.json', dict(status='LOADING', initial=initial, model=model))
    try:
        import torch
        from AlphaBrain.model.framework.base_framework import BaseFramework
        from AlphaBrain.research.dsol.metrics.accel_inference import _validated_trace_output
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        checkpoint = config['models'][model]
        if sha(Path(checkpoint['path'])/'model.safetensors') != checkpoint['weights_sha256']:
            raise ValueError('Checkpoint bytes changed')
        policy = BaseFramework.from_pretrained(checkpoint['path'], strict_checkpoint=True).to(torch.bfloat16).to('cuda').eval()
        policy.gripper_remap = False
        for p in policy.parameters():
            p.requires_grad_(False)
        with tempfile.TemporaryDirectory(prefix='dsol-libero-policy-') as tmp:
            address = str(Path(tmp)/'policy.sock')
            with Listener(address, family='AF_UNIX') as listener:
                # A private 0700 directory prevents untrusted local clients.
                environment = {**os.environ, 'PYTHONPATH': ':'.join([config['sim_site'], config['source'],
                    config['source']+'/scripts/vla_shared']+config['sim_extra_paths']),
                    'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl'}
                command = [config['sim_python'], '-u', '-m', simulator_module or 'benchmarks.LIBERO.window_probe', '--config', str(config_file),
                    '--initial', str(initial), '--model', model, '--output', str(output), '--socket', address, '--execute']
                with (output/'simulator.log').open('xb') as log:
                    child = subprocess.Popen(command, env=environment, cwd=config['source'], stdout=log, stderr=subprocess.STDOUT)
                    listener._listener._socket.settimeout(120)
                    with listener.accept() as connection:
                        while True:
                            if not connection.poll(1):
                                if child.poll() is not None:
                                    raise RuntimeError('Simulator exited without a complete probe; inspect simulator.log')
                                if time.monotonic()-started > config['worker_timeout_seconds']:
                                    raise TimeoutError('Explicit probe worker deadline')
                                continue
                            request = connection.recv()
                            if request['kind'] == 'finish':
                                connection.send(dict(ok=True)); break
                            if request['kind'] != 'infer':
                                raise ValueError('Unknown local policy request')
                            batch = len(request['examples'])
                            if batch not in (1, 16):
                                raise ValueError('Only admitted scoring16 or execution1 batches')
                            with torch.inference_mode():
                                result = policy.predict_action(examples=request['examples'], noise=request['noise'], return_flow_trace=True)
                            trace, times, noise, actions = _validated_trace_output(result, request['noise'],
                                candidate_count=batch, action_horizon=10, action_dim=7)
                            connection.send(dict(velocity=trace, actions=actions))
                    if child.wait(timeout=60) != 0:
                        raise RuntimeError('Simulator failed after final policy call')
        with np.load(output/'probes.npz', allow_pickle=False) as arrays:
            candidate_count = len(arrays['views'])
        metadata = completion_metadata or dict(windows=[1, 2, 3, 4, 5],
            candidate_count=candidate_count, score_repeats=8, batch_size=16, replan_steps=5,
            window1_exact=True, scope='candidate_probe_then_reset_not_online_correction', new_complete_rollouts=0)
        result = dict(**metadata, status='COMPLETE', initial=initial, model=model,
            config_sha256=sha(config_file), source_sha256=sha(__file__),
            arrays_sha256=sha(output/'probes.npz'), seconds=time.monotonic()-started,
            simulator_module=simulator_module or __name__)
        write(output/'status.json', result)
        return result
    except BaseException as error:
        write(output/'status.json', dict(status='FAILED', error=repr(error), initial=initial, model=model,
            config_sha256=sha(config_file), seconds=time.monotonic()-started))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()


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
        run(args.config, args.initial, args.model, args.output)
