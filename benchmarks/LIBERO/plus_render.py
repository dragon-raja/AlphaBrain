"""Native Plus camera rendering, with exact physical/installed-wrist controls.

Explicit command only; no policy inference, training, or simulation on import.
The official camera wrapper is used directly, not approximated by custom97.
"""
import argparse
from pathlib import Path
import tempfile
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, sha, write


def requests_for_initial(requests, record):
    state = record['state']
    selected = [r for r in requests if r['task_id'] == state['task_id']
                and r['physical_initial']['array_sha256'] == state['initialization']['array_sha256']]
    ids = [r['request_id'] for r in selected]
    if not selected or len(set(ids)) != len(ids):
        raise ValueError('Missing or duplicate native Plus requests')
    for row in selected:
        if row['resolved_base_bddl'] != state['bddl_file']:
            raise ValueError('Plus request changed base task')
    return selected


def render(config_file, initial, output):
    config = read(config_file)
    for path, digest in config['input_hashes'].items():
        if sha(path) != digest:
            raise ValueError('Prepared input changed: ' + path)
    inputs = read(config['inputs']); record = inputs['records'][initial]
    requests = requests_for_initial(read(config['requests'])['requests'], record)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write(output/'status.json', dict(status='RUNNING', initial=initial, scope='render_only'))
    from scripts.dsol_paper1.runtime.audit_libero_hdf5_restore import _configure_runtime
    from scripts.dsol_paper1.runtime.render_bridge_common_v1 import render_protocol, array_identity
    from scripts.dsol_paper1.runtime.standard_initialization_v1 import initialize, load_initial
    from scripts.dsol_paper1.runtime.evaluate_dsol_libero_hdf5_views import configure_imports, masked_policy_observation
    configure_imports()
    try:
        with tempfile.TemporaryDirectory(prefix='dsol-plus-native-') as temporary:
            _configure_runtime(Path(config['runtime']), Path(config['dataset']), Path(temporary))
            from libero.libero.envs import OffScreenRenderEnv
            from evaluate_pi05_libero_plus_views import agentview_camera_calibration, physics_state_sha256
            physical = load_initial(record['state'])
            observations = []; references = []; physics = None; wrist = None
            variants = [('canonical', record['state']['bddl_file'])] + [
                (r['request_id'], r['wrapper_bddl_argument']) for r in requests]
            with render_protocol(0) as contexts:
                for view, bddl in variants:
                    env = None; begin = time.monotonic()
                    try:
                        env = OffScreenRenderEnv(bddl_file_name=bddl,
                            camera_names=('agentview', 'robot0_eye_in_hand'),
                            camera_heights=256, camera_widths=256, render_gpu_device_id=0)
                        _, receipt = initialize(env, record['state'], physical)
                        # Match the original reference-camera restore barrier:
                        # refresh derived MuJoCo quantities before observations.
                        env.env.sim.forward()
                        env.env._update_observables(force=True)
                        obs = env.env._get_observations()
                        fingerprint = physics_state_sha256(env)
                        example, _, _ = masked_policy_observation(obs, prompt=record['state']['prompt'],
                            resize_size=224, eval_seed=record['state']['environment_seed'],
                            camera_calibration=agentview_camera_calibration(env), sensor_control='both')
                        # The archived cache and policy adapter both use float32.
                        example['observation/state'] = np.asarray(example['observation/state'], dtype=np.float32)
                        if view == 'canonical':
                            physics = fingerprint; wrist = example['observation/wrist_image'].copy()
                            if sha(record['observation_cache']['path']) != record['observation_cache']['sha256']:
                                raise ValueError('Original canonical cache changed')
                            with np.load(record['observation_cache']['path'], allow_pickle=False) as old:
                                for saved, key in [('external_images', 'observation/image'),
                                        ('wrist_images', 'observation/wrist_image'), ('robot_states', 'observation/state'),
                                        ('camera_intrinsics', 'camera_intrinsics'), ('camera_to_world_opencv', 'camera_to_world_opencv')]:
                                    if not np.array_equal(old[saved][0], example[key]):
                                        raise ValueError('Original canonical input mismatch: ' + key)
                        if fingerprint != physics or not np.array_equal(wrist, example['observation/wrist_image']):
                            raise ValueError('Native camera variant changed physical state or wrist observation')
                        if physics_state_sha256(env) != physics or getattr(env, 'noise', None) != 0:
                            raise ValueError('Rendering changed physics or wrapper observation noise')
                        observations.append(example)
                        references.append(dict(view=view, initialization=receipt,
                            physical_sha256=physics[0], seconds=time.monotonic()-begin,
                            inputs={k: array_identity(v) for k, v in example.items() if isinstance(v, np.ndarray)}))
                    finally:
                        if env is not None:
                            env.close()
                if not contexts or set(contexts) != {0}:
                    raise ValueError('AA0 rendering context was not applied')
            mapping = {'external_images': 'observation/image', 'wrist_images': 'observation/wrist_image',
                       'robot_states': 'observation/state', 'camera_intrinsics': 'camera_intrinsics',
                       'camera_to_world_opencv': 'camera_to_world_opencv'}
            np.savez_compressed(output/'inputs.npz', views=np.asarray([v for v, _ in variants]),
                **{k: np.stack([o[field] for o in observations]) for k, field in mapping.items()})
            write(output/'records.json', dict(initial=initial, pair_key=record['state']['pair_key'],
                prompt=record['state']['prompt'], requests=requests, records=references))
            result = dict(status='COMPLETE', initial=initial, requests=len(requests),
                config_sha256=sha(config_file), source_sha256=sha(__file__),
                inputs_sha256=sha(output/'inputs.npz'), records_sha256=sha(output/'records.json'),
                seconds=time.monotonic()-started, scope='render_only_no_policy_rollouts',
                canonical_exact=True, physical_exact=True, wrist_exact=True)
            write(output/'status.json', result)
            return result
    except BaseException as error:
        write(output/'status.json', dict(status='FAILED', initial=initial, error=repr(error),
            seconds=time.monotonic()-started, scope='render_only'))
        raise


def validate_output(config_file, initial, output):
    output = Path(output); result = read(output/'status.json')
    expected = requests_for_initial(read(read(config_file)['requests'])['requests'],
        read(read(config_file)['inputs'])['records'][initial])
    if (result['status'] != 'COMPLETE' or result['initial'] != initial
            or result['requests'] != len(expected) or result['config_sha256'] != sha(config_file)
            or result['source_sha256'] != sha(__file__)
            or result['inputs_sha256'] != sha(output/'inputs.npz')
            or result['records_sha256'] != sha(output/'records.json')
            or not all(result[k] for k in ['canonical_exact', 'physical_exact', 'wrist_exact'])):
        raise ValueError('Incomplete, stale, or invalid Plus rendering')
    with np.load(output/'inputs.npz', allow_pickle=False) as data:
        if data['views'].tolist() != ['canonical'] + [r['request_id'] for r in expected]:
            raise ValueError('Native Plus candidate axes changed')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--initial', type=int, required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute required')
    render(args.config, args.initial, args.output)
