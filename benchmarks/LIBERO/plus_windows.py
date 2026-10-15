"""Native Plus panel support for the shared five-window probe engine.

No full rollouts are performed here. Initial scores use exactly the native
Plus evaluator's cached-input batch16 path; a later audit compares both files.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores
from .plus_evaluate import policy_example


def load_panel(config, initial):
    root = Path(config['render_root']) / f'{initial:02d}'
    status = read(root/'status.json')
    if (status['status'] != 'COMPLETE' or not status['physical_exact'] or not status['wrist_exact']
            or sha(root/'inputs.npz') != status['inputs_sha256']
            or sha(root/'records.json') != status['records_sha256']):
        raise ValueError('Native Plus render admission changed')
    records = read(root/'records.json')
    with np.load(root/'inputs.npz', allow_pickle=False) as z:
        cached = {k: z[k].copy() for k in z.files}
    expected = ['canonical'] + [r['request_id'] for r in records['requests']]
    if cached['views'].tolist() != expected:
        raise ValueError('Native Plus candidate order changed')
    cached['candidate_ids'] = cached['views']
    return cached, records['requests']


def initial_scores(connection, cached, noise, prompt):
    """Reproduce plus_evaluate's independent eight-noise initial scoring pass."""
    result = np.empty((len(cached['views']), 8))
    mapping = {'external_images': 'observation/image', 'wrist_images': 'observation/wrist_image',
        'robot_states': 'observation/state', 'camera_intrinsics': 'camera_intrinsics',
        'camera_to_world_opencv': 'camera_to_world_opencv'}
    for start in range(0, len(result), 16):
        stop = min(len(result), start+16)
        examples = [policy_example({key: cached[saved][j] for saved, key in mapping.items()}, prompt)
                    for j in range(start, stop)]
        examples += [examples[-1]] * (16-len(examples))
        for repeat in range(8):
            connection.send(dict(kind='infer', examples=examples, noise=np.repeat(noise[repeat][None], 16, 0)))
            response = connection.recv()
            if response.get('error'):
                raise RuntimeError(response['error'])
            metric = compute_accel_scores(response['velocity'][:stop-start], prefixes=[3])[3]
            if metric['degenerate'].any():
                raise ValueError('Degenerate native initial score')
            result[start:stop, repeat] = metric['score']
    return result


def model_without_external_pose(xml):
    """Allow only agentview position/orientation, not intrinsics or physics."""
    root = ET.fromstring(xml)
    cameras = [c for c in root.iter('camera') if c.get('name') == 'agentview']
    if len(cameras) != 1:
        raise ValueError('Expected one external camera')
    for key in ('pos', 'quat', 'axisangle', 'xyaxes', 'zaxis', 'euler'):
        cameras[0].attrib.pop(key, None)
    return ET.tostring(root)


def verify_absolute_endpoints(config, record, variants, output):
    """Check all custom97 endpoints after replacing an official starting pose.

All official geometries already have independent physics/wrist render receipts.
Here the first official start is an engineering bridge per physical initial;
this is not a simulated camera movement or a new success-rate measurement.
"""
    from libero.libero.envs import OffScreenRenderEnv
    from libero_camera_pose import capture_camera_reference, install_camera_pose
    from evaluate_pi05_libero_plus_views import agentview_camera_calibration, physics_state_sha256
    from scripts.dsol_paper1.runtime.standard_initialization_v1 import load_initial
    from .initialization import initialize, capture_model_reference
    from scripts.dsol_paper1.runtime.scan_libero_hdf5_views import _restore_reference
    from scripts.dsol_paper1.runtime.evaluate_dsol_libero_hdf5_views import masked_policy_observation
    spec = record['state']; physical = load_initial(spec); envs = []
    if sha(record['observation_cache']['path']) != record['observation_cache']['sha256']:
        raise ValueError('Original97 cached observations changed')
    with np.load(record['observation_cache']['path'], allow_pickle=False) as z:
        cached = {k: z[k].copy() for k in z.files}
    candidates = read(config['release'])['protocol']['state_blocks'][record['initial_index']]['candidates']
    try:
        for bddl in (spec['bddl_file'], variants[0]['wrapper_bddl_argument']):
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_names=('agentview', 'robot0_eye_in_hand'),
                camera_heights=256, camera_widths=256, render_gpu_device_id=0)
            envs.append(env); initialize(env, spec, physical); env.env.sim.forward()
        base, changed = envs
        if model_without_external_pose(base.env.sim.model.get_xml()) != model_without_external_pose(changed.env.sim.model.get_xml()):
            raise ValueError('Official variant changes model beyond external pose')
        reference = capture_camera_reference(base, camera_name='agentview', table_plane_z=config['table_plane_z'])
        model_reference = capture_model_reference(base)
        if physics_state_sha256(base) != physics_state_sha256(changed):
            raise ValueError('Official-to-custom bridge physics mismatch')
        for j, candidate in enumerate(candidates):
            initialize(changed, spec, physical, model_reference=model_reference); _restore_reference(changed, reference)
            if candidate['pose'] is not None:
                install_camera_pose(changed, reference, candidate['pose'])
            changed.env._update_observables(force=True)
            obs = changed.env._get_observations()
            example, _, _ = masked_policy_observation(obs, prompt=spec['prompt'], resize_size=224,
                eval_seed=spec['environment_seed'], camera_calibration=agentview_camera_calibration(changed), sensor_control='both')
            example['observation/state'] = np.asarray(example['observation/state'], np.float32)
            for saved, key in [('external_images', 'observation/image'), ('wrist_images', 'observation/wrist_image'),
                    ('robot_states', 'observation/state'), ('camera_intrinsics', 'camera_intrinsics'),
                    ('camera_to_world_opencv', 'camera_to_world_opencv')]:
                if not np.array_equal(cached[saved][j], example[key]):
                    raise ValueError('Custom endpoint cache mismatch: '+key)
        write(Path(output)/'endpoint-equivalence.json', dict(status='PASS', candidates=len(candidates),
            starting_view=variants[0]['request_id'], cache_sha256=sha(record['observation_cache']['path']),
            scope='Absolute endpoint equivalence; not a physical camera-motion rollout'))
    finally:
        for env in envs:
            env.close()
