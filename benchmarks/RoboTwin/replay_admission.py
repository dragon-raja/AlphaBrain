"""CPU-only postflight: native replay completion is not rendering admission."""
import json
from pathlib import Path


def render_errors(text):
    markers = ('[error]', 'traceback (most recent call last)', 'segmentation fault')
    return sorted(set(line for line in text.splitlines()
                      if any(marker in line.lower() for marker in markers)))


def audit_attempt(attempt, log_path):
    import numpy as np
    import h5py
    import cv2
    from PIL import Image
    from AlphaBrain.research.dsol.data.resource_import import digest

    attempt, log_path = Path(attempt), Path(log_path)
    state_path = attempt / 'status.json'
    state = json.loads(state_path.read_text())
    errors = render_errors(log_path.read_text())
    result = {'attempt': str(attempt), 'status_sha256': digest(state_path),
              'log_sha256': digest(log_path), 'denoiser': state['denoiser'],
              'render_errors': errors, 'task': state['task'],
              'scope': 'one native expert demonstration; not VLA success rate or all-data certification',
              'physics_replay_verified': False, 'same_state_camera_probe_verified': False,
              'admitted': False}
    if state['status'] != 'COMPLETE':
        result['reason'] = 'Native replay did not complete'
        return result
    for path, sha in state['raw_inputs'].items():
        if digest(path) != sha:
            raise ValueError('Replay input bytes changed')
    raw = next(Path(p) for p in state['raw_inputs'] if p.endswith('.hdf5'))
    actual = np.load(attempt / 'recorded_joint_states.npy', allow_pickle=False)
    pixel_comparison = {}
    with h5py.File(raw, 'r') as source:
        expected = source['joint_action/vector'][:]
        same_joints = bool(actual.shape == expected.shape and np.array_equal(actual, expected))
        for name in ('head_camera', 'left_camera', 'right_camera'):
            blob = source[f'observation/{name}/rgb'][0]
            decoded = cv2.imdecode(np.frombuffer(bytes(blob), dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError('Invalid original compressed RGB')
            rendered = np.asarray(Image.open(attempt / f'canonical-{name}.png').convert('RGB'))
            if decoded.shape != rendered.shape:
                raise ValueError('Original and replayed image shape mismatch')
            # Old RoboTwin OpenCV encoder/decoder preserves array channel order.
            # Report both possibilities; a small error is not an automatic gate.
            pixel_comparison[name] = {
                'raw_cv2_vs_rendered_rgb_mae': float(np.abs(decoded.astype(float)-rendered).mean()),
                'channel_reversed_vs_rendered_rgb_mae': float(np.abs(decoded[..., ::-1].astype(float)-rendered).mean()),
                'shape': list(rendered.shape)}
    result['joint_states_exactly_equal'] = same_joints
    result['replayed_shape'] = list(actual.shape)
    consumed = all(used == total for used, total in state['trajectory_consumption'].values())
    result['physics_replay_verified'] = bool(state['task_success'] and consumed and same_joints)
    probe = state['camera_probe']
    result['same_state_camera_probe_verified'] = bool(
        probe['observable_physics_unchanged'] and all(v['wrist_extrinsics_unchanged'] for v in probe['views'])
        and all(v['pixel_mae_from_first'][name] == 0 for v in probe['views']
                for name in ('left_camera', 'right_camera'))
        and all(v['pixel_mae_from_first']['head_camera'] == 0 for v in probe['views']
                if v['label'] in ('repeat', 'restored'))
        and next(v for v in probe['views'] if v['label'] == 'translate_x_3cm')['pixel_mae_from_first']['head_camera'] > 0)
    result['original_first_frame_comparison'] = pixel_comparison
    result['original_rgb_equivalence_certified'] = False
    result['admitted'] = bool(result['physics_replay_verified']
                              and result['same_state_camera_probe_verified'] and not errors)
    result['status'] = 'ENGINEERING_PILOT_PASS' if result['admitted'] else 'NOT_ADMITTED'
    return result
