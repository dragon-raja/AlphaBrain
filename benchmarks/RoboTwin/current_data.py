"""CPU admission for current official XPolicyLab-format clean demonstrations.

Kept separate from legacy joint_action/observation adapters. Does not imply
native replay, policy input equivalence, or simulation state restoration.
"""
from pathlib import Path


def audit_clean_task(root, expected_episodes=50):
    import numpy as np
    root = Path(root)
    files = sorted(root.rglob('*.hdf5'))
    seeds = list(root.rglob('seed.txt'))
    errors, episodes = [], []
    expected_names = {f'episode_{i:07d}.hdf5' for i in range(expected_episodes)}
    if len(files) != expected_episodes or {p.name for p in files} != expected_names:
        errors.append('Episode coverage/naming mismatch')
    if len(seeds) != 1 or len(seeds[0].read_text().split()) != expected_episodes:
        errors.append('Seed coverage mismatch')
    # Empty/incomplete inventories can be rejected without binary IO dependencies.
    # Actual files still require the explicit full-test/runtime dependencies.
    if files:
        import h5py
        import cv2
    joints = {'left_arm_joint_states': 6, 'left_ee_joint_states': 1,
              'right_arm_joint_states': 6, 'right_ee_joint_states': 1}
    for path in files:
        row = {'path': str(path.relative_to(root)), 'errors': []}
        try:
            with h5py.File(path, 'r') as file:
                count = len(file['state/left_arm_joint_states'])
                row['frames'] = count
                row['format_version'] = str(file['data_format_version'][()])
                for prefix in ('state', 'action'):
                    for key, width in joints.items():
                        values = file[f'{prefix}/{key}'][:]
                        if values.shape != (count, width) or not np.isfinite(values).all():
                            row['errors'].append(f'Invalid aligned field: {prefix}/{key}')
                for name in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
                    camera = file[f'vision/{name}']
                    if camera['colors'].shape != (count,):
                        row['errors'].append('Camera length: ' + name)
                    for key, shape in [('extrinsics_matrix',(count,4,4)),('intrinsic_matrix',(count,3,3))]:
                        array = camera[key][:]
                        if array.shape != shape or not np.isfinite(array).all():
                            row['errors'].append('Camera calibration: ' + name + '/' + key)
                    for index in (0, count-1):
                        blob = bytes(camera['colors'][index])
                        image = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if image is None or image.shape != (240, 320, 3) or b'XPL-RGB1' not in blob:
                            row['errors'].append(f'RGB marker/decode/shape: {name}/{index}')
        except (KeyError, ValueError, OSError, IndexError) as exc:
            row['errors'].append(str(exc))
        episodes.append(row)
        if row['errors']:
            errors.append({'episode': row['path'], 'errors': row['errors']})
    return {'status': 'PASS' if not errors else 'INCOMPLETE', 'root': str(root),
            'episodes': len(episodes), 'frames': sum(x.get('frames',0) for x in episodes),
            'errors': errors, 'records': episodes,
            'scope': 'current XPL schema, aligned finite joints/calibration, first/last RGB; not full pixel or physics replay certification'}
