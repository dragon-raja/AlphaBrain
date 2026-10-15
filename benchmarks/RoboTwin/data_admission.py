"""CPU structural audit of legacy official clean demonstrations; no simulator IO."""
import json
from pathlib import Path
import re


def screen_joint_limits(summary, urdf, ordered_names, tolerance=1e-5):
    """Screen recorded commands, not realized robot states or collision safety."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(urdf)
    limits = {j.attrib['name']: j.find('limit').attrib for j in tree.getroot().findall('joint')
              if j.find('limit') is not None}
    flags = []
    for row in summary['episodes']:
        if len(ordered_names) != len(row['joint_min']):
            raise ValueError('Joint ordering mismatch')
        for i, name in enumerate(ordered_names):
            if name is None:
                continue
            bound = limits[name]
            if (row['joint_min'][i] < float(bound['lower']) - tolerance
                    or row['joint_max'][i] > float(bound['upper']) + tolerance):
                flags.append({'episode': row['episode'], 'joint': name,
                              'min': row['joint_min'][i], 'max': row['joint_max'][i]})
    return {'status': 'COMMAND_LIMIT_SCREEN_PASSED' if not flags else 'FLAGGED',
            'ordered_joint_names': ordered_names, 'tolerance': tolerance,
            'flagged_commands': flags, 'realized_state_or_collision_validated': False}


def inspect(root):
    import h5py
    import numpy as np

    root = Path(root)
    seeds = [int(s) for s in (root / 'seed.txt').read_text().split()]
    scenes = json.loads((root / 'scene_info.json').read_text())
    rows, errors = [], []
    paths = sorted((root / 'data').glob('*.hdf5'))
    indices = set()
    for path in paths:
        match = re.fullmatch(r'episode_?(\d+)\.hdf5', path.name)
        if not match:
            errors.append(f'Unexpected episode filename: {path.name}')
            continue
        index = int(match[1])
        if index in indices or index >= len(seeds):
            errors.append(f'Duplicate/out-of-range episode: {index}')
            continue
        indices.add(index)
        trajectories = [root / '_traj_data' / f'episode{index}.pkl',
                        root / '_traj_data' / f'episode_{index:07d}.pkl']
        trajectory = next((p for p in trajectories if p.is_file()), None)
        if trajectory is None:
            errors.append(f'Missing trajectory: {index}')
        if f'episode_{index}' not in scenes:
            errors.append(f'Missing scene record: {index}')
        with h5py.File(path, 'r') as h:
            q = h['joint_action/vector'][:]
            if q.ndim != 2 or q.shape[1] != 14 or len(q) == 0 or not np.isfinite(q).all():
                errors.append(f'Invalid joint-action array: {index}')
            cameras = {}
            for name, obs in h['observation'].items():
                cameras[name] = {key: {'shape': list(value.shape), 'dtype': str(value.dtype)}
                                 for key, value in obs.items()}
                if 'rgb' not in obs or len(obs['rgb']) != len(q):
                    errors.append(f'Frame/action mismatch: {index}/{name}')
                else:
                    for frame in (0, len(q) - 1):
                        prefix = bytes(obs['rgb'][frame])[:8]
                        if not (prefix.startswith(b'\xff\xd8') or prefix.startswith(b'\x89PNG\r\n\x1a\n')):
                            errors.append(f'Unrecognized image encoding: {index}/{name}/{frame}')
            rows.append({'episode': index, 'seed': seeds[index], 'frames': len(q),
                         'joint_min': q.min(axis=0).tolist(), 'joint_max': q.max(axis=0).tolist(),
                         'trajectory': str(trajectory) if trajectory else None, 'cameras': cameras})
    if indices != set(range(len(seeds))) or len(paths) != len(seeds):
        errors.append('HDF5/seed coverage mismatch')
    return {'status': 'STRUCTURE_CHECKED' if not errors else 'FAILED',
            'root': str(root), 'episodes': sorted(rows, key=lambda x: x['episode']),
            'seed_count': len(seeds), 'unique_seeds': len(set(seeds)), 'errors': errors,
            'runtime_replay_verified': False, 'joint_limits_verified': False,
            'rgb_channel_semantics_verified': False, 'trajectory_pickle_loaded': False}
