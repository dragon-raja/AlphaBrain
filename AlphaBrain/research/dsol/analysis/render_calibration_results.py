"""Verify calibration receipts and summarize independent-process RGB differences."""

from pathlib import Path
import numpy as np

from ..data.artifacts import read, sha
from .render_tolerance import propose_mean_gate, propose_fraction_gate, validate_fixed_gate


def pixel_difference(a, b):
    if a.shape != b.shape or a.dtype != np.uint8 or b.dtype != np.uint8:
        raise ValueError('Matching uint8 RGB arrays required')
    delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return dict(max_abs=int(delta.max()), mean_abs=float(delta.mean()),
                fraction_gt1=float(np.mean(delta > 1)), fraction_gt2=float(np.mean(delta > 2)))


def summarize_calibration(root):
    root = Path(root)
    plan = read(root / 'plan.json')
    for path, expected in plan['provenance'].items():
        if sha(path) != expected:
            archive = read(root / 'source-snapshot-manifest.json')
            if archive[path]['sha256'] != expected or sha(archive[path]['path']) != expected:
                raise ValueError('Calibration source provenance cannot be recovered: ' + path)
    status = read(root / 'status.json')
    if not status['complete'] or len(status['records']) != 2 * len(plan['initials']):
        raise ValueError('Calibration not complete')
    if any(r['returncode'] for r in status['records']):
        raise ValueError('Failed process in calibration')
    groups = {'calibration': [], 'validation': [], 'confirmation': []}
    per_initial = []
    for row in plan['initials']:
        checks, copies = [], []
        for process in range(2):
            folder = root / row['role'] / row['task'] / f'process-{process}'
            receipt = read(folder / 'summary.json')
            if (receipt['numeric_start'] != receipt['numeric_end']
                    or receipt['manifest_sha256'] != sha(Path(row['path']) / 'manifest.json')
                    or receipt['images_sha256'] != sha(folder / 'images.npz')):
                raise ValueError('Calibration provenance / physical restore failure')
            for stage in receipt['stages'].values():
                if not stage['poses_exact'] or stage['unique_geom_buffers'] != 1:
                    raise ValueError('Scene geometry changed during repeated rendering')
                for camera in stage['cameras'].values():
                    checks.extend(camera['historical'])
                    checks.extend(camera['repeat'])
            with np.load(folder / 'images.npz', allow_pickle=False) as arrays:
                copies.append({k: arrays[k] for k in arrays.files})
        cross_process = []
        for key in copies[0]:
            for a in copies[0][key]:
                for b in copies[1][key]:
                    cross_process.append(pixel_difference(a, b))
        checks.extend(cross_process)
        groups[row['role']].extend(checks)
        per_initial.append(dict(task=row['task'], role=row['role'], layout_id=row['layout_id'],
            comparisons=len(checks), max_abs=max(c['max_abs'] for c in checks),
            max_mean_abs=max(c['mean_abs'] for c in checks),
            max_fraction_gt1=max(c['fraction_gt1'] for c in checks),
            cross_process_max_mean_abs=max(c['mean_abs'] for c in cross_process)))
    proposal = (validate_fixed_gate(groups['confirmation'], plan['fixed_mean_abs'])
                if plan.get('mode') == 'confirmation' else propose_mean_gate(groups['calibration'], groups['validation']))
    if plan.get('mode') == 'fraction_calibration':
        proposal = propose_fraction_gate(groups['calibration'], groups['validation'])
    return dict(source_directory=str(root.resolve()), plan_sha256=sha(root / 'plan.json'), proposal=proposal,
                initials=per_initial, comparisons={k: len(v) for k, v in groups.items()},
                independence='Seen initial states selected before rendering; layout-group isolation; 2 fresh processes each',
                limitations=['same host GPU and driver', 'no VLA action or rollout equivalence claim',
                             'finite engineering sample, not a universal rendering bound'])
