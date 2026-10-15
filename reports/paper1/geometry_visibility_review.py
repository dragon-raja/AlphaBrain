"""Render review-only figures/tables; never update the current paper PDF."""
import argparse
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from AlphaBrain.research.dsol.data.artifacts import read, write, sha
from AlphaBrain.research.dsol.plotting.geometry_visibility import (
    geometry_figure, all_tasks_figure, orbit_figure, strata_figure, restricted_figure, orientation_figure)


def build(config_path):
    cfg = read(config_path); out = Path(cfg['output']); result = read(out/'summary.json')
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False})
    with np.load(out/'arrays.npz') as z: camera = z['camera']
    # Reference table plane is read from the frozen release, not guessed.
    states = read(Path(cfg['analysis'])/'states.json')
    root = Path(states[0]['initial_asset_record']).parents[3]
    release = read(root/'release.json'); catalog = read(release['catalog']['path'])
    if sha(release['catalog']['path']) != release['catalog']['sha256']:
        raise ValueError('Catalog identity changed')
    table = catalog['table_plane_z']
    asset = read(states[0]['initial_asset_record'])
    if sha(asset['artifact']) != asset['artifact_sha256']:
        raise ValueError('Camera intrinsics input identity changed')
    with np.load(asset['artifact']) as z:
        intrinsics = z['camera_intrinsics']
        np.testing.assert_allclose(z['camera_to_world_opencv'], camera[0], rtol=0, atol=1e-12)
    from AlphaBrain.research.dsol.plotting.geometry_visibility import pivot_from_camera, orbit_positions
    poses = catalog['broad_training_64'] + catalog['broad_heldout_32']
    position_errors, direction_errors, geometry_rows = [], [], []
    for i, cameras in enumerate(camera):
        pivot = pivot_from_camera(cameras[0], table)
        positions = orbit_positions(cameras[0, :3, 3], pivot,
            np.array([p['azimuth_deg'] for p in poses]), np.array([p['elevation_deg'] for p in poses]),
            np.array([p['radius_scale'] for p in poses]))
        position_errors.append(float(np.abs(positions-cameras[1:, :3, 3]).max()))
        forward = pivot-positions; forward /= np.linalg.norm(forward, axis=1)[:, None]
        direction_errors.append(float(np.abs(forward-cameras[1:, :3, 2]).max()))
        for j, c in enumerate(cameras):
            geometry_rows.append(dict(initial_index=i, task=states[i]['task_id'],
                candidate_id='canonical' if j == 0 else poses[j-1]['pose_id'],
                **dict(zip(['x_m', 'y_m', 'z_m'], c[:3, 3])),
                **dict(zip(['forward_x', 'forward_y', 'forward_z'], c[:3, 2]))))
    if max(position_errors) > 1e-8 or max(direction_errors) > 1e-8:
        raise ValueError('Illustrated orbit does not match measured candidate extrinsics')
    write(out/'geometry-validation.json', dict(max_position_error_m=max(position_errors),
        max_forward_component_error=max(direction_errors), table_plane_z=table,
        catalog_sha256=sha(release['catalog']['path']), convention='Saved OpenCV camera +Z = optical forward'))
    with (out/'camera_poses.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(geometry_rows[0])); w.writeheader(); w.writerows(geometry_rows)
    figures = {'01_camera_world': geometry_figure(camera, table),
               '02_all_tasks_world': all_tasks_figure(camera, table),
               '03_orbit_axes': orbit_figure(camera, table),
               '04_visibility_strata': strata_figure(result),
               '05_visible_destinations': restricted_figure(result),
               '06_camera_orientation': orientation_figure(camera, intrinsics, table)}
    for name, fig in figures.items():
        fig.savefig(out/(name+'.png'), dpi=170, bbox_inches='tight'); plt.close(fig)
    rows = []
    for arm, model in result['models'].items():
        for method, strata in model['primary'].items():
            for name, v in strata.items():
                rows.append(dict(model=arm, method=method, stratum=name, success=v['success'],
                    stay=v['stay'], gain=v['gain'], ci95_low=v['ci95_gain'][0], ci95_high=v['ci95_gain'][1],
                    initials=v['initials'], tasks=v['tasks'], starts=v['starts']))
    with (out/'recovery_strata.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    provenance = read(out/'provenance.json')
    provenance['inputs'][asset['artifact']] = asset['artifact_sha256']
    provenance['report_source_sha256'] = sha(__file__)
    from AlphaBrain.research.dsol.plotting import geometry_visibility
    provenance['plot_source_sha256'] = sha(geometry_visibility.__file__)
    provenance['outputs'] = {p.name: sha(p) for p in out.iterdir() if p.suffix in ('.png', '.csv')}
    write(out/'provenance.json', provenance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', required=True)
    build(parser.parse_args().config)
