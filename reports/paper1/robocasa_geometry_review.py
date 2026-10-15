"""Build review-only RoboCasa geometry audit from frozen caches, CPU only."""
import argparse
import csv
import gzip
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from AlphaBrain.research.dsol.data.artifacts import read, write, sha
from AlphaBrain.research.dsol.analysis.rig_geometry import audit_mounts
from AlphaBrain.research.dsol.plotting.rig_geometry import installation_figure, coverage_figure


def build(config_path):
    cfg = read(config_path); out = Path(cfg['output'])
    if out.exists() and any(out.iterdir()):
        previous = read(out/'provenance.json')
        if previous['inputs'].get(str(config_path)) != sha(config_path):
            raise ValueError('Existing review belongs to another configuration')
    out.mkdir(parents=True, exist_ok=True)
    matrix = read(cfg['matrix_plan']); ref_path = matrix['reference_plan']; reference = read(ref_path)
    release_path = matrix['reference_release']; release = read(release_path)
    if (sha(ref_path) != matrix['reference_plan_sha256'] or sha(release_path) != matrix['reference_release_sha256']
            or not release['complete'] or release['counts'] != {'PASS': len(reference['jobs'])}):
        raise ValueError('Formal reference release identity mismatch')
    released = {r['initial_id']:r for r in release['records']}
    catalog = reference['catalog']; params = np.array([v['parameters'] for v in catalog['views']])
    inputs = {str(config_path): sha(config_path), cfg['matrix_plan']: sha(cfg['matrix_plan']), ref_path: sha(ref_path), release_path: sha(release_path)}
    records, displacements, angles, csv_rows = [], [], [], []
    example = None
    names = ('robot0_agentview_left', 'robot0_agentview_right')
    for job in reference['jobs']:
        record = released[job['initial_id']]
        root = Path(record['candidate_root']); summary_path = root/'summary.json'; summary = read(summary_path)
        if sha(summary_path) != record['candidate_summary_sha256']:
            raise ValueError('Frozen summary changed')
        geometry_path = root/'geometry.npz'
        if sha(geometry_path) != summary['files']['geometry.npz']['sha256']:
            raise ValueError('Frozen geometry changed')
        xml_path = Path(job['path'])/'compiled_model.xml.gz'
        # XML traversal gives model camera order (all six cameras are explicit).
        xml = ET.fromstring(gzip.decompress(xml_path.read_bytes()))
        cameras = list(xml.find('worldbody').iter('camera'))
        camera_names = [c.attrib['name'] for c in cameras]
        ids = [camera_names.index(name) for name in names]
        parents = {c.attrib['name']: body.attrib.get('name') for body in xml.iter('body') for c in body.findall('camera')}
        if parents[names[0]] != parents[names[1]]:
            raise ValueError('Cannot compare local poses from distinct parent frames')
        with np.load(geometry_path, allow_pickle=False) as z:
            if len(cameras) != z['camera_positions'].shape[1] or list(z['view_ids']) != [v['view_id'] for v in catalog['views']]:
                raise ValueError('Camera/candidate ordering mismatch')
            xml_positions = np.array([[float(x) for x in c.attrib.get('pos', '0 0 0').split()] for c in cameras])
            np.testing.assert_allclose(z['camera_positions'][0], xml_positions, atol=1e-5, rtol=0)
            result = audit_mounts(z['camera_positions'], z['camera_quaternions'], ids, params, catalog['focus_distance_m'])
        for path in (summary_path, geometry_path, xml_path): inputs[str(path)] = sha(path)
        records.append(dict(initial_id=job['initial_id'], task=job['task'], split=job['split'],
                            baseline_m=float(result['baseline'][0]), parent=parents[names[0]], **result['checks']))
        displacements.append(result['camera_displacement']); angles.append(result['optical_angle_deg'])
        for i, view in enumerate(catalog['views']):
            for c, name in enumerate(names):
                csv_rows.append(dict(initial_id=job['initial_id'], task=job['task'], split=job['split'],
                    view_id=view['view_id'], camera=name, parent=parents[name],
                    **dict(zip(('x_m','y_m','z_m'), result['positions'][i,c])),
                    **dict(zip(('forward_x','forward_y','forward_z'), -result['rotations'][i,c,:,2]))))
        if example is None and job['split'] == 'pretrain':
            example = result, [float(cameras[i].attrib['fovy']) for i in ids], job
    displacement, angle = np.stack(displacements), np.stack(angles)
    describe = lambda a: dict(min=float(np.min(a)), median=float(np.median(a)), max=float(np.max(a)))
    report = dict(initials=len(records), candidates=len(params), cameras_per_rig=2,
        coordinate_frame='camera parent support body, not world trajectory',
        task_count=len({r['task'] for r in records}),
        splits={s: sum(r['split']==s for r in records) for s in sorted({r['split'] for r in records})},
        camera_displacement_m=describe(displacement), optical_angle_deg=describe(angle),
        baseline_m=describe([r['baseline_m'] for r in records]),
        max_baseline_error_m=max(r['baseline_error_m'] for r in records),
        max_relative_rotation_component_error=max(r['relative_rotation_component_error'] for r in records),
        max_central_ray_pivot_error_m=max(r['central_ray_pivot_error_m'] for r in records),
        all_other_camera_mounts_unchanged=True, illustration_initial_id=example[2]['initial_id'],
        illustration_selection='First pretrain initial in frozen plan; no outcome access',
        per_initial=records, limitations=['No reachability/collision certificate', 'No task-entity segmentation in this audit',
            'No full-episode camera trajectory read', 'No outcomes, rendering, simulation, training or model inference'])
    write(out/'summary.json', report)
    with (out/'camera_mounts.csv').open('w') as f:
        w=csv.DictWriter(f, fieldnames=list(csv_rows[0])); w.writeheader(); w.writerows(csv_rows)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP', 'axes.unicode_minus':False})
    figures = {'01_linked_rig_installations': installation_figure(example[0], example[1],
        'RoboCasa：'+example[2]['task']+'，seen 初态 seed '+str(example[2]['scene_seed'])),
        '02_candidate_coverage': coverage_figure(params, displacement, angle)}
    for name, fig in figures.items():
        fig.savefig(out/(name+'.png'), dpi=160, bbox_inches='tight'); plt.close(fig)
    from AlphaBrain.research.dsol.analysis import rig_geometry as analysis_module
    from AlphaBrain.research.dsol.plotting import rig_geometry as plotting_module
    write(out/'provenance.json', dict(inputs=inputs,
        sources={p:sha(p) for p in (__file__, analysis_module.__file__, plotting_module.__file__)},
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name != 'provenance.json'}))
    print({k:v for k,v in report.items() if k != 'per_initial'})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', required=True)
    build(parser.parse_args().config)
