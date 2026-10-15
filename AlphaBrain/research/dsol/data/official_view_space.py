"""Extend the frozen official panel to every P0 view, without new training."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil

from .artifacts import path_mapping, read, sha, write
from .baseline_panel import validate_baseline_panel
from .diagnostic_panel import digest_file

FORMAT = 'robocasa_official_view_space_v1'


def validate(plan):
    if plan['format'] != FORMAT or sha(plan['baseline_plan']) != plan['baseline_plan_sha256']:
        raise ValueError('Official extension identity mismatch')
    base = read(plan['baseline_plan'])
    p0, reference = validate_baseline_panel(base)
    all_views = [v['view_id'] for v in reference['catalog']['views']]
    pending = [v for v in all_views if v not in base['panel_views']]
    for key in ('initial_ids', 'workers', 'numeric_flags', 'noise_version',
                'checkpoint_repeats', 'confirmation_repeats', 'first_input_source',
                'p0_plan', 'p0_plan_sha256'):
        if plan[key] != base[key]:
            raise ValueError('Original panel convention changed: ' + key)
    if (plan['all_views'] != all_views or plan['panel_views'] != pending
            or plan['reuse_views'] != base['panel_views'] or len(pending) != 92
            or plan['rollout_count'] != 14720 or plan['reused_rollouts'] != 800
            or set(plan['models']) != {'official'}):
        raise ValueError('Expected exactly 40 initials, 97 views, four repeats')
    old, new = base['models']['official'], plan['models']['official']
    for key in ('checkpoint', 'config', 'norm_sha256'):
        if new[key] != old[key]:
            raise ValueError('Official checkpoint changed')
    if set(new['files']) != set(old['files']):
        raise ValueError('Model inventory changed')
    for name, meta in new['files'].items():
        if any(meta[k] != old['files'][name][k] for k in ('sha256', 'bytes')):
            raise ValueError('Official model bytes changed')
    if sha(plan['baseline_audit']) != plan['baseline_audit_sha256']:
        raise ValueError('Baseline audit changed')
    if read(plan['baseline_audit'])['status'] != 'PASS':
        raise ValueError('Only audited baseline trials can be reused')
    return p0, reference


def prepare(baseline_plan, output, source_root, mappings):
    output, source_root = Path(output), Path(source_root)
    base = read(baseline_plan)
    _, ref = validate_baseline_panel(base)
    model = deepcopy(base['models']['official'])
    # Rehash migrated weights, retaining original byte identities and new mtimes.
    for name, meta in model['files'].items():
        p = Path(model['checkpoint']) / name
        if p.stat().st_size != meta['bytes'] or digest_file(p) != meta['sha256']:
            raise ValueError('Migrated checkpoint hash mismatch: ' + name)
        meta['mtime_ns'] = p.stat().st_mtime_ns
    audit_path = Path(baseline_plan).parent/'analysis/audit.json'
    audit = read(audit_path)
    for row in audit['source_records']:
        if sha(row['path']) != row['sha256']:
            raise ValueError('Historical trial changed: ' + row['path'])
    all_views = [v['view_id'] for v in ref['catalog']['views']]
    plan = deepcopy(base)
    plan.update(format=FORMAT, models={'official': model},
        baseline_plan=str(baseline_plan), baseline_plan_sha256=sha(baseline_plan),
        baseline_audit=str(audit_path), baseline_audit_sha256=sha(audit_path),
        path_mappings=mappings, all_views=all_views, reuse_views=base['panel_views'],
        panel_views=[v for v in all_views if v not in base['panel_views']],
        rollout_count=14720, reused_rollouts=800,
        authorization='User approved official frozen model, original 40 initials, all 97 views, original four execution noises; no training',
        scope='40-initial full-view diagnostic; 20 seen and 20 heldout; not the 400-initial P0 population',
        reuse='800 official baseline trials and corresponding final-model P0 cells; no changes to historical receipts',
        monitor_hours=48)
    output.mkdir(parents=True, exist_ok=False)
    source = output/'source'
    for name in ('AlphaBrain', 'benchmarks', 'scripts', 'configs'):
        shutil.copytree(source_root/name, source/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    manifest = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
    write(output/'source-manifest.json', manifest)
    plan.update(source=str(source), source_manifest=str(output/'source-manifest.json'),
                source_manifest_sha256=sha(output/'source-manifest.json'))
    validate(plan)
    write(output/'plan.json', plan)
    write(output/'selection.json', dict(jobs=[j for j in ref['jobs'] if j['initial_id'] in plan['initial_ids']],
                                       views=ref['catalog']['views']))
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-plan', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--source-root', required=True, type=Path)
    p.add_argument('--map-prefix', action='append', nargs=2, default=[])
    a = p.parse_args()
    with path_mapping(a.map_prefix):
        plan = prepare(a.baseline_plan, a.output, a.source_root, a.map_prefix)
    print({'new_rollouts':plan['rollout_count'], 'reused_rollouts':800, 'workers':len(plan['workers'])})


if __name__ == '__main__':
    main()
