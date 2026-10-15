"""Outcome-blind, bounded starting-checkpoint comparison with the frozen P0."""
from collections import defaultdict
from pathlib import Path
import argparse
import shutil

import numpy as np

from .artifacts import read, sha, write
from .evaluation_matrix import validate_plan, NUMERIC_FLAGS, NOISE_VERSION


def select_initials(jobs):
    groups = defaultdict(list)
    for job in jobs:
        groups[job['task'], job['split']].append(job)
    if len(groups) != 20 or {s for _, s in groups} != {'pretrain', 'target'}:
        raise ValueError('Expected ten tasks with seen and heldout scenes')
    selected = []
    for key, rows in sorted(groups.items()):
        if len(rows) < 2:
            raise ValueError('Two initials per task/split required')
        selected.extend(sorted(rows, key=lambda j: (j['scene_seed'], j['initial_id']))[:2])
    return [j['initial_id'] for j in selected]


def select_views(views):
    """Nearest heldout poses to two mild and two strong opposed orbit targets.

    Distances use normalized azimuth/elevation/radius coordinates. These five
    views are a diagnostic panel, not a uniformly sampled candidate average.
    """
    reference = [v for v in views if v['role'] == 'reference']
    if len(reference) != 1:
        raise ValueError('One reference required')
    extra = [v for v in views if v['role'] not in ('reference', 'train')]
    targets = [(-1/3, -1/3, 0), (1/3, 1/3, 0), (-1, -1, 1), (1, 1, 1)]
    chosen = [reference[0]['view_id']]
    for target in targets:
        remaining = [v for v in extra if v['view_id'] not in chosen]
        if not remaining:
            raise ValueError('Four distinct heldout candidates required')
        winner = min(remaining, key=lambda v: (
            float(np.linalg.norm((np.asarray(v['parameters'])-[0, 0, 1])/[30, 15, .2]-target)),
            v['view_id']))
        chosen.append(winner['view_id'])
    return chosen


def validate_baseline_panel(plan):
    intermediate = plan['format'] == 'robocasa_checkpoint_4000_panel_v1'
    phases = ['canonical_4000', 'broad_4000'] if intermediate else ['official']
    if (plan['format'] not in ('robocasa_baseline_panel_v1', 'robocasa_checkpoint_4000_panel_v1')
            or plan['numeric_flags'] != NUMERIC_FLAGS or plan['noise_version'] != NOISE_VERSION
            or plan['checkpoint_repeats'] != [0, 1, 2, 3]
            or plan['confirmation_repeats'] != [] or set(plan['models']) != set(phases)
            or plan['first_input_source'] != 'frozen_p0_canonical_scores'
            or plan['rollout_count'] != 800 * len(phases)):
        raise ValueError('Unrecognized bounded baseline protocol')
    if intermediate:
        if sha(plan['baseline_plan']) != plan['baseline_plan_sha256']:
            raise ValueError('Original baseline panel changed')
        baseline = read(plan['baseline_plan'])
        validate_baseline_panel(baseline)
        for key in ('initial_ids', 'panel_views', 'p0_plan_sha256'):
            if plan[key] != baseline[key]:
                raise ValueError('4000-step panel must match original baseline: ' + key)
        if sha(plan['previous_checkpoint_identity_plan']) != plan['previous_plan_sha256']:
            raise ValueError('Checkpoint identity source changed')
        identities = read(plan['previous_checkpoint_identity_plan'])['models']
        for phase in phases:
            if plan['models'][phase] != identities[phase] or Path(plan['models'][phase]['checkpoint']).name != '4000':
                raise ValueError('4000-step checkpoint identity mismatch')
    if sha(plan['p0_plan']) != plan['p0_plan_sha256']:
        raise ValueError('P0 plan changed')
    p0 = read(plan['p0_plan']); reference = validate_plan(p0)
    if plan['initial_ids'] != select_initials(reference['jobs']):
        raise ValueError('Outcome-blind initial selection changed')
    if plan['panel_views'] != select_views(reference['catalog']['views']):
        raise ValueError('Geometry-only view selection changed')
    assigned = []
    names = set()
    for worker in plan['workers']:
        if (worker['worker_id'] in names or worker['lane'] != 'checkpoints'
                or worker['phases'] != phases or worker['pilot_initials'] != []):
            raise ValueError('Invalid baseline worker')
        names.add(worker['worker_id'])
        for initial in worker['initials']:
            old = next(w for w in p0['workers'] if w['arm'] == 'canonical' and initial in w['initials'])
            if (worker['host'], worker['gpu']) != (old['host'], old['gpu']):
                raise ValueError('Original host/GPU assignment must be preserved')
        assigned.extend(worker['initials'])
    if len(assigned) != 40 or set(assigned) != set(plan['initial_ids']):
        raise ValueError('Missing or duplicated initial')
    return p0, reference


def prepare(previous_plan, output, source_root, *, baseline_plan=None):
    previous_plan, output, source_root = Path(previous_plan), Path(output), Path(source_root)
    old = read(previous_plan); p0 = read(old['p0_plan']); reference = validate_plan(p0)
    initials = select_initials(reference['jobs']); views = select_views(reference['catalog']['views'])
    phases = ['canonical_4000', 'broad_4000'] if baseline_plan else ['official']
    workers = []
    for worker in p0['workers']:
        selected = [i for i in worker['initials'] if i in initials]
        if worker['arm'] == 'canonical' and selected:
            workers.append(dict(worker_id=f"{worker['host']}-gpu{worker['gpu']}-baseline",
                host=worker['host'], gpu=worker['gpu'], lane='checkpoints',
                phases=phases, initials=selected, pilot_initials=[]))
    # Preserve already frozen identities; runtime checks every file's metadata.
    models = {phase: old['models'][phase] for phase in phases}
    for model in models.values():
        if sha(Path(model['checkpoint'])/'assets/norm_stats.json') != model['norm_sha256']:
            raise ValueError('Checkpoint normalization changed')
    output.mkdir(parents=True, exist_ok=False)
    source = output/'source'
    for name in ('AlphaBrain', 'benchmarks', 'scripts', 'configs'):
        shutil.copytree(source_root/name, source/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    hashes = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
    write(output/'source-manifest.json', hashes)
    plan = dict(format='robocasa_baseline_panel_v1', p0_plan=old['p0_plan'],
        p0_plan_sha256=sha(old['p0_plan']), source=str(source),
        source_manifest=str(output/'source-manifest.json'), source_manifest_sha256=sha(output/'source-manifest.json'),
        previous_checkpoint_identity_plan=str(previous_plan), previous_plan_sha256=sha(previous_plan),
        models=models, workers=workers, initial_ids=initials, panel_views=views,
        checkpoint_repeats=[0, 1, 2, 3], confirmation_repeats=[],
        numeric_flags=NUMERIC_FLAGS, noise_version=NOISE_VERSION, rollout_count=800 * len(phases),
        first_input_source='frozen_p0_canonical_scores',
        authorization='User approved 10 tasks x 4 initials x 5 views x 4 repeats; no training or full matrix',
        scope='Diagnostic only: first two seeds per task/split and geometry-only heldout poses; not 97-view mean or oracle',
        reuse='Compare final canonical/broad checkpoints using exact matching P0 cells; no new final-checkpoint rollouts')
    if baseline_plan:
        plan.update(format='robocasa_checkpoint_4000_panel_v1',
            baseline_plan=str(Path(baseline_plan).resolve()), baseline_plan_sha256=sha(baseline_plan),
            authorization='User approved both 4000-step checkpoints on the unchanged 800-trial baseline panel; total 1600; no training',
            reuse='Reuse official and 8250-step endpoint trials on the exact same cells; no full matrix or new oracle')
    validate_baseline_panel(plan)
    write(output/'plan.json', plan)
    write(output/'selection.json', dict(jobs=[j for j in reference['jobs'] if j['initial_id'] in initials],
        views=[next(v for v in reference['catalog']['views'] if v['view_id']==i) for i in views]))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--baseline-plan', type=Path, help='Enable approved paired 4000-step checkpoint panel')
    args = parser.parse_args()
    plan = prepare(args.previous_plan, args.output, args.source_root, baseline_plan=args.baseline_plan)
    print({'rollouts': plan['rollout_count'], 'views': plan['panel_views'], 'workers': len(plan['workers'])})


if __name__ == '__main__':
    main()
