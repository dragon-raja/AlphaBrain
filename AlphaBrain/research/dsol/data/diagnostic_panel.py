"""Supplemental checkpoint/selection diagnosis; never changes the P0 matrix.

View panels are chosen from geometry alone. Confirmation choices are frozen
per initial after averaging P0 execution noise, not reselected per new trial.
"""
from itertools import product
from pathlib import Path
import hashlib
import shutil

import numpy as np

from .artifacts import read, sha, write
from .evaluation_matrix import validate_plan, NUMERIC_FLAGS, NOISE_VERSION


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def geometry_panel(views):
    """Reference + balanced inner/outer tetrahedra + four maximin extra views.

    This is a diagnostic panel, not an unbiased estimator of all 97 views.
    Both tetrahedra have zero mean signed displacement in each coordinate.
    """
    xyz = (np.asarray([v['parameters'] for v in views])- [0, 0, 1]) / [30, 15, .2]
    selected = [next(i for i, v in enumerate(views) if v['role'] == 'reference')]
    train = np.array([i for i, v in enumerate(views) if v['role'] == 'train'])
    for scale in (1/3, 1):
        for signs in product((-1, 1), repeat=3):
            if np.prod(signs) != 1:
                continue
            distance = np.linalg.norm(xyz[train]-np.asarray(signs)*scale, axis=1)
            i = int(train[np.argmin(distance)])
            if distance.min() > 1e-6 or i in selected:
                raise ValueError('Expected balanced original training grid')
            selected.append(i)
    extra = [i for i, v in enumerate(views) if v['role'] not in ('reference', 'train')]
    for _ in range(4):
        i = max(extra, key=lambda i: (np.linalg.norm(xyz[selected]-xyz[i], axis=1).min(), -i))
        selected.append(i)
        extra.remove(i)
    return [views[i]['view_id'] for i in selected]


def frozen_choices(success, accel_scores, learned_scores):
    """Input learned scores are stored costs (-predicted value), as in P0.

    Include canonical; ties follow original catalog order deterministically.
    """
    success = np.asarray(success)
    if success.ndim != 3 or success.shape[-1] != 4:
        raise ValueError('Expected initial x view x four original execution noises')
    if accel_scores.shape != success.shape[:2] or learned_scores.shape != success.shape[:2]:
        raise ValueError('Selection score dimensions differ')
    if not all(np.isfinite(x).all() for x in (success, accel_scores, learned_scores)):
        raise ValueError('Nonfinite selector input')
    return dict(canonical=np.zeros(len(success), dtype=int),
                min_accel=np.argmin(accel_scores, axis=1),
                rgb_accel=np.argmin(learned_scores, axis=1),
                empirical_winner=np.argmax(success.mean(-1), axis=1))


def validate_diagnostic(plan):
    if plan.get('format') == 'robocasa_official_view_space_v1':
        from .official_view_space import validate
        return validate(plan)
    if plan.get('format') in ('robocasa_baseline_panel_v1', 'robocasa_checkpoint_4000_panel_v1'):
        from .baseline_panel import validate_baseline_panel
        return validate_baseline_panel(plan)
    if (plan['format'] != 'robocasa_attribution_v1' or plan['numeric_flags'] != NUMERIC_FLAGS
            or plan['noise_version'] != NOISE_VERSION or plan['checkpoint_repeats'] != list(range(4))
            or plan['confirmation_repeats'] != list(range(4, 12))):
        raise ValueError('Unrecognized supplemental protocol')
    if sha(plan['p0_plan']) != plan['p0_plan_sha256']:
        raise ValueError('P0 identity changed')
    p0 = read(plan['p0_plan'])
    reference = validate_plan(p0)
    for key in ('population.json', 'matrix.npz', 'method-arrays.npz'):
        if digest_file(Path(plan['analysis'])/key) != plan['analysis_hashes'][key]:
            raise ValueError('Analysis inputs changed')
    if plan['panel_views'] != geometry_panel(reference['catalog']['views']):
        raise ValueError('Geometry-only panel changed')
    initials = {j['initial_id'] for j in reference['jobs']}
    for lane in ('checkpoints', 'confirmation'):
        rows = [i for w in plan['workers'] if w['lane'] == lane for i in w['initials']]
        if len(rows) != 400 or set(rows) != initials:
            raise ValueError('Supplement must retain all 400 original initials')
    if len({w['worker_id'] for w in plan['workers']}) != len(plan['workers']):
        raise ValueError('Duplicate worker')
    for arm in ('canonical', 'broad'):
        if set(plan['choices'][arm]) != initials:
            raise ValueError('Confirmation population changed')
        for choices in plan['choices'][arm].values():
            if set(choices) != {'canonical', 'min_accel', 'rgb_accel', 'empirical_winner'}:
                raise ValueError('Confirmation methods changed')
            if not set(choices.values()).issubset(p0['view_ids']):
                raise ValueError('Unknown chosen view')
    return p0, reference


def prepare(config_file, output, source_root):
    cfg, output, source_root = read(config_file), Path(output), Path(source_root)
    p0 = read(cfg['p0_plan']); reference = validate_plan(p0)
    analysis = Path(cfg['analysis']); population = read(analysis/'population.json')
    if ([j['initial_id'] for j in population['jobs']] !=
            [j['initial_id'] for j in sorted(reference['jobs'], key=lambda j: (j['split'], j['task'], j['scene_seed']))]):
        raise ValueError('Unexpected analysis ordering')
    output.mkdir(parents=True, exist_ok=False)
    choices = {}
    with np.load(analysis/'matrix.npz') as matrix, np.load(analysis/'method-arrays.npz') as arrays:
        for m, arm in enumerate(('canonical', 'broad')):
            chosen = frozen_choices(matrix['success'][m], arrays[arm+'__score_accel'], arrays[arm+'__score_rgb_accel'])
            choices[arm] = {j['initial_id']: {method: population['views'][int(ids[n])]['view_id']
                for method, ids in chosen.items()} for n, j in enumerate(population['jobs'])}
    models = {}
    for name, model in cfg['models'].items():
        checkpoint = Path(model['checkpoint']); files = {}
        for file in sorted(checkpoint.rglob('*')):
            if not file.is_file() or file.relative_to(checkpoint).parts[0] not in ('params', 'assets'):
                continue
            stat = file.stat()
            files[str(file.relative_to(checkpoint))] = dict(sha256=digest_file(file), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        models[name] = dict(**model, files=files, norm_sha256=sha(checkpoint/'assets/norm_stats.json'))
        if models[name]['norm_sha256'] != p0['models']['canonical']['norm_sha256']:
            raise ValueError('Normalization differs from P0')
    models.update({arm+'_8250': dict(**p0['models'][arm], parent_arm=arm) for arm in ('canonical', 'broad')})
    workers = []
    for w in p0['workers']:
        lane = 'checkpoints' if w['arm'] == 'canonical' else 'confirmation'
        workers.append(dict(worker_id=f"{w['host']}-gpu{w['gpu']}-{lane}", host=w['host'], gpu=w['gpu'],
                            lane=lane, initials=w['initials'], pilot_initials=w['pilot_initials'],
                            phases=['official', 'canonical_4000', 'broad_4000'] if lane == 'checkpoints'
                            else ['canonical_8250', 'broad_8250']))
    source = output/'source'
    for name in ('AlphaBrain', 'benchmarks', 'scripts', 'configs'):
        shutil.copytree(source_root/name, source/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    source_hashes = {str(p.relative_to(source)): digest_file(p) for p in source.rglob('*') if p.is_file()}
    write(output/'source-manifest.json', source_hashes)
    plan = dict(format='robocasa_attribution_v1', p0_plan=cfg['p0_plan'], p0_plan_sha256=sha(cfg['p0_plan']),
        analysis=str(analysis), analysis_hashes={n: digest_file(analysis/n) for n in ('population.json','matrix.npz','method-arrays.npz')},
        source=str(source), source_manifest=str(output/'source-manifest.json'),
        source_manifest_sha256=sha(output/'source-manifest.json'), models=models, workers=workers,
        choices=choices, panel_views=geometry_panel(reference['catalog']['views']),
        checkpoint_repeats=list(range(4)), confirmation_repeats=list(range(4,12)),
        numeric_flags=NUMERIC_FLAGS, noise_version=NOISE_VERSION,
        scope='Supplement only; original P0 frozen. No new training, no camera motion during rollout.',
        inference='New-noise evaluation estimates frozen selection procedure, not the exact population Oracle.')
    plan['rollout_counts'] = dict(checkpoints=400*len(plan['panel_views'])*3*4,
        confirmation=sum(len(set(c.values()))*8 for arm in choices.values() for c in arm.values()))
    # Analysis lives on fresh. Copy exact frozen inputs into shared output for GNU.
    for name in plan['analysis_hashes']:
        (output/'analysis-inputs').mkdir(exist_ok=True)
        shutil.copy2(analysis/name, output/'analysis-inputs'/name)
    plan['analysis'] = str(output/'analysis-inputs')
    validate_diagnostic(plan)
    write(output/'plan.json', plan)
    return plan
