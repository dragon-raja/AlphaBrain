"""Versioned current-render references; historical evidence stays immutable."""

from pathlib import Path
import numpy as np
from PIL import Image

from .artifacts import read, sha, write
from .evaluation_preparation import verify_files

FORMAT = 'robocasa_current_render_reference_v1'
CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')
LIMITS = dict(max_abs=2, mean_abs=.005, fraction_gt1=.0001, fraction_gt2=0.)


def prepare_plan(old_plan_file, old_cache, output, implementation):
    output = Path(output)
    if (output / 'plan.json').exists():
        raise FileExistsError(output / 'plan.json')
    old = read(old_plan_file)
    if {k: old['pixel_gate'][k] for k in LIMITS} != LIMITS:
        raise ValueError('Numerical thresholds must remain unchanged')
    jobs = []
    for job in old['jobs']:
        cache = Path(old_cache) / 'initials' / job['initial_id']
        source = read(cache / 'summary.json')
        if (source['initial_id'] != job['initial_id'] or source['plan_sha256'] != sha(old_plan_file)
                or source['snapshot_manifest_sha256'] != job['snapshot_manifest_sha256']):
            raise ValueError('Old receipt identity mismatch')
        reference = output / 'references' / job['initial_id']
        if source['status'] == 'PASS':
            reference_from_cache(job, old['catalog'], cache, reference, sha(old_plan_file))
        elif source['status'] != 'ERROR':
            raise ValueError('Unexpected prior status')
        jobs.append(dict(job, previous_status=source['status'], source_cache=str(cache),
            source_summary_sha256=sha(cache / 'summary.json'), reference_file=str(reference / 'reference.json'),
            frozen_reference_sha256=sha(reference / 'reference.json') if source['status'] == 'PASS' else None))
    plan = dict(format=FORMAT, authorization='User approved unified current-render references on 2026-09-18',
        status='REFERENCES_REQUIRE_INDEPENDENT_VALIDATION', historical_results_unchanged=True,
        old_plan=str(Path(old_plan_file).resolve()), old_plan_sha256=sha(old_plan_file),
        catalog=old['catalog'], counts=old['counts'], semantics=old['semantics'],
        pixel_gate=LIMITS, jobs=jobs, implementation_sha256=implementation,
        reference_policy='Existing verified current-runtime cache, otherwise first fresh capture; no retry selection',
        release_scope='Initial candidate cache only; not a closed-loop evaluation or GNU renderer certification')
    write(output / 'plan.json', plan)
    return plan


def validate_candidate(job, catalog, root, expected_plan):
    root = Path(root)
    result = read(root / 'summary.json')
    if (result['status'] != 'PASS' or result['initial_id'] != job['initial_id']
            or result['snapshot_manifest_sha256'] != job['snapshot_manifest_sha256']
            or result['plan_sha256'] != expected_plan):
        raise ValueError('Candidate cache identity or admission mismatch')
    if len(result['views']) != len(catalog['views']):
        raise ValueError('Incomplete candidate catalog')
    for actual, expected in zip(result['views'], catalog['views']):
        if any(actual[k] != v for k, v in expected.items()) or len(actual['images']) != 3:
            raise ValueError('Candidate ordering or geometry definition changed')
        if any(name not in result['files'] for name in actual['images']):
            raise ValueError('Unregistered candidate image')
    manifest = read(Path(job['path']) / 'manifest.json')
    if (result['restored_numeric_sha256'] != manifest['snapshot_numeric_sha256']
            or result['final_numeric_sha256'] != manifest['snapshot_numeric_sha256']
            or not result['physical_fields_unchanged']
            or not all(result['physical_fields_unchanged'].values())):
        raise ValueError('Candidate physical-state checks failed')
    verify_files(root, result['files'])
    return result


def save_reference(job, images, output, *, provenance, runtime=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if set(images) != set(CAMERAS) or any(
            x.shape != (256, 256, 3) or x.dtype != np.uint8 for x in images.values()):
        raise ValueError('Three original-resolution uint8 RGB cameras required')
    manifest = Path(job['path']) / 'manifest.json'
    if sha(manifest) != job['snapshot_manifest_sha256']:
        raise ValueError('Snapshot changed')
    np.savez_compressed(output / 'rgb.npz', **images)
    record = dict(format=FORMAT, status='FROZEN_PENDING_INDEPENDENT_VALIDATION',
                  initial_id=job['initial_id'], snapshot_manifest_sha256=sha(manifest),
                  numeric_sha256=read(manifest)['snapshot_numeric_sha256'],
                  rgb_sha256=sha(output / 'rgb.npz'), provenance=provenance,
                  runtime=runtime, historical_reference_replaced=False,
                  interpretation='New evaluation reference; old RGB is retained as a separate audit')
    write(output / 'reference.json', record)
    return record


def reference_from_cache(job, catalog, cache, output, old_plan_sha256):
    result = validate_candidate(job, catalog, cache, old_plan_sha256)
    views = [v for v in result['views'] if v['role'] == 'reference']
    if len(views) != 1:
        raise ValueError('Exactly one canonical view required')
    images = {}
    for camera, name in zip(CAMERAS, views[0]['images']):
        with Image.open(Path(cache) / name) as image:
            images[camera] = np.asarray(image).copy()
    return save_reference(job, images, output, provenance=dict(
        kind='immutable_existing_current_runtime_cache',
        source_summary=str(Path(cache) / 'summary.json'), source_summary_sha256=sha(Path(cache) / 'summary.json')))


def load_reference(job, reference_file):
    reference_file = Path(reference_file)
    record = read(reference_file)
    manifest = Path(job['path']) / 'manifest.json'
    if (record['format'] != FORMAT or record['initial_id'] != job['initial_id']
            or record['snapshot_manifest_sha256'] != job['snapshot_manifest_sha256']
            or sha(manifest) != job['snapshot_manifest_sha256']
            or record['numeric_sha256'] != read(manifest)['snapshot_numeric_sha256']
            or record['rgb_sha256'] != sha(reference_file.parent / 'rgb.npz')):
        raise ValueError('Current-render reference integrity/identity failure')
    with np.load(reference_file.parent / 'rgb.npz', allow_pickle=False) as arrays:
        if set(arrays.files) != set(CAMERAS):
            raise ValueError('Wrong reference cameras')
        images = {k: arrays[k].copy() for k in CAMERAS}
    if any(x.shape != (256, 256, 3) or x.dtype != np.uint8 for x in images.values()):
        raise ValueError('Wrong reference image contract')
    return record, images


def assert_validation(job, reference_file, receipt):
    def within_limits(check):
        return check['pass'] and all(np.isfinite(check[key]) and 0 <= check[key] <= value
                                     for key, value in LIMITS.items())
    if (receipt['status'] != 'PASS' or receipt['initial_id'] != job['initial_id']
            or receipt['reference_sha256'] != sha(reference_file)
            or receipt['pixel_gate'] != LIMITS or receipt['repeats'] != 4
            or receipt['numeric_start'] != receipt['numeric_end']
            or receipt['numeric_start'] != read(reference_file)['numeric_sha256']
            or len(receipt['checks']) != 4
            or any(set(row) != set(CAMERAS) for row in receipt['checks'])
            or not all(within_limits(c) for row in receipt['checks'] for c in row.values())):
        raise ValueError('Independent reference validation failed')
