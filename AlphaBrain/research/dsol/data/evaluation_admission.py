"""Resolve an initial only through a completed current-render cache release.

This admits data for engineering evaluation; it is not a scientific rollout
protocol release (noise population and method definitions remain separate).
"""
from pathlib import Path

from .artifacts import read, sha
from .render_reference import LIMITS, assert_validation, load_reference


def resolve_initial(plan_file, release_file, initial_id):
    plan, release = read(plan_file), read(release_file)
    jobs, records = plan['jobs'], release['records']
    if (release.get('complete') is not True or release.get('eligible_for_formal_release') is not True
            or release['plan_sha256'] != sha(plan_file) or plan['pixel_gate'] != LIMITS
            or release['total'] != len(jobs) or release['counts'] != {'PASS': len(jobs)}
            or len(records) != len(jobs) or not jobs):
        raise ValueError('Incomplete or mismatched current-reference release')
    job_ids = [job['initial_id'] for job in jobs]
    record_ids = [record['initial_id'] for record in records]
    if (len(set(job_ids)) != len(job_ids) or len(set(record_ids)) != len(record_ids)
            or set(job_ids) != set(record_ids) or any(r['status'] != 'PASS' for r in records)):
        raise ValueError('Release population is not the frozen initial population')
    selected = [job for job in jobs if job['initial_id'] == initial_id]
    if len(selected) != 1:
        raise ValueError('Initial is absent from release')
    job = selected[0]
    record = next(r for r in records if r['initial_id'] == initial_id)
    for path_key, hash_key in (('reference_file', 'reference_sha256'),
                               ('validation_file', 'validation_sha256')):
        if sha(record[path_key]) != record[hash_key]:
            raise ValueError('Released reference or validation has changed')
    candidate_summary = Path(record['candidate_root']) / 'summary.json'
    if sha(candidate_summary) != record['candidate_summary_sha256']:
        raise ValueError('Released candidate summary has changed')
    reference, images = load_reference(job, record['reference_file'])
    assert_validation(job, record['reference_file'], read(record['validation_file']))
    return dict(job=job, record=record, reference=reference, images=images,
                plan_sha256=sha(plan_file), release_sha256=sha(release_file))


def load_candidate(admission, view_id):
    """Read a checksummed, explicitly selected installation and its cached RGB."""
    import numpy as np
    from PIL import Image
    from .evaluation_preparation import verify_files

    root = Path(admission['record']['candidate_root'])
    summary = read(root / 'summary.json')
    if (summary['status'] != 'PASS'
            or summary['initial_id'] != admission['job']['initial_id']
            or summary['snapshot_manifest_sha256'] != admission['job']['snapshot_manifest_sha256']):
        raise ValueError('Candidate cache identity mismatch')
    matches = [(i, view) for i, view in enumerate(summary['views']) if view['view_id'] == view_id]
    if len(matches) != 1:
        raise ValueError('Candidate absent or duplicated')
    index, view = matches[0]
    names = ['geometry.npz', *view['images']]
    verify_files(root, {name: summary['files'][name] for name in names})
    with np.load(root / 'geometry.npz', allow_pickle=False) as geometry:
        if list(geometry['view_ids']) != [v['view_id'] for v in summary['views']]:
            raise ValueError('Candidate geometry ordering mismatch')
        positions = geometry['camera_positions'][index].copy()
        quaternions = geometry['camera_quaternions'][index].copy()
    images = []
    for name in view['images']:
        with Image.open(root / name) as image:
            images.append(np.array(image))
    return dict(view=view, positions=positions, quaternions=quaternions, images=images,
                geometry_sha256=summary['files']['geometry.npz']['sha256'])
