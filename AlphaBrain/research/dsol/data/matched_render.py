"""Fail-closed identity checks for matched canonical/exterior render receipts."""

from pathlib import Path
from .artifacts import read


def select_episode_root(job_id, primary, recoveries=()):
    """Use exactly one receipt; never replace an already successful original."""
    primary = Path(primary) / 'episodes' / job_id
    candidate_paths = [Path(root) / 'episodes' / job_id for root in recoveries]
    recovered = [p for p in candidate_paths if (p / 'summary.json').is_file()
                 and read(p / 'summary.json').get('status') == 'PASS']
    if (primary / 'summary.json').is_file() and read(primary / 'summary.json').get('status') == 'PASS':
        if recovered:
            raise ValueError('Recovery must not override a successful original episode')
        return primary
    if len(recovered) != 1:
        raise ValueError(f'Expected one successful recovery for {job_id}, found {len(recovered)}')
    return recovered[0]


def validate_episode_receipt(job, result, plan_sha):
    if (result.get('status') != 'PASS' or result.get('job_id') != job['job_id']
            or result.get('plan_sha256') != plan_sha or result.get('anchor_frame') != 0
            or result.get('wrist_shared_exact') is not True):
        raise ValueError('Incomplete or mismatched episode provenance')
    if result.get('episode_index') != job['episode_index']:
        raise ValueError('Episode identity mismatch')
    expected = {r['source_window_id']: r for r in job['rows']}
    records = result['records']
    if len(records) != len(expected) or {r['source_window_id'] for r in records} != set(expected):
        raise ValueError('Missing or duplicated source windows')
    for record in records:
        source = expected[record['source_window_id']]
        if any(record[key] != source[key] for key in ('frame_index', 'training_pair_index')):
            raise ValueError('Source window/frame mismatch')
        canonical, broad = record['canonical_exposures'], record['broad_exposures']
        if len(canonical) != 2 or canonical[0] != canonical[1] or canonical[0]['view_id'] != 'canonical':
            raise ValueError('Canonical arm must contain two identical exposures')
        if [r['view_id'] for r in broad] != job['view_ids']:
            raise ValueError('Episode installation changed or evaluation views used')
        reference_wrist = canonical[0]['image_files']['wrist']
        for exposure in canonical + broad:
            if set(exposure['image_files']) != {'left', 'right', 'wrist'}:
                raise ValueError('Expected exactly three RGB inputs')
            if exposure['image_files']['wrist'] != reference_wrist:
                raise ValueError('Wrist pixels must be shared across all exposures')
    return records
