"""Reuse immutable initial caches after a queue-only implementation change."""

from pathlib import Path
import shutil

from .artifacts import read, sha, write
from .evaluation_preparation import verify_files


def reuse_initials(old_plan_file, new_plan_file, old_cache, new_cache):
    old, new = read(old_plan_file), read(new_plan_file)
    if {k: v for k, v in old.items() if k != 'implementation_sha256'} != {
            k: v for k, v in new.items() if k != 'implementation_sha256'}:
        raise ValueError('Cannot reuse cache across scientific protocol changes')
    for path, digest in old['implementation_sha256'].items():
        if path.endswith('/scripts/dsol_paper1/operations/prepare_robocasa_evaluation.py'):
            continue  # Only queue orchestration may change; renderer must not.
        if new['implementation_sha256'].get(path) != digest:
            raise ValueError('Cannot reuse changed renderer or scientific implementation: ' + path)
    old_cache, new_cache = Path(old_cache), Path(new_cache)
    copied = []
    for job in new['jobs']:
        src = old_cache / 'initials' / job['initial_id']
        dst = new_cache / 'initials' / job['initial_id']
        if not (src / 'summary.json').exists() or dst.exists():
            continue
        result = read(src / 'summary.json')
        if (result['plan_sha256'] != sha(old_plan_file) or result['initial_id'] != job['initial_id']
                or result.get('snapshot_manifest_sha256') != job['snapshot_manifest_sha256']
                or result['status'] not in ('PASS', 'ERROR', 'HOLD_CANONICAL_PIXELS')):
            raise ValueError('Source cache identity/status mismatch')
        verify_files(src, result['files'])
        if result['status'] == 'PASS' and len(result['views']) != len(new['catalog']['views']):
            raise ValueError('Incomplete PASS cache')
        shutil.copytree(src, dst)
        # The original receipt remains immutable at source. The new receipt
        # explicitly cites it; failures remain failures, never retried to PASS.
        result['reuse_source'] = dict(summary=str((src / 'summary.json').resolve()),
                                     sha256=sha(src / 'summary.json'), plan_sha256=sha(old_plan_file))
        result['plan_sha256'] = sha(new_plan_file)
        write(dst / 'summary.json', result)
        copied.append(dict(initial_id=job['initial_id'], status=result['status']))
    return copied
