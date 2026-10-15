"""Explicit, resumable worker for one shard of a frozen matched-data plan.

Owns execution and receipts only. Rendering belongs to the benchmark adapter;
candidate assignment belongs to the research data module.
"""

import argparse
from collections import Counter
import os
from pathlib import Path
import traceback

from AlphaBrain.research.dsol.data.artifacts import read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--job-id', action='append', help='Explicit episode job(s), for bounded validation')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute or not 0 <= args.shard_index < args.shard_count or args.shard_count > 8:
        parser.error('Explicit --execute and valid shard in 1..8 required')
    if args.limit is not None and args.limit < 1:
        parser.error('Positive limit required')
    plan = read(args.plan)
    fingerprint = sha(args.plan)
    for path, expected in plan['implementation_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Implementation differs from frozen data plan: ' + path)
    from benchmarks.Robocasa365.training_render import render_episode
    jobs = plan['jobs'][args.shard_index::args.shard_count]
    if args.job_id:
        requested = set(args.job_id)
        jobs = [job for job in jobs if job['job_id'] in requested]
        if {job['job_id'] for job in jobs} != requested:
            raise ValueError('Requested job is absent from this shard')
    if args.limit:
        jobs = jobs[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / f'worker-{args.shard_index}.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    records = []
    consecutive_errors = 0
    try:
        for job in jobs:
            dest = args.output / 'episodes' / job['job_id']
            result_file = dest / 'summary.json'
            try:
                if result_file.exists():
                    result = read(result_file)
                    if result.get('plan_sha256') != fingerprint or result.get('job_id') != job['job_id']:
                        raise ValueError('Existing result has different provenance')
                    # Source and image hashes are rechecked before training release.
                elif dest.exists():
                    raise FileExistsError('Partial output requires audit; not overwriting ' + str(dest))
                else:
                    result = render_episode(job, plan['catalog'], plan['dataset_root'], dest, fingerprint)
                record = dict(job_id=job['job_id'], status=result['status'], seconds=result.get('seconds'))
                consecutive_errors = 0
            except Exception as exc:
                record = dict(job_id=job['job_id'], status='ERROR', error=f'{type(exc).__name__}: {exc}')
                traceback.print_exc()
                error_path = args.output / 'errors' / f"{job['job_id']}.json"
                if not error_path.exists():
                    write(error_path, record)
                consecutive_errors += 1
            records.append(record)
            write(args.output / f'worker-{args.shard_index}.json', dict(plan_sha256=fingerprint,
                complete=False, total=len(jobs), counts=dict(Counter(r['status'] for r in records)), records=records))
            print(record, flush=True)
            if consecutive_errors >= 3:
                raise RuntimeError('Three consecutive failures; stop shard for review, do not skip population')
        write(args.output / f'worker-{args.shard_index}.json', dict(plan_sha256=fingerprint,
            complete=True, total=len(jobs), counts=dict(Counter(r['status'] for r in records)), records=records))
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
