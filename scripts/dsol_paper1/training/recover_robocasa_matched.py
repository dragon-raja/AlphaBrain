"""Repair explicit missing episodes with bounded warm-up placement retries.

Uses the unchanged scientific plan and render body; records the sole permitted
implementation change (warm-up reset retries), without overwriting failures.
"""

import argparse
from pathlib import Path

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.data.matched_render import validate_episode_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--initialization-attempts', type=int, default=5)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute or not 1 <= args.initialization_attempts <= 5:
        parser.error('Explicit --execute and bounded 1..5 initialization attempts required')
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = read(args.plan)
    job, = [r for r in plan['jobs'] if r['job_id'] == args.job_id]
    root = Path(__file__).resolve().parents[3]
    renderer = root / 'benchmarks/Robocasa365/training_render.py'
    for path, expected in plan['implementation_sha256'].items():
        if Path(path) != renderer and sha(path) != expected:
            raise ValueError('Unexpected implementation change: ' + path)
    receipt = dict(job_id=args.job_id, plan_sha256=sha(args.plan),
        renderer_before=plan['implementation_sha256'][str(renderer)], renderer_after=sha(renderer),
        reset_helper_sha256=sha(root / 'benchmarks/Robocasa365/recorded_initialization.py'),
        entrypoint_sha256=sha(__file__), initialization_attempts=args.initialization_attempts,
        source_window_or_camera_changes=False, original_failures_preserved=True)
    args.output.mkdir(parents=True)
    write(args.output / 'recovery.json', receipt)
    from benchmarks.Robocasa365.training_render import render_episode
    try:
        result = render_episode(job, plan['catalog'], plan['dataset_root'],
            args.output / 'episodes' / job['job_id'], receipt['plan_sha256'],
            initialization_attempts=args.initialization_attempts)
        validate_episode_receipt(job, result, receipt['plan_sha256'])
        write(args.output / 'completion.json', dict(status='PASS', job_id=job['job_id'],
            source_windows=len(result['records']), initialization=result['initialization']))
    except Exception as exc:
        write(args.output / 'completion.json', dict(status='ERROR', job_id=job['job_id'],
            error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__ == '__main__':
    main()
