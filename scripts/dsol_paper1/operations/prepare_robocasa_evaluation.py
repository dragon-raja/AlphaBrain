"""Audit a frozen population and optionally cache initial-state candidate images.

Explicit CLI owns execution; pure audit and simulation live in their own modules.
Existing PASS entries are hash-verified, never silently overwritten.
"""

import argparse
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys
import traceback

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.data.evaluation_preparation import prepare_population, verify_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('audit')
    audit.add_argument('--population', required=True, type=Path)
    audit.add_argument('--catalog-plan', required=True, type=Path)
    audit.add_argument('--output', required=True, type=Path)
    audit.add_argument('--confirmed-pixel-gate', type=Path, help='Explicitly adopt an independently confirmed gate proposal')
    render = sub.add_parser('render')
    render.add_argument('--plan', required=True, type=Path)
    render.add_argument('--output', required=True, type=Path)
    render.add_argument('--initial-id', action='append')
    render.add_argument('--execute', action='store_true')
    render.add_argument('--continue-on-error', action='store_true', help='Keep failed initials quarantined, process all others; never release incomplete data')
    render.add_argument('--reuse-plan', type=Path)
    render.add_argument('--reuse-cache', type=Path)
    one = sub.add_parser('render-one', help='Isolated child for one initial; normally called by render')
    one.add_argument('--plan', required=True, type=Path)
    one.add_argument('--output', required=True, type=Path)
    one.add_argument('--initial-id', required=True, action='append')
    one.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.command == 'audit':
        if args.output.exists():
            raise FileExistsError(args.output)
        plan = prepare_population(args.population, args.catalog_plan)
        if args.confirmed_pixel_gate:
            from AlphaBrain.research.dsol.data.render_admission import confirmed_pixel_gate
            plan['pixel_gate'] = confirmed_pixel_gate(args.confirmed_pixel_gate)
        root = Path(__file__).resolve().parents[3]
        paths = [root / 'AlphaBrain/research/dsol/data/evaluation_preparation.py',
                 root / 'AlphaBrain/research/dsol/data/cache_recovery.py',
                 root / 'AlphaBrain/research/dsol/data/render_admission.py',
                 root / 'AlphaBrain/research/dsol/acquisition/camera_rig.py',
                 root / 'benchmarks/Robocasa365/evaluation_render.py',
                 root / 'benchmarks/Robocasa365/camera_rig.py',
                 root / 'benchmarks/Robocasa365/snapshot_bundle.py',
                 root / 'benchmarks/Robocasa365/snapshot.py', Path(__file__).resolve()]
        plan['implementation_sha256'] = {str(p): sha(p) for p in paths}
        write(args.output, plan)
        print(plan['counts'], flush=True)
        return
    if not args.execute:
        parser.error('Rendering requires explicit --execute')
    plan = read(args.plan)
    fingerprint = sha(args.plan)
    if 'pixel_gate' in plan:
        from AlphaBrain.research.dsol.data.render_admission import confirmed_pixel_gate
        if confirmed_pixel_gate(plan['pixel_gate']['confirmation_result']) != plan['pixel_gate']:
            raise ValueError('Pixel gate proposal changed after freezing')
    for path, expected in plan['implementation_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Implementation changed after plan freeze: ' + path)
    jobs = plan['jobs']
    if args.initial_id:
        wanted = set(args.initial_id)
        jobs = [job for job in jobs if job['initial_id'] in wanted]
        if {j['initial_id'] for j in jobs} != wanted:
            raise ValueError('Unknown initial ID')
    if args.command == 'render-one':
        if len(jobs) != 1:
            parser.error('render-one requires exactly one initial ID')
        from benchmarks.Robocasa365.evaluation_render import render_initial
        result = render_initial(jobs[0], plan['catalog'], args.output / 'initials' / jobs[0]['initial_id'], fingerprint,
                                pixel_gate=plan.get('pixel_gate'))
        print(dict(initial_id=jobs[0]['initial_id'], status=result['status'], seconds=result['seconds']), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if bool(args.reuse_plan) != bool(args.reuse_cache):
        parser.error('--reuse-plan and --reuse-cache must be supplied together')
    lock = args.output / 'worker.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    records = []
    try:
        if args.reuse_plan:
            from AlphaBrain.research.dsol.data.cache_recovery import reuse_initials
            copied = reuse_initials(args.reuse_plan, args.plan, args.reuse_cache, args.output)
            write(args.output / 'reuse.json', dict(source_plan=str(args.reuse_plan), records=copied))
        for job in jobs:
            dest = args.output / 'initials' / job['initial_id']
            try:
                if (dest / 'summary.json').exists():
                    result = read(dest / 'summary.json')
                    if result['status'] == 'ERROR' and args.continue_on_error:
                        if result['plan_sha256'] != fingerprint or result['initial_id'] != job['initial_id']:
                            raise ValueError('Quarantined error identity mismatch')
                        raise RuntimeError('Preserved failed initial: ' + result.get('error', 'unknown'))
                    if result['status'] not in ('PASS', 'HOLD_CANONICAL_PIXELS') or result['plan_sha256'] != fingerprint or result['initial_id'] != job['initial_id']:
                        raise ValueError('Existing output is not a matching PASS release; inspect separately')
                    verify_files(dest, result['files'])
                else:
                    # Gym / MuJoCo retains some GL resources across environment
                    # destruction; child exit bounds GPU memory for a long queue.
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), 'render-one',
                        '--plan', str(args.plan), '--output', str(args.output),
                        '--initial-id', job['initial_id'], '--execute'], check=True)
                    result = read(dest / 'summary.json')
                records.append(dict(initial_id=job['initial_id'], status=result['status'], seconds=result['seconds']))
            except Exception as exc:
                records.append(dict(initial_id=job['initial_id'], status='ERROR', error=str(exc)))
                traceback.print_exc()
                write(args.output / 'status.json', dict(complete=False, total=len(jobs), records=records))
                if not args.continue_on_error:
                    raise
                continue  # Keep ERROR in population; never count as released.
            write(args.output / 'status.json', dict(complete=False, total=len(jobs), records=records))
            print(records[-1], flush=True)
        write(args.output / 'status.json', dict(complete=True, total=len(jobs),
              full_population=len(jobs) == len(plan['jobs']),
              eligible_for_formal_release=(len(jobs) == len(plan['jobs'])
                                           and all(r['status'] == 'PASS' for r in records)),
              counts=dict(Counter(r['status'] for r in records)), records=records))
        if any(r['status'] != 'PASS' for r in records):
            raise SystemExit(2)  # Finished enumeration is not successful release.
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
