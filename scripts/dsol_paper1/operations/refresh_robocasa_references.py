"""Authorized reference transition: freeze, independently validate, then index caches."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.data.render_reference import prepare_plan, assert_validation, validate_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'run', 'capture', 'validate', 'candidates'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, help='Explicit preserved engineering revision; defaults to ROOT/plan.json')
    parser.add_argument('--old-plan', type=Path)
    parser.add_argument('--old-cache', type=Path)
    parser.add_argument('--initial-id')
    parser.add_argument('--gpus', nargs='+', type=int, default=[6, 7])
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--authorized-current-reference', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if args.command == 'freeze':
        if not args.authorized_current_reference or not args.old_plan or not args.old_cache:
            parser.error('Explicit reference-transition authorization and original inputs required')
        import shutil
        old = read(args.old_plan)
        for name, expected in old['implementation_sha256'].items():
            if sha(name) != expected:
                preserved = args.root / 'source-before' / Path(name).name
                if not preserved.exists() or sha(preserved) != expected:
                    raise ValueError('Unaccounted implementation change: ' + name)
        paths = list(old['implementation_sha256']) + [str(root / p) for p in (
            'AlphaBrain/research/dsol/data/render_reference.py',
            'benchmarks/Robocasa365/reference_render.py')]+[str(Path(__file__).resolve())]
        snapshots = {}
        for name in paths:
            p = Path(name)
            dest = args.root / 'source_snapshot' / p.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise FileExistsError(dest)
            shutil.copy2(p, dest)
            snapshots[name] = sha(p)
        plan = prepare_plan(args.old_plan, args.old_cache, args.root, snapshots)
        print(plan['counts'], flush=True)
        return
    if not args.execute:
        parser.error('Explicit --execute required')
    plan_file = args.plan or args.root / 'plan.json'
    plan = read(plan_file)
    for path, digest in plan['implementation_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Frozen implementation changed: ' + path)
    if sha(plan['old_plan']) != plan['old_plan_sha256']:
        raise ValueError('Historical plan changed')
    if args.command != 'run':
        matches = [j for j in plan['jobs'] if j['initial_id'] == args.initial_id]
        if len(matches) != 1:
            parser.error('Exactly one known initial ID required')
        job = matches[0]
        reference = Path(job['reference_file'])
        from benchmarks.Robocasa365.reference_render import inspect_reference
        if args.command == 'capture':
            inspect_reference(job, reference.parent)
        elif args.command == 'validate':
            result = inspect_reference(job, args.root / 'validation' / (job['initial_id'] + '.json'), reference_file=reference)
            assert_validation(job, reference, result)
        else:
            from benchmarks.Robocasa365.evaluation_render import render_initial
            assert_validation(job, reference, read(args.root / 'validation' / (job['initial_id'] + '.json')))
            render_initial(job, plan['catalog'], args.root / 'new_candidates' / job['initial_id'],
                           sha(plan_file), pixel_gate=plan['pixel_gate'], reference_file=reference)
        return
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        parser.error('Unique explicit GPU IDs required')
    lock_file = args.root / 'worker.lock'
    with lock_file.open('x') as stream:
        stream.write(str(os.getpid()))
    records, mutex = {}, threading.Lock()

    def update(record):
        with mutex:
            records[record['initial_id']] = record
            write(args.root / 'status.json', dict(complete=False, total=len(plan['jobs']),
                eligible_for_formal_release=False, counts=dict(Counter(r['status'] for r in records.values())),
                records=list(records.values())))
            print(record, flush=True)

    def worker(index, gpu):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_EGL_DEVICE_ID=str(gpu),
                   MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        for job in plan['jobs'][index::len(args.gpus)]:
            identity = job['initial_id']
            started = time.monotonic()
            try:
                source = Path(job['source_cache'])
                if sha(source / 'summary.json') != job['source_summary_sha256']:
                    raise ValueError('Old cache receipt changed')
                reference = Path(job['reference_file'])
                if job['frozen_reference_sha256'] is not None and sha(reference) != job['frozen_reference_sha256']:
                    raise ValueError('Frozen cache-derived reference changed')
                def child(command):
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), command,
                        '--root', str(args.root), '--plan', str(plan_file), '--initial-id', identity, '--execute'], env=env, check=True)
                if not reference.exists():
                    if job['previous_status'] == 'PASS':
                        raise ValueError('Frozen cache-derived reference missing')
                    child('capture')
                validation_file = args.root / 'validation' / (identity + '.json')
                if not validation_file.exists():
                    child('validate')
                assert_validation(job, reference, read(validation_file))
                if job['previous_status'] == 'PASS':
                    candidate = source
                    candidate_plan_sha = plan['old_plan_sha256']
                else:
                    candidate = args.root / 'new_candidates' / identity
                    if not (candidate / 'summary.json').exists():
                        child('candidates')
                    candidate_plan_sha = sha(plan_file)
                validate_candidate(job, plan['catalog'], candidate, candidate_plan_sha)
                update(dict(initial_id=identity, status='PASS', candidate_root=str(candidate),
                    candidate_summary_sha256=sha(candidate / 'summary.json'), reference_file=str(reference),
                    reference_sha256=sha(reference), validation_file=str(validation_file),
                    validation_sha256=sha(validation_file), reused_candidates=job['previous_status']=='PASS',
                    seconds=time.monotonic()-started))
            except Exception as exc:
                update(dict(initial_id=identity, status='ERROR', error=str(exc), seconds=time.monotonic()-started))
    try:
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            futures = [pool.submit(worker, i, gpu) for i, gpu in enumerate(args.gpus)]
            for future in futures:
                future.result()
        ordered = [records[j['initial_id']] for j in plan['jobs']]
        passed = all(r['status'] == 'PASS' for r in ordered)
        final = dict(complete=True, total=len(plan['jobs']), plan_sha256=sha(plan_file),
            eligible_for_formal_release=passed, counts=dict(Counter(r['status'] for r in ordered)), records=ordered,
            scope='Initial candidate cache; no rollout success rates; historical failures retained')
        write(args.root / 'status.json', final)
        if passed:
            write(args.root / 'release.json', final)
        else:
            raise SystemExit(2)
    finally:
        lock_file.unlink()


if __name__ == '__main__':
    main()
