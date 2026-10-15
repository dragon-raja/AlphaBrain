"""Freeze and execute an independent seen-scene numerical render calibration."""

import argparse
from pathlib import Path
import subprocess
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from AlphaBrain.research.dsol.analysis.render_tolerance import calibration_population, confirmation_population, MEAN_CANDIDATES
from AlphaBrain.research.dsol.data.artifacts import read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--population', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--confirmation-of', type=Path)
    parser.add_argument('--fixed-mean', type=float, choices=MEAN_CANDIDATES)
    parser.add_argument('--fraction-exclude-plan', type=Path, action='append',
                        help='Fraction calibration on fresh seen layouts, excluding all listed prior plans')
    parser.add_argument('--gpus', default=os.environ.get('MUJOCO_EGL_DEVICE_ID', '0'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    population = read(args.population)
    if args.fraction_exclude_plan:
        if args.confirmation_of or args.fixed_mean is not None:
            parser.error('Fraction calibration is separate from mean confirmation')
        from AlphaBrain.research.dsol.analysis.render_tolerance import fraction_calibration_population
        excluded = {r['layout_id'] for path in args.fraction_exclude_plan for r in read(path)['initials']}
        rows = fraction_calibration_population(population, excluded)
    elif args.confirmation_of:
        if args.fixed_mean is None:
            parser.error('Confirmation requires a fixed, predeclared --fixed-mean')
        development = read(args.confirmation_of / 'plan.json')
        rows = confirmation_population(population, {r['layout_id'] for r in development['initials']})
    else:
        rows = calibration_population(population)
    gpus = [int(value) for value in args.gpus.split(',')]
    if len(gpus) > 2 or len(set(gpus)) != len(gpus) or any(g not in range(8) for g in gpus):
        parser.error('One or two distinct GPU IDs in 0..7 required')
    root = Path(__file__).resolve().parents[3]
    probe = root / 'scripts/dsol_paper1/diagnostics/robocasa_render_repeatability.py'
    sources = [Path(__file__), probe, root / 'benchmarks/Robocasa365/render_diagnostics.py',
               root / 'benchmarks/Robocasa365/evaluation_render.py',
               root / 'benchmarks/Robocasa365/snapshot_bundle.py',
               root / 'benchmarks/Robocasa365/snapshot.py',
               root / 'AlphaBrain/research/dsol/analysis/render_tolerance.py']
    plan = dict(mode='confirmation' if args.confirmation_of else 'calibration',
                development_plan=str(args.confirmation_of) if args.confirmation_of else None,
                fixed_mean_abs=args.fixed_mean, gpus=gpus,
                population_sha256=sha(args.population), initials=rows, independent_processes=2,
                repeats_per_stage=4, stages=['wrapper', 'direct', 'restore_each'],
                mean_gate_candidates=list(MEAN_CANDIDATES), margin=1.25,
                unchanged_limits=dict(max_abs=2, fraction_gt1=.0001, fraction_gt2=0.),
                provenance={str(p.resolve()): sha(p) for p in sources},
                target_test_data_used_for_threshold=False,
                scope='Proposal only; never applies a new gate or edits an old failure')
    if args.fraction_exclude_plan:
        from AlphaBrain.research.dsol.analysis.render_tolerance import FRACTION_CANDIDATES
        plan.update(mode='fraction_calibration', fraction_gate_candidates=list(FRACTION_CANDIDATES),
                    excluded_plans={str(p): sha(p) for p in args.fraction_exclude_plan},
                    unchanged_limits=dict(max_abs=2, mean_abs=.005, fraction_gt2=0.))
    write(args.output / 'plan.json', plan)
    if not args.execute:
        print('Plan only; no rendering started')
        return
    records = []

    def worker(gpu, assignments):
        finished = []
        for row, repetition in assignments:
            name = f"{row['role']}/{row['task']}/process-{repetition}"
            dest = args.output / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.with_suffix('.log').open('x') as log:
                process = subprocess.run([str(args.python), str(probe), '--snapshot', row['path'],
                    '--output', str(dest), '--repeats', '4', '--calibration', '--execute'],
                    stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, MUJOCO_EGL_DEVICE_ID=str(gpu)))
            finished.append(dict(job=name, returncode=process.returncode, gpu=gpu))
            write(args.output / f'worker-{gpu}.json', dict(complete=False, records=finished, total=len(assignments)))
            print(finished[-1], flush=True)
            if process.returncode:
                raise RuntimeError('Calibration failed; retain records, do not skip initial')
        write(args.output / f'worker-{gpu}.json', dict(complete=True, records=finished, total=len(assignments)))
        return finished

    jobs = [(row, repetition) for row in rows for repetition in range(2)]
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, jobs[i::len(gpus)]) for i, gpu in enumerate(gpus)]
        for future in as_completed(futures):
            records.extend(future.result())
    write(args.output / 'status.json', dict(complete=True, records=records, total=len(rows) * 2))


if __name__ == '__main__':
    main()
