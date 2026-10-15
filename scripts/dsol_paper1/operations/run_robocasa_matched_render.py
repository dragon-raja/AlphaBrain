"""Run the frozen paired-data rendering plan on explicit GPU shards.

Does not stop other processes, update model weights, or run evaluation. Any
missing/error episode blocks the all-data completion gate.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess

from AlphaBrain.research.dsol.data.artifacts import read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    gpu_ids = args.gpus.split(',')
    if not args.execute or not 1 <= len(gpu_ids) <= 8 or len(set(gpu_ids)) != len(gpu_ids) or not all(x.isdigit() for x in gpu_ids):
        parser.error('Explicit --execute and 1..8 distinct GPUs required')
    if args.output.exists():
        raise FileExistsError('Existing queue needs explicit audit before resume: ' + str(args.output))
    plan = read(args.plan)
    root = Path(__file__).resolve().parents[3]
    args.output.mkdir(parents=True)
    write(args.output / 'launch.json', dict(plan=str(args.plan), plan_sha256=sha(args.plan),
        controller_sha256=sha(__file__), python=str(args.python), gpus=gpu_ids,
        tasks=plan['counts']['tasks'], episodes=len(plan['jobs']), pid=os.getpid(),
        training_started=False, no_existing_processes_modified=True))

    def worker(index, gpu):
        command = [str(args.python), '-m', 'scripts.dsol_paper1.training.render_robocasa_matched',
            '--plan', str(args.plan), '--output', str(args.output), '--shard-index', str(index),
            '--shard-count', str(len(gpu_ids)), '--execute']
        with (args.output / f'worker-{index}.log').open('x') as stream:
            child = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                env={**os.environ, 'PYTHONPATH': str(root), 'CUDA_VISIBLE_DEVICES': gpu,
                     'MUJOCO_EGL_DEVICE_ID': gpu, 'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl'})
            write(args.output / f'process-{index}.json', dict(pid=child.pid, gpu=gpu, command=command))
            code = child.wait()
        path = args.output / f'worker-{index}.json'
        status = read(path) if path.exists() else {}
        return dict(shard=index, gpu=gpu, exit_code=code, **status)

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = [pool.submit(worker, i, gpu) for i, gpu in enumerate(gpu_ids)]
        results = [f.result() for f in futures]
    records = [r for result in results for r in result.get('records', [])]
    expected = {job['job_id'] for job in plan['jobs']}
    passed = {r['job_id'] for r in records if r['status'] == 'PASS'}
    complete = (len(records) == len(expected) and passed == expected
                and all(r['exit_code'] == 0 for r in results))
    write(args.output / 'completion.json', dict(all_pass=complete, expected=len(expected),
        passed=len(passed), missing_or_failed=sorted(expected-passed),
        worker_exit_codes=[r['exit_code'] for r in results], training_started=False,
        next_gate='Aggregate hashes, verify real loader and eight-GPU real-data updates before training'))
    if not complete:
        raise RuntimeError('Rendering incomplete; no automatic training permitted')


if __name__ == '__main__':
    main()
