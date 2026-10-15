"""Explicit bounded, resumable training-only camera-support calibration queue.

One predetermined ledger frame per Human300 task. This neither trains models
nor uses evaluation success labels; a PASS is not physical camera feasibility.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import time

from AlphaBrain.research.dsol.data.artifacts import read, write, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--timeout-per-task', type=int, default=600)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute or not 1 <= args.limit <= 300 or not 30 <= args.timeout_per_task <= 1800:
        parser.error('Explicit --execute, limit 1..300 and bounded timeout required')
    cfg = read(args.config)
    ledger_path = Path(cfg['history']) / 'B_source_window_ledger_v1/source_windows.jsonl'
    selected = {}
    with ledger_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            selected.setdefault(row['dataset_identity'], row)
    jobs = [selected[k] for k in sorted(selected)][:args.limit]
    gpu_ids = args.gpus.split(',')
    if len(set(gpu_ids)) != len(gpu_ids) or not all(x.isdigit() for x in gpu_ids):
        parser.error('Distinct numeric GPU identifiers required')
    repo = Path(__file__).resolve().parents[3]
    script = repo / 'scripts/dsol_paper1/diagnostics/robocasa_training_replay.py'
    manifest = dict(config=cfg, config_sha256=sha(args.config), ledger_sha256=sha(ledger_path),
                    worker_sha256=sha(script), jobs=jobs, gpus=gpu_ids, timeout_per_task=args.timeout_per_task,
                    scope='All selected training tasks, no evaluation labels and no optimizer updates')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        if read(manifest_path) != manifest:
            raise ValueError('Cannot resume with different inputs/code/scope')
    else:
        write(manifest_path, manifest)

    def worker(gpu, shard):
        rows = []
        for row in shard:
            name = row['dataset_identity'].split('/')[1]
            destination = args.output / 'tasks' / name
            result_file = destination / 'summary.json'
            if result_file.exists():
                status = read(result_file).get('status', 'ERROR')
                rows.append(dict(task=name, status=status, reused=True))
                continue
            if destination.exists():
                rows.append(dict(task=name, status='INCOMPLETE_OUTPUT_REQUIRES_REVIEW'))
                continue
            log = args.output / 'logs' / f'{name}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            if log.exists():
                rows.append(dict(task=name, status='PREVIOUS_LOG_WITHOUT_RECEIPT_REQUIRES_REVIEW'))
                continue
            command = [str(args.python), str(script), '--dataset', str(Path(cfg['dataset_root']) / row['dataset_identity']),
                       '--episode', str(row['episode_index']), '--frame', str(row['frame_index']),
                       '--output', str(destination), '--rig-calibration', '--execute']
            start = time.monotonic()
            try:
                with log.open('x') as stream:
                    process = subprocess.run(command, cwd=repo, env={**os.environ, 'PYTHONPATH': str(repo),
                        'CUDA_VISIBLE_DEVICES': gpu, 'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl'},
                        stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout_per_task)
                status = read(result_file).get('status', 'ERROR') if result_file.exists() else 'NO_RECEIPT'
                record = dict(task=name, status=status, exit_code=process.returncode, seconds=time.monotonic()-start)
            except subprocess.TimeoutExpired:
                record = dict(task=name, status='TIMEOUT', seconds=time.monotonic()-start)
            rows.append(record)
            write(args.output / f'worker-{gpu}.json', dict(rows=rows, total=len(shard)))
            print(json.dumps(record), flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = [pool.submit(worker, gpu, jobs[index::len(gpu_ids)]) for index, gpu in enumerate(gpu_ids)]
        records = [r for f in futures for r in f.result()]
    write(args.output / 'completion.json', dict(complete=len(records) == len(jobs),
        all_pass=all(r['status'] == 'PASS' for r in records), counts=dict(Counter(r['status'] for r in records)),
        records=records, scope=manifest['scope']))


if __name__ == '__main__':
    main()
