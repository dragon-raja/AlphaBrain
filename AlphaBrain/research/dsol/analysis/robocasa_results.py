"""Audited RoboCasa P0 results. CPU-only; never changes frozen evaluations.

The complete matrix is admitted only after every cell, rollout, noise address,
score payload and model identity has been checked. Outputs are new artifacts.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil

import numpy as np

from ..data.artifacts import read, write, sha
from ..data.evaluation_matrix import validate_plan, flow_noise
from ..data.rollout_replay import array_identity
from ..metrics.accel import compute_accel_scores


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_trial(trial, ref, identity, scores, horizon, noises):
    require(trial['provenance'] == identity and trial['complete_task_attempt'], 'trial provenance')
    require(type(trial['success']) is bool and trial['horizon'] == horizon, 'trial outcome/horizon')
    require(0 < trial['steps'] <= horizon, 'trial step range')
    require(trial['steps'] == horizon or trial['success'] or trial['terminated'] or trial['truncated'],
            'incomplete attempt without terminal condition')
    require(all(trial[k] == ref[k] for k in ('success', 'steps', 'seconds')), 'summary vs trial')
    require(trial['score_arrays_sha256'] == scores['arrays_sha256'], 'trial scoring identity')
    require(len(trial['plans']) == math.ceil(trial['steps'] / 25), 'replan count')
    require(trial['plans'][0]['input_arrays'] == scores['input_arrays'], 'first observation differs')
    require(all(math.isfinite(trial[k]) and trial[k] >= 0 for k in
                ('seconds', 'inference_seconds', 'environment_seconds')), 'invalid duration')
    for r, query in enumerate(trial['plans']):
        require(query['step'] == 25 * r and query['noise'] == noises[r], 'execution noise address')


def audit_worker(plan, reference, plan_hash, worker, output):
    root = Path(plan['source']).parent / 'workers' / worker['worker_id']
    progress = read(root / 'progress.json')
    expected_n = len(worker['initials']) * len(plan['view_ids'])
    require(progress['status'] == 'COMPLETE' and progress['completed_cells'] == expected_n
            and progress['completed_rollouts'] == 4 * expected_n, 'worker incomplete')
    jobs = {j['initial_id']: j for j in reference['jobs']}
    ids, views = worker['initials'], plan['view_ids']
    y = np.zeros((len(ids), len(views), 4), np.uint8)
    acc = np.zeros((len(ids), len(views), 8, 9), np.float64)
    steps = np.zeros_like(y, dtype=np.int16)
    elapsed = np.zeros_like(y, dtype=np.float64)
    degenerate = 0
    pixels_max = {k: 0. for k in ('max_abs', 'mean_abs', 'fraction_gt1', 'fraction_gt2')}
    roots = hashlib.sha256()
    paired_inputs = []
    for i, initial in enumerate(ids):
        job = jobs[initial]
        horizon = plan['task_horizons'][job['task']]
        noise = [[array_identity(flow_noise(initial, 'execution', n, r))
                  for r in range(math.ceil(horizon / 25))] for n in range(4)]
        score_noise = [array_identity(flow_noise(initial, 'score', n)) for n in range(8)]
        inputs = []
        actual_views = {p.name for p in (root / 'cells' / initial).iterdir() if p.is_dir()}
        require(actual_views == set(views), 'extra/missing candidate directory')
        for v, view in enumerate(views):
            cell = root / 'cells' / initial / view
            identity = dict(plan_sha256=plan_hash, worker_id=worker['worker_id'], pilot=False,
                            arm=worker['arm'], initial_id=initial, view_id=view,
                            snapshot_manifest_sha256=job['snapshot_manifest_sha256'])
            complete, scores = read(cell / 'complete.json'), read(cell / 'scores.json')
            require(complete['provenance'] == identity and scores['provenance'] == identity,
                    'cell provenance')
            require(complete['complete'] and len(complete['trials']) == 4, 'cell incomplete')
            require(complete['score_metadata_sha256'] == sha(cell / 'scores.json'), 'score json hash')
            require(scores['arrays_sha256'] == file_hash(cell / 'scores.npz'), 'score payload hash')
            require(scores['primary_accel_prefix'] == 3 and len(scores['rows']) == 8, 'score convention')
            with np.load(cell / 'scores.npz', allow_pickle=False) as payload:
                for key, value in scores['input_arrays'].items():
                    require(array_identity(payload[key]) == value, 'score input payload mismatch')
                trace = payload['velocity_trace']
                require(trace.shape == (8, 10, 50, 32), 'trace shape')
                require(np.isfinite(payload['actions']).all(), 'nonfinite scoring actions')
                computed = compute_accel_scores(trace)
            for n, row in enumerate(scores['rows']):
                require(row['repeat'] == n and row['noise'] == score_noise[n], 'score noise')
                times = np.asarray(row['times'])
                require(times.shape == (10,) and np.allclose(times, 1-np.arange(10)/10,
                                                            atol=1e-6, rtol=0), 'denoise grid')
                for p in range(2, 11):
                    val = row['metrics'][str(p)]
                    require(val['degenerate'] == bool(computed[p]['degenerate'][n]), 'degeneracy')
                    require(np.isclose(val['score'], computed[p]['score'][n], atol=1e-10, rtol=1e-8),
                            'Accel recomputation')
                    acc[i, v, n, p-2] = val['score']
                degenerate += int(row['metrics']['3']['degenerate'])
            physical_hashes = set()
            for checks in (complete['candidate_pixels'], complete['canonical_pixels']):
                for check in checks.values():
                    require(check['pass'], 'render gate failed')
                    for key in pixels_max:
                        require(math.isfinite(check[key]) and check[key] >= 0, 'invalid pixel value')
                        pixels_max[key] = max(pixels_max[key], check[key])
            for n, trial_ref in enumerate(complete['trials']):
                path = cell / f'execution_{n}.json'
                digest = sha(path)
                require(trial_ref['file'] == path.name and trial_ref['sha256'] == digest, 'trial hash')
                trial = read(path)
                require(trial['repeat'] == n, 'execution repeat')
                check_trial(trial, trial_ref, identity, scores, horizon, noise[n])
                physical_hashes.add(trial['initial_physical_sha256'])
                y[i, v, n], steps[i, v, n], elapsed[i, v, n] = trial['success'], trial['steps'], trial['seconds']
            require(complete['success_rate'] == y[i, v].mean(), 'success mean')
            # This runtime fingerprint includes the changed camera mounts. It
            # must agree across repeats/models at one view, NOT across views.
            require(len(physical_hashes) == 1, 'repeat initial state mismatch')
            inputs.append(dict(scores['input_arrays'], initial_physical_sha256=next(iter(physical_hashes))))
            roots.update(f'{initial}/{view}:{sha(cell / "complete.json")}\n'.encode())
        paired_inputs.append(inputs)
    require(degenerate == 0, 'degenerate scores need explicit protocol handling')
    output = Path(output)
    np.savez_compressed(output / (worker['worker_id'] + '.npz'), success=y, accel_prefixes=acc,
                        steps=steps, seconds=elapsed, initial_ids=np.asarray(ids))
    receipt = dict(status='PASS', worker_id=worker['worker_id'], arm=worker['arm'],
                   initials=len(ids), cells=expected_n, rollouts=4*expected_n,
                   cell_receipt_tree_sha256=roots.hexdigest(), degenerate=degenerate,
                   max_pixel_differences=pixels_max, progress=progress)
    write(output / (worker['worker_id'] + '.json'), receipt)
    write(output / (worker['worker_id'] + '.inputs.json'), paired_inputs)
    return receipt


def audit(config):
    cfg = read(config)
    out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=True)
    plan = read(cfg['matrix_plan'])
    reference = validate_plan(plan)
    plan_hash = sha(cfg['matrix_plan'])
    protocol_file = out / 'analysis-protocol.json'
    if protocol_file.exists():
        require(read(protocol_file)['config_sha256'] == sha(config), 'analysis config changed')
    else:
        write(protocol_file, dict(frozen_utc=datetime.now(timezone.utc).isoformat(), config=cfg,
                                  config_sha256=sha(config), plan_sha256=plan_hash))
    work = out / 'audit-workers'
    work.mkdir(exist_ok=True)
    errors, records = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(audit_worker, plan, reference, plan_hash, w, work): w for w in plan['workers']}
        for future in as_completed(futures):
            try:
                record = future.result()
                records.append(record)
                print(json.dumps(dict(audited_workers=len(records), worker=record['worker_id'])), flush=True)
            except Exception as exc:
                errors.append(dict(worker=futures[future]['worker_id'], error=str(exc)))
                print(json.dumps(errors[-1]), flush=True)
            write(out / 'audit-progress.json', dict(audited_workers=len(records), errors=errors))
    if errors:
        write(out / 'audit.json', dict(status='HOLD', errors=errors))
        raise ValueError('Matrix audit failed; see audit.json')
    # Check frozen source and model bytes, not just worker counters.
    require(sha(plan['source_manifest']) == plan['source_manifest_sha256'], 'source manifest hash')
    for name, digest in read(plan['source_manifest']).items():
        require(sha(Path(plan['source']) / name) == digest, 'frozen source changed: '+name)
    for arm, model in plan['models'].items():
        for name, meta in model['files'].items():
            file = Path(model['checkpoint']) / name
            require(file.stat().st_size == meta['bytes'] and file_hash(file) == meta['sha256'],
                    'checkpoint changed: '+arm+'/'+name)
    jobs = sorted(reference['jobs'], key=lambda j: (j['split'], j['task'], j['scene_seed'], j['initial_id']))
    lookup = {j['initial_id']: i for i, j in enumerate(jobs)}
    y = np.zeros((2, 400, 97, 4), np.uint8)
    acc = np.zeros((2, 400, 97, 8, 9))
    steps, seconds = np.zeros_like(y, dtype=np.int16), np.zeros_like(y, dtype=float)
    input_lookup, differences = {}, []
    for w in plan['workers']:
        m = ['canonical', 'broad'].index(w['arm'])
        inputs = read(work / (w['worker_id'] + '.inputs.json'))
        with np.load(work / (w['worker_id'] + '.npz')) as arrays:
            for row, initial in enumerate(arrays['initial_ids']):
                i = lookup[str(initial)]
                y[m, i], acc[m, i] = arrays['success'][row], arrays['accel_prefixes'][row]
                steps[m, i], seconds[m, i] = arrays['steps'][row], arrays['seconds'][row]
                if str(initial) in input_lookup:
                    earlier = input_lookup[str(initial)]
                    for v, (left, right) in enumerate(zip(earlier, inputs[row])):
                        require(left['observation/state'] == right['observation/state'], 'paired proprioception differs')
                        require(left['initial_physical_sha256'] == right['initial_physical_sha256'], 'paired physical initial differs')
                        if left != right:
                            differences.append(dict(initial_id=str(initial), view_id=plan['view_ids'][v]))
                else:
                    input_lookup[str(initial)] = inputs[row]
    np.savez_compressed(out / 'matrix.npz', success=y, accel_prefixes=acc, steps=steps, seconds=seconds)
    write(out / 'population.json', dict(jobs=jobs, views=reference['catalog']['views'], models=plan['models']))
    receipt = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(), plan_sha256=plan_hash,
                   workers=records, cells=77600, complete_rollouts=310400, initial_score_samples=620800,
                   all_score_payloads_hashed=True, all_accel_prefixes_recomputed=True,
                   all_replan_noise_addresses_checked=True, model_files_rehashed=True,
                   paired_input_different_cells=len(differences), paired_input_differences=differences,
                   caveat='Native renderer residual variability retained; hash mismatch of paired RGB is not automatically a protocol failure. No bitwise trajectory or physical-camera feasibility claim.',
                   artifacts={n: sha(out / n) for n in ('matrix.npz', 'population.json', 'analysis-protocol.json')})
    write(out / 'audit.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('workers', 'paired_input_differences')}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=['audit'], required=True)
    args = parser.parse_args()
    audit(args.config)


if __name__ == '__main__':
    main()
