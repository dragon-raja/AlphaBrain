"""Native Plus five-window extension using the standard durable work queue.

Wait for the assigned LIBERO queue to release its GPUs, retain two pilot blocks,
then distribute every remaining initial/model block once. No full rollouts.
"""
import argparse
from pathlib import Path
import shutil
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.execution.queue import MissingResult
from benchmarks.LIBERO.window_queue import WindowBackend


def partition_blocks(counts, capacities):
    """LPT by batch16 candidate blocks, never by scores or success labels."""
    pilots = {'fresh': ('canonical', 0), 'gnu': ('broad', 0)}
    partition = {h: [] for h in capacities}; load = {h: 0 for h in capacities}
    pending = [(m, i) for i in range(len(counts)) for m in ('canonical', 'broad')
               if (m, i) not in pilots.values()]
    for model, initial in sorted(pending, key=lambda x: (-((counts[x[1]]+15)//16), x[1], x[0])):
        host = min(capacities, key=lambda h: (load[h]/capacities[h], h))
        partition[host].append((model, initial)); load[host] += (counts[initial]+15)//16
    return pilots, partition


def validate_block(config_file, model, initial, output):
    config = read(config_file); output = Path(output)
    if not (output/'status.json').exists() or read(output/'status.json')['status'] != 'COMPLETE':
        raise MissingResult(str(output))
    row = read(output/'status.json')
    from .plus_windows import load_panel
    cached, _ = load_panel(config, initial); count = len(cached['views'])
    if (row['initial'] != initial or row['model'] != model or row['config_sha256'] != sha(config_file)
            or row['source_sha256'] != sha(Path(config['source'])/'benchmarks/LIBERO/window_probe.py')
            or row['arrays_sha256'] != sha(output/'probes.npz') or not row['window1_exact']):
        raise ValueError('Native window identity mismatch')
    with np.load(output/'probes.npz', allow_pickle=False) as z:
        if (z['scores'].shape != (count, 8, 5) or not np.isfinite(z['scores']).all()
                or not np.array_equal(z['views'], cached['views'])
                or not np.array_equal(z['scores'][:, :, 0], z['initial_reference_scores'])
                or z['velocities'].shape != (count, 8, 5, 10, 10, 7)
                or not np.isfinite(z['velocities']).all() or not np.isfinite(z['actions']).all()):
            raise ValueError('Native window score/trace coverage mismatch')
    if model == 'canonical':
        bridge = read(output/'endpoint-equivalence.json')
        if bridge['status'] != 'PASS' or bridge['candidates'] != 97:
            raise ValueError('Absolute endpoint bridge failed')
    return dict(status='PASS', receipt_sha256=sha(output/'status.json'),
        native_full_result_comparison='Required separately before publishing results')


class PlusWindowBackend(WindowBackend):
    def validate(self, task):
        return validate_block(self.runtime['config'], task['payload']['model'], task['payload']['initial'], task['output'])


def factory(plan, root):
    return PlusWindowBackend(plan, root)


def prepare(config_file, host, *, pilot=False):
    from AlphaBrain.research.dsol.execution.manifest import task_key
    from AlphaBrain.research.dsol.execution.queue import prepare as prepare_queue
    config_file = Path(config_file).resolve(); cfg = read(config_file)
    worker = read(cfg['worker_config']); root = Path(worker['root']); source = Path(worker['source'])
    records = read(worker['inputs'])['records']
    views = [['canonical'] + [v['request_id'] for v in read(Path(worker['render_root'])/f'{i:02d}'/'records.json')['requests']]
             for i in range(len(records))]
    if len(records) != 32 or sum(len(v)-1 for v in views) != 1364:
        raise ValueError('Authorized native Plus scope changed')
    pilots, partition = partition_blocks([len(v) for v in views], {h: len(g) for h, g in cfg['gpus'].items()})
    if not pilot:
        for model, initial in pilots.values():
            validate_block(cfg['worker_config'], model, initial, root/'plus-probes'/model/f'{initial:02d}')
    blocks = [pilots[host]] if pilot else partition[host]
    fingerprint = sha(cfg['worker_config']); tasks = []
    for model, initial in blocks:
        record = records[initial]
        addresses = [dict(model=model, task=record['state']['task_id'], initial=record['state']['pair_key'],
            view=view, stream='score', repeat=repeat, noise=f'libero-window-20261012::{repeat}::windows1-5')
            for view in views[initial] for repeat in range(8)]
        tasks.append(dict(key=task_key(cfg['experiment_id'], fingerprint, addresses), addresses=addresses,
            payload=dict(initial=initial, model=model), output=str(root/'plus-probes'/model/f'{initial:02d}')))
    paths = {config_file, Path(cfg['worker_config']), Path(worker['inputs']), Path(worker['anchors']), Path(worker['release'])}
    paths.update(p for directory in ('AlphaBrain', 'scripts', 'benchmarks') for p in (source/directory).rglob('*.py'))
    for i in range(32):
        paths.update(Path(worker['render_root'])/f'{i:02d}'/f for f in ('status.json', 'records.json'))
    stage = 'pilot' if pilot else 'production'
    control = Path(cfg['control_roots'][host])/stage
    plan = dict(format='dsol_work_queue_v1', experiment_id=cfg['experiment_id'], protocol_sha256=fingerprint,
        tasks=tasks, slots=[dict(host=host, id=str(g)) for g in (cfg.get('pilot_gpus',{}).get(host,cfg['gpus'][host][:1]) if pilot else cfg['gpus'][host])],
        artifacts=[dict(path=str(p), sha256=sha(p)) for p in sorted(paths)],
        backend=dict(module='benchmarks.LIBERO.plus_window_queue', factory='factory'),
        runtime=dict(host=host, source=str(source), config=cfg['worker_config'], python=cfg['python'],
            workers_per_gpu=1, environment=cfg['environments'][host]))
    prepare_queue(plan, control)
    write(root/f'plus-window-ownership-{host}-{stage}.json', dict(control=str(control), pilots=pilots,
        partition=partition, config_sha256=sha(config_file), worker_config_sha256=fingerprint))
    return control


def wait_for(check, deadline, status_file, phase):
    while not check():
        if time.monotonic() > deadline:
            raise TimeoutError('Extension dependency deadline: '+phase)
        write(status_file, dict(status='WAITING', phase=phase, unix=time.time()))
        time.sleep(20)


def run_pipeline(config_file, host):
    from AlphaBrain.research.dsol.execution.cli import main
    cfg = read(config_file); worker = read(cfg['worker_config']); root = Path(worker['root'])
    status_file = root/f'plus-window-pipeline-{cfg.get("pipeline_tag", "")}{host}.json'; deadline = time.monotonic()+cfg['timeout_seconds']
    def released():
        for dependency in cfg.get('production_dependencies',{}).get(host,[cfg['dependencies'][host]]):
            path=Path(dependency)/'state.json'
            if not path.exists():return False
            state=read(path)
            if state['status'] in ('FAILED','INTERRUPTED'):
                raise RuntimeError('Original queue failed; do not commandeer its GPUs')
            if state['status']!='COMPLETE':return False
        return True
    def admitted():
        for model in ('canonical', 'broad'):
            path = root/'plus-probes'/model/'00'/'status.json'
            if path.exists() and read(path)['status'] == 'FAILED':
                raise RuntimeError('Native Plus pilot failed')
            if not path.exists() or read(path)['status'] != 'COMPLETE':
                return False
        return True
    try:
        if not cfg.get('early_pilot',False):
            wait_for(released, deadline, status_file, 'original_libero_queue_release')
        for pilot in (True, False):
            if not pilot:
                wait_for(admitted, deadline, status_file, 'two_model_native_window_admission')
                wait_for(released, deadline, status_file, 'production_gpu_release')
            control = prepare(config_file, host, pilot=pilot)
            write(status_file, dict(status='RUNNING', phase='pilot' if pilot else 'production', control=str(control)))
            try:
                main(['run', '--execute', '--root', str(control), '--timeout-seconds', str(max(1,int(deadline-time.monotonic()))), '--poll-seconds', '2'])
                if read(control/'state.json')['status'] != 'COMPLETE':
                    raise RuntimeError('Native window queue did not complete')
            finally:
                backup = root/'controls'/f'plus-windows-{host}-{"pilot" if pilot else "production"}'
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(control, backup)
        write(status_file, dict(status='COMPLETE', phase='scoring_complete_full_result_audit_pending'))
        if host == 'fresh':
            def results_ready():
                for initial in range(32):
                    for model in ('canonical', 'broad'):
                        for directory in ('plus-probes', 'plus-evaluation'):
                            path = root/directory/model/f'{initial:02d}'/'status.json'
                            if path.exists() and read(path)['status']=='FAILED':
                                raise RuntimeError('Result failed before joint audit: '+str(path))
                            if not path.exists() or read(path)['status']!='COMPLETE':
                                return False
                return True
            wait_for(results_ready, deadline, status_file, 'all_independent_full_results_and_short_probes')
            from .plus_window_analysis import audit_and_analyze
            audit_and_analyze(config_file)
            write(status_file, dict(status='COMPLETE', phase='joint_window1_and_endpoint_audit_and_analysis_complete'))
    except BaseException as error:
        write(status_file, dict(status='FAILED', error=repr(error))); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--host', choices=('fresh', 'gnu'), required=True)
    parser.add_argument('--execute', action='store_true'); args = parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute required')
    run_pipeline(args.config, args.host)
