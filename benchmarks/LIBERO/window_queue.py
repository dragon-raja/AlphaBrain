"""Five-window LIBERO adapter for the standard durable execution queue.

Only initial/model blocks are distributed. Host identities never enter policy
noise. Completed admission blocks are retained, not resubmitted.
"""
import argparse
from pathlib import Path
import shutil
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.execution.queue import MissingResult
from benchmarks.Robocasa365.probe_queue import ProbeBackend


def block_assignment():
    retained = [('canonical', 0, 'fresh'), ('broad', 0, 'fresh'), ('canonical', 1, 'gnu')]
    pending = [(model, initial) for initial in range(32) for model in ('canonical', 'broad')
               if (model, initial) not in [(m, i) for m, i, _ in retained]]
    return retained, {'fresh': pending[::3], 'gnu': [v for j, v in enumerate(pending) if j % 3 != 0]}


def validate_block(config_file, model, initial, output):
    output = Path(output); config = read(config_file)
    if not (output/'status.json').exists():
        raise MissingResult(str(output))
    row = read(output/'status.json')
    if row['status'] != 'COMPLETE':
        raise MissingResult(str(output))
    if (row['initial'] != initial or row['model'] != model or row['config_sha256'] != sha(config_file)
            or row['source_sha256'] != sha(Path(config['source'])/'benchmarks/LIBERO/window_probe.py')
            or row['arrays_sha256'] != sha(output/'probes.npz') or not row['window1_exact']):
        raise ValueError('Probe identity/hash validation failed')
    with np.load(output/'probes.npz', allow_pickle=False) as data, np.load(config['anchors'], allow_pickle=False) as anchors:
        if (data['scores'].shape != (97, 8, 5) or not np.isfinite(data['scores']).all()
                or not np.array_equal(data['scores'][:, :, 0], anchors['member_scores'][initial, ['canonical', 'broad'].index(model)])):
            raise ValueError('Probe score coverage or original anchor mismatch')
    return dict(status='PASS', receipt_sha256=sha(output/'status.json'))


class WindowBackend(ProbeBackend):
    def admit(self, slot):
        if slot['host'] != self.runtime['host']:
            raise ValueError('Wrong host coordinator')
        return self.gpu_available(slot)

    def worker_command(self, task):
        return [self.runtime['python'], '-u', '-m', 'benchmarks.LIBERO.window_probe',
            '--config', self.runtime['config'], '--initial', str(task['payload']['initial']),
            '--model', task['payload']['model'], '--output', task['output'], '--execute']

    def validate(self, task):
        return validate_block(self.runtime['config'], task['payload']['model'], task['payload']['initial'], task['output'])


def factory(plan, root):
    return WindowBackend(plan, root)


def prepare(config_file, host):
    from AlphaBrain.research.dsol.execution.manifest import task_key
    from AlphaBrain.research.dsol.execution.queue import prepare as prepare_queue
    config_file = Path(config_file).resolve()
    config = read(config_file); worker_config = read(config['worker_config']); root = Path(worker_config['root'])
    retained, partition = block_assignment(); records = read(worker_config['inputs'])['records']
    for model, initial, owner in retained:
        validate_block(config['worker_config'], model, initial, root/'probes'/model/f'{initial:02d}')
    fingerprint = sha(config['worker_config']); tasks = []
    for model, initial in partition[host]:
        record = records[initial]; output = root/'probes'/model/f'{initial:02d}'
        addresses = [dict(model=model, task=record['state']['task_id'], initial=record['state']['pair_key'],
            view=view, stream='score', repeat=repeat, noise=f'libero-window-20261012::{repeat}::windows1-5')
            for view in record['candidate_ids'] for repeat in range(8)]
        tasks.append(dict(key=task_key(config['experiment_id'], fingerprint, addresses), addresses=addresses,
            payload=dict(initial=initial, model=model), output=str(output)))
    paths = {Path(config_file), Path(config['worker_config']), Path(worker_config['inputs']), Path(worker_config['anchors'])}
    for source in (config['source'], worker_config['source']):
        paths.update(p for directory in ('AlphaBrain', 'scripts', 'benchmarks') for p in (Path(source)/directory).rglob('*.py'))
    paths.update(root/'probes'/m/f'{i:02d}'/'status.json' for m, i, _ in retained)
    environment = dict(config['environment'])
    if host == 'gnu':
        environment['LD_LIBRARY_PATH'] = config['gnu_library_path']
    plan = dict(format='dsol_work_queue_v1', experiment_id=config['experiment_id'], protocol_sha256=fingerprint,
        tasks=tasks, slots=[dict(host=host, id=str(gpu)) for gpu in config['gpus'][host]],
        artifacts=[dict(path=str(p), sha256=sha(p)) for p in sorted(paths)],
        backend=dict(module='benchmarks.LIBERO.window_queue', factory='factory'),
        runtime=dict(host=host, source=worker_config['source'], config=config['worker_config'],
            python=config['python'], workers_per_gpu=1, environment=environment))
    control = Path(config['control_roots'][host]); prepare_queue(plan, control)
    write(root/f'window-ownership-{host}.json', dict(retained=retained, partition=partition, control=str(control),
        config_sha256=sha(config_file), worker_config_sha256=fingerprint))
    return control


def after_admission(config_file, host):
    config = read(config_file); root = Path(read(config['worker_config'])['root']); started = time.monotonic()
    while True:
        ready = True
        for model, initial, _ in block_assignment()[0]:
            status = root/'probes'/model/f'{initial:02d}'/'status.json'
            if status.exists() and read(status)['status'] == 'FAILED':
                raise RuntimeError('Admission failed; no production launch: ' + str(status))
            ready &= status.exists() and read(status)['status'] == 'COMPLETE'
        if ready:
            break
        if time.monotonic()-started > config['admission_timeout_seconds']:
            raise TimeoutError('Admission deadline; no production launch')
        time.sleep(20)
    control = prepare(config_file, host)
    from AlphaBrain.research.dsol.execution.cli import main
    try:
        main(['run', '--execute', '--root', str(control), '--timeout-seconds', '43200', '--poll-seconds', '2'])
    finally:
        backup = root/'controls'/host
        backup.parent.mkdir(exist_ok=True)
        shutil.copytree(control, backup)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--host', choices=['fresh', 'gnu'], required=True)
    parser.add_argument('--execute', action='store_true'); args = parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute required')
    after_admission(args.config, args.host)
