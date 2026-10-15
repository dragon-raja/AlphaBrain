"""Admit a new host using fixed-input replay and its first owned full block.

The first block is already in the scientific budget, not a duplicate pilot.
Only one worker may run until its complete task attempts validate. The worker
source, noise, output paths and original selection remain unchanged.
"""
from pathlib import Path
import platform
import time

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from .confirmation_queue import ConfirmationBackend


class TransferredHostBackend(ConfirmationBackend):
    def admit(self, slot):
        binding = self.plan['host_bootstrap']
        if sha(binding['path']) != binding['sha256']:
            raise ValueError('Host bootstrap configuration changed')
        cfg = read(binding['path'])
        if (cfg['host'] != self.runtime['host'] or slot['host'] != cfg['host']
                or cfg['model'] != self.runtime['model']
                or cfg['plan_sha256'] != sha(self.runtime['plan'])
                or cfg['worker_sha256'] != sha(Path(self.runtime['source']) /
                                             'benchmarks/Robocasa365/confirmation_execution.py')):
            raise ValueError('Host bootstrap does not bind this worker/plan/host')
        for replay in cfg['replays']:
            if sha(replay['path']) != replay['sha256']:
                raise ValueError('Host replay changed')
            receipt = read(replay['path'])
            if (receipt['status'] != 'PASS' or receipt['host'] != platform.node()
                    or receipt['model'] != self.runtime['model'] or not receipt['rows']
                    or not all(r['actions_exact'] and r['velocity_exact'] for r in receipt['rows'])):
                raise ValueError('Exact host replay admission required')
        if len(cfg['replays']) != self.runtime['workers_per_gpu']:
            raise ValueError('Replay must cover admitted concurrency')
        original = read(self.runtime['admission'])
        if (original['status'] != 'PASS' or original['model'] != self.runtime['model']
                or original['plan_sha256'] != cfg['plan_sha256']
                or original['source_sha256'] != cfg['worker_sha256']
                or not original['complete_task_attempts_verified']):
            raise ValueError('Original full-task admission missing')
        task = self.plan['tasks'][0]
        state = read(self.root/'state.json')
        row = state['tasks'][task['key']]
        admission = self.root/'host-admission.json'
        if not admission.exists():
            if row['status'] == 'COMPLETE':
                validation = self.validate(task)
                write(admission, dict(status='PASS', host=cfg['host'], model=cfg['model'],
                    plan_sha256=cfg['plan_sha256'], source_sha256=cfg['worker_sha256'],
                    complete_task_attempts_verified=True, bootstrap_sha256=binding['sha256'],
                    task_key=task['key'], validation=validation, admitted_unix=time.time(),
                    pilot_already_in_original_budget=True))
            else:
                # Queue FIFO must dispatch precisely the designated owned block.
                return (slot == self.plan['slots'][0] and row['status'] == 'PENDING'
                        and row['attempt'] == 0 and not any(r['status'] in ('LAUNCHING', 'RUNNING')
                            for r in state['tasks'].values()) and self.gpu_available(slot))
        receipt = read(admission)
        if (receipt['status'] != 'PASS' or receipt['bootstrap_sha256'] != binding['sha256']
                or receipt['task_key'] != task['key'] or row['status'] != 'COMPLETE'
                or receipt['validation'] != row['receipt']):
            raise ValueError('Transferred-host full-task admission invalid')
        return self.gpu_available(slot)


def replay_xr1(config_file):
    import numpy as np
    from .xr1_bridge import XR1Bridge
    from .xr1_evaluation import noise_seed
    cfg = read(config_file); output = Path(cfg['output'])
    if output.exists(): raise FileExistsError(output)
    rows = []; started = time.monotonic()
    bridge = XR1Bridge(cfg['model_python'], cfg['checkpoint'], cfg['upstream'], output.with_suffix('.log'))
    try:
        for entry in cfg['anchors']:
            path = Path(entry['path'])
            if sha(path) != entry['sha256']: raise ValueError('Anchor metadata changed')
            anchor = read(path)
            if anchor['model'] != 'xr1' or anchor['score_repeats'] != 8 or anchor['action_steps'] != 0:
                raise ValueError('XR-1 initial scoring anchor required')
            if sha(path.with_suffix('.npz')) != anchor['arrays_sha256']:
                raise ValueError('Anchor arrays changed')
            with np.load(path.with_suffix('.npz'), allow_pickle=False) as arrays:
                obs = {k: arrays[k] for k in arrays.files if k not in ('velocity_trace', 'actions', 'scores')}
                for repeat in range(8):
                    result = bridge.infer([obs], anchor['prompt'],
                        noise_seed(anchor['initial'], 'score', repeat, 0), capture=True)
                    v, a = result['velocity_trace'][0], result['actions']
                    rows.append(dict(initial=anchor['initial'], view=anchor['view'], repeat=repeat,
                        velocity_exact=bool(np.array_equal(v, arrays['velocity_trace'][repeat])),
                        actions_exact=bool(np.array_equal(a, arrays['actions'][repeat])),
                        velocity_max_abs=float(np.max(np.abs(v-arrays['velocity_trace'][repeat]))),
                        action_max_abs=float(np.max(np.abs(a-arrays['actions'][repeat])))))
    finally:
        bridge.close()
    passed = bool(rows) and all(r['actions_exact'] and r['velocity_exact'] for r in rows)
    receipt = dict(status='PASS' if passed else 'HOLD', model='xr1', host=platform.node(),
        seconds=time.monotonic()-started, rows=rows, config_sha256=sha(config_file),
        scope='Fixed inputs and addressed noise; engineering replay only, not additional scientific trials')
    write(output, receipt)
    if not passed: raise ValueError('Cross-host XR-1 replay differs')
    return receipt


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute: parser.error('Explicit --execute required')
    replay_xr1(args.config)
