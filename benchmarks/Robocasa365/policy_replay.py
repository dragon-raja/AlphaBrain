"""Bounded fixed-input policy replay, isolating compilation from rendering."""
from pathlib import Path
import os
import platform
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.data.rollout_replay import array_identity
from .evaluation_policy import ResidentPolicy


def replay_score_anchor(config_file):
    """Host admission from an existing score anchor, not a new scientific trial."""
    from AlphaBrain.research.dsol.data.evaluation_matrix import flow_noise, NUMERIC_FLAGS
    cfg = read(config_file); output = Path(cfg['output'])
    if output.exists(): raise FileExistsError(output)
    if os.environ.get('XLA_FLAGS') != NUMERIC_FLAGS:
        raise ValueError('Frozen numerical flags required')
    started = time.monotonic(); rows = []
    policy = ResidentPolicy(Path(cfg['checkpoint']), cfg['model_config'])
    for entry in cfg['anchors']:
        path = Path(entry['path'])
        if sha(path) != entry['sha256']: raise ValueError('Anchor metadata changed')
        anchor = read(path)
        if anchor['model'] != 'official' or anchor['score_repeats'] != 8 or anchor['action_steps'] != 0:
            raise ValueError('Official initial score anchor required')
        if sha(path.with_suffix('.npz')) != anchor['arrays_sha256']:
            raise ValueError('Anchor arrays changed')
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as arrays:
            request = dict(prompt=anchor['prompt'], **{k: arrays[k] for k in arrays.files if k.startswith('observation/')})
            for repeat in range(8):
                result = policy.trace(request, flow_noise(anchor['initial'], 'score', repeat, 0))
                v, a = result['velocity'][0], result['actions']
                rows.append(dict(initial=anchor['initial'], view=anchor['view'], repeat=repeat,
                    velocity_exact=bool(np.array_equal(v, arrays['velocity_trace'][repeat])),
                    actions_exact=bool(np.array_equal(a, arrays['actions'][repeat])),
                    velocity_max_abs=float(np.max(np.abs(v-arrays['velocity_trace'][repeat]))),
                    action_max_abs=float(np.max(np.abs(a-arrays['actions'][repeat])))))
    passed = bool(rows) and all(r['velocity_exact'] and r['actions_exact'] for r in rows)
    receipt = dict(status='PASS' if passed else 'HOLD', host=platform.node(), config_sha256=sha(config_file),
                   seconds=time.monotonic()-started, rows=rows,
                   scope='Existing frozen input/noise replay only; excluded from scientific trial counts; full-task pilot still required')
    write(output, receipt)
    if not passed: raise ValueError('Cross-host fixed-input replay differs; no production admission')
    return receipt


def replay_inputs(record_file, checkpoint, config, output, repeats=2):
    record_file, checkpoint, output = map(Path, (record_file, checkpoint, output))
    if output.exists() or output.with_suffix('.npz').exists():
        raise FileExistsError(output)
    record = read(record_file)
    payload = record_file.parent / 'arrays.npz'
    if sha(payload) != record['arrays_sha256']:
        raise ValueError('Replay payload identity changed')
    started = time.monotonic()
    policy = ResidentPolicy(checkpoint, config)
    result = dict(scope='Frozen-input inference only; no environment and no success-rate claim',
                  record_file=str(record_file), record_sha256=sha(record_file),
                  checkpoint=str(checkpoint), config=config, host=platform.node(),
                  numeric_flags=os.environ.get('XLA_FLAGS', ''), rows=[])
    outputs = {}
    with np.load(payload, allow_pickle=False) as arrays:
        for index, row in enumerate(record['plans']):
            request = {'prompt': row['prompt']}
            for name, identity in row['arrays'].items():
                value = arrays[f'plan_{index:04d}/' + name].copy()
                if array_identity(value) != identity:
                    raise ValueError('Replay array identity changed')
                if name.startswith('observation/'):
                    request[name] = value
            noise = arrays[f'plan_{index:04d}/noise'].copy()
            actions = [policy.infer(request, noise) for _ in range(repeats)]
            traced = policy.trace(request, noise)
            outputs[f'plan_{index:04d}/actions'] = np.stack(actions)
            outputs[f'plan_{index:04d}/velocity'] = traced['velocity']
            result['rows'].append(dict(index=index, input_arrays=row['arrays'],
                repeated_actions_exact=all(np.array_equal(a, actions[0]) for a in actions),
                actions=array_identity(actions[0]), velocity=array_identity(traced['velocity']),
                trace_endpoint_max_abs=float(np.max(np.abs(traced['actions']-actions[0]))),
                trace_endpoint_match=bool(np.allclose(traced['actions'], actions[0], atol=1e-5, rtol=1e-5))))
            print(index, result['rows'][-1]['repeated_actions_exact'], flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, **outputs)
    result.update(seconds=time.monotonic()-started, arrays_sha256=sha(output.with_suffix('.npz')),
        status='PASS' if all(r['repeated_actions_exact'] and r['trace_endpoint_match'] for r in result['rows']) else 'HOLD')
    write(output, result)
    return result
