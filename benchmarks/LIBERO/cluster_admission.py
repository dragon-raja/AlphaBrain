"""Exact cross-host admission of native Plus policy and complete trajectories.

Engineering replays never enter the scientific success-rate denominator.
"""
from pathlib import Path
import platform

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from .plus_evaluation_queue import validate_result


def compare_rows(reference, actual):
    fields = ('initial', 'model', 'pair_key', 'view', 'repeat', 'success', 'steps',
              'max_steps', 'calls', 'initial_input_exact')
    if len(reference) != len(actual) or not reference:
        raise ValueError('Admission episode coverage differs')
    for a, b in zip(reference, actual):
        if any(a[k] != b[k] for k in fields):
            raise ValueError('Cross-host complete trajectory differs')
    return dict(episodes=len(actual), action_and_noise_hashes_exact=True,
                completion_steps_exact=True, successes_exact=True)


def validate(config_file):
    config = read(config_file); rows = []; totals = [0., 0.]
    for model in ('canonical', 'broad'):
        paths = [Path(config[k])/model/'00' for k in ('reference_root', 'actual_root')]
        for path in paths:
            validate_result(config['admission_config'], 0, model, path)
        with np.load(paths[0]/'probes.npz', allow_pickle=False) as a, np.load(paths[1]/'probes.npz', allow_pickle=False) as b:
            if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
                raise ValueError('Cross-host initial scores or endpoint results differ')
        episodes = [read(p/'episodes.json')['rows'] for p in paths]
        row = compare_rows(*episodes)
        row.update(model=model, initial_scores_exact=True,
            reference_status_sha256=sha(paths[0]/'status.json'), actual_status_sha256=sha(paths[1]/'status.json'),
            reference_episodes_sha256=sha(paths[0]/'episodes.json'), actual_episodes_sha256=sha(paths[1]/'episodes.json'))
        rows.append(row)
        for i in range(2):
            totals[i] += sum(e['seconds'] for e in episodes[i])
    if sha(config['python_wrapper']) != config['python_wrapper_sha256']:
        raise ValueError('Remote runtime wrapper changed')
    result = dict(status='PASS', host=platform.node(), host_label=config['host_label'], rows=rows,
        production_config_sha256=sha(config['production_config']), config_sha256=sha(config_file),
        python_wrapper=config['python_wrapper'], python_wrapper_sha256=config['python_wrapper_sha256'],
        reference_rollout_seconds=totals[0], actual_rollout_seconds=totals[1],
        relative_throughput=totals[0]/totals[1], scope='engineering_only_not_additional_scientific_trials')
    output = Path(config['output'])
    if output.exists():
        raise FileExistsError(output)
    write(output, result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    print(validate(args.config), flush=True)
