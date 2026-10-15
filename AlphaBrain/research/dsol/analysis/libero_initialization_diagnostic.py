"""Audit a bounded normalization intervention without relabeling old results."""
from pathlib import Path
import argparse

import numpy as np

from ..data.artifacts import read, sha, write


PAIR_KEYS = ('pair_key', 'selected_candidate_id', 'initialization_receipt',
             'noise_bank_manifest_sha256', 'max_steps', 'replan_steps', 'wait_steps',
             'sensor_control', 'prompt', 'policy_repeat_id')


def validate_pair(before, after):
    for key in PAIR_KEYS:
        if before[key] != after[key]:
            raise ValueError('Diagnostic pair mismatch: '+key)
    for row in (before, after):
        if row['status'] != 'complete' or not row['explicit_flow_noise']:
            raise ValueError('Incomplete or implicit-noise diagnostic')
        if not row['success'] and row['completion_steps'] != row['max_steps']:
            raise ValueError('Failed episode is truncated')
        if len(row['policy_calls']) != row['inference_calls'] or not row['policy_calls']:
            raise ValueError('Incomplete policy call trace')
    for a, b in zip(before['policy_calls'], after['policy_calls']):
        if a['noise_sha256'] != b['noise_sha256'] or a['replan_index'] != b['replan_index']:
            raise ValueError('Shared seven-dimensional noise changed')


def summarize(root, original):
    root, original = Path(root), Path(original)
    names = ['alpha-quantile-rollout-checked', 'native-quantile-rollout']
    statuses = {name: read(root/name/'status.json') for name in names}
    if any(v['status'] != 'COMPLETE' or v['completed'] != 8 for v in statuses.values()):
        raise ValueError('Both bounded reference probes must be complete')
    expected = [state*97*32 for state in range(0, 32, 4)]
    plans = {name: read(root/name/'plan.json') for name in names}
    if any(plan['indices'] != expected for plan in plans.values()):
        raise ValueError('Diagnostic population changed')
    sources = []
    task_rows = []
    for index in expected:
        files = [original/'episodes'/f'{index:06d}.json',
                 *[root/name/'episodes'/f'{index:06d}.json' for name in names]]
        rows = [read(path) for path in files]
        for row in rows[1:]:
            validate_pair(rows[0], row)
        for name, row in zip(names, rows[1:]):
            release = read(root/name/'release.json')
            if row['aa0']['release_sha256'] != sha(root/name/'release.json'):
                raise ValueError('Release identity mismatch')
            if row['aa0']['checkpoint_sha256'] != release['models']['baseline']['weights_sha256']:
                raise ValueError('Checkpoint identity mismatch')
        sources.extend({'path': str(path), 'sha256': sha(path)} for path in files)
        task_rows.append(dict(index=index, pair_key=rows[0]['pair_key'],
                              success={k: bool(r['success']) for k, r in zip(['old_meanstd', *names], rows)},
                              steps={k:r['completion_steps'] for k,r in zip(['old_meanstd',*names], rows)}))
    fixed = {name: read(root/(name+'.json')) for name in ['alpha', 'native']}
    differences = []
    for a, b in zip(fixed['alpha']['rows'], fixed['native']['rows']):
        for key in ['pair_key', 'image_sha256', 'wrist_sha256', 'noise7_sha256', 'prompt']:
            if a[key] != b[key]:
                raise ValueError('Fixed-input reference mismatch: '+key)
        differences.append(np.asarray(a['actions'])-np.asarray(b['actions']))
    if len(differences) != 8:
        raise ValueError('Need all eight fixed-input observations')
    differences = np.stack(differences)
    result = dict(status='COMPLETE_AUDITED', scope='Engineering diagnosis, not a benchmark success-rate estimate',
        old_640_result='Retained; not a valid native official performance baseline',
        rows=task_rows, success_counts={name:sum(row['success'][name] for row in task_rows)
                                      for name in ['old_meanstd', *names]}, count_per_condition=8,
        fixed_input_action_mae=float(abs(differences).mean()),
        fixed_input_action_max_abs=float(abs(differences).max()),
        native32_noise_limit='Only first7 coordinates paired to legacy7; remaining25 deterministically extended',
        interpretation='Quantile-to-mean/std decoding mismatch causally changes task outcomes in this probe; '
                       'not a proof that every other porting difference is harmless',
        trained_models='Retain their trained mean/std decoding; do not apply the zero-step correction to them')
    output = root/'analysis'
    output.mkdir(exist_ok=False)
    write(output/'summary.json', result)
    write(output/'audit.json', dict(status='PASS', protocol_fields=list(PAIR_KEYS), sources=sources,
          noise_check='shared seven coordinates and common consumed replan prefix; not full 32D equality'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.original)


if __name__ == '__main__':
    main()
