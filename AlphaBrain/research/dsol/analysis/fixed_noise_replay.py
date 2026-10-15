"""Audit bounded same-noise replay without altering or relaunching evaluations."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import argparse

import numpy as np

from ..data.artifacts import read, sha, write


def compare(a, b):
    """Classify first recorded divergence, not its unobserved physical cause."""
    if a['initial_physical_sha256'] != b['initial_physical_sha256']:
        raise ValueError('Initial physical states differ')
    if a['plans'][0]['input_arrays'] != b['plans'][0]['input_arrays']:
        raise ValueError('Initial policy inputs differ')
    divergence = None
    for x, y in zip(a['plans'], b['plans']):
        if x['step'] != y['step'] or x['noise'] != y['noise']:
            raise ValueError('Execution noise addresses differ')
        if divergence is not None:
            continue
        changed = [key for key in x['input_arrays'] if x['input_arrays'][key] != y['input_arrays'][key]]
        if changed:
            image = any('image' in key for key in changed)
            state = 'observation/state' in changed
            kind = 'image_and_state' if image and state else 'image_only' if image else 'state_or_other_input'
            divergence = dict(step=x['step'], kind=kind, changed_inputs=changed)
        elif x['actions'] != y['actions']:
            divergence = dict(step=x['step'], kind='actions_with_identical_recorded_input')
    if divergence is None and (len(a['plans']) != len(b['plans']) or a['steps'] != b['steps']):
        divergence = dict(step=min(a['steps'], b['steps']), kind='termination_or_duration')
    return dict(success_disagrees=a['success'] != b['success'],
                executed_actions_equal=a['executed_actions_sha256'] == b['executed_actions_sha256'],
                final_physical_equal=a['final_physical_sha256'] == b['final_physical_sha256'],
                first_recorded_divergence=divergence)


def aggregate(rows, key):
    comparisons = [row[key] for row in rows]
    kinds = Counter((c['first_recorded_divergence'] or {}).get('kind', 'none') for c in comparisons)
    return dict(cells=len(rows), success_disagreements=sum(c['success_disagrees'] for c in comparisons),
                equal_action_sequences=sum(c['executed_actions_equal'] for c in comparisons),
                equal_final_physical_states=sum(c['final_physical_equal'] for c in comparisons),
                first_recorded_divergence=dict(kinds))


def run(root, output):
    root, output = Path(root), Path(output)
    plan = read(root / 'plan.json')
    p0 = read(plan['p0_plan'])
    if sha(plan['p0_plan']) != plan['p0_plan_sha256']:
        raise ValueError('P0 plan changed')
    rows = []
    for worker in plan['workers']:
        for arm in ('canonical', 'broad'):
            base = root / 'workers' / worker['worker_id'] / arm
            status = read(base / 'progress.json')
            if status['status'] != 'COMPLETE' or status['completed_rollouts'] != len(worker['initials']) * 4:
                raise ValueError('Replay worker incomplete')
            for initial in worker['initials']:
                for view in plan['view_ids']:
                    paths = [base / 'cells' / initial / view / f'replay_{i}.json' for i in range(2)]
                    trials = [read(p) for p in paths]
                    for i, trial in enumerate(trials):
                        provenance = trial['provenance']
                        expected = dict(plan_sha256=sha(root / 'plan.json'), worker=worker['worker_id'],
                                        arm=arm, initial_id=initial, view_id=view, replay=i)
                        if any(provenance[key] != value for key, value in expected.items()):
                            raise ValueError('Replay provenance mismatch')
                        if sha(provenance['original_trial']) != provenance['original_trial_sha256']:
                            raise ValueError('Original trial changed')
                    old_path = Path(trials[0]['provenance']['original_trial'])
                    old = [read(old_path.with_name(f'execution_{i}.json')) for i in range(4)]
                    disagreement = np.mean([a['success'] != b['success'] for a, b in combinations(old, 2)])
                    rows.append(dict(worker=worker['worker_id'], arm=arm, initial_id=initial, view_id=view,
                                     new_pair=compare(*trials), old_vs_new0=compare(old[0], trials[0]),
                                     old_vs_new1=compare(old[0], trials[1]),
                                     original_distinct_noise_disagreement=float(disagreement),
                                     success=[old[0]['success'], trials[0]['success'], trials[1]['success']],
                                     artifact_hashes={str(p): sha(p) for p in paths}))
    if len(rows) != 40:
        raise ValueError('Expected exactly 40 cells / 80 new rollouts')
    result = dict(scope='10 seen initials, two geometry-selected views, two models; diagnostic not benchmark estimate',
                  caveat='Hash divergence gives first recorded difference, not proof of renderer-only causation. '
                         'The 40 cells are clustered within ten tasks/initials; not 40 independent samples.',
                  plan_sha256=sha(root / 'plan.json'), rows=rows,
                  aggregate={key: aggregate(rows, key) for key in ('new_pair', 'old_vs_new0', 'old_vs_new1')},
                  original_distinct_noise_disagreement=float(np.mean([r['original_distinct_noise_disagreement'] for r in rows])))
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.root, args.output)


if __name__ == '__main__':
    main()
