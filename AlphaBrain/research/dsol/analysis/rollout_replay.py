"""Locate the first observed divergence without asserting an unmeasured cause."""
from pathlib import Path

import numpy as np

from ..data.artifacts import read, sha


def compare_replays(left_file, right_file):
    paths = [Path(left_file), Path(right_file)]
    records = [read(p) for p in paths]
    for path, record in zip(paths, records):
        if sha(path.parent / 'arrays.npz') != record['arrays_sha256']:
            raise ValueError('Replay arrays have changed')
    result = dict(left=str(paths[0]), right=str(paths[1]),
                  plan_counts=[len(x['plans']) for x in records],
                  step_counts=[len(x['steps']) for x in records],
                  first_plan_difference=None, first_step_difference=None,
                  matched_step_physics=True, matched_step_actions=True)
    with np.load(paths[0].parent / 'arrays.npz', allow_pickle=False) as left, np.load(
            paths[1].parent / 'arrays.npz', allow_pickle=False) as right:
        for a, b in zip(records[0]['plans'], records[1]['plans']):
            if a['step'] != b['step'] or a['index'] != b['index'] or a['arrays'].keys() != b['arrays'].keys():
                raise ValueError('Unaligned replay plans')
            changed = [key for key, value in a['arrays'].items() if value != b['arrays'][key]]
            if changed or a['physical_sha256'] != b['physical_sha256'] or a['prompt'] != b['prompt']:
                delta = {}
                for key in changed:
                    name = f"plan_{a['index']:04d}/" + key
                    x, y = left[name].astype(np.float64), right[name].astype(np.float64)
                    if x.shape != y.shape:
                        raise ValueError('Replay array shapes differ')
                    delta[key] = dict(max_abs=float(np.abs(x-y).max()),
                                      mean_abs=float(np.abs(x-y).mean()),
                                      changed_values=int(np.count_nonzero(x != y)))
                result['first_plan_difference'] = dict(index=a['index'], step=a['step'],
                    physical_equal=a['physical_sha256'] == b['physical_sha256'],
                    prompt_equal=a['prompt'] == b['prompt'], array_differences=delta)
                break
    for a, b in zip(records[0]['steps'], records[1]['steps']):
        if a['step'] != b['step']:
            raise ValueError('Unaligned replay steps')
        physics_equal = a['physical_sha256'] == b['physical_sha256']
        actions_equal = a['action'] == b['action']
        result['matched_step_physics'] &= physics_equal
        result['matched_step_actions'] &= actions_equal
        if result['first_step_difference'] is None and a != b:
            result['first_step_difference'] = dict(step=a['step'], physical_equal=physics_equal,
                action_equal=actions_equal, success_equal=a['success'] == b['success'])
    result['scope'] = 'Equality over recorded overlapping steps; unequal lengths are not full replay equivalence'
    return result
