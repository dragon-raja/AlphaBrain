"""Attach existing, ordered two-noise query evidence; never invokes a runtime."""
from copy import deepcopy
from pathlib import Path

import numpy as np

from AlphaBrain.research.dsol.data.online_reward import validate_raw
from AlphaBrain.research.dsol.execution.manifest import file_digest


def attach(episode, query_root):
    """Exclude repeats/returns by measured pose; verify all raw best-score steps.

    The caller binds the source campaign/noise/physics identity and completion
    receipts. This function does not establish cross-campaign equivalence.
    """
    validate_raw(episode)
    count = 1 + sum(t['delta_queries'] for t in episode['transitions'])
    hidden=set(episode.get('terminal_confirmation',{}).get('query_numbers',[]))
    queries, artifacts, seen, pairs = [], [], set(), []
    previous = None
    for number in range(1, count+1):
        path = Path(query_root)/f'query-{number:02d}.npz'
        with np.load(path, allow_pickle=False) as archive:
            scores = np.asarray([float(archive[f'score_{i}']) for i in (0,1)])
            pose = np.asarray(archive['camera_to_world_opencv'])
            if (pose.shape != (4,4) or not np.isfinite(pose).all()
                    or not np.isfinite(scores).all() or (scores < 0).any()):
                raise ValueError('Finite measured pose and paired scores required')
        queries.append(scores)
        key = tuple(pose.reshape(-1))
        if number not in hidden and number > 1 and key not in seen:
            pairs.append(dict(query=number, previous=previous.tolist(), current=scores.tolist()))
        seen.add(key)
        previous = scores
        artifacts.append(dict(path=str(path), sha256=file_digest(path)))
    if (float(queries[0].mean()) != episode['raw_reward']['initial_mean']
            or float(queries[0].std()) != episode['raw_reward']['initial_std']):
        raise ValueError('Initial raw scores differ from saved episode')
    if hidden:
        proof=episode['terminal_confirmation']
        if (queries[proof['query_numbers'][0]-1].tolist()!=proof['initial_scores']
                or queries[proof['query_numbers'][1]-1].tolist()!=proof['selected_scores']):
            raise ValueError('Hidden confirmation receipt differs from durable query scores')
    cursor, best = 1, float(queries[0].mean())
    for transition in episode['transitions']:
        if best != transition['best_previous']:
            raise ValueError('Saved best-score history differs from paired queries')
        for index in range(cursor,cursor+transition['delta_queries']):
            if index+1 not in hidden:best = min(best, float(queries[index].mean()))
        cursor += transition['delta_queries']
        if best != transition['best_current']:
            raise ValueError('Saved best-score step differs from paired queries')
    output = deepcopy(episode)
    output['paired_changes'] = pairs
    output['paired_query_evidence'] = artifacts
    return output
