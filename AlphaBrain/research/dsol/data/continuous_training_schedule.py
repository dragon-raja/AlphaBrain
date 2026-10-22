"""Balanced on-policy batches without regenerating starts or noise identities."""
from collections import Counter
from copy import deepcopy
import numpy as np

SAMPLING_RULE = 'all_initials_equal_per_update_v1'
TRAINING_REVISION = 'continuous_ppo_stop_mixed_v2'


def validate_balanced_schedule(rows, initials, *, episodes_per_update=32, updates=16):
    if (not initials or len(set(initials)) != len(initials)
            or episodes_per_update % len(initials) or episodes_per_update < len(initials)
            or len(rows) != episodes_per_update * updates
            or [r['episode'] for r in rows] != list(range(len(rows)))):
        raise ValueError('Invalid balanced schedule dimensions or episode identities')
    expected = Counter({i: episodes_per_update // len(initials) for i in initials})
    for start in range(0, len(rows), episodes_per_update):
        if Counter(r['initial'] for r in rows[start:start+episodes_per_update]) != expected:
            raise ValueError('Each update must contain equal quotas of every initial')
    if sorted(r['source_episode'] for r in rows) != list(range(len(rows))):
        raise ValueError('Missing or duplicated source episode')
    return dict(rule=SAMPLING_RULE, updates=updates, episodes=len(rows),
                initials_per_update=len(initials), episodes_per_initial_per_update=next(iter(expected.values())))


def balanced_schedule(source, initials, *, episodes_per_update=32, updates=16, seed=0):
    """Reorder a frozen set of episodes; preserve each row's science fields.

    source_episode identifies the original row, independently of host or worker.
    The local RNG affects ordering only, never the existing proposal/noise seeds.
    """
    if (type(seed) is not int or seed < 0 or not initials
            or len(set(initials)) != len(initials)
            or episodes_per_update < len(initials) or episodes_per_update % len(initials)
            or len(source) != episodes_per_update * updates
            or [r['episode'] for r in source] != list(range(len(source)))
            or any('source_episode' in r for r in source)):
        raise ValueError('Original complete episode schedule and divisible quotas required')
    quota = episodes_per_update // len(initials)
    if Counter(r['initial'] for r in source) != Counter({i: quota*updates for i in initials}):
        raise ValueError('Original initial multiplicity differs from balanced quotas')
    rng = np.random.default_rng(seed)
    buckets = {}
    for initial in initials:
        bucket = [r for r in source if r['initial'] == initial]
        buckets[initial] = [bucket[int(i)] for i in rng.permutation(len(bucket))]
    result = []
    for update in range(updates):
        batch = [r for initial in initials for r in buckets[initial][update*quota:(update+1)*quota]]
        for index in rng.permutation(len(batch)):
            row = deepcopy(batch[int(index)])
            row['source_episode'], row['episode'] = row['episode'], len(result)
            result.append(row)
    validate_balanced_schedule(result, initials, episodes_per_update=episodes_per_update, updates=updates)
    return result


def validate_preserved_episodes(source, mixed):
    """Reject altered starts, seeds, states or any other source field."""
    if (len(source) != len(mixed)
            or [r['episode'] for r in source] != list(range(len(source)))
            or [r['episode'] for r in mixed] != list(range(len(mixed)))
            or sorted(r['source_episode'] for r in mixed) != list(range(len(source)))):
        raise ValueError('Source episode set changed')
    for row in mixed:
        original = deepcopy(row)
        original['episode'] = original.pop('source_episode')
        if original != source[original['episode']]:
            raise ValueError('Source start/state/seed identity changed')
