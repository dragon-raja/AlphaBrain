"""Outcome-independent native recovery panel and its per-session budgets."""
from .native_view_panel import views
from .native_baseline import HORIZONS
from .flow_noise import stable_uint64

SCHEMA = 'NATIVE_CONTROLLED_RECOVERY_V1'
METHODS = {
    'validation': ['local', 'random_from_local', 'old_ppo', 'A-seed-0', 'B-seed-0'],
    'seen_new_initials': ['local', 'old_ppo', 'A-seed-0', 'B-seed-0'],
    'unseen_test': ['local', 'random_from_local', 'old_ppo',
                    'A-seed-0', 'A-seed-1', 'A-seed-2', 'B-seed-0', 'B-seed-1', 'B-seed-2'],
}


def rows(states, records):
    """Eight directions, rotated three severity levels; never inspect outcomes.

    The split on each input denotes TASK membership. Seen-task evaluation uses
    new official initial states 40--49, disjoint from training states 0--19.
    """
    if len(states) != 400 or len(records) != 400:
        raise ValueError('Complete forty-task baseline input collection required')
    result = []
    for state in states:
        original = records[state['initial']]['state']
        split = {'train': 'seen_new_initials', 'validation': 'validation',
                 'test_unseen_task': 'unseen_test'}[original['split']]
        if original['pair_key'] != state['state_key']:
            raise ValueError('State index differs')
        expected = range(20, 30) if split == 'validation' else range(40, 50)
        if original['initialization']['init_state_index'] not in expected:
            raise ValueError('Evaluation initial state overlaps training or wrong split')
        for direction in range(8):
            level = (state['initial'] + direction) % 3 + 1
            view = 1 + 8 * (level - 1) + direction
            label = f"{state['state_key']}::{view}"
            for method in METHODS[split]:
                result.append(dict(state, split=split, start=direction, view_id=view,
                    angles_deg=views()[view], method=method, query_cap=8,
                    execution_repeats=[0, 1], proposal_seed=stable_uint64(label, root_seed=20261024),
                    random_seed=stable_uint64(label, root_seed=20261025)))
    counts = {s: sum(r['split'] == s for r in result) for s in METHODS}
    if counts != dict(validation=3200, seen_new_initials=6400, unseen_test=8640):
        raise ValueError('Recovery panel coverage differs from approved budget')
    return result


def limits(rows, *, paired_diagnostic=False):
    return dict(episodes=2 * len(rows), score=(20 if paired_diagnostic else 18) * len(rows),
        paired_execution=sum(2 * (HORIZONS[r['suite']] // 5) for r in rows), nominal_execution=0)


def addresses(rows):
    return [dict(model='native_reference', task=r['task'], initial=r['state_key'],
                 view=f"{r['view_id']}:{r['method']}", stream='execution', repeat=repeat,
                 noise=f'paired-seed-{repeat}') for r in rows for repeat in (0, 1)]


def blocks(rows, *, size=3):
    if size < 1:
        raise ValueError('Positive block size required')
    return [dict(index=i // size, rows=rows[i:i + size],
                 limits=limits(rows[i:i + size]), addresses=addresses(rows[i:i + size]))
            for i in range(0, len(rows), size)]


def waves(blocks, *, size=144):
    """Keep the first complete block as a gate; never reorder scientific rows."""
    if type(size) is not int or size < 1 or not blocks:
        raise ValueError('Nonempty blocks and positive integer wave size required')
    return [blocks[:1]] + [blocks[i:i + size] for i in range(1, len(blocks), size)]


def reference_artifacts(references, rows):
    """Unique byte-hash obligations for these rows, including the nominal view.

    No filesystem cache or NFS timestamps are used. Conflicting declarations for
    the same file fail instead of allowing dictionary overwrite to choose one.
    """
    result = {}
    for row in rows:
        for view in (0, row['view_id']):
            ref = references['views'][f"{row['state_key']}::{view}"]
            for path, expected in ref['artifacts'].items():
                if path in result and result[path] != expected:
                    raise ValueError('Conflicting baseline artifact identities')
                result[path] = expected
    return result
