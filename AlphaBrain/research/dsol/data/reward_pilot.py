"""Explicit contract for the approved B/G/L warm-start screening experiment.

This is a separate protocol, never an amendment of the frozen A/B experiment.
No results are read when constructing its training/evaluation panels.
"""
from .artifacts import read, sha
from .online_reward import SCHEMA as REWARD_SCHEMA

SCHEMA = 'ONLINE_REWARD_ADAPTATION_V1'
COLLECTION_SCHEMA = 'CONTROLLED_REWARD_ADAPTATION_COLLECTION_V1'
LEARNER_SCHEMA = 'CONTROLLED_REWARD_ADAPTATION_LEARNER_V1'
RECOVERY_SCHEMA = 'CONTROLLED_REWARD_ADAPTATION_RECOVERY_V1'
FIXED_SCALE = .01214765438717935
ARMS = ('B', 'G', 'L')


def load_protocol(path, stamp):
    if not stamp or sha(path) != stamp:
        raise ValueError('Unbound adaptation protocol')
    p = read(path)
    from . import reward_cold_start
    if p['schema'] == reward_cold_start.SCHEMA:
        return reward_cold_start.validate(p)
    if (p['schema'] != SCHEMA or p['arms'] != list(ARMS) or p['seeds'] != [0, 1, 2]
            or p['updates'] != 8 or p['episodes_per_update'] != 32
            or p['initial_update'] != 64 or p['initial_cursor'] != 2048
            or p['fixed_scale'] != FIXED_SCALE or p['query_cap'] != 8
            or p['query_cost'] != .01 or p['floor_fraction'] != .1
            or p['diagnostic_calls'] != 4
            or p['call_caps'] != dict(training=36864, evaluation=131328, engineering=1024, total=169216)):
        raise ValueError('Adaptation scientific scope changed')
    for f, h in p['input_hashes'].items():
        if sha(f) != h:
            raise ValueError('Changed adaptation input: '+f)
    required = {p['authorization'], p['approved_plan'], p['paired_schedules'], p['evaluation_cases']}
    required.update(p['schedules'].values())
    required.update(x['path'] for x in p['initial_checkpoints'])
    if not required.issubset(p['input_hashes']):
        raise ValueError('Unbound adaptation science input')
    authorization, plan = read(p['authorization']), read(p['approved_plan'])
    if (authorization['schema'] != 'USER_APPROVED_ONLINE_REWARD_PILOT_V1'
            or authorization['plan'] != p['approved_plan']
            or authorization['plan_sha256'] != sha(p['approved_plan'])
            or authorization['total_additional_call_cap'] != 169216
            or plan['initial_checkpoints'] != p['initial_checkpoints']):
        raise ValueError('Adaptation authorization differs')
    for spec in read(p['paired_schedules']):
        original = spec['rows']
        actual = read(p['schedules'][str(spec['seed'])])
        expected = [dict(row, episode=i, source_episode=row['episode']) for i, row in enumerate(original)]
        if len(actual) != 256 or actual != expected:
            raise ValueError('Adaptation paired schedule changed')
    return p


def from_campaign(c):
    path = c['adaptation_protocol']
    return load_protocol(path, c['input_hashes'].get(path))


def reward_config(arm):
    if arm not in ARMS:
        raise ValueError('Unknown adaptation arm')
    return dict(schema=REWARD_SCHEMA, kind=arm, fixed_scale=FIXED_SCALE,
                query_cost=.01, floor_fraction=.1)


def evaluation_rows(protocol, original):
    """Bind the preselected 64 cases to the original baseline physical indices."""
    from . import reward_cold_start
    if protocol['schema'] == reward_cold_start.SCHEMA:
        return reward_cold_start.evaluation_rows(original)
    cases = read(protocol['evaluation_cases'])
    if len(cases) != 64:
        raise ValueError('Pilot must preserve the approved 64-case panel')
    lookup = {(r['state_key'], r['view_id']): r for r in original if r['method'] == 'B-seed-0'}
    result = []
    for case in cases:
        row = lookup[(case['state_key'], case['view_id'])]
        if row['split'] != 'validation' or any(row[k] != v for k, v in case.items()):
            raise ValueError('Pilot case differs from approved development panel')
        for arm in ARMS:
            for seed in (0, 1, 2):
                result.append(dict(row, method=f'{arm}-seed-{seed}'))
    if len({(r['state_key'], r['view_id'], r['method']) for r in result}) != 576:
        raise ValueError('Repeated pilot case')
    return result
