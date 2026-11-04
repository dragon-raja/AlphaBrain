"""Hash-bound small-domain adaptation inputs, not a dispatch authorization.

The existing controller still owns grants, reservation and recovery. A protocol
can be validated on CPU without constructing a model or starting a collector.
"""
from .artifacts import read, sha
from . import adaptation_reward

SCHEMA = 'DOMAIN_ADAPTATION_PROTOCOL_V1'
LEARNER_SCHEMA = 'CONTROLLED_DOMAIN_ADAPTATION_LEARNER_V1'
COLLECTION_SCHEMA = 'CONTROLLED_DOMAIN_ADAPTATION_COLLECTION_V1'
ARMS = {f'{init}-{mode}': (init, mode)
        for init in ('source', 'random') for mode in ('first_batch_fixed', 'window')}


def from_campaign(config):
    path = config['adaptation_protocol']
    if config['input_hashes'].get(path) != sha(path) or config['protocol'] != path:
        raise ValueError('Unbound domain adaptation protocol')
    p = read(path)
    if (p['schema'] != SCHEMA or p['arms'] != list(ARMS) or p['seeds'] != [0, 1, 2]
            or p['updates'] != 8 or p['episodes_per_update'] != 8
            or p['query_cap'] != 8 or p['query_cost'] != .01
            or p['visual_dim'] != 4608 or set(p['schedules']) != {'0', '1', '2'}
            or set(p['source_checkpoints']) != {'0', '1', '2'}):
        raise ValueError('Unsupported domain adaptation protocol')
    required = {p['domain'], *p['schedules'].values()}
    required.update(x['path'] for x in p['source_checkpoints'].values())
    if not required.issubset(p['input_hashes']):
        raise ValueError('Unbound domain adaptation inputs')
    for filename, stamp in p['input_hashes'].items():
        if sha(filename) != stamp:
            raise ValueError('Changed domain adaptation input: ' + filename)
    domain = read(p['domain'])
    if (domain['schema'] != 'SEARCH_ADAPTATION_DOMAIN_V1'
            or domain['visual_dim'] != p['visual_dim']
            or domain['visual_contract'] not in p['input_hashes']):
        raise ValueError('Missing shared visual interface identity')
    for filename in (domain['sim_config'], domain['training_inputs'], domain['training_noise']):
        if filename not in p['input_hashes']:
            raise ValueError('Unbound domain simulator/state/noise')
    states = read(domain['training_inputs'])['records']
    if not states or any(r['state']['split'] != 'train' for r in states):
        raise ValueError('Adaptation must use training states only')
    visual_sha = p['input_hashes'][domain['visual_contract']]
    for seed, source in p['source_checkpoints'].items():
        if source['visual_contract_sha256'] != visual_sha:
            raise ValueError('Source actor visual contract differs')
        rows = read(p['schedules'][seed])
        if len(rows) != 64 or [r['episode'] for r in rows] != list(range(64)):
            raise ValueError('Exact 64-episode adaptation schedule required')
        for row in rows:
            i = row['initial']
            if type(i) is not int or not 0 <= i < len(states):
                raise ValueError('Adaptation state index out of range')
            state = states[i]['state']
            if row['state_key'] != state['pair_key'] or row['task'] != state['task_id']:
                raise ValueError('Adaptation schedule state identity differs')
    return p


def reward_config(protocol, arm):
    if arm not in ARMS:
        raise ValueError('Unknown adaptation arm')
    return adaptation_reward.validate_config(dict(
        schema=adaptation_reward.SCHEMA, domain_sha256=sha(protocol['domain']),
        mode=ARMS[arm][1], batch_size=8, window_size=32,
        noise_multiplier=2., max_scale_ratio=2., query_cost=.01))
