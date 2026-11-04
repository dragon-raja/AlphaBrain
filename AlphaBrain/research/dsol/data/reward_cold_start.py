"""Hash-bound full native G experiment; separate from the warm-start pilot."""
from .artifacts import read, sha
from . import native_recovery

SCHEMA = 'ONLINE_REWARD_COLD_START_V1'


def evaluation_rows(original):
    rows = [dict(r, method=f'G-seed-{seed}') for r in original
            if r['method'] == 'B-seed-0' for seed in (0, 1, 2)]
    if len(rows) != 9600 or len({(r['state_key'], r['view_id'], r['method']) for r in rows}) != 9600:
        raise ValueError('Complete forty-task G recovery panel required')
    return rows


def validate(p):
    if (p['schema'] != SCHEMA or p['arms'] != ['G'] or p['seeds'] != [0, 1, 2]
            or p['updates'] != 64 or p['episodes_per_update'] != 32
            or p['initial_update'] != 0 or p['initial_cursor'] != 0
            or p['fixed_scale'] != .01214765438717935 or p['query_cap'] != 8
            or p['query_cost'] != .01 or p['floor_fraction'] != .1
            or p['diagnostic_calls'] != 4 or p['episodes_per_block'] != 2):
        raise ValueError('Cold-start G scope changed')
    required = {p['authorization'], p['original_recovery_schedule'], p['reuse_audit']}
    required.update(p['schedules'].values())
    required.update(x['path'] for x in p['initial_checkpoints'])
    required.update(x['path'] for x in p['original_learners'])
    required.update(p['first_batch_references'].values())
    if not required.issubset(p['input_hashes']):
        raise ValueError('Unbound cold-start input')
    for path, stamp in p['input_hashes'].items():
        if sha(path) != stamp:
            raise ValueError('Changed cold-start input: ' + path)
    if [x['seed'] for x in p['initial_checkpoints']] != [0, 1, 2]:
        raise ValueError('Cold-start initialization seeds differ')
    for seed, spec in enumerate(p['original_learners']):
        cfg = read(spec['path'])
        actual = read(p['schedules'][str(seed)])
        if (cfg['run'] != f'B-seed-{seed}' or spec['seed'] != seed
                or p['initial_checkpoints'][seed]['path'] != cfg['initial_checkpoint']
                or sha(cfg['initial_checkpoint']) != cfg['input_hashes'][cfg['initial_checkpoint']]
                or sha(cfg['schedule']) != cfg['input_hashes'][cfg['schedule']]
                or len(actual) != 2048 or actual != read(cfg['schedule'])[:2048]):
            raise ValueError('Cold-start initialization or paired schedule differs')
    rows = evaluation_rows(read(p['original_recovery_schedule']))
    limits = native_recovery.limits(rows, paired_diagnostic=True)
    caps = dict(training=3*2048*8*2,
                evaluation=sum(limits[k] for k in ('score', 'paired_execution', 'nominal_execution')),
                engineering=1024)
    caps['total'] = sum(caps.values())
    auth = read(p['authorization'])
    if (p['call_caps'] != caps or auth['schema'] != 'USER_APPROVED_G_FULL_V1'
            or auth['call_caps'] != caps or not auth['user_approved_full_first_step']
            or read(p['reuse_audit'])['status'] != 'PASS_B_REFERENCE_IDENTITY_AUDIT'):
        raise ValueError('Cold-start authorization, budget or comparison audit differs')
    return p
