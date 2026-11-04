"""Signed independent terminal gain on online change units; opt-in pilot only."""
from copy import deepcopy
import numpy as np
from . import change_reward

SCHEMA='CONFIRMED_CHANGE_REWARD_V1'
RECORD='HIDDEN_TERMINAL_CONFIRMATION_V1'


def config():return dict(schema=SCHEMA, change_config=change_reward.default_config())


def validate_config(c):
    if c != config():raise ValueError('Frozen confirmed-change pilot configuration required')
    return c


def empty_state():return change_reward.empty_state()


def validate_confirmation(e):
    c=e['terminal_confirmation'];q=1+sum(t['delta_queries'] for t in e['transitions'])
    if any(t['delta_queries']>2 for t in e['transitions'][:-1]):raise ValueError('Only terminal transition may include confirmation')
    if (c['schema']!=RECORD or c['hidden_from_actor'] is not True or c['queries']!=q
            or len(c['search_noise_hashes'])!=2 or len(c['confirmation_noise_hashes'])!=2
            or len(set(c['search_noise_hashes']+c['confirmation_noise_hashes']))!=4):
        raise ValueError('Independent hidden confirmation identities required')
    for value in c['search_noise_hashes']+c['confirmation_noise_hashes']+[c['initial_input_sha256'],c['selected_input_sha256']]:
        if not isinstance(value,str) or len(value)!=64 or any(x not in '0123456789abcdef' for x in value):
            raise ValueError('Full confirmation input/noise SHA256 required')
    if c['skipped_same_input']:
        if (c['initial_input_sha256']!=c['selected_input_sha256'] or c['query_numbers']
                or c['initial_scores'] is not None or c['selected_scores'] is not None):
            raise ValueError('Only exact same-input confirmation may be skipped')
        return 0.
    a=np.asarray(c['initial_scores'],float);b=np.asarray(c['selected_scores'],float)
    if (a.shape!=(2,) or b.shape!=(2,) or not np.isfinite([a,b]).all()
            or min(*a,*b)<0 or c['query_numbers']!=[q-1,q]
            or e['transitions'][-1]['delta_queries']<2):
        raise ValueError('Final paired confirmation queries/scores differ')
    return float((a-b).mean())


def assign_batch(episodes,c,state,*,update):
    validate_config(c)
    gains=[validate_confirmation(e) for e in episodes]
    _,next_state,receipt=change_reward.assign_batch(episodes,c['change_config'],state,update=update)
    output=deepcopy(episodes);scale=receipt['denominator']
    for e,gain in zip(output,gains):
        for t in e['transitions']:t['reward']=-.01*t['delta_queries']
        e['transitions'][-1]['reward']+=gain/scale
    receipt.update(schema=SCHEMA,confirmed_gains=gains,reward_rule='signed_terminal_only_minus_all_queries',
        rewards=[sum(t['reward'] for t in e['transitions']) for e in output])
    return output,next_state,receipt
