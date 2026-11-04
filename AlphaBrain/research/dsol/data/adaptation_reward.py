"""Domain-bound, small-batch reward settlement; no model or optimizer imports.

Candidate protocol, distinct from published cumulative G. Only formal initial
queries enter the window. Two-member disagreement is a heuristic, not a noise
confidence bound. A new domain must start an empty state, never inherit a scale.
"""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np

from .online_reward import validate_raw

SCHEMA = 'DOMAIN_BATCH_REWARD_V1'


def validate_config(config):
    expected = {'schema', 'domain_sha256', 'mode', 'batch_size', 'window_size',
                'noise_multiplier', 'max_scale_ratio', 'query_cost'}
    if (set(config) != expected or config['schema'] != SCHEMA
            or config['mode'] not in ('first_batch_fixed', 'window')
            or type(config['batch_size']) is not int or config['batch_size'] != 8
            or type(config['window_size']) is not int or config['window_size'] != 32
            or config['noise_multiplier'] != 2. or config['max_scale_ratio'] != 2.
            or config['query_cost'] != .01):
        raise ValueError('Unsupported small-batch adaptation contract')
    identity = config['domain_sha256']
    if (not isinstance(identity, str) or len(identity) != 64
            or any(c not in '0123456789abcdef' for c in identity)):
        raise ValueError('Explicit domain SHA256 required')
    return config


def _binding(config):
    validate_config(config)
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def empty_state(config):
    return dict(schema=SCHEMA, binding=_binding(config), batches=0, count=0,
                initials=[], disagreements=[], scale=None)


def validate_state(state, config, *, update, count):
    """Validate a restored settlement boundary, including domain and protocol."""
    expected = {'schema', 'binding', 'batches', 'count', 'initials',
                'disagreements', 'scale'}
    if (set(state) != expected or state['schema'] != SCHEMA
            or state['binding'] != _binding(config)
            or type(update) is not int or update < 0
            or type(state['batches']) is not int or type(state['count']) is not int
            or state['batches'] != update or state['count'] != count
            or count != update * config['batch_size']
            or not isinstance(state['initials'], list)
            or not isinstance(state['disagreements'], list)
            or len(state['initials']) != min(count, config['window_size'])
            or len(state['disagreements']) != len(state['initials'])):
        raise ValueError('Domain/protocol/checkpoint boundary mismatch')
    values = state['initials'] + state['disagreements']
    if values and (not np.isfinite(values).all() or min(values) < 0
                   or any(d > a for a, d in zip(state['initials'], state['disagreements']))):
        raise ValueError('Invalid saved initial statistics')
    if update == 0:
        if state['scale'] is not None:
            raise ValueError('New domain must not inherit a reward scale')
    elif (not isinstance(state['scale'], (float, int)) or isinstance(state['scale'], bool)
          or not math.isfinite(state['scale']) or state['scale'] <= 0):
        raise ValueError('Invalid settled scale')


def assign_batch(episodes, config, state, *, update):
    """Pure settlement before GAE; commit next state with optimizer atomically.

    The caller verifies model/domain provenance and unique episode identities.
    This function rejects incomplete/mixed-version batches; it never launches,
    replays, collects calibration data, or changes observation normalization.
    """
    validate_config(config)
    validate_state(state, config, update=update, count=update * config['batch_size'])
    if (len(episodes) != config['batch_size']
            or any(e.get('status') != 'COMPLETE' or e['policy_update'] != update
                   for e in episodes)):
        raise ValueError('Complete same-version eight-episode batch required')
    pairs = [validate_raw(e) for e in episodes]
    window = config['window_size']
    means = (state['initials'] + [a for a, _ in pairs])[-window:]
    disagreements = (state['disagreements'] + [d for _, d in pairs])[-window:]
    median = float(np.median(means))
    noise = float(np.quantile(disagreements, .75, method='linear'))
    proposed = max(median, config['noise_multiplier'] * noise)
    if not math.isfinite(proposed) or proposed <= 0:
        raise ValueError('Formal observations do not identify a positive scale')
    previous = state['scale']
    if previous is None:
        scale = proposed
    elif config['mode'] == 'first_batch_fixed':
        scale = previous
    else:
        ratio = config['max_scale_ratio']
        scale = min(max(proposed, previous / ratio), previous * ratio)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Nonfinite or underflowed settled scale')
    output = deepcopy(episodes)
    for e in output:
        for t in e['transitions']:
            gain = t['best_previous'] - t['best_current']
            t['reward'] = gain / scale - config['query_cost'] * t['delta_queries']
            if not math.isfinite(t['reward']):
                raise ValueError('Nonfinite adaptation reward')
    next_state = dict(schema=SCHEMA, binding=state['binding'], batches=update + 1,
                      count=state['count'] + len(episodes), initials=means,
                      disagreements=disagreements, scale=scale)
    receipt = dict(schema=SCHEMA, domain_sha256=config['domain_sha256'],
                   mode=config['mode'], update=update + 1, count=next_state['count'],
                   window_count=len(means), initial_median=median,
                   disagreement_q75=noise, proposed_scale=proposed,
                   denominator=scale, previous_scale=previous,
                   scale_limited=scale != proposed,
                   disagreement_floor_active=2 * noise > median,
                   rewards=[sum(t['reward'] for t in e['transitions']) for e in output],
                   additional_model_calls=0, frozen_during_optimizer=True)
    return output, next_state, receipt
