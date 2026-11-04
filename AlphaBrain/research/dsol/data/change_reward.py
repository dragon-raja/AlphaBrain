"""Candidate paired-view change units; pure CPU settlement, never a permit.

Statistics use all new-pose signed contrasts, not just record improvements.
The disagreement floor is a two-member heuristic, not a confidence guarantee.
This opt-in reward changes the gain/cost trade-off; legacy G is unchanged.
"""
from copy import deepcopy
import math

import numpy as np

from .online_reward import validate_raw

SCHEMA = 'PAIRED_CHANGE_REWARD_V1'


def default_config():
    return dict(schema=SCHEMA, query_cost=.01, window_size=256, min_pairs=8,
                quantile=.75, noise_multiplier=2., max_scale_ratio=2., smoothing=.25)


def validate_config(config):
    if config != default_config():
        raise ValueError('Explicit frozen paired-change screening configuration required')
    return config


def empty_state():
    return dict(schema=SCHEMA, batches=0, initials=[], disagreements=[],
                changes=[], pair_noises=[], scale=None)


def assign_batch(episodes, config, state, *, update):
    """One complete, on-policy batch; no mutation or additional observation."""
    validate_config(config)
    if (not episodes or state['schema'] != SCHEMA or state['batches'] != update
            or len(state['initials']) != update * len(episodes)
            or len(state['initials']) != len(state['disagreements'])
            or len(state['changes']) != len(state['pair_noises'])
            or any(e['policy_update'] != update for e in episodes)):
        raise ValueError('Paired-change statistics do not match the complete batch boundary')
    initials = [validate_raw(e) for e in episodes]
    changes, noises = list(state['changes']), list(state['pair_noises'])
    added = 0
    for episode in episodes:
        previous_query = 1
        for pair in episode['paired_changes']:
            before = np.asarray(pair['previous'], dtype=float)
            after = np.asarray(pair['current'], dtype=float)
            query = pair['query']
            if (before.shape != (2,) or after.shape != (2,)
                    or not np.isfinite([before, after]).all() or min(*before, *after) < 0
                    or type(query) is not int or not previous_query < query <= 8):
                raise ValueError('Ordered finite same-noise view contrasts required')
            difference = after - before
            changes.append(abs(float(difference.mean())))
            noises.append(abs(float(difference[0] - difference[1])) / 2)
            previous_query = query
            added += 1
    changes, noises = changes[-config['window_size']:], noises[-config['window_size']:]
    means = state['initials'] + [v[0] for v in initials]
    disagreements = state['disagreements'] + [v[1] for v in initials]
    if (not np.isfinite(means + disagreements + changes + noises).all()
            or min(means + disagreements + changes + noises) < 0):
        raise ValueError('Invalid saved change-scale statistics')
    scale = state['scale']
    if scale is not None and (not math.isfinite(scale) or scale <= 0):
        raise ValueError('Invalid saved change scale')
    proposed = None
    if len(changes) >= config['min_pairs'] and added:
        proposed = max(float(np.quantile(changes, config['quantile'])),
                       config['noise_multiplier'] * float(np.quantile(noises, config['quantile'])))
    if proposed is not None and proposed > 0:
        if scale is None:
            scale = proposed
        else:
            smooth = math.exp((1-config['smoothing'])*math.log(scale)
                              + config['smoothing']*math.log(proposed))
            scale = min(max(smooth, scale/config['max_scale_ratio']), scale*config['max_scale_ratio'])
        status = 'CHANGE_SCALE_ESTIMATED'
    elif scale is None:
        scale = float(np.median(means))
        status = 'INITIAL_SCALE_FALLBACK_INSUFFICIENT_CHANGE_EVIDENCE'
    else:
        status = 'PREVIOUS_SCALE_RETAINED_NO_NEW_USABLE_EVIDENCE'
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Interactions do not identify a positive reward scale')
    output = deepcopy(episodes)
    for episode in output:
        for transition in episode['transitions']:
            gain = transition['best_previous'] - transition['best_current']
            transition['reward'] = gain/scale - config['query_cost']*transition['delta_queries']
            if not math.isfinite(transition['reward']):
                raise ValueError('Nonfinite settled change reward')
    next_state = dict(schema=SCHEMA, batches=update+1, initials=means,
                      disagreements=disagreements, changes=changes, pair_noises=noises, scale=scale)
    receipt = dict(schema=SCHEMA, update=update+1, count=len(means), denominator=scale,
                   previous_scale=state['scale'], proposed_scale=proposed,
                   status=status, new_pairs=added, window_pairs=len(changes),
                   contrast_q75=float(np.quantile(changes,.75)) if changes else None,
                   paired_disagreement_q75=float(np.quantile(noises,.75)) if noises else None,
                   rewards=[sum(t['reward'] for t in e['transitions']) for e in output],
                   frozen_during_optimizer=True, additional_model_calls=0)
    return output, next_state, receipt
