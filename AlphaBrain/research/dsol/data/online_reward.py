"""Batch-frozen reward scales from formal interaction, without extra queries.

Only initial scores enter the cumulative reference statistics. Candidate minima
never set their own denominator. Two-member disagreement is a heuristic floor,
not a confidence interval. Raw rollout files remain immutable.
"""
from copy import deepcopy
import math
import numpy as np

SCHEMA = 'FORMAL_BATCH_REWARD_V1'


def validate_config(config):
    if (set(config) != {'schema', 'kind', 'fixed_scale', 'query_cost', 'floor_fraction'}
            or config['schema'] not in (SCHEMA, 'FORMAL_BATCH_QUERY_COST_SENSITIVITY_V1')
            or config['kind'] not in ('B', 'G', 'L')
            or (config['schema']==SCHEMA and config['query_cost']!=.01)
            or (config['schema']!=SCHEMA and (config['kind']!='G' or config['query_cost'] not in (.005,.02)))
            or config['floor_fraction'] != .1
            or not math.isfinite(config['fixed_scale']) or config['fixed_scale'] <= 0):
        raise ValueError('Unsupported formal-interaction reward contract')
    return config


def empty_state():
    return dict(schema=SCHEMA, batches=0, initials=[], disagreements=[])


def validate_raw(episode, metrics=None):
    confirmed = episode.get('terminal_confirmation') is not None
    if confirmed:
        from .confirmed_reward import validate_confirmation
        validate_confirmation(episode)
    raw = episode['raw_reward']
    initial, disagreement = raw['initial_mean'], raw['initial_std']
    if (raw['schema'] != SCHEMA or raw['score_members'] != 2
            or not np.isfinite([initial, disagreement]).all()
            or min(initial, disagreement) < 0 or disagreement > initial
            or not math.isfinite(raw['collector_scale']) or raw['collector_scale'] <= 0):
        raise ValueError('Invalid initial two-member score statistics')
    previous, queries = initial, 1
    for t in episode['transitions']:
        before, after, dq = (t[k] for k in ('best_previous', 'best_current', 'delta_queries'))
        if (not np.isfinite([before, after]).all() or before != previous
                or not 0 <= after <= before or type(dq) is not int or not 0 <= dq <= (4 if confirmed else 2)):
            raise ValueError('Broken raw gain/query chain')
        expected = (before-after)/raw['collector_scale'] - .01*dq
        if t['reward'] != expected:
            raise ValueError('Collector reward disagrees with raw gain/query record')
        previous = after
        queries += dq
    if not 1 <= queries <= 8:
        raise ValueError('Raw query budget exceeded')
    if metrics is not None and (metrics['initial_accel'] != initial
            or metrics['best_accel'] != previous or metrics['queries'] != queries):
        raise ValueError('Raw reward and completion metrics disagree')
    return initial, disagreement


def assign_batch(episodes, config, state, *, update):
    """Return copies, next checkpoint statistics and an auditable settlement.

    Call exactly once per complete on-policy barrier, before GAE. The caller
    checkpoints state together with optimizer weights; no mutable global cache.
    Undefined zero scale fails closed instead of inventing a numerical epsilon.
    """
    validate_config(config)
    if (state['schema'] != SCHEMA or state['batches'] != update
            or len(state['initials']) != len(state['disagreements'])
            or len(state['initials']) != update*len(episodes) or not episodes
            or any(e['policy_update'] != update for e in episodes)):
        raise ValueError('Reward statistics do not match the on-policy boundary')
    pairs = [validate_raw(e) for e in episodes]
    means = state['initials'] + [p[0] for p in pairs]
    noises = state['disagreements'] + [p[1] for p in pairs]
    if (not np.isfinite(means+noises).all() or min(means+noises) < 0
            or any(n > m for m, n in zip(means, noises))):
        raise ValueError('Invalid saved online statistics')
    scale = float(np.median(means))
    noise = float(np.quantile(noises, .75, method='linear'))
    tau = max(config['floor_fraction']*scale, noise)
    if config['kind'] != 'B' and (scale <= 0 or tau <= 0):
        raise ValueError('Formal interactions do not identify a positive scale')
    output = deepcopy(episodes)
    for episode in output:
        for t in episode['transitions']:
            before, after, dq = (t[k] for k in ('best_previous', 'best_current', 'delta_queries'))
            gain = before-after
            if config['kind'] == 'L':
                improvement = math.log1p(gain/(after+tau))
            else:
                improvement = gain/(config['fixed_scale'] if config['kind'] == 'B' else scale)
            t['reward'] = improvement-config['query_cost']*dq
            if not math.isfinite(t['reward']):
                raise ValueError('Nonfinite settled reward')
    next_state = dict(schema=SCHEMA, batches=update+1, initials=means, disagreements=noises)
    receipt = dict(schema=SCHEMA, kind=config['kind'], update=update+1,
        count=len(means), scale=scale, disagreement_q75=noise, tau=tau,
        denominator=config['fixed_scale'] if config['kind'] == 'B' else scale,
        rewards=[sum(t['reward'] for t in e['transitions']) for e in output],
        additional_model_calls=0, frozen_during_optimizer=True, query_cost=config['query_cost'])
    return output, next_state, receipt
