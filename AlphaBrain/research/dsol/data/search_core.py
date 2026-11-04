"""Explicit search-core choices, independent of model backends and Torch."""
from copy import deepcopy


VARIANTS = {
    'U0': ('post', 'linear', 'context'),
    'U1': ('pre', 'linear', 'context'),
    'U2': ('post', 'tanh', 'context'),
    'U3': ('post', 'linear', 'record'),
    'U12': ('pre', 'tanh', 'context'),
    'U13': ('pre', 'linear', 'record'),
    'U23': ('post', 'tanh', 'record'),
    'U123': ('pre', 'tanh', 'record'),
}


def core_spec(variant, *, width=128, heads=4, layers=2):
    if variant not in VARIANTS:
        raise ValueError('Unknown predeclared search-core variant')
    state, activation, auxiliary = VARIANTS[variant]
    result = dict(schema='SHARED_SEARCH_CORE_V1', variant=variant, width=width,
                  heads=heads, layers=layers, state_position=state,
                  context_activation=activation, auxiliary_route=auxiliary,
                  pooling='visited_mean', dropout=0., actor_critic='independent')
    return validate_core_spec(result)


def validate_core_spec(spec):
    keys = {'schema', 'variant', 'width', 'heads', 'layers', 'state_position',
            'context_activation', 'auxiliary_route', 'pooling', 'dropout', 'actor_critic'}
    if not isinstance(spec, dict) or set(spec) != keys:
        raise ValueError('Complete immutable core specification required')
    if (spec['schema'] != 'SHARED_SEARCH_CORE_V1' or spec['variant'] not in VARIANTS
            or tuple(spec[k] for k in ('state_position', 'context_activation', 'auxiliary_route'))
            != VARIANTS[spec['variant']]
            or any(type(spec[k]) is not int or spec[k] <= 0 for k in ('width', 'heads', 'layers'))
            or spec['width'] % spec['heads'] or spec['pooling'] != 'visited_mean'
            or spec['dropout'] != 0. or spec['actor_critic'] != 'independent'):
        raise ValueError('Unreviewed core choice or incompatible dimensions')
    return deepcopy(spec)
