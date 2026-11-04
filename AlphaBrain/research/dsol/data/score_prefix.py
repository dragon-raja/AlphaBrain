"""Explicit scoring axes; a prefix never changes the native solver or actions."""
from copy import deepcopy

AXES = {'pi05': (10, 10, 5, 7), 'groot_n15': (4, 10, 10, 7), 'rldx': (4, 8, 8, 7)}


def contract(family, prefix):
    if family not in AXES or type(prefix) is not int or prefix not in (3, AXES[family][0]):
        raise ValueError('Explicit P3/native-full comparison required')
    steps, score_actions, executed, dimensions = AXES[family]
    return dict(schema='ACCEL_PREFIX_CONTRACT_V1', family=family, denoising_steps=steps,
                prefix=prefix, scored_actions=score_actions, executed_actions=executed,
                valid_dimensions=dimensions)


def validate(value):
    if value != contract(value['family'], value['prefix']):
        raise ValueError('Changed score/solver/action axes')
    return deepcopy(value)

