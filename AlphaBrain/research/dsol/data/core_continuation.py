"""Closed, separately authorized continuation of the two U/T candidates."""
from .search_core import core_spec
from .score_prefix import contract

ENDPOINTS = (32, 48, 64, 80, 96)


def validate_protocol(p):
    if (p['schema'] != 'UNIFIED_CORE_PILOT_V1' or p['design_stage'] != 'long_continuation'
            or p['updates'] != 96 or p['start_update'] != 16
            or p['episodes_per_update'] != 32 or p['seeds'] != [0, 1, 2]
            or p['maximum_model_calls'] != 600000 or p['query_cap'] != 8 or p['query_cost'] != .01):
        raise ValueError('Only the explicitly approved 512 to 3072 continuation is allowed')
    expected = {f'pi05-{v}-seed-{s}' for v in ('U12', 'U3') for s in p['seeds']}
    if set(p['cells']) != expected:
        raise ValueError('Two matched candidates and all three original seeds required')
    for name, cell in p['cells'].items():
        if (name != f"pi05-{cell['core_spec']['variant']}-seed-{cell['seed']}"
                or cell['core_spec'] != core_spec(cell['core_spec']['variant'])
                or cell['family'] != 'pi05' or cell['prefix'] != 10
                or cell['score_contract'] != contract('pi05', 10)
                or cell['native_policy'] is not None or cell.get('feature_producer') is not None
                or cell.get('task_condition') is not None
                or cell['observation_scale'] != .16535300100378683):
            raise ValueError('Continuation cannot change the core, representation or score')
        schedules = cell['evaluation_schedules']
        if (len(schedules) != len(ENDPOINTS)
                or sorted(m['version'] for m in schedules.values()) != list(ENDPOINTS)
                or any(m != dict(purpose='development', version=m['version'], method='ppo') for m in schedules.values())):
            raise ValueError('Exactly the five predeclared development endpoints required')
        origin = cell['continuation']
        if origin['update'] != 16 or len(origin['checkpoint_sha256']) != 64 or not origin['source_training']:
            raise ValueError('Bound complete source checkpoint required')
