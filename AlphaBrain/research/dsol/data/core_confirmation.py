"""Bounded tail/early-checkpoint/deployment evidence; no new method selection.

Separate scope leaves the completed 600k continuation and mechanism contracts
unchanged. Q16 is evaluation-only and may not enter the learner.
"""
from .core_mechanism import DEPLOYMENT_VARIANTS, mechanism_spec
from .search_core import core_spec
from .score_prefix import contract


def validate_protocol(p):
    if (p['schema'] != 'UNIFIED_CORE_PILOT_V1' or p['design_stage'] != 'core_confirmation'
            or p['updates'] != 128 or p['start_update'] != 96
            or p['episodes_per_update'] != 32 or p['seeds'] != [0, 1, 2]
            or p['maximum_model_calls'] != 300000 or p['query_cap'] != 8
            or p['query_cost'] != .01):
        raise ValueError('Only the bounded 3072-to-4096 evidence package is allowed')
    train = {f'pi05-{v}-seed-{s}' for v in ('U12', 'U3') for s in p['seeds']}
    deploy = {f'pi05-U12-{v}-seed-{s}' for v in DEPLOYMENT_VARIANTS for s in p['seeds']}
    if set(p['cells']) != train | deploy:
        raise ValueError('Exactly six continuations and nine deployment conditions required')
    for name, c in p['cells'].items():
        if (c['family'] != 'pi05' or c['prefix'] != 10 or c['score_contract'] != contract('pi05', 10)
                or c['native_policy'] is not None or c.get('feature_producer') is not None
                or c.get('task_condition') is not None or c['seed'] not in p['seeds']
                or c['observation_scale'] != .16535300100378683
                or c['core_spec'] != core_spec(c['core_spec']['variant'])):
            raise ValueError('Changed frozen scoring, representation or core')
        evaluation = name in deploy
        if c.get('evaluation_only', False) != evaluation:
            raise ValueError('Deployment evidence cannot train')
        if evaluation:
            spec = mechanism_spec(c['mechanism']['name'])
            if (spec['name'] not in DEPLOYMENT_VARIANTS or c['mechanism'] != spec
                    or c['core_spec'] != core_spec('U12')
                    or name != f"pi05-U12-{spec['name']}-seed-{c['seed']}"):
                raise ValueError('Changed deployment-only budget comparison')
            versions = [96]
        else:
            if (name != f"pi05-{c['core_spec']['variant']}-seed-{c['seed']}"
                    or c.get('mechanism') is not None):
                raise ValueError('Continuation must preserve the original policy')
            origin = c['continuation']
            if (origin['update'] != 96 or len(origin['checkpoint_sha256']) != 64
                    or not origin['source_training']):
                raise ValueError('Exact settled 3072-episode checkpoint required')
            versions = [0, 4, 8, 112, 128]
        metas = list(c['evaluation_schedules'].values())
        if (len(metas) != len(versions) or sorted(m['version'] for m in metas) != versions
                or any(m != dict(purpose='development', version=m['version'], method='ppo') for m in metas)):
            raise ValueError('Changed or missing predeclared development endpoints')
