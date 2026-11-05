"""Closed matched mechanism screen, distinct from structure continuation."""
from .search_core import core_spec
from .score_prefix import contract
from .reward_pilot import reward_config

VARIANTS = ('cost005', 'cost020', 'no_score', 'fixed_query', 'Q4', 'Q6')
DEPLOYMENT_VARIANTS = ('Q4', 'Q6', 'Q16')

def mechanism_spec(name):
    if name not in (*VARIANTS, *DEPLOYMENT_VARIANTS): raise ValueError('Unknown mechanism condition')
    return dict(schema='SEARCH_MECHANISM_V1', name=name,
        explicit_score=name!='no_score', learned_stop=name!='fixed_query',
        query_cap=int(name[1:]) if name in DEPLOYMENT_VARIANTS else 8,
        query_cost={'cost005':.005,'cost020':.02}.get(name,.01))

def reward_for(spec):
    if spec!=mechanism_spec(spec['name']): raise ValueError('Changed mechanism specification')
    cfg=reward_config('G')
    if spec['query_cost']!=.01:
        cfg.update(schema='FORMAL_BATCH_QUERY_COST_SENSITIVITY_V1',query_cost=spec['query_cost'])
    return cfg

def validate_protocol(p):
    if (p['schema']!='UNIFIED_CORE_PILOT_V1' or p['design_stage']!='mechanism_screen'
            or p['updates']!=16 or p['episodes_per_update']!=32 or p['seeds']!=[0,1,2]
            or p['maximum_model_calls']!=600000 or p['query_cap']!=8 or p['query_cost']!=.01):
        raise ValueError('Only the approved 512-episode mechanism screen is allowed')
    if set(p['cells'])!={f'pi05-U12-{v}-seed-{s}' for v in VARIANTS for s in p['seeds']}:
        raise ValueError('Exactly four training and two deployment conditions required')
    for name,c in p['cells'].items():
        spec=mechanism_spec(c['mechanism']['name'])
        if (c['mechanism']!=spec or name!=f"pi05-U12-{spec['name']}-seed-{c['seed']}"
                or c['core_spec']!=core_spec('U12') or c['family']!='pi05' or c['prefix']!=10
                or c['score_contract']!=contract('pi05',10) or c['native_policy'] is not None
                or c.get('task_condition') is not None or c.get('feature_producer') is not None
                or c['observation_scale']!=.16535300100378683
                or c.get('evaluation_only',False)!=(spec['query_cap']!=8)):
            raise ValueError('Changed score, core, representation or mechanism')
        if len(c['evaluation_schedules'])!=1 or list(c['evaluation_schedules'].values())!=[dict(purpose='development',version=16,method='ppo')]:
            raise ValueError('Only fixed paired development endpoint is allowed')
