"""Closed scopes for separately authorized shared-core screening stages."""
from .search_core import core_spec
from .score_prefix import contract

SCHEMA = 'UNIFIED_CORE_PILOT_V1'
STAGES = {'single_factors': ('U0', 'U1', 'U2', 'U3'),
          'combinations': ('U12', 'U13', 'U23', 'U123')}


def validate_protocol(p):
    if p.get('design_stage') == 'long_continuation':
        from .core_continuation import validate_protocol as validate_long
        return validate_long(p)
    if p.get('design_stage') == 'mechanism_screen':
        from .core_mechanism import validate_protocol as validate_mechanism
        return validate_mechanism(p)
    task_stage=p.get('design_stage')=='task_condition'
    replication=p.get('design_stage')=='seed_replication'
    seeds=[2] if replication else [0,1]
    ceiling=170000 if replication else (90000 if task_stage else 120000)
    if (p['schema'] != SCHEMA or p['updates'] != 16 or p['episodes_per_update'] != 32
            or p['seeds'] != seeds or p['maximum_model_calls'] != ceiling
            or p['query_cap'] != 8 or p['query_cost'] != .01):
        raise ValueError('Only approved first-stage core screening is allowed')
    stage = p.get('design_stage', 'single_factors')
    if stage not in STAGES and not task_stage and not replication:
        raise ValueError('Explicit bounded structure-screening stage required')
    if task_stage or replication:
        from .task_condition import validate_binding
        variants=p['task_variants']
        selected=p['selected_base']
        expected_variants=['U0','U123']+([selected] if selected not in ('U0','U123') else [])
        if (not isinstance(variants,list) or variants[:2]!=['U0','U123'] or len(variants) not in (2,3)
                or variants!=expected_variants or len(set(variants))!=len(variants)
                or any(v not in (*STAGES['single_factors'],*STAGES['combinations']) for v in variants)):
            raise ValueError('Only two task anchors and at most one selected structure allowed')
    else:variants=STAGES[stage]
    if replication:
        if variants != ['U0','U123','U12']:
            raise ValueError('Replication must preserve the three selected task configurations')
    expected = {f'pi05-{v}{"T" if task_stage else ""}-seed-{s}' for v in variants for s in p['seeds']}
    if replication:
        expected={f'pi05-{v}-seed-2' for v in (*STAGES['single_factors'],*STAGES['combinations'])} | {f'pi05-{v}T-seed-2' for v in variants}
    if set(p['cells']) != expected:
        raise ValueError('Exactly four variants and two seeds required')
    for name, cell in p['cells'].items():
        has_task=task_stage or (replication and name in {f'pi05-{v}T-seed-2' for v in variants})
        if (name != f"pi05-{cell['core_spec']['variant']}{'T' if has_task else ''}-seed-{cell['seed']}"
                or cell['family'] != 'pi05' or cell['prefix'] != 10
                or cell['score_contract'] != contract('pi05', 10)
                or cell['native_policy'] is not None or cell.get('feature_producer') is not None
                or cell['core_spec'] != core_spec(cell['core_spec']['variant'])
                or cell['observation_scale'] != 0.16535300100378683):
            raise ValueError('Changed native representation, reward observation scale, or core')
        if has_task:
            binding=validate_binding(cell['task_condition'])
            if (binding!=p['task_condition'] or binding['spec']['family']!='pi05'
                    or set(cell['model_hashes'].values())!={binding['spec']['model_sha256']}):
                raise ValueError('Task encoder must use the same frozen operation model')
        elif cell.get('task_condition') is not None:
            raise ValueError('Task input requires its separately approved stage')
        schedules = cell['evaluation_schedules']
        if (len(schedules) != 1 or any(v != dict(purpose='development', version=16, method='ppo')
                                      for v in schedules.values())):
            raise ValueError('Only the fixed 512-episode development endpoint is approved')


def validate_evaluation(c, cell, row, state):
    expected = cell['evaluation_schedules'].get(c['schedule'])
    if (expected != dict(purpose='development', version=c['evaluation_version'], method='ppo')
            or c.get('stay', False) or state['split'] != 'validation'
            or row['query_cap'] != cell.get('mechanism',{}).get('query_cap',8) or row['execution_repeats'] != [0, 1]
            or row['settle_steps'] != state['initialization']['settle_steps']):
        raise ValueError('Core pilot evaluates its fixed paired development panel only')
