"""Frozen three-model LIBERO scope and development-only checkpoint selection.

Legacy protocols remain immutable. This version uses one U12 core after explicit
real-camera adapters; final labels never enter checkpoint selection.
"""
from collections import defaultdict
from fractions import Fraction

from .artifacts import read
from .search_core import core_spec
from .score_prefix import contract
from ..execution.manifest import digest, file_digest

SCHEMA = 'UNIFIED_LIBERO_MAINLINE_V1'
VERSIONS = (16, 32, 48, 64, 80, 96)
SELECTION = 'development_task_macro_then_fewer_queries_then_earlier_common_three_seeds'


def validate_protocol(p):
    if (p['schema'] != SCHEMA or p.get('design_stage') != 'unified_mainline'
            or p['updates'] != 96 or p['episodes_per_update'] != 32
            or p['seeds'] != [0, 1, 2] or p['query_cap'] != 8 or p['query_cost'] != .01
            or p['maximum_model_calls'] != 4600000 or p['selection_rule'] != SELECTION
            or p['retained_checkpoint_updates'] != list(VERSIONS)):
        raise ValueError('Frozen U12 three-model mainline scope differs')
    expected = {f'{f}-U12-seed-{s}' for f in ('pi05', 'groot_n15', 'rldx') for s in p['seeds']}
    if set(p['cells']) != expected:
        raise ValueError('Exactly three models and three seeds required')
    for name, c in p['cells'].items():
        family = c['family']; prefix = 10 if family == 'pi05' else 4
        if (name != f"{family}-U12-seed-{c['seed']}" or c['prefix'] != prefix
                or c['score_contract'] != contract(family, prefix)
                or c['core_spec'] != core_spec('U12') or c.get('task_condition') is not None
                or c.get('mechanism') is not None or c['observation_scale'] != .16535300100378683
                or c.get('evaluation_only', False) != (family == 'pi05')):
            raise ValueError('Shared core, observation, score or training scope differs')
        if family == 'pi05':
            if c['native_policy'] is not None or c.get('feature_producer') is not None or c['fixed_version'] != 64:
                raise ValueError('Reuse the exact compatible pi05 U12 checkpoint')
        else:
            from ..selectors.search_policy_factory import validate_policy_spec
            spec = c['native_policy']; streams = validate_policy_spec(spec)
            if (spec['schema'] != 'UNIFIED_NATIVE_CONTINUOUS_POLICY_V1' or spec['core'] != c['core_spec']
                    or spec.get('task_spec') is not None or spec['layout'] != c['feature_producer']['layout']
                    or spec['producer_sha256'] != digest(c['feature_producer'])
                    or spec['projection_groups'] != ['visual', 'visual']
                    or [s.camera_ids for s in streams] != [('external',), ('wrist',)]
                    or [s.camera_roles for s in streams] != [('movable',), ('context',)]
                    or any(s.fusion_scope != 'per_view' for s in streams)):
                raise ValueError('Separate native real-camera visual inputs with shared projection required')
        schedules = c['evaluation_schedules']
        dev = [v['version'] for v in schedules.values() if v['purpose'] == 'development']
        if sorted(dev) != ([] if family == 'pi05' else list(VERSIONS)):
            raise ValueError('Complete predeclared development learning curve required')
        if not any(v['purpose'] == 'final' for v in schedules.values()):
            raise ValueError('Final population must be frozen before training')
        for v in schedules.values():
            if v['method'] != 'ppo' or v['purpose'] not in ('development', 'final'):
                raise ValueError('Baselines are audited reuse, not an implicit new matrix')
            if v['purpose'] == 'final' and v['version'] != (64 if family == 'pi05' else 'selected'):
                raise ValueError('Final checkpoint selection rule differs')


def select_checkpoint(rows):
    """Require all 6 versions x 3 seeds x 64 paired development outcomes."""
    groups = defaultdict(list); seen = set()
    for r in rows:
        key = (r['version'], r['seed'], r['state_key'], r['start'], r['repeat'])
        if (key in seen or r['split'] != 'validation' or r['version'] not in VERSIONS
                or r['seed'] not in (0, 1, 2) or type(r['success']) is not bool
                or type(r['queries']) is not int or not 1 <= r['queries'] <= 8):
            raise ValueError('Duplicate, nondevelopment or invalid checkpoint outcome')
        seen.add(key); groups[r['version']].append(r)
    reference = None; table = []; keys = {}
    for version in VERSIONS:
        panel = groups[version]
        if any(sum(r['seed'] == s for r in panel) != 64 for s in (0, 1, 2)):
            raise ValueError('All seeds and complete paired outcomes required')
        identities = {(r['seed'], r['suite'], r['task'], r['state_key'], r['start'], r['repeat']) for r in panel}
        if reference is not None and identities != reference:
            raise ValueError('Checkpoint evaluation populations differ')
        reference = identities
        tasks = defaultdict(list)
        for r in panel: tasks[(r['seed'], r['suite'], r['task'])].append(r)
        if len(tasks) != 24 or any(len(v) != 8 for v in tasks.values()):
            raise ValueError('Fixed eight-task balanced development panel required')
        success = sum((Fraction(sum(r['success'] for r in v), len(v)) for v in tasks.values()), Fraction()) / len(tasks)
        queries = Fraction(sum(r['queries'] for r in panel), len(panel))
        keys[version] = (-success, queries, version)
        table.append(dict(version=version, episodes=32*version, task_macro_success=float(success), mean_queries=float(queries)))
    return dict(rule=SELECTION, version=min(keys, key=keys.get), seeds=[0, 1, 2], curve=table, final_labels_used=False)


def validate_evaluation(c, cell, row, state):
    meta = cell['evaluation_schedules'].get(c['schedule'])
    if meta is None or c.get('stay', False):
        raise ValueError('Unplanned evaluation or baseline replay')
    version = meta['version']
    if version == 'selected':
        ref = c['selection']
        if c['input_hashes'].get(ref) != file_digest(ref):
            raise ValueError('Unbound model-wide checkpoint selection')
        selection = read(ref)
        if (selection['family'] != cell['family'] or selection['protocol_sha256'] != file_digest(c['protocol'])
                or selection['rule'] != SELECTION or selection['seeds'] != [0, 1, 2]
                or selection['final_labels_used'] is not False or selection['version'] not in VERSIONS):
            raise ValueError('Wrong model/protocol/development checkpoint selection')
        version = selection['version']
    if c['evaluation_version'] != version:
        raise ValueError('Checkpoint differs from the frozen common selection')
    expected = ('validation' if meta['purpose'] == 'development' else
                {'seen_new_initials': 'train', 'validation': 'validation', 'unseen_test': 'test_unseen_task'}.get(row.get('split')))
    if (expected is None or state['split'] != expected or row['query_cap'] != 8
            or row['execution_repeats'] != [0, 1] or row['settle_steps'] != state['initialization']['settle_steps']):
        raise ValueError('Changed task split, query cap, paired noise or initialization')
