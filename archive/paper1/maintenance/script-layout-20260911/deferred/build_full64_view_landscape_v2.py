#!/usr/bin/env python3
"""Freeze the authorized 56-state extension without rewriting the first eight."""
from __future__ import annotations

# Repository-local CLI bootstrap: path setup only.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[3]
for _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):
    if str(_layout_path) not in _layout_sys.path:
        _layout_sys.path.insert(0, str(_layout_path))


import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

from scripts.dsol_paper1.build_matched_view_landscape_protocol_v1 import DEFAULT_V2_ROOT, DEFAULT_OUTPUT_ROOT, EXPECTED_CANDIDATES, read_json, write_new_json, source_identity, object_sha256
from scripts.dsol_paper1.evaluate_dsol_libero_hdf5_views import protocol_spec_at, protocol_spec_count
from scripts.dsol_paper1.run_matched_view_accel_v1 import validate_assets, code_identities

REPO = Path(__file__).resolve().parents[3]
PARENT = DEFAULT_OUTPUT_ROOT
ROOT = PARENT / 'full64-extension-v2'


def require(value, message):
    if not value:
        raise ValueError(message)


def partition_states(population, first):
    states = population['population']['development']['states'] + population['population']['test']['states']
    require(len(states) == len({s['pair_key'] for s in states}) == len({s['source_group'] for s in states}) == 64, 'Expected 64 unique source states')
    require(set(Counter(s['task_id'] for s in states).values()) == {8}, 'Expected eight states per task')
    require(Counter(s['split'] for s in states) == {'development': 48, 'test': 16}, 'Historical splits changed')
    by_key = {s['pair_key']: s for s in states}
    require(len(first) == len({s['pair_key'] for s in first}) == 8, 'Expected frozen first eight')
    require(all(by_key.get(s['pair_key']) == s for s in first), 'First-eight state identity changed')
    excluded = {s['pair_key'] for s in first}
    remaining = [s for s in states if s['pair_key'] not in excluded]
    groups = []
    for role, count in (('development', 5), ('test', 2)):
        tasks = sorted({s['task_id'] for s in states})
        by_task = {t: sorted((s for s in remaining if s['task_id'] == t and s['split'] == role), key=lambda s: s['pair_key']) for t in tasks}
        require(all(len(v) == count for v in by_task.values()), 'Extension task/source partition differs')
        for j in range(count):
            groups.append((role, j, [by_task[t][j] for t in tasks]))
    require(len(remaining) == 56 and len(groups) == 7, 'Wrong extension size')
    return states, remaining, groups


def protocol_for(template, blocks, repeats, label, selection, *, benchmark=False):
    result = deepcopy(template)
    result.update(status='PASS_FROZEN_FULL64_MATCHED_EXTENSION',
        phase='matched_view_landscape_full64', wave_id=label,
        episode_identity_prefix=('view-landscape-v2::benchmark' if benchmark else 'view-landscape-v2::O::' + template['role']),
        diagnostic_role='full64_static_matched_diagnostic', state_blocks=deepcopy(blocks),
        policy_repeat_ids=list(repeats), state_count=len(blocks),
        candidate_count_per_state=len(blocks[0]['candidates']),
        episode_count=len(blocks)*len(blocks[0]['candidates'])*len(repeats),
        selection=str(selection), selection_sha256=None,
        claim_scope='Authorized complete historical 64-state matched comparison; not a new unseen-task confirmation',
        camera_motion_within_rollout=False, outcomes_used_for_state_inclusion=False)
    return result


def build(root=ROOT):
    require(not root.exists(), 'Extension output exists; refusing overwrite')
    old_release_path = REPO / 'configs/dsol_paper1/matched_view_landscape_release_v1.json'
    old_release = read_json(old_release_path)
    first_path = PARENT / 'protocols/selection.json'
    first = read_json(first_path)
    population_path = DEFAULT_V2_ROOT / 'population/population-v2.json'
    population = read_json(population_path)
    states, remaining, groups = partition_states(population, first['states'])
    sources = [source_identity(population_path), source_identity(first_path), source_identity(old_release_path)]
    templates = {}
    for role in ('development', 'test'):
        waves = []
        for w in range(8):
            path = DEFAULT_V2_ROOT / f'protocols/dense-O-{role}-wave-{w:02d}.json'
            wave = read_json(path)
            require(wave['noise_bank_id'] == 'O' and wave['role'] == role and wave['policy_repeat_ids'] == list(range(w*4,w*4+4)), 'Historical O wave contract changed')
            require(wave['population_sha256'] == sources[0]['sha256'], 'Population hash mismatch')
            require(wave['catalog_sha256'] == old_release['catalog']['sha256'], 'Candidate catalog hash mismatch')
            waves.append(wave); sources.append(source_identity(path))
        require(all(w['state_blocks'] == waves[0]['state_blocks'] for w in waves), 'Historical state/candidate blocks vary by repeat wave')
        blocks = {b['state']['pair_key']: b for b in waves[0]['state_blocks']}
        for s in [s for s in states if s['split'] == role]:
            b = blocks[s['pair_key']]
            require(b['state'] == s, 'Historical protocol/source state mismatch')
            require(tuple(c['selected_candidate_id'] for c in b['candidates']) == EXPECTED_CANDIDATES, 'Not all 97 candidates')
        templates[role] = waves[0], blocks
    selection_path = root / 'protocols/selection-full64.json'
    selection = dict(schema='dsol_full64_view_landscape_selection_v2', status='FROZEN_COMPLETE_HISTORICAL_POPULATION',
        states=states, selected_state_keys=[s['pair_key'] for s in states], states_per_task=8,
        outcomes_or_view_metrics_used_in_selection=False, historical_splits_preserved=True,
        first_eight_reused=[s['pair_key'] for s in first['states']], new_state_keys=[s['pair_key'] for s in remaining],
        source_population=source_identity(population_path))
    selection_sha = object_sha256(selection)
    pending = {}
    route = []
    seen = set()
    # Give every added state its first four repeats before proceeding to repeat 4.
    for w in range(8):
        for role, j, group in groups:
            template, by_key = templates[role]
            label = f'{role}-sources-{j:02d}-wave-{w:02d}'
            p = protocol_for(template, [by_key[s['pair_key']] for s in group], range(w*4,w*4+4), label, selection_path)
            p['selection_sha256'] = selection_sha
            for i in range(protocol_spec_count(p)):
                spec = protocol_spec_at(p, i)
                cell = (spec['pair_key'], spec['selected_candidate_id'], spec['policy_repeat_id'])
                require(cell not in seen, 'Duplicate extension cell')
                seen.add(cell)
            filename = label + '.json'; pending[filename] = p
            route.append(dict(label=label, path=str(root/'protocols'/filename), sha256=object_sha256(p), episode_count=protocol_spec_count(p)))
    require(len(seen) == 56*97*32, 'Extension matrix is not complete')
    require(not ({s['pair_key'] for s in first['states']} & {key[0] for key in seen}), 'Extension overlaps first eight')
    template, blocks = templates['development']
    benchmark_blocks = [deepcopy(blocks[s['pair_key']]) for s in first['states']]
    for b in benchmark_blocks:
        b['candidates'] = [c for c in b['candidates'] if c['selected_candidate_id'] in {'canonical','broad_train_000','broad_heldout_000'}]
    bench = protocol_for(template, benchmark_blocks, range(8), 'performance-equivalence', selection_path, benchmark=True)
    bench['selection_sha256'] = selection_sha
    pending['performance-equivalence.json'] = bench
    # Reuse the unchanged scorer/runtime contract. Only the explicitly authorized
    # source population and output destination differ from the v1 prepare gate.
    accel = deepcopy(read_json(PARENT/'canonical-accel/manifest.json'))
    require(accel['code'] == code_identities(), 'Scorer code differs from frozen first-eight scorer')
    accel_output = PARENT / 'canonical-accel-full64-extension-v2'
    require(not accel_output.exists(), 'Extension Accel output exists')
    accel.update(scope='Authorized 56 additional historical states; exact legacy scoring seeds/images; no training',
        selection={'path':str(selection_path), 'sha256':selection_sha}, states=sorted(remaining,key=lambda s:s['asset_source_pair_key']),
        state_count=56, historical_broad_rankings=[validate_assets(s) for s in remaining], output_root=str(accel_output))
    scripts = ['build_full64_view_landscape_v2.py','run_full64_view_landscape_v2.py',
               'finalize_full64_view_landscape_v2.py','analyze_view_oracle_comparison_v2.py']
    code = [source_identity(REPO/'scripts/dsol_paper1'/name) for name in scripts]
    code += [source_identity(REPO/'scripts/dsol_paper1'/name) for name in (
        'build_view_landscape_report_v1.py','analyze_view_metric_rules_v1.py',
        'compare_matched_view_landscapes_v1.py','build_view_metric_comparison_brief_v1.py')]
    code += [source_identity(REPO/name, expected) for name,expected in old_release['critical_files'].items()]
    manifest = dict(schema='dsol_full64_view_landscape_release_v2', status='FROZEN_USER_AUTHORIZED_EXTENSION',
        root=str(root), predecessor_root=str(PARENT), old_release=source_identity(old_release_path),
        old_release_content=old_release, source_identities=sources, critical_files=code,
        selection={'path':str(selection_path),'sha256':selection_sha},
        accel_manifest={'path':str(accel_output/'manifest.json'),'sha256':object_sha256(accel)},
        benchmark_protocol={'path':str(root/'protocols/performance-equivalence.json'),'sha256':object_sha256(bench),'episode_count':192},
        route=route, total_state_count=64, new_state_count=56, reused_state_count=8,
        total_dense_episodes=198656, new_dense_episodes=173824, reused_dense_episodes=24832,
        benchmark_extra_episodes=576, total_new_gpu_episodes=174400,
        performance_modes=[{'label':'baseline32','copies':2,'workers':32}, {'label':'copies3-workers48','copies':3,'workers':48}, {'label':'copies3-workers64','copies':3,'workers':64}],
        performance_selection='Fastest passing mode at least 5% faster than baseline; exact action-chunk/physics/noise/output equivalence required, otherwise retain baseline. Benchmark rows never enter scientific matrix.',
        base_port=27200, max_execution_seconds=518400, max_queue_seconds=129600,
        automatic_retry=False, training=False, active_camera=False, robocasa=False,
        unchanged_scientific_controls={'replan_steps':5,'wait_steps':0,'flow_denoising_steps':10,'score_noise_members':8,'rollout_repeat_count':32,'candidate_count':97},
        oracle_comparison=['empirical_statewise_O32_max_optimistic','statewise_O16_crossfit_not_exact_oracle'],
        statistical_scope='Complete same 64 historical source states. Keep development/test labels; not fresh task/source confirmation, not 64 tasks.')
    (root/'protocols').mkdir(parents=True)
    write_new_json(selection_path, selection)
    for filename,value in pending.items(): write_new_json(root/'protocols'/filename,value)
    accel_output.mkdir()
    write_new_json(accel_output/'manifest.json',accel)
    write_new_json(root/'release.json',manifest)
    return manifest


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    value=build()
    print({k:value[k] for k in ('status','root','new_state_count','new_dense_episodes','total_dense_episodes','benchmark_extra_episodes')})
