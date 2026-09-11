#!/usr/bin/env python3
"""Describe completed speed probes without weakening their frozen acceptance gate."""
from __future__ import annotations

# Repository-local CLI bootstrap: path setup only.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_root = _LayoutPath(__file__).resolve().parents[3]
for _layout_path in (_layout_root, _layout_root / 'scripts/dsol_paper1', _layout_root / 'scripts/vla_shared'):
    if str(_layout_path) not in _layout_sys.path:
        _layout_sys.path.insert(0, str(_layout_path))


from collections import Counter
import json
from pathlib import Path

import scripts.dsol_paper1.operations.controllers.run_full64_view_landscape_v2 as control
from scripts.dsol_paper1.protocols.build_matched_view_landscape_protocol_v1 import source_identity, write_new_json


def load_rows(paths):
    rows = {}
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (row['pair_key'], row['selected_candidate_id'], row['policy_repeat_id'])
            control.require(key not in rows, 'Duplicate physical/view/noise key')
            rows[key] = row
    return rows


def compare(reference, candidate):
    control.require(set(reference) == set(candidate), 'Comparison keys differ')
    details = []
    for key, baseline in reference.items():
        tested = candidate[key]
        for field in ('environment_seed', 'sensor_control', 'replan_steps', 'wait_steps',
                      'noise_bank_manifest_sha256', 'pose'):
            control.require(baseline[field] == tested[field], 'Scientific control differs: ' + field)
        common = list(zip(baseline['policy_calls'], tested['policy_calls']))
        first = next((i for i, (a, b) in enumerate(common)
                      if a['action_chunk_sha256'] != b['action_chunk_sha256']), None)
        details.append({
            'pair_key': key[0], 'candidate': key[1], 'repeat': key[2], 'task_id': baseline['task_id'],
            'initial_physics_equal': baseline['initial_metrics']['physics_state_sha256'] == tested['initial_metrics']['physics_state_sha256'],
            'initial_metrics_equal': baseline['initial_metrics'] == tested['initial_metrics'],
            'common_replan_noise_equal': all(all(a[f] == b[f] for f in ('replan_index', 'noise_seed', 'noise_sha256')) for a, b in common),
            'first_different_action_call': first,
            'full_action_path_equal': first is None and len(baseline['policy_calls']) == len(tested['policy_calls']),
            'policy_call_count_equal': len(baseline['policy_calls']) == len(tested['policy_calls']),
            'success_equal': baseline['success'] == tested['success'],
            'completion_steps_equal': baseline['completion_steps'] == tested['completion_steps'],
            'baseline_success': baseline['success'], 'tested_success': tested['success'],
        })
    fields = ('initial_physics_equal', 'initial_metrics_equal', 'common_replan_noise_equal',
              'full_action_path_equal', 'policy_call_count_equal', 'success_equal', 'completion_steps_equal')
    return {
        'episode_count': len(details),
        'mismatch_counts': {field: sum(not row[field] for row in details) for field in fields},
        'path_mismatch_tasks': dict(Counter(row['task_id'] for row in details if not row['full_action_path_equal'])),
        'details': details,
    }


def main():
    release = control.preflight()
    choice_path = control.ROOT / 'performance-choice.json'
    choice = control.read_json(choice_path)
    control.require(choice['status'] == 'PASS_PERFORMANCE_SELECTION', 'Speed probes not complete')
    data = {}
    inputs = [source_identity(choice_path)]
    for mode in release['performance_modes']:
        output = control.ROOT / 'benchmarks' / mode['label']
        control.require(control.completed(output, release, release['benchmark_protocol'], mode) is not None,
                        'Benchmark receipt missing')
        paths = sorted(output.glob('episodes-shard-*.jsonl'))
        data[mode['label']] = load_rows(paths)
        inputs.extend(source_identity(path) for path in paths)
    baseline = data['baseline32']
    old_paths = []
    for directory in sorted((control.PARENT / 'closed-loop/canonical/O').iterdir()):
        receipt_path = directory / 'segment-receipt.json'
        if not receipt_path.exists():
            continue  # Never read or hash the resumed dispatcher's active ledgers.
        receipt = control.read_json(receipt_path)
        control.require(receipt['status'] == 'PASS_COMPLETE', 'Historical receipt incomplete')
        for path, digest in receipt['ledger_sha256'].items():
            control.require(control.sha256_file(Path(path)) == digest, 'Historical ledger changed')
            old_paths.append(Path(path))
        inputs.append(source_identity(receipt_path))
    historical = load_rows(old_paths)
    control.require(set(baseline) <= set(historical), 'Missing same-scope historical episodes')
    inputs.extend(source_identity(path) for path in old_paths)
    historical = {key: historical[key] for key in baseline}
    comparisons = {label: compare(baseline, value) for label, value in data.items() if label != 'baseline32'}
    comparisons['historical32_vs_probe32'] = compare(historical, baseline)
    output = control.ROOT / 'early-speed-gate-v1/equivalence-diagnostics-v1.json'
    write_new_json(output, {
        'status': 'PASS_DESCRIPTIVE_EQUIVALENCE_AUDIT', 'utc': control.stamp(),
        'script': source_identity(Path(__file__)), 'inputs': inputs,
        'acceptance_rule_changed': False, 'selected_mode': choice['selected_mode'],
        'comparisons': comparisons,
        'limits': [
            'Historical32 and probe32 have the same physical/view/noise keys but different protocol packing and worker assignment; this is not an exact replay of the entire execution schedule.',
            'Action hashes expose nonidentity but not numerical magnitude; subsequent observation/state hashes were not logged, so the origin of divergence is not identified.',
            'Initial image statistics equality is not a full RGB-byte equivalence test.',
            'Success agreement on 192 probes is not a proof of distributional equivalence or determinism.',
            'Explicit common-replan noise equality is distinct from bitwise reproducibility of the entire closed-loop path.',
        ],
    })
    print(json.dumps({'output': str(output), 'comparisons': {
        label: {key: value for key, value in result.items() if key != 'details'}
        for label, result in comparisons.items()}}, indent=2))


if __name__ == '__main__':
    main()
