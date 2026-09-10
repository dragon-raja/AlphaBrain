#!/usr/bin/env python3
"""Apply the same frozen seven rules to the strictly matched eight-state pair.

CPU only. This is a descriptive training-condition comparison, not new rule
fitting or a source-generalization claim. Existing output is never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from .analyze_view_metric_rules_v1 import RULES, analyze, source_bootstrap
    from .compare_matched_view_landscapes_v1 import (
        GEOMETRY_FIELDS, REPO, SELECTION_SHA256, catalog_bank, load_report,
        match_reports, read_json, require, sha256,
    )
    from .build_view_landscape_report_v1 import finite_json, write_csv, write_json
except ImportError:
    from analyze_view_metric_rules_v1 import RULES, analyze, source_bootstrap
    from compare_matched_view_landscapes_v1 import (
        GEOMETRY_FIELDS, REPO, SELECTION_SHA256, catalog_bank, load_report,
        match_reports, read_json, require, sha256,
    )
    from build_view_landscape_report_v1 import finite_json, write_csv, write_json


def require_new_output(path: Path) -> None:
    require(not path.exists(), 'Output already exists; this immutable wrapper refuses overwrite or implicit resume')


def pair_rule_results(canonical: dict, broad: dict) -> tuple[dict, list[dict]]:
    maps = []
    for result in (canonical, broad):
        index = {}
        for row in result['state_rule_metrics']:
            key = row['pair_key'], row['rule']
            require(key not in index, 'Duplicate state/rule result')
            index[key] = row
        require(len(index) == 8 * len(RULES), 'Expected all seven rules on exactly eight states')
        maps.append(index)
    require(set(maps[0]) == set(maps[1]), 'Model result cells do not align')
    rows = []
    for key in sorted(maps[0]):
        c, b = maps[0][key], maps[1][key]
        require(all(c[field] == b[field] for field in ('pair_key', 'task_id', 'source_group', 'split', 'rule')), 'State/source identity changed after analysis')
        row = {field: c[field] for field in ('pair_key', 'task_id', 'source_group', 'split', 'rule')}
        for label, values in (('canonical_model', c), ('broad_model', b)):
            for field in ('success', 'gain_vs_canonical_pp', 'gain_vs_uniform_pp'):
                row[label + '_' + field] = values[field]
        row.update(delta_success_pp=100 * (b['success'] - c['success']),
                   gain_interaction_vs_canonical_pp=b['gain_vs_canonical_pp'] - c['gain_vs_canonical_pp'],
                   gain_interaction_vs_uniform_pp=b['gain_vs_uniform_pp'] - c['gain_vs_uniform_pp'])
        rows.append(row)
    unique = {row['pair_key']: row for row in rows}
    require(len(unique) == 8 and len({row['task_id'] for row in unique.values()}) == 8, 'Expected exactly one source state per known task')
    summary = {'schema': 'dsol_matched_view_metric_rule_delta_summary_v1',
        'status': 'PASS_DESCRIPTIVE_MATCHED_RULES', 'state_count': 8, 'task_count': 8,
        'source_states_per_task': 1, 'rule_order': list(RULES),
        'difference_direction': 'Broad-trained minus canonical-trained model',
        'rules': {rule: {field: source_bootstrap([row for row in rows if row['rule']==rule], field)
                          for field in ('delta_success_pp', 'gain_interaction_vs_canonical_pp', 'gain_interaction_vs_uniform_pp')}
                  for rule in RULES},
        'source_generalization_confidence_interval': None,
        'statistical_scope': 'Within-state paired differences followed by equal weighting of the eight fixed known tasks. No within-task source replication: no source bootstrap CI.',
        'new_confirmation': False,
        'limitations': [
            'Same seven rules and historical tie handling, not fitted separately to each model outcome.',
            'Rules may select different candidates for different models; the paired effect compares model-specific use of the same rule.',
            'Training interaction subtracts each model\'s own reference gain, not different pooled cohorts.',
            'Visibility is privileged task-entity pixel occupancy; candidate images are cached offline inputs.',
            'One training seed and one source state per task cannot establish cross-seed or source generalization.',
            'The full-64-state Broad summary must not replace this selected-eight-state Broad counterpart.',
        ]}
    return summary, rows


def verify_catalog(report: dict, model_kind: str) -> None:
    catalog_path = REPO / 'configs/dsol_paper1/libero_view_catalog_v2_m1.json'
    rules_path = REPO / 'configs/dsol_paper1/libero_view_catalog_v2_m1_rules.json'
    require(report['provenance']['catalog_sha256'] == sha256(catalog_path), 'Catalog content identity mismatch')
    require(report['provenance']['catalog_rules_sha256'] == sha256(rules_path), 'Catalog rule identity mismatch')
    expected, _ = catalog_bank(read_json(catalog_path), read_json(rules_path), 'canonical' if model_kind == 'canonical' else 'broad64')
    for pose, frozen in zip(report['poses'], expected):
        require(all(pose[field] == frozen[field] for field in (*GEOMETRY_FIELDS, 'is_model_post_training_support', 'distance_model_support_parameter_normalized')), 'Candidate geometry/model support differs from frozen catalog')


def run(args: argparse.Namespace) -> dict:
    require_new_output(args.output_dir)
    require(sha256(args.selection_manifest) == SELECTION_SHA256, 'Not the frozen outcome-blind eight-state selection')
    selection = read_json(args.selection_manifest)
    canonical, broad = load_report(args.canonical_root, 'canonical'), load_report(args.broad_root, 'broad')
    require(canonical['provenance'].get('selection_manifest_sha256') == SELECTION_SHA256, 'Canonical report selection binding mismatch')
    verify_catalog(canonical, 'canonical'); verify_catalog(broad, 'broad')
    canonical, broad, selected = match_reports(canonical, broad, selection)
    # Explicitly verify selected physical/render asset hashes via their bound
    # receipts/images; no simulator is started and no candidate is rerendered.
    asset_hashes = {}
    for state in selected:
        for field in ('policy_inputs', 'render_receipt', 'visibility_scan'):
            path = Path(state['static_assets'][field])
            digest = sha256(path)
            require(digest == state['static_assets'][field + '_sha256'], 'Selected asset content changed')
            asset_hashes[str(path)] = digest
    code_files = [Path(__file__), REPO / 'scripts/dsol_paper1/analyze_view_metric_rules_v1.py',
                  REPO / 'scripts/dsol_paper1/compare_matched_view_landscapes_v1.py',
                  REPO / 'scripts/dsol_paper1/build_view_landscape_report_v1.py']
    manifest = {'schema': 'dsol_matched_view_metric_rule_manifest_v1', 'status': 'FROZEN_BEFORE_MATCHED_RULE_AGGREGATION',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'rule_order': list(RULES),
        'selection': {'path': str(args.selection_manifest.resolve()), 'sha256': SELECTION_SHA256},
        'input_identity': {'canonical': canonical['input_identity'], 'broad': broad['input_identity']},
        'analysis_code_sha256': {str(path.resolve()): sha256(path) for path in code_files},
        'selected_asset_sha256': asset_hashes, 'state_count': 8, 'candidate_count': 97,
        'score_noise_members': 8, 'rollout_noise_repeats': 32,
        'checkpoint_sha256': {label: report['provenance']['checkpoint_sha256'] for label, report in (('canonical',canonical),('broad',broad))},
        'source_ci_policy': 'No within-task replicated source states; suppress source CI.',
        'rule_definition': 'Reuse analyze_view_metric_rules_v1.analyze unchanged, including historical candidate-ID tie handling and fixed top10.',
        'gpu_used': False, 'new_rollouts': 0, 'training_performed': False}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / 'manifest.json', manifest)
    c_result, b_result = analyze(canonical), analyze(broad)
    paired_summary, paired_rows = pair_rule_results(c_result, b_result)
    for label, result in (('canonical', c_result), ('broad', b_result)):
        write_json(args.output_dir / (label + '_summary.json'), result['summary'])
        write_csv(args.output_dir / (label + '_state_rule_metrics.csv'), finite_json(result['state_rule_metrics']))
        write_csv(args.output_dir / (label + '_state_association_stability.csv'), finite_json(result['state_association_stability']))
        write_json(args.output_dir / (label + '_selected_candidates.json'), {'states_and_rules': result['selected_candidates']})
    write_json(args.output_dir / 'paired_delta_summary.json', paired_summary)
    write_csv(args.output_dir / 'paired_state_rule_metrics.csv', finite_json(paired_rows))
    completion = {'status': 'PASS_COMPLETE', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'state_count': 8, 'rule_count': 7, 'source_bootstrap_ci_reported': False,
        'outputs': {path.name: sha256(path) for path in args.output_dir.iterdir() if path.is_file()},
        'gpu_used': False, 'new_rollouts': 0}
    write_json(args.output_dir / 'completion.json', completion)
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--broad-root', type=Path, required=True)
    parser.add_argument('--canonical-root', type=Path, required=True)
    parser.add_argument('--selection-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run(args)
    print(f'PASS_COMPLETE {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
