#!/usr/bin/env python3
"""Fixed-rule CPU diagnostics on verified view arrays; no training or policy calls.

Freezes the user-proposed seven rules before aggregating their success outcomes.
Visibility is privileged task-entity pixel occupancy, not information gain.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from .compare_matched_view_landscapes_v1 import NEW_ROOT, load_report
    from .build_view_landscape_report_v1 import finite_json, ranks, require, sha256, spearman, write_csv, write_json
except ImportError:
    from compare_matched_view_landscapes_v1 import NEW_ROOT, load_report
    from build_view_landscape_report_v1 import finite_json, ranks, require, sha256, spearman, write_csv, write_json


RULES = ('canonical', 'uniform97', 'min_accel', 'max_visibility', 'accel_top10_uniform',
         'accel_top10_max_visibility', 'visibility_top10_min_accel')
DIFFICULTIES = ('hard_floor', 'middle', 'easy')
ASSOCIATIONS = ('rho_negative_accel_success', 'rho_visibility_success')
STABILITY = ('top10_pairwise_jaccard', 'top10_consensus_overlap', 'top10_all_noise_intersection_count',
             'top10_all_noise_identical', 'top10_member_pair_rank_spearman')
SUCCESS_FIELDS = ('success', 'gain_vs_canonical_pp', 'gain_vs_uniform_pp')
BOOTSTRAP_SEED = 20260908
BOOTSTRAP_RESAMPLES = 10000


def fixed_rule_weights(mean_accel: np.ndarray, visibility: np.ndarray, candidate_ids: list[str]) -> tuple[dict, dict]:
    mean_accel, visibility = np.asarray(mean_accel), np.asarray(visibility)
    require(mean_accel.shape == visibility.shape == (97,), 'Rules require exactly 97 candidates')
    require(np.isfinite(mean_accel).all() and np.isfinite(visibility).all(), 'Nonfinite selection metric')
    require(len(candidate_ids) == len(set(candidate_ids)) == 97, 'Candidate IDs must be unique')
    # Preserve the historical Gate-A ranking/tie rules, not geometry-index ties.
    low = np.asarray(sorted(range(97), key=lambda j: (mean_accel[j], candidate_ids[j]))[:10])
    high = np.asarray(sorted(range(97), key=lambda j: (-visibility[j], candidate_ids[j]))[:10])
    selected = {'canonical': np.array([0]), 'uniform97': np.arange(97),
                'min_accel': np.array([int(low[0])]),
                'max_visibility': np.array([int(high[0])]),
                'accel_top10_uniform': low,
                'accel_top10_max_visibility': np.array([int(min(low, key=lambda j: (-visibility[j], mean_accel[j], candidate_ids[j])))]),
                'visibility_top10_min_accel': np.array([int(min(high, key=lambda j: (mean_accel[j], -visibility[j], candidate_ids[j])))] )}
    weights = {}
    for name, indices in selected.items():
        vector = np.zeros(97)
        vector[indices] = 1 / len(indices)
        weights[name] = vector
    return weights, selected


def difficulty_from_selection_half(outcomes: np.ndarray) -> str:
    require(outcomes.shape == (97, 16) and np.isin(outcomes, [0, 1]).all(), 'Difficulty needs 97 x 16 binary discovery outcomes')
    rate = float(outcomes.mean())
    return 'easy' if rate >= .9 else 'hard_floor' if rate <= .1 else 'middle'


def rule_outcomes(weights: dict, outcomes: np.ndarray) -> dict:
    require(outcomes.ndim == 2 and outcomes.shape[0] == 97 and np.isin(outcomes, [0, 1]).all(), 'Invalid outcome tensor')
    return {rule: vector @ outcomes for rule, vector in weights.items()}


def source_bootstrap(rows: list[dict], field: str) -> dict:
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = float(row[field])
        if np.isfinite(value):
            groups[row['task_id']][row['source_group']].append(value)
    tasks = sorted(groups)
    samples = [np.asarray([np.mean(groups[task][source]) for source in sorted(groups[task])]) for task in tasks]
    result = {'mean': float(np.mean([x.mean() for x in samples])) if samples else None,
        'ci95': [None, None], 'finite_states': sum(np.isfinite(float(row[field])) for row in rows),
        'finite_sources': sum(len(x) for x in samples), 'tasks': len(tasks),
        'sources_per_task': {task: len(groups[task]) for task in tasks},
        'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_resamples': BOOTSTRAP_RESAMPLES,
        'interval_scope': 'Descriptive task-equal, within-task source-cluster bootstrap; conditional on these tasks, fixed rules, cached score noise and realized O-bank measurements.',
        'ci_status': 'NOT_ESTIMABLE_INSUFFICIENT_WITHIN_TASK_SOURCES'}
    if not samples or any(len(values) < 2 for values in samples):
        return result
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.mean([values[rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))].mean(1) for values in samples], axis=0)
    result.update(ci95=np.quantile(draws, [.025, .975]).tolist(), ci_status='DESCRIPTIVE_SOURCE_BOOTSTRAP')
    return result


def top10_stability(accel: np.ndarray, candidate_ids: list[str]) -> dict:
    require(accel.shape == (97, 8) and np.isfinite(accel).all(), 'Need 97 x 8 score-noise members')
    require(len(candidate_ids) == len(set(candidate_ids)) == 97, 'Candidate IDs must be unique')
    sets = [set(sorted(range(97), key=lambda j: (accel[j, m], candidate_ids[j]))[:10]) for m in range(8)]
    means = accel.mean(1)
    consensus = set(sorted(range(97), key=lambda j: (means[j], candidate_ids[j]))[:10])
    overlaps, correlations = [], []
    for i in range(8):
        for j in range(i+1, 8):
            overlaps.append(len(sets[i] & sets[j]) / len(sets[i] | sets[j]))
            correlations.append(spearman(accel[:, i], accel[:, j]))
    finite_correlations = [value for value in correlations if np.isfinite(value)]
    return {'top10_pairwise_jaccard': float(np.mean(overlaps)),
            'top10_consensus_overlap': float(np.mean([len(consensus & member)/10 for member in sets])),
            'top10_all_noise_intersection_count': len(set.intersection(*sets)),
            'top10_all_noise_identical': int(all(member == sets[0] for member in sets)),
            'top10_member_pair_rank_spearman': float(np.mean(finite_correlations)) if finite_correlations else float('nan')}


def aggregate_rules(rows: list[dict], associations: list[dict], stability: list[dict] | None = None) -> dict:
    output = {'rules': {rule: {field: source_bootstrap([row for row in rows if row['rule'] == rule], field)
                              for field in SUCCESS_FIELDS} for rule in RULES},
              'associations': {field: source_bootstrap(associations, field) for field in ASSOCIATIONS}}
    if stability is not None:
        output['stability'] = {field: source_bootstrap(stability, field) for field in STABILITY}
    output['state_count'] = len({row['pair_key'] for row in associations})
    output['source_count'] = len({(row['task_id'], row['source_group']) for row in associations})
    output['task_count'] = len({row['task_id'] for row in associations})
    return output


def analyze(report: dict) -> dict:
    states, success, accel, visibility = (report[key] for key in ('states', 'success', 'accel', 'visibility'))
    n = len(states)
    require(success.shape == (n, 97, 32), 'Requires complete 32-repeat outcomes')
    require(accel.shape == (n, 97, 8), 'Requires eight score-noise members')
    require(np.isin(success, [0, 1]).all(), 'Missing/nonbinary success data')
    state_rules, state_diagnostics, fold_rows, headroom, selected_rows = [], [], [], [], []
    changed = 0
    for i, state in enumerate(states):
        identity = {key: state[key] for key in ('pair_key', 'task_id', 'source_group', 'split')}
        a, v, y = accel[i].mean(1), visibility[i], success[i]
        weights, selected = fixed_rule_weights(a, v, report['candidate_ids'])
        outcomes = rule_outcomes(weights, y)
        p = y.mean(1)
        diagnostics = {**identity, **top10_stability(accel[i], report['candidate_ids']),
            'rho_negative_accel_success': spearman(-a, p), 'rho_visibility_success': spearman(v, p),
            'uniform_view_success': float(p.mean()), 'canonical_view_success': float(p[0]),
            'constant_success_across_candidates': int(np.ptp(p) == 0)}
        state_diagnostics.append(diagnostics)
        top10 = selected['accel_top10_uniform']
        top_max, global_max = float(p[top10].max()), float(p.max())
        headroom.append({**identity, 'top10_empirical_max_success': top_max,
            'all97_empirical_max_success': global_max, 'top10_max_shortfall_pp': 100*(global_max-top_max),
            'top10_contains_any_empirical_maximizer': int(top_max == global_max),
            'optimistic_top10_headroom_over_uniform_pp': 100*(top_max-p.mean()),
            'optimistic_top10_headroom_over_canonical_pp': 100*(top_max-p[0])})
        for rule in RULES:
            values = outcomes[rule]
            dc, du = values-outcomes['canonical'], values-outcomes['uniform97']
            state_rules.append({**identity, 'rule': rule, 'success': float(values.mean()),
                'gain_vs_canonical_pp': 100*float(dc.mean()), 'gain_vs_uniform_pp': 100*float(du.mean()),
                'paired_noise_conditional_se_vs_canonical_pp': 100*float(dc.std(ddof=1))/np.sqrt(32),
                'paired_noise_conditional_se_vs_uniform_pp': 100*float(du.std(ddof=1))/np.sqrt(32)})
            selected_rows.append({**identity, 'rule': rule,
                'candidate_ids': [report['candidate_ids'][j] for j in selected[rule]],
                'probability_per_selected_candidate': 1/len(selected[rule])})
        strata = []
        for direction, discovery, evaluation in (('first16_to_last16', y[:, :16], y[:, 16:]),
                                                 ('last16_to_first16', y[:, 16:], y[:, :16])):
            stratum = difficulty_from_selection_half(discovery)
            strata.append(stratum)
            heldout = rule_outcomes(weights, evaluation)
            common = {**identity, 'direction': direction, 'difficulty': stratum,
                'discovery_uniform_success': float(discovery.mean()),
                'evaluation_uniform_success': float(evaluation.mean()),
                'rho_negative_accel_success': spearman(-a, evaluation.mean(1)),
                'rho_visibility_success': spearman(v, evaluation.mean(1))}
            for rule in RULES:
                fold_rows.append({**common, 'rule': rule, 'success': float(heldout[rule].mean()),
                    'gain_vs_canonical_pp': 100*float((heldout[rule]-heldout['canonical']).mean()),
                    'gain_vs_uniform_pp': 100*float((heldout[rule]-heldout['uniform97']).mean())})
        changed += int(strata[0] != strata[1])
    # A state can move between strata across discovery halves. Within each
    # state/stratum average its available held-out directions before bootstrap.
    grouped = defaultdict(list)
    for row in fold_rows:
        grouped[(row['pair_key'], row['difficulty'], row['rule'])].append(row)
    difficulty_rows = []
    for _, rows in sorted(grouped.items()):
        row = {key: rows[0][key] for key in ('pair_key', 'task_id', 'source_group', 'split', 'difficulty', 'rule')}
        row['heldout_direction_count'] = len(rows)
        for field in (*SUCCESS_FIELDS, *ASSOCIATIONS, 'discovery_uniform_success', 'evaluation_uniform_success'):
            values = [value[field] for value in rows if np.isfinite(value[field])]
            row[field] = float(np.mean(values)) if values else float('nan')
        difficulty_rows.append(row)
    source_groups = defaultdict(list)
    for row in state_rules:
        source_groups[(row['task_id'], row['source_group'], row['rule'])].append(row)
    source_rows = []
    for (task, source, rule), rows in sorted(source_groups.items()):
        source_rows.append({'task_id': task, 'source_group': source, 'rule': rule,
                           'state_count': len(rows), **{field: float(np.mean([row[field] for row in rows])) for field in SUCCESS_FIELDS}})
    summary = {'schema': 'dsol_view_metric_rules_summary_v1', 'status': 'PASS_DESCRIPTIVE_RULE_ANALYSIS',
        'model_label': report['provenance'].get('model_label', ''), 'state_count': n,
        'candidate_count': 97, 'rollout_repeat_count': 32, 'score_noise_member_count': 8, 'top_k': 10,
        'rule_order': list(RULES), 'overall': aggregate_rules(state_rules, state_diagnostics, state_diagnostics),
        'by_task': {task: aggregate_rules([r for r in state_rules if r['task_id']==task],
                     [r for r in state_diagnostics if r['task_id']==task], [r for r in state_diagnostics if r['task_id']==task])
                    for task in sorted({state['task_id'] for state in states})},
        'difficulty_crossfit': {'easy_threshold': .9, 'hard_floor_threshold': .1,
            'grouping_variable': 'uniform97 success on the opposite O16 half',
            'changed_stratum_state_count': changed, 'state_count': n,
            'state_membership_exclusive_across_strata': changed == 0,
            'strata': {name: aggregate_rules([r for r in difficulty_rows if r['difficulty']==name],
                       [r for r in difficulty_rows if r['difficulty']==name and r['rule']=='canonical']) for name in DIFFICULTIES},
            'interpretation': 'Performance-stratified held-out-noise description, not causal task difficulty or physical evidence identification; no new source generalization.'},
        'optimistic_topset': {field: source_bootstrap(headroom, field) for field in
            ('top10_empirical_max_success', 'all97_empirical_max_success', 'top10_max_shortfall_pp',
             'top10_contains_any_empirical_maximizer', 'optimistic_top10_headroom_over_uniform_pp', 'optimistic_top10_headroom_over_canonical_pp')},
        'undefined_full32_correlations': {field: sum(not np.isfinite(row[field]) for row in state_diagnostics) for field in ASSOCIATIONS},
        'constant_success_state_count': sum(row['constant_success_across_candidates'] for row in state_diagnostics),
        'new_confirmation': False,
        'statistical_scope': 'Task-equal and source-equal; source-level descriptive bootstrap. Paired-noise SE in CSV is separate conditional measurement variation, not a source CI.'}
    return {'summary': summary, 'state_rule_metrics': state_rules, 'source_rule_metrics': source_rows,
            'state_association_stability': state_diagnostics, 'difficulty_fold_metrics': fold_rows,
            'difficulty_state_metrics': difficulty_rows, 'diagnostic_topset_headroom': headroom,
            'selected_candidates': selected_rows}


def freeze_manifest(report: dict, input_root: Path, model_kind: str) -> dict:
    return {'schema': 'dsol_view_metric_rules_manifest_v1', 'status': 'FROZEN_BEFORE_RULE_OUTCOME_AGGREGATION',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'model_kind': model_kind,
        'script_sha256': sha256(Path(__file__)), 'input_root': str(input_root.resolve()),
        'input_identity': report['input_identity'], 'checkpoint_sha256': report['provenance']['checkpoint_sha256'],
        'rules': list(RULES), 'top_k': 10, 'candidate_count': 97, 'canonical_candidate_index': 0,
        'tie_break': {'min_accel_and_accel_top10': '(mean Accel, candidate ID)',
                      'max_visibility_and_visibility_top10': '(-visibility, candidate ID)',
                      'accel_top10_max_visibility': '(-visibility, mean Accel, candidate ID) inside low-Accel top10; exact historical Gate-A structure/tie rule',
                      'visibility_top10_min_accel': '(mean Accel, -visibility, candidate ID) inside high-visibility top10; new reverse-order comparator'},
        'uniform_rule_evaluation': 'Expected success under uniform candidate choice, not a newly sampled camera or rollout.',
        'visibility_definition': 'Privileged MuJoCo task-entity instance-mask occupied pixels / full image area, averaged across configured task entities and external/wrist cameras. It is not visible-object fraction, information entropy or learned RGB-only information gain.',
        'visibility_ranking_scope': 'Wrist image term is constant across external candidates within one state, so candidate ordering equals ordering by the external term.',
        'difficulty': {'easy': 'opposite-half uniform97 mean >= 0.9', 'hard_floor': 'opposite-half uniform97 mean <= 0.1',
                       'middle': 'otherwise', 'folds': [[list(range(16)), list(range(16,32))], [list(range(16,32)), list(range(16))]],
                       'aggregation': 'Average held-out directions within state and stratum; changing assignments can place a state in two strata.'},
        'bootstrap': {'seed': BOOTSTRAP_SEED, 'resamples': BOOTSTRAP_RESAMPLES,
                      'unit': 'source within each fixed task; average states within source, sources within task, tasks equally',
                      'insufficient_replicates': 'Suppress CI if any represented task has fewer than two finite sources.'},
        'scope': 'CPU-only prespecified-rule descriptive analysis of all verified states, with no outcome-driven state selection.',
        'legacy_accel_provenance': report['provenance']['static_assets']['accel_checkpoint_provenance'],
        'limitations': [
            'Historical outcomes have already been inspected in the research program; this is not a new confirmatory experiment.',
            'Low global rho can reflect multiple mechanisms; cross-fit stratification tests descriptive dilution, not its cause.',
            'Score-member seeds are archived; legacy initial score-noise tensor hashes were not. Score noise is not paired with rollout noise.',
            'Top-set empirical maxima reuse outcomes and are optimistic diagnostics, not deployable predictors or certified true maxima.',
            'Candidate images and entity masks are privileged offline inputs; no moving-camera execution or cost is validated.',
        ]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=NEW_ROOT / 'broad-existing')
    parser.add_argument('--model-kind', choices=('broad', 'canonical'), default='broad')
    parser.add_argument('--output-dir', type=Path, default=NEW_ROOT / 'metric-rules-v1')
    args = parser.parse_args()
    require(not args.output_dir.exists(), 'Output must not exist; frozen analyses are never overwritten')
    report = load_report(args.input_root, args.model_kind)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output_dir / 'manifest.json'
    # Write this first, before success-rate, association or grouping calculations.
    write_json(manifest_path, freeze_manifest(report, args.input_root, args.model_kind))
    result = analyze(report)
    for name in ('state_rule_metrics', 'source_rule_metrics', 'state_association_stability',
                 'difficulty_fold_metrics', 'difficulty_state_metrics', 'diagnostic_topset_headroom'):
        write_csv(args.output_dir / (name + '.csv'), finite_json(result[name]))
    write_json(args.output_dir / 'selected_candidates.json', {'states_and_rules': result['selected_candidates']})
    result['summary']['manifest_sha256'] = sha256(manifest_path)
    write_json(args.output_dir / 'summary.json', result['summary'])
    write_json(args.output_dir / 'completion.json', {'status': 'PASS_COMPLETE',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': sha256(manifest_path),
        'outputs': {path.name: sha256(path) for path in args.output_dir.iterdir() if path.is_file()},
        'state_count': len(report['states']), 'rule_count': len(RULES), 'gpu_used': False, 'new_rollouts': 0})
    print(f"PASS_COMPLETE {args.output_dir}", flush=True)


if __name__ == '__main__':
    main()
