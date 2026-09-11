#!/usr/bin/env python3
"""CPU-only, fixed-eight-state comparison of verified canonical/Broad landscapes.

No policy calls, no new camera observations, no source-bootstrap confidence
intervals. Noise-conditional measurement SE is reported only for fixed-view
paired model differences, never as evidence of state/task generalization.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from .build_view_landscape_report_v1 import (
        PARAMS, TASK_LABELS, catalog_bank, finite_json, ranks, read_json,
        require, setup_plotting, sha256, spearman, state_metrics, write_csv, write_json,
    )
except ImportError:
    from build_view_landscape_report_v1 import (
        PARAMS, TASK_LABELS, catalog_bank, finite_json, ranks, read_json,
        require, setup_plotting, sha256, spearman, state_metrics, write_csv, write_json,
    )


REPO = Path(__file__).resolve().parents[2]
NEW_ROOT = Path('/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908')
CHECKPOINTS = {
    'canonical': '7e510b752143bb6fd4987988dfab5e94f7db1de5f7e97579f892a58eed22cc68',
    'broad': '123786f53c9823b878fb08fe61ef25fe4931d4bcae8134e0a64940b7a8ac3cad',
}
CONFIGS = {
    'canonical': 'ab2bf0f607def9be8fd231da32a16f91d6b6605244bb5d84c01b4406c4f9723e',
    'broad': 'c22b3ffcb5de139e06e37ff95eadef00b01ee9db65c0ca6b46dd0db1c844b8d9',
}
SELECTION_SHA256 = '290d5ec185e43030cb5425b09a6cd4ddd15901eb28420cdb7a71796c84bf8dc1'
BANK_MANIFEST_SHA256 = '791caaba04b81b1325dfcb9d119a07e20f1e134f10056e3633f8a5baaa513264'
BANK_FILE_SHA256 = '2dcdbeb3f544fe4ee08b4d08ec2ca61dcdb05e0734a18c1f28000971ffbab9e6'
SOURCE_FIELDS = ('pair_key', 'task_id', 'source_group', 'asset_source_pair_key',
                 'source_state_index', 'environment_seed', 'split', 'construction_spec_sha256')
ASSET_HASH_FIELDS = ('physics_state_sha256', 'policy_inputs_sha256', 'render_receipt_sha256', 'visibility_scan_sha256')
GEOMETRY_FIELDS = ('pose_id', *PARAMS, 'orientation_mode', 'catalog_group',
                   'distance_canonical_parameter_normalized', 'distance_train64_parameter_normalized')
RULE_KEYS = ('canonical_success', 'uniform_view_success', 'accel_selected_success',
             'empirical_max_success_diagnostic', 'noise_crossfit_search_success')
RULE_LABELS = ('Canonical', 'Uniform 97', 'Min Accel', 'O32 max\n(in-sample)', 'O16 cross-fit\n(both directions)')


def array_contract(report: dict, label: str) -> None:
    states, poses = report['states'], report['poses']
    n = len(states)
    require(n == 8 if label == 'canonical' else n >= 8, 'Canonical must contain exactly 8 states; Broad at least the selected 8')
    require(len({state['pair_key'] for state in states}) == n, 'Duplicate state metadata')
    require(len(poses) == 97 and len({pose['pose_id'] for pose in poses}) == 97, 'Missing/duplicate candidate identities')
    require(poses[0]['pose_id'] == 'canonical', 'Canonical must be candidate zero')
    require(report['state_keys'] == [state['pair_key'] for state in states], 'Array/state metadata order mismatch')
    require(report['candidate_ids'] == [pose['pose_id'] for pose in poses], 'Array/candidate metadata order mismatch')
    require(report['success'].shape == (n, 97, 32), 'Expected complete state x 97 x 32 outcome tensor')
    require(np.isin(report['success'], [0, 1]).all(), 'Missing/nonbinary success cell')
    require(report['accel'].shape == (n, 97, 8) and np.isfinite(report['accel']).all(), 'Expected finite state x 97 x 8 Accel tensor')
    require(report['visibility'].shape == (n, 97) and np.isfinite(report['visibility']).all(), 'Missing/nonfinite visibility cell')
    for state in states:
        require(state['accel_action_horizon'] == 10, 'Accel action horizon differs from 10')
        seeds = state['accel_ensemble_seeds']
        require(len(seeds) == 8 and len(set(seeds)) == 8, 'Missing/duplicate Accel member seed')


def load_report(root: Path, label: str) -> dict:
    provenance = read_json(root / 'provenance.json')
    require(provenance['schema'] == 'dsol_view_landscape_provenance_v1' and provenance['status'].startswith('PASS_'), 'Report provenance is not PASS')
    require(provenance['checkpoint_sha256'] == CHECKPOINTS[label], 'Report checkpoint hash differs from requested model')
    require(sha256(root / 'verified_arrays.npz') == provenance['cache_sha256'], 'Verified aggregate content hash changed')
    require(provenance['static_assets']['physics_and_image_join'] == 'PASS_ALL', 'Upstream physics/image join failed')
    for item in provenance['file_stats']:
        stat = Path(item['path']).stat()
        require((stat.st_size, stat.st_mtime_ns) == (item['size'], item['mtime_ns']), 'Upstream artifact changed after report verification: ' + item['path'])
    # Recheck small bound manifests/audits; do not scan historical rollout GBs or
    # large weights again. Their verified content hashes and unchanged stats are
    # carried explicitly into this derivative report's provenance.
    manifests, audits = [], []
    for item in provenance['dense']['inputs']:
        path = Path(item['path'])
        if path.name not in ('run_manifest.json', 'audit.json'):
            continue
        require(sha256(path) == item['sha256'], 'Upstream run/audit content changed')
        row = read_json(path)
        if path.name == 'run_manifest.json':
            require(row['checkpoint_sha256'] == CHECKPOINTS[label], 'Dense manifest belongs to another checkpoint')
            require(row['policy_backend'] == 'alphabrain' and row['require_explicit_noise'], 'Dense backend/noise controls differ')
            require((row['replan_steps'], row['wait_steps']) == (5, 0), 'Camera installation/replan controls differ')
            require(row['noise_bank_manifest_sha256'] == BANK_MANIFEST_SHA256, 'Dense run uses another noise bank')
            manifests.append(item)
        else:
            require(row['status'] == 'PASS_COMPLETE' and all(row[key] for key in (
                'every_policy_call_matches_frozen_noise_bank', 'paired_noise_identity_at_common_replan_indices',
                'physics_hash_constant_within_state', 'environment_seed_constant_within_state')), 'Dense integrity audit failed')
            audits.append(item)
    require(len(manifests) == len(audits) == provenance['dense']['waves'] > 0, 'Missing/duplicate dense run or audit manifests')
    config_path = Path(provenance['checkpoint']) / 'framework_config.yaml'
    require(sha256(config_path) == CONFIGS[label], 'Model framework/normalization config changed')
    config = yaml.safe_load(config_path.read_text())
    require(config['framework']['action_model']['num_inference_steps'] == 10, 'Different flow denoising time grid')
    require(config['datasets']['vla_data']['obs'] == ['image_0', 'wrist_image'], 'Model camera input slots differ')
    states = read_json(root / 'states.json')['states']
    poses = read_json(root / 'candidate_geometry.json')['poses']
    with np.load(root / 'verified_arrays.npz', allow_pickle=False) as arrays:
        result = {'states': states, 'poses': poses, 'provenance': provenance,
                  'state_keys': list(map(str, arrays['state_keys'])), 'candidate_ids': list(map(str, arrays['candidate_ids'])),
                  **{key: arrays[key].copy() for key in ('accel', 'success', 'visibility')}}
    array_contract(result, label)
    result['input_identity'] = {name: {'path': str((root / name).resolve()), 'sha256': sha256(root / name)} for name in
                              ('verified_arrays.npz', 'states.json', 'provenance.json', 'candidate_geometry.json')}
    return result


def match_reports(canonical: dict, broad: dict, selection: dict) -> tuple[dict, dict, list[dict]]:
    array_contract(canonical, 'canonical')
    array_contract(broad, 'broad')
    selected = selection['states']
    require(len(selected) == 8 and len({state['pair_key'] for state in selected}) == 8, 'Selection must freeze eight unique states')
    require(len({state['task_id'] for state in selected}) == 8, 'Expected one state from each of eight known tasks')
    require(all(state['split'] == 'development' for state in selected), 'Not the frozen development-only comparison')
    require(selection.get('outcomes_or_view_metrics_used_in_selection') is False, 'Outcome-informed state selection is out of scope')
    require(canonical['provenance']['checkpoint_sha256'] == CHECKPOINTS['canonical'], 'Wrong canonical checkpoint')
    require(broad['provenance']['checkpoint_sha256'] == CHECKPOINTS['broad'], 'Wrong Broad checkpoint')
    require(canonical['provenance']['checkpoint_sha256'] != broad['provenance']['checkpoint_sha256'], 'Two reports use the same checkpoint')
    require(canonical['candidate_ids'] == broad['candidate_ids'], 'Candidate order differs across models')
    for left, right in zip(canonical['poses'], broad['poses']):
        require(all(left[key] == right[key] for key in GEOMETRY_FIELDS), 'Candidate geometry or Broad-reference coordinates differ')
    for report in (canonical, broad):
        dense = report['provenance']['dense']
        require(dense['bank_manifest_sha256'] == BANK_MANIFEST_SHA256 and dense['bank_file_sha256'] == BANK_FILE_SHA256, 'Not shared bank O')
    selected = sorted(selected, key=lambda state: state['task_id'])
    aligned = []
    for report in (canonical, broad):
        index = {state['pair_key']: i for i, state in enumerate(report['states'])}
        require({state['pair_key'] for state in selected} <= set(index), 'Selected state missing from report')
        indices = [index[state['pair_key']] for state in selected]
        aligned_states = [report['states'][i] for i in indices]
        for frozen, state in zip(selected, aligned_states):
            require(all(frozen[key] == state[key] for key in SOURCE_FIELDS), 'Source/state/environment identity differs from frozen selection')
            require(all(frozen['static_assets'][key] == state['static_assets'][key] for key in ASSET_HASH_FIELDS), 'Physics/image/render identity differs from frozen selection')
        aligned.append({**report, 'states': aligned_states, 'state_keys': [state['pair_key'] for state in selected],
                        **{key: report[key][indices] for key in ('accel', 'success', 'visibility')}})
    for left, right in zip(aligned[0]['states'], aligned[1]['states']):
        require(left['accel_ensemble_seeds'] == right['accel_ensemble_seeds'], 'Per-state Accel member seeds/order differ')
    require(np.allclose(aligned[0]['visibility'], aligned[1]['visibility'], rtol=0, atol=1e-12), 'Fixed-state visibility differs between models')
    return aligned[0], aligned[1], selected


def summarize_fixed_states(values: list[float]) -> dict:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return {'mean': float(finite.mean()) if len(finite) else None,
            'min': float(finite.min()) if len(finite) else None,
            'max': float(finite.max()) if len(finite) else None,
            'finite_states': len(finite), 'total_states': len(values),
            'confidence_interval': None,
            'scope': 'Descriptive equal-weight average of these fixed states; no source-generalization interval.'}


def paired_noise_se(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    require(left.shape == right.shape and left.shape[-1] == 32, 'Paired noise SE needs common 32-repeat cells')
    return 100 * (right.astype(float) - left.astype(float)).std(axis=-1, ddof=1) / np.sqrt(32)


def compare_metrics(canonical: dict, broad: dict) -> tuple[dict, list[dict], list[dict], dict]:
    metric_rows = {}
    for name, report in (('canonical', canonical), ('broad', broad)):
        metric_rows[name], _ = state_metrics(report['states'], report['poses'], report['accel'], report['success'], report['visibility'])
    c_rates, b_rates = canonical['success'].mean(2), broad['success'].mean(2)
    c_accel, b_accel = canonical['accel'].mean(2), broad['accel'].mean(2)
    se = paired_noise_se(canonical['success'], broad['success'])
    state_rows, cells = [], []
    for i, (c, b) in enumerate(zip(metric_rows['canonical'], metric_rows['broad'])):
        row = {key: c[key] for key in ('pair_key', 'task_id', 'source_group')}
        for name, source in (('canonical_model', c), ('broad_model', b)):
            for key in (*RULE_KEYS, 'accel_selected_id', 'accel_gain_vs_canonical_pp', 'noise_crossfit_gain_vs_canonical_pp',
                        'spearman_negative_accel_success', 'noise_mean_pairwise_rank_spearman'):
                row[name + '_' + key] = source[key]
        row.update({
            'broad_minus_canonical_fixed_view_pp': 100 * (b['canonical_success'] - c['canonical_success']),
            'broad_minus_canonical_uniform_view_pp': 100 * (b['uniform_view_success'] - c['uniform_view_success']),
            'training_by_min_accel_gain_difference_pp': b['accel_gain_vs_canonical_pp'] - c['accel_gain_vs_canonical_pp'],
            'training_by_o16_search_gain_difference_pp': b['noise_crossfit_gain_vs_canonical_pp'] - c['noise_crossfit_gain_vs_canonical_pp'],
            'cross_model_success_rank_rho': spearman(c_rates[i], b_rates[i]),
            'cross_model_accel_rank_rho': spearman(c_accel[i], b_accel[i]),
            'min_accel_candidate_agrees_between_models': int(c['accel_selected_id'] == b['accel_selected_id']),
        })
        state_rows.append(row)
        cr, br, car, bar = ranks(c_rates[i]), ranks(b_rates[i]), ranks(c_accel[i]), ranks(b_accel[i])
        for j, pose in enumerate(canonical['poses']):
            cells.append({'pair_key': c['pair_key'], 'task_id': c['task_id'], 'candidate_id': pose['pose_id'],
                **{key: pose[key] for key in PARAMS},
                'catalog_reference_group': ('canonical' if j == 0 else 'Broad catalog reference 64' if pose['catalog_group'] == 'train64' else 'Broad held-out reference 32'),
                'canonical_model_success': c_rates[i, j], 'broad_model_success': b_rates[i, j],
                'broad_minus_canonical_success_pp': 100 * (b_rates[i, j] - c_rates[i, j]),
                'paired_noise_conditional_se_pp': se[i, j], 'paired_O_repeat_count': 32,
                'canonical_model_success_rank': cr[j], 'broad_model_success_rank': br[j],
                'canonical_model_accel_mean': c_accel[i, j], 'broad_model_accel_mean': b_accel[i, j],
                'canonical_model_accel_rank': car[j], 'broad_model_accel_rank': bar[j]})
    numeric = [key for key, value in state_rows[0].items() if isinstance(value, (float, np.floating, int, np.integer))]
    summary = {'schema': 'dsol_matched_view_landscape_comparison_v1', 'status': 'PASS_DESCRIPTIVE_MATCHED_PAIR',
        'state_count': 8, 'known_task_count': 8, 'source_states_per_task': 1, 'candidate_count': 97,
        'score_noise_members': 8, 'rollout_noise_repeats': 32, 'new_confirmation': False,
        'source_generalization_confidence_interval': None,
        'fixed_state_statistics': {key: summarize_fixed_states([row[key] for row in state_rows]) for key in numeric},
        'rule_summary': {model: {key: summarize_fixed_states([row[key] for row in rows]) for key in RULE_KEYS}
                         for model, rows in metric_rows.items()},
        'limitations': [
            'Eight known tasks, one frozen development source state each; no source-bootstrap CI is estimable within tasks.',
            'O32 empirical maximum reuses outcomes for selection/evaluation and is optimistic, not a certified oracle.',
            'O16 cross-fit selects on 0..15 and evaluates on 16..31, then reverses; still conditional on these states and historical bank O.',
            'Accel selection uses an independent score-noise ensemble. Score and rollout noise are not sample-wise paired.',
            'Score member seed identities match, but legacy score-noise tensor hashes were not archived; no tensor-level retrospective claim.',
            'Fixed-view paired-noise SE describes sampling variation conditional on these states, not task/state generalization.',
            'Broad catalog reference groups do not denote canonical-model training support or pretraining familiarity.',
            'Cached candidate-image scoring is offline diagnosis; it does not validate cost-aware mobile-camera deployment.',
        ]}
    arrays = {'canonical_success': canonical['success'], 'broad_success': broad['success'],
              'canonical_accel': canonical['accel'], 'broad_accel': broad['accel'],
              'success_difference_pp': 100 * (b_rates - c_rates), 'paired_noise_conditional_se_pp': se,
              'candidate_ids': np.asarray(canonical['candidate_ids']), 'state_keys': np.asarray(canonical['state_keys'])}
    return summary, state_rows, cells, arrays


def plot_report(out: Path, canonical: dict, broad: dict, summary: dict, rows: list[dict]) -> list[str]:
    setup_plotting()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import ListedColormap
    labels = [TASK_LABELS.get(state['task_id'], state['task_id']) for state in canonical['states']]
    figures = out / 'figures'
    figures.mkdir()
    outputs = []
    with PdfPages(out / 'matched_view_landscapes_v1.pdf') as pdf:
        fig = plt.figure(figsize=(16, 9))
        fig.suptitle('Matched training comparison: the same eight states and 97 views', x=.035, y=.97, ha='left', fontsize=19, fontweight='bold')
        fig.text(.035, .91, 'Canonical external-pose post-training versus Broad64 post-training | Wrist images retained | Seed41 only', fontsize=11)
        grid = fig.add_gridspec(4, 2, left=.17, right=.94, bottom=.17, top=.86, width_ratios=[1, .025], height_ratios=[.12, 1, 1, 1], hspace=.47, wspace=.03)
        groups = [0 if i == 0 else 1 if p['catalog_group'] == 'train64' else 2 for i, p in enumerate(canonical['poses'])]
        ax = fig.add_subplot(grid[0, 0])
        ax.imshow([groups], aspect='auto', interpolation='nearest', cmap=ListedColormap(['#222222', '#e8b05d', '#8dabc7']), vmin=0, vmax=2)
        ax.set(xticks=[], yticks=[])
        ax.set_title('Catalog reference: black = canonical; amber = Broad reference 64; blue = Broad held-out reference 32', fontsize=10)
        c, b = canonical['success'].mean(2), broad['success'].mean(2)
        for row, matrix, title, lo, hi, cmap in ((1, c, 'Canonical-trained model: closed-loop success', 0, 100, 'viridis'),
                (2, b, 'Broad-trained model: closed-loop success', 0, 100, 'viridis'),
                (3, b-c, 'Broad minus canonical training: success difference', -100, 100, 'RdBu')):
            ax = fig.add_subplot(grid[row, 0])
            im = ax.imshow(matrix * 100, vmin=lo, vmax=hi, cmap=cmap, aspect='auto', interpolation='nearest')
            ax.set_yticks(range(8), labels, fontsize=8)
            ax.set_title(title, loc='left', fontsize=11)
            ax.axvline(.5, color='white', lw=1)
            ax.set_xticks([0, 16, 32, 48, 64, 80, 96])
            if row == 3:
                ax.set_xlabel('Candidate index: canonical first; then fixed azimuth / elevation / radius order', fontsize=10)
            fig.colorbar(im, cax=fig.add_subplot(grid[row, 1]), label='%' if row < 3 else 'pp')
        fig.text(.035, .055, 'Every cell uses the same 32 O-bank repeat IDs across models. Geometry order is fixed; no interpolation or outcome-based reordering.\nOne source state per task: these are paired development measurements, not unseen-task or source-generalization evidence.', fontsize=10, color='#475569')
        path = figures / '01_matched_success_fields.png'
        fig.savefig(path, dpi=150); pdf.savefig(fig); plt.close(fig); outputs.append(str(path))

        fig, axes = plt.subplots(2, 2, figsize=(16, 9))
        fig.subplots_adjust(left=.14, right=.96, top=.84, bottom=.20, hspace=.65, wspace=.34)
        fig.suptitle('Does training change the value and predictability of view choice?', x=.035, y=.97, ha='left', fontsize=19, fontweight='bold')
        fig.text(.035, .91, 'Descriptive summaries of the same fixed eight states | No source-bootstrap confidence intervals', fontsize=11)
        ax = axes[0, 0]
        x = np.arange(len(RULE_KEYS))
        for offset, model, color, label in ((-.18, 'canonical', '#54778e', 'Canonical training'), (.18, 'broad', '#d79238', 'Broad training')):
            values = [100 * summary['rule_summary'][model][key]['mean'] for key in RULE_KEYS]
            bars = ax.bar(x + offset, values, .34, color=color, label=label)
            ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
        ax.set_xticks(x, RULE_LABELS, fontsize=8); ax.set(ylim=(0, 110), ylabel='Success (%)')
        ax.set_title('A. Fixed-view and selection rules: state-equal means', loc='left', fontsize=11)
        ax.legend(frameon=False, fontsize=8, loc='upper left', ncols=2)
        ax = axes[0, 1]
        valid = 0
        point_labels = {}
        for i, row in enumerate(rows):
            xx, yy = row['canonical_model_spearman_negative_accel_success'], row['broad_model_spearman_negative_accel_success']
            if np.isfinite(xx) and np.isfinite(yy):
                point_labels.setdefault((round(xx, 8), round(yy, 8)), []).append(str(i+1))
                valid += 1
        for (xx, yy), indices in point_labels.items():
            ax.scatter(xx, yy, color='#446b83', s=45)
            ax.annotate(','.join(indices), (xx, yy), xytext=(-4 if xx > .5 else 4, 4), textcoords='offset points', fontsize=8,
                        ha='right' if xx > .5 else 'left')
        ax.plot([-1, 1], [-1, 1], '--', color='#999999', lw=1)
        ax.axhline(0, color='#bbbbbb', lw=.7); ax.axvline(0, color='#bbbbbb', lw=.7)
        ax.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), xlabel='Canonical-trained: rho(-Accel, success)', ylabel='Broad-trained: rho(-Accel, success)')
        ax.set_title(f'B. Within-state metric utility ({valid}/8 defined pairs)', loc='left', fontsize=11)
        ax = axes[1, 0]
        for i, row in enumerate(rows):
            c_gain, b_gain = row['canonical_model_accel_gain_vs_canonical_pp'], row['broad_model_accel_gain_vs_canonical_pp']
            ax.plot([c_gain, b_gain], [i, i], color='#bbbbbb', lw=1)
            ax.scatter(c_gain, i, color='#54778e', s=32)
            ax.scatter(b_gain, i, color='#d79238', s=32)
        ax.axvline(0, color='#444444', lw=1)
        extent = max(10, max(abs(row[key]) for row in rows for key in ('canonical_model_accel_gain_vs_canonical_pp', 'broad_model_accel_gain_vs_canonical_pp')) * 1.15)
        ax.set_xlim(-extent, extent)
        ax.set_yticks(range(8), [str(i+1) + '. ' + label for i, label in enumerate(labels)], fontsize=8)
        ax.invert_yaxis(); ax.set_xlabel('Min-Accel gain over that model\'s canonical view (pp)')
        ax.set_title('C. Selecting a view versus keeping the canonical view', loc='left', fontsize=11)
        ax = axes[1, 1]
        positions = np.arange(8)
        ax.bar(positions-.17, [row['cross_model_success_rank_rho'] for row in rows], .32, label='Success ranks', color='#698b6a')
        ax.bar(positions+.17, [row['cross_model_accel_rank_rho'] for row in rows], .32, label='Accel ranks', color='#9478a5')
        ax.axhline(0, color='#444444', lw=1); ax.set(ylim=(-1.1, 1.1), ylabel='Cross-model Spearman rho', xlabel='State number (same numbering as panel C)')
        ax.set_xticks(positions, list(range(1, 9))); ax.legend(frameon=False, fontsize=8, loc='lower left', ncols=2)
        ax.set_title('D. How much do the two model landscapes agree?', loc='left', fontsize=11)
        fig.text(.035, .065, 'O32 max is in-sample and optimistic; O16 cross-fit is still a state-conditional diagnostic, not a deployable selector or certified oracle.\nRaw Accel scales need not be calibrated across models: interpret rank relations. Undefined constant-vector correlations remain missing.\nScores use cached candidate images; this does not validate mobile-camera acquisition or its cost.', fontsize=9, color='#475569')
        path = figures / '02_matched_selection_diagnostics.png'
        fig.savefig(path, dpi=150); pdf.savefig(fig); plt.close(fig); outputs.append(str(path))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--broad-root', type=Path, required=True)
    parser.add_argument('--canonical-root', type=Path, required=True)
    parser.add_argument('--selection-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), 'Output must be a new directory; do not overwrite a comparison')
    require(sha256(args.selection_manifest) == SELECTION_SHA256, 'Not the frozen outcome-blind eight-state selection')
    selection = read_json(args.selection_manifest)
    canonical, broad = load_report(args.canonical_root, 'canonical'), load_report(args.broad_root, 'broad')
    require(canonical['provenance'].get('selection_manifest_sha256') == SELECTION_SHA256, 'Canonical report is not bound to the selection')
    catalog_path = REPO / 'configs/dsol_paper1/libero_view_catalog_v2_m1.json'
    rules_path = REPO / 'configs/dsol_paper1/libero_view_catalog_v2_m1_rules.json'
    for label, report in (('canonical', canonical), ('broad', broad)):
        require(report['provenance']['catalog_sha256'] == sha256(catalog_path) and report['provenance']['catalog_rules_sha256'] == sha256(rules_path), 'Catalog identity differs')
        expected, _ = catalog_bank(read_json(catalog_path), read_json(rules_path), 'canonical' if label == 'canonical' else 'broad64')
        for pose, frozen in zip(report['poses'], expected):
            require(all(pose[key] == frozen[key] for key in (*GEOMETRY_FIELDS, 'is_model_post_training_support', 'distance_model_support_parameter_normalized')), 'Catalog/order/model-support geometry mismatch')
    canonical, broad, selected = match_reports(canonical, broad, selection)
    checked_assets = {}
    for state in selected:
        for field in ('policy_inputs', 'render_receipt', 'visibility_scan'):
            path = Path(state['static_assets'][field])
            digest = sha256(path)
            require(digest == state['static_assets'][field + '_sha256'], 'Selected static asset content changed')
            checked_assets[str(path)] = digest
    summary, states, cells, arrays = compare_metrics(canonical, broad)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(args.output_dir / 'state_comparison.csv', finite_json(states))
    write_csv(args.output_dir / 'state_view_comparison.csv', finite_json(cells))
    np.savez_compressed(args.output_dir / 'matched_arrays.npz', **arrays)
    summary['figure_paths'] = plot_report(args.output_dir, canonical, broad, summary, states)
    write_json(args.output_dir / 'summary.json', summary)
    provenance = {'schema': 'dsol_matched_view_landscape_comparison_provenance_v1', 'status': 'PASS_COMPLETE_DESCRIPTIVE_ONLY',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': sha256(Path(__file__)),
        'selection': {'path': str(args.selection_manifest.resolve()), 'sha256': SELECTION_SHA256},
        'inputs': {'canonical': canonical['input_identity'], 'broad': broad['input_identity']},
        'checkpoint_sha256': CHECKPOINTS, 'bank_manifest_sha256': BANK_MANIFEST_SHA256, 'bank_file_sha256': BANK_FILE_SHA256,
        'selected_static_asset_hashes_reverified': checked_assets,
        'source_states': [{'pair_key': state['pair_key'], 'environment_seed': state['environment_seed'],
                           'accel_ensemble_seeds': state['accel_ensemble_seeds']} for state in canonical['states']],
        'legacy_accel_provenance': broad['provenance']['static_assets']['accel_checkpoint_provenance'],
        'validation_scope': 'Aggregate bytes, selected static assets, small run/audit/config manifests rehashed; historical raw ledgers/weights keep upstream full hashes plus unchanged file stat metadata.',
        'outputs': {path.name: sha256(path) for path in args.output_dir.iterdir() if path.is_file()}}
    write_json(args.output_dir / 'provenance.json', provenance)
    write_json(args.output_dir / 'completion.json', {'status': 'PASS_COMPLETE', 'state_count': 8, 'candidate_count': 97,
        'rollout_noise_repeats': 32, 'accel_noise_members': 8, 'source_bootstrap_ci_reported': False,
        'summary_sha256': sha256(args.output_dir / 'summary.json'), 'provenance_sha256': sha256(args.output_dir / 'provenance.json')})
    print(json.dumps({'status': 'PASS_COMPLETE', 'output_dir': str(args.output_dir), 'pdf_pages': 2}), flush=True)


if __name__ == '__main__':
    main()
