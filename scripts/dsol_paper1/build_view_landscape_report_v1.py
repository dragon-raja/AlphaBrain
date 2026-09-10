#!/usr/bin/env python3
"""CPU-only, provenance-checked descriptive view landscapes; never runs a policy.

Canonical and future checkpoints can reuse this schema/plotter with their own
Accel and dense-O roots. Historical discovery outcomes are not confirmation data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = Path('/share/longjunyu/alphabrain/experiments')
DEFAULT_CHECKPOINT = EXPERIMENTS / 'dsol-libero-broad-pairing-v1/runs/dsol_broad_unpaired_practical_m-b-formal-v1_seed41_g2_gb32_steps2000/final_model'
DEFAULT_HASH = '123786f53c9823b878fb08fe61ef25fe4931d4bcae8134e0a64940b7a8ac3cad'
PARAMS = ('azimuth_deg', 'elevation_deg', 'radius_scale')
TASK_LABELS = {
    'goal_cream_cheese_bowl': 'Cream cheese / bowl',
    'goal_top_drawer_bowl': 'Bowl / top drawer',
    'goal_wine_rack': 'Wine / rack',
    'libero10_book_caddy': 'Book / caddy',
    'libero10_bowl_bottom_drawer': 'Bowl / bottom drawer',
    'libero10_mug_microwave': 'Mug / microwave',
    'object_cream_cheese_basket': 'Cream cheese / basket',
    'spatial_drawer_bowl_plate': 'Drawer bowl / plate',
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [finite_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite_json(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, so tied empirical success rates are handled correctly."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='stable')
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return result


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left, right = ranks(left), ranks(right)
    if np.ptp(left) == 0 or np.ptp(right) == 0:
        return float('nan')
    return float(np.corrcoef(left, right)[0, 1])


def wilson(successes: int, count: int) -> tuple[float, float]:
    z, p = 1.95996398454, successes / count
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return center - half, center + half


def catalog_bank(catalog: dict, rules: dict, training_support: str = 'broad64') -> tuple[list[dict], np.ndarray]:
    poses = catalog['canonical'] + catalog['broad_training_64'] + catalog['broad_heldout_32']
    require(len(poses) == 97, 'Expected frozen 97-view bank')
    require(len({p['pose_id'] for p in poses}) == 97, 'Duplicate pose identity')
    # Fixed geometry order, independent of scores/outcomes; canonical is distinct.
    poses = [poses[0]] + sorted(poses[1:], key=lambda p: tuple(p[k] for k in PARAMS))
    coords = np.array([[p[k] for k in PARAMS] for p in poses], dtype=float)
    ranges = rules['broad_training']['ranges']
    scale = np.array([ranges[k][1] - ranges[k][0] for k in PARAMS])
    normalized = (coords - np.array([0, 0, 1])) / scale
    train = np.array([i for i, p in enumerate(poses) if p['pose_id'].startswith('broad_train_')])
    model_train = np.array([0]) if training_support == 'canonical' else train
    for i, pose in enumerate(poses):
        pose['distance_canonical_parameter_normalized'] = float(np.linalg.norm(normalized[i]))
        pose['distance_train64_parameter_normalized'] = float(np.min(np.linalg.norm(normalized[train] - normalized[i], axis=1)))
        pose['distance_model_support_parameter_normalized'] = float(np.min(np.linalg.norm(normalized[model_train] - normalized[i], axis=1)))
        pose['is_model_post_training_support'] = bool(i in model_train)
        pose['catalog_group'] = ('canonical' if i == 0 else 'train64' if i in train else 'heldout32')
    return poses, coords


def load_static_assets(args: argparse.Namespace, poses: list[dict]) -> tuple[list[dict], np.ndarray, dict]:
    population = read_json(args.population)
    states = population['population']['development']['states'] + population['population']['test']['states']
    states = sorted(states, key=lambda s: (s['task_id'], s['split'], s['source_group'], s['source_state_index']))
    if args.selection_manifest:
        selected = read_json(args.selection_manifest)
        keys = selected.get('selected_state_keys', selected.get('state_keys'))
        if keys is None:
            records = selected.get('states', selected.get('selected_states', []))
            require(isinstance(records, list), 'Selection must contain an explicit state list, not only a count')
            keys = [s['pair_key'] for s in records]
        require(bool(keys) and len(set(keys)) == len(keys), 'Invalid explicit state selection')
        require(set(keys) <= {s['pair_key'] for s in states}, 'Selection contains unknown states')
        states = [s for s in states if s['pair_key'] in set(keys)]
    require(len({s['pair_key'] for s in states}) == len(states), 'Duplicate state key')
    pose_ids = [p['pose_id'] for p in poses]
    raw, inputs = {}, []
    for path in sorted(args.accel_root.glob('rank-shard-*.jsonl')):
        inputs.append({'path': str(path), 'sha256': sha256(path)})
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                require(row['pair_key'] not in raw, 'Duplicate Accel state')
                raw[row['pair_key']] = row
    require(raw, 'No Accel rank ledgers')
    values = []
    hash_bound_records = 0
    for state in states:
        row = raw[state.get('asset_source_pair_key', state['pair_key'])]
        require(row['status'] == 'PASS' and row['candidate_count'] == 97, 'Invalid Accel record')
        if 'checkpoint_sha256' in row:
            require(row['checkpoint_sha256'] == args.expected_checkpoint_sha256, 'Accel bound checkpoint hash mismatch')
            hash_bound_records += 1
        assets = state['static_assets']
        require(row['physics_state_sha256'] == assets['physics_state_sha256'], 'Physics hash join mismatch')
        require(row['render_artifact_sha256'] == assets['policy_inputs_sha256'], 'Image hash join mismatch')
        for name in ('policy_inputs', 'render_receipt', 'visibility_scan'):
            artifact = Path(assets[name])
            actual = sha256(artifact)
            require(actual == assets[name + '_sha256'], f'Asset content hash mismatch: {artifact}')
            inputs.append({'path': str(artifact), 'sha256': actual})
        render = read_json(Path(assets['render_receipt']))
        require(render['physics_state_sha256'] == assets['physics_state_sha256'], 'Render receipt mismatch')
        with np.load(assets['policy_inputs'], allow_pickle=False) as data:
            require(set(map(str, data['candidate_ids'])) == set(pose_ids), 'Image candidate IDs mismatch')
        ranking = {r['candidate_id']: r for r in row['ranking']}
        require(set(ranking) == set(pose_ids), 'Accel candidate set mismatch')
        array = np.array([ranking[p]['member_accel_3'] for p in pose_ids], dtype=float)
        require(array.shape == (97, row['ensemble_size']), 'Accel ensemble shape mismatch')
        require(np.isfinite(array).all(), 'Non-finite Accel')
        require(np.allclose(array.mean(1), [ranking[p]['mean_accel_3'] for p in pose_ids]), 'Accel means mismatch')
        require(len(row['ensemble_seeds']) == array.shape[1], 'Accel seed metadata mismatch')
        state['accel_ensemble_seeds'] = row['ensemble_seeds']
        state['accel_action_horizon'] = row['checkpoint_action_horizon']
        values.append(array)
    # Legacy rank ledgers omitted checkpoint hashes. Preserve this limitation;
    # corroborate loader logs and never mislabel it as a historical hash binding.
    logs = sorted((args.accel_root / 'logs').glob('rank-shard-*.log'))
    if not hash_bound_records:
        require(len(logs) == len(list(args.accel_root.glob('rank-shard-*.jsonl'))), 'Missing Accel loader logs')
    else:
        require(hash_bound_records == len(states), 'Mixed Accel checkpoint provenance')
    compact_checkpoint = re.sub(r'\s+', '', str(args.checkpoint.resolve()))
    for log in logs if not hash_bound_records else []:
        compact = re.sub(r'\s+', '', log.read_text())
        require(compact_checkpoint in compact, f'Accel checkpoint loader path mismatch: {log}')
        inputs.append({'path': str(log), 'sha256': sha256(log)})
    return states, np.stack(values), {'inputs': inputs, 'state_count': len(states), 'physics_and_image_join': 'PASS_ALL',
        'accel_checkpoint_provenance': ('Every selected Accel record binds verified checkpoint content hash.' if hash_bound_records else
            'Loader-path corroborated; historical ranking records did NOT bind a weight hash. Current weight content verified separately; no retrospective cryptographic guarantee.')}


def load_dense(args: argparse.Namespace, states: list[dict], poses: list[dict], checkpoint_hash: str) -> tuple[np.ndarray, np.ndarray, dict]:
    state_index = {s['pair_key']: i for i, s in enumerate(states)}
    pose_index = {p['pose_id']: i for i, p in enumerate(poses)}
    bank_path = args.noise_bank_manifest or args.rollout_root / 'noise-banks/bank_O.manifest.json'
    bank, bank_hash = read_json(bank_path), sha256(bank_path)
    require(bank['bank_id'] == 'O' and bank['repeat_count'] == 32, 'Wrong noise bank')
    require(sha256(Path(bank['noise_file'])) == bank['noise_file_sha256'], 'Noise file hash mismatch')
    repeats = int(bank['repeat_count'])
    success = np.full((len(states), len(poses), repeats), -1, dtype=np.int8)
    visibility = np.full((len(states), len(poses)), np.nan)
    artifacts, episode_ids, count = [], set(), 0
    audit_fields = ('every_policy_call_matches_frozen_noise_bank', 'paired_noise_identity_at_common_replan_indices', 'physics_hash_constant_within_state', 'environment_seed_constant_within_state')
    if args.dense_dir:
        # Include smoke and both remainder segments, not only wave-*.
        waves = sorted(p.parent for p in args.dense_dir.glob('*/audit.json'))
    else:
        waves = sorted(itertools.chain((args.rollout_root / 'development/O').glob('wave-*'), (args.rollout_root / 'test/O').glob('wave-*')))
    require(waves, 'No dense O waves')
    for wave in waves:
        audit_path, manifest_path = wave / 'audit.json', wave / 'run_manifest.json'
        audit, manifest = read_json(audit_path), read_json(manifest_path)
        require(audit['status'] == 'PASS_COMPLETE' and all(audit[k] for k in audit_fields), f'Bad audit: {wave}')
        require(sha256(manifest_path) == audit['run_manifest_sha256'], 'Run manifest hash mismatch')
        require(sha256(Path(audit['protocol'])) == audit['protocol_sha256'], 'Protocol hash mismatch')
        require(Path(manifest['checkpoint']).resolve() == args.checkpoint.resolve(), 'Checkpoint path mismatch')
        require(manifest['checkpoint_sha256'] == checkpoint_hash, 'Checkpoint weight identity mismatch')
        require(audit['noise_bank_manifest_sha256'] == bank_hash, 'Noise manifest hash mismatch')
        require(manifest['noise_bank_manifest_sha256'] == bank_hash, 'Run noise manifest mismatch')
        require(manifest['replan_steps'] == 5 and manifest['wait_steps'] == 0, 'Rollout controls mismatch')
        wave_count = 0
        for path in sorted(wave.glob('episodes-shard-*.jsonl')):
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for line in stream:
                    digest.update(line)
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    require(row['status'] == 'complete', 'Incomplete episode')
                    require(row['episode_id'] not in episode_ids, 'Duplicate episode ID')
                    episode_ids.add(row['episode_id'])
                    wave_count += 1
                    if row['pair_key'] not in state_index:
                        require(args.selection_manifest is not None, 'Unexpected unselected state without an explicit selection manifest')
                        continue
                    i, j, r = state_index[row['pair_key']], pose_index[row['selected_candidate_id']], int(row['policy_repeat_id'])
                    require(0 <= r < repeats and success[i, j, r] < 0, 'Duplicate/out-of-range state-view-repeat')
                    state = states[i]
                    require(row['asset_source_pair_key'] == state['asset_source_pair_key'], 'State asset identity mismatch')
                    require(row['noise_bank_id'] == 'O' and row['noise_bank_manifest_sha256'] == bank_hash, 'Episode bank mismatch')
                    require(row['explicit_flow_noise'] is True, 'Non-explicit rollout noise')
                    require(row['initial_metrics']['physics_state_sha256'] == state['static_assets']['physics_state_sha256'], 'Episode physics mismatch')
                    require(row['static_assets']['policy_inputs_sha256'] == state['static_assets']['policy_inputs_sha256'], 'Episode input identity mismatch')
                    require(row['environment_seed'] == state['environment_seed'], 'Environment seed mismatch')
                    if j:
                        require(all(abs(float(row['pose'][k]) - float(poses[j][k])) < 1e-10 for k in PARAMS), 'Episode pose mismatch')
                    score = float(row['candidate_features']['visibility_score'])
                    require(np.isnan(visibility[i, j]) or abs(visibility[i, j] - score) < 1e-12, 'Visibility changes between repeats')
                    visibility[i, j] = score
                    success[i, j, r] = int(bool(row['success']))
            artifacts.append({'path': str(path), 'sha256': digest.hexdigest()})
        require(wave_count == audit['episode_count'], f'Audit row count mismatch: {wave}')
        artifacts.extend({'path': str(p), 'sha256': sha256(p)} for p in (audit_path, manifest_path, Path(audit['protocol'])))
        count += wave_count
        print(json.dumps({'stage': 'dense_read', 'wave': str(wave), 'episodes': wave_count}), flush=True)
    require(np.all(success >= 0), f'Missing dense cells: {int((success < 0).sum())}')
    return success, visibility, {'episodes_read': count, 'episodes_selected': int(success.size), 'waves': len(waves), 'inputs': artifacts,
        'bank_manifest': str(bank_path), 'bank_manifest_sha256': bank_hash, 'bank_file_sha256': bank['noise_file_sha256'],
        'scope': 'All ledger identities and asset hashes rechecked; per-policy-call noise tensors use the existing complete frozen-bank audits, not rerun here.'}


def state_metrics(states: list[dict], poses: list[dict], accel: np.ndarray, success: np.ndarray, visibility: np.ndarray) -> tuple[list[dict], list[dict]]:
    rates, means = success.mean(2), accel.mean(2)
    dc = np.array([p['distance_canonical_parameter_normalized'] for p in poses])
    dt = np.array([p['distance_model_support_parameter_normalized'] for p in poses])
    heldout = np.array([i for i, p in enumerate(poses) if i > 0 and not p['is_model_post_training_support']])
    summaries, cells = [], []
    for i, state in enumerate(states):
        a, p = means[i], rates[i]
        member_ranks = np.stack([ranks(accel[i, :, m]) for m in range(accel.shape[2])])
        top1 = np.argmin(accel[i], axis=0)
        pairs = list(itertools.combinations(range(accel.shape[2]), 2))
        jaccard = []
        for x, y in pairs:
            left, right = set(np.argsort(member_ranks[x])[:5]), set(np.argsort(member_ranks[y])[:5])
            jaccard.append(len(left & right) / len(left | right))
        selected = int(np.argmin(a))
        first, last = success[i, :, :16].mean(1), success[i, :, 16:].mean(1)
        # Deterministic tie-break uses frozen geometry order with canonical first.
        # Selection and evaluation halves are disjoint; neither estimate is an exact oracle.
        first_pick, last_pick = int(np.argmax(first)), int(np.argmax(last))
        crossfit_success = (last[first_pick] + first[last_pick]) / 2
        row = {'pair_key': state['pair_key'], 'task_id': state['task_id'], 'split': state['split'],
            'source_group': state['source_group'], 'demo_name': state['demo_name'], 'source_state_index': state['source_state_index'],
            'canonical_success': p[0], 'uniform_view_success': p.mean(), 'empirical_max_success_diagnostic': p.max(),
            'accel_selected_id': poses[selected]['pose_id'], 'accel_selected_success': p[selected],
            'accel_gain_vs_canonical_pp': 100 * (p[selected] - p[0]),
            'noise_crossfit_search_success': crossfit_success,
            'noise_crossfit_gain_vs_canonical_pp': 100 * (crossfit_success - p[0]),
            'noise_crossfit_first16_selected_id': poses[first_pick]['pose_id'],
            'noise_crossfit_last16_selected_id': poses[last_pick]['pose_id'],
            'noise_crossfit_picks_agree': int(first_pick == last_pick),
            'spearman_negative_accel_success': spearman(-a, p),
            'spearman_visibility_success': spearman(visibility[i], p),
            'spearman_canonical_distance_success': spearman(dc[1:], p[1:]),
            'spearman_canonical_distance_accel': spearman(dc[1:], a[1:]),
            'spearman_train_distance_success_heldout': spearman(dt[heldout], p[heldout]),
            'spearman_train_distance_accel_heldout': spearman(dt[heldout], a[heldout]),
            'noise_mean_pairwise_rank_spearman': np.mean([spearman(member_ranks[x], member_ranks[y]) for x, y in pairs]),
            'noise_mean_pairwise_top5_jaccard': np.mean(jaccard),
            'noise_top1_unique_count': len(set(top1)), 'noise_top1_all_agree': int(len(set(top1)) == 1),
            'noise_top1_mode_fraction': max(np.bincount(top1)) / len(top1),
            'outcome_repeat_half_rank_spearman': spearman(first, last),
            'success_rate_candidate_std': p.std(), 'accel_candidate_std': a.std(),
            'top5_accel_average_success': p[np.argsort(a)[:5]].mean(),
            'top10_accel_average_success': p[np.argsort(a)[:10]].mean()}
        summaries.append(row)
        for j, pose in enumerate(poses):
            n, k = success.shape[2], int(success[i, j].sum())
            low, high = wilson(k, n)
            cell = {k: row[k] for k in ('pair_key', 'task_id', 'split', 'source_group')}
            cell.update({k: pose[k] for k in ('pose_id', *PARAMS, 'orientation_mode', 'catalog_group', 'distance_canonical_parameter_normalized',
                'distance_train64_parameter_normalized', 'distance_model_support_parameter_normalized', 'is_model_post_training_support')})
            cell.update({'success_count': k, 'repeat_count': n, 'success_rate': p[j], 'success_wilson_low': low,
                'success_wilson_high': high, 'success_first16': success[i, j, :16].mean(), 'success_last16': success[i, j, 16:].mean(),
                'accel_mean': a[j], 'accel_std': accel[i, j].std(), 'accel_rank': ranks(a)[j],
                'accel_rank_noise_std': member_ranks[:, j].std(), 'visibility_score': visibility[i, j]})
            for member in range(accel.shape[2]):
                cell[f'accel_noise_{member:02d}'] = accel[i, j, member]
            cells.append(cell)
    return summaries, cells


def clustered_summary(rows: list[dict], field: str, seed: int = 20260908, resamples: int = 10000) -> dict:
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = float(row[field])
        if math.isfinite(value):
            groups[row['task_id']][row['source_group']].append(value)
    by_task = [np.array([np.mean(v) for v in g.values()]) for g in groups.values()]
    if not by_task:
        return {'mean': None, 'ci95': [None, None], 'finite_sources': 0, 'tasks': 0}
    if any(len(values) < 2 for values in by_task):
        return {'mean': float(np.mean([v.mean() for v in by_task])), 'ci95': [None, None],
            'finite_sources': sum(map(len, by_task)), 'tasks': len(by_task),
            'interval_status': 'UNSUPPORTED_INSUFFICIENT_SOURCES_PER_TASK',
            'interval_scope': 'At least one task has fewer than 2 finite source groups. Fixed-source descriptive mean only; no source-generalization interval.'}
    rng = np.random.default_rng(seed)
    boot = np.mean([values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(1) for values in by_task], axis=0)
    return {'mean': float(np.mean([v.mean() for v in by_task])), 'ci95': np.quantile(boot, [.025, .975]).tolist(),
        'finite_sources': sum(map(len, by_task)), 'tasks': len(by_task),
        'interval_scope': 'Descriptive task-equal, source-cluster bootstrap conditional on these tasks; rollout/Accel noise measurement error is not resampled.'}


def setup_plotting() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
        'axes.labelsize': 10, 'figure.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 3, 'savefig.facecolor': 'white'})


def number(value: float | None) -> str:
    return f'{value:+.3f}' if value is not None and math.isfinite(value) else 'undefined'


def estimate_label(item: dict) -> str:
    if item['ci95'][0] is None:
        return f"Mean {number(item['mean'])}; fixed sources, no source-generalization CI"
    return f"Mean {number(item['mean'])}; descriptive CI [{number(item['ci95'][0])}, {number(item['ci95'][1])}]"


def title(fig: Any, heading: str, subtitle: str) -> None:
    fig.suptitle(heading, x=.04, y=.975, ha='left', fontsize=19, fontweight='bold', color='#173047')
    fig.text(.04, .922, subtitle, fontsize=10, color='#475569', va='top')
    fig.text(.04, .018, 'Historical exploratory analysis | constructed-occlusion continuation states | not a deployed camera-search result', fontsize=8, color='#64748b')


def scatter3d(ax: Any, coords: np.ndarray, color: np.ndarray, label: str, lo: float = 0, hi: float = 1, cmap: str = 'viridis') -> Any:
    scatter = ax.scatter(*coords[1:].T, c=color[1:], vmin=lo, vmax=hi, cmap=cmap, s=30, alpha=.95)
    ax.scatter(*coords[0], c=[color[0]], vmin=lo, vmax=hi, cmap=cmap, s=130, marker='*', edgecolor='#111827', linewidth=1.2)
    ax.set(xlabel='Azimuth offset (deg)', ylabel='Elevation offset (deg)', zlabel='Radius scale', title=label)
    ax.set_xlim(-60, 60); ax.set_ylim(-25, 25); ax.set_zlim(.9, 1.25)
    ax.view_init(elev=23, azim=-60)
    return scatter


def save_page(fig: Any, name: str, out: Path, pdf: Any) -> None:
    import matplotlib.pyplot as plt
    fig.savefig(out / 'figures' / f'{name}.png', dpi=150)
    pdf.savefig(fig)
    plt.close(fig)


def plot_reports(args: argparse.Namespace, states: list[dict], poses: list[dict], coords: np.ndarray, accel: np.ndarray,
                 success: np.ndarray, visibility: np.ndarray, stats: list[dict], summary: dict) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    out = args.output_dir
    (out / 'figures').mkdir(exist_ok=True)
    (out / 'state_figures').mkdir(exist_ok=True)
    rates, means = success.mean(2), accel.mean(2)
    normalized_rank = np.stack([(ranks(a) - 1) / (len(a) - 1) for a in means])
    selected = []
    for task in sorted({s['task_id'] for s in states}):
        # Fixed before consulting outcomes: first lexicographic development source.
        selected.append(next(i for i, s in enumerate(states) if s['task_id'] == task and s['split'] == 'development'))
    summary['representative_selection'] = {'rule': 'First lexicographic development source_group, then frame, within every task; outcomes and scores unused.',
        'state_keys': [states[i]['pair_key'] for i in selected]}
    with PdfPages(out / 'view_landscape_report.pdf') as pdf:
        fig = plt.figure(figsize=(15, 8.5))
        title(fig, 'View landscape: indicator values and closed-loop behavior',
              f'{args.model_label} | {len(states)} states / 8 known tasks | 97 irregular poses | Accel: 8 noise samples | success: 32 O-bank rollouts')
        ax1 = fig.add_axes([.045, .24, .42, .60], projection='3d')
        ax2 = fig.add_axes([.53, .24, .42, .60], projection='3d')
        a = scatter3d(ax1, coords, rates.mean(0), 'Mean closed-loop success (all states)')
        raw_mean = means.mean(0)
        b = scatter3d(ax2, coords, raw_mean, 'Raw Accel mean (observed-mean color range)',
            lo=float(raw_mean.min()), hi=float(raw_mean.max()) + 1e-12, cmap='magma')
        fig.colorbar(a, ax=ax1, shrink=.55, pad=.08); fig.colorbar(b, ax=ax2, shrink=.55, pad=.08)
        rho = summary['all']['spearman_negative_accel_success']
        interval_text = (f"Descriptive source-bootstrap 95% interval [{number(rho['ci95'][0])}, {number(rho['ci95'][1])}]"
            if rho['ci95'][0] is not None else 'Fixed-source diagnostic: source-generalization interval unsupported.')
        fig.text(.05, .13, f"Within-state rank association: mean rho(-Accel, success) = {number(rho['mean'])}\n" + interval_text, fontsize=12)
        fig.text(.54, .13, 'Star = canonical: original orientation, not a regular look-at sample.\nRaw Accel color range is local to this panel; rank maps follow.\nScoring and rollout noise are independent, not sample-wise paired.', fontsize=10)
        save_page(fig, '01_spatial_overview', out, pdf)

        i = selected[0]
        fig, axes = plt.subplots(3, 2, figsize=(15, 8.5))
        title(fig, 'Radius bands: observed samples, not interpolated slices',
              f"Fixed example: {TASK_LABELS.get(states[i]['task_id'])} / {states[i]['demo_name']} / frame {states[i]['source_state_index']}")
        edges = [.9, 1.0166666667, 1.1333333333, 1.250000001]
        for band in range(3):
            mask = (coords[:, 2] >= edges[band]) & (coords[:, 2] < edges[band + 1]); mask[0] = False
            for col, values in enumerate((rates[i], normalized_rank[i])):
                ax = axes[band, col]
                sc = ax.scatter(coords[mask, 0], coords[mask, 1], c=values[mask], vmin=0, vmax=1,
                    cmap='viridis' if col == 0 else 'magma', s=65, edgecolor='white', linewidth=.4)
                if edges[band] <= 1 < edges[band + 1]:
                    ax.scatter(0, 0, c=[values[0]], vmin=0, vmax=1, cmap='viridis' if col == 0 else 'magma', marker='*', s=180, edgecolor='black')
                ax.set(xlim=(-60, 60), ylim=(-25, 25), xlabel='Azimuth offset (deg)', ylabel='Elevation offset (deg)',
                    title=f"{'Success' if col == 0 else 'Accel rank'} | radius [{edges[band]:.3f}, {edges[band+1]:.3f}) | n={mask.sum()}")
                fig.colorbar(sc, ax=ax, fraction=.025)
        fig.subplots_adjust(left=.065, right=.95, bottom=.09, top=.82, hspace=.75, wspace=.28)
        save_page(fig, '02_radius_bands', out, pdf)

        fig, axes = plt.subplots(1, 2, figsize=(15, 8.5))
        title(fig, 'The complete state x candidate map',
              'Same outcome-independent order in both panels: tasks / split / source; candidate 0 is canonical, others sorted by azimuth / elevation / radius.')
        for ax, values, heading, cmap in zip(axes, (rates, normalized_rank), ('Closed-loop success', 'Within-state Accel rank (lower preferred)'), ('viridis', 'magma')):
            im = ax.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect='auto', interpolation='nearest')
            tasks = sorted({s['task_id'] for s in states})
            task_rows = [[j for j, s in enumerate(states) if s['task_id'] == t] for t in tasks]
            positions = [rows[0] for rows in task_rows]
            ax.set_yticks([np.mean(rows) for rows in task_rows], [TASK_LABELS[t] for t in tasks], fontsize=8)
            for p in positions[1:]: ax.axhline(p - .5, color='white', lw=.7)
            for rows in task_rows:
                for j in rows[1:]:
                    if states[j]['split'] != states[j-1]['split']: ax.axhline(j-.5, color='white', lw=.5, ls=':')
            ax.axvline(.5, color='white', lw=1)
            ax.set(xlabel='Frozen geometry order (97 candidates)', title=heading)
            fig.colorbar(im, ax=ax, fraction=.03)
        fig.subplots_adjust(left=.16, right=.95, bottom=.19, top=.81, wspace=.49)
        fig.text(.17, .062, 'Dotted boundaries separate old development / test rows when both exist; all present rows are exploratory here.', fontsize=9)
        save_page(fig, '03_state_candidate_matrices', out, pdf)

        fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
        title(fig, 'Indicator association and noise stability are different questions',
              'Rank association is measured within each state; a stable ranking can still rank unhelpful views highly. Undefined constant-outcome states are excluded from rho only.')
        metrics = [('spearman_negative_accel_success', 'rho(-Accel, success)', (-1, 1)),
                   ('noise_mean_pairwise_rank_spearman', 'Across-noise rank Spearman', (-1, 1)),
                   ('noise_mean_pairwise_top5_jaccard', 'Across-noise top-5 Jaccard', (0, 1)),
                   ('accel_gain_vs_canonical_pp', 'Accel top-1 minus canonical success (pp)', (-100, 100))]
        for ax, (key, label, limits) in zip(axes.flat, metrics):
            for split, color in (('development', '#3284aa'), ('test', '#d58a37')):
                vals = [s[key] for s in stats if s['split'] == split and np.isfinite(s[key])]
                ax.hist(vals, bins=np.linspace(*limits, 17), alpha=.6, label=f'Historical {split}', color=color)
            ax.axvline(0, color='#334155', lw=1); ax.set(xlabel=label, ylabel='States', xlim=limits)
            point = summary['all'][key]
            ax.set_title(estimate_label(point), fontsize=11)
        axes[0, 0].legend(frameon=False, fontsize=9)
        fig.subplots_adjust(left=.07, right=.96, bottom=.20, top=.80, hspace=.65, wspace=.26)
        agreement = sum(s['noise_top1_all_agree'] for s in stats)
        fig.text(.07, .060, f'The 8 noise samples choose an identical top-1 in {agreement}/{len(states)} states. Accel top-1 includes the keep/canonical option.\nThese O-bank effects are descriptive reuse, not a newly frozen selector confirmation.', fontsize=9)
        save_page(fig, '04_rank_noise_diagnostics', out, pdf)

        fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
        title(fig, 'Training-support geometry is a proxy, not measured familiarity',
              f'Distances use range-normalized azimuth / elevation / radius. Model post-training support: {args.training_support}; nearest-support correlations use its unseen noncanonical poses.')
        x1 = np.array([p['distance_canonical_parameter_normalized'] for p in poses])
        x2 = np.array([p['distance_model_support_parameter_normalized'] for p in poses])
        masks = (np.arange(97) > 0, np.array([i > 0 and not p['is_model_post_training_support'] for i, p in enumerate(poses)]))
        for row, (x, mask, name) in enumerate(zip((x1, x2), masks, ('Distance to canonical parameter origin', f'Nearest model-support distance (unseen n={masks[1].sum()})'))):
            for col, (values, label) in enumerate(((rates, 'Success rate'), (normalized_rank, 'Accel within-state rank'))):
                ax = axes[row, col]
                ax.scatter(np.tile(x[mask], len(states)), values[:, mask].ravel(), s=7, alpha=.075, color='#3b82a0', rasterized=True)
                ax.scatter(x[mask], values[:, mask].mean(0), s=22, color='#a54950', label='Across-state mean at each sampled pose')
                ax.set(xlabel=name, ylabel=label, ylim=(-.03, 1.03))
                key = ('spearman_canonical_distance_' if row == 0 else 'spearman_train_distance_') + ('success' if col == 0 else 'accel') + ('_heldout' if row else '')
                metric = summary['all'][key]
                ax.set_title(estimate_label(metric), fontsize=11)
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.subplots_adjust(left=.07, right=.96, bottom=.20, top=.80, hspace=.65, wspace=.25)
        fig.text(.07, .055, 'Training-support poses have nearest-support distance zero by construction, and are excluded from that correlation.\nThis is post-training pose support, not pretraining familiarity; evidence, occlusion and actual geometry can confound these relations.', fontsize=9)
        save_page(fig, '05_support_geometry', out, pdf)

        fig, axes = plt.subplots(4, 4, figsize=(16, 10))
        title(fig, 'One preselected development source from each task',
              f'Paired panels: success (viridis) and Accel rank (magma), both scaled 0-1. Marker size encodes radius; star is canonical. Full {len(states)}-state atlas is separate.')
        for k, i in enumerate(selected):
            for c, values in enumerate((rates[i], normalized_rank[i])):
                ax = axes[k // 2, 2 * (k % 2) + c]
                ax.scatter(coords[1:, 0], coords[1:, 1], c=values[1:], vmin=0, vmax=1, cmap='viridis' if c == 0 else 'magma',
                    s=9 + (coords[1:, 2] - .9) * 55, alpha=.95)
                ax.scatter(0, 0, c=[values[0]], vmin=0, vmax=1, cmap='viridis' if c == 0 else 'magma', marker='*', s=95, edgecolor='black')
                ax.set(xlim=(-60, 60), ylim=(-25, 25), xlabel='Azimuth', ylabel='Elevation')
                ax.set_title(f"{TASK_LABELS[states[i]['task_id']]}\n{states[i]['demo_name']} | {'Success' if c == 0 else 'Accel rank'}", fontsize=9)
                ax.tick_params(labelsize=8)
        fig.subplots_adjust(left=.055, right=.98, bottom=.08, top=.83, hspace=.92, wspace=.38)
        save_page(fig, '06_fixed_representatives', out, pdf)

    with PdfPages(out / 'view_landscape_all_states_atlas.pdf') as pdf:
        for i, state in enumerate(states):
            fig = plt.figure(figsize=(14, 9))
            title(fig, f"{TASK_LABELS[state['task_id']]} | {state['demo_name']} | frame {state['source_state_index']}",
                f"Historical {state['split']} | {state['pair_key']}")
            ax1 = fig.add_axes([.04, .43, .40, .42], projection='3d')
            ax2 = fig.add_axes([.54, .43, .40, .42], projection='3d')
            sc = scatter3d(ax1, coords, rates[i], 'Success / 32 O-bank repeats'); fig.colorbar(sc, ax=ax1, shrink=.6, pad=.05)
            sc = scatter3d(ax2, coords, means[i], 'Raw Accel mean / 8 scoring noises',
                lo=float(means[i].min()), hi=float(means[i].max()) + 1e-12, cmap='magma'); fig.colorbar(sc, ax=ax2, shrink=.6, pad=.05)
            ax3 = fig.add_axes([.08, .105, .35, .23]); ax4 = fig.add_axes([.58, .105, .35, .23])
            ax3.scatter(normalized_rank[i], rates[i], s=18, c=coords[:, 2], cmap='cividis')
            ax3.scatter(normalized_rank[i, 0], rates[i, 0], s=130, marker='*', c='#b45309', edgecolor='black')
            ax3.set(xlabel='Accel rank (low preferred)', ylabel='Success rate', xlim=(-.03, 1.03), ylim=(-.03, 1.03),
                title=f"Within-state rho(-Accel, success) = {stats[i]['spearman_negative_accel_success']:+.3f}")
            order = np.argsort(means[i], kind='stable')
            for m in range(accel.shape[2]): ax4.plot((ranks(accel[i, :, m])[order] - 1) / 96, alpha=.45, lw=.75)
            ax4.set(xlabel='Candidates ordered by mean Accel', ylabel='Per-noise rank / 96', ylim=(-.03, 1.03),
                title=f"{stats[i]['noise_top1_unique_count']} different top-1 views / 8 noises")
            fig.text(.07, .055, 'Star: canonical original orientation. Raw Accel color range is local to each state: compare colorbar values, not colors across pages.\nNo interpolation. Each success-rate estimate has 32 repeats; Wilson intervals and raw scoring values are in CSV.', fontsize=8)
            slug = hashlib.sha256(state['pair_key'].encode()).hexdigest()[:12]
            fig.savefig(out / 'state_figures' / f'{i:02d}_{slug}.png', dpi=110)
            pdf.savefig(fig); plt.close(fig)
    return [str(out / 'view_landscape_report.pdf'), str(out / 'view_landscape_all_states_atlas.pdf')]


def render_markdown(args: argparse.Namespace, summary: dict) -> str:
    metrics = summary['all']
    def display(key: str) -> str:
        item = metrics[key]
        if item['ci95'][0] is None:
            return f"{number(item['mean'])}（固定来源描述；每任务来源不足，不提供来源泛化 CI）"
        return f"{item['mean']:+.3f}（描述性 95% CI [{item['ci95'][0]:+.3f}, {item['ci95'][1]:+.3f}]）"
    support_name = '规范位姿 1 点' if args.training_support == 'canonical' else 'Broad64 的 64 个位姿'
    unseen_count = 96 if args.training_support == 'canonical' else 32
    return f"""# 视角空间诊断（2026-09-08）

本报告是已有数据的 CPU 只读重分析，不是新闭环实验，不修改原 10 页汇报。模型：`{args.model_label}`。

## 数据与证据边界

- {summary['state_count']} 个构造遮挡中间状态、8 个已知任务；每状态 97 个非规则 Halton 候选，32 次 O-bank 闭环重复；Accel 为独立的 8 个评分初噪。
- 97 点：规范 1、Broad64 参照组 64、该参照下未见组 32。**本模型后训练支持为 {support_name}**；未见非规范候选为 {unseen_count} 点。图 5 使用本模型的支持距离，而非一律把 Broad64 当成本模型训练集。方位偏移 ±60°、仰角偏移 ±25°、半径比例 0.9–1.25；不是完整 SE(3)，也没有覆盖 wide24 外推点。
- 规范点保留原朝向，其余点朝向 pivot。图中规范点用星号单列；无连续插值，半径图是分带散点而非精确二维切面。
- 第 1 页和逐状态图册提供 Accel 的实际 8 噪声均值，色标按各图实际范围显示；不能跨页／跨模型只按颜色比较数值。矩阵、半径分带和代表来源页使用状态内排名，便于检验对应关系。
- 全部状态的物理状态与图像内容 hash、候选身份、O-bank 与运行 manifest 已核对。O-bank 的逐调用噪声一致性沿用已完成的原始审计。
- Accel 旧记录未绑定权重 hash：加载日志证实同一 checkpoint 路径，当前权重内容及 rollout manifest 完整校验；这不能补造历史 cryptographic binding。此限制保留在 provenance.json。
- 原开发／测试划分在此均标为**历史探索**。不能看完这些图再在同一数据上宣称新方法得到独立确认。

## 全体诊断结果

| 指标 | 全体结果 |
|---|---|
| 状态内 ρ(−Accel, 成功率)，正值表示低 Accel 偏向高成功率 | {display('spearman_negative_accel_success')} |
| 不同评分初噪的排名 Spearman | {display('noise_mean_pairwise_rank_spearman')} |
| 不同评分初噪的 Top-5 Jaccard | {display('noise_mean_pairwise_top5_jaccard')} |
| Accel 最小值候选减规范视角成功率（百分点） | {display('accel_gain_vs_canonical_pp')} |
| O16↔O16 双向交叉拟合搜索成功率（0–1） | {display('noise_crossfit_search_success')} |
| O16↔O16 搜索减规范视角成功率（百分点） | {display('noise_crossfit_gain_vs_canonical_pp')} |
| 状态内 ρ(规范参数距离, 成功率)，不含规范点 | {display('spearman_canonical_distance_success')} |
| 状态内 ρ(最近本模型后训练支持距离, 成功率)，仅 {unseen_count} 个未见非规范候选 | {display('spearman_train_distance_success_heldout')} |

8 组初噪 Top-1 完全一致：{summary['noise_top1_all_agree_count']}/{summary['state_count']} 状态。只有每个任务都至少有两个有限来源时，才提供固定任务条件下、任务等权／来源聚类的描述性 bootstrap；每任务仅一个来源的首轮 8 状态**不提供来源泛化 CI**。区间不额外重采样评分与 rollout 噪声的测量误差。常数成功率状态的相关系数不定义，并从该指标的均值排除，数量在 summary.json 中报告。

交叉拟合诊断：前 16 次噪声在全部 97 候选（**包括保持规范视角**）中选经验 Top-1，后 16 次评价；再反向执行并取均值。并列按固定几何序打破，规范点优先。它降低用同一噪声挑赢家又评赢家的偏差，但仍是有限样本、历史探索的搜索程序，**不是 Oracle 真上界，也没有使用每任务固定基线**。

## 如何阅读而不过度推断

1. 空间图回答“指标与闭环成功率怎样分布”，不直接回答模型内部是否真的理解视角、是否具备主动搜索能力。
2. 评分排名的稳定性与评分是否有用是两件事；必须同时报告相关、候选收益和噪声稳定性。
3. 几何距离只是可复现的后训练支持代理，不等于“熟悉度”，也没有定义预训练见过的位姿。遮挡、任务证据、物体尺度和真实姿态差异都可能共同变化。训练点最近支持距离恒为零，所以相关只在本模型未见的 {unseen_count} 个非规范候选内计算。
4. 每候选成功率只有 32 次重复。O-bank 最大值是有选择偏差的经验最大值；不能当精确 Oracle 上限。单元 Wilson 区间在 CSV，绝不以插值制造连续空间精度。
5. 这批是构造遮挡的中途续跑，不等同自然场景初始相机扰动后的恢复验证；不能直接证明主动移动收益或用户拟贡献 1。
6. 本报告单独描述一个 checkpoint，不能自行证明训练交互。单规范／宽视角模型必须按相同来源子集、候选和噪声比较，不能以 Broad 的 64 状态均值对另一个模型的 8 状态均值直接相减。选择指标随后还需冻结，并在独立来源／噪声上检验可执行搜索及获取成本。

## 输出

- [6 页图表报告]({args.output_dir}/view_landscape_report.pdf)
- [{summary['state_count']} 状态完整图册]({args.output_dir}/view_landscape_all_states_atlas.pdf)
- [状态×候选明细]({args.output_dir}/state_view_metrics.csv)
- [状态级统计]({args.output_dir}/state_metrics.csv)
- [汇总与区间]({args.output_dir}/summary.json)
- [来源、hash 与审计范围]({args.output_dir}/provenance.json)

示例选择完全不使用 outcome：每任务按 source_group 字典序及帧序选首个开发来源。全部 {summary['state_count']} 状态均已出图，避免只展示正例。逐状态×候选×32 个噪声结果与逐候选×8 个 Accel 值保存在 `verified_arrays.npz`，其 axis 身份数组一并保存。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--population', type=Path, default=EXPERIMENTS / 'dsol-statewise-view-oracle-v2/population/population-v2.json')
    parser.add_argument('--accel-root', type=Path, default=EXPERIMENTS / 'dsol-view-value-expectation-v1/accel-ensemble')
    parser.add_argument('--rollout-root', type=Path, default=EXPERIMENTS / 'dsol-statewise-view-oracle-v2')
    parser.add_argument('--dense-dir', type=Path, help='Alternative direct O segment directory; loads every child with audit.json, including smoke/remainder segments.')
    parser.add_argument('--noise-bank-manifest', type=Path, help='Explicit shared bank-O manifest; useful when new rollout root does not own the original noise bank.')
    parser.add_argument('--catalog', type=Path, default=REPO / 'configs/dsol_paper1/libero_view_catalog_v2_m1.json')
    parser.add_argument('--catalog-rules', type=Path, default=REPO / 'configs/dsol_paper1/libero_view_catalog_v2_m1_rules.json')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--expected-checkpoint-sha256', default=DEFAULT_HASH)
    parser.add_argument('--model-label', default='Broad M-B / seed41')
    parser.add_argument('--training-support', choices=('broad64', 'canonical'), default='broad64', help='Actual post-training pose support for this checkpoint; catalog train64/heldout32 labels are always Broad reference groups.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--markdown-output', type=Path)
    parser.add_argument('--reference-summary', type=Path, help='Optional existing model summary for manifest-linked, separate-model comparison; does not pool model data.')
    parser.add_argument('--selection-manifest', type=Path, help='Explicit selected_state_keys/state_keys or selected_states[].pair_key. Requires all selected 97x32 cells; never treats missing data as zero.')
    parser.add_argument('--reuse-verified-cache', action='store_true', help='Reuse previously verified aggregate only if all inputs retain path,size,mtime; records this weaker rerender scope.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    poses, coords = catalog_bank(read_json(args.catalog), read_json(args.catalog_rules), args.training_support)
    cache, receipt_path = args.output_dir / 'verified_arrays.npz', args.output_dir / 'provenance.json'
    if args.reuse_verified_cache and cache.exists() and receipt_path.exists():
        provenance = read_json(receipt_path)
        require(provenance['checkpoint_sha256'] == args.expected_checkpoint_sha256, 'Cache belongs to another checkpoint')
        require(provenance.get('selection_manifest_sha256') == (sha256(args.selection_manifest) if args.selection_manifest else None), 'Cache belongs to another state selection')
        for item in provenance['file_stats']:
            stat = Path(item['path']).stat()
            require(stat.st_size == item['size'] and stat.st_mtime_ns == item['mtime_ns'], 'Input changed since verified cache')
        require(sha256(cache) == provenance['cache_sha256'], 'Aggregate cache hash mismatch')
        states = read_json(args.output_dir / 'states.json')['states']
        with np.load(cache, allow_pickle=False) as data:
            require(list(data['state_keys']) == [s['pair_key'] for s in states], 'Cache state order mismatch')
            require(list(data['candidate_ids']) == [p['pose_id'] for p in poses], 'Cache candidate order mismatch')
            accel, success, visibility = data['accel'], data['success'], data['visibility']
        provenance['last_render_mode'] = 'cached; unchanged stat metadata plus aggregate content hash (original full hashes retained)'
        provenance['last_render_script_sha256'] = sha256(Path(__file__))
    else:
        print(json.dumps({'stage': 'checkpoint_hash', 'checkpoint': str(args.checkpoint)}), flush=True)
        checkpoint_hash = sha256(args.checkpoint / 'model.safetensors')
        require(checkpoint_hash == args.expected_checkpoint_sha256, 'Current checkpoint weight hash mismatch')
        print(json.dumps({'stage': 'static_assets'}), flush=True)
        states, accel, static = load_static_assets(args, poses)
        success, visibility, dense = load_dense(args, states, poses, checkpoint_hash)
        provenance = {'schema': 'dsol_view_landscape_provenance_v1', 'status': 'PASS_DESCRIPTIVE_JOIN_WITH_LEGACY_PROVENANCE_LIMITATION',
            'built_at_utc': datetime.now(timezone.utc).isoformat(), 'model_label': args.model_label,
            'checkpoint': str(args.checkpoint), 'checkpoint_sha256': checkpoint_hash, 'static_assets': static, 'dense': dense,
            'score_noise_vs_rollout_noise': 'Independent noise banks; not sample-paired. This report estimates view-level distribution relationships.',
            'script': str(Path(__file__).resolve()), 'script_sha256': sha256(Path(__file__)),
            'population_sha256': sha256(args.population), 'catalog_sha256': sha256(args.catalog), 'catalog_rules_sha256': sha256(args.catalog_rules)}
        provenance['selection_manifest_sha256'] = sha256(args.selection_manifest) if args.selection_manifest else None
        np.savez_compressed(cache, accel=accel, success=success, visibility=visibility, candidate_ids=np.array([p['pose_id'] for p in poses]), state_keys=np.array([s['pair_key'] for s in states]))
        write_json(args.output_dir / 'states.json', {'states': states})
        paths = {Path(x['path']) for x in static['inputs'] + dense['inputs']}
        noise_path = args.noise_bank_manifest or args.rollout_root / 'noise-banks/bank_O.manifest.json'
        paths.update((args.population, args.catalog, args.catalog_rules, args.checkpoint / 'model.safetensors', noise_path, Path(read_json(noise_path)['noise_file'])))
        if args.selection_manifest: paths.add(args.selection_manifest)
        provenance['file_stats'] = [{'path': str(p), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in sorted(paths)]
        provenance['cache_sha256'] = sha256(cache)
        write_json(receipt_path, provenance)
    stats, cells = state_metrics(states, poses, accel, success, visibility)
    keys = [k for k, v in stats[0].items() if isinstance(v, (float, np.floating))]
    summary = {'schema': 'dsol_view_landscape_summary_v1', 'model_label': args.model_label, 'state_count': len(states),
        'model_post_training_support': args.training_support,
        'candidate_count': len(poses), 'accel_noise_count': accel.shape[2], 'rollout_noise_count': success.shape[2],
        'episode_count': success.size, 'noise_top1_all_agree_count': sum(r['noise_top1_all_agree'] for r in stats),
        'historical_exploratory_only': True, 'confirmatory_method_result': False,
        'all': {k: clustered_summary(stats, k) for k in keys},
        'by_historical_split': {split: {k: clustered_summary([r for r in stats if r['split'] == split], k) for k in keys} for split in ('development', 'test')},
        'undefined_negative_accel_success_correlations': sum(not np.isfinite(r['spearman_negative_accel_success']) for r in stats)}
    if args.reference_summary:
        reference = read_json(args.reference_summary)
        summary['comparison_reference'] = {'path': str(args.reference_summary), 'sha256': sha256(args.reference_summary),
            'model_label': reference['model_label'], 'note': 'Separate checkpoint summary; paired model comparisons require matching state/candidate/noise identities, not subtraction of pooled figures.'}
    write_csv(args.output_dir / 'state_metrics.csv', stats)
    write_csv(args.output_dir / 'state_view_metrics.csv', cells)
    write_json(args.output_dir / 'candidate_geometry.json', {'poses': poses, 'order': 'canonical first; noncanonical azimuth/elevation/radius lexicographic'})
    setup_plotting()
    print(json.dumps({'stage': 'plot', 'states': len(states)}), flush=True)
    summary['pdfs'] = plot_reports(args, states, poses, coords, accel, success, visibility, stats, summary)
    write_json(args.output_dir / 'summary.json', summary)
    write_json(receipt_path, provenance)
    markdown = render_markdown(args, summary)
    (args.output_dir / 'README_zh.md').write_text(markdown)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown)
    print(json.dumps({'status': 'PASS_COMPLETE', 'output_dir': str(args.output_dir), 'summary': summary['all']['spearman_negative_accel_success']}), flush=True)


if __name__ == '__main__':
    main()
