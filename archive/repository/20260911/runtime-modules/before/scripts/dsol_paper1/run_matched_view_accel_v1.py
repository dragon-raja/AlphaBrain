#!/usr/bin/env python3
"""Score frozen cached observations with the matched canonical checkpoint.

The legacy Accel seed key is retained deliberately. No rendering and no writes
to the historical asset/ranking directories are performed.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import yaml

try:
    from .run_view_value_expectation_accel_ensemble import (
        append_jsonl, rank_state, sha256_file, stable_seed,
    )
except ImportError:
    from run_view_value_expectation_accel_ensemble import (
        append_jsonl, rank_state, sha256_file, stable_seed,
    )

ROOT = Path('/share/longjunyu/alphabrain/experiments/dsol-view-landscape-v1-20260908')
EXPECTED_WEIGHT_HASH = '7e510b752143bb6fd4987988dfab5e94f7db1de5f7e97579f892a58eed22cc68'
REPO = Path(__file__).resolve().parents[2]
ENSEMBLE_SIZE = 8
ROOT_SEED = 20260921
BATCH_SIZE = 16
EXPECTED_CANDIDATES = {'canonical', *(f'broad_train_{i:03d}' for i in range(64)),
                       *(f'broad_heldout_{i:03d}' for i in range(32))}


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def identity(path: Path) -> dict[str, Any]:
    return {'path': str(path.resolve()), 'sha256': sha256_file(path)}


def code_identities() -> list[dict[str, Any]]:
    paths = [Path(__file__), Path(__file__).with_name('run_view_value_expectation_accel_ensemble.py'),
             Path(__file__).with_name('accel_inference.py'), Path(__file__).with_name('accel_core.py'),
             REPO / 'AlphaBrain/model/framework/PaliGemmaPi.py',
             REPO / 'AlphaBrain/model/framework/base_framework.py',
             REPO / 'AlphaBrain/model/framework/config_utils.py',
             REPO / 'AlphaBrain/model/modules/action_model/pi0_flow_matching_head/pi0_action_head.py',
             REPO / 'AlphaBrain/model/modules/action_model/pi0_flow_matching_head/pi0_transforms.py',
             REPO / 'AlphaBrain/model/modules/action_model/pi0_flow_matching_head/openpi_inference.py']
    return [identity(path) for path in paths]


def seed_list(legacy_key: str) -> list[int]:
    return [stable_seed(f'accel::{legacy_key}::{member}', ROOT_SEED) for member in range(ENSEMBLE_SIZE)]


def validate_ranking_matrix(row: dict[str, Any]) -> None:
    ranking = row.get('ranking', [])
    if len(ranking) != 97 or {r['candidate_id'] for r in ranking} != EXPECTED_CANDIDATES:
        raise ValueError('ranking must contain 97 distinct frozen candidate IDs')
    for candidate in ranking:
        scores = np.asarray(candidate.get('member_accel_3', []), dtype=np.float64)
        if scores.shape != (8,) or not np.all(np.isfinite(scores)) or np.any(scores < 0):
            raise ValueError('ranking must contain eight finite nonnegative Accel values per candidate')
        for key, expected in (('mean_accel_3', float(np.mean(scores))),
                              ('std_accel_3', float(np.std(scores)))):
            if not math.isclose(float(candidate.get(key, math.nan)), expected, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f'ranking {key} does not match its member scores')
    expected_order = sorted(ranking, key=lambda candidate: (candidate['mean_accel_3'], candidate['candidate_id']))
    if ranking != expected_order or row.get('selected_candidate_id') != ranking[0]['candidate_id']:
        raise ValueError('ranking order/selected candidate is inconsistent')


def checkpoint_small_files(checkpoint: Path) -> list[dict[str, Any]]:
    paths = [checkpoint / 'framework_config.yaml']
    paths.extend(sorted((checkpoint / 'vlm_pretrained').glob('*.json')))
    if (checkpoint / 'dataset_statistics.json').exists():
        paths.append(checkpoint / 'dataset_statistics.json')
    return [identity(path) for path in paths]


def validate_model_config(checkpoint: Path) -> None:
    framework = yaml.safe_load((checkpoint / 'framework_config.yaml').read_text())['framework']
    if framework.get('name') != 'PaliGemmaPi05' or not framework.get('pi05'):
        raise ValueError('only the released PaliGemmaPi05 framework is allowed')
    for key, value in {'action_dim': 7, 'action_horizon': 10, 'state_dim': 8,
                       'num_inference_steps': 10}.items():
        if framework['action_model'].get(key) != value:
            raise ValueError(f'checkpoint inference configuration differs at {key}')
    if framework.get('gripper_remap') is not False:
        raise ValueError('checkpoint gripper remapping is not the released setting')


def validate_output_root(output: Path) -> None:
    if output.resolve().parent != ROOT.resolve() or not output.name.startswith('canonical-accel'):
        raise ValueError('Accel output must be a canonical-accel child of the new landscape root')
    if output.is_symlink():
        raise ValueError('symlink output roots are not allowed')


def validate_assets(state: dict[str, Any]) -> dict[str, Any]:
    assets = state['static_assets']
    receipt_path = Path(assets['render_receipt'])
    artifact = Path(assets['policy_inputs'])
    for name in ('render_receipt', 'policy_inputs'):
        if sha256_file(Path(assets[name])) != assets[name + '_sha256']:
            raise ValueError(f'cached {name} hash changed: {state["pair_key"]}')
    render = read(receipt_path)
    if render.get('status') != 'PASS' or render['candidate_count'] != 97:
        raise ValueError('render receipt did not pass full candidate audit')
    if render['pair_key'] != state['asset_source_pair_key']:
        raise ValueError('legacy pair key mismatch; this would change matched Accel noise')
    if Path(render['artifact']).resolve() != artifact.resolve():
        raise ValueError('render artifact path mismatch')
    if render['artifact_sha256'] != assets['policy_inputs_sha256']:
        raise ValueError('render artifact checksum mismatch')
    if render['physics_state_sha256'] != assets['physics_state_sha256']:
        raise ValueError('cached physical state mismatch')
    for key in ('task_id', 'source_group'):
        if render.get(key) != state[key]:
            raise ValueError(f'cached render {key} mismatch')
    with np.load(artifact, allow_pickle=False) as values:
        ids = [str(value) for value in values['candidate_ids']]
        if len(ids) != 97 or set(ids) != EXPECTED_CANDIDATES:
            raise ValueError('cached inputs do not contain 97 distinct frozen candidates')
        for key in ('external_images', 'wrist_images'):
            array = values[key]
            if array.shape != (97, 224, 224, 3) or array.dtype != np.uint8:
                raise ValueError(f'cached {key} image shape/dtype mismatch')
        for key, shape in (('robot_states', (97, 8)), ('camera_intrinsics', (97, 3, 3)),
                            ('camera_to_world_opencv', (97, 4, 4))):
            array = values[key]
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f'cached {key} shape/finite-value mismatch')
    broad_path = receipt_path.with_name('ranking.json')
    broad = read(broad_path)
    if broad.get('status') != 'PASS' or broad.get('pair_key') != state['asset_source_pair_key']:
        raise ValueError('historical Broad rank status/pair identity mismatch')
    for key in ('task_id', 'source_group'):
        if broad.get(key) != state[key]:
            raise ValueError(f'historical Broad {key} mismatch')
    if broad['ensemble_seeds'] != seed_list(state['asset_source_pair_key']):
        raise ValueError('historical Broad Accel noise does not match frozen seeds')
    if broad['ensemble_size'] != 8 or broad['candidate_count'] != 97:
        raise ValueError('historical Broad rank is not 97 x 8')
    if broad['render_artifact_sha256'] != assets['policy_inputs_sha256']:
        raise ValueError('Broad was scored on different images')
    if broad['physics_state_sha256'] != assets['physics_state_sha256']:
        raise ValueError('Broad physical state mismatch')
    if broad.get('checkpoint_action_horizon') != 10:
        raise ValueError('Broad action horizon differs from the released scorer')
    validate_ranking_matrix(broad)
    return identity(broad_path)


def prepare(selection_path: Path, receipt_path: Path, output: Path) -> Path:
    validate_output_root(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('refusing to prepare into a nonempty Accel output directory')
    selection = read(selection_path)
    if selection.get('status') != 'FROZEN_OUTCOME_BLIND_DEVELOPMENT_SUBSET':
        raise ValueError('selection must be outcome-blind frozen development subset')
    states = selection['states']
    if (len(states) != 8 or len({s['task_id'] for s in states}) != 8
            or len({s['pair_key'] for s in states}) != 8
            or len({s['asset_source_pair_key'] for s in states}) != 8
            or len({s['source_group'] for s in states}) != 8):
        raise ValueError('this release is bounded to one state per eight tasks')
    if any(s['split'] != 'development' or s.get('v2_role') != 'development' for s in states):
        raise ValueError('test states are not released')
    receipt = read(receipt_path)
    if receipt.get('status') != 'COMPLETED_ARTIFACT_AUDIT_PASS':
        raise ValueError('checkpoint artifact audit must pass')
    weights = Path(receipt['final_weights']['path'])
    if receipt['final_weights']['content_sha256'] != EXPECTED_WEIGHT_HASH:
        raise ValueError('unreleased checkpoint')
    validate_model_config(weights.parent)
    stat_before = weights.stat()
    if sha256_file(weights) != EXPECTED_WEIGHT_HASH:
        raise ValueError('checkpoint hash changed since acceptance')
    stat = weights.stat()
    if (stat_before.st_size, stat_before.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise ValueError('checkpoint changed during hash verification')
    broad_identities = [validate_assets(state) for state in states]
    manifest = {
        'schema': 'dsol_matched_cached_accel_manifest_v1', 'status': 'PASS_FROZEN',
        'scope': 'eight-state development diagnostic; static counterfactual images are provided, not acquired',
        'selection': identity(selection_path), 'checkpoint_acceptance': identity(receipt_path),
        'checkpoint': str(weights.parent), 'checkpoint_sha256': EXPECTED_WEIGHT_HASH,
        'checkpoint_bytes': stat.st_size, 'checkpoint_mtime_ns': stat.st_mtime_ns,
        'framework_config': identity(weights.parent / 'framework_config.yaml'),
        'checkpoint_small_files': checkpoint_small_files(weights.parent),
        'code': code_identities(), 'historical_broad_rankings': broad_identities,
        'states': sorted(states, key=lambda s: s['asset_source_pair_key']),
        'state_count': 8, 'candidate_count': 97, 'ensemble_size': ENSEMBLE_SIZE,
        'root_seed': ROOT_SEED, 'batch_size': BATCH_SIZE, 'action_horizon': 10,
        'num_shards': 8, 'flow_denoising_steps': 10,
        'action_dim': 7, 'dtype': 'bfloat16', 'gripper_remap': False,
        'seed_key': 'asset_source_pair_key (exact historical Broad Accel seed identity)',
        'noise_relationship': 'same Accel eight-member bank across checkpoints; independent of closed-loop O32',
        'historical_comparison_limit': 'legacy score receipts lack weight/code hashes; their policy provenance must be established separately, not inferred from matched seeds',
        'output_root': str(output.resolve()),
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'manifest.json'
    with path.open('x') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return path


def validate_result(row: dict[str, Any], state: dict[str, Any], manifest_sha: str) -> None:
    expected = {
        'status': 'PASS', 'pair_key': state['asset_source_pair_key'],
        'population_pair_key': state['pair_key'], 'manifest_sha256': manifest_sha,
        'checkpoint_sha256': EXPECTED_WEIGHT_HASH, 'candidate_count': 97,
        'ensemble_size': 8, 'ensemble_seeds': seed_list(state['asset_source_pair_key']),
        'render_artifact_sha256': state['static_assets']['policy_inputs_sha256'],
        'physics_state_sha256': state['static_assets']['physics_state_sha256'],
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f'incompatible existing rank at {key}; refusing to overwrite')
    validate_ranking_matrix(row)


def run(manifest_path: Path, shard_index: int, num_shards: int, device: str) -> None:
    if not 0 <= shard_index < num_shards <= 8:
        raise ValueError('invalid shard allocation')
    manifest = read(manifest_path)
    if manifest.get('status') != 'PASS_FROZEN' or manifest['code'] != code_identities():
        raise ValueError('manifest/code identity mismatch')
    if num_shards != manifest.get('num_shards'):
        raise ValueError('shard allocation differs from the frozen manifest')
    if manifest['checkpoint_sha256'] != EXPECTED_WEIGHT_HASH:
        raise ValueError('unreleased checkpoint')
    weights = Path(manifest['checkpoint']) / 'model.safetensors'
    if (weights.stat().st_size, weights.stat().st_mtime_ns) != (
        manifest['checkpoint_bytes'], manifest['checkpoint_mtime_ns']
    ):
        raise ValueError('weights changed after full-hash preflight')
    if identity(Path(manifest['framework_config']['path'])) != manifest['framework_config']:
        raise ValueError('checkpoint configuration changed')
    if checkpoint_small_files(weights.parent) != manifest['checkpoint_small_files']:
        raise ValueError('checkpoint tokenizer/configuration identity changed')
    for key in ('selection', 'checkpoint_acceptance'):
        if identity(Path(manifest[key]['path'])) != manifest[key]:
            raise ValueError(f'frozen {key} identity changed')
    output = Path(manifest['output_root'])
    validate_output_root(output)
    if output.resolve() != manifest_path.parent.resolve():
        raise ValueError('output manifest relocation is not allowed')
    with (output / f'shard-{shard_index:02d}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another scorer owns this shard') from error
        run_locked(manifest, manifest_path, shard_index, num_shards, device)


def run_locked(manifest: dict[str, Any], manifest_path: Path, shard_index: int,
               num_shards: int, device: str) -> None:
    output = Path(manifest['output_root'])
    manifest_sha = sha256_file(manifest_path)
    states = [s for index, s in enumerate(manifest['states']) if index % num_shards == shard_index]
    ledger = output / f'rank-shard-{shard_index:02d}.jsonl'
    done: dict[str, Any] = {}
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row['pair_key']
            if key in done:
                raise ValueError('duplicate ledger key')
            done[key] = row
    if set(done) - {s['asset_source_pair_key'] for s in states}:
        raise ValueError('existing ledger belongs to a different shard allocation')
    model = None
    for state in states:
        legacy_key = state['asset_source_pair_key']
        broad_identity = validate_assets(state)
        if broad_identity not in manifest['historical_broad_rankings']:
            raise ValueError('historical Broad score receipt changed after preparation')
        state_dir = output / 'states' / hashlib.sha256(legacy_key.encode()).hexdigest()[:20]
        rank_path = state_dir / 'ranking.json'
        if legacy_key in done:
            validate_result(done[legacy_key], state, manifest_sha)
            if not rank_path.exists() or read(rank_path) != done[legacy_key]:
                raise ValueError('ledger and ranking receipt differ')
            continue
        if rank_path.exists():
            row = read(rank_path)
            validate_result(row, state, manifest_sha)
        else:
            if model is None:
                import torch
                from AlphaBrain.model.framework.base_framework import BaseFramework
                model = BaseFramework.from_pretrained(manifest['checkpoint'], strict_checkpoint=True)
                model = model.to(torch.bfloat16).to(device).eval()
                model.gripper_remap = False
                if int(model.action_horizon) != 10 or int(model.action_dim) != 7:
                    raise ValueError('model action shape mismatch')
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
            scoring_state = dict(state, pair_key=legacy_key)
            print(json.dumps({'event': 'state_start', 'pair_key': state['pair_key']}), flush=True)
            # rank_state writes a raw receipt itself. Keep that incomplete
            # identity out of the resumable final path until validation finishes.
            with tempfile.TemporaryDirectory(prefix='.rank-compute-', dir=output) as scratch:
                row = rank_state(scoring_state, model=model,
                                 render_dir=Path(state['static_assets']['render_receipt']).parent,
                                 ensemble_size=8, root_seed=ROOT_SEED, batch_size=16,
                                 output_dir=Path(scratch))
            row.update(population_pair_key=state['pair_key'], v2_split=state['split'],
                       checkpoint_sha256=EXPECTED_WEIGHT_HASH, manifest_sha256=manifest_sha)
            validate_result(row, state, manifest_sha)
            state_dir.mkdir(parents=True, exist_ok=True)
            with rank_path.open('x') as handle:
                json.dump(row, handle, indent=2, sort_keys=True)
                handle.write('\n')
        append_jsonl(ledger, row)
        print(json.dumps({'event': 'state_complete', 'pair_key': state['pair_key'],
                          'inference_seconds': row['inference_seconds']}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--selection', type=Path, default=ROOT / 'protocols/selection.json')
    parser.add_argument('--receipt', type=Path, default=ROOT / 'checkpoint-audit/completed_run_receipt.json')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'canonical-accel')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'canonical-accel/manifest.json')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.prepare:
        print(prepare(args.selection, args.receipt, args.output_root))
    else:
        run(args.manifest, args.shard_index, args.num_shards, args.device)


if __name__ == '__main__':
    main()
