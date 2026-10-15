"""Audit historical training support without interpreting old outcome labels."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json

from AlphaBrain.research.dsol.data.artifacts import read, sha, write


def audit(history, snapshot_root, output):
    history, snapshot_root, output = map(Path, (history, snapshot_root, output))
    if output.exists():
        raise FileExistsError(output)
    inputs = {}

    def load(path):
        inputs[str(path)] = sha(path)
        return read(path)

    def lines(path):
        inputs[str(path)] = sha(path)
        with path.open() as stream:
            for line in stream:
                yield json.loads(line)

    ledger = list(lines(history / 'B_source_window_ledger_v1/source_windows.jsonl'))
    plans = list(lines(history / 'B_view_bundle_plan_v1/view_bundle_plan.jsonl'))
    pose_catalog = {}
    for row in plans:
        for pose in row['poses'].values():
            key = pose['pose_cell_id']
            if key in pose_catalog and pose_catalog[key] != pose:
                raise ValueError(f'Inconsistent geometry for {key}')
            pose_catalog[key] = pose
    arms = {}
    exposures_by_arm = {}
    for arm in ('canonical_repeat', 'mv_unpaired', 'mv_paired', 'image_aug_control'):
        root = history / 'B_training_arm_manifests_v1' / arm
        manifest_path = root / 'training_manifest.json'
        manifest = load(manifest_path)
        run = load(history / 'B_seed41_training_v1/arms' / arm / 'run_config.json')
        exposure_path = root / 'view_exposures.jsonl'
        rows = list(lines(exposure_path))
        exposures_by_arm[arm] = rows
        poses = Counter(e['pose_cell_id'] for row in rows for e in row['exposures'])
        expected_count = run['total_steps'] * run['batch_size']
        arms[arm] = dict(
            run=run, manifest_sha_matches_run=sha(manifest_path) == run['data_manifest_sha256'],
            exposure_sha_matches_manifest=sha(exposure_path) == manifest['view_exposure_manifest_sha256'],
            ledger_sha_matches_manifest=inputs[str(history / 'B_source_window_ledger_v1/source_windows.jsonl')] == manifest['source_window_ledger_sha256'],
            source_windows=len(rows), exposures=sum(poses.values()), expected_consumed_windows=expected_count,
            count_matches_training_budget=sum(poses.values()) == expected_count,
            pose_histogram=dict(poses), manifest_pose_histogram_matches=dict(poses) == manifest['pose_histogram_across_exposures'],
            repeated_view_pairs=sum(len({e['pose_cell_id'] for e in r['exposures']}) == 1 for r in rows),
            sampled_bundle_paths_present=sum(Path(row['exposures'][0]['image_bundle']).is_file() for row in rows[::max(1, len(rows)//32)]),
            source_datasets=manifest['dataset_count'], consumed_rgb_images=manifest['consumed_rgb_images'])
    keys = ('config', 'initial_params', 'norm_stats_sha256', 'total_steps', 'batch_size',
            'seed', 'fsdp_devices', 'training_shuffle', 'model_train_image_augmentation', 'openpi_robocasa_revision')
    reference = arms['canonical_repeat']['run']
    paired = {key: reference.get(key) == arms['mv_unpaired']['run'].get(key) for key in keys}
    ids = [r['source_window_id'] for r in exposures_by_arm['canonical_repeat']]
    source_alignment = {a: [r['source_window_id'] for r in rows] == ids for a, rows in exposures_by_arm.items()}
    ledger_alignment = ids == [r['source_window_id'] for r in ledger]
    integrity_keys = ('manifest_sha_matches_run', 'exposure_sha_matches_manifest',
                      'ledger_sha_matches_manifest', 'count_matches_training_budget', 'manifest_pose_histogram_matches')
    integrity = all(row[key] for row in arms.values() for key in integrity_keys)
    mv_poses = [pose_catalog[k] for k in arms['mv_unpaired']['pose_histogram']]
    geometry = {}
    for camera in ('agent_rig_delta', 'wrist_delta'):
        geometry[camera] = {field: [max(abs(p[camera][field][i]) for p in mv_poses) for i in range(3)]
                            for field in ('translation_m', 'rotation_xyz_deg')}
    banks = {}
    for path in sorted(snapshot_root.glob('*/bank_manifest.json')):
        bank = load(path)
        rows = bank['bundles']
        banks[path.parent.name] = dict(count=len(rows), tasks=dict(Counter(r['task'] for r in rows)),
            layout_ids=sorted({r['layout_id'] for r in rows}), style_ids=sorted({r['style_id'] for r in rows}),
            scope='Snapshot inventory, not fresh successful restoration or capability evidence')
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), inputs=inputs, arms=arms,
        matched_config_fields=paired, source_window_order_matches=source_alignment,
        exposure_order_matches_ledger=ledger_alignment, metadata_integrity=integrity,
        ledger_windows=len(ledger), unique_source_windows=len({r['source_window_id'] for r in ledger}),
        pose_catalog_recovered_from_used_render_plan=pose_catalog, max_abs_components=geometry,
        snapshot_banks=banks,
        verdict=dict(historical_budget_pairing=all(paired.values()) and all(source_alignment.values()) and ledger_alignment and integrity,
            exterior_only_training=not any(x for row in geometry['wrist_delta'].values() for x in row),
            broad_candidate_coverage_not_established=True,
            reuse='Engineering and explicitly labelled sparse all-camera augmentation baseline',
            mainline='New matched canonical/exterior-only broad augmentation branches recommended; do not relabel old weights',
            replay='Image bundles not recovered; metadata audit does not recertify historical decoded RGB'))
    write(output, result)
    print(json.dumps(dict(output=str(output), matched_config_fields=paired, geometry=geometry,
                          banks=banks, verdict=result['verdict']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', required=True)
    parser.add_argument('--snapshot-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit(args.history, args.snapshot_root, args.output)
