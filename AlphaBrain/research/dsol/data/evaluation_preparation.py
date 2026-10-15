"""CPU-only provenance checks for frozen initial-state candidate caches.

This release contains no policy scores, rollout labels, or selector split fit.
It must not be mistaken for an evaluated method or a deployable view sensor.
"""

from collections import Counter
import hashlib
from pathlib import Path

from .artifacts import read, sha


def verify_files(root, files):
    root = Path(root).resolve()
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Artifact path escapes bundle')
        if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            raise ValueError('Artifact integrity failure: ' + str(path))


def prepare_population(population_file, catalog_file):
    """Preserve all supplied entries and verify their immutable snapshot files."""
    rows = read(population_file)
    catalog = read(catalog_file)['catalog']
    views = catalog['views']
    if len({v['view_id'] for v in views}) != len(views):
        raise ValueError('Duplicate view identities')
    if Counter(v['role'] for v in views) != {'reference': 1, 'train': 64, 'extra_eval': 32}:
        raise ValueError('Expected frozen 1+64+32 candidate catalog')
    jobs, identities, layouts = [], set(), {'target': set(), 'pretrain': set()}
    styles = {'target': set(), 'pretrain': set()}
    for row in rows:
        identity = (row['split'], row['task'], row['scene_seed'])
        if identity in identities:
            raise ValueError('Duplicate initial state identity')
        identities.add(identity)
        root = Path(row['path'])
        manifest = read(root / 'manifest.json')
        if manifest['format'] != 'robocasa_eval_snapshot_bundle_v1':
            raise ValueError('Unsupported snapshot format')
        meta = manifest['metadata']
        for key in ('task', 'split', 'scene_seed', 'layout_id', 'style_id'):
            if meta[key] != row[key]:
                raise ValueError('Population/snapshot mismatch: ' + key)
        split, layout = row['split'], row['layout_id']
        if split not in layouts or layout not in (range(1, 11) if split == 'target' else range(11, 61)):
            raise ValueError('Unexpected layout split')
        layouts[split].add(layout)
        if row['style_id'] not in (range(1, 11) if split == 'target' else range(11, 61)):
            raise ValueError('Unexpected style split')
        styles[split].add(row['style_id'])
        verify_files(root, manifest['files'])
        digest = sha(root / 'manifest.json')
        initial_id = hashlib.sha256(('|'.join(map(str, identity)) + '|' + digest).encode()).hexdigest()[:24]
        jobs.append(dict(row, initial_id=initial_id, snapshot_manifest_sha256=digest,
                         use='sealed_layout_test' if split == 'target' else 'seen_layout_pool'))
    if layouts['target'] & layouts['pretrain']:
        raise ValueError('Layout leakage')
    counts = Counter(row['split'] for row in rows)
    return dict(status='SNAPSHOT_FILES_VERIFIED_RENDER_PENDING',
                population_file=str(Path(population_file).resolve()), population_sha256=sha(population_file),
                catalog_file=str(Path(catalog_file).resolve()), catalog_source_sha256=sha(catalog_file),
                catalog=catalog, jobs=jobs, counts=dict(initials=len(rows), by_split=dict(counts),
                    candidate_observations=len(rows) * len(views),
                    navigation=sum(bool(r['navigation']) for r in rows)),
                layouts={k: sorted(v) for k, v in layouts.items()},
                styles={k: sorted(v) for k, v in styles.items()},
                semantics=dict(anchor='persisted_task_initial_state', mounting='constant_support_local',
                    camera_scope='linked_exterior_wrist_mount_unchanged',
                    target_usage='final_test_only_not_selector_training_or_tuning',
                    unseen='joint_heldout_layouts_styles_and_extra_view_coordinates_not_unseen_task_categories',
                    cached_images='offline_field_supervision_or_analysis_not_free_deployment_observations',
                    rollout_images='must_render_online_during_complete_task_execution',
                    noise_budget='not_defined_by_this_data_preparation'))
