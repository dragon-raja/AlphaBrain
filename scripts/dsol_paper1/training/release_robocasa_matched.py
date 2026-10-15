"""Audit every completed render and emit matched OpenPI data manifests.

No training or GPUs. Missing tasks, source changes, image corruption, or changed
wrist inputs block release. Raw render receipts are never rewritten.
"""

import argparse
import json
from pathlib import Path

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.data.matched_render import validate_episode_receipt, select_episode_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--renders', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recovery-root', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = read(args.plan)
    fingerprint = sha(args.plan)
    completion = read(args.renders / 'completion.json')
    if not args.recovery_root and (not completion.get('all_pass') or completion['passed'] != len(plan['jobs'])):
        raise ValueError('All original episode jobs must finish before data release')
    if completion.get('expected') != len(plan['jobs']) or any(completion.get('worker_exit_codes', [1])):
        raise ValueError('Original queue must have completed all shards')
    if sha(plan['source_ledger']) != plan['ledger_sha256']:
        raise ValueError('Source ledger changed')
    records, source_hashes, receipt_hashes = {}, {}, {}
    image_count = 0
    for job_number, job in enumerate(plan['jobs'], 1):
        root = select_episode_root(job['job_id'], args.renders, args.recovery_root)
        path = root / 'summary.json'
        result = read(path)
        rows = validate_episode_receipt(job, result, fingerprint)
        receipt_hashes[str(path)] = sha(path)
        for relative, expected in result['image_sha256'].items():
            target = (root / relative).resolve()
            if not target.is_relative_to(root.resolve()) or sha(target) != expected:
                raise ValueError('Image hash/path failed: ' + relative)
            image_count += 1
        for row in rows:
            for exposure in row['canonical_exposures'] + row['broad_exposures']:
                for path_string in exposure['image_files'].values():
                    relative = str(Path(path_string).relative_to(root))
                    if relative not in result['image_sha256']:
                        raise ValueError('Exposure uses an unaudited image')
            index = row['training_pair_index']
            if index in records:
                raise ValueError('Duplicate global source-window index')
            records[index] = row
        for path_string, expected in result['source_hashes'].items():
            if path_string in source_hashes and source_hashes[path_string] != expected:
                raise ValueError('Source hashes differ across episodes')
            source_hashes[path_string] = expected
        if job_number % 500 == 0 or job_number == len(plan['jobs']):
            print(dict(phase='render_hashes', episodes=job_number, total=len(plan['jobs']),
                       image_files=image_count), flush=True)
    for source_number, (path_string, expected) in enumerate(source_hashes.items(), 1):
        if sha(path_string) != expected:
            raise ValueError('Original training source changed: ' + path_string)
        if source_number % 1000 == 0 or source_number == len(source_hashes):
            print(dict(phase='source_hashes', checked=source_number, total=len(source_hashes)), flush=True)
    if set(records) != set(range(plan['counts']['source_windows'])):
        raise ValueError('Incomplete source population')
    cfg = plan['config']
    previous_manifest = Path(cfg['history']) / 'B_training_arm_manifests_v1/canonical_repeat/training_manifest.json'
    data_dirs = read(previous_manifest)['data_dirs']
    # Keep source-dataset index order, but resolve paths from the explicit root.
    identities = {r['dataset_index']: r['dataset_identity'] for job in plan['jobs'] for r in job['rows']}
    if set(identities) != set(range(300)) or len(data_dirs) != 300:
        raise ValueError('Dataset index population mismatch')
    for index, metadata in enumerate(data_dirs):
        current = Path(plan['dataset_root']) / identities[index]
        if metadata['task'] != identities[index].split('/')[1] or metadata['filter_key'] != '25_demos':
            raise ValueError('Inherited dataset order/filter mismatch')
        metadata['path'] = str(current)
    args.output.mkdir(parents=True)
    manifests = {}
    for arm, key, organization in [('canonical_matched', 'canonical_exposures', 'canonical_repeat'),
                                    ('broad_exterior_matched', 'broad_exposures', 'mv_paired')]:
        exposures = args.output / f'{arm}.jsonl'
        with exposures.open('x') as stream:
            for index in range(len(records)):
                row = records[index]
                stream.write(json.dumps(dict(source_window_id=row['source_window_id'], exposures=row[key]))+'\n')
        manifest = dict(format='robocasa_exterior_matched_training_v1', status='DATA_READY_NOT_TRAINED',
            arm=arm, dataset_count=300, data_dirs=data_dirs, source_window_ledger=plan['source_ledger'],
            source_window_ledger_sha256=plan['ledger_sha256'], view_organization=organization,
            view_exposure_manifest=str(exposures), view_exposure_manifest_sha256=sha(exposures),
            ordinary_image_augmentation=None, training_shuffle=False, source_windows=len(records),
            exposures=2*len(records), plan_sha256=fingerprint, training=cfg['training_proposal'],
            initialization=cfg['initialization'], no_consistency_loss=True,
            semantics=plan['semantics'], codec='lossless_png_rgb256_compress3')
        path = args.output / f'{arm}.json'
        write(path, manifest)
        manifests[arm] = dict(path=str(path), sha256=sha(path))
    write(args.output / 'release.json', dict(status='DATA_READY_NOT_TRAINED', plan_sha256=fingerprint,
        original_completion_sha256=sha(args.renders / 'completion.json'),
        recovery_roots=[str(p) for p in args.recovery_root], original_failures_preserved=True,
        image_files_verified=image_count, source_files_verified=len(source_hashes),
        receipt_sha256=receipt_hashes, manifests=manifests,
        next_gate='Real loader identity audit and real-data FSDP8 optimizer smoke, then matched training'))


if __name__ == '__main__':
    main()
