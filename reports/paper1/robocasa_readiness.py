"""Read-only asset inventory, not scientific result or full shard hash audit."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from AlphaBrain.research.dsol.data.artifacts import read, write, sha


def build(config_path):
    cfg = read(config_path)
    rows = []
    inputs = {str(Path(config_path).resolve()): sha(config_path)}
    for p in sorted(Path(cfg['checkpoint_receipts']).glob('*.json')):
        r = read(p)
        inputs[str(p)] = sha(p)
        root = Path(r['target_path'])
        files = [x for x in root.rglob('*') if x.is_file()] if root.is_dir() else []
        manifest = root / '.transfer_manifest.sha256'
        payload_files = [x for x in files if x != manifest]
        norm = root / 'assets/norm_stats.json'
        rows.append(dict(label=r['label'], path=str(root), present=root.is_dir(), file_count=len(files),
            total_bytes=sum(x.stat().st_size for x in files), receipt_file_count=r['file_count'],
            receipt_total_bytes=r['total_bytes'], norm_sha256=sha(norm) if norm.is_file() else None,
            params_present=(root / 'params').is_dir(), payload_file_count=len(payload_files),
            payload_bytes=sum(x.stat().st_size for x in payload_files),
            transfer_manifest_sha256=sha(manifest) if manifest.is_file() else None,
            receipt_manifest_sha256=r.get('transfer_manifest_sha256', r.get('manifest_sha256'))))
    datasets = []
    for path in sorted(Path(cfg['human300_root']).glob('*/*/*/lerobot/meta/info.json')):
        info = read(path)
        datasets.append(dict(path=str(path.parent.parent), task=path.parents[3].name,
            category=path.parents[4].name, episodes=info.get('total_episodes'), frames=info.get('total_frames')))
    runs = {}
    for arm in ['canonical_repeat', 'mv_unpaired', 'mv_paired', 'image_aug_control']:
        p = Path(cfg['historical_training']) / arm / 'run_config.json'
        if p.is_file():
            r = read(p)
            inputs[str(p)] = sha(p)
            runs[arm] = {k: r.get(k) for k in ['config', 'initial_params', 'norm_stats_sha256', 'total_steps',
                'batch_size', 'seed', 'fsdp_devices', 'data_dataset_count', 'view_organization', 'model_train_image_augmentation']}
    payload = dict(created_utc=datetime.now(timezone.utc).isoformat(), checkpoints=rows, datasets=datasets,
        dataset_count=len(datasets), historical_runs=runs,
        self_trained_75k=dict(path=cfg['self_trained_75k'], exists=Path(cfg['self_trained_75k']).exists(),
                             user_reported=True, official_success_not_attributed_to_this_checkpoint=True),
        inputs=inputs, scope='Current file existence/size and metadata hashes; historical receipts retained, no claim of rehashing all parameter shards or reproducing historical rollouts')
    out = Path(cfg['output']) / cfg.get('inventory_filename', 'inventory.json')
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out, payload)
    print(dict(inventory=str(out), checkpoints=len(rows), datasets=len(datasets),
               counts_match=all(r['payload_file_count'] == r['receipt_file_count'] and r['payload_bytes'] == r['receipt_total_bytes'] for r in rows),
               transfer_manifest_hashes_match=all(r['transfer_manifest_sha256'] == r['receipt_manifest_sha256'] for r in rows)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    build(p.parse_args().config)
