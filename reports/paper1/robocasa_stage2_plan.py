"""Inventory the complete inherited training ledger before new rendering."""

import argparse
from collections import Counter
from pathlib import Path
import json

from AlphaBrain.research.dsol.data.artifacts import read, sha, write


def build(config):
    cfg = read(config)
    output = Path(cfg['output'])
    if output.exists():
        raise FileExistsError(output)
    history = Path(cfg['history'])
    ledger_path = history / 'B_source_window_ledger_v1/source_windows.jsonl'
    with ledger_path.open() as stream:
        ledger = [json.loads(line) for line in stream]
    datasets = Counter(row['dataset_identity'] for row in ledger)
    episodes = sorted({(row['dataset_identity'], row['episode_index']) for row in ledger})
    records = []
    for identity, episode in episodes:
        root = Path(cfg['dataset_root']) / identity
        name = f'episode_{episode:06d}'
        extras = root / 'extras' / name
        parquet = list(root.glob(f'data/*/{name}.parquet'))
        required = [extras / 'ep_meta.json', extras / 'model.xml.gz', extras / 'states.npz', root / 'extras/dataset_meta.json']
        missing = [str(p) for p in required if not p.is_file()]
        if len(parquet) != 1:
            missing.append(f'expected_one_parquet_found_{len(parquet)}')
        records.append(dict(dataset=identity, episode=episode, missing=missing))
    snapshot_records = []
    for bank in sorted(Path(cfg['snapshot_root']).glob('*/bank_manifest.json')):
        for row in read(bank)['bundles']:
            path = bank.parent / 'snapshots' / row['task'] / f"seed_{row['scene_seed']:06d}"
            snapshot_records.append(dict(task=row['task'], scene_seed=row['scene_seed'], split=row['split'],
                layout_id=row['layout_id'], style_id=row['style_id'], path=str(path),
                manifest_present=(path / 'manifest.json').is_file(), navigation=row['task'] == 'NavigateKitchen'))
    result = dict(status='PREPARATION_NOT_RELEASED', config_sha256=sha(config), ledger_sha256=sha(ledger_path),
        source_windows=len(ledger), dataset_count=len(datasets), windows_per_dataset=dict(datasets),
        unique_episodes=len(episodes), episodes_missing_replay_assets=[r for r in records if r['missing']],
        snapshot_count=len(snapshot_records), snapshot_manifests_present=sum(r['manifest_present'] for r in snapshot_records),
        training=cfg['training_proposal'], outstanding_release_gates=cfg['release_gates'],
        no_training_or_rollouts_started_by_this_planner=True,
        scope='File presence only; per-episode state/action alignment, XML assets and rendering remain to validate')
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'preparation.json', result)
    write(output / 'training-episode-inventory.json', records)
    write(output / 'evaluation-population.json', snapshot_records)
    print(json.dumps({k: v for k, v in result.items() if k != 'windows_per_dataset'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    build(parser.parse_args().config)
