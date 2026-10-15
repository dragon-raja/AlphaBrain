"""Freeze a label-independent paired rendering plan; does not launch GPUs."""

import argparse
from collections import Counter
import json
from pathlib import Path

from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.data.rig_training import candidate_catalog, render_jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = read(args.config)
    ledger = Path(cfg['history']) / 'B_source_window_ledger_v1/source_windows.jsonl'
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    if len(rows) != 33000 or len({r['dataset_identity'] for r in rows}) != 300:
        raise ValueError('Do not silently change the all-300-task training population')
    catalog = candidate_catalog(cfg['training_proposal']['seed'])
    jobs = render_jobs(rows, catalog, cfg['training_proposal']['seed'])
    root = Path(__file__).resolve().parents[3]
    sources = ['AlphaBrain/research/dsol/data/rig_training.py',
               'AlphaBrain/research/dsol/acquisition/camera_rig.py',
               'benchmarks/Robocasa365/camera_rig.py', 'benchmarks/Robocasa365/training_render.py',
               'scripts/dsol_paper1/training/render_robocasa_matched.py']
    result = dict(status='RENDER_PLAN_FROZEN_TRAINING_NOT_RELEASED', config=cfg,
        config_sha256=sha(args.config), source_ledger=str(ledger), ledger_sha256=sha(ledger),
        dataset_root=cfg['dataset_root'], catalog=catalog, jobs=jobs,
        implementation_sha256={str(root / p): sha(root / p) for p in sources},
        counts=dict(source_windows=len(rows), episodes=len(jobs), tasks=300,
                    exposures_per_arm=66000, candidate_views=97),
        episode_assignments_per_view=dict(Counter(v for job in jobs for v in job['view_ids'])),
        semantics=dict(anchor_frame=0, installation_constant_within_episode=True,
            native_support_motion_preserved=True, shared_wrist_image_per_frame=True,
            canonical_and_broad_same_png_codec=True, same_source_actions_language_state=True,
            no_test_success_labels_used=True, no_physical_reachability_claim=True))
    write(args.output, result)
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    main()
