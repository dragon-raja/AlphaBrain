"""CPU/data interface audit only. Never renders, queries a VLA, or trains.

Uses one explicitly historical observation per initial to check frozen actor
compatibility. Generated proposals have geometry only. They are NEVER assigned
interpolated historical scores/features or reported as new successful queries.
"""
import argparse
from pathlib import Path

import numpy as np

from AlphaBrain.research.dsol.acquisition.local_view_search import (
    LocalViewSearch, PanTiltMount, PoseReceipt, ViewMeasurement,
)
from AlphaBrain.research.dsol.data.artifacts import read, sha, write
from AlphaBrain.research.dsol.data.search_replay import QueryResult


def audit(config_path, output):
    import torch
    from AlphaBrain.research.dsol.selectors.search_attention import SharedSearchActor

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = read(config_path)
    actor_path = config['actor']['path']
    if sha(actor_path) != config['actor']['sha256']:
        raise ValueError('Frozen actor hash mismatch')
    trained = torch.load(actor_path, map_location='cpu', weights_only=True)
    actor = SharedSearchActor(trained['visual_dim']).eval()
    actor.load_state_dict(trained['state_dict'], strict=True)
    actor.requires_grad_(False)
    features_path = Path(config['native']) / 'siglip-scores.npz'
    with np.load(features_path, allow_pickle=False) as z:
        features, wrist, scores = (z[k].copy() for k in ('external', 'wrist', 'scores'))
    with np.load(config['outcome_matrix'], allow_pickle=False) as z:
        # Only calibration geometry is read. No outcome labels enter this audit.
        poses = z['camera'].copy()
    if (poses.shape[:2] != features.shape[:2] or wrist.shape != features.shape
            or scores.shape[:2] != features.shape[:2]):
        raise ValueError('Historical data axes disagree')
    rows = []
    for initial in range(len(features)):
        mount = PanTiltMount(poses[initial, 0])
        historical = QueryResult(features[initial, 0], float(scores[initial, 0].mean()),
                                 float(scores[initial, 0].std()), len(scores[initial, 0]))

        class HistoricalInitialOnly:
            def move(self, pose):
                raise AssertionError('CPU audit cannot execute a proposed movement')

            def query(self):
                return ViewMeasurement(PoseReceipt(poses[initial, 0], 'historical-initial'), historical)

        session = LocalViewSearch(mount, HistoricalInitialOnly(), initial_angles=[0, 0],
            physical_state_id='historical-initial', wrist=wrist[initial, 0],
            score_scale=trained['score_scale'], seed=20261021 + initial)
        obs = session.start()
        proposal_poses = np.stack([mount.pose(a) for a in session.proposals])
        overlap = [any(np.allclose(p, old, rtol=0, atol=1e-9) for old in poses[initial])
                   for p in proposal_poses]
        tensor = {k: torch.as_tensor(v[None], dtype=torch.float32) for k, v in obs.items()}
        with torch.inference_mode():
            logits = actor(tensor)
        finite = np.isfinite(logits.numpy()[0])
        if not np.array_equal(finite, obs['action_mask'].astype(bool)):
            raise ValueError('Actor output violates dynamic mask')
        if obs['visual'][1:].any() or obs['scores'][1:].any():
            raise ValueError('Unqueried information leaked')
        rows.append(dict(initial=initial, proposals=len(proposal_poses),
                         proposals_outside_old_catalog=sum(not x for x in overlap),
                         historical_initial_only=True, output_shape=list(logits.shape)))
    result = dict(status='CPU_DATA_PASS_NOT_RUNTIME_ADMISSION',
        actor_sha256=sha(actor_path), config_sha256=sha(config_path),
        features_sha256=sha(features_path), geometry_file_sha256=sha(config['outcome_matrix']),
        source_sha256={str(p): sha(p) for p in [Path(__file__),
            Path(__file__).resolve().parents[2] / 'AlphaBrain/research/dsol/acquisition/local_view_search.py']},
        rows=rows, new_vla_calls=0, new_render_calls=0, training_steps=0,
        boundary='Off-catalog geometry and valid logits do not establish search quality or generalization')
    write(output / 'receipt.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    audit(args.reference_config, args.output)
