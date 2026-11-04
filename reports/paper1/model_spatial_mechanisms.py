"""Matched model comparison and archived FastWAM trace diagnostics; CPU only."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from AlphaBrain.research.dsol.execution.manifest import file_digest
from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores
from AlphaBrain.research.dsol.analysis.spatial_task_average import linear_comparison
from reports.paper1.spatial_selection_frontier import build as frontier
from reports.paper1.target_spatial_panel import load_verified_panel


def read(path):
    return json.loads(Path(path).read_text())


def matched_rows(fastwam, groot, native, references, consumed):
    identity = {(x['state_key'], x['view_id']): x['angles_deg'] for x in fastwam}
    if len(identity) != 432: raise ValueError('Expected full 16-state diagnostic')
    converted = []
    for x in native:
        key = x['state_key'], x['view_id']
        if key not in identity: continue
        if not np.allclose(identity[key], [x['yaw_deg'], x['pitch_deg']], atol=1e-12, rtol=0):
            raise ValueError('Native geometry changed')
        ref = references['views'][f"{key[0]}::{key[1]}"]
        path = str(Path(ref['query']).parent/'scores.json')
        if file_digest(path) != ref['artifacts'][path]: raise ValueError('Score JSON hash mismatch')
        consumed[path] = ref['artifacts'][path]
        records = sorted(read(path), key=lambda r:r['member'])
        if [r['member'] for r in records] != [0, 1, 2, 3]: raise ValueError('Missing score members')
        if any(r['state_key'] != key[0] or r['view_id'] != key[1] for r in records):
            raise ValueError('Score identity mismatch')
        scores = [r['score'] for r in records[:2]]
        if abs(np.mean(scores)-x['accel']) > 1e-12: raise ValueError('Archived mean changed')
        converted.append(dict(state_key=key[0], task=x['task'], view_id=key[1],
            angles_deg=identity[key], p3=scores, success=x['success']))
    for rows in (groot, converted):
        other = {(x['state_key'], x['view_id']):x['angles_deg'] for x in rows}
        if len(rows) != 432 or other.keys() != identity.keys():
            raise ValueError('Models do not share the complete diagnostic panel')
        if any(not np.allclose(other[k], identity[k], atol=1e-12, rtol=0) for k in identity):
            raise ValueError('Model camera geometry differs')
    return converted


def build(fastwam_root, groot_panel, native_panel, native_refs, output):
    root, output = Path(fastwam_root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    paths = [root/'report/rows.json', Path(groot_panel), Path(native_panel), Path(native_refs),
             root/'p3-recovery-explanation/pairing-audit.json']
    consumed = {str(p):file_digest(p) for p in paths}
    fastwam, groot, native, refs, audit = map(read, paths)
    pi05 = matched_rows(fastwam, groot, native, refs, consumed)
    # Existing loader validates source blocks and reproduces the archived p3/p9.
    rebuilt, positions = load_verified_panel(root)
    if rebuilt != fastwam: raise ValueError('Original spatial panel reconstruction differs')
    variants = {name:[] for name in ('p3_exec10', 'p9_exec10')}
    grids = set()
    for row in fastwam:
        point = positions[row['state_key'], row['view_id']]
        path = point['score_query']; stamp = file_digest(path)
        if stamp != audit['query_sha256'][path]: raise ValueError('Frozen trace hash changed')
        consumed[path] = stamp
        scores = defaultdict(list)
        with np.load(path, allow_pickle=False) as z:
            for member in (0, 1):
                trace = z[f'trace_{member}']
                if trace.shape != (10, 32, 7): raise ValueError('Unexpected native action shape')
                grids.add(tuple(float(t) for t in z[f'timesteps_{member}']))
                for prefix in (3, 9):
                    values = compute_accel_scores(trace[None, :, :10], prefixes=(prefix,))[prefix]
                    if values['degenerate'][0]: raise ValueError('Degenerate action trace')
                    scores[f'p{prefix}_exec10'].append(float(values['score'][0]))
        for name in variants:
            variants[name].append(dict(row, p3=scores[name]))
    if len(grids) != 1: raise ValueError('Mixed FastWAM time grids')
    all_panels = dict(pi05=pi05, groot=groot, fastwam=fastwam, **variants)
    results = {}
    for name, rows in all_panels.items():
        path = output/f'{name}-rows.json'; path.write_text(json.dumps(rows)+'\n')
        names = ('p3', 'p9') if name=='fastwam' else ('p3',)
        summary = frontier(path, output/name, score_names=names)
        results[name] = dict(reference_success=float(np.mean([x['success'] for x in rows if x['view_id']==0])),
            frontier=[{k:v for k,v in x.items() if k!='per_state'} for x in summary['frontier']],
            member_summary={score:dict(same_minimum_states=x['same_minimum_states'],
                mean_member_pearson=float(np.mean([s['spatial_score_pearson_between_noise_members']
                                                   for s in x['per_state']])),
                mean_relative_spatial_std=float(np.mean([s['relative_spatial_std'] for s in x['per_state']])))
                for score,x in summary['member_stability'].items()},
            linear={score:linear_comparison(rows, score=score)['scopes']['nonreference'] for score in names})
    result = dict(models=results, fastwam_raw_timesteps=list(next(iter(grids))),
        matched_tasks=8, matched_states=16, matched_views=27, new_model_calls=0,
        new_training_updates=0, fastwam_scientific_variants='Posthoc diagnostic only, not deployed',
        limits=['Same state/pose identities, different native model training and execution protocols',
                'Finite-pool selection; not legal continuous acquisition or a new operation evaluation',
                'Only two operation repeats; no outcome-based rule can be validated on this same panel'])
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (output/'consumed-hashes.json').write_text(json.dumps(consumed, indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('fastwam-root','groot-panel','native-panel','native-refs','output'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    result=build(args.fastwam_root,args.groot_panel,args.native_panel,args.native_refs,args.output)
    print(json.dumps(result,indent=2))
