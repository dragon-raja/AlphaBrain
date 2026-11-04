"""CPU finite-pool diagnostic; never launches acquisition or changes a queue."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from AlphaBrain.research.dsol.analysis.spatial_recovery import (
    analyze, uniform_subset_argmin_probabilities,
)


def build(panel_path, output, *, score_names=('p3', 'p9')):
    path, output = Path(panel_path), Path(output)
    raw = path.read_bytes()
    rows = json.loads(raw)
    spatial = analyze(rows)  # Check complete states, poses, outcomes and p3.
    grouped = defaultdict(list)
    for row in rows:
        if row['view_id'] != 0: grouped[row['state_key']].append(row)
    results = []; member_stability = {}
    for score in score_names:
        member_rows = []
        for state, candidates in sorted(grouped.items()):
            candidates.sort(key=lambda x: x['view_id'])
            members = np.asarray([x[score] for x in candidates], dtype=float)
            if members.shape != (26, 2) or not np.isfinite(members).all():
                raise ValueError('Expected two finite score members per candidate')
            selected = np.argmin(members, axis=0)
            correlation = (float(np.corrcoef(members.T)[0, 1])
                           if np.all(members.std(axis=0) > 0) else None)
            member_rows.append(dict(state_key=state,
                same_minimum=bool(selected[0] == selected[1]),
                spatial_score_pearson_between_noise_members=correlation,
                relative_spatial_std=float(members.mean(axis=1).std() / members.mean())
                    if members.mean() > 0 else None,
                selected_view_by_member=[candidates[i]['view_id'] for i in selected],
                observed_success_by_member=[float(np.mean(candidates[i]['success']))
                                            for i in selected]))
        member_stability[score] = dict(
            same_minimum_states=sum(x['same_minimum'] for x in member_rows),
            per_state=member_rows,
            limitation='Two frozen members only; not a calibrated uncertainty estimate')
        for size in (1, 2, 4, 7, 13, 26):
            states = []
            for state, candidates in sorted(grouped.items()):
                candidates.sort(key=lambda x: x['view_id'])
                members = np.asarray([x[score] for x in candidates], dtype=float)
                if members.shape != (26, 2) or not np.isfinite(members).all():
                    raise ValueError('Expected two finite score members per candidate')
                weights = uniform_subset_argmin_probabilities(members.mean(axis=1), size)
                outcomes = np.asarray([x['success'] for x in candidates]).mean(axis=1)
                states.append(dict(state_key=state, task=candidates[0]['task'],
                    expected_success=float(weights @ outcomes),
                    random_endpoint_success=float(outcomes.mean())))
            by_task = defaultdict(list)
            for row in states: by_task[row['task']].append(row['expected_success'])
            results.append(dict(score=score, subset_size=size,
                task_mean_success=float(np.mean([np.mean(v) for v in by_task.values()])),
                per_state=states))
    summary = dict(input=str(path), input_sha256=hashlib.sha256(raw).hexdigest(),
        states=spatial['states'], tasks=spatial['tasks'], reference_excluded=True,
        rule='Uniform subset without replacement; minimum mean score; view-id tie break',
        outcome_labels_used_for_selection=False, new_model_calls=0,
        limitations=['No legal motion path, return-query cost or new score noise modeled',
            'Subset size counts positions, not production query usage',
            'Reuses the same two operation outcomes per point; not new independent trials',
            'Development panel only; not continuous or held-out task performance'],
        frontier=results, member_stability=member_stability)
    output.mkdir(parents=True, exist_ok=False)
    (output/'frontier.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    lines = ['score,subset_positions,expected_task_mean_success']
    lines += [f"{x['score']},{x['subset_size']},{x['task_mean_success']:.12f}" for x in results]
    (output/'frontier.csv').write_text('\n'.join(lines)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); result = build(args.panel, args.output)
    print(json.dumps([{k: v for k, v in x.items() if k != 'per_state'}
                      for x in result['frontier']], indent=2))
