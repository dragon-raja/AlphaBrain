"""Complete measured response panels; no inferred or success-selected trials."""
import numpy as np
from AlphaBrain.research.dsol.data.spatial_completion import key
from .spatial_task_average import linear_comparison
from .spatial_recovery import analyze, uniform_subset_argmin_probabilities
from .spatial_score_validity import summarize as score_validity


def assemble(expected, scores, outcomes):
    """Consume verified records, reject incomplete or multiply counted trials."""
    trials = {key(r): r for r in expected}
    actual = {key(r): r for r in outcomes}
    if (not trials or len(trials) != len(expected) or len(actual) != len(outcomes)
            or set(trials) != set(actual)):
        raise ValueError('Expected exactly one known outcome for every planned trial')
    points = {(r['state_key'], r['view']): r for r in scores}
    if len(points) != len(scores) or set(points) != {(k[0], k[1]) for k in trials}:
        raise ValueError('Missing, duplicate or unplanned score point')
    rows = []
    for (state, view), score in sorted(points.items()):
        success = []
        if len(score['scores']) != 2:
            raise ValueError('Two original scoring members required')
        for repeat in (0, 1):
            trial = trials[state, view, repeat]
            result = actual[state, view, repeat]
            if type(result['success']) is not bool or trial['angles_deg'] != score['angles_deg']:
                raise ValueError('Unknown outcome or changed scored pose')
            success.append(result['success'])
        rows.append(dict(state_key=state, task=state.rsplit('::', 1)[0],
            view_id=view, angles_deg=score['angles_deg'], success=success,
            **{f'p{p}': [s[str(p)] for s in score['scores']] for p in (3, 4)}))
    return rows


def summarize(rows):
    """Report Pearson and outcome-blind endpoint selection with explicit limits."""
    comparisons = {p: linear_comparison(rows, score=p) for p in ('p3', 'p4')}
    recovery = {p: dict(analyze([dict(r, p3=r[p]) for r in rows]), score=p)
                for p in comparisons}
    subsets = {}
    for p in comparisons:
        values = []
        for state in sorted({r['state_key'] for r in rows}):
            candidates = sorted((r for r in rows if r['state_key'] == state and r['view_id'] != 0),
                                key=lambda r: r['view_id'])
            score = [np.mean(r[p]) for r in candidates]
            success = np.asarray([np.mean(r['success']) for r in candidates])
            values.append({str(k): float(uniform_subset_argmin_probabilities(score, k) @ success)
                           for k in (1, 4, 7, 26)})
        subsets[p] = {k: float(np.mean([v[k] for v in values])) for k in values[0]}
    return dict(pearson=comparisons, recovery=recovery, finite_pool_subset=subsets,
        score_validity_secondary=score_validity(rows, score_names=('p3', 'p4')),
        old_two_starts_success=float(np.mean([r['success'] for r in rows if r['view_id'] in (23, 24)])),
        limits=['P4 is stored-native-trace diagnostic, not a newly adopted scoring rule',
                'Finite-pool subsets ignore acquisition paths and return cost; not PPO or online random search',
                'Endpoint rescue/harm pairs reuse selected outcomes; not independent recovery operations',
                'Observed-best views use outcome labels and cannot be deployed as a policy',
                'Development panel with two operation repeats per point; not held-out generalization'])
