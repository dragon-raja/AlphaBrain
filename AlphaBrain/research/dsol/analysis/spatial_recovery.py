"""Separate existence of good viewpoints from score-only endpoint selection.

This is descriptive reuse of a fixed response panel, not a new online search
experiment. Paired starts reuse each selected endpoint's two outcomes. They are
not independent extra trajectories, and no motion/query costs are estimated.
"""
from collections import defaultdict
from math import comb
import numpy as np


def uniform_subset_argmin_probabilities(scores, subset_size):
    """Exact selection weights for a uniform subset, independent of outcomes.

    Ties use input order. This finite-pool diagnostic ignores acquisition paths,
    return queries and fresh score noise; it is not an online search policy.
    """
    scores = np.asarray(scores, dtype=float)
    if (scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all()
            or type(subset_size) is not int or not 1 <= subset_size <= len(scores)):
        raise ValueError('Finite scores and an integer subset size in [1, N] required')
    count = len(scores)
    weights = np.zeros(count)
    order = np.argsort(scores, kind='stable')
    denominator = comb(count, subset_size)
    for rank, index in enumerate(order):
        if count - rank - 1 >= subset_size - 1:
            weights[index] = comb(count - rank - 1, subset_size - 1) / denominator
    return weights


def analyze(rows, *, expected_views=27, reference_id=0, low_count=7):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['state_key']].append(row)
    if not grouped:
        raise ValueError('Empty fixed response panel')
    per_state = []
    for state, values in sorted(grouped.items()):
        if (len(values) != expected_views
                or {x['view_id'] for x in values} != set(range(expected_views))
                or len({x['task'] for x in values}) != 1):
            raise ValueError('Incomplete or inconsistent fixed panel')
        for row in values:
            if (len(row['success']) != 2 or any(type(v) is not bool for v in row['success'])
                    or np.asarray(row['p3']).shape != (2,) or not np.isfinite(row['p3']).all()
                    or np.asarray(row['angles_deg']).shape != (2,)
                    or not np.isfinite(row['angles_deg']).all()):
                raise ValueError('Explicit finite scores, poses and two outcomes required')
        ref = next(x for x in values if x['view_id'] == reference_id)
        candidates = sorted((x for x in values if x['view_id'] != reference_id),
                            key=lambda x: (float(np.mean(x['p3'])), x['view_id']))
        if len(candidates) < low_count:
            raise ValueError('Insufficient non-reference candidates')
        selected = candidates[0]  # Outcomes cannot participate in this selection.
        pair = dict(comparisons=0, before_success=0, after_success=0,
                    rescued=0, harmed=0, failed_starts=0,
                    failed_with_lower_score_success_available=0)
        starts_with_lower_better_mean = 0
        for start in candidates:
            lower = [x for x in candidates if np.mean(x['p3']) < np.mean(start['p3'])]
            starts_with_lower_better_mean += any(
                np.mean(x['success']) > np.mean(start['success']) for x in lower)
            for repeat in (0, 1):
                before, after = start['success'][repeat], selected['success'][repeat]
                pair['comparisons'] += 1
                pair['before_success'] += int(before)
                pair['after_success'] += int(after)
                pair['rescued'] += int(not before and after)
                pair['harmed'] += int(before and not after)
                pair['failed_starts'] += int(not before)
                pair['failed_with_lower_score_success_available'] += int(
                    not before and any(x['success'][repeat] for x in lower))
        ranked = [dict(view_id=x['view_id'], angles_deg=list(x['angles_deg']),
                       rank=i + 1, p3=float(np.mean(x['p3'])),
                       p3_members=list(x['p3']), success=list(x['success']),
                       successes=sum(x['success'])) for i, x in enumerate(candidates)]
        per_state.append(dict(state_key=state, task=values[0]['task'],
            reference_success=sum(ref['success']) / 2,
            perturbed_success=pair['before_success'] / pair['comparisons'],
            argmin=ranked[0], ranked_nonreference=ranked,
            observed_best_success=max(sum(x['success']) / 2 for x in candidates),
            has_two_of_two=any(all(x['success']) for x in candidates),
            low_group_has_two_of_two=any(all(x['success']) for x in candidates[:low_count]),
            low_group_success=float(np.mean([x['success'] for x in candidates[:low_count]])),
            lower_score_better_mean_starts=starts_with_lower_better_mean,
            paired_endpoint_reuse=pair))
    totals = {key: sum(x['paired_endpoint_reuse'][key] for x in per_state)
              for key in per_state[0]['paired_endpoint_reuse']}
    tasks = defaultdict(list)
    for row in per_state:
        tasks[row['task']].append(row['argmin']['successes'] / 2 - row['perturbed_success'])
    task_deltas = np.asarray([np.mean(v) for _, v in sorted(tasks.items())])
    bootstrap = np.random.default_rng(20261029).choice(
        task_deltas, size=(5000, len(task_deltas)), replace=True).mean(axis=1)
    return dict(score='p3', selection='mean of two score members; stable view-id tie break',
        selection_uses_success_labels=False, reference_excluded_from_selection=True,
        states=len(per_state), tasks=len(tasks), positions=len(rows),
        reference_success=float(np.mean([x['reference_success'] for x in per_state])),
        perturbed_success=totals['before_success'] / totals['comparisons'],
        selected_success=sum(x['argmin']['successes'] for x in per_state) / (2 * len(per_state)),
        selected_successes=sum(x['argmin']['successes'] for x in per_state),
        selected_distinct_operations=2 * len(per_state),
        argmin_net_gain_task_mean=float(task_deltas.mean()),
        argmin_net_gain_task_cluster_95=np.quantile(bootstrap, [.025, .975]).tolist(),
        states_with_observed_two_of_two=sum(x['has_two_of_two'] for x in per_state),
        states_with_low_group_two_of_two=sum(x['low_group_has_two_of_two'] for x in per_state),
        starts_with_lower_score_higher_observed_sr=sum(x['lower_score_better_mean_starts'] for x in per_state),
        distinct_perturbed_starts=sum(len(x['ranked_nonreference']) for x in per_state),
        paired_endpoint_reuse=totals, per_state=per_state,
        limits=['Two operation repeats per pose; empirical 2/2 is not a known success probability',
                'Good-view existence and observed-best are outcome-informed diagnostics, not policies',
                'Endpoint pairs reuse outcomes; no additional recovery trajectory or motion validation',
                'Finite development viewpoints do not prove continuous or held-out generalization'])
