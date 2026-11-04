"""Descriptive, state-stratified audits of executed search endpoints.

No unexecuted candidate gets a synthetic success label. Repeated executions with
the same state/angle/noise are counted once, with conflicting labels reported.
Angles are joined at eight decimal places, never by a nearest-neighbour radius.
"""
from collections import defaultdict
import numpy as np


def pose_key(state, angles):
    a = np.asarray(angles, dtype=float)
    if a.shape != (2,) or not np.isfinite(a).all():
        raise ValueError('Two finite angular coordinates required')
    return (state, *[round(float(v), 8) for v in a])


def endpoint_outcomes(cases):
    observations = defaultdict(lambda: defaultdict(set))
    for c in cases:
        key = pose_key(c['state_key'], c['selected_angles'])
        for e in c['executions']:
            if type(e['success']) is not bool:
                raise ValueError('Boolean observed outcome required')
            observations[key][e['repeat']].add(e['success'])
    result = {}
    for key, repetitions in observations.items():
        conflict = any(len(v) != 1 for v in repetitions.values())
        result[key] = dict(success=None if conflict else float(np.mean(
            [next(iter(v)) for v in repetitions.values()])),
            repeats=len(repetitions), conflicting=conflict)
    return result


def diagnose(cases):
    outcomes = endpoint_outcomes(cases)
    groups = defaultdict(list)
    for c in cases:
        groups[c['method']].append(c)
    methods = {}
    for method, rows in sorted(groups.items()):
        scored = [c for c in rows if c['queries_data']]
        gains, net_rewards, disagreement = [], [], []
        known, visited, opportunity, lower_harm = 0, 0, 0, 0
        paired = { (c['state_key'], c['start']): c for c in groups['stay'] }
        for c in scored:
            q = c['queries_data']
            a0 = np.mean(q[0]['p3'])
            best = min(np.mean(t['p3']) for t in q if t['kind'] != 'return_verification')
            gains.append((a0-best)/a0)
            # Retrospective episode-local utility only, NOT the trained reward.
            net_rewards.append((a0-best)/a0 - .01*(len(q)-1))
            p0 = min(q, key=lambda t: t['p3'][0])['angles']
            p1 = min(q, key=lambda t: t['p3'][1])['angles']
            disagreement.append(pose_key(c['state_key'], p0) != pose_key(c['state_key'], p1))
            unique = {pose_key(c['state_key'], t['angles']) for t in q}
            visited += len(unique)
            available = [outcomes[k]['success'] for k in unique
                         if k in outcomes and outcomes[k]['success'] is not None]
            known += len(available)
            success = np.mean([e['success'] for e in c['executions']])
            opportunity += bool(available and max(available) > success)
            stay = paired.get((c['state_key'], c['start']))
            if stay:
                lower_harm += bool(best < a0 and success < np.mean(
                    [e['success'] for e in stay['executions']]))
        methods[method] = dict(searches=len(rows),
            success=float(np.mean([e['success'] for c in rows for e in c['executions']])),
            mean_queries=float(np.mean([c['queries'] for c in rows])),
            immediate_stop_fraction=float(np.mean([c['queries']==1 for c in rows])),
            scored_searches=len(scored),
            mean_relative_best_gain=float(np.mean(gains)) if gains else None,
            fraction_nonpositive_episode_local_utility=float(np.mean(np.array(net_rewards)<=0)) if gains else None,
            score_member_argmin_disagreement=float(np.mean(disagreement)) if disagreement else None,
            visited_pose_instances=visited, visited_with_measured_outcome=known,
            searches_with_known_better_visited_outcome=opportunity,
            lower_best_score_but_worse_than_stay=lower_harm)
    return dict(methods=methods, unique_executed_poses=len(outcomes),
        conflicting_pose_noise_labels=sum(v['conflicting'] for v in outcomes.values()),
        note='Known-better count is a lower bound from measured coordinates only; missing outcomes remain unknown. Utility diagnostic uses episode initial scale, not policy reward. No counterfactual training conclusion.')


def training_diagnostic(episodes, updates, *, episodes_per_update=8):
    if type(episodes_per_update) is not int or episodes_per_update < 1:
        raise ValueError('Positive complete batch size required')
    grouped=defaultdict(list)
    for e in episodes:
        grouped[(e['run'],e['policy_update']+1)].append(e)
    rows=[]
    for u in updates:
        es=grouped[(u['run'],u['update'])]
        if len(es)!=episodes_per_update:
            raise ValueError('Expected declared complete episodes per update')
        settle=u['reward_settlement'];scale=settle['denominator']
        gains=np.array([(e['initial_accel']-e['best_accel'])/scale for e in es])
        costs=np.array([.01*(e['queries']-1) for e in es])
        residual=gains-costs-np.array(settle['rewards'])
        if not np.allclose(residual,0,atol=1e-9,rtol=0):
            raise ValueError('Reward does not reconcile with best-score improvement and billed queries')
        rows.append(dict(run=u['run'],update=u['update'],scale=scale,
            mean_gain=float(gains.mean()),mean_cost=float(costs.mean()),
            mean_reward=float(np.mean(settle['rewards'])),
            positive_reward_fraction=float(np.mean(np.array(settle['rewards'])>0)),
            immediate_stop_fraction=float(np.mean([e['queries']==1 for e in es])),
            stop_probability=u['means']['mean_stop_probability'],kl=u['means']['approximate_kl'],
            clip_fraction=u['means']['clip_fraction'],transitions=u['transitions'],
            max_reward_reconciliation_error=float(abs(residual).max()),
            covered_states=len({e['initial_identity']['state_key'] for e in es})))
    return rows


def spatial_diagnostic(cases):
    """Ranks on the observed endpoint subset, plus offline p9 candidate audit."""
    from scipy.stats import spearmanr
    outcomes=endpoint_outcomes(cases);values=defaultdict(list);audits=[]
    for c in cases:
        qs=[q for q in c['queries_data'] if q['kind']!='return_verification']
        for q in qs:values[pose_key(c['state_key'],q['angles'])].append(q)
        if not qs:continue
        picks={p:min(qs,key=lambda q:np.mean(q[p])) for p in ('p3','p9')}
        keys={p:pose_key(c['state_key'],q['angles']) for p,q in picks.items()}
        audits.append(dict(method=c['method'],state_key=c['state_key'],start=c['start'],
            changed=keys['p3']!=keys['p9'],
            p3_success=outcomes.get(keys['p3'],{}).get('success'),
            p9_success=outcomes.get(keys['p9'],{}).get('success')))
    states=[]
    for state in sorted({c['state_key'] for c in cases}):
        keys=[k for k in values if k[0]==state]
        known=[k for k in keys if outcomes.get(k,{}).get('success') is not None]
        row=dict(state_key=state,scored_poses=len(keys),labelled_poses=len(known))
        for p in ('p3','p9'):
            scores=[np.mean([np.mean(q[p]) for q in values[k]]) for k in known]
            successes=[outcomes[k]['success'] for k in known]
            row[p+'_spearman']=float(spearmanr(scores,successes).statistic) if len(set(scores))>1 and len(set(successes))>1 else None
        states.append(row)
    return dict(states=states,prefix_choices=audits,
        note='Observational, adaptively sampled endpoints; ranks are not unbiased whole-domain correlations. p9 is offline ranking only, not an executed p9 policy.')
