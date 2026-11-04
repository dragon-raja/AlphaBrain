"""Paired reward-adaptation diagnostics; no reward fitting or model execution."""
import numpy as np


def local_incentive_diagnostic(episodes, settlements):
    """Rescore ALL B training decisions without learning or new interactions.

    B trajectories are a common observed-action reference, not counterfactual
    trajectories of G/L. Strata use the formal training scale at that batch.
    This diagnoses incentive changes and coverage, never task effectiveness.
    """
    labels=['below_0.5S','0.5S_to_S','S_to_2S','at_least_2S']
    buckets={k:dict(query_decisions=0,query_increments=0,positive_gain_decisions=0,
        G_nonpositive_L_positive=0,L_nonpositive_G_positive=0,G_reward_sum=0.,L_reward_sum=0.) for k in labels}
    n=0
    for episode in episodes:
        s=settlements[(episode['seed'],episode['policy_update']+1)]
        scale,tau=s['scale'],s['tau']
        if not np.isfinite([scale,tau]).all() or min(scale,tau)<=0:raise ValueError('Invalid training scale')
        for t in episode['transitions']:
            before,after,dq=(t[k] for k in ('best_previous','best_current','delta_queries'))
            if (not np.isfinite([before,after]).all() or not 0<=after<=before
                    or type(dq) is not int or not 0<=dq<=2):raise ValueError('Invalid raw reward transition')
            if not dq:continue
            ratio=before/scale;index=int(np.searchsorted([.5,1.,2.],ratio,side='right'))
            bucket=buckets[labels[index]];gain=before-after
            g=gain/scale-.01*dq;l=float(np.log1p(gain/(after+tau)))-.01*dq
            bucket['query_decisions']+=1;bucket['query_increments']+=dq
            bucket['positive_gain_decisions']+=int(gain>0)
            bucket['G_nonpositive_L_positive']+=int(g<=0<l)
            bucket['L_nonpositive_G_positive']+=int(l<=0<g)
            bucket['G_reward_sum']+=g;bucket['L_reward_sum']+=l
        n+=1
    return dict(reference='all B continuation training episodes, no outcome selection',
        episodes=n,buckets=buckets,model_calls=0,
        interpretation='Observed-action reward recalculation; not G/L policy effectiveness or local-optimum escape proof')


def summarize(episodes, searches, *, bootstrap_draws=5000, seed=20261027):
    """Task-weighted outcomes with paired task/seed exploratory bootstrap.

    Each arm must have the same three seeds and the full same-case panel. Shared
    stay/nominal results are not counted as independently rerun observations.
    """
    methods=[f'{a}-seed-{s}' for a in ('B','G','L') for s in (0,1,2)]
    by_method={m:[] for m in methods};by_search={m:[] for m in methods}
    baseline={};panels={};search_panels={}
    for r in episodes:
        key=(r['state_key'],r['view_id'],r['repeat'])
        m=r['method']
        if m not in by_method or any(type(r[k]) is not bool for k in ('success','stay_success','nominal_success')):
            raise ValueError('Unrecognized method or nonbinary outcome')
        if key in panels.setdefault(m,set()):raise ValueError('Duplicate outcome')
        panels[m].add(key);by_method[m].append(r)
        value=(r['task'],r['stay_success'],r['nominal_success'])
        if key in baseline and baseline[key]!=value:raise ValueError('Unpaired baseline')
        baseline[key]=value
    for r in searches:
        key=(r['state_key'],r['view_id']);m=r['method']
        if m not in by_search or key in search_panels.setdefault(m,set()):raise ValueError('Duplicate or unknown search')
        if not 1<=r['queries']<=8 or len(r['validation_scores'])!=2 or len(r['reference_validation_scores'])!=2:
            raise ValueError('Incomplete search/independent diagnostics')
        search_panels[m].add(key);by_search[m].append(r)
    expected=panels.get(methods[0],set());expected_search={(k[0],k[1]) for k in expected}
    if (not expected or any(panels.get(m)!=expected or search_panels.get(m)!=expected_search for m in methods)
            or expected!={(a,b,s) for a,b in expected_search for s in (0,1)}):
        raise ValueError('Incomplete matched three-arm/three-seed panel')
    tasks=sorted({v[0] for v in baseline.values()})
    def task_mean(rows,fn):
        return [float(np.mean([fn(r) for r in rows if r['task']==task])) for task in tasks]
    per_method={};matrices={}
    for m in methods:
        rows=by_method[m];sr=by_search[m]
        vals=np.array(task_mean(rows,lambda r:r['success']))
        matrices[m]=vals
        rescue=sum(r['success'] and not r['stay_success'] for r in rows)
        harm=sum(not r['success'] and r['stay_success'] for r in rows)
        independent=np.array([np.asarray(r['reference_validation_scores'])-np.asarray(r['validation_scores']) for r in sr])
        selected_improvement=np.array([r['best_accel']<r['initial_accel'] for r in sr])
        per_method[m]=dict(episodes=len(rows),successes=sum(r['success'] for r in rows),
            success_rate=float(vals.mean()),rescue=rescue,harm=harm,
            net_recovery_pp=100*float(np.mean(task_mean(rows,lambda r:int(r['success'])-int(r['stay_success'])))),
            mean_queries=float(np.mean(task_mean(sr,lambda r:r['queries']))),
            immediate_stop_rate=float(np.mean([r['queries']==1 and r['reason']=='stop' for r in sr])),
            independent_mean_improvement_rate=float(np.mean(independent.mean(axis=1)>0)),
            independent_both_members_improve_rate=float(np.mean((independent>0).all(axis=1))),
            selected_score_improved=int(selected_improvement.sum()),
            independent_nonimprovements_after_selected_improvement=int(((independent.mean(axis=1)<=0)&selected_improvement).sum()),
            task_success_rates=dict(zip(tasks,map(float,vals))))
    base_rows=[dict(task=v[0],stay_success=v[1],nominal_success=v[2]) for v in baseline.values()]
    baseline_summary=dict(unique_perturbed_execution_pairs=len(baseline),unique_initials=len({k[0] for k in expected}),
        tasks=len(tasks),stay_success_rate=float(np.mean(task_mean(base_rows,lambda r:r['stay_success']))),
        nominal_success_rate=float(np.mean(task_mean(base_rows,lambda r:r['nominal_success']))))
    arms={};arrays={}
    for arm in ('B','G','L'):
        arrays[arm]=np.stack([matrices[f'{arm}-seed-{s}'] for s in (0,1,2)])
        rows=[per_method[f'{arm}-seed-{s}'] for s in (0,1,2)]
        rates=arrays[arm].mean(axis=1)
        arms[arm]=dict(success_rate=float(rates.mean()),seed_success_rates=rates.tolist(),
            seed_standard_deviation=float(rates.std(ddof=1)),
            mean_queries=float(np.mean([r['mean_queries'] for r in rows])),
            net_recovery_pp=float(np.mean([r['net_recovery_pp'] for r in rows])),
            rescue=sum(r['rescue'] for r in rows),harm=sum(r['harm'] for r in rows))
    rng=np.random.default_rng(seed)
    seed_indices=rng.integers(0,3,size=(bootstrap_draws,3))
    task_indices=rng.integers(0,len(tasks),size=(bootstrap_draws,len(tasks)))
    contrasts={}
    for arm in ('G','L'):
        difference=arrays[arm]-arrays['B']
        draws=difference[seed_indices[:,:,None],task_indices[:,None,:]].mean(axis=(1,2))
        contrasts[arm+'-B']=dict(success_difference_pp=float(100*difference.mean()),
            exploratory_crossed_task_seed_interval_pp=(100*np.quantile(draws,[.025,.975])).tolist(),
            seed_differences_pp=(100*difference.mean(axis=1)).tolist(),bootstrap_draws=bootstrap_draws)
    return dict(arms=arms,per_method=per_method,baseline=baseline_summary,contrasts=contrasts,
        limits=['B warm start; not from-scratch no-calibration evidence',
                'eight updates; null may mean insufficient adaptation',
                'development tasks; not a new blind test',
                'two scoring members; disagreement is not a confidence bound',
                'viewpoints and repeats within a task are correlated',
                'reward values across arms have different units and are not a success metric'])
