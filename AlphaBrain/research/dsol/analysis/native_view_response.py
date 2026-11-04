"""Complete-grid descriptive recovery statistics, never success-based selection."""
import numpy as np


def rank(values):
    values=np.asarray(values);return np.array([np.sum(values<v)+(np.sum(values==v)-1)/2 for v in values])


def analyze(states, angles, tasks, scores):
    results=[]
    for state in states:
        key=state['state_key'];ts=[t for t in tasks if t['state_key']==key];ss=[s for s in scores if s['state_key']==key]
        identities=[(t['view_id'],t['repeat']) for t in ts]
        if len(identities)!=108 or set(identities)!={(v,s) for v in range(27) for s in range(4)}:raise ValueError('Incomplete/duplicate state execution surface')
        scoreids=[(s['view_id'],s['member']) for s in ss]
        if len(scoreids)!=108 or set(scoreids)!={(v,s) for v in range(27) for s in range(4)}:raise ValueError('Incomplete/duplicate state score surface')
        success=np.zeros((27,4),bool);score=np.zeros((27,4),float)
        for t in ts:
            if type(t['success']) is not bool:raise ValueError('Boolean task success required')
            success[t['view_id'],t['repeat']]=t['success']
        for s in ss:score[s['view_id'],s['member']]=s['score']
        if not np.isfinite(score).all():raise ValueError('Nonfinite score')
        primary=score[:,:2].mean(1);validation=score[:,2:].mean(1);chosen=int(np.argmin(primary))
        rates=success.mean(1);a,b=rank(primary),rank(rates)
        rho=float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None
        old=success[[25,26]].reshape(-1);selected=np.tile(success[chosen],2);nominal=np.tile(success[0],2)
        lost=nominal & ~old
        rows=[]
        for v,angle in enumerate(angles):
            candidates=np.flatnonzero(np.max(np.abs((np.asarray(angles)-angle)/[10.,10.]),axis=1)<=1+1e-12)
            local=int(candidates[np.argmin(primary[candidates])])
            rows.append(dict(view_id=v,angles_deg=angle,success=int(success[v].sum()),episodes=4,
                success_rate=float(rates[v]),primary_accel=float(primary[v]),validation_accel=float(validation[v]),
                primary_members=score[v,:2].tolist(),validation_members=score[v,2:].tolist(),
                delta_accel=float(primary[v]-primary[0]),nominal_loss=int(success[0].sum()-success[v].sum()),
                local_diagnostic_candidates=candidates.tolist(),local_low_accel_view=local,
                local_selected_success=int(success[local].sum()),local_oracle_success=int(success[candidates].any(0).sum())))
        results.append(dict(state_key=key,task=state['task'],initial=state['initial'],views=rows,
            primary_min_view=chosen,validation_min_view=int(np.argmin(validation)),
            within_state_score_success_spearman=rho,
            original_starts=dict(episodes=8,stay_success=int(old.sum()),nominal_success=int(nominal.sum()),
                selected_success=int(selected.sum()),nominal_losses=int(lost.sum()),
                rescues=int((~old & selected).sum()),harms=int((old & ~selected).sum()),
                recovery_of_nominal_losses=int((lost & selected).sum())),
            uniform_random_expected_success_rate=float(rates.mean()),
            global_oracle_success_rate=float(success.any(0).mean()),
            scope='Global min uses all 27 primary scores; local diagnostic uses one-step grid neighborhood. Neither is an 8-query online PPO result.'))
    task_names=sorted({s['task'] for s in states})
    per_view=[]
    for v,angle in enumerate(angles):
        task_rates=[float(np.mean([s['views'][v]['success_rate'] for s in results if s['task']==task])) for task in task_names]
        per_view.append(dict(view_id=v,angles_deg=angle,task_macro_success=float(np.mean(task_rates)),
            mean_primary_accel=float(np.mean([s['views'][v]['primary_accel'] for s in results])),
            by_task=dict(zip(task_names,task_rates))))
    sensitivity=[]
    for omitted in [None]+task_names:
        kept=[s for s in results if s['task']!=omitted]
        if not kept:continue
        counts={k:sum(s['original_starts'][k] for s in kept) for k in results[0]['original_starts']}
        sensitivity.append(dict(omitted_task=omitted,counts=counts,
            nominal_minus_perturbed=(counts['nominal_success']-counts['stay_success'])/counts['episodes'],
            selected_minus_perturbed=(counts['selected_success']-counts['stay_success'])/counts['episodes']))
    return dict(states=results,per_view=per_view,leave_one_task_out=sensitivity,
        tasks=len(task_names),initial_states=len(states),
        episodes=len(tasks),score_members=len(scores),
        interpretation='Exploratory four-task surface. Execution seeds share states; view observations are not independent tasks. Oracle sees outcomes and is only an upper bound.')
