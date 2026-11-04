"""Task-paired decision rules fixed before installation-validation outcomes."""
import numpy as np
from ..data.coordinate_validation import CRITERIA


def summarize(pairs, *, criteria):
    if criteria!=CRITERIA:raise ValueError('Predeclared criteria changed')
    keys=[(r['state_key'],r['offset'],r['start'],r['seed'],r['repeat']) for r in pairs]
    if len(keys)!=2592 or len(set(keys))!=2592:raise ValueError('Missing or repeated G comparison')
    tasks=sorted({r['task'] for r in pairs})
    if len(tasks)!=12:raise ValueError('Twelve task clusters required')
    states={r['state_key'] for r in pairs}
    expected={(s,o,start,seed,repeat) for s in states for o in (-15.,0.,15.)
        for start in range(4) for seed in range(3) for repeat in (0,1)}
    if len(states)!=36 or set(keys)!=expected:raise ValueError('Incomplete paired state/offset/seed grid')
    shared={}
    for r in pairs:
        if any(type(r[k]) is not bool for k in ('G','stay','home','local','reference')):
            raise ValueError('Binary task outcomes required')
        if any(type(r[k]) is not int or not 1<=r[k]<=8 for k in ('G_queries','local_queries')):
            raise ValueError('Bounded integer query counts required')
        key=(r['state_key'],r['offset'],r['start'],r['repeat'])
        value=(r['task'],r['stay'],r['home'],r['local'],r['reference'])
        if key in shared and shared[key]!=value:raise ValueError('Seed baselines differ')
        shared[key]=value
    def describe(rows):
        def avg(fn):return float(np.mean([np.mean([fn(r) for r in rows if r['task']==t]) for t in tasks]))
        return dict(success_percent={m:100*avg(lambda r:r[m]) for m in ('G','stay','home','local','reference')},
            net_recovery_pp=100*avg(lambda r:int(r['G'])-int(r['stay'])),
            home_advantage_pp=100*avg(lambda r:int(r['G'])-int(r['home'])),
            rescue_percent=100*avg(lambda r:r['G'] and not r['stay']),
            harm_percent=100*avg(lambda r:r['stay'] and not r['G']),
            mean_queries={m:avg(lambda r:r[m+'_queries']) for m in ('G','local')})
    primary=[r for r in pairs if r['offset'] in criteria['primary_offsets']]
    if len(primary)!=1728:raise ValueError('Wrong primary nonzero-offset population')
    summary=describe(primary);contrasts={};rng=np.random.default_rng(criteria['bootstrap_seed'])
    indices=rng.integers(0,len(tasks),size=(criteria['bootstrap_draws'],len(tasks)))
    for other in ('stay','home','local'):
        values=np.array([np.mean([int(r['G'])-int(r[other]) for r in primary if r['task']==t]) for t in tasks])
        contrasts[other]=dict(difference_pp=float(100*values.mean()),
            paired_task_95_interval_pp=(100*np.quantile(values[indices].mean(axis=1),[.025,.975])).tolist(),
            task_differences_pp=dict(zip(tasks,(100*values).tolist())))
    seeds={str(seed):describe([r for r in primary if r['seed']==seed]) for seed in (0,1,2)}
    gates=dict(net_recovery=summary['net_recovery_pp']>=criteria['minimum_net_recovery_pp'] and contrasts['stay']['paired_task_95_interval_pp'][0]>0,
        beats_home=summary['home_advantage_pp']>=criteria['minimum_home_advantage_pp'] and contrasts['home']['paired_task_95_interval_pp'][0]>0,
        harm=summary['harm_percent']<=criteria['maximum_harm_vs_stay_pp'],
        seeds_vs_stay=sum(s['net_recovery_pp']>0 for s in seeds.values())>=criteria['minimum_positive_seeds_vs_stay'],
        seeds_vs_home=sum(s['home_advantage_pp']>=0 for s in seeds.values())>=criteria['minimum_nonnegative_seeds_vs_home'])
    local_gate=(contrasts['local']['paired_task_95_interval_pp'][0]>-criteria['local_noninferiority_margin_pp']
                and summary['mean_queries']['G']<=summary['mean_queries']['local'])
    decision=('READY_G_MAIN_WITHIN_VALIDATED_SCOPE' if local_gate else 'ACTIVE_SEARCH_ONLY_PPO_NOT_ESTABLISHED') if all(gates.values()) else 'HOLD_G_MAIN'
    return dict(decision=decision,primary=summary,gates=gates,ppo_cost_effectiveness_gate=local_gate,
        contrasts=contrasts,per_seed=seeds,per_offset={str(int(o)):describe([r for r in pairs if r['offset']==o]) for o in (-15.,0.,15.)},
        per_task={t:{m:100*float(np.mean([r[m] for r in primary if r['task']==t])) for m in ('G','stay','home','local','reference')} for t in tasks},
        inference='Paired bootstrap over twelve tasks; three fixed policy seeds averaged within task, not independent scenes.',
        scope='Frozen native pi0.5 and G on declared shifted installations; not robot admission or proof against every possible shortcut.')
