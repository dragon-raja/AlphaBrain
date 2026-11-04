"""Task-clustered descriptive few-shot development results, never actor selection."""
from collections import defaultdict
import numpy as np


def summarize(tasks,*,expected=2336,bootstrap_samples=10000,seed=20261029):
    key=lambda r:(r['state_key'],r['start'],r['repeat'])
    seen=set();groups=defaultdict(list)
    for row in tasks:
        identity=(row['method'],key(row))
        if identity in seen or type(row['success']) is not bool:raise ValueError('Duplicate/invalid result')
        seen.add(identity);groups[row['method']].append(row)
    stay={key(r):r for r in groups.get('stay',[])}
    def stats(rows,reference):
        buckets=defaultdict(list)
        for r in rows:
            if key(r) in reference:
                other=reference[key(r)];buckets[r['task']].append((int(r['success'])-int(other['success']),
                    int(r['success'] and not other['success']),int(not r['success'] and other['success'])))
        if not buckets:return dict(paired_rows=0,tasks=0)
        values=np.asarray([np.mean(v,axis=0) for _,v in sorted(buckets.items())]);mean=values.mean(0)
        rng=np.random.default_rng(seed);draws=rng.integers(len(values),size=(bootstrap_samples,len(values)))
        boot=values[draws,0].mean(1);ci=np.quantile(boot,[.025,.975]).tolist()
        return dict(paired_rows=sum(map(len,buckets.values())),tasks=len(values),net_rescue=float(mean[0]),
            rescued=float(mean[1]),damaged=float(mean[2]),task_cluster_net_rescue_95=ci)
    summaries={}
    for method,rows in sorted(groups.items()):
        tasks_success=defaultdict(list)
        for r in rows:tasks_success[r['task']].append(int(r['success']))
        summaries[method]=dict(episodes=len(rows),successes=sum(r['success'] for r in rows),
            task_equal_success_rate=float(np.mean([np.mean(v) for v in tasks_success.values()])),
            mean_queries=float(np.mean([r['queries'] for r in rows])),versus_stay=stats(rows,stay))
    aggregates={}
    for init in ('source','random'):
        for mode in ('first_batch_fixed','window'):
            for version in (4,8):
                names=[f'actor:{init}-{mode}-seed-{s}:u{version}' for s in (0,1,2)]
                if not all(name in groups for name in names):continue
                rows=[r for name in names for r in groups[name]]
                # Average all seeds within each task before resampling tasks.
                aggregates[f'{init}-{mode}:u{version}']=stats(rows,stay)
    contrasts={}
    for comparator in ('source-first_batch_fixed','random-window'):
        left=[];right={}
        for s in (0,1,2):
            a=f'actor:source-window-seed-{s}:u8';b=f'actor:{comparator}-seed-{s}:u8'
            if a not in groups or b not in groups:break
            # Match each target seed to the same comparator seed, then cluster
            # by task; repeated baseline outcomes are not independent samples.
            left.extend(dict(r,state_key=r['state_key']+f':seed{s}') for r in groups[a])
            for r in groups[b]:
                value=dict(r,state_key=r['state_key']+f':seed{s}');right[key(value)]=value
        else:contrasts['source-window:u8 minus '+comparator+':u8']=stats(left,right)
    candidate=aggregates.get('source-window:u8');complete=len(tasks)==expected
    decision='PENDING_COMPLETE_PANEL'
    if complete and candidate is not None:
        decision=('DEVELOPMENT_RESCUE_TARGET_MET' if candidate['tasks']==8 and candidate['net_rescue']>=.05
            and candidate['task_cluster_net_rescue_95'][0]>0 and candidate['damaged']<=.1 else 'DEVELOPMENT_TARGET_NOT_MET')
    return dict(status='COMPLETE' if complete else 'PARTIAL',episodes=len(tasks),expected=expected,
        methods=summaries,seed_aggregates=aggregates,matched_contrasts=contrasts,decision=decision,
        scope='Eight development tasks; task-cluster percentile intervals, all seeds included; not held-out confirmation or real robot evidence',
        bootstrap=dict(unit='task',samples=bootstrap_samples,seed=seed))
