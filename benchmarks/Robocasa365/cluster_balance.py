"""CPU-only remaining-work observer for cross-model finish-time balancing.

Consumes timing and coverage, never success labels. Publishing a recommendation
does not itself move tasks: the admitted handoff retains that responsibility.
"""
from collections import Counter,defaultdict
from pathlib import Path
import time

from AlphaBrain.research.dsol.data.artifacts import read,sha,write
from AlphaBrain.research.dsol.execution.balance import plan_allocation


def observe(config_file):
    cfg=read(config_file); summaries={}; work={}; qualified={}
    for model,spec in cfg['models'].items():
        plan=read(spec['plan']); targets=Counter(); initial_tasks={}
        for initial,entry in plan['selections'][model].items():
            selection=read(entry['path']);task=selection['task'];initial_tasks[initial]=task
            targets[task]+=len(selection['evaluation_views'])*len(plan['execution_repeats'])
        completed=Counter(); seconds=defaultdict(float);steps=defaultdict(int);paths=set()
        for directory in spec['results']:
            for path in Path(directory).glob('**/execution_*.json'):
                if str(path) in paths:raise ValueError('Overlapping timing result roots')
                paths.add(str(path));row=read(path)
                if row['model']!=model or row['initial'] not in initial_tasks:
                    raise ValueError('Unexpected model/initial in timing inventory')
                task=initial_tasks[row['initial']]
                completed[task]+=1;seconds[task]+=row['seconds'];steps[task]+=row['steps']
        if any(completed[t]>targets[t] for t in completed):raise ValueError('Timing inventory exceeds frozen budget')
        count=sum(completed.values());all_seconds=sum(seconds.values());all_steps=sum(steps.values())
        if not count:
            summaries[model]=dict(status='WAITING_FOR_TIMING',completed=0,total=sum(targets.values()))
            continue
        task_rows={}; remaining_seconds=0
        for task,target in targets.items():
            reliable=completed[task]>=cfg['minimum_task_trials']
            # Unvisited tasks use horizon-based runtime, not an assumption that
            # their success rate equals the early tasks' observed success rate.
            cost=seconds[task]/completed[task] if reliable else all_seconds/all_steps*plan['task_horizons'][task]
            cost+=cfg['startup_seconds_per_trial']
            left=target-completed[task];remaining_seconds+=left*cost
            task_rows[task]=dict(completed=completed[task],total=target,seconds_per_trial=cost,
                                 estimate='observed_task_mean' if reliable else 'unseen_task_horizon_projection')
        work[model]=remaining_seconds/spec['workers_per_gpu']
        qualified[model]=(count>=cfg['minimum_model_trials'] and
                          sum(completed[t]>=cfg['minimum_task_trials'] for t in targets)>=cfg['minimum_observed_tasks'])
        summaries[model]=dict(status='MEASURED',completed=count,total=sum(targets.values()),
            remaining_gpu_hours=work[model]/3600,qualified=qualified[model],tasks=task_rows)
    recommendation=plan_allocation(work,{m:cfg['models'][m]['gpus'] for m in work},
        cfg['total_gpus'],qualified=qualified,minimum_gain=cfg['minimum_gain']) if len(work)==len(cfg['models']) else dict(status='WAITING_FOR_TIMING',change_recommended=False)
    result=dict(status='PASS',sample_unix=time.time(),config_sha256=sha(config_file),models=summaries,
        recommendation=recommendation,
        caveat='Timing projection, not a deadline guarantee. Repartition requires host/model admission and an atomic disjoint handoff; no result-based scientific filtering.')
    write(cfg['output'],result)
    if recommendation.get('change_recommended'):
        write(str(Path(cfg['output']).with_name('rebalance-recommended.json')),result)
    return result


def watch(config_file):
    cfg=read(config_file);fingerprint=sha(config_file);deadline=time.monotonic()+cfg['watch_seconds']
    while True:
        if sha(config_file)!=fingerprint:raise ValueError('Observer configuration changed')
        result=observe(config_file)
        print(result['recommendation'],flush=True)
        if all(r['completed']==r['total'] for r in result['models'].values()) or time.monotonic()>=deadline:return result
        time.sleep(cfg['interval_seconds'])


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    (watch if args.watch else observe)(args.config)
