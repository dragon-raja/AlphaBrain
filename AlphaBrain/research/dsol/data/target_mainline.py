"""Outcome-blind target-model mainline schedules aligned to the frozen G panel.

This is a scientific planning contract, never a controller binding or permit.
Models retain their admitted native operation horizons and action chunk sizes.
"""
from dataclasses import dataclass
from collections import Counter
from copy import deepcopy
from .native_baseline import HORIZONS
from .native_view_panel import views
from ..execution.manifest import digest


@dataclass(frozen=True)
class OperationSpec:
    name: str
    horizons: dict
    replan_steps: int
    settle_steps: int
    warmup_per_block: int = 0

    def validate(self):
        if (not isinstance(self.name,str) or not self.name.strip()
                or set(self.horizons)!=set(HORIZONS)
                or any(type(n) is not int or n<1 for n in self.horizons.values())
                or type(self.replan_steps) is not int or self.replan_steps<1
                or type(self.settle_steps) is not int or self.settle_steps<0
                or type(self.warmup_per_block) is not int or self.warmup_per_block not in (0,1)):
            raise ValueError('Explicit native operation contract required')


def _identity(key):
    parts=key.split('::')
    if len(parts)!=4 or parts[1] not in HORIZONS:
        raise ValueError('Explicit original state identity required')
    return parts[1],parts[2],int(parts[3])


def compile_mainline(schedules, training_records, recovery_rows, models):
    """Freeze primary baseline/G and secondary local/random without GPU work.

    Training and operation model calls are separate units. Reused endpoints are
    not multiplied into independent outcomes. No reuse credit is guessed here.
    """
    if set(schedules)!={0,1,2} or len(training_records)!=400:
        raise ValueError('Three original schedules and 400 training states required')
    train_ids=[]
    for record in training_records:
        state=record['state'];key=state['pair_key'];_,task,initial=_identity(key)
        if (state['split']!='train' or state['task_id']!=task
                or state['initialization']['init_state_index']!=initial or not 0<=initial<20):
            raise ValueError('Training state identity/split differs')
        train_ids.append(key)
    training_tasks={_identity(k)[1] for k in train_ids}
    if len(set(train_ids))!=400 or len(training_tasks)!=20:
        raise ValueError('Twenty training tasks / twenty unique initials required')
    for seed,rows in schedules.items():
        if len(rows)!=2048 or [r['episode'] for r in rows]!=list(range(2048)):
            raise ValueError('Exact 2048-episode original prefix required')
        for row in rows:
            if not 0<=row['initial']<400 or row['state_key']!=train_ids[row['initial']]:
                raise ValueError('Training schedule identity differs')
            suite,task,_=_identity(row['state_key'])
            if (row['suite'],row['task'])!=(suite,task):raise ValueError('Training task differs')
        if {r['initial'] for r in rows}!=set(range(400)):
            raise ValueError('Training initial coverage reduced')
    if len(recovery_rows)!=3200:
        raise ValueError('Complete forty-task eight-start panel required')
    groups={};splits={}
    for row in recovery_rows:
        suite,task,initial=_identity(row['state_key'])
        if ((row['suite'],row['task'])!=(suite,task) or row['method']!='B-seed-0'
                or row['execution_repeats']!=[0,1] or row['query_cap']!=8
                or not 1<=row['view_id']<=24 or row['angles_deg']!=views()[row['view_id']]):
            raise ValueError('Original recovery identity/geometry differs')
        split=row['split'];expected=range(20,30) if split=='validation' else range(40,50)
        if split not in ('seen_new_initials','validation','unseen_test') or initial not in expected:
            raise ValueError('Evaluation initial/split differs')
        if (task in training_tasks)!=(split=='seen_new_initials'):
            raise ValueError('Search training/evaluation task membership differs')
        if task in splits and splits[task]!=split:raise ValueError('Mixed task splits')
        splits[task]=split;groups.setdefault(row['state_key'],[]).append(row)
    if (len(groups)!=400 or len({_identity(k) for k in groups})!=400
            or Counter(splits.values())!=dict(seen_new_initials=20,validation=8,unseen_test=12)
            or Counter(_identity(k)[0] for k in groups)!=dict.fromkeys(HORIZONS,100)):
        raise ValueError('Full split/suite coverage differs')
    for rows in groups.values():
        if len(rows)!=8 or len({r['view_id'] for r in rows})!=8 or {r['start'] for r in rows}!=set(range(8)):
            raise ValueError('Eight unique starting directions required per state')
    if not models or len({m.name for m in models})!=len(models):raise ValueError('Unique model IDs required')
    output={}
    for model in models:
        model.validate();phases={k:[] for k in ('baseline','ours','local','random_from_local')}
        def trial(row,method):
            r=deepcopy(row);r.update(model=model.name,method=method,
                source_max_steps=row['max_steps'],max_steps=model.horizons[row['suite']],
                replan_steps=model.replan_steps,settle_steps=model.settle_steps)
            r['trial_id']=digest(dict(model=model.name,method=method,state=r['state_key'],view=r['view_id']))
            return r
        phases['baseline']=[trial(r,'stay') for r in recovery_rows]
        for rows in groups.values():
            r=trial(rows[0],'reference');r.update(view_id=0,angles_deg=[0.,0.],start=None)
            r['trial_id']=digest(dict(model=model.name,method='reference',state=r['state_key'],view=0))
            phases['baseline'].append(r)
        phases['ours']=[trial(r,f'G-seed-{seed}') for r in recovery_rows for seed in (0,1,2)]
        for method in ('local','random_from_local'):phases[method]=[trial(r,method) for r in recovery_rows]
        all_rows=[r for rr in phases.values() for r in rr]
        if len({r['trial_id'] for r in all_rows})!=19600:raise ValueError('Duplicate scientific trial')
        budgets={}
        for phase,rr in phases.items():
            budgets[phase]=dict(operation_episodes=2*len(rr),search_sessions=0 if phase=='baseline' else len(rr),
                score_calls=0 if phase=='baseline' else 20*len(rr),
                operation_calls=sum(2*((r['max_steps']+r['replan_steps']-1)//r['replan_steps']) for r in rr),
                warmup_calls=sum((count+1)//2 for count in Counter((r['suite'],r['method']) for r in rr).values())*model.warmup_per_block)
        # Suite-specific checkpoints cannot share a block. In a complete
        # 32-episode publication, four odd suite counts may require 18 blocks
        # rather than 16. Reserve that ceiling without changing episode order,
        # batch size, or mixing seed/policy versions to fill a block.
        training_block_cap=192*(18 if model.warmup_per_block else 16)
        budgets['training']=dict(search_episodes=6144,score_calls=98304,
            warmup_calls=training_block_cap*model.warmup_per_block,operation_episodes=0,
            maximum_two_episode_blocks=training_block_cap)
        budgets['engineering']=dict(model_calls=1024)
        total=sum(b.get('score_calls',0)+b.get('operation_calls',0)+b.get('warmup_calls',0)+b.get('model_calls',0) for b in budgets.values())
        output[model.name]=dict(operation_spec=dict(horizons=model.horizons,replan_steps=model.replan_steps,settle_steps=model.settle_steps),
            phases=phases,budgets=budgets,total_call_cap_before_verified_reuse=total,
            primary_call_cap=sum(sum(budgets[k].get(x,0) for x in ('score_calls','operation_calls','warmup_calls','model_calls')) for k in ('training','baseline','ours','engineering')),
            operation_episodes=39200,primary_operation_episodes=26400)
    return dict(schema='TARGET_NATIVE_MAINLINE_PLAN_V1',status='SCIENCE_SCHEDULE_ONLY_RUNTIME_BINDING_PENDING',
        dispatch_enabled=False,models=output,training_schedule_hashes={str(k):digest(v) for k,v in schedules.items()},
        source_recovery_hash=digest(recovery_rows),score_selection='Preserve admitted prefix3 scalar and minimum-visited selection; no automatic best-prefix selection',
        training=dict(tasks=20,states=400,seeds=[0,1,2],episodes_per_seed=2048,episodes_per_update=32,updates=64,query_cap=8),
        evaluation=dict(tasks=40,states=400,starts_per_state=8,operation_repeats=2,search_seeds=[0,1,2],splits=dict(seen_new_initials=20,validation=8,unseen_test=12)),
        total_call_cap_before_verified_reuse=sum(x['total_call_cap_before_verified_reuse'] for x in output.values()))
