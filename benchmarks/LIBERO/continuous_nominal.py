"""Matched nominal-view reference: eight states by four execution seeds.

Reuses the paired evaluation's input/noise/time-limit bindings, without scoring,
searching, duplicating the two perturbation starts, or loading a search actor.
"""
import argparse
from pathlib import Path
import math
import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.execution.manifest import file_digest
from .continuous_evaluation import validate_inputs,EvaluationCalls,append_event
from .local_search_rollout import rollout_local_episode


def nominal_rows(plan,rows):
    selected={}
    for row in rows:
        key=(row['cohort'],row['initial'])
        if key not in selected:selected[key]=dict(row,method='nominal',angles_deg=[0.,0.],start='nominal')
    expected={(t['cohort'],i) for t in plan['evaluation']['tasks'] for i in t['indices']}
    if set(selected)!=expected or len(selected)!=8:raise ValueError('Nominal reference needs the same eight physical states')
    if any(r['execution_repeats']!=[0,1,2,3] for r in selected.values()):
        raise ValueError('Nominal reference needs the same four execution seeds')
    budget=sum(4*len(t['indices'])*math.ceil(plan['evaluation']['horizons'][t['suite']]/5)
               for t in plan['evaluation']['tasks'])
    if budget!=1760:raise ValueError('Matched nominal horizon budget changed')
    return list(selected.values())


def run(config_path,output,*,execute=False):
    cfg=read(config_path)
    if not execute or cfg.get('authorization')!='EXPLICIT_NOMINAL_EVALUATION_ONLY':
        raise ValueError('Nominal execution not authorized')
    parent=cfg['paired_config']
    if file_digest(parent)!=cfg['paired_config_sha256'] or cfg['execution_cap']!=1760:
        raise ValueError('Nominal parent binding/budget changed')
    paired,plan,_=validate_inputs(parent)
    rows=nominal_rows(plan,read(paired['schedule']))
    root=Path(output);root.mkdir(parents=True,exist_ok=False);calls=None;tasks=[]
    try:
        from .search_runtime import create_policy_runtime
        from .search_live import simulator,check_simulator_dependencies
        from .search_domain import exact_input
        from AlphaBrain.research.dsol.data.flow_noise import ExplicitFlowNoiseBank
        sims={k:read(v['sim_config']) for k,v in paired['cohorts'].items()}
        for sim in sims.values():check_simulator_dependencies(sim)
        banks={k:ExplicitFlowNoiseBank(Path(v['execution_noise']),verify_file=True) for k,v in paired['cohorts'].items()}
        runtime=create_policy_runtime(sims['A'],root,1760)
        calls=EvaluationCalls(runtime,root,dict(execution=1760));calls.stage='execution'
        for row in rows:
            spec=paired['cohorts'][row['cohort']];sim=sims[row['cohort']]
            suite=next(t['suite'] for t in plan['evaluation']['tasks'] if t['task']==row['task'])
            identity={k:row[k] for k in ('cohort','initial','state_key','task','start','method')}
            calls.identity=identity
            for repeat in row['execution_repeats']:
                case=root/f"{row['cohort']}-{row['initial']:02d}-seed-{repeat}";case.mkdir()
                with simulator(spec['sim_config'],sim,row['initial'],case) as request:
                    pose=request(dict(kind='continuous_info'))
                    initial=request(dict(kind='query_pose'))
                    if not np.allclose(initial['receipt']['camera_to_world'],pose['reference'],rtol=0,atol=1e-9):
                        raise ValueError('Nominal reference is not the canonical camera pose')
                    response=request(dict(kind='begin_execution'));exact_input(initial['example'],response['example'])
                    result,arrays=rollout_local_episode(request,calls.infer,response['example'],banks[row['cohort']],
                        row['state_key'],repeat,identity,plan=plan,suite=suite,
                        record=lambda e:append_event(case/'events.jsonl',e))
                np.savez_compressed(case/'execution.npz',**arrays)
                result.update(identity,execution_sha256=file_digest(case/'execution.npz'))
                write(case/'result.json',result);tasks.append(result)
        write(root/'status.json',dict(status='COMPLETE',tasks=tasks,completed_tasks=len(tasks),
            calls=calls.counts,score_calls=0,search_queries=0,training_updates=0,config_sha256=file_digest(config_path)))
    except BaseException as error:
        write(root/'failure.json',dict(status='FAILED_STOPPED',error=repr(error),completed_tasks=len(tasks),
            calls=calls.counts if calls else {},automatic_retry=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args();run(a.config,a.output,execute=a.execute)
