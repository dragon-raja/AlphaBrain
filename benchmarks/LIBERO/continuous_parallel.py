"""Four-worker synchronous PPO through the existing durable execution queue.

Each worker keeps one frozen VLA resident for its block. Workers exit at the
update barrier; only the central CPU learner can optimize or publish a version.
No processes or model imports are started by importing this adapter.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time
from AlphaBrain.research.dsol.data.artifacts import read
from AlphaBrain.research.dsol.execution.manifest import digest,file_digest,task_key,verify_artifacts
from AlphaBrain.research.dsol.execution.queue import prepare,run as run_queue,MissingResult
from AlphaBrain.research.dsol.execution.rollout_budget import RolloutBudget
from benchmarks.Robocasa365.probe_queue import ProbeBackend


class CollectorBackend(ProbeBackend):
    def admit(self,slot):
        if (self.runtime['mode']!='continuous_parallel_collection' or self.runtime['workers_per_gpu']!=1
                or slot['host']!=self.runtime['host']):raise ValueError('Single-host distinct-GPU collectors only')
        return self.gpu_available(slot)

    def worker_command(self,task):
        return [self.runtime['python'],'-u','-m','benchmarks.LIBERO.continuous_parallel',
                'worker','--request',task['payload']['request'],'--output',task['output'],'--execute']

    def validate(self,task):
        p=Path(task['output'])/'status.json'
        if not p.exists():raise MissingResult(str(p))
        s=read(p);req=read(task['payload']['request'])
        if (s['status']!='COMPLETE' or s['request_sha256']!=file_digest(task['payload']['request'])
                or s['policy_sha256']!=req['policy_sha256'] or s['version']!=req['version']
                or set(s['records'])!={str(r['episode']) for r in req['rows']}):
            raise ValueError('Worker version/coverage mismatch')
        budget=RolloutBudget(req['budget_root'])
        units=budget.recoverable([f"{req['prefix']}:{r['episode']}" for r in req['rows']])
        for key,ref in s['records'].items():
            if file_digest(ref['path'])!=ref['sha256'] or units[f"{req['prefix']}:{key}"]['receipt_sha256']!=ref['sha256']:
                raise ValueError('Rollout digest changed')
        return dict(status='PASS_COLLECTED_ONLY',receipt_sha256=file_digest(p))


def factory(plan,root):return CollectorBackend(plan,root)


def worker(request_path,output,*,execute=False,shared_runtime=None):
    req=read(request_path)
    if not execute or req.get('authorization')!='EXPLICIT_COLLECTION_ONLY':raise ValueError('Collection not authorized')
    if file_digest(req['policy'])!=req['policy_sha256']:raise ValueError('Policy snapshot changed')
    cfg=read(req['config']);plan=read(cfg['design'])
    if cfg.get('authorization') not in ('EXPLICIT_TRAINING_ONLY','EXPLICIT_ADMISSION_ONLY'):
        raise ValueError('Training/admission manifest not authorized')
    import torch
    from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
    from AlphaBrain.research.dsol.selectors.continuous_parallel import seeded_episode
    from AlphaBrain.research.dsol.data.continuous_training_store import atomic_file,pack,write_json
    from .continuous_training import LiveEpisodes
    torch.set_num_threads(2)
    payload=torch.load(req['policy'],map_location='cpu',weights_only=True)
    if payload['version']!=req['version']:raise ValueError('Stale policy version')
    policy=ContinuousActorCritic(4608).eval();policy.load_state_dict(payload['state_dict'])
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    budget=RolloutBudget(req['budget_root']);owner=req['owner'];version=req['policy_sha256']
    backend=LiveEpisodes(cfg,plan,root,global_budget=(budget,owner,version,req['prefix']),
        runtime=shared_runtime.get('runtime') if shared_runtime is not None else None)
    records={};started=time.monotonic()
    try:
        for row in req['rows']:
            key=f"{req['prefix']}:{row['episode']}";budget.claim(key,owner,version)
            query_root=Path(req['query_root'])/f"{row['episode']:04d}-queries"
            episode,metrics=seeded_episode(policy,backend,row,query_root,version=req['version'],
                seed=plan['training']['seed'],source_kind=cfg['source_kind'])
            p=root/f"episode-{row['episode']:04d}.pt"
            atomic_file(p,lambda f:torch.save(pack(dict(episode=episode,metrics=metrics)),f))
            ref=dict(path=str(p),sha256=file_digest(p));records[str(row['episode'])]=ref
            budget.complete(key,owner,version,metrics['scoring_trajectories'],ref['sha256'])
        status=dict(status='COMPLETE',version=req['version'],policy_sha256=req['policy_sha256'],
            request_sha256=file_digest(request_path),records=records,elapsed_seconds=time.monotonic()-started,
            episode_timings=backend.timings)
        if shared_runtime is not None:shared_runtime['runtime']=backend.runtime
        write_json(root/'status.json',status);return status
    except BaseException as error:
        budget.fail(repr(error));write_json(root/'failure.json',dict(status='FAILED_STOPPED',error=repr(error)));raise


def collect_batch(cfg, runtime, policy, rows, version, batch_root, query_root, control_root, budget, *, workers=4,resident=None,call_caps=None):
    """Freeze actor once; admit the complete declared quota, never fastest-N."""
    import torch
    from AlphaBrain.research.dsol.data.continuous_training_store import atomic_file,write_json
    root=Path(batch_root);control=Path(control_root)
    if workers not in (1,4,8) or len(rows)%workers:raise ValueError('One/four/eight-worker complete quotas required')
    if root.exists():
        raise ValueError('Existing batch requires explicit completed-batch recovery; never automatic replay')
    root.mkdir(parents=True);Path(query_root).mkdir(parents=True,exist_ok=True)
    snapshot=root/'policy.pt'
    atomic_file(snapshot,lambda f:torch.save(dict(version=version,state_dict=policy.state_dict()),f))
    policy_sha=file_digest(snapshot);prefix=root.name
    tasks=[];artifacts=[dict(path=str(snapshot),sha256=policy_sha),
        dict(path=cfg,sha256=file_digest(cfg))]
    units=[]
    for w in range(workers):
        selected=rows[w::workers];owner=f'resident-collector-{w}' if resident is not None else f'{prefix}-collector-{w}'
        req=dict(authorization='EXPLICIT_COLLECTION_ONLY',config=cfg,policy=str(snapshot),policy_sha256=policy_sha,
            version=version,owner=owner,prefix=prefix,rows=selected,query_root=str(query_root),budget_root=str(budget.root))
        if resident is not None:req['resident_output']=str(root/f'worker-{w}')
        path=root/f'request-{w}.json';write_json(path,req)
        artifacts.append(dict(path=str(path),sha256=file_digest(path)))
        addresses=[dict(model='pi05-batch1',task=r['task'],initial=r['state_key'],view=f"episode-{r['episode']}",
            stream='score',repeat=r['episode'],noise=f"fixed-two-members-initial-{r['initial']}") for r in selected]
        tasks.append(dict(key=task_key(prefix,file_digest(cfg),addresses),addresses=addresses,
            payload=dict(request=str(path)),output=str(root/f'worker-{w}')))
        for row in selected:
            cap=16 if call_caps is None else call_caps[row['episode']]
            if type(cap) is not int or cap<2 or cap>16 or cap%2:raise ValueError('Invalid exact admission quota')
            units.append(dict(key=f"{prefix}:{row['episode']}",owner=owner,version=policy_sha,cap=cap))
    budget.reserve(units)
    plan=dict(format='dsol_work_queue_v1',experiment_id=prefix,protocol_sha256=file_digest(cfg),
        slots=runtime['slots'][:workers],tasks=tasks,artifacts=artifacts+runtime['source_artifacts'],
        backend=dict(module='benchmarks.LIBERO.continuous_parallel',factory='factory'),
        runtime={**runtime,'mode':'continuous_parallel_collection','workers_per_gpu':1})
    if len(plan['slots'])!=workers or len({s['gpu_id'] for s in plan['slots']})!=workers:
        raise ValueError('Distinct physical GPUs required')
    if resident is not None:
        try:
            resident.collect([t['payload']['request'] for t in tasks])
            control.mkdir(parents=True,exist_ok=False)
            backend=CollectorBackend(plan,control);records={};attempts=[]
            for task in tasks:
                backend.validate(task)
                status=read(Path(task['output'])/'status.json')
                for k,value in status['records'].items():
                    if int(k) in records:raise ValueError('Duplicate resident episode')
                    records[int(k)]=value
                attempts.extend(json.loads(l) for l in (Path(task['output'])/'model-attempts.jsonl').read_text().splitlines())
            attempts.sort(key=lambda a:a['global_index'])
            write_json(root/'resident-complete.json',dict(status='COMPLETE',version=version,
                       records=records,requests=[file_digest(t['payload']['request']) for t in tasks]))
            return records,attempts
        except BaseException as error:
            budget.fail(repr(error));raise
    prepare(plan,control);backend=CollectorBackend(plan,control)
    try:run_queue(control,backend,poll_seconds=1,timeout_seconds=runtime.get('batch_timeout_seconds',7200))
    except BaseException as error:budget.fail(repr(error));raise
    return completed_batch(control)


def completed_batch(control_root):
    """Revalidate existing COMPLETE queue; never start or recover a failed worker."""
    from AlphaBrain.research.dsol.execution.queue import load
    plan,state=load(control_root);verify_artifacts(plan)
    if state['status']!='COMPLETE':raise ValueError('Batch not complete; investigate before explicit recovery')
    backend=CollectorBackend(plan,control_root);records={};attempts=[]
    for task in plan['tasks']:
        handle=state['tasks'][task['key']].get('handle')
        if handle and backend.alive(handle):raise ValueError('Old collector still alive')
        backend.validate(task);s=read(Path(task['output'])/'status.json')
        for k,v in s['records'].items():
            if int(k) in records:raise ValueError('Duplicate collector ownership')
            records[int(k)]=v
        attempts.extend(json.loads(l) for l in (Path(task['output'])/'model-attempts.jsonl').read_text().splitlines())
    attempts.sort(key=lambda a:a['global_index'])
    return records,attempts


def admission(config_path,execution_path,output,control_root,*,execute=False):
    cfg=read(config_path)
    if not execute or cfg.get('authorization')!='EXPLICIT_ADMISSION_ONLY':raise ValueError('Admission not authorized')
    from .continuous_training import validate_inputs
    _,cfg,plan=validate_inputs(config_path)
    import numpy as np
    import torch
    from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
    from AlphaBrain.research.dsol.selectors.continuous_parallel import validate_rollout
    from AlphaBrain.research.dsol.data.continuous_training_store import write_json,numpy_tree
    runtime=read(execution_path);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    cap=runtime.get('admission_cap',128)
    if not isinstance(cap,int) or not 1<=cap<=128:raise ValueError('Invalid bounded admission budget')
    budget=RolloutBudget.create(root/'budget',cap);torch.set_num_threads(2)
    policy=ContinuousActorCritic(4608).eval()
    initial=torch.load(cfg['initial_checkpoint'],map_location='cpu',weights_only=True)
    if initial['optimizer_updates']!=0:raise ValueError('Admission must use the original initialization')
    policy.load_state_dict(initial['state_dict']);schedule=read(cfg['training_schedule'])
    # Four distinct training tasks, predetermined without looking at rewards.
    rows=[next(r for r in schedule if r['initial']==i) for i in (0,8,16,24)]
    serial,_=collect_batch(str(Path(config_path).resolve()),runtime,policy,rows,0,root/'serial',
        root/'serial-queries',Path(control_root)/'serial',budget,workers=1)
    # Reference counts are known before replay; a differing path fails at its
    # reserved bound, rather than reserving three independent worst-case runs.
    call_caps={i:int(torch.load(ref['path'],map_location='cpu',weights_only=True)['metrics']['scoring_trajectories'])
               for i,ref in serial.items()}
    resident_enabled=runtime.get('collector_lifecycle','per_batch')=='resident'
    warm=None
    with ExitStack() as cleanup:
        resident=None
        if resident_enabled:
            from .continuous_resident import ResidentPool
            resident=cleanup.enter_context(ResidentPool(str(Path(config_path).resolve()),runtime,
                root/'resident',Path(control_root)/'resident',workers=4))
        parallel,_=collect_batch(str(Path(config_path).resolve()),runtime,policy,rows,0,root/'parallel',
            root/'parallel-queries',Path(control_root)/'parallel',budget,workers=4,resident=resident,call_caps=call_caps)
        if resident is not None:
            warm,_=collect_batch(str(Path(config_path).resolve()),runtime,policy,rows,1,root/'warm',
                root/'warm-queries',Path(control_root)/'warm',budget,workers=4,resident=resident,call_caps=call_caps)
    files=0
    def exact(a,b,label):
        if isinstance(a,dict):
            if a.keys()!=b.keys():raise ValueError('Parallel keys differ: '+label)
            for k in a:exact(a[k],b[k],label+'/'+k)
        elif isinstance(a,list):
            if len(a)!=len(b):raise ValueError('Parallel lengths differ: '+label)
            for i,(x,y) in enumerate(zip(a,b)):exact(x,y,label+'/'+str(i))
        elif not np.array_equal(a,b):raise ValueError('Parallel values differ: '+label)
    for row in rows:
        i=row['episode'];values=[]
        for records in (serial,parallel):
            value=torch.load(records[i]['path'],map_location='cpu',weights_only=True)
            validate_rollout(policy,value,row,0);value=numpy_tree(value)
            value['metrics'].pop('elapsed_seconds');values.append(value)
        exact(*values,f'episode-{i}')
        if warm is not None:
            value=torch.load(warm[i]['path'],map_location='cpu',weights_only=True)
            validate_rollout(policy,value,row,1);value=numpy_tree(value)
            value['metrics'].pop('elapsed_seconds')
            value['metrics']['policy_update']=0;value['episode']['policy_update']=0
            exact(values[0],value,f'warm-episode-{i}')
        for q in range(1,values[0]['metrics']['queries']+1):
            a=root/'serial-queries'/f'{i:04d}-queries'/f'query-{q:02d}.npz'
            b=root/'parallel-queries'/f'{i:04d}-queries'/f'query-{q:02d}.npz'
            with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
                exact(dict(x),dict(y),f'episode-{i}/query-{q}')
            if warm is not None:
                c=root/'warm-queries'/f'{i:04d}-queries'/f'query-{q:02d}.npz'
                with np.load(a,allow_pickle=False) as x,np.load(c,allow_pickle=False) as y:
                    exact(dict(x),dict(y),f'warm-episode-{i}/query-{q}')
            files+=1
    state=budget.inspect()
    if state['reserved'] or state['spent']>cap or state['failed']:raise ValueError('Admission budget inconsistent')
    report=dict(status='PASS_EXACT_PARALLEL',episodes=12 if warm is not None else 8,training_updates=0,manipulation_actions=0,
        calls=state['spent'],limit=cap,paired_query_files=files,workers=4,
        collector_lifecycle='resident' if resident_enabled else 'per_batch',
        initial_checkpoint_sha256=file_digest(cfg['initial_checkpoint']),source_commit=cfg['source_commit'])
    write_json(root/'status.json',report);return report


def train(config_path,execution_path,output,control_root,*,execute=False,resume=False):
    cfg=read(config_path)
    if not execute or cfg.get('authorization')!='EXPLICIT_TRAINING_ONLY':raise ValueError('Training not authorized')
    from .continuous_training import validate_inputs
    _,cfg,plan=validate_inputs(config_path);runtime=read(execution_path)
    proof=runtime['parallel_admission']
    resident_enabled=runtime.get('collector_lifecycle','per_batch')=='resident'
    if runtime.get('collector_lifecycle','per_batch') not in ('resident','per_batch'):
        raise ValueError('Unknown collector lifecycle')
    if resident_enabled and (resume or read(proof['path']).get('collector_lifecycle')!='resident'):
        raise ValueError('Resident lifecycle needs separate admission; interrupted pools require explicit audit')
    if (file_digest(proof['path'])!=proof['sha256'] or read(proof['path'])['status']!='PASS_EXACT_PARALLEL'
            or len(runtime['slots'])!=4):raise ValueError('Four-worker exact admission required')
    import torch
    from AlphaBrain.research.dsol.selectors.continuous_ppo import ContinuousPPO,ContinuousPPOConfig
    from AlphaBrain.research.dsol.selectors.continuous_search_policy import ContinuousActorCritic
    from AlphaBrain.research.dsol.selectors.continuous_parallel import ingest,RNG_RULE
    from AlphaBrain.research.dsol.data.continuous_training_store import write_json,atomic_file
    from .continuous_training_gate import audit_first_batch
    torch.set_num_threads(2);policy=ContinuousActorCritic(4608)
    initial=torch.load(cfg['initial_checkpoint'],map_location='cpu',weights_only=True)
    if initial['optimizer_updates']!=0:raise ValueError('Original untrained initialization required')
    policy.load_state_dict(initial['state_dict'])
    settings=plan['training'];p=settings['ppo']
    config=ContinuousPPOConfig(episodes_per_update=32,updates=16,seed=settings['seed'],learning_rate=p['learning_rate'],
        n_epochs=p['n_epochs'],batch_size=p['batch_size'],max_grad_norm=p['max_grad_norm'])
    root=Path(output);started=time.monotonic()
    with ContinuousPPO(policy,config,read(cfg['training_schedule']),None,root,
            binding=dict(source_kind=cfg['source_kind'],launch_config_sha256=file_digest(config_path),
                execution_config_sha256=file_digest(execution_path),source_commit=cfg['source_commit'],
                sampling_rng=RNG_RULE,cpu_threads=2),resume=resume) as trainer, ExitStack() as cleanup:
        budget=RolloutBudget(root/'budget') if resume else RolloutBudget.create(root/'budget',8192)
        resident=None
        if resident_enabled:
            from .continuous_resident import ResidentPool
            resident=cleanup.enter_context(ResidentPool(str(Path(config_path).resolve()),runtime,
                root/'resident',Path(control_root)/'resident',workers=4))
        if trainer.update and not (root/'first-batch-gate.json').exists():
            raise ValueError('First update has no validated gate; do not resume automatically')
        while trainer.update<16:
            k=trainer.update;rows=trainer.schedule[k*32:(k+1)*32]
            control=Path(control_root)/f'update-{k:03d}'
            if control.exists():
                if not resume:raise ValueError('Explicit resume required')
                records,attempts=completed_batch(control)
            else:
                records,attempts=collect_batch(str(Path(config_path).resolve()),runtime,trainer.policy,rows,k,
                    root/'batches'/f'update-{k:03d}',root/'episodes',control,budget,resident=resident)
            ledger=root/'model-attempts.jsonl'
            previous=[json.loads(l) for l in ledger.read_text().splitlines()] if ledger.exists() else []
            by_index={a['global_index']:a for a in previous}
            for a in attempts:
                a=dict(a,index=a['global_index'])
                if a['global_index'] in by_index and by_index[a['global_index']]!=a:raise ValueError('Changed call ledger')
                by_index[a['global_index']]=a
            all_calls=[dict(by_index[i],index=i) for i in sorted(by_index)]
            if sorted(by_index)!=list(range(1,budget.inspect()['spent']+1)):raise ValueError('Global call coverage gap')
            atomic_file(ledger,lambda f:f.write((''.join(json.dumps(a)+'\n' for a in all_calls)).encode()))
            ingest(trainer,records);trainer.optimize()
            if trainer.queries*2!=budget.inspect()['spent'] or budget.inspect()['reserved']:
                raise ValueError('Global spent/reserved accounting mismatch')
            if trainer.update==1:
                trainer.store.begin('first_batch_gate',1)
                try:write_json(root/'first-batch-gate.json',audit_first_batch(trainer));trainer.store.finish()
                except BaseException as error:trainer.store.fail(error);raise
            write_json(root/'progress.json',dict(status='RUNNING',episodes=trainer.cursor,updates=trainer.update,
                queries=trainer.queries,calls=budget.inspect()['spent'],elapsed_seconds=time.monotonic()-started))
        result=dict(status='COMPLETE',episodes=512,updates=16,queries=trainer.queries,calls=budget.inspect()['spent'],
            scoring_trajectories=trainer.queries*2,source_kind=cfg['source_kind'],manipulation_actions=0,
            config_sha256=file_digest(config_path),elapsed_seconds=time.monotonic()-started)
        # A completed training receipt must also imply all resident workers exited.
        cleanup.close()
        write_json(root/'status.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['worker','train','admission'])
    p.add_argument('--request');p.add_argument('--config');p.add_argument('--execution');p.add_argument('--output',required=True)
    p.add_argument('--control');p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    try:
        if a.mode=='worker':result=worker(a.request,a.output,execute=a.execute)
        elif a.mode=='admission':result=admission(a.config,a.execution,a.output,a.control,execute=a.execute)
        else:result=train(a.config,a.execution,a.output,a.control,execute=a.execute,resume=a.resume)
        print(json.dumps(result,indent=2))
    except BaseException as error:
        if a.mode!='worker' and Path(a.output).exists():
            from AlphaBrain.research.dsol.execution.queue import _durable_write
            _durable_write(Path(a.output)/'failure.json',dict(status='FAILED_STOPPED',error=repr(error)))
        raise
