"""Host-local standard queues, shared campaign stop, disjoint bounded evaluation blocks.

Preparation imports no policy or simulator. Real execution requires an explicit
permit separate from the frozen campaign. No automatic retry or partial replay.
"""
import argparse
import json
import signal
from pathlib import Path
from AlphaBrain.research.dsol.data.artifacts import read
from AlphaBrain.research.dsol.data.continuous_evaluation_blocks import blocks,STAGES,phase
from AlphaBrain.research.dsol.execution.manifest import file_digest,task_key,validate
from AlphaBrain.research.dsol.execution.queue import prepare,run as run_queue,_durable_write,MissingResult
from AlphaBrain.research.dsol.execution.campaign_control import CampaignGuard
from benchmarks.Robocasa365.probe_queue import ProbeBackend


def inventory(campaign):
    cfg=read(campaign);result={}
    if cfg['schema']!='continuous_evaluation_fleet_v1' or len(cfg['conditions'])!=3:
        raise ValueError('Three separate frozen model conditions required')
    if len({v['host'] for v in cfg['conditions'].values()})!=3:raise ValueError('One condition per host required')
    for model,spec in cfg['conditions'].items():
        paired=read(spec['paired']);nominal=read(spec['nominal'])
        if (file_digest(spec['paired'])!=spec['paired_sha256'] or file_digest(spec['nominal'])!=spec['nominal_sha256']
                or file_digest(spec['runtime'])!=spec['runtime_sha256']
                or nominal['paired_config']!=spec['paired'] or nominal['paired_config_sha256']!=spec['paired_sha256']):
            raise ValueError('Model config binding changed')
        result[model]=blocks(read(paired['design']),read(paired['schedule']))
    return cfg,result


def request_for(campaign,model,index):
    cfg,all_blocks=inventory(campaign);spec=cfg['conditions'][model];block=all_blocks[model][index]
    return dict(campaign=str(Path(campaign).resolve()),campaign_sha256=file_digest(campaign),model=model,
        host=spec['host'],stage=phase(index),block=block,paired=spec['paired'],nominal=spec['nominal'])


def validate_block(request,output):
    req=read(request) if not isinstance(request,dict) else request
    root=Path(output);path=root/'complete.json'
    if not path.exists():raise MissingResult(str(path))
    done=read(path)
    from AlphaBrain.research.dsol.execution.manifest import digest
    if done['status']!='COMPLETE' or done['request_sha256']!=digest(req):raise ValueError('Block identity changed')
    required={'results.json','status.json','timing.json','model-attempts.jsonl'}
    if not required.issubset(done['artifacts']):raise ValueError('Required block evidence missing')
    actual_files={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p!=path}
    if actual_files!=set(done['artifacts']):raise ValueError('Unbound block evidence')
    for relative,expected in done['artifacts'].items():
        p=root/relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or file_digest(p)!=expected:
            raise ValueError('Completed block artifact changed')
    payload=read(root/'results.json');block=req['block'];tasks=payload['tasks'];searches=payload['searches']
    status=read(root/'status.json')
    config=req['paired'] if block['kind']=='paired' else req['nominal']
    if (payload['block_index']!=block['index'] or status['status']!='COMPLETE'
            or status['config_sha256']!=file_digest(config)):
        raise ValueError('Incomplete or wrong block')
    expected={(r['cohort'],r['initial'],r['start'],r['method'],s)
              for r in block['rows'] for s in r['execution_repeats']}
    actual=[(r['cohort'],r['initial'],r['start'],r['method'],r['repeat']) for r in tasks]
    if len(actual)!=len(set(actual)) or set(actual)!=expected or len(searches)!=block['searches']:
        raise ValueError('Missing/duplicate/out-of-block results')
    if block['kind']=='paired':
        from .continuous_evaluation import audit_block
        audit_block(tasks,searches)
        for row in tasks:
            asset=Path(row['trajectory'])
            if not asset.is_relative_to(root) or file_digest(asset)!=row['trajectory_sha256']:
                raise ValueError('Trajectory binding differs')
        for row in searches:
            if not 1<=row['queries']<=(1 if row['method']=='stay' else 8):
                raise ValueError('Search query budget differs')
    else:
        for row in tasks:
            asset=root/f"{row['cohort']}-{row['initial']:02d}-seed-{row['repeat']}"/'execution.npz'
            if file_digest(asset)!=row['execution_sha256']:raise ValueError('Nominal trajectory binding differs')
    counts=status['calls']
    if set(counts)!=set(block['limits']) or any(type(v)!=int or not 0<=v<=block['limits'][k] for k,v in counts.items()):
        raise ValueError('Block budget exceeded')
    if (counts['execution']!=sum(r['replans'] for r in tasks)
            or (block['kind']=='paired' and counts['score']!=2*sum(r['queries'] for r in searches))):
        raise ValueError('Results and call accounting differ')
    ledger=root/'model-attempts.jsonl'
    attempts=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    if [x['index'] for x in attempts]!=list(range(1,sum(counts.values())+1)):
        raise ValueError('Block durable call coverage differs')
    for stage,count in counts.items():
        if [x['stage_index'] for x in attempts if x['identity']['stage']==stage]!=list(range(1,count+1)):
            raise ValueError('Per-stage ledger mismatch')
    return dict(status='PASS_COMPLETE_BLOCK',receipt_sha256=file_digest(path),calls=counts)


def prior_stages(campaign,stage):
    cfg,all_blocks=inventory(campaign)
    previous=[] if stage=='admission' else ['admission'] if stage=='capacity' else ['admission','capacity']
    for model in all_blocks:
        for name in previous:
            for index in STAGES[name]:
                validate_block(request_for(campaign,model,index),Path(cfg['output'])/model/'blocks'/f'{index:02d}')


def worker(request_path,output,*,execute=False):
    req=read(request_path)
    if not execute:raise ValueError('Explicit block execution required')
    if file_digest(req['campaign'])!=req['campaign_sha256'] or req!=request_for(req['campaign'],req['model'],req['block']['index']):
        raise ValueError('Unbound block request')
    guard=CampaignGuard(req['campaign'],req['host'],req['stage'])
    try:
        guard.check()
        proof=read(guard.root/f'phase-verified-{guard.host}-{guard.stage}.json')
        if proof!=dict(manifest_sha256=guard.digest,stage=guard.stage):raise ValueError('Coordinator phase gate missing')
        if req['block']['kind']=='paired':from .continuous_evaluation import run
        else:from .continuous_nominal import run
        config=req['paired'] if req['block']['kind']=='paired' else req['nominal']
        run(config,output,execute=True,block_index=req['block']['index'],guard=guard.check)
        root=Path(output)
        artifacts={str(p.relative_to(root)):file_digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
        from AlphaBrain.research.dsol.execution.manifest import digest
        _durable_write(root/'complete.json',dict(status='COMPLETE',request_sha256=digest(req),artifacts=artifacts))
        return validate_block(req,output)
    except BaseException as error:
        guard.fail(error);raise


class BlockBackend(ProbeBackend):
    def __init__(self,plan,root):
        super().__init__(plan,root)
        self.guard=CampaignGuard(self.runtime['campaign'],self.runtime['host'],self.runtime['stage'])

    def admit(self,slot):
        if not self.guard.check(dispatch=True):return False
        if self.runtime['workers_per_gpu']!=1 or slot['host']!=self.runtime['host']:
            raise ValueError('One worker per admitted local GPU required')
        return self.gpu_available(slot)

    def start(self,task,slot,attempt):
        if not self.guard.check(dispatch=True):raise RuntimeError('Dispatch cutoff reached')
        return super().start(task,slot,attempt)

    def poll(self,handle):
        self.guard.check()
        return super().poll(handle)

    def worker_command(self,task):
        return [self.runtime['python'],'-u','-m','benchmarks.LIBERO.continuous_evaluation_fleet',
                'worker','--request',task['payload']['request'],'--output',task['output'],'--execute']

    def validate(self,task):return validate_block(task['payload']['request'],task['output'])


def factory(plan,root):return BlockBackend(plan,root)


def build(campaign,model,stage,directory):
    cfg,all_blocks=inventory(campaign);spec=cfg['conditions'][model];directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False);tasks=[];artifacts=[]
    for index in STAGES[stage]:
        req=request_for(campaign,model,index);f=directory/f'request-{index:02d}.json';_durable_write(f,req)
        addresses=[dict(model=model,task=r['task'],initial=r['state_key'],view=f"{r['start']}:{r['method']}",
            stream='execution',repeat=s,noise=f"paired-seed-{s}") for r in req['block']['rows'] for s in r['execution_repeats']]
        tasks.append(dict(key=task_key(f"{cfg['run_id']}-{model}-{stage}",req['campaign_sha256'],addresses),addresses=addresses,
            output=str(Path(cfg['output'])/model/'blocks'/f'{index:02d}'),payload=dict(request=str(f))))
        artifacts.append(dict(path=str(f),sha256=file_digest(f)))
    rt=read(spec['runtime']);rt.update(campaign=str(Path(campaign).resolve()),stage=stage,workers_per_gpu=1)
    artifacts+=rt['source_artifacts']+[dict(path=str(Path(campaign).resolve()),sha256=file_digest(campaign)),
        dict(path=spec['paired'],sha256=spec['paired_sha256']),dict(path=spec['nominal'],sha256=spec['nominal_sha256'])]
    unique={}
    for artifact in artifacts:
        if artifact['path'] in unique and unique[artifact['path']]!=artifact:
            raise ValueError('Conflicting artifact identities')
        unique[artifact['path']]=artifact
    artifacts=list(unique.values())
    slots=rt['slots'][:1] if stage=='admission' else rt['slots'][:8]
    if len(slots)!=(1 if stage=='admission' else 8):raise ValueError('Explicit one/eight-GPU phase required')
    plan=dict(format='dsol_work_queue_v1',experiment_id=f"{cfg['run_id']}-{model}-{stage}",
        protocol_sha256=file_digest(campaign),slots=slots,tasks=tasks,artifacts=artifacts,
        backend=dict(module='benchmarks.LIBERO.continuous_evaluation_fleet',factory='factory'),runtime=rt)
    validate(plan);_durable_write(directory/'queue.json',plan);return plan


def run_phase(plan_path,control,*,execute=False):
    plan=read(plan_path);r=plan['runtime'];guard=CampaignGuard(r['campaign'],r['host'],r['stage'])
    if not execute:raise ValueError('No execution without explicit flag')
    def interrupt(_signal,_frame):raise KeyboardInterrupt('Coordinator stop requested')
    previous={s:signal.signal(s,interrupt) for s in (signal.SIGTERM,signal.SIGHUP)}
    try:
        with guard.lease():
            if not guard.check(dispatch=True):raise RuntimeError('Dispatch cutoff reached before launch')
            prior_stages(r['campaign'],r['stage'])
            _durable_write(guard.root/f'phase-verified-{guard.host}-{guard.stage}.json',dict(manifest_sha256=guard.digest,stage=guard.stage))
            prepare(plan,control);backend=BlockBackend(plan,control)
            state=run_queue(control,backend,timeout_seconds=86400,allow_dispatch=lambda:guard.check(dispatch=True))
            _durable_write(Path(guard.config['control'])/f"done-{r['host']}-{r['stage']}.json",dict(status=state['status'],manifest_sha256=guard.digest,stage=guard.stage))
    except BaseException as error:guard.fail(error);raise
    finally:
        for signum,handler in previous.items():signal.signal(signum,handler)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['worker','run-phase'])
    for k in ('request','output','plan','control'):p.add_argument('--'+k)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.mode=='worker':worker(a.request,a.output,execute=a.execute)
    else:run_phase(a.plan,a.control,execute=a.execute)
