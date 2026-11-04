"""Known, settled PPO-boundary continuation after a planned resource handoff.

The optimizer/collector stay on their original frozen source and configuration.
Only new unlaunched phases move to the shared pool. Unknown calls, pending CPU
updates, changed artifacts or a still-live old owner block continuation.
"""
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor,as_completed
import subprocess,sys,shlex,time,json
from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.execution.manifest import file_digest as sha,digest
from AlphaBrain.research.dsol.execution.phase_budget import PhaseBudget
from AlphaBrain.research.dsol.execution.learner_publication import execute_update
from AlphaBrain.research.dsol.execution.authority_wire import ssh_command
from .prefix_campaign import PrefixCampaign
from .controlled_campaign import collection_records


def process_alive(receipt):
    try:
        text=Path(f"/proc/{receipt['pid']}/stat").read_text();fields=text[text.rfind(')')+2:].split()
        return int(fields[19])==receipt['start_ticks'] and fields[0] not in ('Z','X')
    except FileNotFoundError:return False


def frozen_stage(source,config,phase,campaign,log,*,kind='training'):
    if kind not in ('training','target_recovery'):raise ValueError('Reviewed frozen phase kind required')
    """A single separately fenced standard phase; its durable intent forbids replay."""
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True)
    intent=log.with_suffix('.attempt.json')
    with intent.open('x') as f:
        json.dump(dict(source=str(source),config=str(config),campaign=str(campaign),phase=phase,automatic_retry=False),f);f.flush()
        import os;os.fsync(f.fileno())
    script=('import sys,json;sys.path.insert(0,sys.argv[1]);'
            'from pathlib import Path;read=lambda p:json.loads(Path(p).read_text());'
            'from benchmarks.LIBERO.controlled_campaign import run_stage;'
            'print(json.dumps(run_stage(read(sys.argv[2]),sys.argv[3],sys.argv[4],kind=sys.argv[5],cpu=False)))')
    with log.open('xb') as stderr:
        result=subprocess.run([sys.executable,'-I','-B','-c',script,str(source),str(config),phase,str(campaign),kind],stdout=subprocess.PIPE,stderr=stderr,timeout=15000)
        stderr.write(result.stdout);stderr.flush();result.check_returncode()
    return json.loads(result.stdout.decode().splitlines()[-1])


class SettledContinuation:
    def __init__(self,evaluator,plan):
        self.evaluator=evaluator;self.plan=plan;self.original=Path(plan['original']);self.cfg=read(self.original/'coordinator-config.json')

    def run(self):
        e=self.evaluator;old=self.original;receipt=read(old/'coordinator-process.json')
        while process_alive(receipt):
            if e.stop.wait(5):return
        assert (old/'completion.json').exists(),'Missing explicit old-owner completion'
        # Delegated budgets are not unresolved scientific attempts.
        root=Path(self.cfg['local_root'])/'budget'
        for f in root.glob('*.json'):
            r=read(f)
            if read(r['campaign']).get('kind')=='EXPLICIT_DELEGATED_CUMULATIVE_BUDGET':continue
            if r['status']!='COMPLETE':raise RuntimeError('Unsettled old phase blocks continuation: '+r['key'])
        p=read(self.cfg['protocol']);pending={}
        for name in self.plan['cells']:
            pubs=old/'publications'/name
            for intent in pubs.glob('*.attempt.json'):
                if not intent.with_name(intent.name.replace('.attempt.json','.json')).exists():raise RuntimeError('Unknown CPU optimizer outcome')
            receipts=sorted(pubs.glob('[0-9][0-9][0-9][0-9].json'));version=read(receipts[-1])['version'] if receipts else 0
            if receipts:
                value=read(receipts[-1]);assert value['episodes']==32*version and sha(value['policy'])==value['policy_sha256']
            if version<64:pending[name]=version
        write(e.root/'continuation-audit.json',dict(status='VERIFIED_STOPPED_SETTLED_PPO_BOUNDARIES_ONLY',pending=pending,optimizer_resets=0,replayed_complete_episodes=0,maximum_update=64,source_manifest_sha256=sha(self.cfg['source_manifest'])))
        # Activate only the old owner's disjoint cards after proving live idleness.
        for host,ids in self.plan['capacity'].items():
            fresh=[i for i in ids if i not in e.activated.get(host,set())]
            while fresh and not e.idle_cards(host,fresh):
                if e.stop.wait(5):return
            if fresh:
                proof=e.root/'resource-grants'/('settled-training-'+host+'.json');write(proof,dict(status='VERIFIED_STOPPED_OLD_OWNER_AND_IDLE',host=host,gpu_ids=fresh,audit_sha256=sha(e.root/'continuation-audit.json')))
                with e.lock:
                    fresh=[i for i in fresh if i not in e.activated.get(host,set())]
                    if fresh:e.pool.grant(host,fresh,evidence_sha256=sha(proof));e.activated.setdefault(host,set()).update(fresh)
        budget=PhaseBudget(root,old/'budget',p['authorization'],authorization_sha256=sha(p['authorization']))
        with ThreadPoolExecutor(max(1,len(pending))) as pool:
            for f in as_completed([pool.submit(self.cell,n,v,budget) for n,v in pending.items()]):f.result()

    def cell(self,name,start,budget):
        e=self.evaluator;old=self.original;p=read(self.cfg['protocol']);cell=p['cells'][name];family=cell['family'];out=e.root/'settled-continuation';out.mkdir(exist_ok=True)
        cfg=deepcopy(self.cfg);cfg.update(output=str(out),local_root=str(Path(e.config['owner']).parent/'settled-continuation'),source_manifest=self.cfg['source_manifest'],actor_publications={n:str(old/'training'/n/'publications') for n in p['cells']})
        path=out/(name+'-config.json');write(path,cfg);ctx=PrefixCampaign(path);ctx.pool=e.pool;ctx.budget=budget;ctx.proofs=read(old/'admission-proofs.json');snapshot=read(cfg['source_manifest']);ctx.manifest=snapshot
        self.development(ctx,name,start,budget,snapshot,out)
        for version in range(start,64):
            rows=read(cell['training_schedule'])[version*32:(version+1)*32];groups={digest(read(cell['training_routes'][r['suite']])) for r in rows};count=max(2,len(groups))
            hosts=['gnu','video'] if family=='groot_n15' else self.cfg['hosts'][family]
            with e.pool.lease(name+f'-continuation-{version}',hosts,count=count) as (host,ids):
                campaign,label=ctx.build(name,'training',cell['training_schedule'],host,version=version);execution=ctx.execution(name,family,host,ids)
                execution.update(source_manifest=self.cfg['source_manifest'],node_template=self.cfg['node_templates'][family],experiment_id='settled-prefix-continuation-'+name,local_root=str(Path(cfg['local_root'])/name),output=str(out/'execution'/name))
                execution['input_hashes']=self.cfg['input_hashes'];ep=out/'bindings'/name/(label+'.json');write(ep,execution)
                key=name+'-'+label;budget.reserve(key,campaign);result=frozen_stage(snapshot['release'],ep,label,campaign,out/'controller-logs'/name/(label+'.log'));budget.settle(key,result)
                records=collection_records(execution,label,campaign)
            batch=out/'batches'/(name+f'-{version:04d}.json');write(batch,dict(cell=name,version=version,policy_sha256=sha(ctx.publication(name,version)),records=records))
            args=['env','CUDA_VISIBLE_DEVICES=','WANDB_MODE=disabled','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+snapshot['release'],self.cfg['learner_python'],'-B','-m','benchmarks.LIBERO.controlled_learner','--config',self.cfg['learners'][name],'--execute-cpu','--records',str(batch)]
            log=out/'cpu-updates'/(name+f'-{version+1:04d}.log');log.parent.mkdir(exist_ok=True)
            execute_update(ssh_command(self.cfg['learner_target'])+[shlex.join(args)],log,old/'publications',run=name,version=version+1,final_version=64)
            self.development(ctx,name,version+1,budget,snapshot,out)
        with e.lock:e.completed[name+'-continuation']='COMPLETE_ORIGINAL_OPTIMIZER_TO_2048'
        e.status()

    def development(self,ctx,name,version,budget,snapshot,out):
        if version not in (16,32):return
        cell=ctx.protocol['cells'][name];family=cell['family']
        for schedule,meta in cell['evaluation_schedules'].items():
            if meta['purpose']!='development' or meta['version']!=version:continue
            label=Path(schedule).stem+f'-v{version:04d}';key=name+'-'+label
            old=budget.root/(key+'.json')
            if old.exists():
                if read(old)['status']!='COMPLETE':raise RuntimeError('Unsettled development cannot replay')
                continue
            rows=read(schedule);groups={digest(read(cell['evaluation_routes'][r['suite']])) for r in rows}
            hosts=['gnu','video'] if family=='groot_n15' else self.cfg['hosts'][family]
            with self.evaluator.pool.lease(name+'-'+label,hosts,count=max(2,len(groups))) as (host,ids):
                path,label=ctx.build(name,'evaluation',schedule,host,version=version);execution=ctx.execution(name,family,host,ids)
                execution.update(source_manifest=self.cfg['source_manifest'],node_template=self.cfg['node_templates'][family],experiment_id='settled-prefix-development-'+name,local_root=str(Path(ctx.cfg['local_root'])/name),output=str(out/'execution'/name),input_hashes=self.cfg['input_hashes'])
                ep=out/'bindings'/name/(label+'.json');write(ep,execution);budget.reserve(key,path)
                result=frozen_stage(snapshot['release'],ep,label,path,out/'controller-logs'/name/(label+'.log'),kind='target_recovery');budget.settle(key,result)
