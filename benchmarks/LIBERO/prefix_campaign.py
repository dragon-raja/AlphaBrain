"""Full-length prefix experiment composition over existing controlled phases.

No independent launcher/queue or automatic replay. Resource leases precede
immutable campaign publication; every scientific forward uses center authority.
"""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from threading import RLock
import subprocess,shlex,time
from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.execution.manifest import file_digest as sha,digest
from AlphaBrain.research.dsol.execution.queue import exclusive
from AlphaBrain.research.dsol.execution.phase_budget import PhaseBudget
from AlphaBrain.research.dsol.execution.fleet_slots import FleetSlots
from AlphaBrain.research.dsol.execution.learner_publication import execute_update
from AlphaBrain.research.dsol.execution.authority_wire import ssh_command
from .controlled_campaign import run_stage,collection_records
from .prefix_pilot import MAINLINE,limits,validate_campaign


def admission_identity(cell):
    """Separate identical layouts with different task/feature semantics."""
    return digest({key:cell.get(key) for key in ('core_spec','task_condition','feature_producer','model_hashes','score_contract','mechanism')})


class PrefixCampaign:
    def __init__(self,path):
        self.config_path=Path(path);self.cfg=read(path);self.root=Path(self.cfg['output']);self.local=Path(self.cfg['local_root'])
        for p,h in self.cfg['input_hashes'].items():
            if sha(p)!=h:raise ValueError('Changed production configuration input')
        self.protocol=read(self.cfg['protocol']);self.manifest=read(self.cfg['source_manifest']);self.lock=RLock()
        self.budget=PhaseBudget(self.local/'budget',self.root/'budget',self.protocol['authorization'],authorization_sha256=sha(self.protocol['authorization']),allocation=self.cfg.get('budget_allocation'))
        self.pool=FleetSlots(self.cfg['capacity']);self.proofs={};self.variant_proofs={};self.states={};self.failures={}

    def status(self,name,**values):
        with self.lock:
            self.states.setdefault(name,{}).update(values,updated_unix=time.time())
            write(self.root/'status.json',dict(status='RUNNING' if not self.failures else 'PARTIAL_HOLD',cells=self.states,budget=self.budget.status(),
                resources=self.pool.snapshot(),failures=dict(self.failures),
                automatic_retry=False,automatic_extension=False,updated_unix=time.time()))

    def execution(self,name,family,host,ids):
        c=self.cfg
        return dict(experiment_id=c['experiment_id']+'-'+name,source_manifest=c['source_manifest'],node_template=c['node_templates'][family],aliases=c['aliases'],
            capacity={host:dict(gpu_ids=ids,workers_per_gpu=1)},local_root=str(self.local/name),output=str(self.root/'execution'/name),excluded_hosts=[],
            dispatch_deadline_utc_seconds=c['dispatch_deadline_utc_seconds'],phase_timeout_seconds=14400,launch_timeout_seconds=600,
            persistent_inspection=True,parallel_launches=len(ids),node_health=c['node_health'],input_hashes=c['input_hashes'])

    def publication(self,name,version):
        root=Path(self.cfg.get('actor_publications',{}).get(name,self.root/'training'/name/'publications'))
        return root/f'policy-{version:04d}.pt'

    def build(self,name,kind,schedule,host,*,version=None,stay=False):
        p=self.protocol;cell=p['cells'][name];rows=read(schedule)
        if kind=='training':rows=rows[32*version:32*(version+1)]
        c=dict(schema=MAINLINE,kind=kind,cell=name,protocol=self.cfg['protocol'],native_policy=cell['native_policy'],source_identity=self.manifest['source_identity'],
            schedule=schedule,hosts=[host],execution_hosts=[host],admission_proofs=self.proofs.get(cell['family'],{}),
            score_noise=cell['score_noise'],execution_noise=cell['execution_noise'],stay=stay,blocks=[],input_hashes={})
        if cell.get('core_spec'):
            c['admission_proofs']=self.variant_proofs.get(admission_identity(cell),{})
        required={self.cfg['protocol'],p['authorization'],schedule}
        if cell.get('task_condition') is not None:required.add(cell['task_condition']['cache'])
        if kind=='evaluation':
            c['evaluation_version']=version
            required.update([cell['score_noise'],cell['execution_noise'],read(cell['execution_noise'])['noise_file']])
            if not stay:
                pub=self.publication(name,version);c['actor']=dict(path=str(pub),sha256=sha(pub));required.add(str(pub))
        if kind=='admission':
            c['execution_noise']=self.cfg['engineering_noise'];required.update([c['execution_noise'],read(c['execution_noise'])['noise_file']])
            if c['schema']==MAINLINE or cell['family']=='rldx':
                c['engineering_actor']=str(self.publication(name,0));required.add(c['engineering_actor'])
        groups=defaultdict(list)
        for r in rows:groups[r['suite']].append(r)
        stage='training' if kind=='admission' else kind
        for suite,selected in sorted(groups.items()):
            route=cell[stage+'_routes'][suite];sim=read(route);required.update([route,sim['inputs'],sim['anchors']])
            model=sim['models'][sim['operation_model']]
            if 'runtime_contract' in model:required.add(model['runtime_contract'])
            if 'pilot_execution_plan' in sim:required.add(sim['pilot_execution_plan'])
            if kind!='admission':required.add(c['admission_proofs'][host+'/'+suite])
            size=1 if kind=='admission' else 2
            for i in range(0,len(selected),size):
                pair=selected[i:i+size];b=dict(index=len(c['blocks']),host=host,rows=pair,sim_config=route,limits=limits(pair,sim,kind,cell['family'],stay=stay,collector_smoke=True,query_cap=cell.get('mechanism',{}).get('query_cap',8)))
                if kind=='training':
                    actor=self.publication(name,version);required.add(str(actor))
                    b.update(version=version,policy=str(actor),policy_sha256=sha(actor),run=name,arm='G',seed=cell['seed'])
                    b['addresses']=[dict(model=name,task=r['task'],initial=r['state_key'],view='episode-'+str(r['episode']),stream='score',repeat=0,noise='two-fixed-members') for r in pair]
                else:b['addresses']=[dict(model=name,task=r['task'],initial=r['state_key'],view=str(r.get('start','admission'))+':'+r.get('method',kind),stream='score' if kind=='admission' else 'execution',repeat=n,noise='admission' if kind=='admission' else f'paired-{n}') for r in pair for n in ([0] if kind=='admission' else [0,1])]
                c['blocks'].append(b)
        c['limits']={k:sum(b['limits'][k] for b in c['blocks']) for k in c['blocks'][0]['limits']};c['input_hashes']={x:sha(x) for x in sorted(required)}
        label=f'train-{version:04d}' if kind=='training' else 'admission-'+host+('-'+Path(schedule).stem if self.cfg.get('separate_admission_suites') else '') if kind=='admission' else Path(schedule).stem+f'-v{version:04d}'
        path=self.root/'campaigns'/name/(label+'.json')
        if path.exists():raise ValueError('Existing phase requires audit, never replay')
        write(path,c);validate_campaign(path);return path,label

    def execute(self,name,kind,schedule,*,version=None,stay=False,host_only=None):
        family=self.protocol['cells'][name]['family'];hosts=[host_only] if host_only else self.cfg['hosts'][family]
        count=self.cfg.get('training_slots',4) if kind=='training' else self.cfg.get('admission_slots',4) if kind=='admission' else 2
        if type(count) is not int or not 1<=count<=8:raise ValueError('Invalid phase slot count')
        if kind=='evaluation':
            suites={r['suite'] for r in read(schedule)}
            spec=self.protocol['cells'][name].get('core_spec')
            proofs=self.variant_proofs.get(admission_identity(self.protocol['cells'][name]),{}) if spec else self.proofs.get(family,{})
            hosts=[h for h in hosts if all(h+'/'+s in proofs for s in suites)]
            if not hosts:raise ValueError('No host admitted for this exact evaluation suite')
        maximum=self.cfg.get('evaluation_slots_max',8) if kind=='evaluation' else count
        if type(maximum) is not int or not count<=maximum<=8:raise ValueError('Invalid maximum phase slots')
        with self.pool.lease(name+'-'+Path(schedule).stem,hosts,count=count,max_count=maximum) as (host,ids):
            path,label=self.build(name,kind,schedule,host,version=version,stay=stay);config=self.execution(name,family,host,ids)
            config_path=self.root/'bindings'/name/(label+'.json');write(config_path,config)
            key=name+'-'+label;self.budget.reserve(key,path);self.status(name if kind=='training' else name+'-'+kind,state='RUNNING',phase=label,host=host,gpu_ids=ids)
            result=run_stage(config,label,path,kind='training' if kind in ('training','admission') else 'target_recovery',cpu=False)
            self.budget.settle(key,result);write(self.root/'settlements'/name/(label+'.json'),result)
            return config,path,label

    def cpu_update(self,name,version,batch):
        c=self.cfg;env=['env','CUDA_VISIBLE_DEVICES=','WANDB_MODE=disabled','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+self.manifest['release']]
        args=env+[c['learner_python'],'-B','-m','benchmarks.LIBERO.controlled_learner','--config',c['learners'][name],'--execute-cpu','--records',str(batch)]
        command=ssh_command(c['learner_target'])+[shlex.join(args)];log=self.root/'cpu-updates'/(name+f'-{version:04d}.log');log.parent.mkdir(exist_ok=True)
        return execute_update(command,log,self.root/'publications',run=name,version=version,final_version=self.protocol['updates'])

    def admission(self,family,host):
        names=[n for n,c in self.protocol['cells'].items() if c['family']==family and c['seed']==min(self.protocol['seeds'])
            and n in self.cfg.get('active_cells',self.protocol['cells'])]
        if self.protocol['schema']!='UNIFIED_CORE_PILOT_V1':names=names[:1]
        for name in names:self.admit_cell(name,family,host)

    def admit_cell(self,name,family,host):
        cfg,path,label=self.execute(name,'admission',self.cfg['engineering_schedules'][family],host_only=host)
        from .controlled_campaign import phase_evidence
        campaign=read(path);root=phase_evidence(cfg,label)
        with self.lock:
            cell=self.protocol['cells'][name]
            proof=(self.variant_proofs.setdefault(admission_identity(cell),{})
                   if cell.get('core_spec') else self.proofs.setdefault(family,{}))
            for b in campaign['blocks']:proof[host+'/'+b['rows'][0]['suite']]=str(root/'results'/f"block-{b['index']:04d}"/'admission.json')
            # Model numerical references are shared; execution still requires the
            # exact variant proof above and in validate_admission. No U0 dependency.
            if cell.get('core_spec'):
                self.proofs.setdefault(family,{})
                for b in campaign['blocks']:
                    key=host+'/'+b['rows'][0]['suite']
                    self.proofs[family].setdefault(key,proof[key])
            write(self.root/'admission-proofs.json',self.proofs)
            if cell.get('core_spec'):write(self.root/'variant-admission-proofs.json',self.variant_proofs)

    def cell(self,name):
        evaluator=None
        try:
            cell=self.protocol['cells'][name]
            if cell.get('evaluation_only'):
                if self.protocol.get('design_stage') not in ('mechanism_screen','core_confirmation'):raise ValueError('Explicit deployment-only protocol required')
                for schedule,meta in cell['evaluation_schedules'].items():self.execute(name,'evaluation',schedule,version=meta['version'])
                self.status(name,state='COMPLETE',optimizer_updates=0,new_training_episodes=0)
                return
            start=self.protocol.get('start_update',0)
            if start and self.protocol.get('design_stage') not in ('long_continuation','core_confirmation'):
                raise ValueError('Starting later requires an explicit continuation protocol')
            pending=[]
            evaluator=ThreadPoolExecutor(1) if self.cfg.get('async_development',False) else None
            if self.protocol.get('design_stage')=='core_confirmation':
                for schedule,meta in cell['evaluation_schedules'].items():
                    if meta['version']<start:
                        if evaluator is None:self.execute(name,'evaluation',schedule,version=meta['version'])
                        else:pending.append(evaluator.submit(self.execute,name,'evaluation',schedule,version=meta['version']))
            for version in range(start,self.protocol['updates']):
                for future in pending:
                    if future.done():future.result()
                cfg,path,label=self.execute(name,'training',cell['training_schedule'],version=version)
                actor=self.publication(name,version);batch=self.root/'batches'/(name+f'-{version:04d}.json')
                write(batch,dict(cell=name,version=version,policy_sha256=sha(actor),records=collection_records(cfg,label,path)))
                self.status(name,state='CPU_OPTIMIZER',optimizer_updates=version,episodes=32*(version+1));self.cpu_update(name,version+1,batch)
                self.status(name,state='UPDATE_COMPLETE',optimizer_updates=version+1,episodes=32*(version+1))
                for schedule,meta in cell['evaluation_schedules'].items():
                    if meta['purpose']=='development' and meta['version']==version+1:
                        if evaluator is None:self.execute(name,'evaluation',schedule,version=version+1)
                        else:pending.append(evaluator.submit(self.execute,name,'evaluation',schedule,version=version+1))
            updates=self.protocol['updates']
            write(self.root/'training-completions'/(name+'.json'),dict(status='COMPLETE',episodes=32*updates,updates=updates,operation_model_updates=0))
            for future in pending:future.result()
            for schedule,meta in cell['evaluation_schedules'].items():
                if meta['purpose']=='final':self.execute(name,'evaluation',schedule,version=updates)
            self.status(name,state='COMPLETE',optimizer_updates=updates,episodes=32*updates)
        except BaseException as error:
            with self.lock:self.failures[name]=repr(error)
            self.status(name,state='HOLD_NO_AUTOMATIC_RETRY',error=repr(error))
            raise
        finally:
            if 'evaluator' in locals() and evaluator is not None:evaluator.shutdown(wait=True,cancel_futures=True)

    def baseline(self):
        name='rldx-p4-seed-0'
        for schedule,meta in self.protocol['cells'][name]['evaluation_schedules'].items():
            if meta['purpose']=='baseline':self.execute(name,'evaluation',schedule,version=64,stay=True)

    def run(self):
        owner=self.local/'coordinator-owner';owner.mkdir(parents=True,exist_ok=False)
        with exclusive(owner):
            active=self.cfg.get('active_cells',list(self.protocol['cells']))
            if not active or len(set(active))!=len(active) or not set(active).issubset(self.protocol['cells']):raise ValueError('Explicit approved active cells required')
            families={self.protocol['cells'][n]['family'] for n in active}
            pairs=[(f,h) for f,hs in self.cfg['hosts'].items() if f in families for h in hs]
            with ThreadPoolExecutor(len(pairs)) as pool:
                for future in as_completed([pool.submit(self.admission,f,h) for f,h in pairs]):future.result()
            from .prefix_admission import peer_numerical_proof
            write(self.root/'peer-numerical-proof.json',peer_numerical_proof(self.proofs,self.cfg.get('reference_proofs')))
            write(self.root/'admission-completion.json',dict(status='PASS_ALL_MODEL_HOST_SUITE_ENTRIES',proofs=self.proofs,budget=self.budget.status()))
            with ThreadPoolExecutor(max(1,len(active)+int(self.cfg.get('run_baseline',True)))) as pool:
                futures={pool.submit(self.cell,name):name for name in active}
                if self.cfg.get('run_baseline',True):futures[pool.submit(self.baseline)]='rldx-baseline'
                for future in as_completed(futures):
                    try:future.result()
                    except BaseException as error:
                        with self.lock:self.failures[futures[future]]=repr(error)
            write(self.root/'completion.json',dict(status='COMPLETE' if not self.failures else 'PARTIAL_HOLD',failures=self.failures,budget=self.budget.status(),dispatch_enabled=False,automatic_retry=False))
