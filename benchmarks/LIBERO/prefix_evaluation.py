"""Evaluation handoff from immutable full-prefix training publications.

Uses the existing controlled campaign adapter, phase budgets and shared resource
pool. A final marker only fences old ownership; fresh model grants still require
node/suite admission, a settled actor publication and a newly reserved phase.
"""
from pathlib import Path
from threading import Thread,Event,RLock
from concurrent.futures import ThreadPoolExecutor,as_completed
import time,subprocess,shlex
from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.execution.manifest import file_digest as sha,digest
from AlphaBrain.research.dsol.execution.fleet_slots import FleetSlots
from AlphaBrain.research.dsol.execution.authority_wire import ssh_command
from .prefix_campaign import PrefixCampaign
from .controlled_campaign import phase_evidence
from .prefix_admission import peer_numerical_proof


class PrefixEvaluation:
    def __init__(self,path):
        self.config=read(path);self.deadline=self.config['dispatch_deadline_utc_seconds'];self.root=Path(self.config['output']);self.lock=RLock();self.stop=Event();self.errors={};self.admitted={};self.completed={}
        self.contexts={key:PrefixCampaign(p) for key,p in self.config['contexts'].items()}
        self.pool=FleetSlots(self.config['capacity'],initial_capacity=self.config['initial_capacity'])
        self.proofs={};self.activated={h:set(ids) for h,ids in self.config['initial_capacity'].items()}
        for ctx in self.contexts.values():ctx.pool=self.pool;ctx.proofs=self.proofs

    def status(self):
        with self.lock:
            write(self.root/'status.json',dict(status='RUNNING' if not self.errors else 'PARTIAL_HOLD',completed=self.completed,errors=self.errors,
                resources=self.pool.snapshot(),admitted=self.admitted,budgets={k:c.budget.status() for k,c in self.contexts.items()},updated_unix=time.time(),automatic_retry=False))

    def idle_cards(self,host,ids):
        node=read(self.config['nodes'])['nodes'][host]
        code='''import json,sys
sys.path.insert(0,__SOURCE__)
from AlphaBrain.research.dsol.execution import gpu_guard
n=__NODE__;inventory={g['index']:g['uuid'] for g in n['gpu_guard']['devices']};idle=True
for index in __IDS__:
 for attempt in range(3):
  try:s=gpu_guard.collect(n['gpu_guard'],target_gpu_uuid=inventory[index]);break
  except (gpu_guard.TransientMonitorError,FileNotFoundError,ProcessLookupError):
   if attempt==2:raise
   import time;time.sleep(1)
 k=gpu_guard.verify_status(n['gpu_guard'],s)
 idle=idle and not any(a['gpu_uuid']==inventory[index] and a['pid']!=k['pid'] for a in s['apps'])
print(json.dumps(dict(idle=idle,keeper_supervisor=k['supervisor_pid'])))
'''.replace('__SOURCE__',repr(self.config['source'])).replace('__NODE__',repr(node)).replace('__IDS__',repr(ids))
        target=dict(alias=self.config['aliases'][host],**({'public_endpoint':node['public_endpoint']} if 'public_endpoint' in node else {}))
        result=subprocess.run(ssh_command(target)+['python3','-B','-'],input=code,text=True,capture_output=True,timeout=30)
        if result.returncode:raise ConnectionError('Resource handoff live probe failed: '+host)
        import json
        return json.loads(result.stdout)['idle']

    def grant_loop(self):
        plan=read(self.config['handoff_plan'])
        while not self.stop.wait(10):
            try:
                if time.time()>=self.deadline:raise TimeoutError('Evaluation handoff window expired')
                for name,spec in plan['cells'].items():
                    if name in self.completed.get('resource_handoffs',{}):continue
                    receipt=Path(spec['publication'])
                    if not receipt.exists():continue
                    p=read(receipt)
                    if p['version']!=64 or p['episodes']!=2048 or sha(p['policy'])!=p['policy_sha256']:raise ValueError('Invalid final publication in handoff')
                    state=read(Path(spec['root'])/'status.json');held=state.get('resources',{}).get('held',{}).get(spec['resource_quarantine_key'])
                    if held is None:continue
                    if sha(spec['sentinel'])!=spec['sentinel_sha256']:raise ValueError('Evaluation handoff marker changed')
                    host=held['host'];ids=[i for i in held['gpu_ids'] if i not in self.activated.get(host,set())]
                    if ids and not self.idle_cards(host,ids):continue
                    proof=dict(status='VERIFIED_V64_PUBLICATION_OLD_OWNER_QUARANTINE_AND_LIVE_IDLE',run=name,resource=held,publication_sha256=sha(receipt),marker_sha256=spec['sentinel_sha256'],checked_unix=time.time())
                    path=self.root/'resource-grants'/(name+'.json');write(path,proof)
                    with self.lock:
                        ids=[i for i in ids if i not in self.activated.get(host,set())]
                        if ids:self.pool.grant(host,ids,evidence_sha256=sha(path));self.activated.setdefault(host,set()).update(ids)
                        self.completed.setdefault('resource_handoffs',{})[name]=proof
                    self.status()
            except BaseException as error:
                with self.lock:self.errors['resource-handoff']=repr(error)
                self.status();self.stop.set();return

    def admit(self,family,host,ctx):
        name=next(n for n,c in ctx.protocol['cells'].items() if c['family']==family and c['seed']==0)
        for row in read(ctx.cfg['engineering_schedules'][family]):
            suite=row['suite'];schedule=self.root/'admission-schedules'/(family+'-'+suite+'.json')
            with self.lock:
                if not schedule.exists():write(schedule,[row])
            cfg,path,label=ctx.execute(name,'admission',str(schedule),host_only=host)
            b=read(path)['blocks'][0];proof=phase_evidence(cfg,label)/'results'/f"block-{b['index']:04d}"/'admission.json'
            # Compare to an already reviewed fixed native input, not its version string.
            reference=self.config['reference_proofs'][family+'/'+suite]
            checked=peer_numerical_proof({family:{host+'/'+suite:str(proof)}},{family+'/'+suite:reference})
            write(self.root/'admission-peer-proofs'/(family+'-'+host+'-'+suite+'.json'),checked)
            with self.lock:self.proofs.setdefault(family,{})[host+'/'+suite]=str(proof)
        with self.lock:self.admitted[family+'/'+host]='PASS_NATIVE_COLLECTOR_TRAJECTORY_AND_PEER_EQUALITY'
        self.status()

    def admission_loop(self):
        while not self.stop.is_set():
            if time.time()>=self.deadline:self.stop.set();return
            progressed=False
            for family,context in self.config['family_contexts'].items():
                ctx=self.contexts[context]
                for host in ctx.cfg['hosts'][family]:
                    if family+'/'+host in self.admitted:continue
                    if not self.pool.snapshot()['available'].get(host):continue
                    try:self.admit(family,host,ctx);progressed=True
                    except BaseException as error:
                        with self.lock:self.errors['admission-'+family+'-'+host]=repr(error);self.admitted[family+'/'+host]='HOLD'
                        self.status()
            if not progressed:self.stop.wait(10)

    def evaluate(self,name):
        context=self.config['cell_contexts'][name];ctx=self.contexts[context];cell=ctx.protocol['cells'][name];family=cell['family']
        final_receipt=Path(read(self.config['handoff_plan'])['cells'][name]['publication'])
        while not final_receipt.exists():
            if 'settled-continuation' in self.errors and self.config.get('settled_continuation',{}).get('original')==read(self.config['handoff_plan'])['cells'][name]['root']:raise RuntimeError('Planned continuation requires explicit audit')
            if self.stop.wait(5) or time.time()>=self.deadline:raise RuntimeError('Evaluation owner stopped before final actor')
        p=read(final_receipt)
        if p['run']!=name or p['version']!=64 or p['episodes']!=2048 or sha(p['policy'])!=p['policy_sha256']:raise ValueError('Unsettled final actor')
        for schedule,meta in cell['evaluation_schedules'].items():
            if meta['purpose']!='final':continue
            self.wait_ready(ctx,family,schedule);ctx.execute(name,'evaluation',schedule,version=64)
        with self.lock:self.completed[name]='COMPLETE_FINAL_EVALUATION'
        self.status()

    def wait_ready(self,ctx,family,schedule):
        if self.stop.is_set() or time.time()>=self.deadline:raise RuntimeError('Evaluation owner halted')
        suites={r['suite'] for r in read(schedule)}
        while not any(all(h+'/'+s in self.proofs.get(family,{}) for s in suites) for h in ctx.cfg['hosts'][family]):
            if self.stop.wait(5) or time.time()>=self.deadline:raise RuntimeError('No admitted evaluation host')

    def baseline(self):
        ctx=self.contexts[self.config['family_contexts']['rldx']];name='rldx-p4-seed-0';cell=ctx.protocol['cells'][name]
        for schedule,meta in cell['evaluation_schedules'].items():
            if meta['purpose']=='baseline':
                self.wait_ready(ctx,'rldx',schedule);ctx.execute(name,'evaluation',schedule,version=64,stay=True)
        with self.lock:self.completed['baseline']='COMPLETE_STAY_REFERENCE'
        self.status()

    def run(self):
        from AlphaBrain.research.dsol.execution.queue import exclusive
        owner=Path(self.config['owner']);owner.mkdir(parents=True,exist_ok=False)
        with exclusive(owner):
            threads=[Thread(target=self.grant_loop,daemon=True),Thread(target=self.admission_loop,daemon=True)]
            if self.config.get('settled_continuation'):
                from .prefix_continuation import SettledContinuation
                def continue_settled():
                    try:SettledContinuation(self,self.config['settled_continuation']).run()
                    except BaseException as error:
                        with self.lock:self.errors['settled-continuation']=repr(error)
                        self.status()
                threads.append(Thread(target=continue_settled,daemon=True))
            for t in threads:t.start()
            with ThreadPoolExecutor(10) as pool:
                fs={pool.submit(self.evaluate,n):n for n in self.config['cell_contexts']};fs[pool.submit(self.baseline)]='baseline'
                for future in as_completed(fs):
                    try:future.result()
                    except BaseException as error:
                        with self.lock:self.errors[fs[future]]=repr(error)
                        self.status()
            self.stop.set()
            for t in threads:t.join(timeout=40)
            write(self.root/'completion.json',dict(status='COMPLETE' if not self.errors else 'PARTIAL_HOLD',errors=self.errors,completed=self.completed,budgets={k:c.budget.status() for k,c in self.contexts.items()},dispatch_enabled=False))
