"""One-shot approved reward adaptation using the standard distributed executor.

Nine CPU learners publish immutable actors; collectors synchronize at complete
32-episode barriers. Every new GPU phase has a durable reservation and no replay.
Configuration supplies all private paths. No execution happens on import.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import argparse
import datetime
import json
import shlex
import subprocess
import time
from AlphaBrain.research.dsol.data.artifacts import read
from AlphaBrain.research.dsol.data import reward_pilot,native_recovery
from AlphaBrain.research.dsol.execution.manifest import file_digest
from AlphaBrain.research.dsol.execution.queue import exclusive,_durable_write
from AlphaBrain.research.dsol.execution.learner_publication import accept,published,execute_update
from .controlled_campaign import run_stage


def run_names(protocol=None):
    arms = reward_pilot.ARMS if protocol is None else protocol['arms']
    return [f'{arm}-seed-{seed}' for arm in arms for seed in (0,1,2)]


class Adaptation:
    def __init__(self,config):
        self.config=config;self.root=Path(config['output'])
        self.protocol=reward_pilot.load_protocol(config['protocol'],config['input_hashes'][config['protocol']])
        self.publications=Path(config.get('publication_root',str(Path(config['local_root'])/'learner-publications')))
        self.names=run_names(self.protocol)
        self.updates=self.protocol['updates']
        self.calls=dict(training=0,evaluation=0,engineering=0)
        self.committed=dict(training=0,evaluation=0,engineering=0)

    def status(self,status,**extra):
        value=dict(status=status,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            actual_model_calls=dict(self.calls),committed_call_caps=dict(self.committed),
            automatic_retry=False,workers=48,**extra)
        _durable_write(self.root/'current.json',value)
        print(json.dumps(value),flush=True)

    def command(self,module,args,*,host='fresh'):
        node=read(self.config['node_template'])['nodes'][host]
        release=read(self.config['source_manifest'])['release']
        env={k:v for k,v in node['environment'].items() if k in ('PYTHONPATH','LD_LIBRARY_PATH')}
        env['PYTHONPATH']=release+':'+':'.join(env['PYTHONPATH'].split(':')[1:])
        env.update(CUDA_VISIBLE_DEVICES='',WANDB_MODE='disabled',PYTHONDONTWRITEBYTECODE='1')
        cmd=['env',*[k+'='+v for k,v in env.items()],node['python'],'-m',module,*map(str,args)]
        return ['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8',
            self.config['aliases'][host],shlex.join(cmd)]

    def initialize(self):
        folder=self.root/'training/configs';folder.mkdir(parents=True,exist_ok=False)
        for name in self.names:
            arm,_,seed=name.partition('-seed-');seed=int(seed)
            schedule=self.protocol['schedules'][str(seed)];initial=self.protocol['initial_checkpoints'][seed]['path']
            protocol=self.config['protocol']
            cfg=dict(schema=reward_pilot.LEARNER_SCHEMA,arm=arm,seed=seed,run=name,updates=self.updates,episodes_per_update=32,
                schedule=schedule,initial_checkpoint=initial,protocol=protocol,adaptation_protocol=protocol,
                output=str(self.root/'training/runs'/name),
                input_hashes={p:file_digest(p) for p in (schedule,initial,protocol)})
            _durable_write(folder/(name+'.json'),cfg)
        def one(name):
            # Exclusive init intent remains even when SSH succeeds but receipt is lost.
            with (folder/(name+'-initialize.attempt')).open('x') as f:f.write('ONE_ATTEMPT_NO_REPLAY\n')
            command=self.command('benchmarks.LIBERO.controlled_learner',
                ['--config',folder/(name+'.json'),'--execute-cpu'])
            with (folder/(name+'-initialize.log')).open('xb') as log:
                process=subprocess.run(command,stdout=subprocess.PIPE,stderr=log)
                log.write(process.stdout);log.flush();process.check_returncode()
            return accept(process.stdout,self.publications,run=name,version=0,final_version=self.updates)
        with ThreadPoolExecutor(3) as pool:results=list(pool.map(one,self.names))
        _durable_write(self.root/'training/initialization.json',dict(status='COMPLETE_CPU_INITIALIZATION',runs=results))

    def batch(self,version):
        from .controlled_training import validate_campaign
        common=deepcopy(read(self.config['training_common']))
        # Retain model/simulator/input bytes; replace only the approved protocol.
        common.update(schema=reward_pilot.COLLECTION_SCHEMA,kind='rollout',protocol=self.config['protocol'],
            adaptation_protocol=self.config['protocol'],schedules=self.protocol['schedules'],settle_reward_at_learner=True)
        common.pop('calibration_requests',None);common.pop('calibration_receipt',None)
        hashes=common['input_hashes'];hashes[self.config['protocol']]=file_digest(self.config['protocol'])
        for p in self.protocol['schedules'].values():hashes[p]=file_digest(p)
        blocks=[]
        for name in self.names:
            arm,_,seed=name.partition('-seed-');seed=int(seed)
            publication=published(self.publications,name,version)
            hashes[publication['policy']]=publication['policy_sha256']
            rows=read(self.protocol['schedules'][str(seed)])[version*32:(version+1)*32]
            size=self.protocol.get('episodes_per_block',4)
            for offset in range(0,32,size):
                selected=rows[offset:offset+size];index=len(blocks)
                addresses=[dict(model='native_reference',task=x['task'],initial=x['state_key'],
                    view=f"{name}:episode-{x['episode']}",stream='score',repeat=m,
                    noise=f"training-initial-{x['initial']}-member-{m}") for x in selected for m in (0,1)]
                blocks.append(dict(index=index,host=['fresh','gnu','video'][index%3],rows=selected,
                    run=name,arm=arm,seed=seed,version=version,policy=publication['policy'],
                    policy_sha256=publication['policy_sha256'],reward_scale=reward_pilot.FIXED_SCALE,
                    addresses=addresses,limits=dict(episodes=len(selected),score=16*len(selected),paired_execution=0,nominal_execution=0)))
        common.update(blocks=blocks,limits=dict(episodes=32*len(self.names),score=512*len(self.names),paired_execution=0,nominal_execution=0))
        folder=self.root/'training/batches'/f'{version:04d}';folder.mkdir(parents=True,exist_ok=False)
        path=folder/'campaign.json';_durable_write(path,common);validate_campaign(path)
        return path

    def stage(self,phase,campaign,*,kind,category,cpu=False):
        limits=read(campaign)['limits'];cap=sum(limits[k] for k in ('score','paired_execution','nominal_execution'))
        if not cpu:
            self.committed[category]+=cap
            if self.committed[category]>self.protocol['call_caps'][category]:raise ValueError('Category reservation exceeds approval')
            if sum(self.committed.values())>self.protocol['call_caps']['total']:raise ValueError('Global reservation exceeds approval')
            _durable_write(self.root/'reservations'/(phase+'.json'),dict(status='RESERVED_BEFORE_DISPATCH',
                campaign=str(campaign),sha256=file_digest(campaign),category=category,cap=cap,
                cumulative_committed=dict(self.committed),unknown_consumption_refundable=False))
        result=run_stage(self.config,phase,campaign,kind=kind,cpu=cpu)
        if not cpu:
            self.calls[category]+=result['actual_model_calls']
            _durable_write(self.root/'settlements'/(phase+'.json'),result)
        return result

    def update(self,version):
        phase=self.config.get('phase_overrides',{}).get(f'train-{version:04d}',f'train-{version:04d}')
        folder=self.root/'execution'/(phase+'-real')
        binding=read(folder/'binding.json');by_run={name:{} for name in self.names}
        for spec in binding['tasks'].values():
            block=spec['request']['block'];result=read(Path(spec['output'])/'collection.json')
            target=by_run[block['run']]
            if block['version']!=version or set(target).intersection(result['records']):raise ValueError('Mixed rollout barrier')
            target.update(result['records'])
        paths=[]
        for name,records in by_run.items():
            if set(map(int,records))!=set(range(version*32,(version+1)*32)):raise ValueError('Incomplete on-policy batch')
            pub=published(self.publications,name,version)
            path=self.root/'training/batches'/f'{version:04d}'/(name+'-records.json')
            _durable_write(path,dict(run=name,version=version,policy_sha256=pub['policy_sha256'],records=records));paths.append((name,path))
        if version==0:
            # Actual same-initialization, same-noise rollout equivalence is checked
            # before any arm is allowed to update or another GPU phase starts.
            command=self.command('benchmarks.LIBERO.reward_adaptation',
                ['--config',self.config['config_path'],'--check-first-batch'])
            with (self.root/'training/first-batch-gate.log').open('xb') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        def one(item):
            name,path=item
            command=self.command('benchmarks.LIBERO.controlled_learner',
                ['--config',self.root/'training/configs'/(name+'.json'),'--records',path,'--execute-cpu'])
            return execute_update(command,path.with_suffix('.log'),self.publications,
                run=name,version=version+1,final_version=self.updates)
        with ThreadPoolExecutor(3) as pool:results=list(pool.map(one,paths))
        _durable_write(self.root/'training/batches'/f'{version:04d}'/'optimizer-complete.json',dict(status='COMPLETE',runs=results))

    def prepare_evaluation(self):
        from .controlled_recovery import validate_campaign,common_asset_hashes
        command=self.command('benchmarks.LIBERO.reward_adaptation',
            ['--config',self.config['config_path'],'--export-cpu'])
        with Path(self.config.get('export_log',str(self.root/'training/export.log'))).open('xb') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        actor_root=Path(self.config.get('actor_root',str(self.root/'actors')))
        actors={name:dict(path=str(actor_root/name/'actor.pt'),controls=str(actor_root/name/'controls.pt')) for name in self.names}
        baseline=read(self.config['baseline_campaign'])
        rows=reward_pilot.evaluation_rows(self.protocol,read(self.config['original_recovery_schedule']))
        folder=Path(self.config.get('evaluation_root',str(self.root/'evaluation')));folder.mkdir(exist_ok=False)
        schedule=folder/'schedule.json';_durable_write(schedule,rows)
        common=dict(schema=reward_pilot.RECOVERY_SCHEMA,model='native_reference',
            baseline_campaign=self.config['baseline_campaign'],schedule=str(schedule),paired_config=baseline['paired_config'],
            score_noise=baseline['score_noise'],baseline_references=self.config['baseline_references'],
            score_scale=.16535300100378683,query_cap=8,selection_members=[0,1],validation_members=[2,3],
            adaptation_protocol=self.config['protocol'])
        hashes=dict(baseline['input_hashes'])
        hashes.update(common_asset_hashes(self.config['baseline_campaign']))
        paths=[str(schedule),common['baseline_campaign'],common['baseline_references'],self.config['protocol']]
        for a in actors.values():paths.extend(a.values())
        hashes.update({p:file_digest(p) for p in paths});common['input_hashes']=hashes
        # Case order is frozen; first block is a real gate included in the panel.
        blocks=native_recovery.blocks(rows,size=3)
        for b in blocks:b['limits']=native_recovery.limits(b['rows'],paired_diagnostic=True)
        phases=[]
        for i,chunk in enumerate(native_recovery.waves(blocks,size=192)):
            local=[dict(b,index=j,host=['fresh','gnu','video'][j%3]) for j,b in enumerate(chunk)]
            selected=[r for b in local for r in b['rows']];methods={r['method'] for r in selected}
            c=dict(common,actors={k:v for k,v in actors.items() if k in methods},blocks=local,
                limits=native_recovery.limits(selected,paired_diagnostic=True))
            path=folder/f'campaign-{i:03d}.json';_durable_write(path,c);validate_campaign(path)
            phases.append(dict(phase=f'eval-{i:03d}',path=str(path),sha256=file_digest(path),searches=len(selected)))
        _durable_write(folder/'allocation.json',dict(phases=phases,searches=len(rows),episodes=2*len(rows)))
        return phases

    def run(self):
        resume=self.config.get('initialization_recovery')
        continuation=self.config.get('continuation_recovery')
        evaluation_resume=self.config.get('evaluation_recovery')
        start_version=0
        if self.protocol['initial_update']==0 and (resume or continuation or evaluation_resume):
            raise ValueError('Pilot recovery proofs cannot authorize cold-start replay')
        if evaluation_resume:
            proof=read(evaluation_resume)
            if (resume or self.config['input_hashes'].get(evaluation_resume)!=file_digest(evaluation_resume)
                    or proof['status']!='VERIFIED_FINAL_TRAINING_ZERO_EVALUATION'
                    or proof['evaluation_calls']!=0 or proof['automatic_retry']):
                raise ValueError('Explicit final-training zero-evaluation proof required')
            for p,h in proof['input_hashes'].items():
                if file_digest(p)!=h:raise ValueError('Evaluation recovery evidence changed')
            for name in self.names:
                folder=self.root/'training/runs'/name
                if (read(folder/'latest.json')['update']!=8 or len(list((folder/'episodes').glob('*.json')))!=256
                        or (folder/'failure.json').exists() or (folder/'inflight.json').exists()):
                    raise ValueError('Final training boundary is not clean')
            if list((self.root/'reservations').glob('eval-*.json')):
                raise ValueError('Evaluation-only recovery cannot replay reserved evaluation')
            self.calls.update(training=proof['training_calls'])
            self.committed.update(training=proof['training_reservation_bound'])
            start_version=8
        if continuation and not evaluation_resume:
            if resume:raise ValueError('Recovery modes are mutually exclusive')
            proof=read(continuation)
            if (self.config['input_hashes'].get(continuation)!=file_digest(continuation)
                    or proof['status']!='VERIFIED_STOPPED_PARTIAL_COLLECTION'
                    or not proof['all_workers_stopped'] or proof['resume_update']!=6
                    or proof['automatic_retry'] or proof['held_failed_phase_cap']!=4608):
                raise ValueError('Explicit stopped partial-collection proof required')
            for p,h in proof['input_hashes'].items():
                if file_digest(p)!=h:raise ValueError('Recovery evidence changed')
            for name in self.names:
                if published(self.publications,name,6)!=proof['publications'][name]:raise ValueError('Recovery actor changed')
                folder=self.root/'training/runs'/name
                if (read(folder/'latest.json')['update']!=6 or len(list((folder/'episodes').glob('*.json')))!=192
                        or (folder/'failure.json').exists() or (folder/'inflight.json').exists()):
                    raise ValueError('Recovery learner boundary is not clean')
            self.calls.update(training=proof['completed_training_calls']+proof['interrupted_attempts'])
            self.committed.update(training=proof['completed_training_calls']+proof['held_failed_phase_cap'])
            start_version=6
        if resume:
            proof=read(resume)
            if (self.config['input_hashes'].get(resume)!=file_digest(resume)
                    or proof['status']!='VERIFIED_INITIALIZED_ZERO_GPU_DISPATCH'
                    or proof['resume_update']!=0 or proof['model_calls']!=0):
                raise ValueError('Explicit initialization-only recovery proof required')
            for p,h in proof['input_hashes'].items():
                if file_digest(p)!=h:raise ValueError('Initialization recovery evidence changed')
            for name in self.names:
                pub=published(self.publications,name,0)
                if pub!=proof['publications'][name]:raise ValueError('Initialization publication differs')
                root=self.root/'training/runs'/name
                if (list((root/'episodes').iterdir()) or list((root/'updates').iterdir())
                        or (root/'inflight.json').exists() or (root/'failure.json').exists()):
                    raise ValueError('Initialization recovery cannot replay training work')
        marker=self.root/('execution-start-'+self.config.get('attempt','initial'));marker.mkdir(exist_ok=False)
        (self.root/'reservations').mkdir(exist_ok=bool(resume or continuation or evaluation_resume));(self.root/'settlements').mkdir(exist_ok=bool(resume or continuation or evaluation_resume))
        with exclusive(marker):
            try:
                self.status('INITIALIZING_CPU_LEARNERS',dispatch_enabled=False)
                if not resume and not continuation and not evaluation_resume:self.initialize()
                for version in range(start_version,self.updates):
                    campaign=Path(proof['campaign']) if (resume and version==0 or continuation and version==start_version) else self.batch(version)
                    if continuation and version==start_version:
                        self.stage('recovery-contract-'+self.config['attempt'],campaign,kind='training',category='training',cpu=True)
                    if version==0:
                        self.stage('training-contract-'+self.config.get('attempt','initial'),campaign,
                            kind='training',category='training',cpu=True)
                    self.status('TRAINING_COLLECTION',update=version,completed_episodes=version*32*len(self.names))
                    phase=self.config.get('phase_overrides',{}).get(f'train-{version:04d}',f'train-{version:04d}')
                    self.stage(phase,campaign,kind='training',category='training')
                    self.status('CPU_PPO_UPDATE',update=version+1)
                    self.update(version)
                self.status('TRAINING_COMPLETE_PREPARING_EVALUATION',completed_episodes=self.updates*32*len(self.names),dispatch_enabled=False)
                prepared=self.config.get('prepared_evaluation')
                if prepared:
                    if self.config['input_hashes'].get(prepared)!=file_digest(prepared):raise ValueError('Unbound prepared evaluation')
                    phases=read(prepared)['phases']
                    for phase in phases:
                        if file_digest(phase['path'])!=phase['sha256']:raise ValueError('Prepared evaluation changed')
                else:phases=self.prepare_evaluation()
                self.stage('evaluation-contract',phases[0]['path'],kind='recovery',category='evaluation',cpu=True)
                for index,p in enumerate(phases):
                    self.status('RECOVERY_EVALUATION',phase=p['phase'])
                    self.stage(p['phase'],p['path'],kind='recovery',category='evaluation')
                    if index==0:recovery_gate(self.root/'execution'/(p['phase']+'-real'),self.config['baseline_references'],Path(self.config.get('evaluation_root',str(self.root/'evaluation')))/'first-block-gate.json')
                searches=sum(p['searches'] for p in phases)
                self.status('ALL_SCIENTIFIC_WORK_COMPLETE',completed_searches=searches,completed_operation_episodes=2*searches,dispatch_enabled=False)
            except BaseException as error:
                self.status('HALTED_NO_RETRY',error=repr(error),dispatch_enabled=False)
                raise


def first_batch_gate(config):
    """CPU comparison of actual GPU rollouts, before reward adaptation starts."""
    import numpy as np
    import torch
    from AlphaBrain.research.dsol.data.continuous_training_store import numpy_tree
    protocol=reward_pilot.load_protocol(config['protocol'],config['input_hashes'][config['protocol']])
    if protocol['initial_update']==0:
        return cold_start_first_batch_gate(config,protocol)
    root=Path(config['output']);folder=root/'training/batches/0000'
    def same(a,b):
        if isinstance(a,np.ndarray):return np.array_equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
        if isinstance(a,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        return a==b
    checked=0
    for seed in (0,1,2):
        arms={a:read(folder/f'{a}-seed-{seed}-records.json')['records'] for a in reward_pilot.ARMS}
        for episode in range(32):
            values=[]
            for arm,records in arms.items():
                ref=records[str(episode)]
                if file_digest(ref['path'])!=ref['sha256']:raise ValueError('Gate rollout hash changed')
                values.append(numpy_tree(torch.load(ref['path'],map_location='cpu',weights_only=True)))
            for candidate in values[1:]:
                if not same(values[0]['episode'],candidate['episode']):raise ValueError('First-batch paired trajectories differ')
                for key in ('queries','best_accel','initial_accel','selected_angles','initial_identity'):
                    if not same(values[0]['metrics'][key],candidate['metrics'][key]):raise ValueError('First-batch paired metrics differ')
            checked+=1
    _durable_write(root/'training/first-batch-gate.json',dict(status='PASS_PAIRED_ACTORS_ACTIONS_INPUTS_SCORES',
        paired_seed_episodes=checked,arms=3,model_calls=0,training_episodes_reused=True))


def cold_start_first_batch_gate(config, protocol):
    """Compare the unchanged first collection to historical B before G updates."""
    import numpy as np
    import torch
    from AlphaBrain.research.dsol.data.continuous_training_store import numpy_tree
    from AlphaBrain.research.dsol.selectors.continuous_parallel import _same_log_probability
    root=Path(config['output']);checked=0
    def same(a,b,key=None):
        if key=='log_prob':return _same_log_probability(a,b)
        if isinstance(a,np.ndarray):return np.array_equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k],k) for k in a)
        if isinstance(a,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        return a==b
    for seed in (0,1,2):
        old=read(protocol['first_batch_references'][str(seed)])['records']
        new=read(root/f'training/batches/0000/G-seed-{seed}-records.json')['records']
        for episode in range(32):
            values=[]
            for refs in (old,new):
                ref=refs[str(episode)]
                if file_digest(ref['path'])!=ref['sha256']:raise ValueError('Changed first-batch reference')
                values.append(numpy_tree(torch.load(ref['path'],map_location='cpu',weights_only=True)))
            a,b=values
            ea={k:v for k,v in a['episode'].items() if k!='raw_reward'}
            eb={k:v for k,v in b['episode'].items() if k!='raw_reward'}
            # Historical collection predates the raw-reward audit fields. Keep
            # observations, actions, reward/value/log-prob unchanged in this gate.
            for episode_value in (ea,eb):
                episode_value['transitions']=[{k:v for k,v in t.items()
                    if k not in ('best_previous','best_current','delta_queries')}
                    for t in episode_value['transitions']]
            if not same(ea,eb):raise ValueError(f'Cold-start first rollout differs: seed={seed},episode={episode}')
            for key in ('queries','best_accel','initial_accel','selected_angles','initial_identity'):
                if not same(a['metrics'][key],b['metrics'][key]):raise ValueError('Cold-start metric differs: '+key)
            checked+=1
    _durable_write(root/'training/first-batch-gate.json',dict(status='PASS_COLD_START_PAIRED_WITH_HISTORICAL_B',
        paired_episodes=checked,model_calls=0,log_probability_rule='existing four binary64 ULP rule; actions/values exact'))


def export_cpu(config):
    from AlphaBrain.research.dsol.selectors.continuous_export import export_final
    root=Path(config['output'])
    protocol=reward_pilot.load_protocol(config['protocol'],config['input_hashes'][config['protocol']])
    for name in run_names(protocol):
        export_final(root/'training/runs'/name,Path(config.get('actor_root',str(root/'actors')))/name,
                     episodes=protocol['updates']*32,updates=protocol['updates'])


def recovery_gate(folder,references,output):
    import numpy as np
    from .controlled_recovery import validate_block
    refs=read(references)['views'];checked=[]
    for spec in read(folder/'binding.json')['tasks'].values():
        validate_block(spec['request'],spec['output'])
        results=read(Path(spec['output'])/'results.json')
        for task in results['tasks']:
            prior=read(refs[f"{task['state_key']}::{task['view_id']}"]['episode_results'][task['repeat']])
            n=min(task['replans'],prior['replans'])
            if task['execution_noise_sha256'][:n]!=prior['execution_noise_sha256'][:n]:raise ValueError('Execution noises differ')
            row=next(r for r in spec['request']['block']['rows'] if all(r[k]==task[k] for k in ('state_key','view_id','method')))
            exact=task['selected_angles']==row['angles_deg']
            if exact:
                for field in ('success','steps','replans','horizon','input_sha256','execution_noise_sha256'):
                    if task[field]!=prior[field]:raise ValueError('Retained-start outcome differs: '+field)
                with np.load(task['trajectory'],allow_pickle=False) as a,np.load(prior['trajectory'],allow_pickle=False) as b:
                    if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files):raise ValueError('Retained-start trajectory differs')
            checked.append(dict(method=task['method'],repeat=task['repeat'],retained_start_exact_trajectory=exact,paired_noise=True))
    _durable_write(output,dict(status='PASS_INPUT_PREDICTION_NOISE_AND_TRAJECTORY_CHECKS',checks=checked,
        included_in_scientific_denominators=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute-authorized',action='store_true');mode.add_argument('--check-first-batch',action='store_true')
    mode.add_argument('--export-cpu',action='store_true')
    args=parser.parse_args();config=read(args.config)
    if config['config_path']!=str(Path(args.config).resolve()):raise ValueError('Configuration path identity differs')
    for p,h in config['input_hashes'].items():
        if file_digest(p)!=h:raise ValueError('Configuration input changed')
    if args.check_first_batch:first_batch_gate(config)
    elif args.export_cpu:export_cpu(config)
    else:Adaptation(config).run()


if __name__=='__main__':main()
