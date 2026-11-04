"""Reusable composition of immutable LIBERO campaigns and center authority.

Science lives in the registered adapters; this runner owns only fresh bindings,
node probes, per-phase ledgers and bounded permits. Unknown outcomes never retry.
All locations and existing environment identities come from explicit config.
"""
from copy import deepcopy
from pathlib import Path
import subprocess
import time
import uuid
from AlphaBrain.research.dsol.data.artifacts import read
from AlphaBrain.research.dsol.execution.authority_wire import Channel,identity,ssh_command
from AlphaBrain.research.dsol.execution.manifest import digest,file_digest,task_key
from AlphaBrain.research.dsol.execution.controller_ledger import ControllerLedger
from AlphaBrain.research.dsol.execution.controlled_queue import run_controlled_queue
from AlphaBrain.research.dsol.execution.model_authority import ModelAuthorityBackend
from AlphaBrain.research.dsol.execution.resources import worker_slots
from AlphaBrain.research.dsol.execution.queue import exclusive,_durable_write
from .continuous_authority import source_identity,validate_scientific_spec
from .continuous_authority_plan import prepare_controller,plans


def adapter_for(kind):
    if kind=='rldx_panel':
        from . import rldx_spatial_panel
        return rldx_spatial_panel,'controlled_rldx_operation_v1'
    if kind=='rldx_operation':
        from . import controlled_rldx_operation
        return controlled_rldx_operation,'controlled_rldx_operation_v1'
    if kind=='rldx_basic':
        from . import controlled_rldx
        return controlled_rldx,'controlled_rldx_v1'
    if kind=='groot_admission':
        from . import controlled_groot_admission
        return controlled_groot_admission,'controlled_groot_admission_v1'
    if kind=='target_panel':
        from . import controlled_target_panel
        return controlled_target_panel,'controlled_target_panel_v1'
    if kind=='target_recovery':
        from . import controlled_target_recovery
        return controlled_target_recovery,'controlled_target_recovery_v1'
    if kind=='target_evaluation':
        from . import controlled_target_evaluation
        return controlled_target_evaluation,'controlled_target_evaluation_v1'
    if kind=='admission':
        from . import controlled_admission
        return controlled_admission,'controlled_admission_v1'
    if kind=='training':
        from . import controlled_training
        return controlled_training,'controlled_training_v1'
    if kind=='recovery':
        from . import controlled_recovery
        return controlled_recovery,'controlled_recovery_v1'
    raise ValueError('Unregistered campaign kind')


def bind_node(node,snapshot,*,workers_per_gpu=2):
    worker_slots('validation',[0],workers_per_gpu=workers_per_gpu)
    node=deepcopy(node);previous=node['source']
    node.update(source=snapshot['release'],source_artifacts=snapshot['artifacts'],workers_per_gpu=workers_per_gpu)
    node['environment']['PYTHONPATH']=node['environment']['PYTHONPATH'].replace(previous,snapshot['release'])
    # Channel itself enforces CUDA_VISIBLE_DEVICES='' and WANDB_MODE=disabled.
    # They are intentionally NOT permitted caller overrides.
    return node


def campaign_nodes(template, tasks, aliases, excluded_hosts=()):
    """Select explicit task hosts before probing; never migrate a task.

    Historical templates may include temporarily unavailable machines. Unused
    nodes must not become dependencies of an otherwise independent phase.
    """
    hosts={task['host'] for task in tasks}
    if (not hosts or hosts & set(excluded_hosts)
            or not hosts.issubset(template) or not hosts.issubset(aliases)):
        raise ValueError('Campaign contains an excluded or unbound host; no automatic reassignment')
    return {host:deepcopy(template[host]) for host in sorted(hosts)}


def campaign_capacity(config,hosts):
    """Explicit resource slots, independent of task/noise identity.

    Legacy native campaigns keep their admitted 8x2 capacity. A new model must
    supply a separate measured/bounded profile; this does not itself admit it.
    """
    capacity=config.get('capacity')
    if capacity is None:
        capacity={h:dict(gpu_ids=list(range(8)),workers_per_gpu=2) for h in hosts}
    # A final wave can use fewer hosts than the admitted fleet. Keep the
    # declared host/slot assignment, selecting only its configured resources.
    # Missing hosts remain an error; this never assigns work to another node.
    hosts=set(hosts)
    if not hosts or not hosts.issubset(capacity):
        raise ValueError('Capacity must cover the selected hosts')
    capacity={h:capacity[h] for h in sorted(hosts)}
    slots={}
    for h,c in capacity.items():
        if set(c)!={'gpu_ids','workers_per_gpu'}:raise ValueError('Explicit GPU/worker capacity required')
        slots[h]=worker_slots(h,c['gpu_ids'],workers_per_gpu=c['workers_per_gpu'])
    return deepcopy(capacity),slots


def resident_runtime_slots(campaign, slots):
    """Keep each resident slot on one immutable runtime for this campaign.

    Allocate only among the explicitly assigned host's blocks. Identical
    simulator contents share a pool even when stored at different paths. Pools
    are recomputed for the next campaign, after the old workers have exited.
    """
    if campaign.get('native_policy') is None:
        return {b['index']:slots[b['host']] for b in campaign['blocks']}
    groups={host:{} for host in slots};identities={}
    for block in campaign['blocks']:
        path=block.get('sim_config',campaign.get('sim_config'))
        if not path:raise ValueError('Native resident block requires an explicit runtime')
        if path not in identities:identities[path]=digest(read(path))
        groups[block['host']].setdefault(identities[path],[]).append(block['index'])
    result={}
    for host,runtimes in groups.items():
        keys=sorted(runtimes);available=slots[host]
        if len(keys)>len(available):raise ValueError('Not enough resident slots for immutable runtime groups')
        assigned={key:[available[i]] for i,key in enumerate(keys)}
        for slot in available[len(keys):]:
            candidates=[key for key in keys if len(assigned[key])<len(runtimes[key])]
            if not candidates:break
            key=min(candidates,key=lambda key:((len(assigned[key])+1)/len(runtimes[key]),key))
            assigned[key].append(slot)
        for key,indices in runtimes.items():
            for index in indices:result[index]=assigned[key]
    return result



def phase_duration(config,block_count,*,cpu,now):
    """Bind an explicit new-model phase window; preserve legacy defaults.

    A longer permit never changes task/call limits or renews a running lease.
    Its duration is chosen once before dispatch and capped by the frozen deadline.
    """
    duration=1800 if cpu else 7200*max(1,(block_count+143)//144)
    if not cpu:
        explicit=config.get('phase_timeout_seconds',duration)
        if type(explicit) is not int or not 60<=explicit<=86400:
            raise ValueError('Explicit bounded phase timeout required')
        remaining=config['dispatch_deadline_utc_seconds']-now
        if remaining<=0:raise ValueError('Campaign dispatch deadline expired; explicit recovery required')
        duration=min(explicit,remaining)
    return duration


def phase_evidence(config,phase,*,cpu=False):
    """Canonical receipt location shared by collectors, coordinators and audits."""
    if not isinstance(phase,str) or not phase or Path(phase).name!=phase or phase in ('.','..'):
        raise ValueError('One explicit phase component required')
    return Path(config['output'])/'execution'/(phase+('-cpu' if cpu else '-real'))


def collection_records(config,phase,campaign):
    """Return only a settled, complete and hash-verified on-policy barrier."""
    from .controlled_training import request_for
    root=phase_evidence(config,phase);done=read(root/'completion.json');c=read(campaign)
    if done['status']!='COMPLETE' or done['dispatch_enabled'] is not False:
        raise ValueError('Collection owner has not settled')
    records={};expected={str(r['episode']) for b in c['blocks'] for r in b['rows']}
    for b in c['blocks']:
        folder=root/'results'/f"block-{b['index']:04d}";receipt=read(folder/'complete.json');path=folder/'collection.json'
        if receipt['request_sha256']!=digest(request_for(campaign,b['index'])) or receipt['artifacts']['collection.json']!=file_digest(path):
            raise ValueError('Collection request/receipt changed')
        value=read(path)
        if value['status']!='COMPLETE':raise ValueError('Incomplete collector')
        for key,ref in value['records'].items():
            if key in records or file_digest(ref['path'])!=ref['sha256']:raise ValueError('Repeated/changed rollout')
            records[key]=ref
    if set(records)!=expected or done['counts']['episodes']!=len(expected):raise ValueError('Incomplete on-policy collection barrier')
    return records


def remote_control_base(config,node,base,host):
    """Preserve experiment isolation when a shared template overrides the root.

    Frozen releases keep their paths; recovery uses a separate node binding.
    """
    base=Path(base)
    if 'control_root' not in node:return base/host
    control=Path(node['control_root']);experiment=config['experiment_id']
    if not control.is_absolute():raise ValueError('Explicit absolute node control root required')
    if any(not isinstance(v,str) or not v or Path(v).name!=v or v in ('.','..')
           for v in (experiment,host,base.name)):
        raise ValueError('Single experiment, host and phase path components required')
    return control/experiment/base.name/host


def run_stage(config, phase, campaign, *, kind, cpu):
    """One explicit attempt. The enclosing coordinator reserves campaign budget."""
    adapter,adapter_name=adapter_for(kind)
    if kind in ('rldx_panel','rldx_operation','rldx_basic','admission','groot_admission','target_evaluation','target_recovery','target_panel') and 'capacity' not in config:
        raise ValueError('New runtime admission needs an explicit bounded capacity')
    campaign=Path(campaign).resolve();c=adapter.validate_campaign(campaign)
    if not 1<=len(c['blocks'])<=288:raise ValueError('Bounded campaign required')
    label=phase+('-cpu' if cpu else '-real')
    base=Path(config['local_root'])/label
    evidence=phase_evidence(config,phase,cpu=cpu)
    evidence.mkdir(parents=True,exist_ok=False)
    lock=Path(config['local_root'])/('launcher-'+label);lock.mkdir(parents=True,exist_ok=False)
    with exclusive(lock):
        return _run(config,campaign,c,adapter,adapter_name,base,evidence,phase,cpu)


def _run(config,campaign,c,adapter,adapter_name,base,evidence,phase,cpu):
    for p,h in config['input_hashes'].items():
        if file_digest(p)!=h:raise ValueError('Changed execution configuration input')
    snapshot=read(config['source_manifest'])
    if snapshot['source_identity']!=source_identity():raise ValueError('Controller source identity changed')
    b=deepcopy(read(config['node_template']))
    if b.get('schema')!='LIBERO_LIVE_BINDING_V1':
        raise ValueError('Reviewed node template schema required before any remote write')
    b['nodes']=campaign_nodes(b['nodes'],c['blocks'],config['aliases'],config.get('excluded_hosts',()))
    capacity,slots=campaign_capacity(config,b['nodes'])
    eligible=resident_runtime_slots(c,slots)
    b.update(campaign=str(campaign),authority=identity(),tasks={},resident_workers=True,
        resident_simulator=True,mode='cpu_contract' if cpu else 'libero',scientific_adapter=adapter_name,
        status='NEW_BOUND_PHASE',execution_source_archive_sha256=snapshot['archive_sha256'])
    b['nodes']={h:bind_node(n,snapshot,workers_per_gpu=capacity[h]['workers_per_gpu']) for h,n in b['nodes'].items()}
    experiment=config['experiment_id']+'-'+phase
    for index in range(len(c['blocks'])):
        request=adapter.request_for(campaign,index);block=request['block'];host=request['host']
        key=task_key(experiment,file_digest(campaign),block['addresses'])
        b['tasks'][key]=dict(host=host,stage='formal',addresses=block['addresses'],
            eligible_slots=eligible[index],
            identity_experiment_id=experiment,output=str(evidence/'results'/f'block-{index:04d}'),
            request=request,limits=block['limits'])
    b['budget']=dict(campaign_sha256=file_digest(campaign),peers=list(b['nodes']),stages=['formal'],limits=c['limits'],
        tasks=[dict(key=k,**{x:s[x] for x in ('addresses','stage','limits','output')}) for k,s in b['tasks'].items()])
    bp=evidence/'binding.json';_durable_write(bp,b)
    for key in b['tasks']:validate_scientific_spec(b,key)
    plans(b,bp)  # Check the actual queue address schema before any remote write.
    (evidence/'checkpoints').mkdir()
    backend_config=dict(schema='LIVE_MODEL_AUTHORITY_V1',mode=b['mode'],resident_workers=True,
        binding_sha256=digest(b),authority=identity(),checkpoint_directory=str(evidence/'checkpoints'),peers={})
    health={h:v for h,v in config.get('node_health',{}).items() if h in b['nodes']}
    if not cpu and c.get('native_policy') is not None and 'cy' in b['nodes'] and 'cy' not in health:
        raise ValueError('Borrowed cy native production requires center-local health monitoring')
    if health:backend_config['node_health']=health
    if 'launch_timeout_seconds' in config:
        backend_config['launch_timeout_seconds'] = config['launch_timeout_seconds']
    if 'persistent_inspection' in config:
        backend_config['persistent_inspection'] = config['persistent_inspection']
    if 'parallel_launches' in config:
        backend_config['parallel_launches'] = config['parallel_launches']
    for host,node in b['nodes'].items():
        remote_base=remote_control_base(config,node,base,host)
        remote_binding=remote_base/'binding.json';store=remote_base/'store'
        code=('from pathlib import Path;p=Path('+repr(str(remote_binding))+');'
              'p.parent.mkdir(parents=True,exist_ok=False);p.write_bytes('+repr(bp.read_bytes())+')')
        target=dict(transport='ssh',alias=config['aliases'][host],python=node['python'],source=snapshot['release'],
            root=str(store),binding=str(remote_binding),agent='benchmarks.LIBERO.continuous_authority',environment=node['environment'])
        if 'public_endpoint' in node:target['public_endpoint']=deepcopy(node['public_endpoint'])
        subprocess.run(ssh_command(target)+['python3','-'],input=code,text=True,check=True,timeout=120,capture_output=True)
        with Channel(target,prepare=True,agent=target['agent'],expected_source=source_identity(),environment=target['environment'],
                     hello_timeout=config.get('launch_timeout_seconds',60)) as channel:
            hello=channel.hello;channel.wire.send(dict(kind='PROBE',session=hello['session']))
            if channel.wire.receive()['kind']!='HEALTHY':raise RuntimeError('Node probe failed')
        target['expected']={k:hello[k] for k in ('source','identity','store_id')}
        backend_config['peers'][host]=target
    root=base/'center';prepare_controller(bp,root,backend_config);ledger=ControllerLedger(root/'custody')
    ident=identity();owner=dict(host=ident['host'],boot_id=ident['boot'],session=uuid.uuid4().hex)
    epoch=ledger.claim(owner);plan=plans(b,bp)['formal'];now=time.time()
    duration=phase_duration(config,len(c['blocks']),cpu=cpu,now=now)
    permit=dict(authorization='CPU_SIMULATION_ONLY' if cpu else 'EXPLICIT_MODEL_STAGE_V1',binding_sha256=digest(b),
        ledger_plan_sha256=digest(b['budget']),queue_sha256=digest(plan),owner=owner,epoch=epoch,stage='formal',
        issued_at=now-1,dispatch_until=now+duration,finish_until=now+duration+600)
    pp=root/'permit.json';_durable_write(pp,permit);_durable_write(evidence/'permit.json',permit)
    backend=ModelAuthorityBackend(root/'backend',{k:s['limits'] for k,s in b['tasks'].items()},config=backend_config)
    try:
        run_controlled_queue(root/'formal',ledger,owner,epoch,pp,backend,simulate=cpu,execute=not cpu,
            poll_seconds=.05,timeout_seconds=duration+600)
    except BaseException as error:
        _durable_write(evidence/'failure.json',dict(status='HALTED_NO_RETRY',error=repr(error)))
        raise
    state=ledger.read()[1]
    if state['dispatch_enabled'] or any(t['status']!='COMPLETE' for t in state['tasks'].values()):
        raise ValueError('Campaign did not finish with settled ownership')
    counts={k:sum(t['consumed'][k] for t in state['tasks'].values()) for k in c['limits']}
    result=dict(status='COMPLETE',phase=phase,cpu=cpu,seconds=time.time()-now,counts=counts,
        actual_model_calls=0 if cpu else sum(counts[k] for k in ('score','paired_execution','nominal_execution')),
        actual_episodes=0 if cpu else counts['episodes'],metrics=backend.metrics,optimizer_updates=0,dispatch_enabled=False)
    _durable_write(evidence/'completion.json',result)
    return result
