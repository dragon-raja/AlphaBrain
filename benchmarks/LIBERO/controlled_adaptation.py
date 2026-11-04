"""Finite few-shot PPO coordinator using existing collectors and CPU learners.

No retries, host migration or science generation. Every phase reserves its
maximum before dispatch; each published actor is returned by the learner over
SSH rather than inferred from a mutable NFS pointer.
"""
from pathlib import Path
import json
import shlex
import subprocess
import time
from AlphaBrain.research.dsol.data.artifacts import read,write
from AlphaBrain.research.dsol.execution.manifest import file_digest
from AlphaBrain.research.dsol.execution.queue import exclusive
from AlphaBrain.research.dsol.data import domain_adaptation
from .controlled_campaign import run_stage,bind_node


def learner(config,path,records=None):
    node=bind_node(read(config['node_template'])['nodes']['fresh'],read(config['source_manifest']),workers_per_gpu=1)
    env=dict(node['environment'],CUDA_VISIBLE_DEVICES='',WANDB_MODE='disabled')
    command=['env',*[k+'='+v for k,v in env.items()],node['python'],'-m','benchmarks.LIBERO.controlled_learner',
             '--config',str(path),'--execute-cpu']
    if records is not None:command+=['--records',str(records)]
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
        '-o','ConnectTimeout=8',config['aliases']['fresh'],shlex.join(command)],capture_output=True,text=True,timeout=900)
    if result.returncode:raise RuntimeError('CPU learner failed: '+result.stderr[-4000:])
    return json.loads(result.stdout.splitlines()[-1])


def collection(common,protocol,publications,version):
    c=dict(common,kind='rollout',blocks=[],limits=dict(episodes=0,score=0,paired_execution=0,nominal_execution=0))
    c['input_hashes']=dict(common['input_hashes'])
    for arm in protocol['arms']:
        for seed in protocol['seeds']:
            run=f'{arm}-seed-{seed}';p=publications[run]
            if p['version']!=version or p['episodes']!=8*version or file_digest(p['policy'])!=p['policy_sha256']:
                raise ValueError('Mixed/stale learner publication')
            rows=read(protocol['schedules'][str(seed)])[version*8:(version+1)*8]
            limits=dict(episodes=8,score=128,paired_execution=0,nominal_execution=0)
            addresses=[dict(model=c['operation_model'],task=r['task'],initial=r['state_key'],
                view=f"{run}:episode-{r['episode']}",stream='score',repeat=seed,noise=f"train-{r['initial']}") for r in rows]
            c['blocks'].append(dict(index=len(c['blocks']),host='fresh',run=run,arm=arm,seed=seed,version=version,
                policy=p['policy'],policy_sha256=p['policy_sha256'],reward_scale=c['observation_scale'],
                rows=rows,limits=limits,addresses=addresses))
            c['input_hashes'][p['policy']]=p['policy_sha256']
            for k,n in limits.items():c['limits'][k]+=n
    return c


def first_batch_gate(config,publications):
    node=bind_node(read(config['node_template'])['nodes']['fresh'],read(config['source_manifest']),workers_per_gpu=1)
    pairs=[(publications[f'{init}-first_batch_fixed-seed-{s}']['policy'],
            publications[f'{init}-window-seed-{s}']['policy']) for init in ('source','random') for s in (0,1,2)]
    code='''import torch,json
pairs=PAIRS
result=[]
for a,b in pairs:
 x=torch.load(a,map_location='cpu',weights_only=True)['state_dict'];y=torch.load(b,map_location='cpu',weights_only=True)['state_dict']
 result.append(set(x)==set(y) and all(torch.equal(x[k],y[k]) for k in x))
print(json.dumps(dict(status='PASS' if all(result) else 'FAIL',equal=result)))
'''.replace('PAIRS',repr(pairs))
    command=['env','CUDA_VISIBLE_DEVICES=','WANDB_MODE=disabled',node['python'],'-']
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',config['aliases']['fresh'],
        shlex.join(command)],input=code,capture_output=True,text=True,timeout=120)
    if r.returncode:raise RuntimeError('First batch CPU comparison failed: '+r.stderr[-1000:])
    return json.loads(r.stdout.splitlines()[-1])


def run(config_path,*,execute=False):
    if not execute:raise ValueError('Explicit bounded adaptation execution required')
    config=read(config_path);root=Path(config['output'])
    if set(config['aliases'])!={'fresh'} or set(config['capacity'])!={'fresh'}:
        raise ValueError('Fresh-only phase; unavailable nodes cannot receive work')
    for p,h in config['input_hashes'].items():
        if file_digest(p)!=h:raise ValueError('Changed coordinator input: '+p)
    proof=read(config['trajectory_proof'])
    if proof['status']!='PASS':raise ValueError('Full operation trajectory gate has not passed')
    common=read(config['collection_common']);protocol=domain_adaptation.from_campaign(common)
    if config['training_limits']!=dict(episodes=768,score=12288,paired_execution=0,nominal_execution=0):
        raise ValueError('Changed bounded adaptation budget')
    start=root/'coordinator-start';start.mkdir(parents=True,exist_ok=False)
    with exclusive(start):
        try:return _run(config,root,common,protocol)
        except BaseException as error:
            write(root/'failure.json',dict(status='HALTED_NO_RETRY',error=repr(error),time=time.time()))
            raise


def _run(config,root,common,protocol):
    publications={};settled=[]
    for arm in protocol['arms']:
        for seed in protocol['seeds']:
            run=f'{arm}-seed-{seed}'
            publications[run]=learner(config,config['learners'][run])
    write(root/'initial-publications.json',publications)
    for version in range(8):
        c=collection(common,protocol,publications,version);phase=f'update-{version:02d}'
        cp=root/'campaigns'/f'{phase}.json';write(cp,c)
        reservation=root/'reservations'/f'{phase}.json'
        if reservation.exists():raise ValueError('Existing reservation; explicit recovery required')
        write(reservation,dict(campaign=str(cp),sha256=file_digest(cp),limits=c['limits'],status='RESERVED'))
        run_stage(config,phase,cp,kind='training',cpu=True)
        result=run_stage(config,phase,cp,kind='training',cpu=False)
        settled.append(result);write(root/'settlements'/f'{phase}.json',result)
        for block in c['blocks']:
            folder=root/'execution'/f'{phase}-real/results'/f"block-{block['index']:04d}"
            record=read(folder/'collection.json');run=block['run']
            batch=root/'batches'/f'{run}-{version:02d}.json'
            write(batch,dict(run=run,version=version,policy_sha256=block['policy_sha256'],records=record['records']))
            publications[run]=learner(config,config['learners'][run],batch)
        write(root/'endpoints'/f'update-{version+1:02d}.json',publications)
        if version==0:
            gate=first_batch_gate(config,publications);write(root/'first-batch-gate.json',gate)
            if gate['status']!='PASS':raise ValueError('Matched first-batch policies diverged; stop before second batch')
        write(root/'current.json',dict(status='TRAINING' if version<7 else 'TRAINING_COMPLETE',
            updates_per_run=version+1,episodes_per_run=8*(version+1),runs=12,
            training_calls=sum(s['actual_model_calls'] for s in settled),evaluation_complete=False))
    return dict(status='TRAINING_COMPLETE',publications=publications,settlements=settled)
