"""One frozen search cell, followed by all paired execution seeds.

Every queried image/feature/score and every task execution is fresh. Historical
records are optional validation references, never query responses or labels
available to the actor. No training and no success-based stopping of the block.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read,sha,write
from AlphaBrain.research.dsol.data.flow_noise import ExplicitFlowNoiseBank
from AlphaBrain.research.dsol.acquisition.search_env import SearchEnv,SearchConfig,baseline_action
from AlphaBrain.research.dsol.acquisition.search_budget import attention_observation
from AlphaBrain.research.dsol.data.search_replay import QueryResult,CachedQueries
from .search_live import simulator,verify_example
from .search_features import load_observations
from .search_runtime import PolicyRuntime,exact_prediction
from .search_domain import exact_input


def validate_block(block):
    if (block['cohort'] not in ('A','B','capability') or type(block['initial']) is not int
            or block['initial']<0 or block['repeats']!=list(range(8))):
        raise ValueError('Frozen cohort, initial and eight paired seeds required')
    expected=['nominal'] if block['cohort']=='capability' else ['stay','local_accel','ppo']
    if block['methods']!=expected or block['start'] not in ([0] if block['cohort']=='capability' else range(65,97)):
        raise ValueError('Frozen method/start contract changed')
    if block['cohort']=='B' and block['start'] not in (65,69,73,77,81,85,89,93):
        raise ValueError('Unknown frozen B starting candidate')
    return block


def verify_actor_controls(actor,trained,config):
    """Frozen CPU controls only; cached values never enter a live search cell."""
    import torch
    controls=read(config['deployment_controls'])['controls']
    with np.load(Path(config['native'])/'siglip-scores.npz',allow_pickle=False) as z:
        scores=z['scores'];visual=z['external'];wrist=z['wrist']
    with np.load(config['outcome_matrix'],allow_pickle=False) as z:poses=z['camera']
    for control in controls:
        i=control['initial']
        core=SearchEnv(poses[i],CachedQueries(visual[i],scores[i]),wrist[i,65],
            trained['score_scale'],SearchConfig(query_budget=8,**trained['search']))
        core.reset(start=65);actions=[]
        while not core.done:
            observation=attention_observation(core,core.observation())
            tensor={k:torch.as_tensor(v[None],dtype=torch.float32) for k,v in observation.items()}
            with torch.inference_mode():action=int(actor(tensor).argmax(-1).item())
            actions.append(action);core.step(action)
        if actions!=control['actions'] or core.best!=control['selected']:
            raise ValueError('Actor runtime changed frozen control decisions')
    return len(controls)


def rollout_episode(request,infer,example,bank,state_key,repeat,identity,*,horizon=300):
    """Native five-step replanning and step-observation timing, no final reset."""
    if horizon!=300 or repeat not in range(8):
        raise ValueError('Frozen execution horizon/seed changed')
    calibration={k:np.array(example[k],copy=True) for k in ('camera_intrinsics','camera_to_world_opencv')}
    steps=0;replan=0;success=False;actions=[];noises=[];frames=[];states=[]
    while steps<horizon and not success:
        noise=bank.get(state_key,repeat,replan)
        result=infer(example,noise['noise'],dict(identity,repeat=repeat,replan=replan),capture=False)
        chunk=np.clip(result['actions'][:min(5,horizon-steps)],-1,1)
        if replan%10==0:
            frames.append(np.concatenate(example['image'],axis=1))
        states.append(np.asarray(example['state']).copy())
        response=request(dict(kind='step',actions=chunk))
        count=response['executed']
        if type(count) is not int or not 1<=count<=len(chunk):
            raise ValueError('Invalid simulator executed-step count')
        example=response['example'];success=bool(response['success'])
        actions.extend(chunk[:count]);steps+=count;replan+=1;noises.append(noise['noise_sha256'])
        if any(not np.array_equal(example[k],v) for k,v in calibration.items()):
            raise ValueError('External camera moved during task')
    return dict(success=success,steps=steps,replans=replan,execution_noise_sha256=noises),dict(
        actions=np.asarray(actions),sample_frames=np.asarray(frames),replan_states=np.asarray(states))


def run(config_path,block_path,output,*,execute=False):
    if not execute:raise ValueError('Explicit --execute required')
    config=read(config_path);block=validate_block(read(block_path));output=Path(output)
    output.mkdir(parents=True,exist_ok=False);runtime=None;tasks=[];searches=[];begin=time.time()
    try:
        for path,digest in config['input_hashes'].items():
            if sha(path)!=digest:raise ValueError('Bound evaluation asset changed: '+path)
        if sha(block_path)!=config['block_hashes'].get(str(block_path)):
            raise ValueError('Frozen evaluation block changed')
        import torch
        from AlphaBrain.research.dsol.selectors.search_attention import SharedSearchActor
        actor_asset=config['actor']
        if sha(actor_asset['path'])!=actor_asset['sha256']:raise ValueError('Frozen actor changed')
        limit=8*60 if block['cohort']=='capability' else 3*8*60+2*(1+8+8)
        runtime=PolicyRuntime(config,output,limit)
        trained=torch.load(actor_asset['path'],map_location='cpu',weights_only=True)
        actor=SharedSearchActor(trained['visual_dim']).eval();actor.load_state_dict(trained['state_dict'],strict=True)
        actor.requires_grad_(False)
        controls=verify_actor_controls(actor,trained,config)
        write(output/'actor-runtime-admission.json',dict(status='PASS',controls=controls))
        search_config=SearchConfig(query_budget=8,**trained['search'])
        initial=block['initial'];spec=read(config['inputs'])['records'][initial];state_key=spec['state']['pair_key']
        if block['cohort']=='A':
            with np.load(config['anchors'],allow_pickle=False) as z:score_noise=z['first_score_noise'][initial,:2].copy()
            cached=load_observations(spec)
            with np.load(Path(config['native'])/'initials'/f'{initial:02d}.npz',allow_pickle=False) as z:
                native={k:z[k].copy() for k in ('trace','actions')}
            with np.load(Path(config['native'])/'siglip-scores.npz',allow_pickle=False) as z:
                native.update({k:z[k][initial].copy() for k in ('scores','external','wrist')})
            bank_path=config['execution_noise_manifest']
        else:
            with np.load(config['domain_noise'],allow_pickle=False) as z:score_noise=z['score'][initial].copy()
            cached=None;native=None;bank_path=config['domain_execution_noise_manifest']
        bank=ExplicitFlowNoiseBank(Path(bank_path),verify_file=True)
        with simulator(config_path,config,initial,output) as request:
            poses=request(dict(kind='catalog'))['poses']
            if block['cohort']=='A' and not np.array_equal(poses,cached['camera_to_world_opencv']):
                raise ValueError('Public camera catalog changed from frozen graph')
            for method in block['methods']:
                request(dict(kind='reset'));directory=output/method;directory.mkdir()
                observations={};query_seconds=[]
                class Source:
                    size=97;feature_dim=4608;score_samples=2
                    def query(self,candidate):
                        before=time.perf_counter()
                        item=request(dict(kind='render',candidate=int(candidate)))['example']
                        if cached is not None:verify_example(item,cached,candidate)
                        values=[runtime.infer(item,score_noise[m],dict(stage='score',method=method,candidate=int(candidate),member=m),capture=True) for m in (0,1)]
                        if native is not None:
                            for m,value in enumerate(values):
                                reference={k:native[k][candidate,m] for k in ('trace','actions')}
                                reference.update(score=native['scores'][candidate,m],external=native['external'][candidate],wrist=native['wrist'][candidate])
                                exact_prediction(value,reference)
                        observations[int(candidate)]=item
                        scores=np.array([v['score'] for v in values]);self.wrist=values[0]['wrist']
                        np.savez_compressed(directory/f'query-{candidate:03d}.npz',scores=scores,
                            trace=np.stack([v['trace'] for v in values]),actions=np.stack([v['actions'] for v in values]),
                            external_image=item['image'][0],wrist_image=item['image'][1])
                        query_seconds.append(time.perf_counter()-before)
                        return QueryResult(values[0]['external'],float(scores.mean()),float(scores.std()),2)
                started=time.perf_counter()
                if method=='nominal':
                    selected=0;observations[0]=request(dict(kind='render',candidate=0))['example']
                    info=dict(selected=0,queries=0,reason='fixed_nominal_capability',visited=[])
                else:
                    source=Source();prefetch=source.query(block['start'])
                    # reset charges and consumes the already acquired first observation once.
                    original_query=source.query;pending=[prefetch]
                    def query(candidate):
                        if pending:
                            if candidate!=block['start']:raise ValueError('Initial prefetch identity changed')
                            return pending.pop()
                        return original_query(candidate)
                    source.query=query
                    core=SearchEnv(poses,source,source.wrist,trained['score_scale'],search_config)
                    core.reset(start=block['start'])
                    while not core.done:
                        if method=='ppo':
                            observation=attention_observation(core,core.observation())
                            tensor={k:torch.as_tensor(v[None],dtype=torch.float32) for k,v in observation.items()}
                            with torch.inference_mode():action=int(actor(tensor).argmax(-1).item())
                        else:action=baseline_action(core,method)
                        core.step(action)
                    info=core.info();selected=info['selected']
                info.update(cohort=block['cohort'],initial=initial,start=block['start'],method=method,
                    query_seconds=query_seconds,search_seconds=time.perf_counter()-started)
                write(directory/'search.json',info);searches.append(info)
                for repeat in block['repeats']:
                    request(dict(kind='reset'))
                    item=request(dict(kind='render',candidate=int(selected)))['example']
                    exact_input(item,observations[selected])
                    result,arrays=rollout_episode(request,runtime.infer,item,bank,state_key,repeat,
                        dict(stage='execution',cohort=block['cohort'],initial=initial,method=method,start=block['start']))
                    result.update(cohort=block['cohort'],initial=initial,state_key=state_key,start=block['start'],
                        method=method,repeat=repeat,selected=selected,queries=info['queries'])
                    asset=directory/f'episode-{repeat}.npz';np.savez_compressed(asset,**arrays)
                    result.update(trajectory=str(asset),trajectory_sha256=sha(asset))
                    write(directory/f'episode-{repeat}.json',result);tasks.append(result)
                    write(output/'progress.json',dict(tasks=len(tasks),calls=runtime.calls))
        write(output/'status.json',dict(status='COMPLETE',config_sha256=sha(config_path),block_sha256=sha(block_path),
            calls=runtime.calls,tasks=tasks,searches=searches,elapsed_seconds=time.time()-begin))
    except BaseException as error:
        write(output/'failure.json',dict(status='FAILED_STOPPED',error=repr(error),
            calls=0 if runtime is None else runtime.calls,completed_tasks=len(tasks)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--block',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();run(args.config,args.block,args.output,execute=args.execute)
