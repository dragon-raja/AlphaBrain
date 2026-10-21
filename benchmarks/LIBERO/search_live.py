"""Bounded real-render queries and paired full tasks for the frozen search actor.

No cached observation is returned to the actor. Saved inputs/scores are used
only as exact engineering checks after each real render/inference.
"""

import argparse
from contextlib import contextmanager
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read,sha,write


def verify_example(example,cached,candidate):
    values=dict(external_images=example['image'][0],wrist_images=example['image'][1],
                robot_states=example['state'],camera_intrinsics=example['camera_intrinsics'],
                camera_to_world_opencv=example['camera_to_world_opencv'])
    for key,value in values.items():
        if not np.array_equal(value,cached[key][candidate]):
            raise ValueError('Live renderer differs from frozen input: '+key)


@contextmanager
def simulator(config_path,config,initial,output):
    child=None
    with tempfile.TemporaryDirectory(prefix='dsol-live-search-') as temporary:
        address=str(Path(temporary)/'sim.sock')
        with Listener(address,family='AF_UNIX') as listener:
            environment={**os.environ,'PYTHONPATH':':'.join([config['sim_site'],config['source'],
                config['source']+'/scripts/vla_shared']+config['sim_extra_paths']),
                'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl'}
            command=[config['sim_python'],'-u','-m','benchmarks.LIBERO.search_simulator',
                     '--config',str(config_path),'--initial',str(initial),'--socket',address,'--execute']
            try:
                with (output/'simulator.log').open('xb') as log:
                    child=subprocess.Popen(command,env=environment,cwd=config['source'],stdout=log,stderr=subprocess.STDOUT)
                    listener._listener._socket.settimeout(180)
                    with listener.accept() as connection:
                        def request(payload=None):
                            if payload is not None:
                                connection.send(payload)
                            if not connection.poll(180):
                                raise TimeoutError('Simulator response deadline exceeded')
                            value=connection.recv()
                            if value.get('error'):
                                raise RuntimeError(value['error'])
                            return value
                        if request()['status']!='READY':
                            raise ValueError('Simulator failed to initialize')
                        yield request
                        request(dict(kind='close'))
                    if child.wait(timeout=60)!=0:
                        raise RuntimeError('Simulator exited unsuccessfully')
            finally:
                if child is not None and child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        child.kill();child.wait()


def run(config_path,output,*,execute=False):
    if not execute:
        raise ValueError('Explicit --execute required')
    import torch
    from AlphaBrain.model.framework.base_framework import BaseFramework
    from AlphaBrain.model.pi05_features import SiglipFeatureCapture
    from AlphaBrain.research.dsol.acquisition.search_env import SearchEnv,SearchConfig,baseline_action
    from AlphaBrain.research.dsol.acquisition.search_budget import attention_observation
    from AlphaBrain.research.dsol.data.search_replay import QueryResult,CachedQueries
    from AlphaBrain.research.dsol.data.flow_noise import ExplicitFlowNoiseBank,tensor_sha256
    from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores
    from AlphaBrain.research.dsol.metrics.accel_inference import _validated_trace_output
    from AlphaBrain.research.dsol.selectors.search_attention import SharedSearchActor
    from .search_features import load_observations

    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    config=read(config_path);counts={'live_search':0,'task_search':0,'execution':0}
    limits={'live_search':128,'task_search':68,'execution':1440}
    started=time.monotonic();queries=[];tasks=[]
    try:
        for path,digest in config['input_hashes'].items():
            if sha(path)!=digest:
                raise ValueError('Bound live source changed: '+path)
        native=Path(config['native']);receipt=read(native/'features.json')
        if receipt['status']!='COMPLETE' or sha(native/'siglip-scores.npz')!=receipt['asset_sha256']:
            raise ValueError('Native batch1 table not admitted')
        training=Path(config['training']);trained=read(training/'receipt.json')
        if trained['status']!='COMPLETE':
            raise ValueError('Frozen training/evaluation not complete')
        actor_path=training/'seed-0'/'actor.pt'
        if sha(actor_path)!=read(training/'seed-0'/'receipt.json')['actor_sha256']:
            raise ValueError('Predetermined seed0 actor changed')
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        checkpoint=config['models']['canonical']
        if sha(Path(checkpoint['path'])/'model.safetensors')!=checkpoint['weights_sha256']:
            raise ValueError('Frozen manipulation policy changed')
        policy=BaseFramework.from_pretrained(checkpoint['path'],strict_checkpoint=True).to(torch.bfloat16).to('cuda').eval()
        policy.gripper_remap=False;policy.requires_grad_(False)
        actor_data=torch.load(actor_path,map_location='cpu',weights_only=True)
        actor=SharedSearchActor(actor_data['visual_dim']).eval()
        actor.load_state_dict(actor_data['state_dict'],strict=True)
        scale=actor_data['score_scale']
        inputs=read(config['inputs'])['records']
        with np.load(config['anchors'],allow_pickle=False) as z:
            scoring_noises=z['first_score_noise'].copy()
        with np.load(native/'siglip-scores.npz',allow_pickle=False) as z:
            native_scores=z['scores'].copy();native_visual=z['external'].copy();native_wrist=z['wrist'].copy()
        with np.load(config['outcome_matrix'],allow_pickle=False) as z:
            poses=z['camera'].copy();historical_success=z['success'][0].copy()
        # Different isolated Torch environments must select the same actions on
        # these fixed deployment controls before any new VLA inference.
        controls=read(training/'seed-0'/'deployment-controls.json')['controls']
        for control in controls:
            i=control['initial']
            core=SearchEnv(poses[i],CachedQueries(native_visual[i],native_scores[i]),native_wrist[i,65],
                           scale,SearchConfig(query_budget=8,**actor_data['search']))
            core.reset(start=65);actions=[]
            while not core.done:
                obs=attention_observation(core,core.observation())
                tensor={k:torch.as_tensor(v[None],dtype=torch.float32) for k,v in obs.items()}
                with torch.inference_mode():
                    action=int(actor(tensor).argmax(-1).item())
                actions.append(action);core.step(action)
            if actions!=control['actions'] or core.best!=control['selected']:
                raise ValueError('Actor changed selections across isolated Torch runtimes')
        write(output/'actor-runtime-admission.json',dict(status='PASS',controls=len(controls)))
        noise_bank=ExplicitFlowNoiseBank(Path(config['execution_noise_manifest']),verify_file=True)
        tower=policy.vlm_interface.model.vision_tower

        def infer(example,noise,stage,identity,capture):
            if counts[stage]>=limits[stage]:
                raise ValueError('Bounded model trajectory budget exhausted')
            counts[stage]+=1
            with (output/'calls.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(stage=stage,index=counts[stage],**identity))+'\n')
            tap=SiglipFeatureCapture(tower,patch_grid=(16,16),view_names=('external','wrist','masked_dummy'))
            from contextlib import nullcontext
            torch.cuda.synchronize();begin=time.perf_counter()
            with torch.inference_mode(),(tap if capture else nullcontext()):
                prediction=policy.predict_action(examples=[example],noise=noise[None],return_flow_trace=True)
            torch.cuda.synchronize();seconds=time.perf_counter()-begin
            trace,_,_,actions=_validated_trace_output(prediction,noise[None],candidate_count=1,action_horizon=10,action_dim=7)
            result=dict(trace=trace[0],actions=actions[0],seconds=seconds)
            if capture:
                score=compute_accel_scores(trace,prefixes=[3])[3]
                if score['degenerate'].any():
                    raise ValueError('Degenerate live score')
                result['score']=float(score['score'][0])
                result.update({k:v.cpu().numpy().reshape(-1) for k,v in tap.features().items() if k in ('external','wrist')})
            return result

        def search(request,initial,method,stage,directory):
            directory.mkdir()
            cached=load_observations(inputs[initial])

            class Source:
                size=97;feature_dim=4608;score_samples=2
                def __init__(self):
                    self.prefetched=None;self.wrist=None
                def acquire(self,candidate):
                    response=request(dict(kind='render',candidate=candidate))
                    example=response['example'];verify_example(example,cached,candidate)
                    values=[infer(example,scoring_noises[initial,m],stage,
                                  dict(initial=initial,method=method,candidate=candidate,member=m),True) for m in (0,1)]
                    scores=np.asarray([v['score'] for v in values])
                    if not np.array_equal(scores,native_scores[initial,candidate]):
                        raise ValueError('Live batch1 scores differ from admitted native table')
                    for value in values:
                        if (not np.array_equal(value['external'],native_visual[initial,candidate])
                                or not np.array_equal(value['wrist'],native_wrist[initial,candidate])):
                            raise ValueError('Live native feature differs from reference')
                    self.wrist=values[0]['wrist']
                    np.savez_compressed(directory/f'query-{candidate:03d}.npz',external_image=example['image'][0],
                        wrist_image=example['image'][1],scores=scores,trace=np.stack([v['trace'] for v in values]),
                        actions=np.stack([v['actions'] for v in values]),seconds=[v['seconds'] for v in values])
                    return QueryResult(values[0]['external'],float(scores.mean()),float(scores.std()),2)
                def query(self,candidate):
                    if self.prefetched is not None:
                        if candidate!=65:
                            raise ValueError('Initial prefetch identity changed')
                        value,self.prefetched=self.prefetched,None
                        return value
                    return self.acquire(candidate)

            source=Source();source.prefetched=source.acquire(65)
            core=SearchEnv(poses[initial],source,source.wrist,scale,SearchConfig(query_budget=8,**actor_data['search']))
            core.reset(start=65)
            while not core.done:
                if method=='ppo':
                    observation=attention_observation(core,core.observation())
                    tensor={k:torch.as_tensor(v[None],dtype=torch.float32) for k,v in observation.items()}
                    with torch.inference_mode():
                        action=int(actor(tensor).argmax(-1).item())
                else:
                    action=baseline_action(core,method)
                core.step(action)
            info=dict(core.info(),initial=initial,method=method,stage=stage,actor_seed=0,
                      input_exact=True,score_exact=True,feature_exact=True,camera_motion='external_pose_teleport')
            write(directory/'search.json',info);queries.append(info)
            return info

        for initial in range(0,32,4):
            directory=output/f'live-{initial:02d}';directory.mkdir()
            with simulator(config_path,config,initial,directory) as request:
                search(request,initial,'ppo','live_search',directory/'search')
            write(output/'progress.json',dict(counts=counts,completed_live=len(queries),completed_tasks=len(tasks)))
        write(output/'live-admission.json',dict(status='PASS',episodes=8,counts=counts.copy(),all_queries_exact=True))
        for initial in (0,4):
            directory=output/f'tasks-{initial:02d}';directory.mkdir()
            with simulator(config_path,config,initial,directory) as request:
                for method in ('stay','local_accel','ppo'):
                    request(dict(kind='reset'))
                    info=search(request,initial,method,'task_search',directory/method)
                    selected=info['selected']
                    for repeat in range(4):
                        request(dict(kind='reset'))
                        example=request(dict(kind='render',candidate=selected))['example']
                        verify_example(example,load_observations(inputs[initial]),selected)
                        start_calibration={k:np.asarray(example[k]).copy() for k in ('camera_intrinsics','camera_to_world_opencv')}
                        steps=0;replan=0;success=False;frames=[];executed=[];noise_ids=[]
                        while steps<300 and not success:
                            noise=noise_bank.get(inputs[initial]['state']['pair_key'],repeat,replan)
                            response=infer(example,noise['noise'],'execution',
                                dict(initial=initial,method=method,repeat=repeat,replan=replan),False)
                            chunk=np.clip(response['actions'][:min(5,300-steps)],-1,1)
                            frames.append(np.concatenate(example['image'],axis=1))
                            response=request(dict(kind='step',actions=chunk))
                            example=response['example'];success=response['success'];count=response['executed']
                            executed.extend(chunk[:count]);steps+=count;replan+=1
                            noise_ids.append(noise['noise_sha256'])
                            if any(not np.array_equal(example[k],v) for k,v in start_calibration.items()):
                                raise ValueError('External camera moved during full task')
                        task=dict(initial=initial,method=method,repeat=repeat,selected=selected,success=success,
                            steps=steps,replans=replan,queries=info['queries'],scoring_members=info['scoring_members'],
                            historical_same_seed_success=bool(historical_success[initial,selected,repeat]),
                            execution_noise_sha256=noise_ids)
                        np.savez_compressed(directory/method/f'task-{repeat}.npz',actions=np.asarray(executed),frames=np.asarray(frames))
                        write(directory/method/f'task-{repeat}.json',task);tasks.append(task)
                        write(output/'progress.json',dict(counts=counts,completed_live=8,completed_tasks=len(tasks)))
                        print(f"task initial={initial} method={method} repeat={repeat} success={success} steps={steps}",flush=True)
        if len(tasks)!=24:
            raise ValueError('Full-task coverage incomplete')
        write(output/'receipt.json',dict(status='COMPLETE',counts=counts,searches=queries,tasks=tasks,
            elapsed_seconds=time.monotonic()-started,actor_seed=0,
            scope='Eight static external-camera search admissions and two development cases x three methods x four paired execution seeds; no head hardware/generalization claim'))
    except BaseException as error:
        write(output/'failure.json',dict(status='FAILED_STOPPED',error=repr(error),counts=counts,
            completed_tasks=len(tasks),completed_searches=len(queries)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    run(args.config,args.output,execute=args.execute)
