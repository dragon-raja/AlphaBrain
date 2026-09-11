#!/usr/bin/env python3
"""Fresh-process episode / full-candidate render / metric scoring, bridge only."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
from shared_runtime_paths import shared_scripts
from render_bridge_common_v1 import ROOT,RUNTIME,REPO,read,sha,write_new,array_identity,render_protocol

sys.path.insert(0,str(shared_scripts(REPO)))

def configure(state,out):
    from audit_libero_hdf5_restore import _configure_runtime
    _configure_runtime(RUNTIME,Path(state['hdf5']).parent.parent,out/'libero-config')

def episode(args,release):
    import evaluate_dsol_libero_hdf5_views as evaluator
    evaluator.configure_imports()
    spec=release['specs'][args.index]; output=ROOT/'rollouts'/args.cell/f'{args.index:04d}.json'
    if output.exists(): raise FileExistsError(output)
    configure(spec,output.parent/f'config-{args.index:04d}')
    import libero.libero.envs as env_module
    from trace_view_repeatability_v1 import physical_arrays
    from explicit_flow_noise import ExplicitFlowNoiseBank
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    original=env_module.OffScreenRenderEnv; holder={}; calls=[]; steps=[]
    def factory(*a,**kw):
        env=original(*a,**kw); holder['env']=env; step0=env.step
        def step(action):
            before={k:array_identity(v) for k,v in physical_arrays(env).items()}
            result=step0(action)
            steps.append({'action':array_identity(action),'pre':before,
                'post':{k:array_identity(v) for k,v in physical_arrays(env).items()},'success':bool(result[2])})
            return result
        env.step=step; return env
    env_module.OffScreenRenderEnv=factory
    client=WebsocketClientPolicy(host='127.0.0.1',port=args.port)
    class Client:
        def infer(self,request):
            inputs={k:array_identity(v) for k,v in request.items() if isinstance(v,np.ndarray)}
            metadata={k:v for k,v in request.items() if not isinstance(v,np.ndarray)}
            response=client.infer(request)
            calls.append({'inputs':inputs,'metadata':metadata,'actions':array_identity(response['actions'])})
            return response
    env=None
    try:
        with render_protocol(args.offsamples) as contexts:
            row,env=evaluator.run_episode(spec,runtime=RUNTIME,config_root=output.parent/f'config-{args.index:04d}',
                client=Client(),replan_steps=5,wait_steps=0,resize_size=224,seed=20260818,save_video=False,
                video_dir=output.parent/'unused-video',render_gpu=args.gpu,
                noise_bank=ExplicitFlowNoiseBank(Path(release['noise_bank']['path']),verify_file=False),require_explicit_noise=True)
            if not contexts: raise ValueError('Render intervention did not run')
            row['bridge']={'release_sha256':sha(ROOT/'release.json'),'cell':args.cell,'index':args.index,
                'render_samples':contexts,'render_gpu':args.gpu,'calls':calls,'steps':steps,
                'formal_legacy_ledger':False,'actual_wrapper_noise':getattr(env,'noise',None)}
            if row['bridge']['actual_wrapper_noise']!=0: raise ValueError('Unexpected action-wrapper noise')
            asset_cell=args.cell.split('/',1)[1].rsplit('-',1)[0]
            state_index=next(i for i,s in enumerate(release['states']) if s['pair_key']==spec['pair_key'])
            asset=read(ROOT/'assets'/asset_cell/f'{state_index:02d}'/'render.json')
            visible=next(v for v in asset['visibility'] if v['candidate_id']==spec['selected_candidate_id'])
            row['legacy_candidate_features_not_used']=row.pop('candidate_features',{})
            row['candidate_features']={'visibility_score':visible['score'],
                'per_camera_scores':{k:v['score'] for k,v in visible['per_camera'].items()},
                'render_artifact_sha256':asset['artifact_sha256']}
            write_new(output,row)
    finally:
        if env is not None: env.close()
        env_module.OffScreenRenderEnv=original

def render(args,release):
    import h5py
    from audit_libero_hdf5_restore import _decode,_rewrite_model_paths
    from run_view_value_expectation_accel_ensemble import operational_records
    state=release['states'][args.index]; scan=read(state['static_assets']['visibility_scan'])
    output=ROOT/'assets'/args.cell/f'{args.index:02d}'
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True); configure(state,output)
    from libero.libero.envs import OffScreenRenderEnv
    from libero_camera_pose import capture_camera_reference,install_camera_pose
    from libero_constructed_view import inject_static_visual_occluder
    from libero_visibility import task_entity_visibility
    from scan_libero_hdf5_views import _restore_reference
    from evaluate_pi05_libero_plus_views import agentview_camera_calibration,prepare_policy_observation,physics_state_sha256
    with h5py.File(state['hdf5'],'r') as h:
        data=h['data']; demo=data[state['demo_name']]
        physical=np.asarray(demo['states'][state['source_state_index']])
        xml,rewrites=_rewrite_model_paths(_decode(demo.attrs['model_file']),RUNTIME)
        bddl=Path(_decode(data.attrs['bddl_file_name'])).name
        import json
        prompt=json.loads(_decode(data.attrs['problem_info']))['language_instruction']
    xml=inject_static_visual_occluder(xml,scan['scene_construction'])
    candidates=operational_records(scan); values={k:[] for k in ('external_images','wrist_images','robot_states','camera_intrinsics','camera_to_world_opencv')}
    visibility=[]; env=None
    try:
        with render_protocol(args.offsamples) as contexts:
            env=OffScreenRenderEnv(bddl_file_name=str(RUNTIME/'libero/libero/bddl_files'/state['suite']/bddl),
                camera_names=('agentview','robot0_eye_in_hand'),camera_heights=256,camera_widths=256,render_gpu_device_id=args.gpu)
            env.seed(state['environment_seed']); env.reset(); env.reset_from_xml_string(xml); env.set_init_state(physical)
            before,size=physics_state_sha256(env)
            if before!=state['static_assets']['physics_state_sha256']: raise ValueError('Restored asset physics differs from frozen source')
            reference=capture_camera_reference(env,camera_name='agentview',table_plane_z=read(release['catalog']['path'])['table_plane_z'])
            def install(record):
                _restore_reference(env,reference)
                if record['pose_id']!='canonical': install_camera_pose(env,reference,record['pose'])
            # All RGB first, preserving the historical render_state call order.
            for record in candidates:
                install(record); env.env._update_observables(force=True)
                request,_,_=prepare_policy_observation(env.env._get_observations(),prompt=prompt,resize_size=224,
                    eval_seed=state['environment_seed'],camera_calibration=agentview_camera_calibration(env))
                for dest,src,dtype in [('external_images','observation/image',np.uint8),('wrist_images','observation/wrist_image',np.uint8),
                    ('robot_states','observation/state',np.float32),('camera_intrinsics','camera_intrinsics',np.float64),
                    ('camera_to_world_opencv','camera_to_world_opencv',np.float64)]: values[dest].append(np.asarray(request[src],dtype=dtype))
            # Segmentation after RGB; never insert additional renders into RGB generation.
            for record in candidates:
                install(record)
                visibility.append({'candidate_id':record['pose_id'],**task_entity_visibility(env,
                    entity_names=list(env.env.obj_of_interest),camera_names=('agentview','robot0_eye_in_hand'),height=224,width=224)})
            if physics_state_sha256(env)!=(before,size): raise ValueError('Rendering/visibility changed physical state')
            artifact=output/'policy_inputs.npz'
            np.savez_compressed(artifact,candidate_ids=np.asarray([r['pose_id'] for r in candidates]),**{k:np.stack(v) for k,v in values.items()})
            write_new(output/'render.json',{'status':'PASS','pair_key':state['asset_source_pair_key'],'split':state['split'],
                'task_id':state['task_id'],'source_group':state['source_group'],'language':prompt,'candidate_count':97,
                'physics_state_sha256':before,'physics_state_size':size,'artifact':str(artifact),'artifact_sha256':sha(artifact),
                'render_samples':contexts,'scene_construction_sha256':scan['scene_construction']['sha256'],
                'release_sha256':sha(ROOT/'release.json'),'visibility':visibility,'asset_path_rewrites':rewrites})
    finally:
        if env is not None: env.close()

def score(args,release):
    import torch
    from AlphaBrain.model.framework.base_framework import BaseFramework
    from run_view_value_expectation_accel_ensemble import rank_state
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    checkpoint=release['models'][args.model]
    model=BaseFramework.from_pretrained(checkpoint['path'],strict_checkpoint=True).to(torch.bfloat16).to('cuda:0').eval()
    model.gripper_remap=False
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for index in range(args.index,len(release['states']),args.shards):
        state=dict(release['states'][index],pair_key=release['states'][index]['asset_source_pair_key'])
        for cell in release['asset_cells']:
            output=ROOT/'scores'/args.model/cell/f'{index:02d}'
            if output.exists(): raise FileExistsError(output)
            row=rank_state(state,model=model,render_dir=ROOT/'assets'/cell/f'{index:02d}',ensemble_size=8,
                root_seed=20260921,batch_size=16,output_dir=output)
            write_new(output/'bridge-receipt.json',{'release_sha256':sha(ROOT/'release.json'),
                'checkpoint_sha256':checkpoint['weights_sha256'],'ranking_sha256':sha(output/'ranking.json')})

def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('episode','render','score'))
    p.add_argument('--index',type=int,required=True); p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--offsamples',type=int,choices=(0,4),default=4); p.add_argument('--cell',default='')
    p.add_argument('--port',type=int,default=29700); p.add_argument('--model',choices=('canonical','broad'))
    p.add_argument('--shards',type=int,default=4); args=p.parse_args()
    release=read(ROOT/'release.json')
    {'episode':episode,'render':render,'score':score}[args.mode](args,release)

if __name__=='__main__': main()
