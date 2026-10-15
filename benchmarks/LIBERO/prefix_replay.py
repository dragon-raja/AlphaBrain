"""Bounded original-noise prefix replay using the native batch16 policy bridge."""
import argparse
from multiprocessing.connection import Client
from pathlib import Path
import tempfile
import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read,sha,write
from AlphaBrain.research.dsol.data.prefix_telemetry import frame,names,exact,save_branch
from AlphaBrain.research.dsol.metrics.accel import compute_accel_scores
from .window_probe import later_noise,run


def simulator(config_file,initial,model,output,socket_path):
    config=read(config_file);record=read(config['inputs'])['records'][initial];spec=record['state']
    case=next(c for c in config['replay_cases'] if c['initial']==initial)
    original=Path(case['original'])
    if sha(original/'probes.npz')!=case['original_sha256']:raise ValueError('Original trace changed')
    with np.load(original/'probes.npz') as z:old={k:z[k].copy() for k in ('views','scores','velocities','actions')}
    with np.load(record['observation_cache']['path']) as z:cached={k:z[k].copy() for k in z.files}
    with np.load(config['anchors']) as z:first_noise=z['first_score_noise'][initial].copy()
    candidates=read(config['release'])['protocol']['state_blocks'][initial]['candidates']
    from scripts.dsol_paper1.runtime.audit_libero_hdf5_restore import _configure_runtime
    from scripts.dsol_paper1.runtime.evaluate_dsol_libero_hdf5_views import configure_imports,masked_policy_observation
    from scripts.dsol_paper1.runtime.standard_initialization_v1 import load_initial,initialize
    from scripts.dsol_paper1.runtime.render_bridge_common_v1 import render_protocol
    from scripts.dsol_paper1.runtime.scan_libero_hdf5_views import _restore_reference
    configure_imports();connection=Client(socket_path,family='AF_UNIX');out=Path(output)
    complete=[];physical=load_initial(spec)
    with tempfile.TemporaryDirectory(prefix='dsol-prefix-sim-') as temp:
        _configure_runtime(Path(config['runtime']),Path(config['dataset']),Path(temp))
        from libero.libero.envs import OffScreenRenderEnv
        from libero_camera_pose import capture_camera_reference,install_camera_pose
        from evaluate_pi05_libero_plus_views import agentview_camera_calibration
        with render_protocol(0):
            for view in case['views']:
                j=old['views'].tolist().index(view)
                env=OffScreenRenderEnv(bddl_file_name=spec['bddl_file'],camera_names=('agentview','robot0_eye_in_hand'),camera_heights=256,camera_widths=256,render_gpu_device_id=0)
                try:
                    initialize(env,spec,physical)
                    reference=capture_camera_reference(env,camera_name='agentview',table_plane_z=config['table_plane_z'])
                    write(out/(view+'-names.json'),names(env.env.sim))
                    for repeat in range(8):
                        arrays={};records=[];valid=False
                        identity=dict(model=model,initial=initial,view=view,repeat=repeat,original=str(original),stream='score',batch_size=16)
                        branch=out/view/f'{repeat:02d}'
                        try:
                            initialize(env,spec,physical);_restore_reference(env,reference)
                            if candidates[j]['pose'] is not None:install_camera_pose(env,reference,candidates[j]['pose'])
                            env.env._update_observables(force=True);obs=env.env._get_observations();calibration=agentview_camera_calibration(env)
                            for window in range(5):
                                step=window*5
                                arrays.update({f'frame/{step:03d}/'+k:v for k,v in frame(env.env.sim,obs).items()})
                                example,_,_=masked_policy_observation(obs,prompt=spec['prompt'],resize_size=224,eval_seed=spec['environment_seed'],camera_calibration=calibration,sensor_control='both')
                                example['observation/state']=np.asarray(example['observation/state'],np.float32)
                                checks={}
                                if window==0:
                                    for saved,key in [('external_images','observation/image'),('wrist_images','observation/wrist_image'),('robot_states','observation/state'),('camera_intrinsics','camera_intrinsics'),('camera_to_world_opencv','camera_to_world_opencv')]:
                                        checks[key]=exact(example[key],cached[saved][j])
                                    records.append(dict(kind='first_input',checks=checks))
                                    if not all(c['exact'] for c in checks.values()):raise ValueError('Original first input differs')
                                examples=[dict(image=[example['observation/image'],example['observation/wrist_image']],lang=spec['prompt'],language=spec['prompt'],state=example['observation/state'],camera_intrinsics=example['camera_intrinsics'],camera_to_world_opencv=example['camera_to_world_opencv'])]*16
                                noise=first_noise[repeat] if window==0 else later_noise(spec['pair_key'],repeat,window)
                                connection.send(dict(kind='infer',examples=examples,noise=np.repeat(noise[None],16,0)));response=connection.recv()
                                if response.get('error'):raise RuntimeError(response['error'])
                                velocity=response['velocity'][0];action=response['actions'][0]
                                metric=compute_accel_scores(velocity[None],prefixes=[3])[3]
                                score=float(metric['score'][0]);gate=dict(velocity=exact(velocity,old['velocities'][j,repeat,window]),action=exact(action,old['actions'][j,repeat,window]),score=exact(score,old['scores'][j,repeat,window]))
                                arrays[f'plan/{window}/velocity']=velocity;arrays[f'plan/{window}/actions']=action
                                records.append(dict(kind='plan',window=window,step=step,accel=score,gate=gate))
                                write(out/'progress.json',dict(initial=initial,view=view,repeat=repeat,window=window,gate=gate))
                                if not all(v['exact'] for v in gate.values()):raise ValueError('Original prefix differs; no tolerance relaxation')
                                if window<4:
                                    for k,command in enumerate(action[:5]):
                                        issued=np.clip(command,-1,1);obs,_,done,_=env.step(issued);n=step+k+1
                                        arrays[f'command/{n:03d}']=issued.copy()
                                        arrays.update({f'frame/{n:03d}/'+key:value for key,value in frame(env.env.sim,obs).items()})
                                        records.append(dict(kind='step',step=n,success=bool(done)))
                            valid=True;complete.append(dict(view=view,repeat=repeat))
                        finally:save_branch(branch,arrays,records,identity,valid)
                finally:env.close()
    np.savez_compressed(out/'probes.npz',views=np.array(case['views']),completed=np.array([v['repeat'] for v in complete]))
    write(out/'prefix-audit.json',dict(status='PASS',all_five_windows_exact=True,completed=complete,original_sha256=case['original_sha256']))
    connection.send(dict(kind='finish'));connection.recv();connection.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--initial',type=int,required=True);p.add_argument('--model',required=True,choices=['canonical']);p.add_argument('--output',required=True);p.add_argument('--socket');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:p.error('Explicit --execute required')
    if a.socket:simulator(a.config,a.initial,a.model,a.output,a.socket)
    else:run(a.config,a.initial,a.model,a.output,simulator_module=__name__ if __name__!='__main__' else 'benchmarks.LIBERO.prefix_replay',completion_metadata=dict(scope='original-prefix-replay',new_complete_rollouts=0,windows=[1,2,3,4,5],score_repeats=8))
