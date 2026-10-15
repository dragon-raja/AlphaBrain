"""Read-only-in-experiment first-frame diagnostic; never relax equality gates."""
import argparse
from pathlib import Path
import tempfile
import numpy as np
from AlphaBrain.research.dsol.data.artifacts import read, write


def diagnose(config_file, initial, output, count=16, restore=False):
    cfg=read(config_file); record=read(cfg['inputs'])['records'][initial]; spec=record['state']
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    from scripts.dsol_paper1.runtime.audit_libero_hdf5_restore import _configure_runtime
    from scripts.dsol_paper1.runtime.evaluate_dsol_libero_hdf5_views import configure_imports, masked_policy_observation
    from scripts.dsol_paper1.runtime.standard_initialization_v1 import load_initial
    from .initialization import initialize, capture_model_reference
    from scripts.dsol_paper1.runtime.scan_libero_hdf5_views import _restore_reference
    from scripts.dsol_paper1.runtime.render_bridge_common_v1 import render_protocol
    configure_imports()
    with np.load(record['observation_cache']['path'],allow_pickle=False) as z: cache={k:z[k].copy() for k in z.files}
    candidates=read(cfg['release'])['protocol']['state_blocks'][initial]['candidates']
    old=read(spec['initial_asset_record']); rows=[];envs=[];refs=[];physical=load_initial(spec); models={}; model_refs=[]
    with tempfile.TemporaryDirectory(prefix='dsol-first-frame-') as temporary:
        _configure_runtime(Path(cfg['runtime']),Path(cfg['dataset']),Path(temporary))
        from libero.libero.envs import OffScreenRenderEnv
        from libero_camera_pose import capture_camera_reference,install_camera_pose
        from evaluate_pi05_libero_plus_views import agentview_camera_calibration
        def compare(env,j,phase,receipt):
            env.env._update_observables(force=True)
            example,_,_=masked_policy_observation(env.env._get_observations(),prompt=spec['prompt'],resize_size=224,
                eval_seed=spec['environment_seed'],camera_calibration=agentview_camera_calibration(env),sensor_control='both')
            fields={}
            for saved,key in [('external_images','observation/image'),('wrist_images','observation/wrist_image'),
                ('robot_states','observation/state'),('camera_intrinsics','camera_intrinsics'),('camera_to_world_opencv','camera_to_world_opencv')]:
                a=np.asarray(example[key],np.float32) if saved=='robot_states' else np.asarray(example[key]); b=cache[saved][j]
                delta=np.abs(a.astype(np.float64)-b.astype(np.float64))
                fields[saved]=dict(exact=bool(np.array_equal(a,b)),max=float(delta.max()),mae=float(delta.mean()),different=int(np.count_nonzero(delta)))
            attributes=('body_pos','body_quat','geom_pos','geom_quat','geom_rgba','site_pos','site_quat','light_pos','mat_rgba','cam_pos','cam_quat')
            model_arrays={key:np.asarray(getattr(env.env.sim.model,key)).copy() for key in attributes}
            if phase=='first':models[j]=model_arrays
            changes={key:dict(max=float(np.max(np.abs(value-models[j][key]))),
                indices=np.flatnonzero(np.any(value!=models[j][key],axis=1)).tolist())
                for key,value in model_arrays.items() if not np.array_equal(value,models[j][key])}
            for key,change in changes.items():
                if key.startswith('body_'):
                    change['body_names']=[env.env.sim.model.body_id2name(i) for i in change['indices']]
            rows.append(dict(candidate=j,phase=phase,fields=fields,initialization=receipt,model_changes=changes,
                physical_matches_old=receipt['settled_physics_sha256']==old['initialization_receipt']['settled_physics_sha256']))
            np.savez_compressed(output/f'{phase}-{j:02d}.npz',external=example['observation/image'],
                wrist=example['observation/wrist_image'],qpos=env.env.sim.data.qpos,qvel=env.env.sim.data.qvel)
            write(output/'report.json',dict(initial=initial,rows=rows))
        with render_protocol(0):
            try:
                for j in range(count):
                    env=OffScreenRenderEnv(bddl_file_name=spec['bddl_file'],camera_names=('agentview','robot0_eye_in_hand'),camera_heights=256,camera_widths=256,render_gpu_device_id=0)
                    envs.append(env);_,receipt=initialize(env,spec,physical)
                    model_refs.append(capture_model_reference(env))
                    ref=capture_camera_reference(env,camera_name='agentview',table_plane_z=cfg['table_plane_z']);refs.append(ref)
                    _restore_reference(env,ref)
                    if candidates[j]['pose'] is not None:install_camera_pose(env,ref,candidates[j]['pose'])
                    compare(env,j,'first',receipt)
                for repeat in range(2):
                    for j,env in enumerate(envs):
                        _,receipt=initialize(env,spec,physical,model_reference=model_refs[j] if restore else None);_restore_reference(env,refs[j])
                        if candidates[j]['pose'] is not None:install_camera_pose(env,refs[j],candidates[j]['pose'])
                        compare(env,j,f'reset{repeat+1}',receipt)
            finally:
                for env in envs:env.close()
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--initial',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--count',type=int,default=16);p.add_argument('--restore',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:p.error('Explicit --execute required')
    diagnose(a.config,a.initial,a.output,a.count,a.restore)
