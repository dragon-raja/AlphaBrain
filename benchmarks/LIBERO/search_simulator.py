"""Private Python3.8 renderer/action server for the bounded search validation.

Camera queries change the external camera pose in a fixed physical state. They
are simulator teleports, not a claim of collision-free head/robot movement.
"""

from __future__ import annotations

import argparse
import hashlib
from multiprocessing.connection import Client
from pathlib import Path
import tempfile

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import read


def physical_digest(env):
    state = np.asarray(env.env.sim.get_state().flatten(),dtype=np.float64)
    return hashlib.sha256(state.tobytes()).hexdigest()


def serve(config_path,initial,address):
    from scripts.dsol_paper1.runtime.audit_libero_hdf5_restore import _configure_runtime
    from scripts.dsol_paper1.runtime.evaluate_dsol_libero_hdf5_views import configure_imports,masked_policy_observation
    from scripts.dsol_paper1.runtime.standard_initialization_v1 import load_initial
    from scripts.dsol_paper1.runtime.render_bridge_common_v1 import render_protocol
    from scripts.dsol_paper1.runtime.scan_libero_hdf5_views import _restore_reference
    from .initialization import initialize,capture_model_reference
    configure_imports()
    config=read(config_path);record=read(config['inputs'])['records'][initial];spec=record['state']
    candidates=read(config['release'])['protocol']['state_blocks'][initial]['candidates']
    physical=load_initial(spec)
    with tempfile.TemporaryDirectory(prefix='dsol-search-sim-') as temporary:
        _configure_runtime(Path(config['runtime']),Path(config['dataset']),Path(temporary))
        from libero.libero.envs import OffScreenRenderEnv
        from libero_camera_pose import capture_camera_reference,install_camera_pose
        from evaluate_pi05_libero_plus_views import agentview_camera_calibration
        with Client(address,family='AF_UNIX') as connection,render_protocol(0) as contexts:
            env=OffScreenRenderEnv(bddl_file_name=spec['bddl_file'],
                camera_names=('agentview','robot0_eye_in_hand'),camera_heights=256,
                camera_widths=256,render_gpu_device_id=0)
            try:
                initialize(env,spec,physical)
                model_reference=capture_model_reference(env)
                camera_reference=capture_camera_reference(env,camera_name='agentview',table_plane_z=config['table_plane_z'])
                baseline=physical_digest(env)
                observation=None

                def current_example(refresh=True):
                    nonlocal observation
                    if refresh:
                        env.env._update_observables(force=True)
                        observation=env.env._get_observations()
                    calibration=agentview_camera_calibration(env)
                    item,_,_=masked_policy_observation(observation,prompt=spec['prompt'],resize_size=224,
                        eval_seed=spec['environment_seed'],camera_calibration=calibration,sensor_control='both')
                    return dict(image=[item['observation/image'],item['observation/wrist_image']],
                        lang=spec['prompt'],language=spec['prompt'],state=np.asarray(item['observation/state'],np.float32),
                        camera_intrinsics=item['camera_intrinsics'],camera_to_world_opencv=item['camera_to_world_opencv'])

                connection.send(dict(status='READY',physics=baseline))
                while True:
                    request=connection.recv();kind=request['kind']
                    if kind=='catalog':
                        # Public camera geometry only: no RGB, score or outcome
                        # is returned for unvisited candidates.
                        before=physical_digest(env);poses=[]
                        for candidate in candidates:
                            _restore_reference(env,camera_reference)
                            if candidate['pose'] is not None:
                                install_camera_pose(env,camera_reference,candidate['pose'])
                            poses.append(agentview_camera_calibration(env)['camera_to_world_opencv'])
                        _restore_reference(env,camera_reference)
                        if physical_digest(env)!=before:
                            raise ValueError('Camera catalog changed physical state')
                        connection.send(dict(poses=np.asarray(poses),physics=before));continue
                    if kind=='close':
                        connection.send(dict(ok=True));break
                    if kind=='reset':
                        initialize(env,spec,physical,model_reference=model_reference)
                        if physical_digest(env)!=baseline:
                            raise ValueError('Physical reset did not reproduce the admitted initial')
                        connection.send(dict(ok=True,physics=baseline));continue
                    if kind=='render':
                        candidate=int(request['candidate'])
                        if not 0<=candidate<97:
                            raise ValueError('Candidate outside frozen catalog')
                        before=physical_digest(env)
                        _restore_reference(env,camera_reference)
                        pose=candidates[candidate]['pose']
                        if pose is not None:
                            install_camera_pose(env,camera_reference,pose)
                        example=current_example()
                        if physical_digest(env)!=before:
                            raise ValueError('Camera query changed physical state')
                        connection.send(dict(example=example,physics=before));continue
                    if kind=='step':
                        actions=np.asarray(request['actions'],np.float32)
                        if actions.ndim!=2 or actions.shape[1]!=7 or not 1<=len(actions)<=5 or not np.isfinite(actions).all():
                            raise ValueError('Invalid execution action chunk')
                        success=False;executed=0
                        for action in actions:
                            observation,_,done,_=env.step(action)
                            executed+=1
                            if done:
                                success=True;break
                        # Preserve the native env.step observation timing during execution.
                        connection.send(dict(example=current_example(refresh=False),success=success,executed=executed,
                                             physics=physical_digest(env)));continue
                    raise ValueError('Unknown simulator command')
                if not contexts or set(contexts)!={0}:
                    raise ValueError('AA0 renderer changed')
            except BaseException as error:
                try:
                    connection.send(dict(error=repr(error)))
                except (EOFError,BrokenPipeError):
                    pass
                raise
            finally:
                env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--initial',required=True,type=int)
    parser.add_argument('--socket',required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute required')
    serve(args.config,args.initial,args.socket)
