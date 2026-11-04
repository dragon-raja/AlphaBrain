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


def serve_session(config_path,initial,connection):
    from benchmarks.LIBERO.simulation_inputs import _configure_runtime
    from benchmarks.LIBERO.simulation_inputs import configure_imports,masked_policy_observation
    from benchmarks.LIBERO.simulation_inputs import load_initial
    from benchmarks.LIBERO.simulation_inputs import render_protocol
    from benchmarks.LIBERO.simulation_inputs import _restore_reference
    from .initialization import initialize,capture_model_reference
    configure_imports()
    config=read(config_path);record=read(config['inputs'])['records'][initial];spec=record['state']
    from AlphaBrain.research.dsol.data.operation_contract import simulator_action_contract
    action_limit,action_dtype=simulator_action_contract(config)
    # Continuous pose queries do not load or snap to the historical catalog.
    candidates=[] if config.get('continuous_camera') is True else read(config['release'])['protocol']['state_blocks'][initial]['candidates']
    physical=load_initial(spec)
    with tempfile.TemporaryDirectory(prefix='dsol-search-sim-') as temporary:
        _configure_runtime(Path(config['runtime']),Path(config['dataset']),Path(temporary))
        from libero.libero.envs import OffScreenRenderEnv
        from AlphaBrain.common.camera_pose import capture_camera_reference,install_camera_pose
        from benchmarks.LIBERO.simulation_inputs import agentview_camera_calibration
        with render_protocol(0) as contexts:
            env=OffScreenRenderEnv(bddl_file_name=spec['bddl_file'],
                camera_names=('agentview','robot0_eye_in_hand'),camera_heights=256,
                camera_widths=256,render_gpu_device_id=0)
            try:
                initialize(env,spec,physical)
                model_reference=capture_model_reference(env)
                camera_reference=capture_camera_reference(env,camera_name='agentview',table_plane_z=config['table_plane_z'])
                baseline=physical_digest(env)
                observation=None
                continuous=None
                from .continuous_execution import ContinuousExecutionGate,FixedOperationCamera
                execution_gate=ContinuousExecutionGate(config)
                fixed_camera=FixedOperationCamera()
                if config.get('continuous_camera') is True:
                    from .continuous_camera import ContinuousCamera
                    continuous=ContinuousCamera(env.env.sim, camera_reference['camera_id'],
                        agentview_camera_calibration(env)['camera_to_world_opencv'])

                def current_example(refresh=True):
                    nonlocal observation
                    if refresh:
                        env.env._update_observables(force=True)
                        observation=env.env._get_observations()
                    calibration=(fixed_camera.read(lambda:agentview_camera_calibration(env),refresh=refresh)
                        if 'operation_contract' in config else agentview_camera_calibration(env))
                    item,_,_=masked_policy_observation(observation,prompt=spec['prompt'],resize_size=224,
                        eval_seed=spec['environment_seed'],camera_calibration=calibration,sensor_control='both')
                    example=dict(image=[item['observation/image'],item['observation/wrist_image']],
                        lang=spec['prompt'],language=spec['prompt'],state=np.asarray(item['observation/state'],np.float32),
                        camera_intrinsics=item['camera_intrinsics'],camera_to_world_opencv=item['camera_to_world_opencv'])
                    if 'raw_observation' in config:
                        from .simulation_inputs import RAW_OBSERVATION, raw_observation
                        if config['raw_observation'] != RAW_OBSERVATION:
                            raise ValueError('Unknown raw observation contract')
                        example['fastwam_raw']=raw_observation(observation)
                    return example

                connection.send(dict(status='READY',physics=baseline))
                while True:
                    request=connection.recv();kind=request['kind']
                    if continuous is not None:
                        execution_gate.allow(kind)
                        if kind=='begin_execution':
                            receipt=continuous.receipt()
                            example=current_example()
                            execution_gate.begin()
                            connection.send(dict(example=example,receipt=receipt));continue
                        if kind=='continuous_info':
                            connection.send(dict(reference=continuous.mount.reference.copy(),
                                receipt=continuous.receipt()));continue
                        if kind=='move_pose':
                            connection.send(dict(receipt=continuous.move(request['camera_to_world'])));continue
                        if kind=='query_pose':
                            example=current_example()
                            connection.send(dict(example=example,receipt=continuous.receipt()));continue
                        # Only explicitly unlocked manipulation reaches the shared
                        # native step handler below. Search remains motion-free.
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
                        actions=np.asarray(request['actions'],dtype=action_dtype)
                        if actions.ndim!=2 or actions.shape[1]!=7 or not 1<=len(actions)<=action_limit or not np.isfinite(actions).all():
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


def serve(config_path,initial,address):
    with Client(address,family='AF_UNIX') as connection:
        serve_session(config_path,initial,connection)


def serve_resident(config_path,address):
    # Keep imports/interpreter warm; each session still constructs and closes a
    # fresh environment and renderer with the original initialization protocol.
    with Client(address,family='AF_UNIX') as connection:
        sequence=0
        connection.send(dict(status='IDLE',sequence=sequence))
        while True:
            try:message=connection.recv()
            except EOFError:return
            if message==dict(kind='shutdown',sequence=sequence):return
            if (message.get('kind')!='session' or message.get('sequence')!=sequence+1
                    or type(message.get('initial')) is not int or message['initial']<0):
                raise RuntimeError('Invalid resident simulator session')
            sequence+=1
            serve_session(config_path,message['initial'],connection)
            connection.send(dict(status='IDLE',sequence=sequence))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--initial',type=int)
    parser.add_argument('--resident',action='store_true')
    parser.add_argument('--socket',required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute required')
    if args.resident:
        if args.initial is not None:parser.error('Resident sessions bind initial per request')
        serve_resident(args.config,args.socket)
    else:
        if args.initial is None:parser.error('--initial required')
        serve(args.config,args.initial,args.socket)
