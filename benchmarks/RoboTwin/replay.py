"""Explicit native recorded-path replay and same-state camera admission pilot.

No simulator, Torch or GPU initialization at module import. This is engineering
validation, not a VLA rollout, benchmark score or active-camera controller.
"""
import io
import json
import pickle
from pathlib import Path


def load_numpy_pickle(path):
    import numpy as np
    allowed = {
        ('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype,
        ('numpy.core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
        ('numpy.core.multiarray', 'scalar'): np.core.multiarray.scalar,
        ('numpy.core.numeric', '_frombuffer'): np.core.numeric._frombuffer,
    }
    class ArraysOnly(pickle.Unpickler):
        def find_class(self, module, name):
            key = (module.replace('numpy._core', 'numpy.core'), name)
            if key not in allowed:
                raise ValueError(f'Unapproved trajectory global: {module}.{name}')
            return allowed[key]
    result = ArraysOnly(io.BytesIO(Path(path).read_bytes())).load()
    if not isinstance(result, dict) or set(result) != {'left_joint_path', 'right_joint_path'}:
        raise ValueError('Unexpected native trajectory structure')
    return result


def physics_snapshot(task):
    """Observable rigid-body and articulation state, excluding camera entities."""
    import numpy as np
    rows = []
    for entity in task.scene.entities:
        for component in entity.components:
            if not type(component).__name__.startswith('Physx'):
                continue
            row = {'entity': entity.name, 'component': type(component).__name__,
                   'pose': np.r_[entity.pose.p, entity.pose.q].tolist()}
            for key in ('linear_velocity', 'angular_velocity'):
                if hasattr(component, key):
                    row[key] = np.asarray(getattr(component, key)).tolist()
            if hasattr(component, 'articulation') and component.articulation is not None:
                articulation = component.articulation
                row['qpos'] = np.asarray(articulation.get_qpos()).tolist()
                row['qvel'] = np.asarray(articulation.get_qvel()).tolist()
            rows.append(row)
    if not rows:
        raise RuntimeError('No observable physics state captured')
    return rows


def camera_probe(task, output):
    import numpy as np
    import sapien.core as sapien
    from PIL import Image
    camera = task.cameras.static_camera_list[task.cameras.static_camera_name.index('head_camera')]
    pose = camera.entity.get_pose()
    original = sapien.Pose(pose.p.copy(), pose.q.copy())
    before = physics_snapshot(task)
    observations = []
    for label, delta in [('canonical', 0), ('repeat', 0), ('translate_x_3cm', .03), ('restored', 0)]:
        position = original.p.copy()
        position[0] += delta
        camera.entity.set_pose(sapien.Pose(position, original.q))
        obs = task.get_obs()
        if physics_snapshot(task) != before:
            raise RuntimeError('Camera rendering changed observable physical state')
        for name in ('head_camera', 'left_camera', 'right_camera'):
            Image.fromarray(obs['observation'][name]['rgb']).save(output / f'{label}-{name}.png')
        observations.append(obs)
    baseline = observations[0]['observation']
    return {'observable_physics_unchanged': True, 'camera_translation_m': .03,
            'rigid_head_motion_executed': False, 'simulation_camera_pose_edit_only': True,
            'views': [{
                'label': label,
                'head_cam2world': obs['observation']['head_camera']['cam2world_gl'].tolist(),
                'wrist_extrinsics_unchanged': all(np.array_equal(
                    baseline[name]['cam2world_gl'], obs['observation'][name]['cam2world_gl'])
                    for name in ('left_camera', 'right_camera')),
                'pixel_mae_from_first': {name: float(np.abs(
                    baseline[name]['rgb'].astype(float) - obs['observation'][name]['rgb'].astype(float)).mean())
                    for name in ('head_camera', 'left_camera', 'right_camera')},
            } for label, obs in zip(('canonical', 'repeat', 'translate_x_3cm', 'restored'), observations)]}


def run(config_path, task_name, episode, output, denoiser):
    import os
    import sys
    import time
    import traceback
    import types
    import importlib
    import numpy as np
    import h5py
    import yaml
    from AlphaBrain.research.dsol.data.resource_import import digest, save
    from AlphaBrain.research.dsol.data.resource_catalog import check_pointer

    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if os.environ.get('CUDA_VISIBLE_DEVICES') != config['physical_cuda_visible_devices']:
        raise ValueError('CUDA_VISIBLE_DEVICES differs from frozen pilot assignment')
    if task_name not in config['tasks'] or episode not in config['episodes']:
        raise ValueError('Task/episode outside engineering pilot')
    resource_identity = check_pointer(config['resource_pointer'])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw = Path(config['raw_root']) / task_name / 'extracted/aloha-agilex_clean_50'
    receipt = json.loads((raw.parent.parent / 'IMPORT_RECEIPT.json').read_text())
    hashes = {x['path']: x['sha256'] for x in receipt['files']}
    paths = [raw / '_traj_data' / f'episode{episode}.pkl', raw / 'data' / f'episode{episode}.hdf5', raw / 'seed.txt']
    for path in paths:
        if digest(path) != hashes[str(path.relative_to(raw.parent.parent))]:
            raise ValueError('Raw demonstration identity changed')
    trajectory = load_numpy_pickle(paths[0])
    seed = int(paths[2].read_text().split()[episode])
    runtime = Path(config['runtime_root'])
    os.chdir(runtime)
    sys.path.insert(0, str(runtime))
    sys.path.insert(0, str(runtime / 'envs/curobo/src'))
    os.environ['SIM_BENCH_DENOISER'] = denoiser
    state = {'status': 'RUNNING', 'task': task_name, 'episode': episode, 'seed': seed,
             'started_unix': time.time(), 'denoiser': denoiser, 'vla_loaded': False,
             'source_sha256': digest(__file__), 'config_sha256': digest(config_path),
             'raw_inputs': {str(p): digest(p) for p in paths},
             'resource_identity': resource_identity,
             'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']}
    save(output / 'status.json', state)
    try:
        import sapien.core as sapien
        # Renderer GPU selection is explicit; avoid the busy default physical GPU.
        renderer = sapien.SapienRenderer
        sapien.SapienRenderer = lambda *a, **kw: renderer(*a, **dict(kw, device=config['render_device']))
        import torch
        state['torch_gpu'] = torch.cuda.get_device_name(0)
        cls = getattr(importlib.import_module('envs.' + task_name), task_name)
        task = cls()
        args = yaml.safe_load((runtime / 'env_cfg/task_config/demo_clean.yml').read_text())
        embodiment = runtime / 'assets/embodiments/aloha-agilex'
        robot_config = yaml.safe_load((embodiment / 'config.yml').read_text())
        args.update(task_name=task_name, now_ep_num=episode, seed=seed, need_plan=False,
                    save_data=False, render_freq=0, eval_mode=False,
                    save_path=str(output / 'native-output'), dual_arm_embodied=True,
                    left_robot_file=str(embodiment), right_robot_file=str(embodiment),
                    left_embodiment_config=robot_config, right_embodiment_config=robot_config)
        state['phase'] = 'INITIALIZING_NATIVE_TASK'
        save(output / 'status.json', state)
        task.setup_demo(**args)
        state['phase'] = 'CAMERA_PROBE'
        save(output / 'status.json', state)
        probe = camera_probe(task, output)
        save(output / 'camera_probe.json', probe)
        frames, samples = [], []
        def capture(self):
            obs = self.get_obs()
            frames.append(np.asarray(obs['joint_action']['vector']).copy())
            if len(frames) == 1:
                from PIL import Image
                Image.fromarray(obs['observation']['head_camera']['rgb']).save(output / 'replay-first.png')
            samples.append(physics_snapshot(self))
            self.FRAME_IDX += 1
        task._take_picture = types.MethodType(capture, task)
        task.set_path_lst(dict(trajectory, need_plan=False))
        state['phase'] = 'PLAYING_RECORDED_PATHS'
        save(output / 'status.json', state)
        task.play_once()
        commands = np.stack(frames)
        np.save(output / 'recorded_joint_states.npy', commands)
        save(output / 'physics_frames.json', samples)
        with h5py.File(paths[1], 'r') as original:
            reference = original['joint_action/vector'][:]
            aligned = commands.shape == reference.shape
            comparison = {'original_shape': list(reference.shape), 'replayed_shape': list(commands.shape),
                          'same_frame_count': aligned,
                          'max_joint_abs_difference': float(np.max(np.abs(commands-reference))) if aligned else None}
        success = bool(task.check_success())
        consumed = {'left': [task.left_cnt, len(trajectory['left_joint_path'])],
                    'right': [task.right_cnt, len(trajectory['right_joint_path'])]}
        complete = all(used == total for used, total in consumed.values())
        same = aligned and comparison['max_joint_abs_difference'] <= config['joint_tolerance']
        state.update(status='COMPLETE', task_success=success, trajectory_consumption=consumed,
                     trajectory_comparison=comparison, physics_replay_verified=bool(success and complete and same),
                     rendering_admission='REQUIRES_POSTFLIGHT_LOG_AND_IMAGE_AUDIT',
                     camera_probe=probe, completed_unix=time.time())
        task.close_env()
    except BaseException as exc:
        state.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        save(output / 'status.json', state)
    return state
