"""Initial-state candidate RGB cache; no VLA inference and no simulator steps."""

from pathlib import Path
import random
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import sha, write
from .camera_rig import CAMERA_NAMES, ReferenceRig
from .snapshot_bundle import (load_snapshot_bundle, reconstruct_and_restore,
                              refresh_bundle_observation, env_numeric_fingerprint)
from .snapshot import restore_snapshot


def physical_fields(sim):
    """Fields that camera-only forward/render must not change (warm-start excepted)."""
    fields = {'sim_state': np.asarray(sim.get_state().flatten()).copy()}
    for owner, names in ((sim.data, ('ctrl', 'mocap_pos', 'mocap_quat')),
                         (sim.model, ('site_rgba', 'site_size', 'geom_rgba', 'body_pos', 'body_quat'))):
        for name in names:
            if hasattr(owner, name):
                fields[name] = np.asarray(getattr(owner, name)).copy()
    return fields


def pixel_check(actual, reference, mean_abs_limit=.002):
    if mean_abs_limit not in (.002, .005, .01):
        raise ValueError('A predeclared numerical gate is required')
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape or actual.dtype != np.uint8 or reference.dtype != np.uint8:
        raise ValueError('Expected shape-matched uint8 images')
    delta = np.abs(actual.astype(np.int16) - reference.astype(np.int16))
    result = dict(max_abs=int(delta.max()), mean_abs=float(delta.mean()),
                  fraction_gt1=float(np.mean(delta > 1)), fraction_gt2=float(np.mean(delta > 2)))
    result['pass'] = (result['max_abs'] <= 2 and result['mean_abs'] <= mean_abs_limit
                      and result['fraction_gt1'] <= .0001 and result['fraction_gt2'] == 0.)
    return result


def pixel_disposition(checks):
    """Quarantine tiny rounding differences; never promote them to a PASS."""
    if len(checks) != 3:
        raise ValueError('Three snapshot cameras required')
    if all(v['pass'] for v in checks.values()):
        return 'PASS'
    if all(v['max_abs'] <= 1 for v in checks.values()):
        return 'HOLD_CANONICAL_PIXELS'
    raise ValueError('Canonical snapshot pixel gate failed beyond one-level quarantine')


def render_initial(job, catalog, destination, plan_sha256, *, pixel_gate=None, reference_file=None):
    # Heavy registration is explicit and absent from imports / default tests.
    import gymnasium as gym
    import robocasa  # noqa: F401
    from PIL import Image

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = dict(initial_id=job['initial_id'], plan_sha256=plan_sha256, status='ERROR',
                  snapshot_manifest_sha256=job['snapshot_manifest_sha256'], files={}, views=[])
    mean_limit = .002 if pixel_gate is None else pixel_gate['mean_abs']
    result['pixel_gate'] = pixel_gate or dict(mean_abs=.002, max_abs=2, fraction_gt1=.0001, fraction_gt2=0.)
    env = None
    try:
        root = Path(job['path'])
        if sha(root / 'manifest.json') != job['snapshot_manifest_sha256']:
            raise ValueError('Snapshot changed after plan freeze')
        bundle = load_snapshot_bundle(root)
        for key in ('task', 'split', 'scene_seed', 'layout_id', 'style_id'):
            if bundle.manifest['metadata'][key] != job[key]:
                raise ValueError('Job/snapshot identity mismatch')
        random.seed(job['scene_seed'])
        np.random.seed(job['scene_seed'])
        env = gym.make('robocasa/' + job['task'], split=job['split'], seed=job['scene_seed'],
                       randomize_cameras=False)
        env.reset()
        reconstruct_and_restore(env, bundle)
        result['restored_numeric_sha256'] = env_numeric_fingerprint(env)
        obs, _ = refresh_bundle_observation(env, bundle)
        result['snapshot_pixels'] = {key: pixel_check(obs[key], value, mean_limit)
                                     for key, value in (bundle.canonical_images or {}).items()}
        result['legacy_snapshot_pixels'] = {key: pixel_check(obs[key], value)
                                            for key, value in (bundle.canonical_images or {}).items()}
        if reference_file is None:
            disposition = pixel_disposition(result['snapshot_pixels'])
        else:
            from AlphaBrain.research.dsol.data.render_reference import load_reference
            _, reference_images = load_reference(job, reference_file)
            result['current_reference_sha256'] = sha(reference_file)
            result['current_reference_pixels'] = {
                'video.' + key: pixel_check(obs['video.' + key], image, mean_limit)
                for key, image in reference_images.items()}
            # Historical comparison stays in snapshot_pixels, including failures.
            disposition = pixel_disposition(result['current_reference_pixels'])
        if disposition != 'PASS':
            with (destination / 'pixel_gate_diagnostic.npz').open('xb') as stream:
                np.savez_compressed(stream, **{key: obs[key] for key in bundle.canonical_images})
            diagnostic = destination / 'pixel_gate_diagnostic.npz'
            result['files'][diagnostic.name] = dict(bytes=diagnostic.stat().st_size, sha256=sha(diagnostic))
            again, _ = refresh_bundle_observation(env, bundle)
            result['repeat_pixels'] = {key: pixel_check(again[key], obs[key]) for key in bundle.canonical_images}
        sim = env.unwrapped.env.sim
        before_fields = physical_fields(sim)
        rig = ReferenceRig(sim)
        before = np.asarray(sim.get_state().flatten()).copy()
        untouched = [i for i in range(len(rig.local_positions)) if i not in rig.ids[:2]]
        files = result['files']

        def register(name):
            path = destination / name
            files[name] = dict(bytes=path.stat().st_size, sha256=sha(path))

        def save_rgb(name, array):
            Image.fromarray(np.ascontiguousarray(array)).save(destination / name, compress_level=3)
            register(name)

        # Wrist is shared only at this same physical initial state, not across a rollout.
        save_rgb('wrist.png', obs['video.' + CAMERA_NAMES[2]])
        state = np.concatenate([obs['state.' + key] for key in
            ('end_effector_position_relative', 'end_effector_rotation_relative',
             'base_position', 'base_rotation', 'gripper_qpos')])
        result['prompt'] = obs['annotation.human.task_description']
        positions, quaternions = [], []
        for view in catalog['views']:
            if view['role'] == 'reference':
                rig.restore()
            else:
                rig.apply(*view['parameters'])
            mount = rig.mounting()
            rig.install(mount)  # Explicit unchanged-wrist/non-exterior validation.
            if not np.array_equal(sim.get_state().flatten(), before):
                raise ValueError('Camera change advanced physical state')
            if not np.array_equal(mount[0][untouched], rig.local_positions[untouched]):
                raise ValueError('Non-exterior camera changed')
            images = []
            for name, camera in zip(('left', 'right'), CAMERA_NAMES[:2]):
                image = np.ascontiguousarray(sim.render(height=256, width=256, camera_name=camera)[::-1])
                if view['role'] == 'reference' and not pixel_check(image, obs['video.' + camera], mean_limit)['pass']:
                    raise ValueError('Direct renderer and policy observation differ')
                filename = view['view_id'] + '_' + name + '.png'
                save_rgb(filename, image)
                images.append(filename)
            positions.append(mount[0]); quaternions.append(mount[1])
            result['views'].append(dict(view, images=images + ['wrist.png']))
        rig.restore()
        after_fields = physical_fields(sim)
        result['physical_fields_unchanged'] = {key: bool(np.array_equal(value, after_fields[key]))
                                                for key, value in before_fields.items()}
        if not all(result['physical_fields_unchanged'].values()):
            raise ValueError('Rendering changed physical state')
        with (destination / 'geometry.npz').open('xb') as stream:
            np.savez_compressed(stream, camera_positions=np.stack(positions),
                                camera_quaternions=np.stack(quaternions), state=state,
                                view_ids=np.asarray([v['view_id'] for v in catalog['views']]))
        register('geometry.npz')
        # MuJoCo forward() can update solver warm-start even without step().
        # Do not use this rendering environment as an un-restored rollout start.
        result['warmstart_changed_by_forward'] = not np.array_equal(sim.data.qacc_warmstart, bundle.snapshot.qacc_warmstart)
        restore_snapshot(env, bundle.snapshot)
        result['final_numeric_sha256'] = env_numeric_fingerprint(env)
        if result['final_numeric_sha256'] != result['restored_numeric_sha256']:
            raise ValueError('Final numeric state differs after restoring canonical rig')
        result['status'] = disposition
        result['eligible_for_formal_release'] = disposition == 'PASS'
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if env is not None:
            env.close()
        result['seconds'] = time.monotonic() - started
        write(destination / 'summary.json', result)
    return result
