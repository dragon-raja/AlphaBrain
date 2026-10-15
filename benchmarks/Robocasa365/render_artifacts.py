"""Localize historical RGB discrepancies without admitting or editing images."""

from pathlib import Path
import random
import numpy as np

from AlphaBrain.research.dsol.data.artifacts import sha, write
from .camera_rig import CAMERA_NAMES
from .evaluation_render import pixel_check
from .snapshot import restore_snapshot
from .snapshot_bundle import load_snapshot_bundle, reconstruct_and_restore, refresh_bundle_observation, env_numeric_fingerprint


def inspect_artifact(snapshot, output):
    import gymnasium as gym
    import robocasa  # noqa: F401
    import mujoco
    from openpi_client import image_tools

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bundle = load_snapshot_bundle(Path(snapshot))
    meta = bundle.manifest['metadata']
    random.seed(meta['scene_seed']); np.random.seed(meta['scene_seed'])
    env = gym.make('robocasa/' + meta['task'], split=meta['split'], seed=meta['scene_seed'], randomize_cameras=False)
    result = dict(scope='Diagnostic only; no gate changes, no physics steps, no released candidates',
                  snapshot=str(snapshot), manifest_sha256=sha(Path(snapshot) / 'manifest.json'), cameras={})
    arrays, inputs = {}, {}
    try:
        env.reset()
        reconstruct_and_restore(env, bundle)
        result['numeric_start'] = env_numeric_fingerprint(env)
        obs, _ = refresh_bundle_observation(env, bundle)
        sim = env.unwrapped.env.sim
        context = sim._render_context_offscreen
        inputs['state'] = np.concatenate([obs['state.' + k] for k in (
            'end_effector_position_relative', 'end_effector_rotation_relative', 'base_position', 'base_rotation', 'gripper_qpos')])
        inputs['prompt'] = np.asarray(obs['annotation.human.task_description'])
        original_flags = np.array(context.scn.flags).copy()
        for name, request_key in zip(CAMERA_NAMES, ('image', 'right_image', 'wrist_image')):
            historical, actual = bundle.canonical_images['video.' + name], obs['video.' + name]
            repeat = [np.ascontiguousarray(sim.render(height=256, width=256, camera_name=name)[::-1]) for _ in range(8)]
            segmentation = np.ascontiguousarray(sim.render(height=256, width=256, camera_name=name, segmentation=True)[::-1])
            delta = np.abs(actual.astype(np.int16) - historical.astype(np.int16))
            mask = np.any(delta > 1, axis=-1)
            ids, counts = np.unique(segmentation[..., 1][mask & (segmentation[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM))], return_counts=True)
            geoms = []
            for geom, count in sorted(zip(ids, counts), key=lambda x: -x[1]):
                material = int(sim.model.geom_matid[geom])
                geoms.append(dict(geom_id=int(geom), name=sim.model.geom_id2name(int(geom)),
                    affected_pixels=int(count), material_id=material,
                    material_name=mujoco.mj_id2name(sim.model._model, mujoco.mjtObj.mjOBJ_MATERIAL, material)
                    if material >= 0 else None))
            where = np.argwhere(mask)
            variants = {}
            for label, flag in [('no_shadow', mujoco.mjtRndFlag.mjRND_SHADOW),
                                ('no_reflection', mujoco.mjtRndFlag.mjRND_REFLECTION)]:
                context.scn.flags[:] = original_flags
                context.scn.flags[flag] = 0
                variant = np.ascontiguousarray(sim.render(height=256, width=256, camera_name=name)[::-1])
                variants[label] = pixel_check(variant, historical, .005)
                arrays[name + '_' + label] = variant
            context.scn.flags[:] = original_flags
            restored = np.ascontiguousarray(sim.render(height=256, width=256, camera_name=name)[::-1])
            for label, image in [('historical', historical), ('current', actual)]:
                arrays[name + '_' + label] = image
                inputs[label + '_' + request_key] = image_tools.convert_to_uint8(image_tools.resize_with_pad(
                    np.ascontiguousarray(image), 224, 224))
            arrays[name + '_repeats'] = np.stack(repeat)
            arrays[name + '_segmentation'] = segmentation
            result['cameras'][name] = dict(historical_difference=pixel_check(actual, historical, .005),
                repeat_differences=[pixel_check(a, actual, .005) for a in repeat],
                after_flags_restored=pixel_check(restored, actual, .005),
                bbox_yx_minmax=[where.min(0).tolist(), where.max(0).tolist()] if len(where) else None,
                affected_geoms=geoms, diagnostic_variants=variants,
                model_input_224_difference=pixel_check(inputs['current_' + request_key], inputs['historical_' + request_key], .005))
        restore_snapshot(env, bundle.snapshot)
        result['numeric_end'] = env_numeric_fingerprint(env)
        if result['numeric_start'] != result['numeric_end']:
            raise ValueError('Diagnostic did not restore exact physical state')
        np.savez_compressed(output / 'render_images.npz', **arrays)
        np.savez_compressed(output / 'paired_inputs.npz', **inputs)
        result['files'] = {name: sha(output / name) for name in ('render_images.npz', 'paired_inputs.npz')}
        result['status'] = 'DIAGNOSTIC_COMPLETE_NOT_ADMITTED'
    except Exception as exc:
        result.update(status='ERROR', error=str(exc))
        raise
    finally:
        env.close()
        write(output / 'summary.json', result)
    return result
