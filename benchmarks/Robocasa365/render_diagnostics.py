"""Explicit render repeatability diagnostic; never changes a scientific release."""

from pathlib import Path
import hashlib
import random
import time

import numpy as np

from AlphaBrain.research.dsol.data.artifacts import sha, write
from .camera_rig import CAMERA_NAMES
from .evaluation_render import pixel_check
from .snapshot_bundle import load_snapshot_bundle, reconstruct_and_restore, refresh_bundle_observation, env_numeric_fingerprint
from .snapshot import restore_snapshot


def probe_snapshot(snapshot, output, repeats=8, calibration=False, pipeline_ablation=False, sample_readback=False):
    import gymnasium as gym
    import robocasa  # noqa: F401
    import mujoco
    from OpenGL import GL

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bundle = load_snapshot_bundle(Path(snapshot))
    meta = bundle.manifest['metadata']
    random.seed(meta['scene_seed']); np.random.seed(meta['scene_seed'])
    started = time.monotonic()
    env = gym.make('robocasa/' + meta['task'], split=meta['split'], seed=meta['scene_seed'], randomize_cameras=False)
    result = dict(snapshot=str(snapshot), manifest_sha256=sha(Path(snapshot) / 'manifest.json'), stages={})
    arrays = {}
    try:
        env.reset()
        reconstruct_and_restore(env, bundle)
        sim = env.unwrapped.env.sim
        result['numeric_start'] = env_numeric_fingerprint(env)
        context = sim._render_context_offscreen
        context.gl_ctx.make_current()
        result['pending_gl_errors'] = []
        for _ in range(16):
            error = int(GL.glGetError())
            if error == GL.GL_NO_ERROR:
                break
            result['pending_gl_errors'].append(error)
        result['gl'] = {key: GL.glGetString(value).decode() for key, value in
                        [('vendor', GL.GL_VENDOR), ('renderer', GL.GL_RENDERER), ('version', GL.GL_VERSION)]}
        result['dither'] = bool(GL.glIsEnabled(GL.GL_DITHER))
        result['offsamples'] = int(sim.model.vis.quality.offsamples)
        result['flags'] = list(map(int, context.scn.flags))
        camera_ids = [sim.model.camera_name2id(n) for n in CAMERA_NAMES]
        poses = np.concatenate([sim.data.cam_xpos[camera_ids].flatten(), sim.data.cam_xmat[camera_ids].flatten()])

        def stage(name, mode='direct', finish=False):
            names = CAMERA_NAMES[-1:] if mode == 'fixed_scene' else CAMERA_NAMES
            values = {n: [] for n in names}
            scene_ids = []
            if mode == 'fixed_scene':
                sim.render(height=256, width=256, camera_name=CAMERA_NAMES[-1])
            for repeat in range(repeats):
                if mode == 'restore':
                    restore_snapshot(env, bundle.snapshot)
                if mode in ('wrapper', 'restore'):
                    obs, _ = refresh_bundle_observation(env, bundle)
                    images = [obs['video.' + n] for n in CAMERA_NAMES]
                else:
                    images = []
                    for n in names:
                        if finish:
                            while GL.glGetError() != GL.GL_NO_ERROR:
                                pass
                            GL.glFinish()
                        if mode == 'fixed_scene':
                            # The same immutable wrist scene is re-rendered: no
                            # mjv_updateScene, controller, observable or physics.
                            mujoco.mjr_render(mujoco.MjrRect(0, 0, 256, 256), context.scn, context.con)
                            image = context.read_pixels(256, 256)[::-1]
                        else:
                            image = sim.render(height=256, width=256, camera_name=n)[::-1]
                        images.append(np.ascontiguousarray(image))
                geom_values = np.concatenate([np.concatenate([g.pos, g.mat.flatten(), g.rgba])
                                               for g in context.scn.geoms[:context.scn.ngeom]])
                scene_ids.append(hashlib.sha256(geom_values.tobytes()).hexdigest())
                for n, image in zip(names, images):
                    values[n].append(image.copy())
            stats = {}
            for n, images in values.items():
                stats[n] = dict(historical=[pixel_check(a, bundle.canonical_images['video.' + n]) for a in images],
                                repeat=[pixel_check(a, images[0]) for a in images],
                                unique_images=len({hashlib.sha256(a.tobytes()).hexdigest() for a in images}))
                arrays[name + '_' + n] = np.stack(images)
            current = np.concatenate([sim.data.cam_xpos[camera_ids].flatten(), sim.data.cam_xmat[camera_ids].flatten()])
            pending_errors = []
            for _ in range(16):
                error = int(GL.glGetError())
                if error == GL.GL_NO_ERROR:
                    break
                pending_errors.append(error)
            result['stages'][name] = dict(cameras=stats, poses_exact=bool(np.array_equal(current, poses)),
                                         unique_geom_buffers=len(set(scene_ids)),
                                         framebuffer_samples=int(context.con.offSamples),
                                         dither=bool(GL.glIsEnabled(GL.GL_DITHER)),
                                         pending_gl_errors=pending_errors)
            write(output / 'progress.json', result)

        stage('wrapper', 'wrapper')
        stage('direct')
        stage('restore_each', 'restore')
        if sample_readback:
            from .multisample_diagnostic import MultisampleDiagnostic
            diagnostic = MultisampleDiagnostic(context)
            try:
                stage('texture_native_resolve')
                context.read_pixels = diagnostic.read_pixels
                stage('texture_explicit_resolve')
                result['sample_stack_unique_per_camera'] = {
                    name: len(set(diagnostic.sample_hashes[i::len(CAMERA_NAMES)]))
                    for i, name in enumerate(CAMERA_NAMES)}
            finally:
                diagnostic.close()
            restore_snapshot(env, bundle.snapshot)
            result['numeric_end'] = env_numeric_fingerprint(env)
            result['scope'] = 'Diagnostic MSAA resolve isolation; NOT a released evaluation renderer'
            np.savez_compressed(output / 'images.npz', **arrays)
            result['images_sha256'] = sha(output / 'images.npz')
            return result
        if pipeline_ablation:
            # mjr_render re-enables GL_MULTISAMPLE on every call. Disabling the
            # GL capability before rendering is therefore NOT a no-MSAA test.
            # Rebuild only this diagnostic context with explicit sample counts.
            original_samples = int(sim.model.vis.quality.offsamples)
            for samples in (0, 2, original_samples):
                sim.model.vis.quality.offsamples = samples
                context.con.free()
                context._set_mujoco_context_and_buffers()
                GL.glDisable(GL.GL_DITHER)
                stage(f'framebuffer_samples_{samples}')
            # Preserve the original framebuffer and evaluate whether forcing
            # per-sample shading removes shared-fragment interpolation drift.
            GL.glEnable(GL.GL_SAMPLE_SHADING)
            GL.glMinSampleShading(1.0)
            stage('native_samples_full_sample_shading')
            GL.glDisable(GL.GL_SAMPLE_SHADING)
            sim.model.vis.quality.offsamples = original_samples
            restore_snapshot(env, bundle.snapshot)
            result['numeric_end'] = env_numeric_fingerprint(env)
            result['scope'] = 'Diagnostic framebuffer ablation; NOT an approved evaluation renderer'
            np.savez_compressed(output / 'images.npz', **arrays)
            result['images_sha256'] = sha(output / 'images.npz')
            return result
        if calibration:
            restore_snapshot(env, bundle.snapshot)
            result['numeric_end'] = env_numeric_fingerprint(env)
            np.savez_compressed(output / 'images.npz', **arrays)
            result['images_sha256'] = sha(output / 'images.npz')
            return result
        stage('finish', finish=True)
        while GL.glGetError() != GL.GL_NO_ERROR:
            pass
        GL.glDisable(GL.GL_DITHER)
        stage('no_dither')
        while GL.glGetError() != GL.GL_NO_ERROR:
            pass
        GL.glEnable(GL.GL_DITHER)
        original_flags = np.array(context.scn.flags).copy()
        for label, flag in [('no_shadow', mujoco.mjtRndFlag.mjRND_SHADOW),
                            ('no_reflection', mujoco.mjtRndFlag.mjRND_REFLECTION)]:
            context.scn.flags[:] = original_flags
            context.scn.flags[flag] = 0
            stage(label)
        context.scn.flags[:] = original_flags
        stage('fixed_scene', mode='fixed_scene')
        if hasattr(mujoco.mjtRndFlag, 'mjRND_ADDITIVE'):
            context.scn.flags[mujoco.mjtRndFlag.mjRND_ADDITIVE] = 1
            stage('additive_blending')
        context.scn.flags[:] = original_flags
        while GL.glGetError() != GL.GL_NO_ERROR:
            pass
        GL.glDisable(GL.GL_MULTISAMPLE)
        stage('no_multisample_diagnostic_only')
        while GL.glGetError() != GL.GL_NO_ERROR:
            pass
        GL.glEnable(GL.GL_MULTISAMPLE)
        original_materials = np.asarray(sim.model.mat_rgba).copy()
        sim.model.geom_rgba[:, 3] = 1.
        sim.model.mat_rgba[:, 3] = 1.
        stage('opaque_diagnostic_only')
        sim.model.mat_rgba[:] = original_materials
        restore_snapshot(env, bundle.snapshot)
        result['numeric_end'] = env_numeric_fingerprint(env)
        np.savez_compressed(output / 'images.npz', **arrays)
        result['images_sha256'] = sha(output / 'images.npz')
    finally:
        env.close()
        result['seconds'] = time.monotonic() - started
        write(output / 'summary.json', result)
    return result
