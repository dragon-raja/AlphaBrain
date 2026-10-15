"""Explicit reference capture and independent restoration; no scene changes."""

import os
import platform
from pathlib import Path
import random
import numpy as np

from AlphaBrain.research.dsol.data.artifacts import sha, write
from AlphaBrain.research.dsol.data.render_reference import CAMERAS, LIMITS, save_reference, load_reference
from .evaluation_render import pixel_check
from .snapshot import restore_snapshot
from .snapshot_bundle import load_snapshot_bundle, reconstruct_and_restore, refresh_bundle_observation, env_numeric_fingerprint


def inspect_reference(job, output, *, reference_file=None):
    import gymnasium as gym
    import robocasa
    import mujoco
    from OpenGL import GL

    bundle = load_snapshot_bundle(Path(job['path']))
    if sha(Path(job['path']) / 'manifest.json') != job['snapshot_manifest_sha256']:
        raise ValueError('Snapshot identity changed')
    images = None
    if reference_file is not None:
        _, images = load_reference(job, reference_file)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    random.seed(job['scene_seed']); np.random.seed(job['scene_seed'])
    env = gym.make('robocasa/' + job['task'], split=job['split'], seed=job['scene_seed'], randomize_cameras=False)
    result = dict(status='ERROR', initial_id=job['initial_id'], pixel_gate=LIMITS,
                  checks=[], historical_checks=[], repeats=4, pid=os.getpid(), host=platform.node())
    try:
        env.reset()
        reconstruct_and_restore(env, bundle)
        result['numeric_start'] = env_numeric_fingerprint(env)
        obs, _ = refresh_bundle_observation(env, bundle)
        sim = env.unwrapped.env.sim
        sim._render_context_offscreen.gl_ctx.make_current()
        # Record pending GL errors before querying metadata: PyOpenGL otherwise
        # attributes an earlier renderer error to a successful glGetString call.
        # This does not alter rendering flags, pixels, or admission checks.
        result['pending_gl_errors_before_metadata'] = []
        for _ in range(16):
            error = int(GL.glGetError())
            if error == GL.GL_NO_ERROR:
                break
            result['pending_gl_errors_before_metadata'].append(error)
        else:
            raise RuntimeError('Persistent GL error before metadata query')
        result['runtime'] = dict(mujoco=mujoco.__version__, robocasa=robocasa.__version__,
            pending_gl_errors_before_metadata=result['pending_gl_errors_before_metadata'],
            renderer=GL.glGetString(GL.GL_RENDERER).decode(), driver=GL.glGetString(GL.GL_VERSION).decode())
        if reference_file is None:
            restore_snapshot(env, bundle.snapshot)
            if env_numeric_fingerprint(env) != result['numeric_start']:
                raise ValueError('Capture changed restored state')
            return save_reference(job, {k: obs['video.' + k] for k in CAMERAS}, output,
                provenance=dict(kind='first_capture_not_selected_by_pixel_agreement', pid=os.getpid(),
                                host=platform.node()), runtime=result['runtime'])
        result['reference_sha256'] = sha(reference_file)
        for repeat in range(4):
            restore_snapshot(env, bundle.snapshot)
            obs, _ = refresh_bundle_observation(env, bundle)
            result['checks'].append({k: pixel_check(obs['video.' + k], images[k], .005) for k in CAMERAS})
            result['historical_checks'].append({k: pixel_check(obs['video.' + k], bundle.canonical_images['video.' + k], .005)
                                                 for k in CAMERAS})
        restore_snapshot(env, bundle.snapshot)
        result['numeric_end'] = env_numeric_fingerprint(env)
        if result['numeric_start'] != result['numeric_end']:
            raise ValueError('Independent validation changed state')
        result['status'] = 'PASS' if all(c['pass'] for row in result['checks'] for c in row.values()) else 'HOLD'
    except Exception as exc:
        result['error'] = str(exc)
        raise
    finally:
        env.close()
        if reference_file is not None:
            write(output, result)
    return result
