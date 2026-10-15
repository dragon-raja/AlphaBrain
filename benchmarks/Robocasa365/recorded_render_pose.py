"""Observation-only replay of recorded world poses; never a dynamics restore.

Robosuite telemetry may pair integrated qpos with pre-integration rendering
caches. Keep the raw state intact and reconstruct render caches independently.
No forward or integration may run after installation and before rendering.
"""

import numpy as np


def install_recorded_render_pose(sim, positions, quaternions, *, native=None):
    """Derive geometry/sites/cameras from recorded body poses using MuJoCo.

    Historical sites are deliberately not inputs: callers must independently
    compare the reconstructed sites and rendered pixels against their records.
    Tracking cameras and deformable geometry need additional replay semantics.
    """
    model, data = sim.model._model, sim.data._data
    if np.any(model.cam_mode != 0) or np.any(model.light_mode != 0):
        raise ValueError('Recorded render replay requires fixed cameras and lights')
    if getattr(model, 'nflex', 0) or getattr(model, 'nskin', 0):
        raise ValueError('Deformable render state was not recorded')
    for name, value in (('xpos', positions), ('xquat', quaternions)):
        value = np.asarray(value)
        if value.shape != getattr(data, name).shape or not np.isfinite(value).all():
            raise ValueError('Invalid recorded render pose: '+name)
    if native is None:
        import mujoco as native
    data.xpos[:] = positions
    data.xquat[:] = quaternions
    for body in range(model.nbody):
        native.mju_quat2Mat(data.xmat[body], data.xquat[body])
        native.mj_local2Global(data, data.xipos[body], data.ximat[body],
                              model.body_ipos[body], model.body_iquat[body], body, 0)
    for kind in ('geom', 'site'):
        for index in range(getattr(model, 'n'+kind)):
            native.mj_local2Global(
                data, getattr(data, kind+'_xpos')[index], getattr(data, kind+'_xmat')[index],
                getattr(model, kind+'_pos')[index], getattr(model, kind+'_quat')[index],
                int(getattr(model, kind+'_bodyid')[index]),
                int(getattr(model, kind+'_sameframe')[index]))
    native.mj_camlight(model, data)
