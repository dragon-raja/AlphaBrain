"""Explicit static-model restoration around the immutable official reset path."""
import numpy as np
from scripts.dsol_paper1.runtime.standard_initialization_v1 import initialize as official_initialize


def capture_model_reference(env):
    return {key: np.asarray(getattr(env.env.sim.model,key)).copy() for key in ('body_pos','body_quat')}


class _StaticModelReset:
    def __init__(self,env,reference):
        self._wrapped=env; self._reference=reference

    def __getattr__(self,name):
        return getattr(self._wrapped,name)

    def reset(self):
        observation=self._wrapped.reset()
        if set(self._reference)!={'body_pos','body_quat'}:
            raise ValueError('Explicit complete static-body reference required')
        for key,values in self._reference.items():
            target=getattr(self._wrapped.env.sim.model,key)
            if target.shape!=values.shape or not np.isfinite(values).all():
                raise ValueError('Static-body reference shape/value mismatch')
            target[:]=values
        return observation


def initialize(env,spec,physical,*,model_reference=None):
    wrapped=env if model_reference is None else _StaticModelReset(env,model_reference)
    return official_initialize(wrapped,spec,physical)
