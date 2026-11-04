"""Explicit same-model MuJoCo integration-state replay with task/RNG hooks.

MuJoCo is injected, never imported or constructed here. A full task adapter must
provide capture/restore hooks for its Python state, RNGs and native model caches.
Changing the robot topology requires the separate named-state migration and a
new reference snapshot; a flat old state must never be reused after adding DOFs.
"""

from copy import deepcopy
import hashlib
import json
import re
import numpy as np


EXTERNAL_FIELDS = {"task", "numpy_rng", "python_rng", "model_rng", "native_cache"}


class IntegrationReplay:
    def __init__(self, api, model, data, external, *, model_sha256, guard):
        if not isinstance(model_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", model_sha256):
            raise ValueError("Explicit compiled model source identity required")
        self.api, self.model, self.data, self.external = api, model, data, external
        self.guard = guard
        self.spec = api.mjtState.mjSTATE_INTEGRATION
        self.size = int(api.mj_stateSize(model, self.spec))
        self.identity = dict(
            model_sha256=model_sha256,
            mujoco_version=api.__version__,
            state_spec=int(self.spec),
            state_size=self.size,
            topology={
                k: int(getattr(model, k))
                for k in ("nq", "nv", "na", "nu", "nbody", "nmocap", "nuserdata")
            },
        )
        self.status = "READY"

    def _live(self):
        if self.status != "READY":
            raise RuntimeError("Explicit audit required before replay recovery")
        self.guard()

    def _state(self):
        state = np.empty(self.size, dtype=np.float64)
        self.api.mj_getState(self.model, self.data, state, self.spec)
        if not np.isfinite(state).all():
            raise ValueError("Nonfinite integration state")
        return state

    def capture(self):
        try:
            self._live()
            external = deepcopy(self.external.capture())
            if not isinstance(external, dict) or set(external) != EXTERNAL_FIELDS:
                raise ValueError("Complete explicit task/RNG/cache snapshot required")
            state = self._state()
            self._live()
            return dict(
                identity=deepcopy(self.identity), integration=state.copy(), external=external
            )
        except BaseException:
            self.status = "HOLD"
            raise

    def restore(self, snapshot):
        """Restores simulation state only; does not resume a queue or refund calls.

        forward refreshes derived kinematics/observations but performs no physics
        timestep. Reapply the integration state afterwards to retain warm-start
        and other saved integration buffers that forward may recompute.
        """
        try:
            self._live()
            if (
                set(snapshot) != {"identity", "integration", "external"}
                or snapshot["identity"] != self.identity
            ):
                raise ValueError("Snapshot belongs to a different compiled model/runtime")
            state = np.array(snapshot["integration"], dtype=np.float64, copy=True)
            if state.shape != (self.size,) or not np.isfinite(state).all():
                raise ValueError("Invalid integration vector")
            external = deepcopy(snapshot["external"])
            if not isinstance(external, dict) or set(external) != EXTERNAL_FIELDS:
                raise ValueError("Task/RNG/cache snapshot incomplete")
            self.guard()
            self.external.restore(external)
            self.api.mj_setState(self.model, self.data, state, self.spec)
            self.api.mj_forward(self.model, self.data)
            self.api.mj_setState(self.model, self.data, state, self.spec)
            if not np.array_equal(self._state(), state):
                raise ValueError("Integration state changed during restoration")
            self._live()
            return dict(
                status="RESTORED_STATE_ONLY",
                integration_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                identity_sha256=hashlib.sha256(
                    json.dumps(self.identity, sort_keys=True).encode()
                ).hexdigest(),
                dispatch_resumed=False,
                model_calls=0,
            )
        except BaseException:
            self.status = "HOLD"
            raise
