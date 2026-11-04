"""Named native-MuJoCo IO for HeadServo; no implicit simulator construction.

advance is injected deliberately: the experiment must specify its fixed-scene
control protocol. Plain mj_step on an uncontrolled task is not declared safe.
"""

import hashlib
import numpy as np
from .head_layout import HeadLayout, HEAD_NAMES


class MujocoHeadIO:
    def __init__(self, api, model, data, *, naming_prefix, advance, guard):
        self.api, self.model, self.data, self.guard = api, model, data, guard
        self.advance_fn = advance
        joints = [
            api.mj_name2id(model, api.mjtObj.mjOBJ_JOINT, naming_prefix + n) for n in HEAD_NAMES
        ]
        if min(joints) < 0 or any(model.jnt_type[j] != api.mjtJoint.mjJNT_HINGE for j in joints):
            raise ValueError("Three named native head hinges required")
        self.qpos = np.array([model.jnt_qposadr[j] for j in joints])
        self.qvel = np.array([model.jnt_dofadr[j] for j in joints])
        self.actuators = []
        names = []
        for joint in joints:
            found = [
                i
                for i in range(model.nu)
                if model.actuator_trntype[i] == api.mjtTrn.mjTRN_JOINT
                and model.actuator_trnid[i, 0] == joint
            ]
            if len(found) != 1:
                raise ValueError("Unique native head actuator required")
            i = found[0]
            if (
                model.actuator_dyntype[i] != api.mjtDyn.mjDYN_NONE
                or model.actuator_biastype[i] != api.mjtBias.mjBIAS_NONE
            ):
                raise ValueError("Only direct native motors supported")
            self.actuators.append(i)
            names.append(api.mj_id2name(model, api.mjtObj.mjOBJ_ACTUATOR, i))
        self.layout = HeadLayout(
            tuple(map(tuple, model.jnt_range[joints])),
            tuple(map(tuple, model.jnt_axis[joints])),
            tuple(names),
            "motor",
        )
        self.non_head_qpos = np.setdiff1d(np.arange(model.nq), self.qpos)
        self.non_head_qvel = np.setdiff1d(np.arange(model.nv), self.qvel)

    def read_head_state(self):
        self.guard()
        return dict(
            positions=tuple(self.data.qpos[self.qpos]),
            velocities=tuple(self.data.qvel[self.qvel]),
            observed_at=float(self.data.time),
        )

    def non_head_state_identity(self):
        h = hashlib.sha256()
        for values in (
            self.data.qpos[self.non_head_qpos],
            self.data.qvel[self.non_head_qvel],
            self.data.mocap_pos,
            self.data.mocap_quat,
        ):
            h.update(np.ascontiguousarray(values).tobytes())
        return h.hexdigest()

    def write_head_efforts(self, named):
        self.guard()
        if set(named) != set(self.layout.actuator_names):
            raise ValueError("Exact named head motors required")
        values = np.array([named[n] for n in self.layout.actuator_names], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Finite motor controls required")
        for i, value in zip(self.actuators, values):
            if (
                self.model.actuator_ctrllimited[i]
                and not self.model.actuator_ctrlrange[i, 0]
                <= value
                <= self.model.actuator_ctrlrange[i, 1]
            ):
                raise ValueError("Motor control exceeds native bound")
        self.data.ctrl[self.actuators] = values

    def advance(self):
        self.guard()
        self.advance_fn()
        self.guard()

    def stop_head_motion(self):
        # An explicit zero-effort command, not a claim of a physical brake.
        self.data.ctrl[self.actuators] = 0.0
