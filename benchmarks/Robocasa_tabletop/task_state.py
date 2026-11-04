"""Explicit Python-state hooks complementing full MuJoCo integration replay.

Only the audited BASIC, non-interpolating joint-controller path is supported.
Renderer state is deliberately invalidated by the injected native-cache hook.
No snapshot resumes a dispatcher or restores/refunds a model-call ledger.
"""

from copy import deepcopy
import random
import numpy as np


def _plain(value):
    if value is None or isinstance(value, (str, bool, int, float, np.generic)):
        return True
    if isinstance(value, np.ndarray):
        # robosuite 1.5.1 FixedBaseRobot.control pushes self.torques (None)
        # into recent_torques after BASIC composite control. Preserve that exact
        # scalar sentinel; arbitrary Python objects remain forbidden.
        return not value.dtype.hasobject or (value.shape == () and value.item() is None)
    if isinstance(value, (tuple, list)):
        return all(_plain(v) for v in value)
    if isinstance(value, dict):
        return all(_plain(k) and _plain(v) for k, v in value.items())
    return False


def _attributes(obj, excluded=()):
    result = {}
    for name, value in vars(obj).items():
        if name in excluded:
            continue
        if not _plain(value):
            raise ValueError(f"Unaudited mutable state: {type(obj).__name__}.{name}")
        result[name] = deepcopy(value)
    return result


def _buffer_state(value):
    if isinstance(value, dict):
        return {key: _buffer_state(item) for key, item in value.items()}
    if type(value).__name__ not in ("DeltaBuffer", "RingBuffer"):
        raise ValueError("Unknown robot history buffer")
    return {"class": type(value).__name__, "state": _attributes(value)}


def _restore_buffer(value, snapshot):
    if isinstance(value, dict):
        if value.keys() != snapshot.keys():
            raise ValueError("Robot buffer keys changed")
        for key in value:
            _restore_buffer(value[key], snapshot[key])
    else:
        if (
            type(value).__name__ != snapshot["class"]
            or vars(value).keys() != snapshot["state"].keys()
        ):
            raise ValueError("Robot buffer type changed")
        for key, item in snapshot["state"].items():
            setattr(value, key, deepcopy(item))


class RobosuiteTaskState:
    """Same constructed task only; never crosses reset/recompile boundaries."""

    ENV_FIELDS = ("timestep", "cur_time", "done", "_obs_cache", "_ep_meta")
    OBS_FIELDS = (
        "_time_since_last_sample",
        "_current_delay",
        "_current_observed_value",
        "_sampled",
    )

    def __init__(self, env, native_cache, model_rng):
        if len(env.robots) != 1:
            raise ValueError("Audited single GR1 required")
        self.env, self.cache, self.model_rng = env, native_cache, model_rng
        self.cc = env.robots[0].composite_controller
        self.robot = env.robots[0]
        if type(self.cc).__name__ != "CompositeController":
            raise ValueError("Audited BASIC controller required")
        for part in self.cc.part_controllers.values():
            if getattr(part, "interpolator", None) is not None:
                raise ValueError("Interpolator state requires separate admission")
        self.observable_names = tuple(env._observables)
        self.part_names = tuple(self.cc.part_controllers)

    def capture(self):
        task = dict(
            env={name: deepcopy(getattr(self.env, name)) for name in self.ENV_FIELDS},
            observables={
                name: {key: deepcopy(getattr(obs, key)) for key in self.OBS_FIELDS}
                for name, obs in self.env._observables.items()
            },
            controllers={
                name: _attributes(part, ("sim",)) for name, part in self.cc.part_controllers.items()
            },
            applied_action=deepcopy(self.cc._applied_action_dict),
            buffers={
                name: _buffer_state(value)
                for name, value in vars(self.robot).items()
                if name.startswith("recent_")
            },
            env_rng=deepcopy(self.env.rng.bit_generator.state),
            observable_names=self.observable_names,
            part_names=self.part_names,
        )
        return dict(
            task=task,
            numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),
            model_rng=deepcopy(self.model_rng.capture()),
            native_cache=deepcopy(self.cache.capture()),
        )

    def restore(self, snapshot):
        task = snapshot["task"]
        if tuple(task["observable_names"]) != tuple(self.env._observables) or tuple(
            task["part_names"]
        ) != tuple(self.cc.part_controllers):
            raise ValueError("Task/controller schema changed across replay")
        if set(task["env"]) != set(self.ENV_FIELDS):
            raise ValueError("Incomplete environment state")
        if set(task["controllers"]) != set(self.part_names):
            raise ValueError("Incomplete controller state")
        if set(task["observables"]) != set(self.observable_names):
            raise ValueError("Incomplete observable state")
        for name, values in task["buffers"].items():
            _restore_buffer(getattr(self.robot, name), values)
        for name, value in task["env"].items():
            if name not in self.ENV_FIELDS:
                raise ValueError("Unexpected environment state")
            setattr(self.env, name, deepcopy(value))
        for name, values in task["controllers"].items():
            part = self.cc.part_controllers[name]
            if set(values) != set(vars(part)) - {"sim"}:
                raise ValueError("Controller state keys changed")
            for key, value in values.items():
                setattr(part, key, deepcopy(value))
        for name, values in task["observables"].items():
            if set(values) != set(self.OBS_FIELDS):
                raise ValueError("Incomplete observable state")
            for key, value in values.items():
                setattr(self.env._observables[name], key, deepcopy(value))
        self.cc._applied_action_dict = deepcopy(task["applied_action"])
        self.env.rng.bit_generator.state = deepcopy(task["env_rng"])
        np.random.set_state(snapshot["numpy_rng"])
        random.setstate(snapshot["python_rng"])
        self.model_rng.restore(deepcopy(snapshot["model_rng"]))
        self.cache.restore(deepcopy(snapshot["native_cache"]))


class ObservationEpoch:
    """Invalidate observations on movement/replay; never roll freshness backward."""

    def __init__(self):
        self.epoch = 0
        self.values = {}

    def invalidate(self):
        self.epoch += 1
        self.values.clear()
        return self.epoch

    def capture(self):
        return {"epoch": self.epoch}

    def restore(self, snapshot):
        if (
            set(snapshot) != {"epoch"}
            or type(snapshot["epoch"]) is not int
            or snapshot["epoch"] < 0
        ):
            raise ValueError("Invalid cache epoch")
        self.epoch = max(self.epoch, snapshot["epoch"]) + 1
        self.values.clear()
