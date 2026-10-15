# Promoted from the audited 20260814 RoboCasa recovery source on 20260916.
# Historical sources remain immutable; runtime no longer imports recovery paths.
"""Complete reversible RoboCasa runtime snapshots for paired evaluation.

This is an isolated copy of the snapshot machinery previously validated by
``check_robocasa_roundtrip.py`` in the active-evidence work. It intentionally
captures controller, fixture, observable, wrapper, RNG, and MuJoCo warm-start
state in addition to qpos/qvel.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EnvSnapshot:
    sim_state: np.ndarray
    camera_positions: np.ndarray
    camera_quaternions: np.ndarray
    model_runtime: dict[str, np.ndarray]
    ctrl: np.ndarray | None
    qacc_warmstart: np.ndarray | None
    mocap_positions: np.ndarray | None
    mocap_quaternions: np.ndarray | None
    rng_state: dict[str, Any] | None
    numpy_random_state: tuple[Any, ...]
    wrapper_runtime: tuple[dict[str, Any], ...]
    env_runtime: dict[str, Any]
    fixture_runtime: dict[str, dict[str, Any]]
    controller_runtime: tuple[dict[str, dict[str, Any]], ...]
    robot_runtime: tuple[dict[str, Any], ...]
    gripper_actions: tuple[dict[str, np.ndarray], ...]
    observable_runtime: dict[str, dict[str, Any]]


def inner_env(env: Any) -> Any:
    unwrapped = getattr(env, "unwrapped", env)
    return getattr(unwrapped, "env", unwrapped)


def refresh_observation(env: Any) -> dict[str, Any]:
    """Render and remap a fresh Gym observation without advancing physics."""

    wrapper = getattr(env, "unwrapped", env)
    base = inner_env(env)
    raw_observation = (
        base.viewer._get_observations(force_update=True)
        if base.viewer_get_obs
        else base._get_observations(force_update=True)
    )
    if not hasattr(wrapper, "get_observation"):
        raise TypeError("RoboCasa Gym wrapper does not expose get_observation")
    return wrapper.get_observation(raw_observation)


def _copy_attr(obj: Any, name: str) -> np.ndarray | None:
    value = getattr(obj, name, None)
    return None if value is None else np.asarray(value).copy()


_UNSAFE_RUNTIME_VALUE = object()

_ENV_RUNTIME_FIELDS = {
    "_initial_yaw",
    "_obs_cache",
    "_robot_touched_food",
    "board_contact_positions",
    "board_contact_timer",
    "contact_timesteps",
    "cooldown_timer",
    "cooling_done",
    "cur_time",
    "dishes",
    "done",
    "fish_rinsed",
    "fish_water_timer",
    "food_washed",
    "heat_timer",
    "num_filled_timesteps",
    "off_too_early",
    "pan_rotated",
    "press_done",
    "press_timer",
    "prev_pan_pos",
    "rinsed_objects",
    "rinsing_time",
    "scale_wait_complete",
    "scale_wait_counter",
    "sink_contact_positions",
    "sink_contact_timer",
    "sponge_contact_height",
    "sponge_contact_time",
    "steps_elapsed",
    "success_contact_time",
    "success_time",
    "success_timer",
    "swept_sink",
    "table_contact_positions",
    "table_contact_timer",
    "task_timestep",
    "teapot_contact_time",
    "timestep",
    "toaster_on",
    "vegetable_was_in_sink",
    "vegetable_washed",
    "vegetables_washed",
    "washed_loc",
    "washed_time",
    "water_contact_timers",
    "water_running",
    "water_running_steps",
    "water_was_turned_on",
}

_FIXTURE_RUNTIME_FIELDS = {
    "_button_contact_prev_timestep",
    "_button_head_lock",
    "_cooldown",
    "_cooldown_time",
    "_doneness",
    "_door",
    "_door_target",
    "_function",
    "_last_lid_update",
    "_last_time_update",
    "_lid_on_blender",
    "_lid_speed",
    "_num_steps_on",
    "_rack",
    "_state",
    "_switch",
    "_target_lid_angle",
    "_temperature",
    "_time",
    "_timer",
    "_tray",
    "_turned_on",
}

_WRAPPER_RUNTIME_FIELDS = {
    "_elapsed_steps",
    "_geom_ids",
    "_has_reset",
    "_last_model_ptr",
    "_saved_alpha",
    "enabled",
}


def _copy_plain_runtime_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 12:
        return _UNSAFE_RUNTIME_VALUE
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return value
    if isinstance(value, np.generic):
        return value.copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        copied: dict[Any, Any] = {}
        for key, item in value.items():
            copied_key = _copy_plain_runtime_value(key, depth=depth + 1)
            copied_item = _copy_plain_runtime_value(item, depth=depth + 1)
            if (
                copied_key is _UNSAFE_RUNTIME_VALUE
                or copied_item is _UNSAFE_RUNTIME_VALUE
            ):
                return _UNSAFE_RUNTIME_VALUE
            copied[copied_key] = copied_item
        return copied
    if isinstance(value, list):
        copied = [_copy_plain_runtime_value(item, depth=depth + 1) for item in value]
        return (
            _UNSAFE_RUNTIME_VALUE
            if any(item is _UNSAFE_RUNTIME_VALUE for item in copied)
            else copied
        )
    if isinstance(value, tuple):
        copied = tuple(
            _copy_plain_runtime_value(item, depth=depth + 1) for item in value
        )
        return (
            _UNSAFE_RUNTIME_VALUE
            if any(item is _UNSAFE_RUNTIME_VALUE for item in copied)
            else copied
        )
    if isinstance(value, (set, frozenset)):
        copied = [_copy_plain_runtime_value(item, depth=depth + 1) for item in value]
        if any(item is _UNSAFE_RUNTIME_VALUE for item in copied):
            return _UNSAFE_RUNTIME_VALUE
        return type(value)(copied)
    return _UNSAFE_RUNTIME_VALUE


def _capture_named_runtime(obj: Any, names: set[str]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    attributes = vars(obj)
    for name in names:
        if name not in attributes:
            continue
        value = _copy_plain_runtime_value(attributes[name])
        if value is not _UNSAFE_RUNTIME_VALUE:
            copied[name] = value
    return copied


def _capture_model_runtime(model: Any) -> dict[str, np.ndarray]:
    names = ("site_rgba", "site_size", "geom_rgba", "body_pos", "body_quat")
    return {
        name: np.asarray(getattr(model, name)).copy()
        for name in names
        if hasattr(model, name)
    }


def _capture_fixture_runtime(base: Any) -> dict[str, dict[str, Any]]:
    fixtures = getattr(base, "fixtures", {})
    return {
        str(name): _capture_named_runtime(fixture, _FIXTURE_RUNTIME_FIELDS)
        for name, fixture in fixtures.items()
    }


def _copy_object_attributes(obj: Any, *, exclude: set[str]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for name, value in vars(obj).items():
        if name in exclude or callable(value):
            continue
        try:
            copied[name] = deepcopy(value)
        except Exception:
            continue
    return copied


def _capture_controller_runtime(
    base: Any,
) -> tuple[dict[str, dict[str, Any]], ...]:
    all_robots: list[dict[str, dict[str, Any]]] = []
    for robot in base.robots:
        controllers: dict[str, dict[str, Any]] = {}
        for name, controller in robot.composite_controller.part_controllers.items():
            controllers[name] = _copy_object_attributes(controller, exclude={"sim"})
        all_robots.append(controllers)
    return tuple(all_robots)


def _capture_robot_runtime(base: Any) -> tuple[dict[str, Any], ...]:
    states: list[dict[str, Any]] = []
    for robot in base.robots:
        names = [
            name
            for name in vars(robot)
            if name == "torques" or name.startswith("recent_")
        ]
        states.append({name: deepcopy(getattr(robot, name)) for name in names})
    return tuple(states)


def _capture_gripper_actions(base: Any) -> tuple[dict[str, np.ndarray], ...]:
    states: list[dict[str, np.ndarray]] = []
    for robot in base.robots:
        states.append(
            {
                name: np.asarray(gripper.current_action).copy()
                for name, gripper in robot.gripper.items()
                if hasattr(gripper, "current_action")
            }
        )
    return tuple(states)


def _capture_observable_runtime(base: Any) -> dict[str, dict[str, Any]]:
    runtime_names = {
        "_time_since_last_sample",
        "_current_delay",
        "_current_observed_value",
        "_sampled",
    }
    return {
        name: {
            key: deepcopy(value)
            for key, value in vars(observable).items()
            if key in runtime_names
        }
        for name, observable in base._observables.items()
    }


def _wrapper_chain(env: Any) -> tuple[Any, ...]:
    wrappers: list[Any] = []
    current = env
    while hasattr(current, "env"):
        wrappers.append(current)
        current = current.env
    return tuple(wrappers)


def _capture_wrapper_runtime(env: Any) -> tuple[dict[str, Any], ...]:
    return tuple(
        _capture_named_runtime(wrapper, _WRAPPER_RUNTIME_FIELDS)
        for wrapper in _wrapper_chain(env)
    )


def capture_snapshot(env: Any) -> EnvSnapshot:
    base = inner_env(env)
    sim = base.sim
    rng = getattr(base, "rng", None)
    rng_state = None
    if rng is not None and hasattr(rng, "bit_generator"):
        rng_state = deepcopy(rng.bit_generator.state)

    return EnvSnapshot(
        sim_state=np.asarray(sim.get_state().flatten()).copy(),
        camera_positions=np.asarray(sim.model.cam_pos).copy(),
        camera_quaternions=np.asarray(sim.model.cam_quat).copy(),
        model_runtime=_capture_model_runtime(sim.model),
        ctrl=_copy_attr(sim.data, "ctrl"),
        qacc_warmstart=_copy_attr(sim.data, "qacc_warmstart"),
        mocap_positions=_copy_attr(sim.data, "mocap_pos"),
        mocap_quaternions=_copy_attr(sim.data, "mocap_quat"),
        rng_state=rng_state,
        numpy_random_state=deepcopy(np.random.get_state()),
        wrapper_runtime=_capture_wrapper_runtime(env),
        env_runtime=_capture_named_runtime(base, _ENV_RUNTIME_FIELDS),
        fixture_runtime=_capture_fixture_runtime(base),
        controller_runtime=_capture_controller_runtime(base),
        robot_runtime=_capture_robot_runtime(base),
        gripper_actions=_capture_gripper_actions(base),
        observable_runtime=_capture_observable_runtime(base),
    )


def _restore_attr(obj: Any, name: str, value: np.ndarray | None) -> None:
    if value is not None and hasattr(obj, name):
        getattr(obj, name)[:] = value


def _restore_runtime(base: Any, snapshot: EnvSnapshot) -> None:
    for name, value in snapshot.env_runtime.items():
        setattr(base, name, deepcopy(value))

    fixtures = getattr(base, "fixtures", {})
    if set(fixtures) != set(snapshot.fixture_runtime):
        raise RuntimeError("RoboCasa fixture registry changed after snapshot capture")
    for name, attributes in snapshot.fixture_runtime.items():
        fixture = fixtures[name]
        for attribute, value in attributes.items():
            setattr(fixture, attribute, deepcopy(value))

    for robot, controller_states in zip(
        base.robots, snapshot.controller_runtime, strict=True
    ):
        for name, attributes in controller_states.items():
            controller = robot.composite_controller.part_controllers[name]
            for attribute, value in attributes.items():
                setattr(controller, attribute, deepcopy(value))

    for robot, attributes in zip(base.robots, snapshot.robot_runtime, strict=True):
        for name, value in attributes.items():
            setattr(robot, name, deepcopy(value))

    for robot, actions in zip(base.robots, snapshot.gripper_actions, strict=True):
        for name, value in actions.items():
            robot.gripper[name].current_action = np.asarray(value).copy()

    for name, attributes in snapshot.observable_runtime.items():
        observable = base._observables[name]
        for attribute, value in attributes.items():
            setattr(observable, attribute, deepcopy(value))


def _restore_wrapper_runtime(env: Any, snapshot: EnvSnapshot) -> None:
    wrappers = _wrapper_chain(env)
    if len(wrappers) != len(snapshot.wrapper_runtime):
        raise RuntimeError("Gym wrapper stack changed after snapshot capture")
    for wrapper, attributes in zip(
        wrappers, snapshot.wrapper_runtime, strict=True
    ):
        for name, value in attributes.items():
            setattr(wrapper, name, deepcopy(value))


def restore_snapshot(env: Any, snapshot: EnvSnapshot) -> None:
    base = inner_env(env)
    sim = base.sim

    sim.model.cam_pos[:] = snapshot.camera_positions
    sim.model.cam_quat[:] = snapshot.camera_quaternions
    for name, value in snapshot.model_runtime.items():
        getattr(sim.model, name)[:] = value
    sim.set_state_from_flattened(snapshot.sim_state)
    _restore_attr(sim.data, "ctrl", snapshot.ctrl)
    _restore_attr(sim.data, "mocap_pos", snapshot.mocap_positions)
    _restore_attr(sim.data, "mocap_quat", snapshot.mocap_quaternions)
    _restore_runtime(base, snapshot)
    _restore_wrapper_runtime(env, snapshot)

    rng = getattr(base, "rng", None)
    if snapshot.rng_state is not None and rng is not None and hasattr(
        rng, "bit_generator"
    ):
        rng.bit_generator.state = deepcopy(snapshot.rng_state)
    np.random.set_state(deepcopy(snapshot.numpy_random_state))
    sim.forward()
    _restore_attr(sim.data, "qacc_warmstart", snapshot.qacc_warmstart)

