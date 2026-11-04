"""Frozen GR1 29D absolute-control and actual ego observation mapping.

Uses upstream hand unformatting/key conversion and image padding. No fake wrist
view, camera crop change, delta conversion, simulator step or model call here.
"""

import numpy as np

PARTS = (("left_arm", 7), ("right_arm", 7), ("left_hand", 6), ("right_hand", 6), ("waist", 3))


def _active(value):
    array = np.asarray(value)
    if array.shape != (29,) or not np.isfinite(array).all():
        raise ValueError("One finite canonical 29D absolute action required")
    return array


def action_dict(value):
    array = _active(value)
    offset = 0
    result = {}
    for part, size in PARTS:
        result["action." + part] = array[offset : offset + size].copy()
        offset += size
    return result


def environment_action(env, value):
    """Same flattening order as upstream Gym; no clipping or stepping."""
    from robocasa.models.robots import GR1ArmsAndWaistKeyConverter

    if len(env.robots) != 1:
        raise ValueError("Single audited GR1 required")
    robot = env.robots[0]
    cc = robot.composite_controller
    keyed = GR1ArmsAndWaistKeyConverter.unmap_action(action_dict(value))
    out = np.zeros(cc.action_limits[0].shape, dtype=np.float64)
    for part in cc.part_controllers:
        key = robot.robot_model.naming_prefix + part
        start, end = cc._action_split_indexes[part]
        if key not in keyed or keyed[key].shape != (end - start,):
            raise ValueError("Native controller/action dimensions differ")
        out[start:end] = keyed.pop(key)
    if keyed or out.shape != (29,):
        raise ValueError("Unexpected active controller layout")
    return out


def active_state(env):
    from robocasa.models.robots import gather_robot_observations, GR1ArmsAndWaistKeyConverter

    values = GR1ArmsAndWaistKeyConverter.map_obs(gather_robot_observations(env))
    result = []
    for part, size in PARTS:
        prefix = "hand." if part.endswith("_hand") else "body."
        value = np.asarray(values[prefix + part], dtype=np.float32)
        if value.shape != (size,):
            raise ValueError("Native state dimensions differ")
        result.append(value)
    return _active(np.concatenate(result))


def model_input(env, raw_mujoco_ego):
    """Exact upstream 800x1280 bottom-up RGB -> padded 256 ego convention."""
    from robocasa.utils.gym_utils.gymnasium_groot import GrootRoboCasaEnv
    from robocasa.models.robots.manipulators.gr1_robot import GR1ArmsAndWaist

    image = np.asarray(raw_mujoco_ego)
    if image.shape != (800, 1280, 3) or image.dtype != np.uint8:
        raise ValueError("Actual native 800x1280 uint8 MuJoCo ego frame required")
    if not isinstance(env.robots[0].robot_model, GR1ArmsAndWaist):
        raise ValueError("Task instruction convention requires audited GR1 waist embodiment")
    prompt = env.get_ep_meta().get("lang")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Actual task instruction required")
    return {
        "observation/image": GrootRoboCasaEnv.process_img(np.copy(image[::-1])),
        "observation/state": active_state(env),
        "prompt": "unlocked_waist: " + prompt,
    }
