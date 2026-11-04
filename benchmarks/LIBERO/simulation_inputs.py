"""LIBERO input construction, initialization and rendering contracts.

Imported by adapters without reaching into executable scripts. Functions were
extracted without numerical changes; runtime roots and render protocol remain
explicit. Importing this module never creates a simulator or model.
"""

from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping
import hashlib
import json
import math
import os
import sys
import numpy as np
from AlphaBrain.common.camera_pose import mujoco_camera_calibration
from AlphaBrain.research.dsol.data.artifacts import read, sha
from AlphaBrain.research.dsol.data.rollout_replay import array_identity

DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
POLICY_CAMERA_RESOLUTION = 224
KIND = "official_libero_initial_state_v1"


def configure_imports():
    """Compatibility no-op: runtime imports are now qualified package imports."""


def prepare_policy_observation(
    observation: Mapping[str, Any],
    *,
    prompt: str,
    resize_size: int,
    eval_seed: int,
    camera_calibration: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    from openpi_client import image_tools

    agent = np.ascontiguousarray(np.asarray(observation["agentview_image"])[::-1, ::-1])
    wrist = np.ascontiguousarray(np.asarray(observation["robot0_eye_in_hand_image"])[::-1, ::-1])
    agent = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(agent, resize_size, resize_size)
    )
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist, resize_size, resize_size)
    )
    state = np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"]),
            quat_to_axis_angle(np.asarray(observation["robot0_eef_quat"])),
            np.asarray(observation["robot0_gripper_qpos"]),
        )
    )
    return (
        {
            "observation/image": agent,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "prompt": str(prompt),
            "_eval_seed": int(eval_seed),
            "camera_intrinsics": np.asarray(
                camera_calibration["camera_intrinsics"], dtype=np.float64
            ),
            "camera_to_world_opencv": np.asarray(
                camera_calibration["camera_to_world_opencv"], dtype=np.float64
            ),
        },
        agent,
        wrist,
    )


def agentview_camera_calibration(env: Any) -> dict[str, np.ndarray]:
    calibration = mujoco_camera_calibration(
        env,
        camera_name="agentview",
        height=POLICY_CAMERA_RESOLUTION,
        width=POLICY_CAMERA_RESOLUTION,
    )
    return {
        "camera_intrinsics": np.asarray(calibration["intrinsics"], dtype=np.float64),
        "camera_to_world_opencv": np.asarray(
            calibration["camera_to_world_opencv"], dtype=np.float64
        ),
    }


def quat_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float64)
    return quat[:3] * (2.0 * math.acos(quat[3]) / denominator)


def physics_state_sha256(env: Any) -> tuple[str, int]:
    state = np.asarray(env.env.sim.get_state().flatten(), dtype=np.float64)
    return hashlib.sha256(state.tobytes()).hexdigest(), int(state.size)


def _runtime_config(runtime: Path, dataset_root: Path) -> dict[str, str]:
    benchmark_root = runtime / "libero" / "libero"
    return {
        "assets": str(benchmark_root / "assets"),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "benchmark_root": str(benchmark_root),
        "datasets": str(dataset_root),
        "init_states": str(benchmark_root / "init_files"),
    }


def _configure_runtime(runtime: Path, dataset_root: Path, config_root: Path) -> None:
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "config.yaml").write_text(
        json.dumps(
            _runtime_config(runtime=runtime, dataset_root=dataset_root),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    sys.path.insert(0, str(runtime))


def _restore_reference(env: Any, reference: Mapping[str, Any]) -> None:
    sim = env.env.sim
    camera_id = int(reference["camera_id"])
    sim.model.cam_pos[camera_id] = np.asarray(reference["position"])
    sim.model.cam_quat[camera_id] = np.asarray(reference["quaternion"])
    sim.forward()


def masked_policy_observation(
    observation: Mapping[str, Any],
    *,
    prompt: str,
    resize_size: int,
    eval_seed: int,
    camera_calibration: Mapping[str, Any],
    sensor_control: str,
    explicit_noise: np.ndarray | None = None,
    noise_sha256: str | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:

    example, agent, wrist = prepare_policy_observation(
        observation,
        prompt=prompt,
        resize_size=resize_size,
        eval_seed=eval_seed,
        camera_calibration=camera_calibration,
    )
    if sensor_control == "external_only":
        wrist = np.zeros_like(wrist)
        example["observation/wrist_image"] = wrist
    elif sensor_control == "wrist_only":
        agent = np.zeros_like(agent)
        example["observation/image"] = agent
    elif sensor_control == "all_blackout":
        agent = np.zeros_like(agent)
        wrist = np.zeros_like(wrist)
        example["observation/image"] = agent
        example["observation/wrist_image"] = wrist
    elif sensor_control != "both":
        raise ValueError(f"unsupported sensor control: {sensor_control}")
    if explicit_noise is not None:
        example["_eval_noise"] = np.ascontiguousarray(explicit_noise, dtype=np.float32)
        example["_eval_noise_sha256"] = str(noise_sha256)
    return example, agent, wrist


@contextmanager
def render_protocol(offsamples):
    if offsamples not in (0, 4):
        raise ValueError("Only preregistered AA0/AA4 protocols are permitted")
    from robosuite.utils.binding_utils import MjRenderContext

    original = MjRenderContext._set_mujoco_context_and_buffers
    contexts = []

    def initialize(context):
        if offsamples == 4 and int(context.model.vis.quality.offsamples) != 4:
            raise ValueError("Historical XML does not have the expected default MSAA4 setting")
        context.model.vis.quality.offsamples = offsamples
        result = original(context)
        if int(context.con.offSamples) != offsamples:
            raise ValueError("Actual framebuffer sample count differs")
        contexts.append(int(context.con.offSamples))
        return result

    MjRenderContext._set_mujoco_context_and_buffers = initialize
    try:
        yield contexts
    finally:
        MjRenderContext._set_mujoco_context_and_buffers = original


def load_initial(spec):
    init = spec["initialization"]
    assert init["kind"] == KIND
    assert sha(init["array_path"]) == init["array_sha256"]
    assert sha(spec["bddl_file"]) == spec["bddl_sha256"]
    state = np.load(init["array_path"], allow_pickle=False)
    assert array_identity(state) == init["array_identity"]
    return state


def initialize(env, spec, physical):
    """No demonstration XML/state; official reset, set initial, 10 no-op steps."""
    env.seed(int(spec["environment_seed"]))
    env.reset()
    observation = env.set_init_state(physical)
    if env.check_success():
        raise ValueError(
            "Official initial state already satisfies task: inspect, do not replace silently"
        )
    raw, _ = physics_state_sha256(env)
    for _ in range(int(spec["initialization"]["settle_steps"])):
        observation, _, done, _ = env.step(DUMMY_ACTION)
        if done:
            raise ValueError("Task completed during initialization; inspect protocol")
    settled, size = physics_state_sha256(env)
    return observation, dict(
        kind=KIND,
        official_file=spec["initialization"]["official_file"],
        official_file_sha256=spec["initialization"]["official_file_sha256"],
        init_state_index=spec["initialization"]["init_state_index"],
        raw_physics_sha256=raw,
        settled_physics_sha256=settled,
        settled_physics_size=size,
        settle_steps=10,
        demonstration_state_used=False,
        demonstration_xml_used=False,
    )


def verify_policy_input(spec, example):
    asset = read(spec["initial_asset_record"])
    expected = next(
        v for v in asset["input_identities"] if v["candidate_id"] == spec["selected_candidate_id"]
    )
    for key, identity in expected["inputs"].items():
        if array_identity(example[key]) != identity:
            raise ValueError("Rendered metric/rollout input mismatch: " + key)
    return sha(spec["initial_asset_record"])


RAW_OBSERVATION = 'LIBERO_RAW_RGB256_PROPRIO_V1'

def raw_observation(observation):
    """Copy raw simulator fields before either model's image preprocessing."""
    out = {'schema': RAW_OBSERVATION}
    shapes = {'agentview_image': (256,256,3), 'robot0_eye_in_hand_image': (256,256,3),
              'robot0_eef_pos': (3,), 'robot0_eef_quat': (4,), 'robot0_gripper_qpos': (2,)}
    for key, shape in shapes.items():
        value = np.asarray(observation[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Invalid raw LIBERO observation: '+key)
        if key.endswith('_image') and value.dtype != np.uint8:
            raise ValueError('Raw images must be uint8; no already-normalized pixels')
        out[key] = value.copy()
    return out
