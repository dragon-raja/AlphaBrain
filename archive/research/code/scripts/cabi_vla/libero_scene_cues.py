from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import numpy as np


SCENE_CUE_MODES = ("fixed", "cue_randomized")
BACKGROUND_GEOM_NAMES = {
    "floor",
    "living_room",
    "wall_decoration",
    "wall_leftcorner_visual",
    "wall_rightcorner_visual",
    "wall_left_visual",
    "wall_right_visual",
    "wall_rear_visual",
    "wall_front_visual",
}


def stable_scene_seed(seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{seed}::{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _body_name(model: Any, body_id: int) -> str:
    name = model.body_id2name(int(body_id))
    return "" if name is None else str(name)


def _yaw_quaternion(angle: float) -> np.ndarray:
    return np.asarray(
        [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)],
        dtype=np.float64,
    )


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = map(float, left)
    rw, rx, ry, rz = map(float, right)
    result = np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )
    return result / np.linalg.norm(result)


def capture_scene_cue_reference(env: Any) -> dict[str, Any]:
    """Capture render-only scene assets without touching simulator state."""

    model = env.env.sim.model
    named_background = [
        index
        for index, name in enumerate(model.geom_names)
        if name in BACKGROUND_GEOM_NAMES
    ]
    visual_table = [
        index
        for index in range(model.ngeom)
        if _body_name(model, model.geom_bodyid[index]) == "living_room_table_col"
        and int(model.geom_group[index]) == 1
        and int(model.geom_contype[index]) == 0
        and int(model.geom_conaffinity[index]) == 0
    ]
    robot_base = [
        index
        for index in range(model.ngeom)
        if _body_name(model, model.geom_bodyid[index]) == "robot0_link0"
        and int(model.geom_group[index]) == 1
        and int(model.geom_contype[index]) == 0
    ]
    floor_body_id = int(model.body_name2id("floor"))
    background_ids = sorted(set(named_background + visual_table))
    if not visual_table:
        raise RuntimeError("LIBERO scene has no visual-only living-room table geom")
    if not background_ids:
        raise RuntimeError("LIBERO scene has no recognized background geoms")
    return {
        "body_pos": np.asarray(model.body_pos).copy(),
        "body_quat": np.asarray(model.body_quat).copy(),
        "geom_pos": np.asarray(model.geom_pos).copy(),
        "geom_quat": np.asarray(model.geom_quat).copy(),
        "geom_rgba": np.asarray(model.geom_rgba).copy(),
        "geom_group": np.asarray(model.geom_group).copy(),
        "mat_rgba": np.asarray(model.mat_rgba).copy(),
        "light_pos": np.asarray(model.light_pos).copy(),
        "light_diffuse": np.asarray(model.light_diffuse).copy(),
        "light_specular": np.asarray(model.light_specular).copy(),
        "light_ambient": np.asarray(model.light_ambient).copy(),
        "floor_body_id": floor_body_id,
        "background_geom_ids": np.asarray(background_ids, dtype=np.int32),
        "visual_table_geom_ids": np.asarray(visual_table, dtype=np.int32),
        "robot_base_geom_ids": np.asarray(robot_base, dtype=np.int32),
    }


def restore_scene_cues(env: Any, reference: Mapping[str, Any]) -> None:
    model = env.env.sim.model
    for key in (
        "body_pos",
        "body_quat",
        "geom_pos",
        "geom_quat",
        "geom_rgba",
        "geom_group",
        "mat_rgba",
        "light_pos",
        "light_diffuse",
        "light_specular",
        "light_ambient",
    ):
        getattr(model, key)[:] = np.asarray(reference[key])
    env.env.sim.forward()


def install_scene_cues(
    env: Any,
    reference: Mapping[str, Any],
    *,
    mode: str,
    seed: int,
    sample_id: str,
) -> dict[str, Any]:
    """Install a deterministic visual scene intervention.

    The intervention is restricted to render assets. Task objects, robot
    kinematics, collision geometry, and camera parameters are unchanged.
    """

    if mode not in SCENE_CUE_MODES:
        raise ValueError(f"scene cue mode must be one of {SCENE_CUE_MODES}")
    restore_scene_cues(env, reference)
    if mode == "fixed":
        return {
            "scene_cue_mode": mode,
            "scene_cue_seed": None,
            "robot_base_visual_hidden": False,
        }

    resolved_seed = stable_scene_seed(seed, sample_id)
    rng = np.random.default_rng(resolved_seed)
    model = env.env.sim.model

    floor_id = int(reference["floor_body_id"])
    floor_xy = rng.uniform(-2.0, 2.0, size=2)
    floor_yaw = float(rng.uniform(-math.pi, math.pi))
    model.body_pos[floor_id, :2] = (
        np.asarray(reference["body_pos"])[floor_id, :2] + floor_xy
    )
    model.body_quat[floor_id] = _quaternion_multiply(
        _yaw_quaternion(floor_yaw),
        np.asarray(reference["body_quat"])[floor_id],
    )

    table_xy = rng.uniform(-0.20, 0.20, size=2)
    table_yaw = float(rng.uniform(-math.pi, math.pi))
    for geom_id in np.asarray(reference["visual_table_geom_ids"], dtype=np.int32):
        model.geom_pos[geom_id, :2] = (
            np.asarray(reference["geom_pos"])[geom_id, :2] + table_xy
        )
        model.geom_quat[geom_id] = _quaternion_multiply(
            _yaw_quaternion(table_yaw),
            np.asarray(reference["geom_quat"])[geom_id],
        )

    background_ids = np.asarray(reference["background_geom_ids"], dtype=np.int32)
    table_ids = set(
        map(int, np.asarray(reference["visual_table_geom_ids"], dtype=np.int32))
    )
    for geom_id in background_ids:
        if int(geom_id) not in table_ids and model.geom_names[int(geom_id)] != "floor":
            model.geom_group[geom_id] = 0
        material_id = int(model.geom_matid[geom_id])
        if material_id >= 0:
            tint = rng.uniform(0.55, 1.25, size=3)
            base = np.asarray(reference["mat_rgba"])[material_id, :3]
            model.mat_rgba[material_id, :3] = np.clip(base * tint, 0.05, 1.0)

    robot_base_ids = np.asarray(reference["robot_base_geom_ids"], dtype=np.int32)
    model.geom_group[robot_base_ids] = 0

    if model.nlight:
        model.light_pos[:] = (
            np.asarray(reference["light_pos"])
            + rng.uniform(-0.45, 0.45, size=model.light_pos.shape)
        )
        diffuse_scale = rng.uniform(0.65, 1.20, size=(model.nlight, 1))
        model.light_diffuse[:] = np.clip(
            np.asarray(reference["light_diffuse"]) * diffuse_scale,
            0.05,
            1.0,
        )
        model.light_ambient[:] = np.clip(
            np.asarray(reference["light_ambient"])
            + rng.uniform(0.0, 0.12, size=model.light_ambient.shape),
            0.0,
            1.0,
        )

    env.env.sim.forward()
    return {
        "scene_cue_mode": mode,
        "scene_cue_seed": int(resolved_seed),
        "floor_texture_xy": floor_xy.tolist(),
        "floor_texture_yaw_deg": math.degrees(floor_yaw),
        "visual_table_xy": table_xy.tolist(),
        "visual_table_yaw_deg": math.degrees(table_yaw),
        "robot_base_visual_hidden": bool(len(robot_base_ids)),
        "fixed_room_visuals_hidden": True,
    }


def background_segmentation_mask(
    segmentation: np.ndarray,
    background_geom_ids: np.ndarray,
) -> np.ndarray:
    import mujoco

    values = np.asarray(segmentation)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("segmentation must have shape [H, W, 2]")
    return (
        values[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM)
    ) & np.isin(values[..., 1], np.asarray(background_geom_ids, dtype=np.int32))


def render_background_only(
    env: Any,
    *,
    camera_name: str,
    height: int,
    width: int,
    background_geom_ids: np.ndarray,
) -> np.ndarray:
    """Render recognized scene cues without robot or task-object silhouettes."""

    sim = env.env.sim
    original_geom_groups = np.asarray(sim.model.geom_group).copy()
    context = sim._render_context_offscreen
    if context is None:
        sim.render(
            camera_name=camera_name,
            height=height,
            width=width,
        )
        context = sim._render_context_offscreen
    original_render_groups = np.asarray(context.vopt.geomgroup).copy()
    if len(original_render_groups) < 6:
        raise RuntimeError("MuJoCo render context has no spare geometry group")
    background_ids = np.asarray(background_geom_ids, dtype=np.int32)
    active_ids = np.asarray(
        [
            int(geom_id)
            for geom_id in background_ids
            if original_render_groups[
                int(original_geom_groups[int(geom_id)])
            ]
        ],
        dtype=np.int32,
    )
    if len(active_ids) == 0:
        raise RuntimeError("recognized background has no normally rendered geoms")
    selected_group = len(original_render_groups) - 1
    try:
        sim.model.geom_group[:] = original_geom_groups
        sim.model.geom_group[active_ids] = selected_group
        context.vopt.geomgroup[:] = 0
        context.vopt.geomgroup[selected_group] = 1
        sim.forward()
        return np.asarray(
            sim.render(
                camera_name=camera_name,
                height=height,
                width=width,
            )
        ).copy()
    finally:
        sim.model.geom_group[:] = original_geom_groups
        context.vopt.geomgroup[:] = original_render_groups
        sim.forward()
