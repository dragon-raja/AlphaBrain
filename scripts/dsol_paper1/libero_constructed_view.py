from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


CONSTRUCTION_SCHEMA = "dsol_libero_static_visual_occluder_v1"
TASK_VIEW_CONTEXT_SCHEMA = "dsol_libero_task_view_context_v1"


def _finite_vector(value: Sequence[float], *, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of length {length}")
    return array


def _entity_geom_ids(env: Any, entity_name: str) -> np.ndarray:
    from libero_visibility import _entity_geom_ids as visibility_entity_geom_ids

    ids, _source = visibility_entity_geom_ids(env, entity_name)
    return ids


def entity_world_center(env: Any, entity_name: str) -> np.ndarray:
    ids = _entity_geom_ids(env, entity_name)
    positions = np.asarray(
        [env.env.sim.data.geom_xpos[int(index)] for index in ids],
        dtype=np.float64,
    )
    return positions.mean(axis=0)


def equal_entity_world_center(env: Any, entity_names: Iterable[str]) -> np.ndarray:
    names = tuple(dict.fromkeys(str(value) for value in entity_names))
    if not names:
        raise ValueError("entity_names must not be empty")
    return np.asarray([entity_world_center(env, name) for name in names]).mean(axis=0)


def _yaw_quaternion_wxyz(normal_xy: np.ndarray) -> np.ndarray:
    if normal_xy.shape != (2,) or float(np.linalg.norm(normal_xy)) <= 1e-9:
        raise ValueError("occluder normal must have a nonzero horizontal component")
    yaw = math.atan2(float(normal_xy[1]), float(normal_xy[0]))
    return np.asarray(
        [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
        dtype=np.float64,
    )


def resolve_task_view_context(
    env: Any,
    specification: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the state-dependent task pivot without modifying the scene."""

    target_entity = str(specification["target_entity"])
    pivot_entities = tuple(
        str(value)
        for value in specification.get("camera_pivot_entities", [target_entity])
    )
    sim = env.env.sim
    sim.forward()
    camera_name = str(specification.get("camera_name", "agentview"))
    camera_id = int(sim.model.camera_name2id(camera_name))
    camera_position = np.asarray(sim.data.cam_xpos[camera_id], dtype=np.float64)
    target_center = entity_world_center(env, target_entity)
    camera_pivot = equal_entity_world_center(env, pivot_entities)
    resolved = {
        "schema": TASK_VIEW_CONTEXT_SCHEMA,
        "target_entity": target_entity,
        "camera_pivot_entities": list(pivot_entities),
        "camera_name": camera_name,
        "target_world_center": target_center.tolist(),
        "camera_pivot_world": camera_pivot.tolist(),
        "reference_camera_world": camera_position.tolist(),
    }
    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    resolved["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return resolved


def resolve_static_visual_occluder(
    env: Any,
    specification: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve a state-dependent, non-colliding visual occluder in world space."""

    context = resolve_task_view_context(env, specification)
    fraction = float(specification.get("camera_ray_fraction", 0.2))
    if not 0.02 <= fraction <= 0.8:
        raise ValueError("camera_ray_fraction must be in [0.02, 0.8]")
    half_size = _finite_vector(
        specification.get("half_size_xyz", [0.008, 0.12, 0.16]),
        length=3,
        name="half_size_xyz",
    )
    if np.any(half_size <= 0.0):
        raise ValueError("occluder half sizes must be positive")

    camera_position = np.asarray(context["reference_camera_world"], dtype=np.float64)
    target_center = np.asarray(context["target_world_center"], dtype=np.float64)
    center = target_center + fraction * (camera_position - target_center)
    horizontal_normal = camera_position[:2] - target_center[:2]
    horizontal_normal /= np.linalg.norm(horizontal_normal)
    horizontal_tangent = np.asarray(
        [-horizontal_normal[1], horizontal_normal[0]], dtype=np.float64
    )
    lateral_offset = float(specification.get("lateral_offset_m", 0.0))
    if not math.isfinite(lateral_offset) or abs(lateral_offset) > 0.5:
        raise ValueError("lateral_offset_m must be finite and within [-0.5, 0.5]")
    center[:2] += lateral_offset * horizontal_tangent
    quaternion = _yaw_quaternion_wxyz(horizontal_normal)
    rgba = _finite_vector(
        specification.get("rgba", [0.12, 0.12, 0.12, 1.0]),
        length=4,
        name="rgba",
    )
    resolved = {
        "schema": CONSTRUCTION_SCHEMA,
        "target_entity": context["target_entity"],
        "camera_pivot_entities": context["camera_pivot_entities"],
        "camera_name": context["camera_name"],
        "camera_ray_fraction": fraction,
        "lateral_offset_m": lateral_offset,
        "horizontal_tangent_world_xy": horizontal_tangent.tolist(),
        "target_world_center": target_center.tolist(),
        "camera_pivot_world": context["camera_pivot_world"],
        "reference_camera_world": camera_position.tolist(),
        "occluder": {
            "name": str(specification.get("name", "dsol_visual_occluder")),
            "position_world": center.tolist(),
            "quaternion_wxyz": quaternion.tolist(),
            "half_size_xyz": half_size.tolist(),
            "rgba": rgba.tolist(),
            "contype": 0,
            "conaffinity": 0,
            "render_group": 1,
            "has_joint": False,
        },
    }
    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    resolved["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return resolved


def task_orbit_pose_from_specification(
    *,
    context: Mapping[str, Any],
    specification: Mapping[str, Any],
    pair_id: str,
    pair_member: str,
) -> dict[str, Any]:
    """Build one frozen member of a task-centric mirrored camera pair."""

    member_sign = {"negative": -1.0, "positive": 1.0}
    if pair_member not in member_sign:
        raise ValueError("pair_member must be 'negative' or 'positive'")
    matching = [
        pair
        for pair in specification.get("candidate_pairs", [])
        if str(pair.get("pair_id")) == str(pair_id)
    ]
    if len(matching) != 1:
        raise ValueError(
            f"expected exactly one candidate pair {pair_id!r}, found {len(matching)}"
        )
    pair = matching[0]
    magnitude = abs(float(pair["azimuth_deg"]))
    return explicit_task_orbit_pose(
        reference_position=context["reference_camera_world"],
        pivot=context["camera_pivot_world"],
        azimuth_deg=member_sign[pair_member] * magnitude,
        elevation_deg=float(pair.get("elevation_deg", 0.0)),
        radius_scale=float(pair.get("radius_scale", 1.0)),
        pose_id=f"taskcentric_{pair_id}_{pair_member}",
        pair_id=str(pair_id),
        pair_member=pair_member,
    )


def inject_static_visual_occluder(
    model_xml: str,
    construction: Mapping[str, Any],
) -> str:
    if construction.get("schema") != CONSTRUCTION_SCHEMA:
        raise ValueError("unsupported scene-construction schema")
    occluder = construction["occluder"]
    root = ET.fromstring(model_xml)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MuJoCo XML has no worldbody")
    body_name = str(occluder["name"])
    if root.find(f".//body[@name='{body_name}']") is not None:
        raise ValueError(f"MuJoCo XML already contains body {body_name!r}")

    def values(name: str) -> str:
        return " ".join(f"{float(value):.12g}" for value in occluder[name])

    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": body_name,
            "pos": values("position_world"),
            "quat": values("quaternion_wxyz"),
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{body_name}_geom",
            "type": "box",
            "size": values("half_size_xyz"),
            "rgba": values("rgba"),
            "contype": "0",
            "conaffinity": "0",
            # LIBERO renders visual geoms from group 1; group 0 is collision-only.
            "group": "1",
        },
    )
    return ET.tostring(root, encoding="unicode")


def explicit_task_orbit_pose(
    *,
    reference_position: Sequence[float],
    pivot: Sequence[float],
    azimuth_deg: float,
    elevation_deg: float,
    radius_scale: float,
    pose_id: str,
    pair_id: str,
    pair_member: str,
) -> dict[str, Any]:
    from libero_camera_pose import look_at_rotation, rotation_matrix_to_wxyz

    reference = _finite_vector(reference_position, length=3, name="reference_position")
    target = _finite_vector(pivot, length=3, name="pivot")
    relative = reference - target
    radius = float(np.linalg.norm(relative))
    if radius <= 1e-9:
        raise ValueError("reference camera and task pivot must differ")
    if not 0.25 <= float(radius_scale) <= 2.5:
        raise ValueError("radius_scale must be in [0.25, 2.5]")
    azimuth = math.atan2(float(relative[1]), float(relative[0]))
    elevation = math.asin(float(relative[2]) / radius)
    new_elevation = elevation + math.radians(float(elevation_deg))
    if not math.radians(-85.0) < new_elevation < math.radians(85.0):
        raise ValueError("camera elevation leaves supported range")
    new_azimuth = azimuth + math.radians(float(azimuth_deg))
    position = target + radius * float(radius_scale) * np.asarray(
        [
            math.cos(new_elevation) * math.cos(new_azimuth),
            math.cos(new_elevation) * math.sin(new_azimuth),
            math.sin(new_elevation),
        ],
        dtype=np.float64,
    )
    quaternion = rotation_matrix_to_wxyz(look_at_rotation(position, target))
    return {
        "pose_id": str(pose_id),
        "orientation_mode": "explicit_world_look_at",
        "position_world": position.tolist(),
        "quaternion_wxyz": quaternion.tolist(),
        "pivot_world": target.tolist(),
        "azimuth_deg": float(azimuth_deg),
        "elevation_deg": float(elevation_deg),
        "radius_scale": float(radius_scale),
        "pair_id": str(pair_id),
        "pair_member": str(pair_member),
    }


def paired_task_orbit_poses(
    construction: Mapping[str, Any],
    candidate_pairs: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    poses = []
    for index, pair in enumerate(candidate_pairs):
        pair_id = str(pair.get("pair_id", f"constructed_pair_{index:02d}"))
        magnitude = abs(float(pair["azimuth_deg"]))
        if magnitude <= 0.0:
            raise ValueError("paired candidate azimuth magnitude must be positive")
        for member, sign in (("negative", -1.0), ("positive", 1.0)):
            poses.append(
                explicit_task_orbit_pose(
                    reference_position=construction["reference_camera_world"],
                    pivot=construction["camera_pivot_world"],
                    azimuth_deg=sign * magnitude,
                    elevation_deg=float(pair.get("elevation_deg", 0.0)),
                    radius_scale=float(pair.get("radius_scale", 1.0)),
                    pose_id=f"{pair_id}_{member}",
                    pair_id=pair_id,
                    pair_member=member,
                )
            )
    return poses


def install_constructed_camera_pose(
    env: Any,
    reference: Mapping[str, Any],
    pose: Mapping[str, Any],
) -> dict[str, Any]:
    if pose.get("orientation_mode") != "explicit_world_look_at":
        raise ValueError("constructed camera pose must use explicit_world_look_at")
    position = _finite_vector(pose["position_world"], length=3, name="position_world")
    quaternion = _finite_vector(
        pose["quaternion_wxyz"], length=4, name="quaternion_wxyz"
    )
    sim = env.env.sim
    camera_id = int(reference["camera_id"])
    if int(sim.model.cam_bodyid[camera_id]) != 0:
        raise ValueError("explicit world pose currently requires a world-body camera")
    sim.model.cam_pos[camera_id] = position
    sim.model.cam_quat[camera_id] = quaternion
    sim.forward()
    return {
        "camera_name": str(reference["camera_name"]),
        "camera_position": position.tolist(),
        "camera_quaternion_wxyz": quaternion.tolist(),
        "camera_pivot": list(pose["pivot_world"]),
        "camera_fovy": float(reference["fovy"]),
        "camera_azimuth_deg": float(pose["azimuth_deg"]),
        "camera_elevation_deg": float(pose["elevation_deg"]),
        "camera_radius_scale": float(pose["radius_scale"]),
        "pair_id": str(pose["pair_id"]),
        "pair_member": str(pose["pair_member"]),
    }
