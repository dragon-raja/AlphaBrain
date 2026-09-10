from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


POSE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
CAMERA_PARAMETERS = ("azimuth_deg", "elevation_deg", "radius_scale")
CAMERA_PARAMETER_NEUTRALS = {
    "azimuth_deg": 0.0,
    "elevation_deg": 0.0,
    "radius_scale": 1.0,
}
CAMERA_PARAMETER_PREFIXES = {
    "azimuth_deg": "az",
    "elevation_deg": "el",
    "radius_scale": "rad",
}


def _as_vector(value: Any, *, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of length {length}")
    return array


def rotation_matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0:
        quaternion *= -1
    return quaternion


def look_at_rotation(
    position: Sequence[float],
    target: Sequence[float],
    *,
    world_up: Sequence[float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    position_vector = _as_vector(position, length=3, name="position")
    target_vector = _as_vector(target, length=3, name="target")
    up_vector = _as_vector(world_up, length=3, name="world_up")
    forward = target_vector - position_vector
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1e-9:
        raise ValueError("camera position and target must differ")
    forward /= forward_norm
    right = np.cross(forward, up_vector)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-9:
        raise ValueError("camera optical axis may not be parallel to world_up")
    right /= right_norm
    camera_up = np.cross(-forward, right)
    camera_up /= np.linalg.norm(camera_up)
    return np.column_stack((right, camera_up, -forward))


def capture_camera_reference(
    env: Any,
    *,
    camera_name: str,
    table_plane_z: float,
) -> dict[str, Any]:
    sim = env.env.sim
    sim.forward()
    camera_id = int(sim.model.camera_name2id(camera_name))
    position = np.asarray(sim.model.cam_pos[camera_id], dtype=np.float64).copy()
    quaternion = np.asarray(sim.model.cam_quat[camera_id], dtype=np.float64).copy()
    rotation = np.asarray(sim.data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3).copy()
    forward = -rotation[:, 2]
    if abs(float(forward[2])) <= 1e-9:
        raise ValueError("camera optical axis does not intersect the table plane")
    distance = (float(table_plane_z) - float(position[2])) / float(forward[2])
    if distance <= 0:
        raise ValueError("table plane lies behind the camera")
    pivot = position + distance * forward
    return {
        "camera_name": camera_name,
        "camera_id": camera_id,
        "position": position,
        "quaternion": quaternion,
        "rotation": rotation,
        "pivot": pivot,
        "table_plane_z": float(table_plane_z),
        "fovy": float(sim.model.cam_fovy[camera_id]),
    }


def camera_intrinsics(
    *,
    fovy_deg: float,
    height: int,
    width: int,
) -> np.ndarray:
    """Return pinhole intrinsics for MuJoCo's square-pixel camera model."""
    if height <= 0 or width <= 0:
        raise ValueError("camera height and width must be positive")
    fovy = float(fovy_deg)
    if not math.isfinite(fovy) or not 0.0 < fovy < 180.0:
        raise ValueError("camera fovy must be in (0, 180) degrees")
    focal = float(height) / (2.0 * math.tan(math.radians(fovy) / 2.0))
    return np.asarray(
        [
            [focal, 0.0, float(width) / 2.0],
            [0.0, focal, float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def mujoco_camera_calibration(
    env: Any,
    *,
    camera_name: str,
    height: int,
    width: int,
) -> dict[str, np.ndarray | float | int | str]:
    """Read OpenCV-convention intrinsics and camera-to-world from MuJoCo.

    MuJoCo cameras use +x right, +y up, and -z forward. OpenCV uses +x right,
    +y down, and +z forward, so both the y and z camera axes are negated.
    """
    sim = env.env.sim
    sim.forward()
    camera_id = int(sim.model.camera_name2id(camera_name))
    rotation_mujoco = np.asarray(
        sim.data.cam_xmat[camera_id], dtype=np.float64
    ).reshape(3, 3)
    position = np.asarray(sim.data.cam_xpos[camera_id], dtype=np.float64).copy()
    rotation_opencv = np.column_stack(
        (
            rotation_mujoco[:, 0],
            -rotation_mujoco[:, 1],
            -rotation_mujoco[:, 2],
        )
    )
    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = rotation_opencv
    camera_to_world[:3, 3] = position
    fovy = float(sim.model.cam_fovy[camera_id])
    return {
        "camera_name": camera_name,
        "camera_id": camera_id,
        "height": int(height),
        "width": int(width),
        "fovy_deg": fovy,
        "intrinsics": camera_intrinsics(
            fovy_deg=fovy,
            height=height,
            width=width,
        ),
        "camera_to_world_opencv": camera_to_world,
    }


def project_world_points(
    points: np.ndarray | Sequence[Sequence[float]],
    *,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to raw MuJoCo image coordinates."""
    world = np.asarray(points, dtype=np.float64)
    if world.ndim == 1:
        world = world[None, :]
    if world.ndim != 2 or world.shape[1] != 3 or not np.all(np.isfinite(world)):
        raise ValueError("points must be a finite Nx3 array")
    matrix = np.asarray(camera_to_world, dtype=np.float64)
    intrinsic = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (4, 4) or intrinsic.shape != (3, 3):
        raise ValueError("camera_to_world and intrinsics must be 4x4 and 3x3")
    homogeneous = np.concatenate(
        [world, np.ones((len(world), 1), dtype=np.float64)],
        axis=1,
    )
    camera = (np.linalg.inv(matrix) @ homogeneous.T).T[:, :3]
    in_front = camera[:, 2] > 1e-9
    projected = np.full((len(world), 2), np.nan, dtype=np.float64)
    valid = camera[in_front]
    if len(valid):
        pixels = (intrinsic @ valid.T).T
        projected[in_front] = pixels[:, :2] / pixels[:, 2:3]
    return projected, camera[:, 2]


def opencv_pixels_to_policy(
    pixels: np.ndarray | Sequence[Sequence[float]],
    *,
    width: int,
) -> np.ndarray:
    """Map OpenCV-top-left pixels to the LIBERO policy image convention."""

    values = np.asarray(pixels, dtype=np.float64)
    if values.shape[-1:] != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("pixels must be a finite array ending in two coordinates")
    if width <= 0:
        raise ValueError("width must be positive")
    result = values.copy()
    result[..., 0] = float(width - 1) - result[..., 0]
    return result


def plucker_raymap(
    *,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    height: int,
    width: int,
    image_transform: str = "mujoco_upright",
) -> np.ndarray:
    """Generate KYC's per-pixel (direction, moment) Plucker ray map."""
    intrinsic = np.asarray(intrinsics, dtype=np.float64)
    matrix = np.asarray(camera_to_world, dtype=np.float64)
    if intrinsic.shape != (3, 3) or matrix.shape != (4, 4):
        raise ValueError("intrinsics and camera_to_world must be 3x3 and 4x4")
    vv, uu = np.meshgrid(
        np.arange(height, dtype=np.float64) + 0.5,
        np.arange(width, dtype=np.float64) + 0.5,
        indexing="ij",
    )
    rays = np.stack([uu, vv, np.ones_like(uu)], axis=-1)
    directions_camera = rays @ np.linalg.inv(intrinsic).T
    directions_world = directions_camera @ matrix[:3, :3].T
    directions_world /= np.linalg.norm(directions_world, axis=-1, keepdims=True)
    origin = matrix[:3, 3].reshape(1, 1, 3)
    moment = np.cross(origin, directions_world)
    result = np.concatenate([directions_world, moment], axis=-1)
    if image_transform == "rot180":
        result = np.flip(result, axis=(0, 1))
    elif image_transform == "mujoco_upright":
        result = np.flip(result, axis=1)
    elif image_transform != "none":
        raise ValueError(f"unsupported camera image transform: {image_transform}")
    return result.astype(np.float32, copy=True)


def _object_geom_ids(env: Any, object_name: str) -> np.ndarray:
    ids = [
        index
        for index, name in enumerate(env.env.sim.model.geom_names)
        if name and object_name in name
    ]
    if not ids:
        raise KeyError(f"no MuJoCo geoms found for {object_name}")
    return np.asarray(ids, dtype=np.int32)


def _visibility_class(
    *,
    center_in_frame: bool,
    visible_pixels: int,
    touches_border: bool,
    minimum_pixels: int,
) -> str:
    if visible_pixels == 0:
        return "occluded_or_subpixel" if center_in_frame else "out_of_frame"
    if visible_pixels < minimum_pixels:
        return "tiny"
    if touches_border:
        return "partial"
    return "full"


def _segmentation_geom_mask(
    segmentation: np.ndarray,
    geom_ids: np.ndarray,
) -> np.ndarray:
    import mujoco

    return (
        segmentation[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM)
    ) & np.isin(segmentation[..., 1], geom_ids)


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    if not np.any(mask):
        return None
    yy, xx = np.where(mask)
    return [
        int(xx.min()),
        int(yy.min()),
        int(xx.max()),
        int(yy.max()),
    ]


def _patch_support(mask: np.ndarray, *, patch_size: int = 14) -> int:
    height, width = mask.shape
    count = 0
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            if np.any(mask[top : top + patch_size, left : left + patch_size]):
                count += 1
    return count


def _isolated_guard_mask(
    env: Any,
    *,
    camera_name: str,
    geom_ids: np.ndarray,
    height: int,
    width: int,
    initial_factor: int = 2,
    maximum_factor: int = 4,
) -> dict[str, Any]:
    """Render an isolated object on a wider sensor at unchanged focal length."""

    sim = env.env.sim
    camera_id = int(sim.model.camera_name2id(camera_name))
    original_fovy = float(sim.model.cam_fovy[camera_id])
    original_geom_groups = np.asarray(sim.model.geom_group).copy()
    context = sim._render_context_offscreen
    if context is None:
        sim.render(
            camera_name=camera_name,
            height=height,
            width=width,
            segmentation=True,
        )
        context = sim._render_context_offscreen
    original_render_groups = np.asarray(context.vopt.geomgroup).copy()
    if len(original_render_groups) < 6:
        raise RuntimeError("MuJoCo render context has no spare geometry group")
    active_geom_ids = np.asarray(
        [
            int(geom_id)
            for geom_id in geom_ids
            if original_render_groups[
                int(original_geom_groups[int(geom_id)])
            ]
        ],
        dtype=np.int32,
    )
    if len(active_geom_ids) == 0:
        raise RuntimeError("object has no geoms enabled in the normal render")

    selected_group = len(original_render_groups) - 1
    try:
        sim.model.geom_group[:] = original_geom_groups
        sim.model.geom_group[active_geom_ids] = selected_group
        context.vopt.geomgroup[:] = 0
        context.vopt.geomgroup[selected_group] = 1

        factor = int(initial_factor)
        guard_mask = None
        while True:
            guard_height = height * factor
            guard_width = width * factor
            guard_fovy = math.degrees(
                2.0
                * math.atan(
                    factor * math.tan(math.radians(original_fovy) / 2.0)
                )
            )
            sim.model.cam_fovy[camera_id] = guard_fovy
            sim.forward()
            segmentation = np.asarray(
                sim.render(
                    camera_name=camera_name,
                    height=guard_height,
                    width=guard_width,
                    segmentation=True,
                )
            )
            guard_mask = _segmentation_geom_mask(
                segmentation,
                active_geom_ids,
            )
            bbox = _mask_bbox(guard_mask)
            touches_guard = bool(
                bbox is not None
                and (
                    bbox[0] == 0
                    or bbox[1] == 0
                    or bbox[2] == guard_width - 1
                    or bbox[3] == guard_height - 1
                )
            )
            if not touches_guard or factor >= maximum_factor:
                break
            factor *= 2

        offset_y = (guard_mask.shape[0] - height) // 2
        offset_x = (guard_mask.shape[1] - width) // 2
        in_frame = guard_mask[
            offset_y : offset_y + height,
            offset_x : offset_x + width,
        ]
        guard_bbox = _mask_bbox(guard_mask)
        projected_bbox = (
            None
            if guard_bbox is None
            else [
                guard_bbox[0] - offset_x,
                guard_bbox[1] - offset_y,
                guard_bbox[2] - offset_x,
                guard_bbox[3] - offset_y,
            ]
        )
        projected_pixels = int(guard_mask.sum())
        in_frame_pixels = int(in_frame.sum())
        clipping_fraction = (
            1.0 - in_frame_pixels / projected_pixels
            if projected_pixels
            else 1.0
        )
        signed_margin = (
            None
            if projected_bbox is None
            else float(
                min(
                    projected_bbox[0],
                    projected_bbox[1],
                    width - 1 - projected_bbox[2],
                    height - 1 - projected_bbox[3],
                )
            )
        )
        return {
            "projected_mask": in_frame,
            "projected_pixels_total": projected_pixels,
            "projected_pixels_in_frame": in_frame_pixels,
            "projected_bbox_raw": projected_bbox,
            "projected_signed_border_margin_px": signed_margin,
            "fov_clipping_fraction": float(
                np.clip(clipping_fraction, 0.0, 1.0)
            ),
            "guard_factor": factor,
            "guard_truncated": bool(
                guard_bbox is not None
                and (
                    guard_bbox[0] == 0
                    or guard_bbox[1] == 0
                    or guard_bbox[2] == guard_mask.shape[1] - 1
                    or guard_bbox[3] == guard_mask.shape[0] - 1
                )
            ),
        }
    finally:
        sim.model.cam_fovy[camera_id] = original_fovy
        sim.model.geom_group[:] = original_geom_groups
        context.vopt.geomgroup[:] = original_render_groups
        sim.forward()


def camera_task_visibility(
    env: Any,
    observation: Mapping[str, Any],
    *,
    camera_name: str,
    source_object: str,
    target_object: str,
    height: int,
    width: int,
    minimum_pixels: int = 64,
    detailed_geometry: bool = False,
) -> dict[str, Any]:
    """Measure task-object visibility using projection and rendered segmentation."""
    if minimum_pixels <= 0:
        raise ValueError("minimum_pixels must be positive")
    calibration = mujoco_camera_calibration(
        env,
        camera_name=camera_name,
        height=height,
        width=width,
    )
    segmentation = np.asarray(
        env.env.sim.render(
            camera_name=camera_name,
            height=height,
            width=width,
            segmentation=True,
        )
    )
    if segmentation.shape != (height, width, 2):
        raise ValueError(f"unexpected segmentation shape: {segmentation.shape}")

    result: dict[str, Any] = {
        "camera_intrinsics": np.asarray(calibration["intrinsics"]).tolist(),
        "camera_to_world_opencv": np.asarray(
            calibration["camera_to_world_opencv"]
        ).tolist(),
    }
    task_objects = (
        ("source", source_object),
        ("target", target_object),
    )
    fully_visible = []
    center_visible = []
    rendered_visible = []
    geometrically_inside = []
    for label, object_name in task_objects:
        center = np.asarray(
            observation[f"{object_name}_pos"], dtype=np.float64
        ).reshape(1, 3)
        uv, depth = project_world_points(
            center,
            intrinsics=np.asarray(calibration["intrinsics"]),
            camera_to_world=np.asarray(calibration["camera_to_world_opencv"]),
        )
        u, v = map(float, uv[0])
        in_frame = bool(
            depth[0] > 0.0
            and 0.0 <= u < float(width)
            and 0.0 <= v < float(height)
        )
        geom_ids = _object_geom_ids(env, object_name)
        mask = _segmentation_geom_mask(segmentation, geom_ids)
        visible_pixels = int(mask.sum())
        if visible_pixels:
            yy, xx = np.where(mask)
            bbox = [
                int(xx.min()),
                int(yy.min()),
                int(xx.max()),
                int(yy.max()),
            ]
            touches_border = bool(
                bbox[0] == 0
                or bbox[1] == 0
                or bbox[2] == width - 1
                or bbox[3] == height - 1
            )
        else:
            bbox = None
            touches_border = False
        visibility = _visibility_class(
            center_in_frame=in_frame,
            visible_pixels=visible_pixels,
            touches_border=touches_border,
            minimum_pixels=minimum_pixels,
        )
        result.update(
            {
                f"{label}_center_world": center[0].tolist(),
                f"{label}_center_uv_raw": [u, v],
                f"{label}_center_uv_policy": opencv_pixels_to_policy(
                    [[u, v]],
                    width=width,
                )[0].tolist(),
                f"{label}_center_depth": float(depth[0]),
                f"{label}_center_in_frame": in_frame,
                f"{label}_visible_pixels": visible_pixels,
                f"{label}_visible_fraction": visible_pixels / float(height * width),
                f"{label}_visible_bbox_raw": bbox,
                f"{label}_touches_border": touches_border,
                f"{label}_visibility": visibility,
            }
        )
        if detailed_geometry:
            guard = _isolated_guard_mask(
                env,
                camera_name=camera_name,
                geom_ids=geom_ids,
                height=height,
                width=width,
            )
            projected_pixels = int(guard["projected_pixels_in_frame"])
            occlusion_fraction = (
                1.0 - visible_pixels / projected_pixels
                if projected_pixels
                else 0.0
            )
            clipping_fraction = float(guard["fov_clipping_fraction"])
            if guard["projected_pixels_total"] == 0:
                fov_tier = "not_projected"
            elif clipping_fraction <= 0.01:
                fov_tier = "inside"
            elif clipping_fraction <= 0.10:
                fov_tier = "clipped_01_10"
            elif clipping_fraction <= 0.50:
                fov_tier = "clipped_10_50"
            elif clipping_fraction < 0.99:
                fov_tier = "clipped_over_50"
            else:
                fov_tier = "out_of_frame"
            result.update(
                {
                    f"{label}_projected_pixels_total": int(
                        guard["projected_pixels_total"]
                    ),
                    f"{label}_projected_pixels_in_frame": projected_pixels,
                    f"{label}_projected_bbox_raw": guard[
                        "projected_bbox_raw"
                    ],
                    f"{label}_projected_signed_border_margin_px": guard[
                        "projected_signed_border_margin_px"
                    ],
                    f"{label}_fov_clipping_fraction": clipping_fraction,
                    f"{label}_external_occlusion_fraction": float(
                        np.clip(occlusion_fraction, 0.0, 1.0)
                    ),
                    f"{label}_projected_patch_support": _patch_support(
                        guard["projected_mask"]
                    ),
                    f"{label}_visible_patch_support": _patch_support(mask),
                    f"{label}_fov_tier": fov_tier,
                    f"{label}_guard_factor": int(guard["guard_factor"]),
                    f"{label}_guard_truncated": bool(
                        guard["guard_truncated"]
                    ),
                }
            )
            geometrically_inside.append(clipping_fraction <= 0.10)
        center_visible.append(in_frame)
        rendered_visible.append(visible_pixels >= minimum_pixels)
        fully_visible.append(visibility == "full")

    eef_uv, eef_depth = project_world_points(
        np.asarray(observation["robot0_eef_pos"], dtype=np.float64),
        intrinsics=np.asarray(calibration["intrinsics"]),
        camera_to_world=np.asarray(calibration["camera_to_world_opencv"]),
    )
    eef_u, eef_v = map(float, eef_uv[0])
    eef_in_frame = bool(
        eef_depth[0] > 0.0
        and 0.0 <= eef_u < float(width)
        and 0.0 <= eef_v < float(height)
    )
    result.update(
        {
            "eef_center_uv_raw": [eef_u, eef_v],
            "eef_center_depth": float(eef_depth[0]),
            "eef_center_in_frame": eef_in_frame,
            "task_centers_in_frame": bool(all(center_visible)),
            "task_objects_visible": bool(all(rendered_visible)),
            "task_objects_fully_visible": bool(all(fully_visible)),
            **(
                {
                    "task_objects_geometrically_inside": bool(
                        all(geometrically_inside)
                    )
                }
                if detailed_geometry
                else {}
            ),
        }
    )
    return result


def orbit_pose(reference: Mapping[str, Any], pose: Mapping[str, Any]) -> dict[str, Any]:
    azimuth_delta = float(pose.get("azimuth_deg", 0.0))
    elevation_delta = float(pose.get("elevation_deg", 0.0))
    radius_scale = float(pose.get("radius_scale", 1.0))
    if not math.isfinite(azimuth_delta) or not math.isfinite(elevation_delta):
        raise ValueError("camera angle offsets must be finite")
    if not math.isfinite(radius_scale) or not 0.1 <= radius_scale <= 3.0:
        raise ValueError("camera radius_scale must be in [0.1, 3.0]")

    base_position = _as_vector(reference["position"], length=3, name="reference position")
    pivot = _as_vector(reference["pivot"], length=3, name="reference pivot")
    relative = base_position - pivot
    radius = float(np.linalg.norm(relative))
    azimuth = math.atan2(float(relative[1]), float(relative[0]))
    elevation = math.asin(float(relative[2]) / radius)
    new_elevation = elevation + math.radians(elevation_delta)
    if not math.radians(-85.0) < new_elevation < math.radians(85.0):
        raise ValueError("camera elevation leaves the supported range")
    new_azimuth = azimuth + math.radians(azimuth_delta)
    new_radius = radius * radius_scale
    position = pivot + new_radius * np.asarray(
        [
            math.cos(new_elevation) * math.cos(new_azimuth),
            math.cos(new_elevation) * math.sin(new_azimuth),
            math.sin(new_elevation),
        ],
        dtype=np.float64,
    )

    is_baseline = (
        abs(azimuth_delta) <= 1e-12
        and abs(elevation_delta) <= 1e-12
        and abs(radius_scale - 1.0) <= 1e-12
    )
    if is_baseline:
        quaternion = _as_vector(
            reference["quaternion"], length=4, name="reference quaternion"
        ).copy()
    else:
        quaternion = rotation_matrix_to_wxyz(look_at_rotation(position, pivot))
    return {
        "position": position,
        "quaternion": quaternion,
        "pivot": pivot.copy(),
        "azimuth_deg": azimuth_delta,
        "elevation_deg": elevation_delta,
        "radius_scale": radius_scale,
    }


def install_camera_pose(
    env: Any,
    reference: Mapping[str, Any],
    pose: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = orbit_pose(reference, pose)
    sim = env.env.sim
    camera_id = int(reference["camera_id"])
    # Always write the resolved pose. This matters when a baseline sample follows
    # a perturbed sample in an on-the-fly or replay-based camera sequence.
    sim.model.cam_pos[camera_id] = resolved["position"]
    sim.model.cam_quat[camera_id] = resolved["quaternion"]
    sim.forward()
    metadata = {
        "camera_name": str(reference["camera_name"]),
        "camera_position": resolved["position"].tolist(),
        "camera_quaternion_wxyz": resolved["quaternion"].tolist(),
        "camera_pivot": resolved["pivot"].tolist(),
        "camera_fovy": float(reference["fovy"]),
        "camera_azimuth_deg": resolved["azimuth_deg"],
        "camera_elevation_deg": resolved["elevation_deg"],
        "camera_radius_scale": resolved["radius_scale"],
    }
    return metadata


def set_camera_pose(
    env: Any,
    reference: Mapping[str, Any],
    pose: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    metadata = install_camera_pose(env, reference, pose)
    env.env._update_observables(force=True)
    observation = env.env._get_observations()
    return observation, metadata


def _axis_values(axis: Mapping[str, Any]) -> list[float]:
    if "values" in axis:
        values = [float(value) for value in axis["values"]]
    else:
        start = float(axis["start"])
        stop = float(axis["stop"])
        step = float(axis["step"])
        if not all(math.isfinite(value) for value in (start, stop, step)):
            raise ValueError("camera axis range must be finite")
        if step <= 0.0 or stop < start:
            raise ValueError("camera axis range requires stop >= start and step > 0")
        count = int(math.floor((stop - start) / step + 1e-9))
        values = [start + index * step for index in range(count + 1)]
        if not math.isclose(values[-1], stop, abs_tol=1e-8):
            values.append(stop)
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("camera axis values must be non-empty and finite")
    return values


def _pose_value_name(parameter: str, value: float) -> str:
    prefix = CAMERA_PARAMETER_PREFIXES[parameter]
    if parameter == "radius_scale":
        return f"{prefix}_{int(round(value * 1000)):04d}"
    sign = "m" if value < 0 else "p"
    magnitude = f"{abs(value):.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{prefix}_{sign}{magnitude}"


def _expand_one_factor_axes(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    axes = payload.get("one_factor_axes")
    if not isinstance(axes, list) or not axes:
        raise ValueError("one_factor_axes must be a non-empty list")
    poses: list[dict[str, Any]] = [
        {
            "name": "baseline",
            "azimuth_deg": 0.0,
            "elevation_deg": 0.0,
            "radius_scale": 1.0,
            "sweep_axis": "baseline",
            "sweep_value": 0.0,
        }
    ]
    for axis in axes:
        if not isinstance(axis, Mapping):
            raise ValueError("each one-factor axis must be an object")
        parameter = str(axis.get("parameter", ""))
        if parameter not in CAMERA_PARAMETERS:
            raise ValueError(f"unsupported camera sweep parameter: {parameter!r}")
        neutral = CAMERA_PARAMETER_NEUTRALS[parameter]
        for value in _axis_values(axis):
            if math.isclose(value, neutral, abs_tol=1e-10):
                continue
            pose = {
                "name": _pose_value_name(parameter, value),
                "azimuth_deg": 0.0,
                "elevation_deg": 0.0,
                "radius_scale": 1.0,
                "sweep_axis": parameter,
                "sweep_value": value,
            }
            pose[parameter] = value
            poses.append(pose)
    return poses


def load_camera_sweep_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("camera sweep config must use schema_version=1")
    camera_name = str(payload.get("camera_name", ""))
    if not camera_name:
        raise ValueError("camera_name is required")
    table_plane_z = float(payload.get("table_plane_z", float("nan")))
    if not math.isfinite(table_plane_z):
        raise ValueError("table_plane_z must be finite")
    poses = payload.get("poses")
    if poses is None and "one_factor_axes" in payload:
        poses = _expand_one_factor_axes(payload)
    if not isinstance(poses, list) or not poses:
        raise ValueError(
            "camera sweep config requires poses or one_factor_axes"
        )
    names = []
    normalized = []
    for value in poses:
        if not isinstance(value, dict):
            raise ValueError("each camera pose must be an object")
        name = str(value.get("name", ""))
        if not POSE_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"invalid camera pose name: {name!r}")
        orbit_pose(
            {
                # Validate relative offsets independently of a task-specific
                # baseline elevation. The real reference is checked again when
                # the pose is installed.
                "position": np.asarray([1.0, 0.0, 0.0]),
                "pivot": np.asarray([0.0, 0.0, 0.0]),
                "quaternion": np.asarray([1.0, 0.0, 0.0, 0.0]),
            },
            value,
        )
        names.append(name)
        normalized.append(
            {
                "name": name,
                "azimuth_deg": float(value.get("azimuth_deg", 0.0)),
                "elevation_deg": float(value.get("elevation_deg", 0.0)),
                "radius_scale": float(value.get("radius_scale", 1.0)),
                **(
                    {"sweep_axis": str(value["sweep_axis"])}
                    if "sweep_axis" in value
                    else {}
                ),
                **(
                    {"sweep_value": float(value["sweep_value"])}
                    if "sweep_value" in value
                    else {}
                ),
            }
        )
    if len(names) != len(set(names)):
        raise ValueError("camera pose names must be unique")
    if "baseline" not in names:
        raise ValueError("camera sweep config must include a baseline pose")
    return {
        **payload,
        "camera_name": camera_name,
        "table_plane_z": table_plane_z,
        "poses": normalized,
    }
