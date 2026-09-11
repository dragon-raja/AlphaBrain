from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


_REGION_QUALIFIERS = frozenset(
    {
        "back",
        "bottom",
        "contain",
        "front",
        "heating",
        "init",
        "inside",
        "left",
        "middle",
        "right",
        "side",
        "top",
    }
)


def _entity_geom_ids(env: Any, entity_name: str) -> tuple[np.ndarray, str]:
    names = env.env.sim.model.geom_names
    candidates = [entity_name]
    if entity_name.endswith("_region"):
        parts = entity_name[: -len("_region")].split("_")
        while parts and parts[-1] in _REGION_QUALIFIERS:
            parts.pop()
            candidates.append("_".join(parts))
    for candidate in candidates:
        ids = [
            index
            for index, name in enumerate(names)
            if name and (name == candidate or name.startswith(f"{candidate}_"))
        ]
        if ids:
            return np.asarray(ids, dtype=np.int32), candidate
    raise KeyError(
        f"no MuJoCo geoms found for task entity {entity_name!r}; "
        f"tried {candidates!r}"
    )


def _segmentation_mask(segmentation: np.ndarray, geom_ids: np.ndarray) -> np.ndarray:
    import mujoco

    return (
        segmentation[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM)
    ) & np.isin(segmentation[..., 1], geom_ids)


def task_entity_visibility(
    env: Any,
    *,
    entity_names: Iterable[str],
    camera_names: Iterable[str],
    height: int,
    width: int,
) -> dict[str, Any]:
    """Compute equal-weight task-entity visibility from instance segmentation."""

    entities = tuple(dict.fromkeys(str(name) for name in entity_names))
    cameras = tuple(dict.fromkeys(str(name) for name in camera_names))
    if not entities:
        raise ValueError("entity_names must not be empty")
    if not cameras:
        raise ValueError("camera_names must not be empty")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    resolved_geoms = {name: _entity_geom_ids(env, name) for name in entities}
    per_camera: dict[str, Any] = {}
    all_fractions = []
    for camera_name in cameras:
        segmentation = np.asarray(
            env.env.sim.render(
                camera_name=camera_name,
                height=height,
                width=width,
                segmentation=True,
            )
        )
        if segmentation.shape != (height, width, 2):
            raise ValueError(
                f"unexpected segmentation shape for {camera_name}: "
                f"{segmentation.shape}"
            )
        entity_records = {}
        camera_fractions = []
        for entity_name in entities:
            geom_ids, geom_source = resolved_geoms[entity_name]
            mask = _segmentation_mask(segmentation, geom_ids)
            visible_pixels = int(mask.sum())
            visible_fraction = visible_pixels / float(height * width)
            camera_fractions.append(visible_fraction)
            all_fractions.append(visible_fraction)
            entity_records[entity_name] = {
                "visible_pixels": visible_pixels,
                "visible_fraction": visible_fraction,
                "geom_source": geom_source,
                "touches_border": bool(
                    mask[0].any()
                    or mask[-1].any()
                    or mask[:, 0].any()
                    or mask[:, -1].any()
                ),
            }
        per_camera[camera_name] = {
            "score": float(np.mean(camera_fractions)),
            "entities": entity_records,
        }

    return {
        "definition": "equal_mean_visible_pixel_fraction_over_entities_and_cameras",
        "height": int(height),
        "width": int(width),
        "entity_names": list(entities),
        "entity_geom_sources": {
            name: source for name, (_, source) in resolved_geoms.items()
        },
        "camera_names": list(cameras),
        "score": float(np.mean(all_fractions)),
        "per_camera": per_camera,
    }
