from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


VISIBILITY_DEFINITION = (
    "equal_mean_visible_pixel_fraction_over_entities_and_cameras"
)
REQUIRED_ROLES = (
    "canonical",
    "strong_info",
    "matched_control",
    "blind",
    "look_away",
    "all_camera_blackout",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_snapshot_identity(
    *, task_id: str, components: Mapping[str, Path]
) -> dict[str, Any]:
    """Bind a restored simulator snapshot to all files needed to recreate it."""

    if not task_id:
        raise ValueError("task_id must not be empty")
    if not components:
        raise ValueError("snapshot components must not be empty")

    records = {}
    for name, path in sorted(components.items()):
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"snapshot component is not a file: {resolved}")
        records[str(name)] = {
            "path": str(resolved),
            "size_bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }

    canonical = {
        "task_id": task_id,
        "components": {
            name: {
                "size_bytes": record["size_bytes"],
                "sha256": record["sha256"],
            }
            for name, record in records.items()
        },
    }
    snapshot_sha256 = hashlib.sha256(
        json.dumps(
            canonical, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "dsol_constructed_snapshot_identity_v1",
        "task_id": task_id,
        "snapshot_sha256": snapshot_sha256,
        "components": records,
    }


def recompute_equal_weight_visibility(
    visibility: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute I_task without trusting cached aggregate scores."""

    if visibility.get("definition") != VISIBILITY_DEFINITION:
        raise ValueError("unexpected visibility definition")
    entities = tuple(str(value) for value in visibility.get("entity_names", ()))
    cameras = tuple(str(value) for value in visibility.get("camera_names", ()))
    if not entities or not cameras:
        raise ValueError("visibility must contain nonempty entity and camera lists")
    height = int(visibility.get("height", 0))
    width = int(visibility.get("width", 0))
    if height <= 0 or width <= 0:
        raise ValueError("visibility height and width must be positive")

    fractions = []
    per_camera = {}
    raw_cameras = visibility.get("per_camera", {})
    for camera in cameras:
        if camera not in raw_cameras:
            raise ValueError(f"missing camera visibility: {camera}")
        raw_entities = raw_cameras[camera].get("entities", {})
        camera_values = []
        for entity in entities:
            if entity not in raw_entities:
                raise ValueError(
                    f"missing entity visibility: camera={camera} entity={entity}"
                )
            value = float(raw_entities[entity]["visible_fraction"])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("visible fractions must be finite and in [0, 1]")
            visible_pixels = int(raw_entities[entity]["visible_pixels"])
            if visible_pixels < 0 or visible_pixels > height * width:
                raise ValueError("visible pixel counts must fit the image area")
            expected_fraction = visible_pixels / float(height * width)
            if not math.isclose(value, expected_fraction, abs_tol=1e-12):
                raise ValueError(
                    "visible_fraction does not match visible_pixels / image_area"
                )
            camera_values.append(value)
            fractions.append(value)
        per_camera[camera] = sum(camera_values) / len(camera_values)

    return {
        "definition": VISIBILITY_DEFINITION,
        "score": sum(fractions) / len(fractions),
        "per_camera_scores": per_camera,
        "entity_count": len(entities),
        "camera_count": len(cameras),
    }


def masked_visibility(
    visibility: Mapping[str, Any], *, masked_cameras: Sequence[str]
) -> dict[str, Any]:
    """Create an auditable sensor-blackout visibility record."""

    masked = set(masked_cameras)
    entities = tuple(str(value) for value in visibility["entity_names"])
    cameras = tuple(str(value) for value in visibility["camera_names"])
    unknown = masked.difference(cameras)
    if unknown:
        raise ValueError(f"cannot mask unknown cameras: {sorted(unknown)}")

    result = json.loads(json.dumps(visibility))
    for camera in masked:
        camera_record = result["per_camera"][camera]
        camera_record["score"] = 0.0
        for entity in entities:
            entity_record = camera_record["entities"][entity]
            entity_record["visible_pixels"] = 0
            entity_record["visible_fraction"] = 0.0
            entity_record["touches_border"] = False

    recomputed = recompute_equal_weight_visibility(result)
    result["score"] = recomputed["score"]
    for camera, score in recomputed["per_camera_scores"].items():
        result["per_camera"][camera]["score"] = score
    return result


def is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def check_record_visibility(
    record: Mapping[str, Any], *, tolerance: float
) -> dict[str, Any]:
    recomputed = recompute_equal_weight_visibility(record["visibility"])
    stored_score = float(record["visibility_score"])
    cached_score = float(record["visibility"]["score"])
    per_camera = {
        str(name): float(value)
        for name, value in record.get("per_camera_scores", {}).items()
    }
    camera_errors = {
        name: max(
            abs(per_camera.get(name, math.inf) - score),
            abs(float(record["visibility"]["per_camera"][name]["score"]) - score),
        )
        for name, score in recomputed["per_camera_scores"].items()
    }
    errors = {
        "record_score": abs(stored_score - recomputed["score"]),
        "cached_score": abs(cached_score - recomputed["score"]),
        "per_camera_max": max(camera_errors.values(), default=0.0),
    }
    return {
        "passed": all(value <= tolerance for value in errors.values()),
        "recomputed_score": recomputed["score"],
        "recomputed_per_camera_scores": recomputed["per_camera_scores"],
        "absolute_errors": errors,
    }


def role_index(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        role = str(record["condition_role"])
        if role in indexed:
            raise ValueError(f"duplicate condition role: {role}")
        indexed[role] = record
    return indexed
