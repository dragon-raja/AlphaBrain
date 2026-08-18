from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


HALTON_BASES = (2, 3, 5)
POSE_KEYS = ("azimuth_deg", "elevation_deg", "radius_scale")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _halton(index: int, base: int) -> float:
    value = 0.0
    denominator = 1.0
    while index:
        index, remainder = divmod(index, base)
        denominator *= base
        value += remainder / denominator
    return value


def _validate_ranges(ranges: Mapping[str, Sequence[float]]) -> None:
    if set(ranges) != set(POSE_KEYS):
        raise ValueError(f"pose ranges must contain exactly {POSE_KEYS}")
    for name in POSE_KEYS:
        values = ranges[name]
        if len(values) != 2 or float(values[1]) <= float(values[0]):
            raise ValueError(f"invalid range for {name}: {values}")


def _halton_poses(
    *,
    count: int,
    skip: int,
    ranges: Mapping[str, Sequence[float]],
    prefix: str,
) -> list[dict[str, Any]]:
    _validate_ranges(ranges)
    poses = []
    for offset in range(count):
        values = []
        for base, name in zip(HALTON_BASES, POSE_KEYS):
            unit = _halton(skip + offset + 1, base)
            low, high = map(float, ranges[name])
            values.append(low + unit * (high - low))
        poses.append(
            {
                "pose_id": f"{prefix}_{offset:03d}",
                **dict(zip(POSE_KEYS, values)),
                "orientation_mode": "look_at_task_pivot",
            }
        )
    return poses


def _outside_box(
    pose: Mapping[str, float], ranges: Mapping[str, Sequence[float]]
) -> bool:
    return any(
        float(pose[name]) < float(ranges[name][0])
        or float(pose[name]) > float(ranges[name][1])
        for name in POSE_KEYS
    )


def _named_poses(rows: Iterable[Mapping[str, Any]], prefix: str) -> list[dict[str, Any]]:
    return [
        {
            "pose_id": f"{prefix}_{index:03d}",
            **row,
            "orientation_mode": "look_at_task_pivot",
        }
        for index, row in enumerate(rows)
    ]


def build(rules: Mapping[str, Any]) -> dict[str, Any]:
    broad_rules = rules["broad_training"]
    broad_sizes = sorted(int(value) for value in broad_rules["sizes"])
    broad_all = _halton_poses(
        count=max(broad_sizes),
        skip=int(broad_rules["skip"]),
        ranges=broad_rules["ranges"],
        prefix="broad_train",
    )
    heldout_rules = rules["broad_heldout"]
    heldout = _halton_poses(
        count=int(heldout_rules["size"]),
        skip=int(heldout_rules["skip"]),
        ranges=heldout_rules["ranges"],
        prefix="broad_heldout",
    )
    extrapolation_rules = rules["wide_extrapolation"]
    extrapolation_candidates = _halton_poses(
        count=int(extrapolation_rules["size"]) * 8,
        skip=int(extrapolation_rules["skip"]),
        ranges=extrapolation_rules["ranges"],
        prefix="wide_extrapolation_candidate",
    )
    extrapolation = [
        pose
        for pose in extrapolation_candidates
        if _outside_box(pose, broad_rules["ranges"])
    ][: int(extrapolation_rules["size"])]
    for index, pose in enumerate(extrapolation):
        pose["pose_id"] = f"wide_extrapolation_{index:03d}"
    if len(extrapolation) != int(extrapolation_rules["size"]):
        raise RuntimeError("could not generate enough extrapolation poses")

    training_sets = {
        f"broad_{size}": [pose["pose_id"] for pose in broad_all[:size]]
        for size in broad_sizes
    }
    if not set(training_sets["broad_32"]).issubset(training_sets["broad_64"]):
        raise RuntimeError("Broad-32 must be an exact subset of Broad-64")
    training_ids = set(training_sets["broad_64"])
    if training_ids.intersection(pose["pose_id"] for pose in heldout):
        raise RuntimeError("training and held-out pose IDs overlap")

    look_away = []
    for index, row in enumerate(rules["diagnostic_look_away"]):
        look_away.append(
            {
                "pose_id": f"look_away_{index:03d}",
                **row,
                "radius_scale": 1.0,
                "orientation_mode": "relative_look_away",
            }
        )

    catalog = {
        "schema": "dsol_libero_view_catalog_v2",
        "status": "CANDIDATE_PENDING_RENDER_VISIBILITY_AUDIT",
        "source_rules_schema": rules["schema"],
        "seed": int(rules["seed"]),
        "camera_name": rules["camera_name"],
        "table_plane_z": float(rules["table_plane_z"]),
        "canonical": [
            {
                "pose_id": "canonical",
                "azimuth_deg": 0.0,
                "elevation_deg": 0.0,
                "radius_scale": 1.0,
                "orientation_mode": "original_camera_pose",
            }
        ],
        "narrow_legacy_8": _named_poses(
            rules["narrow_legacy_8"], "narrow_legacy"
        ),
        "broad_training_64": broad_all,
        "broad_training_sets": training_sets,
        "broad_heldout_32": heldout,
        "wide_extrapolation_24": extrapolation,
        "diagnostic_extreme_orbit": _named_poses(
            rules["diagnostic_extreme_orbit"], "extreme_orbit"
        ),
        "diagnostic_look_away": look_away,
        "sensor_controls": list(rules["sensor_controls"]),
        "information_view_definition": rules["information_view_definition"],
        "training_exclusions": rules["exclusions"],
        "audit_requirements": [
            "task_region_visibility",
            "object_pixel_area",
            "image_border_contact",
            "camera_collision",
            "train_test_nearest_pose_distance",
            "per_task_acceptance_rate",
        ],
    }
    if "diagnostic_crossed_orbit" in rules:
        catalog["diagnostic_crossed_orbit"] = _named_poses(
            rules["diagnostic_crossed_orbit"], "crossed_orbit"
        )
    return catalog


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the nested LIBERO view catalog.")
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    rules = json.loads(args.rules.read_text(encoding="utf-8"))
    catalog = build(rules)
    output = args.output.resolve()
    _atomic_json(output, catalog)
    checksum = _sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{checksum}  {output.name}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": catalog["status"],
                "broad_training": len(catalog["broad_training_64"]),
                "broad_heldout": len(catalog["broad_heldout_32"]),
                "wide_extrapolation": len(catalog["wide_extrapolation_24"]),
                "output": str(output),
                "sha256": checksum,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
