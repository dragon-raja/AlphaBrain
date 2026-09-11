#!/usr/bin/env python3
"""Package one rendered constructed state into the strict M0 audit schema."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

try:
    from scripts.dsol_paper1.constructed_blind_reveal import (
        build_snapshot_identity,
        masked_visibility,
        recompute_equal_weight_visibility,
        sha256_file,
    )
except ModuleNotFoundError:
    from constructed_blind_reveal import (
        build_snapshot_identity,
        masked_visibility,
        recompute_equal_weight_visibility,
        sha256_file,
    )


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


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _index_source_records(scan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = {}
    for record in scan["records"]:
        pose_id = str(record["pose_id"])
        if pose_id in records:
            raise ValueError(f"duplicate source pose_id: {pose_id}")
        records[pose_id] = record
    return records


def _condition_visibility(
    *,
    source: Mapping[str, Any],
    canonical_visibility: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    if source.get("visibility") is not None:
        return json.loads(json.dumps(source["visibility"]))
    if role == "all_camera_blackout":
        return masked_visibility(
            canonical_visibility,
            masked_cameras=canonical_visibility["camera_names"],
        )
    if role == "external_blackout":
        return masked_visibility(
            canonical_visibility,
            masked_cameras=(canonical_visibility["camera_names"][0],),
        )
    if role == "wrist_blackout":
        return masked_visibility(
            canonical_visibility,
            masked_cameras=(canonical_visibility["camera_names"][-1],),
        )
    raise ValueError(
        f"source record {source.get('pose_id')} has no visibility for role {role}"
    )


def _manual_audit(base: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    manual = dict(manifest.get("manual_visual_audit", {"status": "PENDING"}))
    if manual.get("status") != "PASS":
        return manual
    if not manual.get("montage_path"):
        raise ValueError("PASS manual audit requires montage_path")
    montage = _resolve(base, str(manual["montage_path"])).resolve()
    if not montage.is_file():
        raise FileNotFoundError(f"manual audit montage does not exist: {montage}")
    actual_sha256 = sha256_file(montage)
    expected_sha256 = manual.get("montage_sha256")
    if expected_sha256 is not None and expected_sha256 != actual_sha256:
        raise ValueError("manual audit montage SHA-256 mismatch")
    manual["montage_path"] = str(montage)
    manual["montage_sha256"] = actual_sha256
    return manual


def package(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    base = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "dsol_constructed_blind_reveal_package_v1":
        raise ValueError("unexpected package manifest schema")

    source_scan_path = _resolve(base, manifest["source_scan"]).resolve()
    source_scan = json.loads(source_scan_path.read_text(encoding="utf-8"))
    if source_scan.get("status") != "PASS":
        raise ValueError("source scan must have PASS status")
    source_records = _index_source_records(source_scan)
    canonical_source = source_records.get("canonical")
    if canonical_source is None or canonical_source.get("visibility") is None:
        raise ValueError("source scan must contain canonical visibility")

    components = {
        str(name): _resolve(base, str(value))
        for name, value in manifest["snapshot_components"].items()
    }
    identity = build_snapshot_identity(
        task_id=str(manifest["task_id"]), components=components
    )
    canonical_score = recompute_equal_weight_visibility(
        canonical_source["visibility"]
    )["score"]

    records = []
    for condition in manifest["conditions"]:
        pose_id = str(condition["source_pose_id"])
        if pose_id not in source_records:
            raise KeyError(f"source scan has no pose_id {pose_id!r}")
        source = source_records[pose_id]
        role = str(condition["condition_role"])
        visibility = _condition_visibility(
            source=source,
            canonical_visibility=canonical_source["visibility"],
            role=role,
        )
        recomputed = recompute_equal_weight_visibility(visibility)
        records.append(
            {
                "condition_id": str(condition.get("condition_id", pose_id)),
                "condition_role": role,
                "source_pose_id": pose_id,
                "source_group": source.get("group"),
                "snapshot_sha256": identity["snapshot_sha256"],
                "evaluation_only": bool(condition.get("evaluation_only", False)),
                "training_eligible": bool(condition.get("training_eligible", False)),
                "operational": bool(condition.get("operational", True)),
                "is_extreme": bool(condition.get("is_extreme", False)),
                "visibility_score": recomputed["score"],
                "delta_visibility": recomputed["score"] - canonical_score,
                "per_camera_scores": recomputed["per_camera_scores"],
                "visibility": visibility,
                "camera": source.get("camera"),
                "camera_displacement_from_canonical": source.get(
                    "camera_displacement_from_canonical",
                    {"translation_m": 0.0, "rotation_geodesic_deg": 0.0},
                ),
                "image_artifacts": condition.get("image_artifacts", {}),
            }
        )

    split = str(manifest["split"])
    if split != "train" and any(record["training_eligible"] for record in records):
        raise ValueError("validation/test conditions cannot be training eligible")
    output = {
        "schema": "dsol_constructed_blind_reveal_scan_v1",
        "status": "PACKAGED_UNAUDITED",
        "snapshot_group_id": str(manifest["snapshot_group_id"]),
        "task_id": str(manifest["task_id"]),
        "split": split,
        "scene_variant_id": str(manifest["scene_variant_id"]),
        "source_episode_id": str(
            source_scan.get(
                "episode_id",
                (
                    f"{source_scan.get('suite', 'unknown')}::"
                    f"{Path(source_scan.get('hdf5', 'unknown')).stem}::"
                    f"{source_scan.get('demo', 'unknown')}"
                ),
            )
        ),
        "source_demo": source_scan.get("demo"),
        "source_frame": source_scan.get("frame"),
        "snapshot": identity,
        "task_entities": list(canonical_source["visibility"]["entity_names"]),
        "camera_names": list(canonical_source["visibility"]["camera_names"]),
        "visibility_definition": canonical_source["visibility"]["definition"],
        "source_scan": {
            "path": str(source_scan_path),
            "sha256": sha256_file(source_scan_path),
        },
        "manual_visual_audit": _manual_audit(base, manifest),
        "records": records,
    }
    return output


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = package(args.manifest)
    _atomic_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "snapshot_group_id": result["snapshot_group_id"],
                "snapshot_sha256": result["snapshot"]["snapshot_sha256"],
                "condition_count": len(result["records"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
