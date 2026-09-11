from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


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


def _load_collection(root: Path) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "dsol_libero_hdf5_view_pair_collection_v1":
        raise ValueError(f"unsupported collection manifest: {root}")
    if manifest.get("status") != "VERIFIED":
        raise ValueError(f"collection is not verified: {root}")
    rows = []
    for shard in manifest["shards"]:
        shard_root = root / shard["path"]
        shard_manifest = json.loads(
            (shard_root / "manifest.json").read_text(encoding="utf-8")
        )
        for line in (shard_root / shard_manifest["records"]).read_text().splitlines():
            if line.strip():
                rows.append(
                    {
                        **json.loads(line),
                        "shard_path": str(shard_root / shard_manifest["shard"]),
                    }
                )
    return manifest, sorted(rows, key=lambda row: row["sample_id"])


def audit(info_root: Path, control_root: Path) -> dict[str, Any]:
    from libero_pair_records import read_record

    info_manifest, info_rows = _load_collection(info_root)
    control_manifest, control_rows = _load_collection(control_root)
    if len(info_rows) != len(control_rows):
        raise ValueError("collection record counts differ")

    exact_image_names = ("canonical", "broad_b", "wrist")
    handles: dict[str, Any] = {}

    def read(row: Mapping[str, Any]) -> Mapping[str, Any]:
        shard_path = str(row["shard_path"])
        handle = handles.get(shard_path)
        if handle is None:
            handle = Path(shard_path).open("rb")
            handles[shard_path] = handle
        return read_record(
            handle,
            offset=int(row["offset"]),
            image_names=("canonical", "broad_a", "broad_b", "wrist"),
        )

    mismatches = []
    task_view_images_different = 0
    image_exact_mismatch_counts = {name: 0 for name in exact_image_names}
    image_tolerated_mismatch_counts = {name: 0 for name in exact_image_names}
    maximum_image_absolute_difference = {name: 0 for name in exact_image_names}
    image_max_abs_tolerance = 1
    image_changed_value_fraction_tolerance = 0.001
    split_counts = {"train": 0, "val": 0, "test": 0}
    try:
        for info_row, control_row in zip(info_rows, control_rows):
            sample_id = str(info_row["sample_id"])
            if sample_id != str(control_row["sample_id"]):
                mismatches.append({"sample_id": sample_id, "field": "sample_id"})
                continue
            info_record = read(info_row)
            control_record = read(control_row)
            info_header = info_record["header"]
            control_header = control_record["header"]
            image_checks = {}
            for name in exact_image_names:
                left = info_record["images"][name]
                right = control_record["images"][name]
                delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
                maximum = int(delta.max())
                changed_fraction = float(np.count_nonzero(delta) / delta.size)
                exact = bool(maximum == 0)
                if not exact:
                    image_exact_mismatch_counts[name] += 1
                tolerated = bool(
                    maximum <= image_max_abs_tolerance
                    and changed_fraction <= image_changed_value_fraction_tolerance
                )
                if not tolerated:
                    image_tolerated_mismatch_counts[name] += 1
                maximum_image_absolute_difference[name] = max(
                    maximum_image_absolute_difference[name], maximum
                )
                image_checks[f"image_{name}_within_tolerance"] = tolerated
            checks = {
                "split": info_row["split"] == control_row["split"],
                "source_state_sha256": (
                    info_header["source_state_sha256"]
                    == control_header["source_state_sha256"]
                ),
                "action_chunk": info_header["action_chunk"] == control_header["action_chunk"],
                "robot_state": info_header["robot_state"] == control_header["robot_state"],
                "pose_b_id": info_row["pose_b_id"] == control_row["pose_b_id"],
                "pose_b": info_header["pose_b"] == control_header["pose_b"],
                "scene_construction_disabled": (
                    info_header["task_view_support"]["scene_construction_applied"]
                    is False
                    and control_header["task_view_support"][
                        "scene_construction_applied"
                    ]
                    is False
                ),
                **image_checks,
            }
            failed = [field for field, passed in checks.items() if not passed]
            if failed:
                mismatches.append({"sample_id": sample_id, "fields": failed})
            if not np.array_equal(
                info_record["images"]["broad_a"],
                control_record["images"]["broad_a"],
            ):
                task_view_images_different += 1
            split_counts[str(info_row["split"])] += 1
    finally:
        for handle in handles.values():
            handle.close()

    result = {
        "schema": "dsol_taskcentric_support_pair_audit_v1",
        "status": "PASS" if not mismatches else "FAIL",
        "info_root": str(info_root.resolve()),
        "control_root": str(control_root.resolve()),
        "info_plan_sha256": info_manifest["plan_sha256"],
        "control_plan_sha256": control_manifest["plan_sha256"],
        "record_count": len(info_rows),
        "split_counts": split_counts,
        "exact_fields": [
            "sample_id",
            "split",
            "source_state_sha256",
            "action_chunk",
            "robot_state",
            "pose_b_id",
            "pose_b",
            "image_canonical_within_tolerance",
            "image_broad_b_within_tolerance",
            "image_wrist_within_tolerance",
        ],
        "image_tolerance": {
            "maximum_absolute_uint8_difference": image_max_abs_tolerance,
            "maximum_changed_value_fraction": image_changed_value_fraction_tolerance,
        },
        "image_exact_mismatch_counts": image_exact_mismatch_counts,
        "image_outside_tolerance_counts": image_tolerated_mismatch_counts,
        "maximum_image_absolute_difference": maximum_image_absolute_difference,
        "task_view_images_different": task_view_images_different,
        "task_view_images_different_fraction": (
            task_view_images_different / len(info_rows) if info_rows else 0.0
        ),
        "scene_construction_applied": False,
        "mismatch_count": len(mismatches),
        "mismatches_first_20": mismatches[:20],
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["content_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit matched task-centric information/control collections."
    )
    parser.add_argument("--info-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = audit(args.info_root, args.control_root)
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "record_count": result["record_count"],
                "mismatch_count": result["mismatch_count"],
                "task_view_images_different": result["task_view_images_different"],
            },
            sort_keys=True,
        )
    )
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
