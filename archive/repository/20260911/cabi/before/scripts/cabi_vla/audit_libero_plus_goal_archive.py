from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import zipfile
from collections import Counter
from collections.abc import Mapping
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def archive_inventory(archive: Path) -> dict[str, Any]:
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        unsafe = [member.filename for member in members if not _safe_member(member.filename)]
        if unsafe:
            raise ValueError(f"archive contains unsafe members: {unsafe[:3]}")
        files = [member for member in members if not member.is_dir()]
        tfrecords = sorted(
            member.filename
            for member in files
            if ".tfrecord-" in member.filename
        )
        metadata = sorted(
            member.filename
            for member in files
            if PurePosixPath(member.filename).name
            in {"dataset_info.json", "features.json", "checksums.tsv"}
        )
        suffixes = Counter(PurePosixPath(member.filename).suffix for member in files)
        return {
            "member_count": len(members),
            "file_count": len(files),
            "compressed_bytes": int(sum(member.compress_size for member in files)),
            "uncompressed_bytes": int(sum(member.file_size for member in files)),
            "tfrecord_shard_count": len(tfrecords),
            "tfrecord_members": tfrecords,
            "metadata_members": metadata,
            "suffix_counts": dict(sorted(suffixes.items())),
        }


def _extract_member(handle: zipfile.ZipFile, name: str, target_root: Path) -> Path:
    relative = PurePosixPath(name)
    if not _safe_member(name):
        raise ValueError(f"unsafe archive member: {name}")
    target = target_root.joinpath(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    with handle.open(name) as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
    return target


def _feature_summary(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    summary = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "size": int(array.size),
    }
    if array.dtype.kind in "SUO":
        flattened = array.reshape(-1)
        summary["byte_element_count"] = int(len(flattened))
        summary["total_bytes"] = int(
            sum(len(bytes(item)) for item in flattened if isinstance(item, (bytes, np.bytes_)))
        )
    elif array.size and np.issubdtype(array.dtype, np.number):
        numeric = np.asarray(array, dtype=np.float64)
        summary["finite"] = bool(np.all(np.isfinite(numeric)))
    return summary


def _scalar_text(value: Any) -> str | None:
    array = np.asarray(value)
    if array.size != 1:
        return None
    item = array.reshape(()).item()
    if isinstance(item, (bytes, np.bytes_)):
        return bytes(item).decode("utf-8", errors="replace")
    if isinstance(item, str):
        return item
    return None


def source_factor_class(value: str) -> str:
    lowered = value.lower()
    has_camera = any(token in lowered for token in ("camera", "extrinsic", "_view_"))
    has_background = any(token in lowered for token in ("background", "texture", "_tb_"))
    has_background = has_background or bool(re.search(r"_table_\d+", lowered))
    if has_camera and has_background:
        return "camera_background"
    if has_camera:
        return "camera"
    if has_background:
        return "background"
    return "other"


def source_factor_hint(value: str) -> dict[str, Any]:
    path = PurePosixPath(value)
    basename = path.name
    background_match = re.search(
        r"_(?:table|tb)_(\d+)(?:_demo)?(?:\.hdf5)?$",
        basename,
        flags=re.IGNORECASE,
    )
    view_match = re.search(
        r"_view_([^/]+?)(?:_initstate_\d+)?(?:_demo)?(?:\.hdf5)?$",
        basename,
        flags=re.IGNORECASE,
    )
    return {
        "factor_class": source_factor_class(value),
        "parent": path.parent.name,
        "grandparent": path.parent.parent.name,
        "basename": basename,
        "background_id": int(background_match.group(1)) if background_match else None,
        "view_id": view_match.group(1) if view_match else None,
    }


def inspect_record_sample(path: Path, *, record_limit: int = 512) -> dict[str, Any]:
    try:
        from tfrecord.reader import tfrecord_loader
    except ImportError as error:
        raise RuntimeError("the isolated TFRecord reader environment is required") from error
    records = list(islice(tfrecord_loader(str(path), None), record_limit))
    if not records:
        raise ValueError(f"TFRecord shard is empty: {path}")
    features = {
        key: _feature_summary(value) for key, value in sorted(records[0].items())
    }
    keys = set().union(*(record.keys() for record in records))
    camera_fields = sorted(
        key
        for key in keys
        if any(token in key.lower() for token in ("camera", "extrinsic", "intrinsic"))
    )
    background_fields = sorted(
        key
        for key in keys
        if any(token in key.lower() for token in ("background", "texture", "floor", "wall"))
    )
    source_fields = sorted(
        key for key in keys if "file_path" in key.lower() or "source" in key.lower()
    )
    source_hints = []
    for record in records:
        for field in source_fields:
            if field not in record:
                continue
            value = _scalar_text(record[field])
            if value is not None:
                source_hints.append(source_factor_hint(value))
                break
    factor_classes = Counter(hint["factor_class"] for hint in source_hints)
    parents = Counter(hint["parent"] for hint in source_hints)
    grandparents = Counter(hint["grandparent"] for hint in source_hints)
    background_ids = sorted(
        {int(hint["background_id"]) for hint in source_hints if hint["background_id"] is not None}
    )
    view_ids = sorted(
        {str(hint["view_id"]) for hint in source_hints if hint["view_id"] is not None}
    )
    camera_signatures = set()
    for record in records:
        for field in camera_fields:
            if field not in record:
                continue
            array = np.asarray(record[field])
            if array.size in {9, 12, 16} and np.issubdtype(array.dtype, np.number):
                camera_signatures.add(tuple(np.round(array.astype(np.float64), 4).reshape(-1)))
    return {
        "sampled_episode_count": len(records),
        "record_limit": record_limit,
        "feature_count": len(features),
        "features": features,
        "direct_camera_factor_fields": camera_fields,
        "direct_background_factor_fields": background_fields,
        "source_identity_fields": source_fields,
        "unique_direct_camera_signatures": len(camera_signatures),
        "source_factor_class_counts": dict(sorted(factor_classes.items())),
        "source_parent_counts": dict(sorted(parents.items())),
        "source_grandparent_counts": dict(sorted(grandparents.items())),
        "source_background_ids": background_ids,
        "source_view_id_count": len(view_ids),
        "source_view_id_examples": view_ids[:20],
        "source_basename_examples": sorted({hint["basename"] for hint in source_hints})[:20],
    }


def _read_metadata(handle: zipfile.ZipFile, names: list[str]) -> dict[str, Any]:
    result = {}
    for name in names:
        basename = PurePosixPath(name).name
        if basename.endswith(".json"):
            payload = json.loads(handle.read(name))
            result[name] = payload
        else:
            result[name] = {"bytes": handle.getinfo(name).file_size}
    return result


def audit_archive(*, archive: Path, sample_root: Path) -> dict[str, Any]:
    inventory = archive_inventory(archive)
    tfrecords = inventory.pop("tfrecord_members")
    if not tfrecords:
        raise ValueError("archive contains no TFRecord shards")
    sample_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        bad_member = handle.testzip()
        if bad_member is not None:
            raise ValueError(f"CRC failure in archive member: {bad_member}")
        metadata = _read_metadata(handle, inventory["metadata_members"])
        sample_path = _extract_member(handle, tfrecords[0], sample_root)
    record = inspect_record_sample(sample_path)
    camera_direct = bool(record["direct_camera_factor_fields"])
    background_direct = bool(record["direct_background_factor_fields"])
    source_available = bool(record["source_identity_fields"])
    if camera_direct and background_direct:
        gate = "DIRECT_FACTOR_FIELDS_AVAILABLE"
    elif source_available:
        gate = "FACTOR_RECOVERY_REQUIRED_FROM_SOURCE_IDENTITY"
    else:
        gate = "STRICT_COMPOSITION_DATA_INVALID"
    return {
        "schema_version": 1,
        "study": "libero_plus_goal_archive_factor_audit",
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "inventory": inventory,
        "metadata": metadata,
        "sample": {
            "member": tfrecords[0],
            "extracted_path": str(sample_path),
            **record,
        },
        "strict_composition_gate": {
            "decision": gate,
            "camera_factor_direct": camera_direct,
            "background_factor_direct": background_direct,
            "source_identity_available": source_available,
            "training_started": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a LIBERO-Plus suite archive")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.output}")
    report = audit_archive(archive=args.archive, sample_root=args.sample_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision": report["strict_composition_gate"]["decision"],
                "tfrecord_shards": report["inventory"]["tfrecord_shard_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
