from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from collections import Counter
from collections.abc import Mapping
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


def inspect_first_record(path: Path) -> dict[str, Any]:
    try:
        from tfrecord.reader import tfrecord_loader
    except ImportError as error:
        raise RuntimeError("the isolated TFRecord reader environment is required") from error
    try:
        record = next(iter(tfrecord_loader(str(path), None)))
    except StopIteration as error:
        raise ValueError(f"TFRecord shard is empty: {path}") from error
    features = {key: _feature_summary(value) for key, value in sorted(record.items())}
    keys = set(features)
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
    return {
        "feature_count": len(features),
        "features": features,
        "direct_camera_factor_fields": camera_fields,
        "direct_background_factor_fields": background_fields,
        "source_identity_fields": source_fields,
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
    record = inspect_first_record(sample_path)
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
