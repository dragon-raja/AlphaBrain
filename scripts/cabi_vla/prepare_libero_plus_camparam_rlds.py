from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

from resume_aria2_ranges import sha256


SHARD_PATTERN = re.compile(
    r"^libero_mix/1\.0\.0/libero_mix-train\.tfrecord-(\d{5})-of-(\d{5})$"
)
METADATA_MEMBERS = {
    "libero_mix/1.0.0/dataset_info.json",
    "libero_mix/1.0.0/features.json",
}


def inspect_archive(archive: Path, *, expected_shards: int = 256) -> dict[str, Any]:
    if expected_shards <= 0:
        raise ValueError("expected shard count must be positive")
    shards: dict[int, str] = {}
    file_count = 0
    uncompressed_bytes = 0
    members: set[str] = set()
    with ZipFile(archive) as bundle:
        infos = bundle.infolist()
        for info in infos:
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe zip member: {info.filename}")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise ValueError(f"zip symlink is not allowed: {info.filename}")
            if info.is_dir():
                continue
            file_count += 1
            uncompressed_bytes += int(info.file_size)
            members.add(info.filename)
            match = SHARD_PATTERN.fullmatch(info.filename)
            if match:
                index, total = map(int, match.groups())
                if total != expected_shards:
                    raise ValueError(
                        f"shard {info.filename} declares total {total}, "
                        f"expected {expected_shards}"
                    )
                if index in shards:
                    raise ValueError(f"duplicate TFRecord shard index: {index}")
                shards[index] = info.filename
    missing_metadata = sorted(METADATA_MEMBERS - members)
    if missing_metadata:
        raise ValueError(f"archive is missing metadata: {missing_metadata}")
    expected_indices = set(range(expected_shards))
    if set(shards) != expected_indices:
        missing = sorted(expected_indices - set(shards))
        extra = sorted(set(shards) - expected_indices)
        raise ValueError(f"invalid shard set: missing={missing}, extra={extra}")
    allowed = METADATA_MEMBERS | set(shards.values())
    unexpected = sorted(members - allowed)
    if unexpected:
        raise ValueError(f"archive has unexpected files: {unexpected[:10]}")
    return {
        "member_count": len(infos),
        "file_count": file_count,
        "shard_count": len(shards),
        "uncompressed_bytes": uncompressed_bytes,
        "dataset_subdirectory": "libero_mix/1.0.0",
    }


def prepare(
    *,
    archive: Path,
    output: Path,
    expected_sha256: str,
    expected_shards: int = 256,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite extracted dataset: {output}")
    actual_sha256 = sha256(archive)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"RLDS archive SHA256 mismatch: {actual_sha256}")
    archive_info = inspect_archive(archive, expected_shards=expected_shards)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir()
    try:
        subprocess.run(
            ["unzip", "-q", str(archive), "-d", str(staging)],
            check=True,
        )
        dataset = staging / archive_info["dataset_subdirectory"]
        shards = sorted(
            dataset.glob(
                f"libero_mix-train.tfrecord-*-of-{expected_shards:05d}"
            )
        )
        if len(shards) != expected_shards:
            raise ValueError(
                f"extracted {len(shards)} TFRecord shards, expected {expected_shards}"
            )
        for name in ("dataset_info.json", "features.json"):
            if not (dataset / name).is_file():
                raise FileNotFoundError(dataset / name)
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "archive": str(archive),
            "archive_sha256": actual_sha256,
            "archive_info": archive_info,
            "dataset_path": archive_info["dataset_subdirectory"],
        }
        (staging / "libero_plus_camparam_rlds_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        staging.rename(output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely prepare LIBERO-Plus camparam RLDS")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    if len(args.expected_sha256) != 64:
        raise ValueError("expected SHA256 must contain 64 hexadecimal characters")
    int(args.expected_sha256, 16)
    manifest = prepare(
        archive=args.archive,
        output=args.output,
        expected_sha256=args.expected_sha256.lower(),
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
