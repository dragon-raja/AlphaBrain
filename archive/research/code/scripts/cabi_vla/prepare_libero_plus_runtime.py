from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

from resume_aria2_ranges import sha256


REQUIRED_ASSET_ENTRIES = (
    "scenes",
    "textures",
)


def inspect_archive(archive: Path) -> dict[str, Any]:
    prefixes: set[tuple[str, ...]] = set()
    file_count = 0
    uncompressed_bytes = 0
    with ZipFile(archive) as bundle:
        infos = bundle.infolist()
        for info in infos:
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe zip member: {info.filename}")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise ValueError(f"zip symlink is not allowed: {info.filename}")
            try:
                assets_index = path.parts.index("assets")
            except ValueError as error:
                raise ValueError(
                    f"zip member is outside the assets subtree: {info.filename}"
                ) from error
            prefixes.add(tuple(path.parts[:assets_index]))
            if not info.is_dir():
                file_count += 1
                uncompressed_bytes += int(info.file_size)
    if len(prefixes) != 1:
        raise ValueError(f"archive has multiple assets prefixes: {sorted(prefixes)}")
    prefix = next(iter(prefixes))
    return {
        "member_count": len(infos),
        "file_count": file_count,
        "uncompressed_bytes": uncompressed_bytes,
        "stripped_prefix": list(prefix),
    }


def extraction_groups(
    archive: Path,
    *,
    stripped_prefix: Iterable[str],
) -> list[dict[str, Any]]:
    assets_root = (*stripped_prefix, "assets")
    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    with ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                continue
            parts = PurePosixPath(info.filename).parts
            relative = parts[len(assets_root) :]
            if tuple(parts[: len(assets_root)]) != assets_root or not relative:
                raise ValueError(f"zip member is outside the assets subtree: {info.filename}")
            key = relative[:2] if relative[0] == "new_objects" and len(relative) > 1 else relative[:1]
            group = groups.setdefault(
                key,
                {
                    "key": key,
                    "file_count": 0,
                    "uncompressed_bytes": 0,
                    "has_descendants": False,
                },
            )
            group["file_count"] += 1
            group["uncompressed_bytes"] += int(info.file_size)
            group["has_descendants"] |= len(relative) > len(key)

    result: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        if any(any(character in part for character in "*?[]\\") for part in key):
            raise ValueError(f"unsupported wildcard character in archive path: {key}")
        path = PurePosixPath(*assets_root, *key).as_posix()
        result.append(
            {
                **group,
                "pattern": f"{path}/*" if group["has_descendants"] else path,
            }
        )
    return result


def partition_extraction_groups(
    groups: Iterable[dict[str, Any]],
    *,
    workers: int,
) -> list[list[dict[str, Any]]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    ordered = sorted(groups, key=lambda group: int(group["file_count"]), reverse=True)
    if not ordered:
        raise ValueError("archive contains no asset files")
    partitions: list[list[dict[str, Any]]] = [[] for _ in range(min(workers, len(ordered)))]
    counts = [0] * len(partitions)
    for group in ordered:
        partition_index = min(range(len(partitions)), key=counts.__getitem__)
        partitions[partition_index].append(group)
        counts[partition_index] += int(group["file_count"])
    return partitions


def extract_assets(
    *,
    archive: Path,
    extraction: Path,
    stripped_prefix: Iterable[str],
    workers: int,
) -> None:
    groups = extraction_groups(archive, stripped_prefix=stripped_prefix)
    partitions = partition_extraction_groups(groups, workers=workers)
    extraction.mkdir(parents=True, exist_ok=True)

    def extract_partition(partition: list[dict[str, Any]]) -> None:
        subprocess.run(
            [
                "unzip",
                "-q",
                "-o",
                str(archive),
                *(str(group["pattern"]) for group in partition),
                "-d",
                str(extraction),
            ],
            check=True,
        )

    with ThreadPoolExecutor(max_workers=len(partitions)) as executor:
        list(executor.map(extract_partition, partitions))


def _copy_ignore(directory: str, names: list[str]) -> set[str]:
    path = Path(directory)
    ignored = {".git", "__pycache__"} & set(names)
    if path.name == "static":
        ignored |= {"videos"} & set(names)
    return ignored


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def prepare(
    *,
    source_repo: Path,
    archive: Path,
    output: Path,
    expected_sha256: str,
    workers: int = 1,
    resume_staging: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite runtime: {output}")
    if _git(source_repo, "status", "--porcelain"):
        raise ValueError(f"source LIBERO-Plus checkout is dirty: {source_repo}")
    archive_sha256 = sha256(archive)
    if archive_sha256 != expected_sha256:
        raise ValueError(f"asset archive SHA256 mismatch: {archive_sha256}")
    archive_info = inspect_archive(archive)

    output.parent.mkdir(parents=True, exist_ok=True)
    if resume_staging is None:
        staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
        if staging.exists():
            raise FileExistsError(staging)
        shutil.copytree(source_repo, staging, ignore=_copy_ignore)
        extraction = staging / ".archive-extraction"
    else:
        staging = resume_staging.resolve()
        expected_prefix = f".{output.name}.staging-"
        if staging.parent != output.parent.resolve() or not staging.name.startswith(expected_prefix):
            raise ValueError(f"resume staging is outside the expected runtime directory: {staging}")
        extraction = staging / ".archive-extraction"
        if not extraction.is_dir():
            raise FileNotFoundError(f"resume extraction directory is missing: {extraction}")

    extract_assets(
        archive=archive,
        extraction=extraction,
        stripped_prefix=archive_info["stripped_prefix"],
        workers=workers,
    )
    source_assets = extraction.joinpath(
        *archive_info["stripped_prefix"],
        "assets",
    )
    destination_assets = staging / "libero" / "libero" / "assets"
    if not source_assets.is_dir():
        raise FileNotFoundError(f"extracted assets directory is missing: {source_assets}")
    if destination_assets.exists():
        raise FileExistsError(destination_assets)
    shutil.move(str(source_assets), str(destination_assets))
    shutil.rmtree(extraction)

    missing = [
        name for name in REQUIRED_ASSET_ENTRIES if not (destination_assets / name).exists()
    ]
    if missing:
        raise ValueError(f"runtime assets are missing required entries: {missing}")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "source_repo": str(source_repo),
        "source_commit": _git(source_repo, "rev-parse", "HEAD"),
        "asset_archive": str(archive),
        "asset_archive_sha256": archive_sha256,
        "archive": archive_info,
        "assets_path": "libero/libero/assets",
        "extraction_workers": workers,
    }
    (staging / "libero_plus_runtime_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    staging.rename(output)
    return manifest


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare an isolated LIBERO-Plus runtime")
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--assets-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume-staging", type=Path)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    if len(args.expected_sha256) != 64:
        raise ValueError("expected SHA256 must contain 64 hexadecimal characters")
    int(args.expected_sha256, 16)
    manifest = prepare(
        source_repo=args.source_repo,
        archive=args.assets_archive,
        output=args.output,
        expected_sha256=args.expected_sha256.lower(),
        workers=args.workers,
        resume_staging=args.resume_staging,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
