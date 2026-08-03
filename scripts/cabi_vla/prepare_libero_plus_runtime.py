from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
from collections.abc import Iterable
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
    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    shutil.copytree(source_repo, staging, ignore=_copy_ignore)

    extraction = staging / ".archive-extraction"
    extraction.mkdir()
    subprocess.run(
        ["unzip", "-q", str(archive), "-d", str(extraction)],
        check=True,
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
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
