from __future__ import annotations

import stat
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from prepare_libero_plus_runtime import (
    extract_assets,
    extraction_groups,
    inspect_archive,
    partition_extraction_groups,
)


def test_inspect_archive_finds_and_strips_unique_prefix(tmp_path: Path) -> None:
    archive = tmp_path / "assets.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("release/tree/assets/scenes/scene.xml", b"scene")
        bundle.writestr("release/tree/assets/textures/texture.png", b"texture")
    result = inspect_archive(archive)
    assert result == {
        "member_count": 2,
        "file_count": 2,
        "uncompressed_bytes": 12,
        "stripped_prefix": ["release", "tree"],
    }


def test_inspect_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("../assets/scenes/scene.xml", b"scene")
    with pytest.raises(ValueError, match="unsafe zip member"):
        inspect_archive(archive)


def test_inspect_archive_rejects_symlink(tmp_path: Path) -> None:
    archive = tmp_path / "symlink.zip"
    info = ZipInfo("release/assets/scenes/link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(archive, "w") as bundle:
        bundle.writestr(info, "target")
    with pytest.raises(ValueError, match="symlink"):
        inspect_archive(archive)


def test_parallel_extraction_groups_are_disjoint_and_balanced(tmp_path: Path) -> None:
    archive = tmp_path / "assets.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("release/assets/new_objects/apple/a.obj", b"apple-a")
        bundle.writestr("release/assets/new_objects/apple/b.obj", b"apple-b")
        bundle.writestr("release/assets/new_objects/bowl/a.obj", b"bowl")
        bundle.writestr("release/assets/scenes/kitchen/scene.xml", b"scene")
        bundle.writestr("release/assets/textures/table.png", b"texture")
        bundle.writestr("release/assets/wall.xml", b"wall")

    groups = extraction_groups(archive, stripped_prefix=("release",))
    assert {group["pattern"] for group in groups} == {
        "release/assets/new_objects/apple/*",
        "release/assets/new_objects/bowl/*",
        "release/assets/scenes/*",
        "release/assets/textures/*",
        "release/assets/wall.xml",
    }
    partitions = partition_extraction_groups(groups, workers=3)
    assert len(partitions) == 3
    assert sum(len(partition) for partition in partitions) == len(groups)
    assert len({group["pattern"] for partition in partitions for group in partition}) == len(groups)


def test_parallel_extraction_rejects_empty_group_list() -> None:
    with pytest.raises(ValueError, match="no asset files"):
        partition_extraction_groups([], workers=2)


def test_parallel_extraction_overwrites_interrupted_file(tmp_path: Path) -> None:
    archive = tmp_path / "assets.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("release/assets/new_objects/apple/a.obj", b"complete-apple")
        bundle.writestr("release/assets/scenes/kitchen/scene.xml", b"complete-scene")
        bundle.writestr("release/assets/textures/table.png", b"complete-texture")
    extraction = tmp_path / "extraction"
    partial = extraction / "release/assets/new_objects/apple/a.obj"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")

    extract_assets(
        archive=archive,
        extraction=extraction,
        stripped_prefix=("release",),
        workers=3,
    )

    assert partial.read_bytes() == b"complete-apple"
    assert (
        extraction / "release/assets/scenes/kitchen/scene.xml"
    ).read_bytes() == b"complete-scene"
    assert (
        extraction / "release/assets/textures/table.png"
    ).read_bytes() == b"complete-texture"
