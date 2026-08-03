from __future__ import annotations

import stat
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from prepare_libero_plus_runtime import inspect_archive


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
