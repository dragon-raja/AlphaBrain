from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from audit_libero_plus_goal_archive import archive_inventory


def test_archive_inventory_finds_rlds_members(tmp_path: Path) -> None:
    archive = tmp_path / "suite.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("libero_goal/1.0.0/dataset_info.json", "{}")
        handle.writestr("libero_goal/1.0.0/features.json", "{}")
        handle.writestr("libero_goal/1.0.0/data.tfrecord-00000-of-00001", b"record")

    inventory = archive_inventory(archive)

    assert inventory["tfrecord_shard_count"] == 1
    assert len(inventory["metadata_members"]) == 2
    assert inventory["file_count"] == 3


def test_archive_inventory_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape", b"bad")

    with pytest.raises(ValueError, match="unsafe members"):
        archive_inventory(archive)
