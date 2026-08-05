from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from audit_libero_plus_goal_archive import archive_inventory, source_factor_hint


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


def test_source_factor_hint_distinguishes_base_task_table_word() -> None:
    hint = source_factor_hint(
        "/data/canonical/pick_up_the_bowl_from_table_center_demo.hdf5"
    )

    assert hint["factor_class"] == "other"
    assert hint["background_id"] is None


def test_source_factor_hint_recovers_background_and_view_ids() -> None:
    background = source_factor_hint("/data/background/task_name_tb_21_demo.hdf5")
    camera = source_factor_hint(
        "/data/extrinsics_camera_view/task_name_view_1_15_100_0_0_initstate_0_demo.hdf5"
    )

    assert background["factor_class"] == "background"
    assert background["background_id"] == 21
    assert camera["factor_class"] == "camera"
    assert camera["view_id"] == "1_15_100_0_0"
