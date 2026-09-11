from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from zipfile import ZipFile

import pytest

from prepare_libero_plus_camparam_rlds import inspect_archive, prepare


def _archive(path: Path, *, unsafe: bool = False) -> str:
    with ZipFile(path, "w") as bundle:
        bundle.writestr("libero_mix/1.0.0/dataset_info.json", "{}")
        bundle.writestr("libero_mix/1.0.0/features.json", "{}")
        bundle.writestr(
            "libero_mix/1.0.0/libero_mix-train.tfrecord-00000-of-00002",
            b"first",
        )
        bundle.writestr(
            "libero_mix/1.0.0/libero_mix-train.tfrecord-00001-of-00002",
            b"second",
        )
        if unsafe:
            bundle.writestr("../escape", b"bad")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_validates_and_extracts_atomically() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        archive = root / "dataset.zip"
        digest = _archive(archive)
        info = inspect_archive(archive, expected_shards=2)
        assert info["shard_count"] == 2

        output = root / "prepared"
        manifest = prepare(
            archive=archive,
            output=output,
            expected_sha256=digest,
            expected_shards=2,
        )
        assert manifest["status"] == "complete"
        assert len(list((output / "libero_mix/1.0.0").glob("*.tfrecord-*"))) == 2
        saved = json.loads(
            (output / "libero_plus_camparam_rlds_manifest.json").read_text()
        )
        assert saved == manifest


def test_inspect_archive_rejects_path_traversal() -> None:
    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / "unsafe.zip"
        _archive(archive, unsafe=True)
        with pytest.raises(ValueError, match="unsafe zip member"):
            inspect_archive(archive, expected_shards=2)
