from __future__ import annotations

import json
from pathlib import Path

import pytest

from download_verified_ranges import (
    _load_or_create_state,
    chunk_ranges,
)


def test_chunk_ranges_covers_tail_without_overlap() -> None:
    assert chunk_ranges(10, 4) == [(0, 3), (4, 7), (8, 9)]


def test_state_creation_and_resume_validation(tmp_path: Path) -> None:
    output = tmp_path / "resource.bin"
    state_path = tmp_path / "resource.bin.ranges.json"
    kwargs = {
        "output": output,
        "state_path": state_path,
        "url": "https://example.invalid/resource.bin",
        "expected_size": 10,
        "expected_sha256": "0" * 64,
        "chunk_bytes": 4,
    }
    state = _load_or_create_state(**kwargs)
    assert output.stat().st_size == 10
    assert state["completed_chunks"] == []

    payload = json.loads(state_path.read_text())
    payload["completed_chunks"] = [0]
    state_path.write_text(json.dumps(payload))
    resumed = _load_or_create_state(**kwargs)
    assert resumed["completed_chunks"] == [0]

    with pytest.raises(ValueError, match="expected_size"):
        _load_or_create_state(**{**kwargs, "expected_size": 11})


def test_refuses_untracked_output(tmp_path: Path) -> None:
    output = tmp_path / "resource.bin"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        _load_or_create_state(
            output=output,
            state_path=tmp_path / "resource.bin.ranges.json",
            url="https://example.invalid/resource.bin",
            expected_size=8,
            expected_sha256="0" * 64,
            chunk_bytes=4,
        )
