from __future__ import annotations

import struct
from pathlib import Path

from resume_aria2_ranges import missing_byte_ranges, parse_control, sha256


def _control(path: Path, *, piece_length: int, total: int, bits: bytes) -> None:
    payload = bytearray(34 + len(bits))
    struct.pack_into(">H", payload, 0, 1)
    struct.pack_into(">I", payload, 10, piece_length)
    struct.pack_into(">Q", payload, 14, total)
    struct.pack_into(">I", payload, 30, len(bits))
    payload[34:] = bits
    path.write_bytes(payload)


def test_parse_control_and_missing_ranges(tmp_path: Path) -> None:
    path = tmp_path / "sample.aria2"
    # Six pieces: complete, incomplete x2, complete x2, incomplete.
    _control(path, piece_length=4, total=22, bits=bytes([0b10011000]))
    piece, total, completed = parse_control(path)
    assert piece == 4
    assert total == 22
    assert completed == [True, False, False, True, True, False]
    assert missing_byte_ranges(piece, total, completed) == [(4, 11), (20, 21)]


def test_sha_fixture_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "fixture.bin"
    path.write_bytes(b"range-repair")
    assert sha256(path) == (
        "09b606855024b407a151cdb2e86f13f69350b08330014d6a15b5db1191c5bb75"
    )
