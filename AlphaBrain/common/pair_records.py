from __future__ import annotations

import hashlib
import io
import json
import struct
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from PIL import Image


FILE_MAGIC = b"DSOLPAIR1\n"
RECORD_PREFIX = struct.Struct("<QI")
IMAGE_LENGTH = struct.Struct("<I")
IMAGE_ORDER = ("canonical", "broad_a", "broad_b", "wrist")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_jpeg(image: np.ndarray, *, quality: int = 95) -> bytes:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("JPEG input must be an HxWx3 uint8 array")
    output = io.BytesIO()
    Image.fromarray(array).save(
        output,
        format="JPEG",
        quality=quality,
        subsampling=0,
        optimize=False,
    )
    return output.getvalue()


def decode_jpeg(value: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(value)) as image:
        return np.asarray(image.convert("RGB"))


def initialize_shard(handle: BinaryIO) -> None:
    if handle.tell() != 0:
        raise ValueError("a new shard must start at byte 0")
    handle.write(FILE_MAGIC)


def write_record(
    handle: BinaryIO,
    *,
    header: Mapping[str, Any],
    images: Mapping[str, np.ndarray],
    jpeg_quality: int = 95,
) -> dict[str, int]:
    if tuple(images) != IMAGE_ORDER:
        raise ValueError(f"images must be ordered as {IMAGE_ORDER}")
    header_payload = {
        **header,
        "image_order": list(IMAGE_ORDER),
        "image_codec": "jpeg_rgb",
        "jpeg_quality": int(jpeg_quality),
    }
    header_bytes = json.dumps(
        header_payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    encoded_images = [
        encode_jpeg(images[name], quality=jpeg_quality) for name in IMAGE_ORDER
    ]
    payload_length = len(header_bytes) + sum(
        IMAGE_LENGTH.size + len(image) for image in encoded_images
    )
    offset = handle.tell()
    handle.write(RECORD_PREFIX.pack(payload_length, len(header_bytes)))
    handle.write(header_bytes)
    for image in encoded_images:
        handle.write(IMAGE_LENGTH.pack(len(image)))
        handle.write(image)
    return {
        "offset": offset,
        "total_bytes": RECORD_PREFIX.size + payload_length,
    }


def _read_exact(handle: BinaryIO, length: int, label: str) -> bytes:
    value = handle.read(length)
    if len(value) != length:
        raise ValueError(f"truncated {label}: expected {length}, got {len(value)}")
    return value


def validate_magic(handle: BinaryIO) -> None:
    handle.seek(0)
    value = _read_exact(handle, len(FILE_MAGIC), "file magic")
    if value != FILE_MAGIC:
        raise ValueError("invalid DSOL pair shard magic")


def read_record(
    handle: BinaryIO,
    *,
    offset: int,
    image_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    if offset < len(FILE_MAGIC):
        raise ValueError("record offset overlaps file magic")
    handle.seek(offset)
    payload_length, header_length = RECORD_PREFIX.unpack(
        _read_exact(handle, RECORD_PREFIX.size, "record prefix")
    )
    if header_length > payload_length:
        raise ValueError("header length exceeds record payload")
    header = json.loads(_read_exact(handle, header_length, "record header"))
    selected = set(IMAGE_ORDER if image_names is None else image_names)
    unknown = selected.difference(IMAGE_ORDER)
    if unknown:
        raise ValueError(f"unknown image names: {sorted(unknown)}")
    images = {}
    consumed = header_length
    for name in IMAGE_ORDER:
        (length,) = IMAGE_LENGTH.unpack(
            _read_exact(handle, IMAGE_LENGTH.size, f"{name} length")
        )
        consumed += IMAGE_LENGTH.size
        encoded = _read_exact(handle, length, name)
        if name in selected:
            images[name] = decode_jpeg(encoded)
        consumed += length
    if consumed != payload_length:
        raise ValueError(
            f"record payload mismatch: expected {payload_length}, consumed {consumed}"
        )
    return {"header": header, "images": images}
