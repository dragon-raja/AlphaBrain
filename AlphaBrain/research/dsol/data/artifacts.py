"""Extracted from the audited INITIAL32 analysis; preserve scientific semantics."""

from pathlib import Path
import hashlib
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar


_PATH_MAP = ContextVar('artifact_path_map', default=())


def mapped_path(value):
    """Resolve explicit relocation prefixes; never create compatibility paths."""
    value = str(value)
    for old, new in _PATH_MAP.get():
        if value == old or value.startswith(old + '/'):
            return new + value[len(old):]
    return value


def _map_record(value):
    if isinstance(value, str):
        return mapped_path(value)
    if isinstance(value, list):
        return [_map_record(v) for v in value]
    if isinstance(value, dict):
        return {mapped_path(k): _map_record(v) for k, v in value.items()}
    return value


@contextmanager
def path_mapping(pairs):
    """Map read-time paths only. SHA256 always covers original file bytes."""
    pairs = tuple((str(a).rstrip('/'), str(b).rstrip('/')) for a, b in pairs)
    if any(not a.startswith('/') or not b.startswith('/') or not a or not b
           for a, b in pairs) or len({a for a, _ in pairs}) != len(pairs):
        raise ValueError('Unique absolute relocation prefixes required')
    token = _PATH_MAP.set(tuple(sorted(pairs, key=lambda p: -len(p[0]))))
    try:
        yield
    finally:
        _PATH_MAP.reset(token)


def read(p):
    return _map_record(json.loads(Path(mapped_path(p)).read_text()))


def sha(p):
    return hashlib.sha256(Path(mapped_path(p)).read_bytes()).hexdigest()


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, p)
