"""Data-only replay files; no pickle, queue state, or implicit restoration.

The caller supplies the independently recorded file digest and task contract.
IntegrationReplay still checks compiled model identity before restoration. A
file passing these checks does not grant permission to resume a research task.
"""

import base64
import hashlib
import json
from pathlib import Path
import numpy as np


def _encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject and value.shape == () and value.item() is None:
            return {"type": "none_array"}
        if value.dtype.kind not in "biuf":
            raise ValueError("Only numeric replay arrays supported")
        return {
            "type": "array",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": base64.b64encode(value.tobytes(order="C")).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return _encode(value.item())
    if value is None or type(value) in (str, bool, int, float):
        return {"type": "scalar", "value": value}
    if type(value) in (tuple, list):
        return {"type": type(value).__name__, "items": [_encode(x) for x in value]}
    if type(value) is dict:
        return {"type": "dict", "items": [[_encode(k), _encode(v)] for k, v in value.items()]}
    raise ValueError("Unsupported replay state type: " + type(value).__name__)


def _decode(value):
    kind = value["type"]
    if kind == "none_array":
        if set(value) != {"type"}:
            raise ValueError("Invalid None buffer sentinel")
        return np.array(None, dtype=object)
    if kind == "scalar":
        if set(value) != {"type", "value"} or type(value["value"]) not in (
            type(None),
            str,
            bool,
            int,
            float,
        ):
            raise ValueError("Invalid scalar")
        return value["value"]
    if kind == "array":
        if set(value) != {"type", "dtype", "shape", "data"}:
            raise ValueError("Invalid array envelope")
        dtype = np.dtype(value["dtype"])
        shape = value["shape"]
        if (
            dtype.kind not in "biuf"
            or not isinstance(shape, list)
            or any(type(x) is not int or x < 0 for x in shape)
        ):
            raise ValueError("Invalid numeric replay array")
        raw = base64.b64decode(value["data"], validate=True)
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if set(value) != {"type", "items"}:
        raise ValueError("Invalid container envelope")
    if kind in ("tuple", "list"):
        items = [_decode(x) for x in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "dict":
        pairs = [(_decode(k), _decode(v)) for k, v in value["items"]]
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("Duplicate replay state key")
        return result
    raise ValueError("Unknown replay value type")


def save_snapshot(path, snapshot, *, task_contract):
    """Write a new immutable file; interrupted writes cannot match its digest."""
    import os

    payload = {
        "schema": "GR1_REPLAY_DATA_V1",
        "task_contract": task_contract,
        "state": _encode(snapshot),
    }
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(data).hexdigest()


def load_snapshot(path, *, expected_sha256, task_contract):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Replay file digest differs")
    payload = json.loads(data)
    if (
        set(payload) != {"schema", "task_contract", "state"}
        or payload["schema"] != "GR1_REPLAY_DATA_V1"
        or payload["task_contract"] != task_contract
    ):
        raise ValueError("Replay task contract differs")
    return _decode(payload["state"])
