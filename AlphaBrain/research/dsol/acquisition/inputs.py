"""Audited current-frame input whitelists, NOT a complete evaluation runner.

This boundary takes the raw endpoint observation returned by the acquisition
executor, before ``prepare_policy_observation`` flips/resizes images. It never
imports that adapter, a policy, or a simulator, and performs no image encoding,
flipping, resizing, camera queries, or budget charges. One observed bundle
contains BOTH external and wrist RGB; constructing two consumers is not two
queries. The caller remains responsible for current-frame temporal provenance,
RGB channel order, and correspondence of explicitly supplied calibration.

The supported draft policy contract is external RGB + wrist RGB + language,
WITHOUT proprioception. The existing websocket interface requires an 8D state
and cannot consume this policy mapping unchanged. Resolving that mismatch needs
a separately audited adapter/contract; this module does not silently add state.

Selector-only robot_state is caller-supplied [eef_position(3), eef_axis_angle(3),
gripper_qpos(2)], matching the current LIBERO interface. No state is extracted
from raw observation. Geometry is ONLY the caller-supplied current external
camera's OpenCV intrinsics and camera-to-world transform. Matrices are not
reinterpreted for raw image indexing here; geometric/pixel registration must be
verified by the eventual adapter. No wrist/candidate calibration is inferred.

Only the two literal raw RGB keys are read. Object state, simulator truth,
history, future/candidate images, noise controls, and other raw fields are not
forwarded or inspected. Structural validation cannot prove that a caller has
not deliberately mislabeled privileged data as an allowed input.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np


INPUT_SCHEMA = "view-acquisition-inputs/v1"
INPUT_CONTRACT_ID = "dsol_external_wrist_language_v1"
RAW_IMAGE_KEYS = MappingProxyType({
    "external_rgb": "agentview_image",
    "wrist_rgb": "robot0_eye_in_hand_image",
})
_POLICY_KEYS = ("external_rgb", "wrist_rgb", "language")
_SELECTOR_KEYS = _POLICY_KEYS + ("robot_state", "camera_geometry")
_GEOMETRY_KEYS = ("camera_intrinsics", "camera_to_world_opencv")


class InputContractError(ValueError):
    """Unsupported permission contract or invalid explicitly allowed input."""


def supported_input_contract() -> dict[str, Any]:
    """Return a fresh copy of the one implemented contract, not release approval."""
    return {
        "id": INPUT_CONTRACT_ID,
        "policy_inputs": list(_POLICY_KEYS),
        "selector_inputs": list(_SELECTOR_KEYS),
        "candidate_images": "forbidden",
        "along_path_images": False,
        "image_readout": "after_arrival_only",
        "query_unit": "observation_bundle",
        "bundle_image_keys": ["external_rgb", "wrist_rgb"],
        "rules_share_input_permissions": True,
        "base_policy_history": "current_frame",
        "extra_selector_state_declared": True,
    }


def _validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise InputContractError("input_contract must be a mapping")
    expected = supported_input_contract()
    if set(contract) != set(expected):
        raise InputContractError("unknown or missing contract fields require a new audited contract")
    for key, required in expected.items():
        actual = contract[key]
        # In particular, 0/1 must not pass for the boolean permission flags.
        matches = type(actual) is type(required)
        if isinstance(required, list):
            matches = matches and len(actual) == len(required) and all(
                type(item) is str and item == reference for item, reference in zip(actual, required)
            )
        else:
            matches = matches and actual == required
        if not matches:
            raise InputContractError(f"unsupported input_contract field: {key}")
    return expected


def _immutable_array(array: np.ndarray) -> np.ndarray:
    # A bytes-backed copy is C-contiguous and cannot be made writable again.
    # Each consumer receives a separate copy, even if source images alias.
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _rgb(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype("uint8"):
        raise InputContractError(f"{name} must be a uint8 numpy RGB array; no implicit conversion")
    if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) <= 0:
        raise InputContractError(f"{name} must have nonempty HWC shape (H, W, 3)")
    return _immutable_array(value)


def _numeric_array(value: Any, *, shape: tuple[int, ...], name: str, dtype: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputContractError(f"{name} must be finite numeric data with shape {shape}") from exc
    if array.shape != shape or array.dtype.kind not in "fiu":
        raise InputContractError(f"{name} must be real numeric data with shape {shape}")
    if not np.all(np.isfinite(array)):
        raise InputContractError(f"{name} must be finite")
    target = np.dtype(dtype)
    if np.any(np.abs(array.astype(np.longdouble)) > np.finfo(target).max):
        raise InputContractError(f"{name} cannot be represented as finite {dtype}")
    return _immutable_array(np.asarray(array, dtype=target))


def _camera_geometry(value: Any) -> Mapping[str, np.ndarray]:
    if not isinstance(value, Mapping) or set(value) != set(_GEOMETRY_KEYS):
        raise InputContractError("camera_geometry requires only current external intrinsics and camera-to-world")
    intrinsics = _numeric_array(value["camera_intrinsics"], shape=(3, 3), name="camera_intrinsics", dtype="float64")
    transform = _numeric_array(
        value["camera_to_world_opencv"], shape=(4, 4), name="camera_to_world_opencv", dtype="float64"
    )
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0 or not np.allclose(
        intrinsics[2], [0, 0, 1], atol=1e-8, rtol=0
    ):
        raise InputContractError("camera_intrinsics must have positive focal lengths and bottom row [0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise InputContractError("camera_to_world_opencv must have homogeneous bottom row [0, 0, 0, 1]")
    if (
        np.any(np.abs(rotation) > 1.0 + 1e-6)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0)
    ):
        raise InputContractError("camera_to_world_opencv rotation must be a proper orthonormal matrix")
    return MappingProxyType({"camera_intrinsics": intrinsics, "camera_to_world_opencv": transform})


def _json_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _descriptor(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        return {
            "kind": "ndarray", "shape": list(value.shape), "dtype": value.dtype.name,
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
        }
    if isinstance(value, str):
        return {
            "kind": "utf8", "characters": len(value),
            "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        }
    if isinstance(value, Mapping):
        fields = {key: _descriptor(item) for key, item in value.items()}
        return {"kind": "mapping", "keys": list(value), "fields": fields, "sha256": _json_hash(fields)}
    raise InputContractError("unexpected value type in constructed input")


@dataclass(frozen=True)
class InputBundle:
    """Read-only payloads with independent storage and a pixel-free audit copy."""

    policy: Mapping[str, Any] = field(repr=False)
    selector: Mapping[str, Any] = field(repr=False)
    _receipt: dict[str, Any] = field(repr=False)

    def audit_record(self) -> dict[str, Any]:
        """Return only metadata, keys, shapes, dtypes and hashes; never values."""
        return copy.deepcopy(self._receipt)


def build_input_bundle(
    observation: Mapping[str, Any],
    *,
    language: str,
    input_contract: Mapping[str, Any],
    robot_state: Any = None,
    camera_geometry: Any = None,
) -> InputBundle:
    """Build policy/selector inputs from one already-read current RGB bundle.

    robot_state and camera_geometry are mandatory under this supported contract,
    but they are selector-only. Raw fields with those names are never used as a
    fallback. Missing one camera rejects the entire bundle. Caller must supply
    current observations, not candidate/path/history frames under current keys.
    """
    contract = _validate_contract(input_contract)
    if not isinstance(observation, Mapping):
        raise InputContractError("observation must be a mapping")
    if not isinstance(language, str) or not language.strip():
        raise InputContractError("language must be an explicit nonempty string")
    try:
        language.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InputContractError("language must be valid UTF-8 text") from exc
    images = {}
    for output_key, raw_key in RAW_IMAGE_KEYS.items():
        try:
            value = observation[raw_key]
        except KeyError as exc:
            raise InputContractError(f"missing required current RGB image: {raw_key}") from exc
        images[output_key] = _rgb(value, raw_key)
    if robot_state is None:
        raise InputContractError("selector robot_state must be supplied explicitly, not read from observation")
    if camera_geometry is None:
        raise InputContractError("selector camera_geometry must be supplied explicitly, not read from observation")
    state = _numeric_array(robot_state, shape=(8,), name="robot_state", dtype="float32")
    geometry = _camera_geometry(camera_geometry)
    policy = MappingProxyType({**images, "language": language})
    selector = MappingProxyType({
        "external_rgb": _immutable_array(images["external_rgb"]),
        "wrist_rgb": _immutable_array(images["wrist_rgb"]),
        "language": language,
        "robot_state": state,
        "camera_geometry": geometry,
    })
    receipt = {
        "schema_version": INPUT_SCHEMA,
        "input_contract_id": contract["id"],
        "input_contract_sha256": _json_hash(contract),
        "input_stage": "raw_libero_observation_before_existing_policy_adapter",
        "policy_encoding_applied": False,
        "image_preprocessing_applied": False,
        "existing_websocket_request_ready": False,
        "base_policy_robot_state_included": False,
        "base_policy_history": "current_frame",
        "query_unit": "observation_bundle",
        "query_units_per_observation_read": 1,
        "bundle_image_keys": list(RAW_IMAGE_KEYS),
        "bundle_image_count": 2,
        "builder_performs_or_charges_queries": False,
        "robot_state_source": "explicit_caller_argument",
        "robot_state_layout": "eef_position_3_eef_axis_angle_3_gripper_qpos_2",
        "camera_geometry_source": "explicit_caller_argument_current_external_camera",
        "raw_pixel_geometry_registration_verified": False,
        "temporal_provenance_verified": False,
        "policy": _descriptor(policy),
        "selector": _descriptor(selector),
    }
    receipt["receipt_sha256"] = _json_hash(receipt)
    return InputBundle(policy, selector, receipt)
