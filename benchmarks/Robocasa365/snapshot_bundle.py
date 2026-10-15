# Promoted from the audited 20260814 RoboCasa recovery source on 20260916.
# Historical sources remain immutable; runtime no longer imports recovery paths.
"""Persistent, checksummed RoboCasa environment snapshot bundles.

``robocasa_snapshot`` handles exact in-process capture / restore.  This module
adds the missing disk layer required to evaluate several checkpoints against
the same physical initial states across independent processes.

Snapshot payloads contain trusted Python pickle data and must only be loaded
from a locally generated, checksummed archive.  The XML and episode metadata
are stored separately so that the exact MuJoCo model can be reconstructed
before controller, fixture, observable, wrapper, RNG, and warm-start state is
restored.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import tempfile
from typing import Any

import numpy as np

from .snapshot import (
    EnvSnapshot,
    inner_env,
    refresh_observation,
    restore_snapshot,
)


FORMAT_VERSION = "robocasa_eval_snapshot_bundle_v1"
PICKLE_FILE = "snapshot.pkl.gz"
MODEL_FILE = "model.xml.gz"
COMPILED_MODEL_FILE = "compiled_model.xml.gz"
EP_META_FILE = "ep_meta.json"
CANONICAL_RGB_FILE = "canonical_rgb.npz"
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class SnapshotBundle:
    path: Path
    manifest: dict[str, Any]
    ep_meta: dict[str, Any]
    model_xml: str
    snapshot: EnvSnapshot
    canonical_images: dict[str, np.ndarray] | None


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(digest: Any, name: str, value: np.ndarray | None) -> None:
    digest.update(name.encode("utf-8"))
    if value is None:
        digest.update(b"<none>")
        return
    array = np.ascontiguousarray(value)
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())


def snapshot_numeric_fingerprint(snapshot: EnvSnapshot) -> str:
    """Hash all numeric simulator/model fields needed for physical pairing."""

    digest = hashlib.sha256()
    _array_digest(digest, "sim_state", snapshot.sim_state)
    _array_digest(digest, "camera_positions", snapshot.camera_positions)
    _array_digest(digest, "camera_quaternions", snapshot.camera_quaternions)
    _array_digest(digest, "ctrl", snapshot.ctrl)
    _array_digest(digest, "qacc_warmstart", snapshot.qacc_warmstart)
    _array_digest(digest, "mocap_positions", snapshot.mocap_positions)
    _array_digest(digest, "mocap_quaternions", snapshot.mocap_quaternions)
    for name, value in sorted(snapshot.model_runtime.items()):
        _array_digest(digest, f"model_runtime.{name}", value)
    return digest.hexdigest()


def env_numeric_fingerprint(env: Any) -> str:
    """Hash the live fields covered by ``snapshot_numeric_fingerprint``."""

    base = inner_env(env)
    sim = base.sim
    digest = hashlib.sha256()
    _array_digest(digest, "sim_state", np.asarray(sim.get_state().flatten()))
    _array_digest(digest, "camera_positions", np.asarray(sim.model.cam_pos))
    _array_digest(digest, "camera_quaternions", np.asarray(sim.model.cam_quat))
    _array_digest(digest, "ctrl", getattr(sim.data, "ctrl", None))
    _array_digest(
        digest, "qacc_warmstart", getattr(sim.data, "qacc_warmstart", None)
    )
    _array_digest(digest, "mocap_positions", getattr(sim.data, "mocap_pos", None))
    _array_digest(
        digest, "mocap_quaternions", getattr(sim.data, "mocap_quat", None)
    )
    for name in sorted(("site_rgba", "site_size", "geom_rgba", "body_pos", "body_quat")):
        if hasattr(sim.model, name):
            _array_digest(digest, f"model_runtime.{name}", getattr(sim.model, name))
    return digest.hexdigest()


def _names(model: Any, attribute: str) -> list[str]:
    values = getattr(model, attribute, ())
    return [str(value) for value in values]


def model_signature(env: Any) -> dict[str, Any]:
    return compiled_model_signature(inner_env(env).sim.model)


def compiled_model_signature(model: Any) -> dict[str, Any]:
    """Describe a compiled model without requiring an environment or reset."""
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "nbody": int(model.nbody),
        "ncam": int(model.ncam),
        "ngeom": int(model.ngeom),
        "body_names": _names(model, "body_names"),
        "camera_names": _names(model, "camera_names"),
        "joint_names": _names(model, "joint_names"),
        "geom_names": _names(model, "geom_names"),
    }


def load_compiled_model_xml(bundle: SnapshotBundle) -> str:
    """Read the archived runtime model for direct, no-reset reconstruction.

    The source MODEL_FILE is intentionally retained for environment reconstruction,
    where edit_model_xml adds runtime cameras. Direct MjSim construction must use
    the separately archived compiled XML. Never synthesize missing cameras or fall
    back to the source XML when this payload is missing or has changed.
    """
    expected = bundle.manifest['files'].get(COMPILED_MODEL_FILE)
    if expected is None:
        raise ValueError('No checksummed compiled model XML in snapshot bundle')
    path = bundle.path / COMPILED_MODEL_FILE
    if path.stat().st_size != int(expected['bytes']) or _sha256(path) != expected['sha256']:
        raise RuntimeError('Archived compiled model XML checksum/size mismatch')
    return _read_gzip_text(path)


def _write_gzip_text(path: Path, value: str) -> None:
    with gzip.open(path, "wt", compresslevel=6) as stream:
        stream.write(value)


def _read_gzip_text(path: Path) -> str:
    with gzip.open(path, "rt") as stream:
        return stream.read()


def save_snapshot_bundle(destination, *, env, snapshot, canonical_images, metadata):
    """Publish a trusted snapshot, manifest last; never overwrite a bundle.

    Promoted from the same audited 20260814 archive as the reader. A partial
    publication remains visible without a manifest and must be investigated,
    not silently regenerated with another physical initial state.
    """
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    base = inner_env(env)
    payloads = (PICKLE_FILE, MODEL_FILE, COMPILED_MODEL_FILE,
                EP_META_FILE, CANONICAL_RGB_FILE)
    with tempfile.TemporaryDirectory(prefix=f'.{destination.name}.tmp-',
                                     dir=destination.parent) as temporary:
        temp = Path(temporary)
        (temp / EP_META_FILE).write_text(json.dumps(_jsonable(base.get_ep_meta()),
                                                  indent=2, sort_keys=True) + '\n')
        _write_gzip_text(temp / MODEL_FILE, str(base.model.get_xml()))
        _write_gzip_text(temp / COMPILED_MODEL_FILE, str(base.sim.model.get_xml()))
        with gzip.open(temp / PICKLE_FILE, 'wb', compresslevel=3) as stream:
            pickle.dump(snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)
        np.savez_compressed(temp / CANONICAL_RGB_FILE,
                            **{str(k): np.asarray(v).copy() for k, v in canonical_images.items()})
        manifest = dict(format=FORMAT_VERSION,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            metadata=_jsonable(metadata), model_signature=model_signature(env),
            snapshot_numeric_sha256=snapshot_numeric_fingerprint(snapshot),
            files={name: dict(sha256=_sha256(temp / name), bytes=(temp / name).stat().st_size)
                   for name in payloads})
        (temp / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        destination.mkdir(exist_ok=False)
        for name in payloads:
            shutil.copy2(temp / name, destination / name)
            if _sha256(destination / name) != manifest['files'][name]['sha256']:
                raise RuntimeError('Snapshot publication checksum mismatch: ' + name)
        shutil.copy2(temp / MANIFEST_FILE, destination / MANIFEST_FILE)
    return manifest


class _LegacySnapshotUnpickler(pickle.Unpickler):
    """Resolve the old snapshot class without importing archival code.

    This is NOT an untrusted-pickle sandbox. Only trusted local, checksummed
    experimental archives may be loaded.
    """

    def find_class(self, module, name):
        if module == "robocasa_snapshot" and name == "EnvSnapshot":
            return EnvSnapshot
        return super().find_class(module, name)


def load_snapshot_bundle(path: Path, *, verify: bool = True) -> SnapshotBundle:
    path = path.resolve()
    manifest = json.loads((path / MANIFEST_FILE).read_text())
    if manifest.get("format") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported snapshot format {manifest.get('format')!r} at {path}"
        )
    if verify:
        for name, expected in manifest["files"].items():
            payload = path / name
            actual_size = payload.stat().st_size
            actual_hash = _sha256(payload)
            if actual_size != int(expected["bytes"]):
                raise RuntimeError(f"size mismatch for {payload}")
            if actual_hash != expected["sha256"]:
                raise RuntimeError(f"SHA256 mismatch for {payload}")

    ep_meta = json.loads((path / EP_META_FILE).read_text())
    model_xml = _read_gzip_text(path / MODEL_FILE)
    with gzip.open(path / PICKLE_FILE, "rb") as stream:
        snapshot = _LegacySnapshotUnpickler(stream).load()  # trusted, checksummed local archive only
    canonical_images = None
    if CANONICAL_RGB_FILE in manifest["files"]:
        with np.load(path / CANONICAL_RGB_FILE, allow_pickle=False) as payload:
            canonical_images = {
                str(name): np.asarray(payload[name]).copy() for name in payload.files
            }
    if not isinstance(snapshot, EnvSnapshot):
        raise TypeError(f"unexpected snapshot payload type: {type(snapshot)!r}")
    if snapshot_numeric_fingerprint(snapshot) != manifest["snapshot_numeric_sha256"]:
        raise RuntimeError(f"numeric snapshot fingerprint mismatch at {path}")
    return SnapshotBundle(
        path=path,
        manifest=manifest,
        ep_meta=ep_meta,
        model_xml=model_xml,
        snapshot=snapshot,
        canonical_images=canonical_images,
    )


def reconstruct_and_restore(env: Any, bundle: SnapshotBundle) -> None:
    """Rebuild the stored scene XML in ``env`` and restore all runtime state."""

    base = inner_env(env)
    if hasattr(base, "set_attrs_from_ep_meta"):
        base.set_attrs_from_ep_meta(bundle.ep_meta)
    elif hasattr(base, "set_ep_meta"):
        base.set_ep_meta(bundle.ep_meta)
    else:
        raise TypeError("RoboCasa environment cannot restore episode metadata")

    # Match RoboCasa's official dataset playback order. The hard reset rebuilds
    # Python fixture/controller registries; reset_from_xml_string then installs
    # the exact persisted MuJoCo scene.
    base.reset()
    xml = base.edit_model_xml(bundle.model_xml)
    base.reset_from_xml_string(xml)
    base.sim.reset()

    actual_signature = model_signature(env)
    expected_signature = bundle.manifest["model_signature"]
    if actual_signature != expected_signature:
        raise RuntimeError(
            "reconstructed RoboCasa model signature differs from snapshot: "
            f"expected={expected_signature}, actual={actual_signature}"
        )
    restore_snapshot(env, bundle.snapshot)
    actual_fingerprint = env_numeric_fingerprint(env)
    expected_fingerprint = bundle.manifest["snapshot_numeric_sha256"]
    if actual_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "restored numeric state differs from persisted snapshot: "
            f"expected={expected_fingerprint}, actual={actual_fingerprint}"
        )


def refresh_bundle_observation(
    env: Any, bundle: SnapshotBundle
) -> tuple[dict[str, Any], str]:
    """Refresh sensors and install the bundle's persisted task instruction.

    Some RoboCasa task classes regenerate wording such as ``left`` / ``right``
    from transient Python attributes that are not part of MuJoCo state. The
    original instruction in ``ep_meta.json`` is the experimental ground truth,
    so evaluation must never silently replace it with regenerated wording.

    Returns the policy observation and the environment-regenerated instruction
    for audit purposes.
    """

    expected_prompt = str(bundle.manifest["metadata"]["prompt"])
    ep_meta_prompt = str(bundle.ep_meta.get("lang", ""))
    if ep_meta_prompt != expected_prompt:
        raise RuntimeError(
            "persisted prompt differs between ep_meta and manifest: "
            f"{ep_meta_prompt!r} != {expected_prompt!r}"
        )
    observation = refresh_observation(env)
    prompt_key = "annotation.human.task_description"
    regenerated_prompt = str(observation[prompt_key])
    observation[prompt_key] = expected_prompt
    return observation, regenerated_prompt

