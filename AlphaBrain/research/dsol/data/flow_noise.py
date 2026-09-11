#!/usr/bin/env python3
"""Materialize and load auditable per-replan flow-noise banks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float32)
    return sha256_bytes(array.tobytes(order="C"))


def stable_uint64(label: str, *, root_seed: int) -> int:
    payload = f"{root_seed}::{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def materialize_bank(
    *,
    output_dir: Path,
    bank_id: str,
    state_keys: Sequence[str],
    repeat_count: int,
    max_replans: int,
    action_horizon: int,
    action_dim: int,
    root_seed: int,
) -> dict[str, Any]:
    if len(state_keys) != len(set(state_keys)) or not state_keys:
        raise ValueError("state keys must be nonempty and unique")
    if min(repeat_count, max_replans, action_horizon, action_dim) <= 0:
        raise ValueError("all bank dimensions must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    noise_path = output_dir / f"bank_{bank_id}.npy"
    if noise_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen noise bank: {noise_path}")
    shape = (
        len(state_keys),
        repeat_count,
        max_replans,
        action_horizon,
        action_dim,
    )
    bank = np.lib.format.open_memmap(noise_path, mode="w+", dtype=np.float32, shape=shape)
    seed_digest = hashlib.sha256()
    for state_index, state_key in enumerate(state_keys):
        for repeat_id in range(repeat_count):
            for replan_index in range(max_replans):
                seed = stable_uint64(
                    f"{bank_id}::{state_key}::{repeat_id}::{replan_index}",
                    root_seed=root_seed,
                )
                seed_digest.update(seed.to_bytes(8, "little"))
                bank[state_index, repeat_id, replan_index] = np.random.Generator(
                    np.random.PCG64(seed)
                ).standard_normal((action_horizon, action_dim), dtype=np.float32)
    bank.flush()
    del bank
    manifest = {
        "schema": "dsol_explicit_flow_noise_bank_v1",
        "status": "MATERIALIZED",
        "bank_id": bank_id,
        "distribution": "numpy_pcg64_iid_standard_normal_float32",
        "root_seed": root_seed,
        "seed_derivation": "sha256(root_seed::bank_id::state_key::repeat_id::replan_index)[:8]_little_uint64",
        "seed_stream_sha256": seed_digest.hexdigest(),
        "state_keys": list(state_keys),
        "repeat_count": repeat_count,
        "max_replans": max_replans,
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "shape": list(shape),
        "dtype": "float32",
        "noise_file": str(noise_path.resolve()),
        "noise_file_size": noise_path.stat().st_size,
        "noise_file_sha256": sha256_file(noise_path),
    }
    manifest_path = output_dir / f"bank_{bank_id}.manifest.json"
    atomic_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path.resolve())
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


class ExplicitFlowNoiseBank:
    def __init__(self, manifest_path: Path, *, verify_file: bool = True) -> None:
        self.manifest_path = manifest_path.resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != "dsol_explicit_flow_noise_bank_v1":
            raise ValueError("unsupported explicit flow-noise bank schema")
        self.noise_path = Path(self.manifest["noise_file"]).resolve()
        if verify_file and sha256_file(self.noise_path) != self.manifest["noise_file_sha256"]:
            raise ValueError("explicit flow-noise bank SHA-256 mismatch")
        self.array = np.load(self.noise_path, mmap_mode="r")
        if list(self.array.shape) != list(self.manifest["shape"]) or self.array.dtype != np.float32:
            raise ValueError("explicit flow-noise bank shape or dtype mismatch")
        self.state_indices = {
            key: index for index, key in enumerate(self.manifest["state_keys"])
        }

    def get(self, state_key: str, repeat_id: int, replan_index: int) -> dict[str, Any]:
        if state_key not in self.state_indices:
            raise KeyError(f"state is absent from noise bank: {state_key}")
        if not 0 <= repeat_id < int(self.manifest["repeat_count"]):
            raise IndexError(f"repeat ID outside noise bank: {repeat_id}")
        if not 0 <= replan_index < int(self.manifest["max_replans"]):
            raise IndexError(f"replan index outside noise bank: {replan_index}")
        seed = stable_uint64(
            f"{self.manifest['bank_id']}::{state_key}::{repeat_id}::{replan_index}",
            root_seed=int(self.manifest["root_seed"]),
        )
        noise = np.asarray(
            self.array[self.state_indices[state_key], repeat_id, replan_index],
            dtype=np.float32,
        )
        return {"noise": noise, "noise_seed": seed, "noise_sha256": tensor_sha256(noise)}


def state_keys_from_population(payload: Mapping[str, Any], split: str) -> list[str]:
    population = payload["population"][split]
    keys = [str(row["pair_key"]) for row in population["states"]]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate pair keys in population split: {split}")
    return sorted(keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-replans", type=int, default=104)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--root-seed", type=int, default=20260901)
    args = parser.parse_args()
    population = json.loads(args.population.read_text(encoding="utf-8"))
    definitions = (
        ("A", "calibration", 4),
        ("B", "calibration", 8),
        ("C", "calibration", 16),
        ("D", "calibration", 64),
        ("E", "heldout_test", 32),
        ("F", "heldout_test", 32),
    )
    manifests = []
    for offset, (bank_id, split, repeats) in enumerate(definitions):
        manifests.append(
            materialize_bank(
                output_dir=args.output_dir,
                bank_id=bank_id,
                state_keys=state_keys_from_population(population, split),
                repeat_count=repeats,
                max_replans=args.max_replans,
                action_horizon=args.action_horizon,
                action_dim=args.action_dim,
                root_seed=args.root_seed + offset * 1_000_003,
            )
        )
    receipt = {
        "schema": "dsol_explicit_flow_noise_banks_receipt_v1",
        "status": "PASS",
        "population": str(args.population.resolve()),
        "population_sha256": sha256_file(args.population),
        "banks": manifests,
    }
    atomic_json(args.output_dir / "noise_banks_receipt.json", receipt)
    print(json.dumps({"status": "PASS", "banks": len(manifests)}, sort_keys=True))


if __name__ == "__main__":
    main()
