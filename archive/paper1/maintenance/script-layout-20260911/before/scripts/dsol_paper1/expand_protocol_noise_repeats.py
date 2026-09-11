#!/usr/bin/env python3
"""Expand a frozen evaluation protocol over explicitly recorded noise seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def build(
    protocol: Mapping[str, Any],
    seeds: Sequence[int],
    *,
    source_path: Path,
) -> dict[str, Any]:
    if protocol.get("status") != "PASS" or not protocol.get("specs"):
        raise ValueError("source protocol must PASS and contain specs")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("noise seeds must be nonempty and unique")
    specs = []
    for seed in seeds:
        for source in protocol["specs"]:
            row = dict(source)
            base_episode_id = str(source["episode_id"])
            identity = f"{base_episode_id}::noise::{seed}"
            row.update(
                {
                    "base_episode_id": base_episode_id,
                    "episode_id": hashlib.sha256(identity.encode()).hexdigest()[:20],
                    "evaluation_seed": int(seed),
                }
            )
            specs.append(row)
    payload = dict(protocol)
    payload.update(
        {
            "schema": "dsol_noise_repeated_evaluation_protocol_v1",
            "base_protocol_schema": protocol.get("schema"),
            "base_protocol": str(source_path.resolve()),
            "base_protocol_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "noise_seeds": [int(seed) for seed in seeds],
            "noise_repeat_count": len(seeds),
            "base_episode_count": len(protocol["specs"]),
            "episode_count": len(specs),
            "specs": specs,
        }
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    payload = build(
        json.loads(args.protocol.read_text(encoding="utf-8")),
        seeds,
        source_path=args.protocol,
    )
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "base_episodes": payload["base_episode_count"],
                "noise_repeats": payload["noise_repeat_count"],
                "episodes": payload["episode_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
