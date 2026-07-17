from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


def _load(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_fingerprint(group: Mapping[str, Any]) -> str:
    payload = {
        "initial_state_index": group.get("initial_state_index"),
        "randomization": group.get("randomization"),
        "task": group.get("task"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_seal(
    development_source: Path,
    confirmation_source: Path,
    confirmation_episodes: Path,
    *,
    expected_groups: int,
) -> dict[str, Any]:
    development = _load(development_source / "manifest.json")
    source = _load(confirmation_source / "manifest.json")
    episodes = _load(confirmation_episodes / "manifest.json")
    source_quality = _load(confirmation_source / "quality_report.json")
    episode_quality = _load(confirmation_episodes / "quality_report.json")
    if not source_quality.get("passed") or not episode_quality.get("passed"):
        raise ValueError("confirmation generation quality gate did not pass")

    source_groups = [row for row in source["pairs"] if row["task"] == "grasp_slip"]
    episode_groups = list(episodes["groups"])
    if len(source_groups) != expected_groups or len(episode_groups) != expected_groups:
        raise ValueError("confirmation group count does not match the frozen protocol")

    development_fingerprints = {
        _group_fingerprint(row)
        for row in development["pairs"]
        if row["task"] == "grasp_slip"
    }
    confirmation_fingerprints = [_group_fingerprint(row) for row in source_groups]
    if len(set(confirmation_fingerprints)) != expected_groups:
        raise ValueError("confirmation snapshots are not unique")
    if development_fingerprints.intersection(confirmation_fingerprints):
        raise ValueError("confirmation snapshots overlap development snapshots")

    source_ids = [str(row["pair_id"]) for row in source_groups]
    episode_ids = [str(row["pair_id"]) for row in episode_groups]
    if source_ids != episode_ids:
        raise ValueError("source and full-episode confirmation group order changed")

    inventory = [path for path in confirmation_episodes.rglob("*") if path.is_file()]
    return {
        "schema_version": 1,
        "status": "SEALED_FOR_GATE3_ONLY",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "development_source": str(development_source.resolve()),
        "confirmation_source": str(confirmation_source.resolve()),
        "confirmation_episode_root": str(confirmation_episodes.resolve()),
        "group_count": expected_groups,
        "confirmation_groups": [
            {
                "confirmation_id": f"cora-confirmation-{index:04d}",
                "source_pair_id": pair_id,
                "snapshot_fingerprint": fingerprint,
            }
            for index, (pair_id, fingerprint) in enumerate(
                zip(source_ids, confirmation_fingerprints, strict=True)
            )
        ],
        "development_snapshot_overlap": 0,
        "allowed_use": "single formal Gate 3 comparison after method, N, checkpoint, and thresholds are frozen",
        "forbidden_use": [
            "data design",
            "threshold selection",
            "model selection",
            "ablation",
            "debugging",
        ],
        "key_hashes": {
            "source_manifest_sha256": _sha256(confirmation_source / "manifest.json"),
            "source_quality_sha256": _sha256(confirmation_source / "quality_report.json"),
            "episode_manifest_sha256": _sha256(confirmation_episodes / "manifest.json"),
            "episode_quality_sha256": _sha256(confirmation_episodes / "quality_report.json"),
        },
        "episode_inventory": {
            "file_count": len(inventory),
            "total_bytes": sum(path.stat().st_size for path in inventory),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seal untouched CORA confirmation groups")
    parser.add_argument("--development-source", type=Path, required=True)
    parser.add_argument("--confirmation-source", type=Path, required=True)
    parser.add_argument("--confirmation-episodes", type=Path, required=True)
    parser.add_argument("--expected-groups", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_groups < 1:
        raise ValueError("expected-groups must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite seal: {args.output}")
    seal = build_seal(
        args.development_source,
        args.confirmation_source,
        args.confirmation_episodes,
        expected_groups=args.expected_groups,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "status": seal["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
