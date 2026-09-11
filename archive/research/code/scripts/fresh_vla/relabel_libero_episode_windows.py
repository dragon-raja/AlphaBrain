from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

from build_libero_episode_windows import assign_window_labels, build_quality_report


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relabel(window_root: Path, *, seed: int, apply: bool) -> dict[str, Any]:
    labels_path = window_root / "training_labels.json"
    quality_path = window_root / "quality_report.json"
    manifest = json.loads((window_root / "manifest.json").read_text())
    records = [json.loads(line) for line in (window_root / "records.jsonl").read_text().splitlines() if line]
    horizon = int(manifest["horizon"])
    labels = assign_window_labels(records, horizon=horizon, seed=seed)
    quality = build_quality_report(records, labels, horizon=horizon)
    old_quality = json.loads(quality_path.read_text())
    if "source_initial_state_disjoint" in old_quality["checks"]:
        quality["checks"]["source_initial_state_disjoint"] = bool(
            old_quality["checks"]["source_initial_state_disjoint"]
        )
    for key in ("split_group_counts", "split_source_counts"):
        if key in old_quality.get("metrics", {}):
            quality["metrics"][key] = old_quality["metrics"][key]
    quality["passed"] = bool(all(quality["checks"].values()))
    if not quality["passed"]:
        raise RuntimeError(f"relabelled dataset failed quality checks: {quality['checks']}")

    old_sha = _sha256(labels_path)
    payload = {"schema_version": 1, "horizon": horizon, "records": labels}
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    new_sha = hashlib.sha256(encoded).hexdigest()
    result = {
        "window_root": str(window_root),
        "record_count": len(records),
        "seed": seed,
        "old_labels_sha256": old_sha,
        "new_labels_sha256": new_sha,
        "changed": old_sha != new_sha,
        "quality_checks": quality["checks"],
        "applied": apply,
    }
    if not apply:
        return result

    backup = window_root / "training_labels.before_sample_marginal_fix.json"
    if backup.exists():
        if _sha256(backup) != old_sha:
            raise FileExistsError(f"existing backup does not match current labels: {backup}")
    else:
        temporary_backup = backup.with_name(f".{backup.name}.tmp-{os.getpid()}")
        shutil.copyfile(labels_path, temporary_backup)
        temporary_backup.replace(backup)
    _atomic_write_json(labels_path, payload)
    _atomic_write_json(quality_path, quality)
    _atomic_write_json(
        window_root / "label_revision.json",
        {
            **result,
            "reason": "preserve Oracle/Shuffled-Oracle sample marginals within split and window-group multiplicity",
            "backup": str(backup),
        },
    )
    if _sha256(labels_path) != new_sha:
        raise RuntimeError("atomic label rewrite checksum mismatch")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate only FRESH full-episode loss labels")
    parser.add_argument("--window-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(relabel(args.window_root, seed=args.seed, apply=args.apply), sort_keys=True))


if __name__ == "__main__":
    main()
