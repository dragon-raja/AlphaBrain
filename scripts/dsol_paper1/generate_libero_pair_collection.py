from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _verified_shard(path: Path, source: Path) -> Mapping[str, Any] | None:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "VERIFIED":
        return None
    if Path(manifest.get("source_hdf5", "")).resolve() != source.resolve():
        raise ValueError(f"existing shard source mismatch: {path}")
    shard = path / manifest["shard"]
    records = path / manifest["records"]
    if _sha256(shard) != manifest["shard_sha256"]:
        raise ValueError(f"existing shard checksum mismatch: {shard}")
    if _sha256(records) != manifest["records_sha256"]:
        raise ValueError(f"existing records checksum mismatch: {records}")
    return manifest


def _run_task(
    *,
    task: Mapping[str, Any],
    task_index: int,
    args: argparse.Namespace,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    source = args.hdf5_root / str(task["source"])
    if not source.is_file():
        raise FileNotFoundError(source)
    shard_root = args.output / "shards" / str(task["task_id"])
    existing = _verified_shard(shard_root, source)
    if existing is not None:
        return {"task_id": task["task_id"], "skipped": True, "manifest": existing}

    log_path = args.output / "logs" / f"{task['task_id']}.log"
    config_root = args.config_root / str(task["task_id"])
    command = [
        str(args.python),
        str(args.generator),
        "--hdf5",
        str(source),
        "--runtime",
        str(args.runtime),
        "--catalog",
        str(args.catalog),
        "--acquisition",
        str(args.acquisition),
        "--config-root",
        str(config_root),
        "--output",
        str(shard_root),
        "--pose-set",
        str(plan["pose_set"]),
        "--seed",
        str(plan["seed"]),
        "--resolution",
        str(plan["resolution"]),
        "--action-horizon",
        str(plan["action_horizon"]),
        "--frame-stride",
        str(plan["frame_stride"]),
        "--jpeg-quality",
        str(plan["jpeg_quality"]),
        "--render-gpu",
        str(task_index % args.workers),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(
            f"pair generation failed for {task['task_id']} (see {log_path})"
        )
    manifest = _verified_shard(shard_root, source)
    if manifest is None:
        raise RuntimeError(f"generator did not produce a verified shard: {shard_root}")
    return {"task_id": task["task_id"], "skipped": False, "manifest": manifest}


def build_collection(args: argparse.Namespace) -> dict[str, Any]:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema") != "dsol_libero_pair_collection_plan_v1":
        raise ValueError("unsupported pair collection plan")
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in [1, 8]")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "logs").mkdir(exist_ok=True)
    (args.output / "shards").mkdir(exist_ok=True)
    plan_sha = _sha256(args.plan)
    identity_path = args.output / "plan_identity.json"
    identity = {"plan": str(args.plan.resolve()), "plan_sha256": plan_sha}
    if identity_path.is_file():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("output root already belongs to a different plan")
    else:
        _atomic_json(identity_path, identity)

    failures = []
    completed: dict[str, dict[str, Any]] = {}
    # One serial executor per renderer GPU prevents two long-running tasks from
    # being scheduled onto the same device when task runtimes differ.
    pools = [
        concurrent.futures.ThreadPoolExecutor(max_workers=1)
        for _ in range(args.workers)
    ]
    try:
        futures = {
            pools[index % args.workers].submit(
                _run_task,
                task=task,
                task_index=index,
                args=args,
                plan=plan,
            ): task
            for index, task in enumerate(plan["tasks"])
        }
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                completed[str(task["task_id"])] = result
                print(
                    json.dumps(
                        {
                            "task_id": task["task_id"],
                            "status": "VERIFIED",
                            "skipped": result["skipped"],
                            "record_count": result["manifest"]["record_count"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except Exception as exc:
                failures.append({"task_id": task["task_id"], "error": str(exc)})
                print(json.dumps(failures[-1], sort_keys=True), file=sys.stderr, flush=True)
    finally:
        for pool in pools:
            pool.shutdown(wait=True)

    ordered = []
    total_records = 0
    total_bytes = 0
    counts_by_split = {"train": 0, "val": 0, "test": 0}
    for task in plan["tasks"]:
        task_id = str(task["task_id"])
        if task_id not in completed:
            continue
        shard_manifest = completed[task_id]["manifest"]
        actual = int(shard_manifest["record_count"])
        expected = int(task["expected_records"])
        if actual != expected:
            failures.append(
                {
                    "task_id": task_id,
                    "error": f"expected {expected} records, generated {actual}",
                }
            )
        total_records += actual
        total_bytes += int(shard_manifest["shard_size_bytes"])
        for split, count in shard_manifest["counts_by_split"].items():
            counts_by_split[split] += int(count)
        ordered.append(
            {
                "task_id": task_id,
                "path": f"shards/{task_id}",
                "record_count": actual,
                "shard_sha256": shard_manifest["shard_sha256"],
                "records_sha256": shard_manifest["records_sha256"],
            }
        )

    if total_records != int(plan["expected_record_count"]):
        failures.append(
            {
                "task_id": "collection",
                "error": (
                    f"expected {plan['expected_record_count']} total records, "
                    f"generated {total_records}"
                ),
            }
        )
    manifest = {
        "schema": "dsol_libero_hdf5_view_pair_collection_v1",
        "status": "VERIFIED" if not failures else "FAILED",
        "plan": str(args.plan.resolve()),
        "plan_sha256": plan_sha,
        "source_revision": json.loads(
            args.acquisition.read_text(encoding="utf-8")
        )["revision"],
        "catalog": str(args.catalog.resolve()),
        "record_count": total_records,
        "counts_by_split": counts_by_split,
        "shard_size_bytes": total_bytes,
        "shards": ordered,
        "failures": failures,
    }
    _atomic_json(args.output / "manifest.json", manifest)
    if failures:
        raise RuntimeError(f"collection failed validation: {failures}")
    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a resumable DSOL pair collection.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--hdf5-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args(argv)


def main() -> None:
    manifest = build_collection(parse_args())
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "record_count": manifest["record_count"],
                "counts_by_split": manifest["counts_by_split"],
                "shard_size_bytes": manifest["shard_size_bytes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
