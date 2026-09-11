from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile


REPO_ID = "yifengzhu-hf/LIBERO-datasets"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def inventory(api: HfApi) -> tuple[str, list[dict[str, object]]]:
    revision = api.dataset_info(REPO_ID, token=False).sha
    records: list[dict[str, object]] = []
    for entry in api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True, expand=True, revision=revision):
        if not isinstance(entry, RepoFile) or not entry.path.endswith(".hdf5"):
            continue
        suite = entry.path.split("/", 1)[0]
        if suite not in SUITES:
            continue
        if entry.lfs is None:
            raise RuntimeError(f"missing LFS identity for {entry.path}")
        records.append(
            {
                "expected_sha256": entry.lfs.sha256,
                "path": entry.path,
                "size_bytes": entry.size,
                "status": "PENDING",
            }
        )
    records.sort(key=lambda record: str(record["path"]))
    if len(records) != 40:
        raise RuntimeError(f"expected 40 HDF5 files, found {len(records)}")
    return revision, records


def download_one(target: Path, revision: str, record: dict[str, object]) -> tuple[str, str]:
    relative = str(record["path"])
    expected_sha256 = str(record["expected_sha256"])
    expected_size = int(record["size_bytes"])
    destination = target / relative

    if destination.is_file() and destination.stat().st_size == expected_size:
        actual_sha256 = sha256_file(destination)
        if actual_sha256 == expected_sha256:
            return relative, "VERIFIED_EXISTING"

    downloaded = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=relative,
            repo_type="dataset",
            revision=revision,
            local_dir=target,
            token=False,
        )
    )
    if downloaded.stat().st_size != expected_size:
        raise RuntimeError(f"size mismatch for {relative}")
    if sha256_file(downloaded) != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {relative}")
    return relative, "VERIFIED_DOWNLOADED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify the 40-task canonical LIBERO HDF5 data.")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers must be between 1 and 8")
    target = args.target.resolve()
    receipt = args.receipt.resolve()
    target.mkdir(parents=True, exist_ok=True)
    receipt.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=False)
    revision, records = inventory(api)
    by_path = {str(record["path"]): record for record in records}
    state: dict[str, object] = {
        "completed_at": None,
        "expected_files": len(records),
        "expected_size_bytes": sum(int(record["size_bytes"]) for record in records),
        "files": records,
        "repo_id": REPO_ID,
        "revision": revision,
        "schema": "libero_hdf5_acquisition_v1",
        "started_at": utc_now(),
        "status": "DOWNLOADING",
        "suites": list(SUITES),
        "target": os.fspath(target),
        "workers": args.workers,
    }
    state_path = receipt / "acquisition.json"
    atomic_json(state_path, state)
    print(
        f"inventory files={len(records)} bytes={state['expected_size_bytes']} "
        f"revision={revision} workers={args.workers}",
        flush=True,
    )

    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures: dict[Future[tuple[str, str]], str] = {
            executor.submit(download_one, target, revision, record): str(record["path"]) for record in records
        }
        for future in as_completed(futures):
            relative = futures[future]
            try:
                _, status = future.result()
                by_path[relative]["status"] = status
                print(f"{status} {relative}", flush=True)
            except Exception as exc:
                by_path[relative]["status"] = "FAILED"
                failures.append((relative, type(exc).__name__))
                print(f"FAILED {relative} {type(exc).__name__}", flush=True)
            atomic_json(state_path, state)

    state["completed_at"] = utc_now()
    state["status"] = "VERIFIED" if not failures else "FAILED"
    state["verified_files"] = sum(str(record["status"]).startswith("VERIFIED") for record in records)
    state["failures"] = [{"error_type": error, "path": path} for path, error in failures]
    atomic_json(state_path, state)
    print(f"status={state['status']} verified={state['verified_files']}/{len(records)}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
