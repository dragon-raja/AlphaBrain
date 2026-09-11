from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from resume_aria2_ranges import _download_range, _write_range, sha256


SCHEMA_VERSION = 1


def chunk_ranges(total_bytes: int, chunk_bytes: int) -> list[tuple[int, int]]:
    if total_bytes <= 0 or chunk_bytes <= 0:
        raise ValueError("total and chunk byte counts must be positive")
    return [
        (start, min(total_bytes - 1, start + chunk_bytes - 1))
        for start in range(0, total_bytes, chunk_bytes)
    ]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _initial_state(
    *,
    url: str,
    expected_size: int,
    expected_sha256: str,
    chunk_bytes: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "partial",
        "url": url,
        "expected_size": expected_size,
        "expected_sha256": expected_sha256,
        "chunk_bytes": chunk_bytes,
        "completed_chunks": [],
    }


def _load_or_create_state(
    *,
    output: Path,
    state_path: Path,
    url: str,
    expected_size: int,
    expected_sha256: str,
    chunk_bytes: int,
) -> dict[str, Any]:
    expected = _initial_state(
        url=url,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        chunk_bytes=chunk_bytes,
    )
    if state_path.exists():
        state = json.loads(state_path.read_text())
        for key in (
            "schema_version",
            "url",
            "expected_size",
            "expected_sha256",
            "chunk_bytes",
        ):
            if state.get(key) != expected[key]:
                raise ValueError(f"range-download state mismatch for {key}")
        if not output.exists():
            raise FileNotFoundError(f"state exists but output is missing: {output}")
        if output.stat().st_size != expected_size:
            raise ValueError("range-download output has unexpected logical size")
        return state
    if output.exists():
        raise FileExistsError(f"refusing to overwrite untracked output: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.truncate(expected_size)
    _atomic_json(state_path, expected)
    return expected


def download(
    *,
    url: str,
    output: Path,
    expected_size: int,
    expected_sha256: str,
    chunk_bytes: int,
    workers: int,
    part_dir: Path,
) -> dict[str, Any]:
    state_path = output.with_name(f"{output.name}.ranges.json")
    state = _load_or_create_state(
        output=output,
        state_path=state_path,
        url=url,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        chunk_bytes=chunk_bytes,
    )
    ranges = chunk_ranges(expected_size, chunk_bytes)
    completed = {int(index) for index in state.get("completed_chunks", [])}
    if any(index < 0 or index >= len(ranges) for index in completed):
        raise ValueError("state contains an invalid completed chunk index")

    part_dir.mkdir(parents=True, exist_ok=True)
    for stale in part_dir.glob("*.part"):
        stale.unlink()

    pending = [
        (index, start, end)
        for index, (start, end) in enumerate(ranges)
        if index not in completed
    ]
    if pending:
        with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            futures = {
                pool.submit(
                    _download_range,
                    url=url,
                    start=start,
                    end=end,
                    part_dir=part_dir,
                ): (index, start, end)
                for index, start, end in pending
            }
            for future in as_completed(futures):
                index, start, end = futures[future]
                part = future.result()
                try:
                    _write_range(output, part, offset=start)
                finally:
                    part.unlink(missing_ok=True)
                completed.add(index)
                state["completed_chunks"] = sorted(completed)
                _atomic_json(state_path, state)
                print(
                    f"completed chunk {len(completed)}/{len(ranges)}: "
                    f"offset={start} bytes={end - start + 1}",
                    flush=True,
                )

    actual_sha256 = sha256(output)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"downloaded file SHA256 mismatch: {actual_sha256}")
    state["status"] = "complete"
    state["actual_sha256"] = actual_sha256
    _atomic_json(state_path, state)
    return {
        "output": str(output),
        "chunk_count": len(ranges),
        "total_bytes": expected_size,
        "sha256": actual_sha256,
    }


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and verify a public file by HTTP ranges")
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--chunk-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--part-dir", type=Path, required=True)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.workers > 8:
        raise ValueError("workers must be in [1, 8]")
    if args.chunk_bytes < 1024 * 1024:
        raise ValueError("chunk bytes must be at least 1 MiB")
    if len(args.expected_sha256) != 64:
        raise ValueError("expected SHA256 must contain 64 hexadecimal characters")
    int(args.expected_sha256, 16)
    result = download(
        url=args.url,
        output=args.output,
        expected_size=args.expected_size,
        expected_sha256=args.expected_sha256.lower(),
        chunk_bytes=args.chunk_bytes,
        workers=args.workers,
        part_dir=args.part_dir,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
