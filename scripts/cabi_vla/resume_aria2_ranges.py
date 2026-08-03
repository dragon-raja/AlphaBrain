from __future__ import annotations

import argparse
import hashlib
import math
import os
import struct
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterable, Iterator
from pathlib import Path


def parse_control(path: Path) -> tuple[int, int, list[bool]]:
    payload = path.read_bytes()
    if len(payload) < 34:
        raise ValueError(f"aria2 control file is too short: {path}")
    version = struct.unpack(">H", payload[0:2])[0]
    if version != 1:
        raise ValueError(f"unsupported aria2 control version: {version}")
    piece_length = struct.unpack(">I", payload[10:14])[0]
    total_length = struct.unpack(">Q", payload[14:22])[0]
    bitfield_length = struct.unpack(">I", payload[30:34])[0]
    if piece_length <= 0 or total_length <= 0 or bitfield_length <= 0:
        raise ValueError("invalid aria2 piece, total, or bitfield length")
    bitfield = payload[34 : 34 + bitfield_length]
    if len(bitfield) != bitfield_length:
        raise ValueError("truncated aria2 bitfield")
    piece_count = math.ceil(total_length / piece_length)
    if bitfield_length * 8 < piece_count:
        raise ValueError("aria2 bitfield cannot represent every piece")
    completed = [
        bool(bitfield[index // 8] & (0x80 >> (index % 8)))
        for index in range(piece_count)
    ]
    return piece_length, total_length, completed


def missing_byte_ranges(
    piece_length: int,
    total_length: int,
    completed: list[bool],
) -> list[tuple[int, int]]:
    missing = [index for index, done in enumerate(completed) if not done]
    if not missing:
        return []
    piece_ranges: list[list[int]] = []
    for index in missing:
        if not piece_ranges or index != piece_ranges[-1][1] + 1:
            piece_ranges.append([index, index])
        else:
            piece_ranges[-1][1] = index
    return [
        (
            first * piece_length,
            min(total_length - 1, (last + 1) * piece_length - 1),
        )
        for first, last in piece_ranges
    ]


def sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _download_range(
    *,
    url: str,
    start: int,
    end: int,
    part_dir: Path,
) -> Path:
    expected = end - start + 1
    part_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"range-{start}-{end}.",
        suffix=".part",
        dir=part_dir,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        subprocess.run(
            [
                "curl",
                "-fsSL",
                "--retry",
                "20",
                "--retry-all-errors",
                "--retry-delay",
                "5",
                "--connect-timeout",
                "30",
                "--max-time",
                "900",
                "--range",
                f"{start}-{end}",
                "--output",
                str(temporary),
                url,
            ],
            check=True,
        )
        actual = temporary.stat().st_size
        if actual != expected:
            raise ValueError(
                f"range {start}-{end} returned {actual} bytes, expected {expected}"
            )
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_range(target: Path, source: Path, *, offset: int) -> None:
    with target.open("r+b") as output, source.open("rb") as input_stream:
        output.seek(offset)
        while chunk := input_stream.read(8 * 1024 * 1024):
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())


def repair(
    *,
    target: Path,
    control: Path,
    url: str,
    expected_size: int,
    expected_sha256: str,
    part_dir: Path,
    workers: int,
) -> dict[str, int | str]:
    piece_length, total_length, completed = parse_control(control)
    if total_length != expected_size:
        raise ValueError(
            f"control total {total_length} does not match expected {expected_size}"
        )
    if not target.exists():
        raise FileNotFoundError(target)
    with target.open("r+b") as stream:
        stream.truncate(total_length)

    ranges = missing_byte_ranges(piece_length, total_length, completed)
    parts: list[Path] = []
    repaired_bytes = 0
    try:
        part_by_range: dict[tuple[int, int], Path] = {}
        with ThreadPoolExecutor(max_workers=min(workers, len(ranges) or 1)) as pool:
            futures = {
                pool.submit(
                    _download_range,
                    url=url,
                    start=start,
                    end=end,
                    part_dir=part_dir,
                ): (index, start, end)
                for index, (start, end) in enumerate(ranges, start=1)
            }
            for future in as_completed(futures):
                index, start, end = futures[future]
                part = future.result()
                parts.append(part)
                part_by_range[(start, end)] = part
                print(
                    f"downloaded range {index}/{len(ranges)}: "
                    f"offset={start} bytes={end - start + 1}",
                    flush=True,
                )

        for index, (start, end) in enumerate(ranges, start=1):
            size = end - start + 1
            print(
                f"writing range {index}/{len(ranges)}: "
                f"offset={start} bytes={size}",
                flush=True,
            )
            part = part_by_range[(start, end)]
            _write_range(target, part, offset=start)
            repaired_bytes += size

        actual_sha256 = sha256(target)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"repaired file SHA256 mismatch: {actual_sha256}"
            )
        completed_control = control.with_name(f"{control.name}.completed")
        if completed_control.exists():
            raise FileExistsError(completed_control)
        os.replace(control, completed_control)
        return {
            "range_count": len(ranges),
            "repaired_bytes": repaired_bytes,
            "total_bytes": total_length,
            "sha256": actual_sha256,
        }
    finally:
        for part in parts:
            part.unlink(missing_ok=True)


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair aria2 sparse files with HTTP ranges")
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--part-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    if len(args.expected_sha256) != 64:
        raise ValueError("expected SHA256 must contain 64 hexadecimal characters")
    int(args.expected_sha256, 16)
    if args.workers <= 0 or args.workers > 8:
        raise ValueError("workers must be in [1, 8]")
    result = repair(
        target=args.file,
        control=args.control,
        url=args.url,
        expected_size=args.expected_size,
        expected_sha256=args.expected_sha256.lower(),
        part_dir=args.part_dir,
        workers=args.workers,
    )
    print(result)


if __name__ == "__main__":
    main()
