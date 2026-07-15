from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

import av
from video_io import write_h264_video


@dataclass(frozen=True)
class VideoInfo:
    codec: str
    codec_tag: str
    pixel_format: str
    width: int
    height: int
    frame_count: int
    fps: float


@dataclass(frozen=True)
class TranscodeResult:
    path: str
    status: str
    backup: str | None
    before: VideoInfo
    after: VideoInfo


def inspect_video(path: Path, *, decode: bool = True) -> VideoInfo:
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.base_rate or Fraction(10, 1)
        frame_count = sum(1 for _ in container.decode(stream)) if decode else int(stream.frames or 0)
        return VideoInfo(
            codec=str(stream.codec_context.name or ""),
            codec_tag=str(stream.codec_context.codec_tag or ""),
            pixel_format=str(stream.codec_context.pix_fmt or ""),
            width=int(stream.codec_context.width),
            height=int(stream.codec_context.height),
            frame_count=frame_count,
            fps=float(rate),
        )


def is_compatible(info: VideoInfo) -> bool:
    return info.codec == "h264" and info.codec_tag == "avc1" and info.pixel_format == "yuv420p"


def _decoded_frames(path: Path, frame_counter: list[int]) -> tuple[Iterable, float]:
    container = av.open(str(path), mode="r")
    stream = container.streams.video[0]
    rate = stream.average_rate or stream.base_rate or Fraction(10, 1)

    def frames():
        try:
            for frame in container.decode(stream):
                frame_counter[0] += 1
                yield frame.to_ndarray(format="rgb24")
        finally:
            container.close()

    return frames(), float(rate)


def _backup_path(path: Path, backup_root: Path) -> Path:
    absolute = path.resolve()
    return backup_root / absolute.relative_to(absolute.anchor)


def transcode_video(path: Path, backup_root: Path, *, crf: int = 23, preset: str = "fast") -> TranscodeResult:
    path = path.resolve()
    before = inspect_video(path)
    if is_compatible(before):
        return TranscodeResult(str(path), "already-compatible", None, before, before)

    temporary = path.with_name(f".{path.stem}.{os.getpid()}.avc1.mp4")
    backup = _backup_path(path, backup_root.resolve())
    if backup.exists():
        raise FileExistsError(f"refusing to overwrite existing backup: {backup}")

    frame_counter = [0]
    frames, fps = _decoded_frames(path, frame_counter)
    try:
        write_h264_video(temporary, frames, fps=fps, crf=crf, preset=preset)
        after = inspect_video(temporary)
        if not is_compatible(after):
            raise RuntimeError(f"incompatible transcoded stream for {path}: {after}")
        if after.frame_count != frame_counter[0] or after.frame_count != before.frame_count:
            raise RuntimeError(
                f"frame count changed for {path}: before={before.frame_count} "
                f"decoded={frame_counter[0]} after={after.frame_count}"
            )
        if (after.width, after.height) != (before.width + before.width % 2, before.height + before.height % 2):
            raise RuntimeError(f"dimensions changed unexpectedly for {path}: before={before} after={after}")

        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(backup))
        try:
            os.replace(temporary, path)
        except Exception:
            shutil.move(str(backup), str(path))
            raise
        return TranscodeResult(str(path), "transcoded", str(backup), before, after)
    finally:
        temporary.unlink(missing_ok=True)


def discover_videos(inputs: Sequence[Path], backup_root: Path) -> list[Path]:
    backup_root = backup_root.resolve()
    videos = set()
    for value in inputs:
        path = value.resolve()
        if path == backup_root or backup_root in path.parents:
            continue
        if path.is_file():
            if path.suffix.lower() == ".mp4":
                videos.add(path)
        elif path.is_dir():
            for video in path.rglob("*.mp4"):
                resolved = video.resolve()
                if resolved != backup_root and backup_root not in resolved.parents:
                    videos.add(resolved)
        else:
            raise FileNotFoundError(path)
    return sorted(videos)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcode FRESH-VLA MP4 videos to VS Code compatible H.264/avc1")
    parser.add_argument("inputs", nargs="+", type=Path, help="Video files or directories to scan recursively")
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=Path("/share/longjunyu/fresh-vla/video_mp4v_backup"),
        help="Root that preserves original absolute paths",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preset", default="fast")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    videos = discover_videos(args.inputs, args.backup_root)
    if args.dry_run:
        payload = {"count": len(videos), "paths": [str(path) for path in videos]}
    else:
        if args.workers <= 0:
            raise ValueError("workers must be positive")
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(transcode_video, path, args.backup_root, crf=args.crf, preset=args.preset): path
                for path in videos
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                print(f"[{completed}/{len(videos)}] {result.status}: {result.path}", flush=True)
        payload = {
            "count": len(results),
            "transcoded": sum(result.status == "transcoded" for result in results),
            "already_compatible": sum(result.status == "already-compatible" for result in results),
            "results": [
                {
                    **asdict(result),
                    "before": asdict(result.before),
                    "after": asdict(result.after),
                }
                for result in sorted(results, key=lambda item: item.path)
            ],
        }

    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
