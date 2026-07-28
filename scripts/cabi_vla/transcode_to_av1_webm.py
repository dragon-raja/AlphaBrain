from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import av

FRESH_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "fresh_vla"
if str(FRESH_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(FRESH_SCRIPT_DIR))

from video_io import write_av1_video


def inspect_video(path: Path) -> dict[str, Any]:
    with av.open(str(path), mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ValueError(f"expected one video stream in {path}")
        stream = streams[0]
        frame_count = sum(1 for _ in container.decode(stream))
        return {
            "codec": str(stream.codec_context.name),
            "width": int(stream.width),
            "height": int(stream.height),
            "frame_count": frame_count,
            "fps": (
                float(stream.average_rate)
                if stream.average_rate is not None
                else None
            ),
        }


def transcode_video(
    source: Path,
    output: Path,
    *,
    crf: int,
    cpu_used: int,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite AV1 video: {output}")
    before = inspect_video(source)
    fps = before["fps"] or 10.0
    with av.open(str(source), mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ValueError(f"expected one video stream in {source}")
        stream = streams[0]
        frames = (
            frame.to_ndarray(format="rgb24")
            for frame in container.decode(stream)
        )
        write_av1_video(
            output,
            frames,
            fps=fps,
            crf=crf,
            cpu_used=cpu_used,
        )
    after = inspect_video(output)
    if after["codec"] not in {"av1", "libaom-av1", "libdav1d"}:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"transcoded video is not AV1: {output}")
    for field in ("width", "height", "frame_count"):
        if after[field] != before[field]:
            output.unlink(missing_ok=True)
            raise RuntimeError(
                f"transcoded {field} changed: {before[field]} -> {after[field]}"
            )
    return {
        "source": str(source),
        "output": str(output),
        "source_video": before,
        "av1_video": after,
    }


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create verified AV1/WebM copies without modifying source videos"
    )
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--relative-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--crf", type=int, default=35)
    parser.add_argument("--cpu-used", type=int, default=8)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args(args)


def main(args: Iterable[str] | None = None) -> None:
    parsed = parse_args(args)
    results = []
    for source in parsed.input:
        relative = source.resolve().relative_to(parsed.relative_root.resolve())
        output = (parsed.output_root / relative).with_suffix(".webm")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            result = {
                "source": str(source),
                "output": str(output),
                "status": "already_exists",
                "av1_video": inspect_video(output),
            }
            if result["av1_video"]["codec"] not in {
                "av1",
                "libaom-av1",
                "libdav1d",
            }:
                raise RuntimeError(f"existing output is not AV1: {output}")
        else:
            result = {
                **transcode_video(
                    source,
                    output,
                    crf=parsed.crf,
                    cpu_used=parsed.cpu_used,
                ),
                "status": "transcoded",
            }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "codec": "AV1",
        "container": "WebM",
        "source_files_preserved": True,
        "results": results,
    }
    if parsed.manifest is not None:
        if parsed.manifest.exists():
            raise FileExistsError(
                f"refusing to overwrite video manifest: {parsed.manifest}"
            )
        parsed.manifest.parent.mkdir(parents=True, exist_ok=True)
        parsed.manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
