from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

import av
import numpy as np

FRESH_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "fresh_vla"
if str(FRESH_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(FRESH_SCRIPT_DIR))

from video_io import write_av1_video, write_h264_video


def verify_video(path: Path, expected_codec: str) -> dict:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frame_count = sum(1 for _ in container.decode(stream))
        codec = stream.codec_context.name
        pixel_format = stream.codec_context.pix_fmt
        width, height = stream.width, stream.height
    accepted_codecs = {expected_codec}
    if expected_codec == "av1":
        accepted_codecs.add("libdav1d")
    if codec not in accepted_codecs:
        raise ValueError(f"{path} codec {codec!r} != {expected_codec!r}")
    if frame_count == 0:
        raise ValueError(f"video has no decodable frames: {path}")
    return {
        "bitstream_codec": expected_codec,
        "decoder": codec,
        "pixel_format": pixel_format,
        "width": width,
        "height": height,
        "frame_count": frame_count,
    }


def render(
    collection_root: Path,
    output_dir: Path,
    *,
    state_indices: set[int],
    codecs: tuple[str, ...],
) -> dict:
    manifest = json.loads((collection_root / "manifest.json").read_text())
    rows = [
        row
        for row in manifest["rows"]
        if row.get("success")
        and "episode_file" in row
        and int(row["canonical_state_index"]) in state_indices
    ]
    if not rows:
        raise ValueError("no successful episodes matched the requested video states")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    unknown = sorted(set(codecs) - {"h264", "av1"})
    if unknown:
        raise ValueError(f"unsupported video codecs: {unknown}")

    staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=False)
    rendered = []
    try:
        for row in rows:
            with np.load(collection_root / row["episode_file"], allow_pickle=False) as arrays:
                frames = np.concatenate(
                    [np.asarray(arrays["agentview"]), np.asarray(arrays["wrist"])],
                    axis=2,
                )
            sample_id = row["sample_id"]
            outputs = {}
            if "h264" in codecs:
                path = staging / "h264" / f"{sample_id}.mp4"
                write_h264_video(path, frames, fps=10.0)
                outputs["h264"] = {
                    "path": str(Path("h264") / path.name),
                    **verify_video(path, "h264"),
                }
            if "av1" in codecs:
                path = staging / "av1" / f"{sample_id}.webm"
                write_av1_video(path, frames, fps=10.0)
                outputs["av1"] = {
                    "path": str(Path("av1") / path.name),
                    **verify_video(path, "av1"),
                }
            rendered.append({"sample_id": sample_id, "outputs": outputs})
        report = {
            "schema_version": 1,
            "source_collection": str(collection_root),
            "state_indices": sorted(state_indices),
            "video_count": len(rendered),
            "codecs": list(codecs),
            "videos": rendered,
        }
        (staging / "manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        staging.rename(output_dir)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def parse_indices(value: str) -> set[int]:
    result = {int(part.strip()) for part in value.split(",") if part.strip()}
    if not result or min(result) < 0:
        raise ValueError("state indices must be non-negative")
    return result


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render verified LIBERO-Bind QA videos")
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--state-indices", default="0")
    parser.add_argument("--codecs", default="h264,av1")
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    report = render(
        args.collection_root,
        args.output_dir,
        state_indices=parse_indices(args.state_indices),
        codecs=tuple(value.strip() for value in args.codecs.split(",") if value.strip()),
    )
    print(json.dumps({"output_dir": str(args.output_dir), "video_count": report["video_count"]}))


if __name__ == "__main__":
    main()
