from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from render_libero_bind_videos import verify_video

import sys

FRESH_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "fresh_vla"
if str(FRESH_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(FRESH_SCRIPT_DIR))

from video_io import write_av1_video, write_h264_video


def make_contact_sheet(
    frames: np.ndarray,
    *,
    tile_count: int = 16,
    columns: int = 4,
) -> tuple[np.ndarray, list[int]]:
    values = np.asarray(frames)
    if values.ndim != 4 or values.shape[-1] != 3 or len(values) == 0:
        raise ValueError(f"contact-sheet frames must be non-empty NxHxWx3, got {values.shape}")
    if tile_count <= 0 or columns <= 0:
        raise ValueError("tile count and columns must be positive")
    indices = np.linspace(0, len(values) - 1, tile_count).round().astype(int).tolist()
    height, width = values.shape[1:3]
    rows = (tile_count + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * height), color="black")
    for position, frame_index in enumerate(indices):
        tile = Image.fromarray(np.asarray(values[frame_index], dtype=np.uint8), mode="RGB")
        if height >= 24 and width >= 60:
            draw = ImageDraw.Draw(tile)
            label = f"t={frame_index}"
            label_width = min(width - 1, max(52, 8 * len(label)))
            draw.rectangle((0, 0, label_width, 20), fill="black")
            draw.text((5, 4), label, fill="white")
        canvas.paste(tile, ((position % columns) * width, (position // columns) * height))
    return np.asarray(canvas), indices


def render_evaluation(
    evaluation_path: Path,
    frame_dir: Path,
    output_dir: Path,
    *,
    codecs: tuple[str, ...],
    fps: float,
) -> dict:
    evaluation = json.loads(evaluation_path.read_text())
    rows = [row for row in evaluation["rows"] if "frame_file" in row]
    if not rows:
        raise ValueError("evaluation has no recorded frame episodes")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    unknown = sorted(set(codecs) - {"h264", "av1"})
    if unknown:
        raise ValueError(f"unsupported codecs: {unknown}")
    if fps <= 0:
        raise ValueError("fps must be positive")

    staging = output_dir.parent / f".{output_dir.name}.staging-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=False)
    rendered = []
    try:
        for row in rows:
            source = frame_dir / row["frame_file"]
            with np.load(source, allow_pickle=False) as archive:
                frames = np.asarray(archive["frames"])
            stem = source.stem
            outputs = {}
            contact_sheet, contact_indices = make_contact_sheet(frames)
            contact_path = staging / "contact_sheets" / f"{stem}.jpg"
            contact_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(contact_sheet).save(contact_path, quality=90)
            outputs["contact_sheet"] = {
                "path": str(Path("contact_sheets") / contact_path.name),
                "width": int(contact_sheet.shape[1]),
                "height": int(contact_sheet.shape[0]),
                "sampled_frame_indices": contact_indices,
            }
            if "h264" in codecs:
                path = staging / "h264" / f"{stem}.mp4"
                write_h264_video(path, frames, fps=fps)
                outputs["h264"] = {
                    "path": str(Path("h264") / path.name),
                    **verify_video(path, "h264"),
                }
            if "av1" in codecs:
                path = staging / "av1" / f"{stem}.webm"
                write_av1_video(path, frames, fps=fps)
                outputs["av1"] = {
                    "path": str(Path("av1") / path.name),
                    **verify_video(path, "av1"),
                }
            rendered.append(
                {
                    "edge_id": row["edge_id"],
                    "canonical_state_index": row["canonical_state_index"],
                    "execution_horizon": row["execution_horizon"],
                    "success": row["success"],
                    **(
                        {"camera_pose": row["camera_pose"]}
                        if "camera_pose" in row
                        else {}
                    ),
                    "outputs": outputs,
                }
            )
        report = {
            "schema_version": 1,
            "source_evaluation": str(evaluation_path),
            "source_frame_dir": str(frame_dir),
            "video_count": len(rendered),
            "codecs": list(codecs),
            "fps": fps,
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


def parse_args(args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render LIBERO-Bind rollout frames")
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--codecs", default="h264,av1")
    parser.add_argument("--fps", type=float, default=20.0)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    report = render_evaluation(
        args.evaluation,
        args.frame_dir,
        args.output_dir,
        codecs=tuple(value.strip() for value in args.codecs.split(",") if value.strip()),
        fps=args.fps,
    )
    print(
        json.dumps(
            {"output_dir": str(args.output_dir), "video_count": report["video_count"]}
        )
    )


if __name__ == "__main__":
    main()
