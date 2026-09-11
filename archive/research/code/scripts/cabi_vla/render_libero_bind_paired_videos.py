from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from PIL import Image, ImageDraw

from render_libero_bind_eval_frames import make_contact_sheet
from render_libero_bind_videos import verify_video


FRESH_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "fresh_vla"
if str(FRESH_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(FRESH_SCRIPT_DIR))

from video_io import write_av1_video, write_h264_video


def rollout_key(row: Mapping[str, object]) -> tuple[str, int, int]:
    return (
        str(row["edge_id"]),
        int(row["canonical_state_index"]),
        int(row["execution_horizon"]),
    )


def pair_rollout_frames(
    baseline: np.ndarray,
    method: np.ndarray,
    *,
    baseline_label: str,
    method_label: str,
    header_height: int = 24,
    divider_width: int = 4,
) -> tuple[np.ndarray, dict[str, int]]:
    """Align two rollout videos and place them under a stable comparison header."""

    left = np.asarray(baseline)
    right = np.asarray(method)
    if (
        left.ndim != 4
        or right.ndim != 4
        or left.shape[-1] != 3
        or right.shape[-1] != 3
        or len(left) == 0
        or len(right) == 0
    ):
        raise ValueError("paired rollouts must be non-empty NxHxWx3 arrays")
    if left.shape[1:] != right.shape[1:]:
        raise ValueError(
            f"paired rollout frame shapes differ: {left.shape[1:]} vs {right.shape[1:]}"
        )
    if header_height <= 0 or divider_width <= 0:
        raise ValueError("header and divider dimensions must be positive")

    left = np.asarray(np.clip(left, 0, 255), dtype=np.uint8)
    right = np.asarray(np.clip(right, 0, 255), dtype=np.uint8)
    frame_count = max(len(left), len(right))
    if len(left) < frame_count:
        left = np.concatenate(
            [left, np.repeat(left[-1:], frame_count - len(left), axis=0)], axis=0
        )
    if len(right) < frame_count:
        right = np.concatenate(
            [right, np.repeat(right[-1:], frame_count - len(right), axis=0)], axis=0
        )

    height, width = left.shape[1:3]
    output_width = width * 2 + divider_width
    header = Image.new("RGB", (output_width, header_height), color=(20, 20, 20))
    draw = ImageDraw.Draw(header)
    draw.text((6, 5), baseline_label, fill="white")
    draw.text((width + divider_width + 6, 5), method_label, fill="white")
    header_array = np.asarray(header, dtype=np.uint8)

    paired = np.zeros(
        (frame_count, height + header_height, output_width, 3), dtype=np.uint8
    )
    paired[:, :header_height] = header_array
    paired[:, header_height:, :width] = left
    paired[:, header_height:, width : width + divider_width] = 220
    paired[:, header_height:, width + divider_width :] = right
    return paired, {
        "baseline_frame_count": int(len(baseline)),
        "method_frame_count": int(len(method)),
        "paired_frame_count": frame_count,
    }


def _load_rows(path: Path) -> tuple[dict, dict[tuple[str, int, int], dict]]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"paired rendering requires a complete evaluation: {path}")
    rows = {
        rollout_key(row): row for row in payload["rows"] if "frame_file" in row
    }
    if not rows:
        raise ValueError(f"evaluation has no recorded frames: {path}")
    return payload, rows


def render_paired_evaluations(
    baseline_evaluation: Path,
    baseline_frame_dir: Path,
    method_evaluation: Path,
    method_frame_dir: Path,
    output_dir: Path,
    *,
    baseline_name: str,
    method_name: str,
    codecs: tuple[str, ...],
    fps: float,
) -> dict:
    baseline_payload, baseline_rows = _load_rows(baseline_evaluation)
    method_payload, method_rows = _load_rows(method_evaluation)
    if set(baseline_rows) != set(method_rows):
        raise ValueError("baseline and method recorded rollout keys differ")
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
        for key in sorted(baseline_rows):
            baseline_row = baseline_rows[key]
            method_row = method_rows[key]
            with np.load(
                baseline_frame_dir / str(baseline_row["frame_file"]),
                allow_pickle=False,
            ) as archive:
                baseline_frames = np.asarray(archive["frames"])
            with np.load(
                method_frame_dir / str(method_row["frame_file"]),
                allow_pickle=False,
            ) as archive:
                method_frames = np.asarray(archive["frames"])

            baseline_success = bool(baseline_row["success"])
            method_success = bool(method_row["success"])
            paired, frame_metadata = pair_rollout_frames(
                baseline_frames,
                method_frames,
                baseline_label=f"{baseline_name} success={int(baseline_success)}",
                method_label=f"{method_name} success={int(method_success)}",
            )
            edge_id, state_index, horizon = key
            stem = f"{edge_id}--state-{state_index:02d}--k{horizon}"
            outputs = {}
            contact_sheet, contact_indices = make_contact_sheet(paired)
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
                write_h264_video(path, paired, fps=fps)
                outputs["h264"] = {
                    "path": str(Path("h264") / path.name),
                    **verify_video(path, "h264"),
                }
            if "av1" in codecs:
                path = staging / "av1" / f"{stem}.webm"
                write_av1_video(path, paired, fps=fps)
                outputs["av1"] = {
                    "path": str(Path("av1") / path.name),
                    **verify_video(path, "av1"),
                }
            rendered.append(
                {
                    "edge_id": edge_id,
                    "canonical_state_index": state_index,
                    "execution_horizon": horizon,
                    "baseline_success": baseline_success,
                    "method_success": method_success,
                    **frame_metadata,
                    "outputs": outputs,
                }
            )

        report = {
            "schema_version": 1,
            "baseline_evaluation": str(baseline_evaluation),
            "baseline_policy_identity": baseline_payload.get("policy_identity"),
            "baseline_name": baseline_name,
            "method_evaluation": str(method_evaluation),
            "method_policy_identity": method_payload.get("policy_identity"),
            "method_name": method_name,
            "pair_count": len(rendered),
            "codecs": list(codecs),
            "fps": fps,
            "pairs": rendered,
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
    parser = argparse.ArgumentParser(description="Render paired LIBERO-Bind rollouts")
    parser.add_argument("--baseline-evaluation", type=Path, required=True)
    parser.add_argument("--baseline-frame-dir", type=Path, required=True)
    parser.add_argument("--method-evaluation", type=Path, required=True)
    parser.add_argument("--method-frame-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-name", default="BC")
    parser.add_argument("--method-name", default="CABI")
    parser.add_argument("--codecs", default="h264,av1")
    parser.add_argument("--fps", type=float, default=20.0)
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    report = render_paired_evaluations(
        args.baseline_evaluation,
        args.baseline_frame_dir,
        args.method_evaluation,
        args.method_frame_dir,
        args.output_dir,
        baseline_name=args.baseline_name,
        method_name=args.method_name,
        codecs=tuple(value.strip() for value in args.codecs.split(",") if value.strip()),
        fps=args.fps,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "pair_count": report["pair_count"]}))


if __name__ == "__main__":
    main()
