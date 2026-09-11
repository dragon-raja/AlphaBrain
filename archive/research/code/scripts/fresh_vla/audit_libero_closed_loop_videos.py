from __future__ import annotations

import argparse
import json
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import av
import numpy as np


def mp4_box_offsets(path: Path) -> dict[str, list[int]]:
    file_size = path.stat().st_size
    offsets: dict[str, list[int]] = defaultdict(list)
    with path.open("rb") as stream:
        offset = 0
        while offset < file_size:
            stream.seek(offset)
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError(f"truncated MP4 box header at byte {offset}: {path}")
            size, raw_type = struct.unpack(">I4s", header)
            header_size = 8
            if size == 1:
                extended = stream.read(8)
                if len(extended) != 8:
                    raise ValueError(f"truncated extended MP4 box at byte {offset}: {path}")
                size = struct.unpack(">Q", extended)[0]
                header_size = 16
            elif size == 0:
                size = file_size - offset
            if size < header_size or offset + size > file_size:
                raise ValueError(f"invalid MP4 box size={size} at byte {offset}: {path}")
            offsets[raw_type.decode("latin-1")].append(offset)
            offset += size
    return dict(offsets)


def inspect_video_artifact(path: Path, *, expected_frames: int) -> dict[str, Any]:
    boxes = mp4_box_offsets(path)
    faststart = bool(boxes.get("moov") and boxes.get("mdat")) and min(boxes["moov"]) < min(
        boxes["mdat"]
    )
    decoded_frames = 0
    max_frame_std = 0.0
    max_temporal_delta = 0.0
    previous = None
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        codec = str(stream.codec_context.name or "")
        codec_tag = str(stream.codec_context.codec_tag or "")
        pixel_format = str(stream.codec_context.pix_fmt or "")
        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)
        for frame in container.decode(stream):
            current = frame.to_ndarray(format="rgb24").astype(np.float32)
            decoded_frames += 1
            max_frame_std = max(max_frame_std, float(np.std(current)))
            if previous is not None:
                max_temporal_delta = max(
                    max_temporal_delta,
                    float(np.mean(np.abs(current - previous))),
                )
            previous = current
    compatible = codec == "h264" and codec_tag == "avc1" and pixel_format == "yuv420p"
    exact_frames = decoded_frames == expected_frames
    nonblank = max_frame_std > 1.0
    motion = max_temporal_delta > 0.1
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "codec": codec,
        "codec_tag": codec_tag,
        "pixel_format": pixel_format,
        "width": width,
        "height": height,
        "expected_frames": expected_frames,
        "decoded_frames": decoded_frames,
        "max_frame_std": max_frame_std,
        "max_temporal_delta": max_temporal_delta,
        "faststart": faststart,
        "compatible": compatible,
        "exact_frames": exact_frames,
        "nonblank": nonblank,
        "motion": motion,
        "passed": compatible and exact_frames and nonblank and motion and faststart,
    }


def expected_videos(
    payload: Mapping[str, Any],
    *,
    video_groups: int,
) -> dict[str, int]:
    if payload.get("status") != "complete":
        raise ValueError("closed-loop evaluation must be complete before video audit")
    if payload.get("evaluation") != "end_to_end":
        raise ValueError("video audit expects end-to-end evaluation")
    grouped: dict[int, dict[str, dict[str, Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in payload["rows"]:
        grouped[int(row["execution_horizon"])][str(row["pair_id"])][
            str(row["branch_outcome"])
        ] = row
    result = {}
    for horizon in sorted(grouped):
        pair_ids = sorted(grouped[horizon])[:video_groups]
        for pair_id in pair_ids:
            branches = grouped[horizon][pair_id]
            if branches.keys() != {"attached", "slipped"}:
                raise ValueError(f"unpaired video rows for K={horizon}, pair={pair_id}")
            result[f"end_to_end-k{horizon}-{pair_id}.mp4"] = (
                max(int(branches[branch]["completion_steps"]) for branch in branches) + 1
            )
    return result


def audit_runs(
    run_dirs: Sequence[Path],
    *,
    tag: str,
    video_groups: int,
) -> dict[str, Any]:
    if video_groups < 1:
        raise ValueError("video_groups must be positive")
    records = []
    missing = []
    unexpected = []
    errors = []
    run_summaries = []
    for run_dir in run_dirs:
        result_path = run_dir / f"closed_loop_end_to_end_{tag}.json"
        payload = json.loads(result_path.read_text())
        expected = expected_videos(payload, video_groups=video_groups)
        video_dir = run_dir / f"closed_loop_videos_{tag}"
        actual = {path.name: path for path in video_dir.glob("*.mp4")}
        run_missing = sorted(set(expected) - set(actual))
        run_unexpected = sorted(set(actual) - set(expected))
        missing.extend(str(video_dir / name) for name in run_missing)
        unexpected.extend(str(actual[name]) for name in run_unexpected)
        for name in sorted(set(expected) & set(actual)):
            try:
                record = inspect_video_artifact(actual[name], expected_frames=expected[name])
                record.update({"run_dir": str(run_dir), "filename": name})
                records.append(record)
            except Exception as error:
                errors.append({"path": str(actual[name]), "error": repr(error)})
        run_summaries.append(
            {
                "run_dir": str(run_dir),
                "result": str(result_path),
                "expected_video_count": len(expected),
                "actual_video_count": len(actual),
                "missing": run_missing,
                "unexpected": run_unexpected,
            }
        )

    expected_count = sum(row["expected_video_count"] for row in run_summaries)
    actual_count = sum(row["actual_video_count"] for row in run_summaries)
    checks = {
        "video_count": actual_count == expected_count and not missing and not unexpected,
        "decoded_without_error": not errors and len(records) == expected_count,
        "codec_compatible": len(records) == expected_count
        and all(row["compatible"] for row in records),
        "exact_frame_count": len(records) == expected_count
        and all(row["exact_frames"] for row in records),
        "faststart": len(records) == expected_count and all(row["faststart"] for row in records),
        "nonblank": len(records) == expected_count and all(row["nonblank"] for row in records),
        "motion": len(records) == expected_count and all(row["motion"] for row in records),
    }
    return {
        "schema_version": 1,
        "passed": all(checks.values()),
        "checks": checks,
        "tag": tag,
        "video_groups_per_horizon": video_groups,
        "expected_video_count": expected_count,
        "actual_video_count": actual_count,
        "total_decoded_frames": sum(row["decoded_frames"] for row in records),
        "total_expected_frames": sum(row["expected_frames"] for row in records),
        "total_size_bytes": sum(row["size_bytes"] for row in records),
        "runs": run_summaries,
        "missing": missing,
        "unexpected": unexpected,
        "errors": errors,
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit closed-loop H.264 video artifacts")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--tag", default="val_gate")
    parser.add_argument("--video-groups", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_runs(args.run_dirs, tag=args.tag, video_groups=args.video_groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "video_count": result["actual_video_count"],
                "decoded_frames": result["total_decoded_frames"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
