from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path
from typing import Iterable

import av
import numpy as np


def _rgb_frame(frame: np.ndarray, *, width: int | None = None, height: int | None = None) -> np.ndarray:
    value = np.asarray(frame)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"video frame must have shape HxWx3, got {value.shape}")
    if value.dtype != np.uint8:
        value = np.clip(value, 0, 255).astype(np.uint8)
    if width is not None and height is not None and value.shape[:2] != (height, width):
        raise ValueError(f"video frame shape changed from {(height, width)} to {value.shape[:2]}")
    return np.ascontiguousarray(value)


def _pad_even(frame: np.ndarray) -> np.ndarray:
    pad_height = frame.shape[0] % 2
    pad_width = frame.shape[1] % 2
    if not pad_height and not pad_width:
        return frame
    return np.pad(frame, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")


def write_h264_video(
    path: Path,
    frames: Iterable[np.ndarray],
    *,
    fps: float = 10.0,
    crf: int = 23,
    preset: str = "fast",
) -> None:
    """Atomically write browser/VS Code compatible H.264 video from RGB frames."""
    if fps <= 0:
        raise ValueError("video fps must be positive")
    iterator = iter(frames)
    try:
        first = _rgb_frame(next(iterator))
    except StopIteration as error:
        raise ValueError(f"cannot write empty video: {path}") from error

    original_height, original_width = first.shape[:2]
    first = _pad_even(first)
    height, width = first.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.h264.tmp")
    container = None
    try:
        container = av.open(
            str(temporary),
            mode="w",
            format="mp4",
            options={"movflags": "+faststart"},
        )
        stream = container.add_stream("libx264", rate=Fraction(str(fps)).limit_denominator(1000))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": preset}

        def encode(frame: np.ndarray) -> None:
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)

        encode(first)
        for frame in iterator:
            value = _rgb_frame(frame, width=original_width, height=original_height)
            encode(_pad_even(value))
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        container = None
        os.replace(temporary, path)
    except Exception:
        if container is not None:
            container.close()
        temporary.unlink(missing_ok=True)
        raise


def write_av1_video(
    path: Path,
    frames: Iterable[np.ndarray],
    *,
    fps: float = 10.0,
    crf: int = 35,
    cpu_used: int = 8,
) -> None:
    """Atomically write an AV1 WebM fallback for Chromium/VS Code viewers."""
    if fps <= 0:
        raise ValueError("video fps must be positive")
    iterator = iter(frames)
    try:
        first = _rgb_frame(next(iterator))
    except StopIteration as error:
        raise ValueError(f"cannot write empty video: {path}") from error

    original_height, original_width = first.shape[:2]
    first = _pad_even(first)
    height, width = first.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.av1.tmp")
    container = None
    try:
        container = av.open(str(temporary), mode="w", format="webm")
        stream = container.add_stream(
            "libaom-av1",
            rate=Fraction(str(fps)).limit_denominator(1000),
        )
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {
            "crf": str(crf),
            "cpu-used": str(cpu_used),
            "row-mt": "1",
        }

        def encode(frame: np.ndarray) -> None:
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)

        encode(first)
        for frame in iterator:
            value = _rgb_frame(frame, width=original_width, height=original_height)
            encode(_pad_even(value))
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        container = None
        os.replace(temporary, path)
    except Exception:
        if container is not None:
            container.close()
        temporary.unlink(missing_ok=True)
        raise
