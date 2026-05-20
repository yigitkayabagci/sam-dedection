from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int


def _open(path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    return cap


def read_metadata(path: str | Path) -> VideoMetadata:
    cap = _open(path)
    try:
        return VideoMetadata(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def read_first_frame(path: str | Path) -> np.ndarray:
    """Returns the first frame as RGB uint8."""
    cap = _open(path)
    try:
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Empty video: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def extract_frames(video_path: str | Path, out_dir: str | Path) -> VideoMetadata:
    """Extract every frame to JPGs named 00000.jpg, 00001.jpg, ... in out_dir.

    SAM 2 / EdgeTAM's video predictor consumes a folder of JPGs in this layout.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = _open(video_path)
    meta = VideoMetadata(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    try:
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            cv2.imwrite(str(out / f"{idx:05d}.jpg"), bgr)
            idx += 1
    finally:
        cap.release()
    return meta


def write_video(
    frames: Iterable[np.ndarray],
    out_path: str | Path,
    fps: float,
    size: tuple[int, int],
) -> None:
    """Write RGB frames to an MP4."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {out_path}")
    try:
        for frame_rgb in frames:
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
