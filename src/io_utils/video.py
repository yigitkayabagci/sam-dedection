from __future__ import annotations

from contextlib import contextmanager
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


# The first entry keeps `out_path` exactly as given -- this is the existing
# behaviour and the common case works first try. The rest are the fallback,
# each forcing its own container: some ARM builds (observed: a headless
# Jetson whose FFmpeg maps H.264-family fourccs to the `h264_v4l2m2m`
# hardware encoder) have no software MPEG-4/H.264 encoder registered at all
# and no accessible /dev/video encode node either, so mp4v's `isOpened()`
# comes back False with nothing on stderr to explain why. MJPG and XVID in an
# .avi container are the two that were confirmed to actually open there, in
# that order because MJPG is the more widely playable of the two, at the cost
# of a larger file.
_FOURCC_FALLBACKS = (("mp4v", None), ("MJPG", ".avi"), ("XVID", ".avi"))


@contextmanager
def open_video_writer(out_path: str | Path, fps: float, size: tuple[int, int]):
    """Yield a `write(frame_rgb)` callable that encodes straight to disk.

    Use this instead of collecting frames and calling `write_video` at the end.
    A 500-frame 720p clip is ~1.4 GB of RGB held at once, and on a Jetson --
    where CPU and GPU share both the memory and its bandwidth -- that is not
    free: it competes with the model for exactly the resource the model is
    bound by.

    `write_video.path` on the yielded callable is where the video actually
    landed. Usually that is `out_path` unchanged; it differs only after a
    fallback swapped the container, and a caller that reports "wrote X" needs
    to say what was actually written, not what was asked for.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer, used_path, used_fourcc, failed = None, None, None, []
    for fourcc, suffix in _FOURCC_FALLBACKS:
        candidate = out_path if suffix is None else out_path.with_suffix(suffix)
        w = cv2.VideoWriter(str(candidate), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            writer, used_path, used_fourcc = w, candidate, fourcc
            break
        w.release()
        failed.append(fourcc)

    if writer is None:
        raise RuntimeError(
            f"Could not open a video writer for {out_path} -- tried "
            f"{', '.join(f for f, _ in _FOURCC_FALLBACKS)}, none of which "
            "cv2.VideoWriter would open. Check `ffmpeg -encoders` for a "
            "software MPEG-4 or MJPEG encoder on this build."
        )
    if failed:
        print(f"[video] {'/'.join(failed)} would not open a writer for "
              f"{out_path.name} on this machine; wrote {used_path.name} "
              f"instead, using {used_fourcc}.")

    def write(frame_rgb):
        return writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

    write.path = used_path
    try:
        yield write
    finally:
        writer.release()


def write_video(
    frames: Iterable[np.ndarray],
    out_path: str | Path,
    fps: float,
    size: tuple[int, int],
) -> None:
    """Write RGB frames to an MP4."""
    with open_video_writer(out_path, fps, size) as write:
        for frame_rgb in frames:
            write(frame_rgb)
