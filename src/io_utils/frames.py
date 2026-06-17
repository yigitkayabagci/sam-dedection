from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .video import VideoMetadata

# Image suffixes we accept as a standalone frame sequence. TIFF is the
# primary target (frame_000000.tiff, ...), but the same loader handles the
# common 8-bit formats too.
FRAME_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def list_frame_files(frames_dir: str | Path, pattern: str = "*.tif*") -> list[Path]:
    """Return frame files in a directory, sorted by name.

    Filenames are expected to be zero-padded (frame_000000.tiff, ...), so a
    plain lexical sort already yields chronological order.
    """
    d = Path(frames_dir)
    if not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {d}")
    files = sorted(p for p in d.glob(pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(f"No frames matching {pattern!r} in {d}")
    return files


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Collapse any bit depth (e.g. 16-bit TIFF) down to 8-bit using the
    frame's own dynamic range, so low-valued sensor data stays visible."""
    if img.dtype == np.uint8:
        return img
    out = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return out.astype(np.uint8)


def load_frame_rgb8(path: str | Path) -> np.ndarray:
    """Load a single frame as 8-bit RGB, regardless of source bit depth or
    channel count (handles 16-bit / grayscale / BGRA TIFFs)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read frame: {path}")
    img = _to_uint8(img)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_first_frame_dir(frames_dir: str | Path, pattern: str = "*.tif*") -> np.ndarray:
    """First frame of a directory sequence as RGB uint8 (for prompt picking)."""
    return load_frame_rgb8(list_frame_files(frames_dir, pattern)[0])


def frames_metadata(
    frames_dir: str | Path, pattern: str = "*.tif*", fps: float = 30.0
) -> VideoMetadata:
    """Build VideoMetadata for a frame sequence. fps is supplied by the caller
    since an image sequence carries no inherent frame rate."""
    files = list_frame_files(frames_dir, pattern)
    first = load_frame_rgb8(files[0])
    h, w = first.shape[:2]
    return VideoMetadata(width=int(w), height=int(h), fps=float(fps), frame_count=len(files))
