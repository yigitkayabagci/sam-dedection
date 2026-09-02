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


def to_rgb8(img: np.ndarray) -> np.ndarray:
    """8-bit RGB from a decoded frame of any bit depth or channel count.

    Split out from `load_frame_rgb8` so a caller that is about to downscale can
    do this *after* the resize: on a 16-bit mono frame it is a full-frame
    min/max pass followed by tripling the data, and both are far cheaper at the
    model's input size than at the sensor's.
    """
    img = _to_uint8(img)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_frame(path: str | Path) -> np.ndarray:
    """Decode a frame file, unchanged: whatever depth and channels it holds."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read frame: {path}")
    return img


def load_frame_rgb8(path: str | Path) -> np.ndarray:
    """Load a single frame as 8-bit RGB, regardless of source bit depth or
    channel count (handles 16-bit / grayscale / BGRA TIFFs)."""
    return to_rgb8(decode_frame(path))


def read_first_frame_dir(frames_dir: str | Path, pattern: str = "*.tif*") -> np.ndarray:
    """First frame of a directory sequence as RGB uint8 (for prompt picking)."""
    return load_frame_rgb8(list_frame_files(frames_dir, pattern)[0])


def frames_metadata(
    frames_dir: str | Path, pattern: str = "*.tif*", fps: float = 30.0
) -> VideoMetadata:
    """Build VideoMetadata for a frame sequence. fps is supplied by the caller
    since an image sequence carries no inherent frame rate.

    Read off frame 0 alone, which is why `odd_sized_frames` exists: everything
    downstream is fixed to this one frame's geometry.
    """
    files = list_frame_files(frames_dir, pattern)
    first = load_frame_rgb8(files[0])
    h, w = first.shape[:2]
    return VideoMetadata(width=int(w), height=int(h), fps=float(fps), frame_count=len(files))


def frame_size(path: str | Path) -> tuple[int, int] | None:
    """`(width, height)` from the file's header, or None if it cannot be read.

    Pillow parses the header and stops, so this costs a stat and a few hundred
    bytes per file rather than a full decode -- which is what makes checking
    every frame before a run affordable. None is not "no frame": an unreadable
    file is `decode_frame`'s error to raise, with its own message, and guessing
    here would replace it with a worse one.
    """
    try:
        from PIL import Image

        with Image.open(str(path)) as img:
            return int(img.width), int(img.height)
    except Exception:
        return None


def odd_sized_frames(files, width: int, height: int, limit: int = 5):
    """The frames that are not `width`x`height`, as `(path, (w, h))` pairs.

    A frame sequence has to be one size, and nothing in the pipeline says so
    until it is far too late. `frames_metadata` reads frame 0; the centre crop
    is computed from it once; the JPG cache is written through that one slice;
    the model returns every mask at that one geometry. A frame of another size
    then meets the overlay as a NumPy slice of a different shape -- and because
    `rgb[y0:y0 + ch, x0:x0 + cw]` clips silently rather than raising, a smaller
    frame arrives as a smaller array instead of an error. What surfaces is
    `IndexError: boolean index did not match indexed array`, minutes in, from
    `visualize.overlay_masks`, naming two numbers and no file.

    `limit` caps how many are named in the message; the count is always exact.
    """
    odd = []
    for path in files:
        size = frame_size(path)
        if size is not None and size != (width, height):
            odd.append((path, size))
            if len(odd) >= limit:
                break
    return odd
