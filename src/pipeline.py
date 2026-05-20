from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .io_utils import extract_frames, read_first_frame, write_video
from .io_utils.video import read_metadata
from .prompts import PromptSet
from .trackers import VideoTracker
from .visualize import overlay_masks


@dataclass
class PipelineConfig:
    video_path: Path
    output_path: Path
    # "auto" -> hand mp4 directly to EdgeTAM if decord is available (no JPG dump);
    # "mp4"  -> require direct mp4 path (fail if decord is missing);
    # "jpg"  -> always extract JPGs first (legacy, works without decord).
    video_mode: str = "auto"
    frames_cache: Path | None = None
    keep_frames: bool = False
    draw_bbox: bool = True
    mask_alpha: float = 0.5


def _decord_available() -> bool:
    try:
        import decord  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_video_mode(cfg: PipelineConfig) -> str:
    suffix = Path(cfg.video_path).suffix.lower()
    if cfg.video_mode == "jpg":
        return "jpg"
    if cfg.video_mode == "mp4":
        if suffix not in (".mp4",):
            raise ValueError(f"video_mode=mp4 requires .mp4 file, got {suffix}")
        if not _decord_available():
            raise RuntimeError("video_mode=mp4 needs `decord` (or eva-decord). "
                               "pip install eva-decord  or use video_mode=jpg.")
        return "mp4"
    # auto
    if suffix == ".mp4" and _decord_available():
        return "mp4"
    return "jpg"


def run(tracker: VideoTracker, prompts: PromptSet, cfg: PipelineConfig) -> Path:
    """Run a full video -> tracked video pipeline.

    Two flavors, chosen by `cfg.video_mode`:
      mp4 mode: hand the .mp4 path straight to the tracker (EdgeTAM decodes
                via decord on the fly). cv2.VideoCapture reads frames in
                parallel for the overlay step. No JPG cache on disk.
      jpg mode: extract all frames to JPGs first, then track. Slower start,
                but works without `decord`.
    """
    video_path = Path(cfg.video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    mode = _resolve_video_mode(cfg)
    if mode == "mp4":
        meta = read_metadata(video_path)
        return _run_mp4(tracker, prompts, cfg, video_path, meta)
    return _run_jpg(tracker, prompts, cfg, video_path)


def _run_mp4(tracker, prompts, cfg, video_path, meta):
    """Direct-mp4 path: no frame extraction, overlay via sequential cv2 read."""
    tracker.prepare(str(video_path))
    tracker.set_prompts(prompts)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for overlay: {video_path}")

    rendered: list[np.ndarray] = []
    cursor = 0
    progress = tqdm(total=meta.frame_count, desc="tracking [mp4]", unit="frame")
    t0 = time.perf_counter()
    try:
        for result in tracker.propagate():
            # Sequential read; propagate yields frames in chronological order.
            while cursor < result.frame_idx:
                cap.read()
                cursor += 1
            ok, bgr = cap.read()
            if not ok:
                break
            cursor += 1
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rendered.append(
                overlay_masks(rgb, result.masks, alpha=cfg.mask_alpha, draw_bbox=cfg.draw_bbox)
            )
            progress.update(1)
    finally:
        progress.close()
        cap.release()
        tracker.reset()

    elapsed = time.perf_counter() - t0
    n = len(rendered)
    if n:
        print(f"[pipeline] tracked {n} frames in {elapsed:.2f}s ({n/elapsed:.1f} FPS)")
    write_video(rendered, cfg.output_path, fps=meta.fps, size=(meta.width, meta.height))
    return Path(cfg.output_path).resolve()


def _run_jpg(tracker, prompts, cfg, video_path):
    """Legacy path: dump JPGs to a temp dir then track."""
    cache_root = Path(cfg.frames_cache) if cfg.frames_cache else Path(tempfile.mkdtemp(prefix="frames_"))
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        meta = extract_frames(video_path, cache_root)
        tracker.prepare(cache_root)
        tracker.set_prompts(prompts)

        frame_files = sorted(cache_root.glob("*.jpg"))
        rendered: list[np.ndarray] = []
        progress = tqdm(total=len(frame_files), desc="tracking [jpg]", unit="frame")
        t0 = time.perf_counter()
        for result in tracker.propagate():
            bgr = cv2.imread(str(frame_files[result.frame_idx]))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rendered.append(
                overlay_masks(rgb, result.masks, alpha=cfg.mask_alpha, draw_bbox=cfg.draw_bbox)
            )
            progress.update(1)
        progress.close()
        elapsed = time.perf_counter() - t0
        n = len(rendered)
        if n:
            print(f"[pipeline] tracked {n} frames in {elapsed:.2f}s ({n/elapsed:.1f} FPS)")
        write_video(rendered, cfg.output_path, fps=meta.fps, size=(meta.width, meta.height))
        return Path(cfg.output_path).resolve()
    finally:
        tracker.reset()
        if not cfg.keep_frames and cfg.frames_cache is None:
            shutil.rmtree(cache_root, ignore_errors=True)


def grab_first_frame(video_path: str | Path) -> np.ndarray:
    return read_first_frame(video_path)
