from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .io_utils import extract_frames, read_first_frame, write_video
from .prompts import PromptSet
from .trackers import VideoTracker
from .visualize import overlay_masks


@dataclass
class PipelineConfig:
    video_path: Path
    output_path: Path
    frames_cache: Path | None = None  # if None, use a tempdir
    keep_frames: bool = False
    draw_bbox: bool = True
    mask_alpha: float = 0.5


def run(tracker: VideoTracker, prompts: PromptSet, cfg: PipelineConfig) -> Path:
    """Run a full video -> tracked video pipeline.

    Steps:
        1. Extract frames from video to a directory of JPGs.
        2. Initialise the tracker on that directory.
        3. Push prompts (boxes/points) on their declared frames.
        4. Propagate, overlay masks on each frame, and write to mp4.
    """
    video_path = Path(cfg.video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cache_root = Path(cfg.frames_cache) if cfg.frames_cache else Path(tempfile.mkdtemp(prefix="frames_"))
    cache_root.mkdir(parents=True, exist_ok=True)

    try:
        meta = extract_frames(video_path, cache_root)
        tracker.prepare(cache_root)
        tracker.set_prompts(prompts)

        frame_files = sorted(cache_root.glob("*.jpg"))
        rendered: list[np.ndarray] = []

        progress = tqdm(total=len(frame_files), desc="tracking", unit="frame")
        for result in tracker.propagate():
            frame_bgr = cv2.imread(str(frame_files[result.frame_idx]))
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            overlay = overlay_masks(
                frame_rgb,
                result.masks,
                alpha=cfg.mask_alpha,
                draw_bbox=cfg.draw_bbox,
            )
            rendered.append(overlay)
            progress.update(1)
        progress.close()

        write_video(
            rendered,
            cfg.output_path,
            fps=meta.fps,
            size=(meta.width, meta.height),
        )
        return Path(cfg.output_path).resolve()
    finally:
        tracker.reset()
        if not cfg.keep_frames and cfg.frames_cache is None:
            shutil.rmtree(cache_root, ignore_errors=True)


def grab_first_frame(video_path: str | Path) -> np.ndarray:
    return read_first_frame(video_path)
