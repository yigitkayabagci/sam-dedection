from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .io_utils import (
    extract_frames,
    frames_metadata,
    list_frame_files,
    load_frame_rgb8,
    read_first_frame,
    read_first_frame_dir,
    write_video,
)
from .io_utils.video import read_metadata
from .metrics import format_benchmark_note, fps_summary, write_latency_chart, write_stage_chart
from .prompts import PromptSet
from .trackers import VideoTracker
from .visualize import overlay_masks


@dataclass
class PipelineConfig:
    output_path: Path
    # Exactly one input source must be set:
    #   video_path  -> an .mp4 (or any cv2-readable video) file, OR
    #   frames_dir  -> a directory of pre-extracted frames (frame_000000.tiff, ...).
    video_path: Path | None = None
    frames_dir: Path | None = None
    # Glob used to pick up frames inside frames_dir (frames mode only).
    frame_pattern: str = "*.tif*"
    # Output frame rate for frames mode (an image sequence has no inherent fps).
    fps: float = 30.0
    # "auto" -> hand mp4 directly to EdgeTAM if decord is available (no JPG dump);
    # "mp4"  -> require direct mp4 path (fail if decord is missing);
    # "jpg"  -> always extract JPGs first (legacy, works without decord).
    video_mode: str = "auto"
    frames_cache: Path | None = None
    keep_frames: bool = False
    draw_bbox: bool = True
    mask_alpha: float = 0.5
    # FPS reporting: drop the first `fps_warmup` frames (model load / CUDA
    # warm-up) from the average, and optionally save per-frame charts.
    fps_warmup: int = 0
    fps_chart: Path | None = None
    # Per-frame breakdown: decode, model inference, mask render. Separate from
    # fps_chart because it needs three timed stages instead of one.
    stage_chart: Path | None = None


def _benchmark_note(tracker: VideoTracker, meta, prompts: PromptSet) -> str:
    model_size = getattr(getattr(tracker, "_predictor", None), "image_size", None)
    objects = len(prompts.object_ids()) if prompts is not None else None
    return format_benchmark_note(
        tracker_name=getattr(tracker, "name", None),
        precision=getattr(tracker, "precision", None),
        source_size=(meta.width, meta.height) if meta is not None else None,
        model_size=model_size,
        batch=objects,
    )


def _report_timing(
    cfg: PipelineConfig,
    tracker: VideoTracker,
    meta,
    prompts: PromptSet,
    per_frame_dt: list[float],
    pre_ms: list[float],
    infer_ms: list[float],
    post_ms: list[float],
) -> None:
    """Print FPS stats (with warm-up exclusion) and optionally write charts."""
    if not per_frame_dt:
        return
    stats = fps_summary(per_frame_dt, warmup=cfg.fps_warmup)
    print(f"[pipeline] tracked {stats['frames']} frames in {stats['total_s']:.2f}s "
          f"({stats['avg_fps_all']:.1f} FPS all)")
    if stats["warmup"] > 0:
        print(f"[pipeline] avg {stats['avg_fps_post_warmup']:.1f} FPS over "
              f"{stats['kept_frames']} frames (excluded first {stats['warmup']} warm-up)")

    if cfg.fps_chart is None and cfg.stage_chart is None:
        return
    note = _benchmark_note(tracker, meta, prompts)
    label = getattr(tracker, "name", None)
    if cfg.fps_chart is not None:
        out = write_latency_chart(per_frame_dt, cfg.fps_chart, warmup=cfg.fps_warmup,
                                   note=note, label=label)
        if out:
            print(f"[pipeline] wrote latency chart -> {out}")
    if cfg.stage_chart is not None:
        out = write_stage_chart(pre_ms, infer_ms, post_ms, cfg.stage_chart,
                                 warmup=cfg.fps_warmup, note=note, label=label)
        if out:
            print(f"[pipeline] wrote stage chart -> {out}")


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
    if cfg.frames_dir is not None:
        frames_dir = Path(cfg.frames_dir).resolve()
        if not frames_dir.is_dir():
            raise NotADirectoryError(frames_dir)
        return _run_frames(tracker, prompts, cfg, frames_dir)

    if cfg.video_path is None:
        raise ValueError("PipelineConfig needs either video_path or frames_dir.")
    video_path = Path(cfg.video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    mode = _resolve_video_mode(cfg)
    if mode == "mp4":
        meta = read_metadata(video_path)
        return _run_mp4(tracker, prompts, cfg, video_path, meta)
    return _run_jpg(tracker, prompts, cfg, video_path)


def _run_frames(tracker, prompts, cfg, frames_dir):
    """Frame-sequence path: track a directory of pre-extracted frames
    (e.g. frame_000000.tiff, frame_000001.tiff, ...).

    EdgeTAM's init_state only loads JPGs from a directory, so we transcode the
    source frames into a temporary 00000.jpg cache for the model. The overlay
    is rendered from the *original* frames to preserve their look.
    """
    frame_files = list_frame_files(frames_dir, cfg.frame_pattern)
    meta = frames_metadata(frames_dir, cfg.frame_pattern, fps=cfg.fps)

    cache_root = Path(cfg.frames_cache) if cfg.frames_cache else Path(tempfile.mkdtemp(prefix="frames_"))
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        for idx, src in enumerate(tqdm(frame_files, desc="transcoding", unit="frame")):
            rgb = load_frame_rgb8(src)
            cv2.imwrite(str(cache_root / f"{idx:05d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        tracker.prepare(cache_root)
        tracker.set_prompts(prompts)

        rendered: list[np.ndarray] = []
        per_frame_dt: list[float] = []
        pre_ms: list[float] = []
        infer_ms: list[float] = []
        post_ms: list[float] = []
        progress = tqdm(total=len(frame_files), desc="tracking [frames]", unit="frame")
        t0 = time.perf_counter()
        for result in tracker.propagate():
            t1 = time.perf_counter()
            rgb = load_frame_rgb8(frame_files[result.frame_idx])
            t2 = time.perf_counter()
            rendered.append(
                overlay_masks(rgb, result.masks, alpha=cfg.mask_alpha, draw_bbox=cfg.draw_bbox)
            )
            t3 = time.perf_counter()
            infer_ms.append((t1 - t0) * 1000.0)
            pre_ms.append((t2 - t1) * 1000.0)
            post_ms.append((t3 - t2) * 1000.0)
            per_frame_dt.append(t3 - t0)
            t0 = t3
            progress.update(1)
        progress.close()
        _report_timing(cfg, tracker, meta, prompts, per_frame_dt, pre_ms, infer_ms, post_ms)
        write_video(rendered, cfg.output_path, fps=meta.fps, size=(meta.width, meta.height))
        return Path(cfg.output_path).resolve()
    finally:
        tracker.reset()
        if not cfg.keep_frames and cfg.frames_cache is None:
            shutil.rmtree(cache_root, ignore_errors=True)


def _run_mp4(tracker, prompts, cfg, video_path, meta):
    """Direct-mp4 path: no frame extraction, overlay via sequential cv2 read."""
    tracker.prepare(str(video_path))
    tracker.set_prompts(prompts)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for overlay: {video_path}")

    rendered: list[np.ndarray] = []
    per_frame_dt: list[float] = []
    pre_ms: list[float] = []
    infer_ms: list[float] = []
    post_ms: list[float] = []
    cursor = 0
    progress = tqdm(total=meta.frame_count, desc="tracking [mp4]", unit="frame")
    t0 = time.perf_counter()
    try:
        for result in tracker.propagate():
            t1 = time.perf_counter()
            # Sequential read; propagate yields frames in chronological order.
            while cursor < result.frame_idx:
                cap.read()
                cursor += 1
            ok, bgr = cap.read()
            if not ok:
                break
            cursor += 1
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t2 = time.perf_counter()
            rendered.append(
                overlay_masks(rgb, result.masks, alpha=cfg.mask_alpha, draw_bbox=cfg.draw_bbox)
            )
            t3 = time.perf_counter()
            infer_ms.append((t1 - t0) * 1000.0)
            pre_ms.append((t2 - t1) * 1000.0)
            post_ms.append((t3 - t2) * 1000.0)
            per_frame_dt.append(t3 - t0)
            t0 = t3
            progress.update(1)
    finally:
        progress.close()
        cap.release()
        tracker.reset()

    _report_timing(cfg, tracker, meta, prompts, per_frame_dt, pre_ms, infer_ms, post_ms)
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
        per_frame_dt: list[float] = []
        pre_ms: list[float] = []
        infer_ms: list[float] = []
        post_ms: list[float] = []
        progress = tqdm(total=len(frame_files), desc="tracking [jpg]", unit="frame")
        t0 = time.perf_counter()
        for result in tracker.propagate():
            t1 = time.perf_counter()
            bgr = cv2.imread(str(frame_files[result.frame_idx]))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t2 = time.perf_counter()
            rendered.append(
                overlay_masks(rgb, result.masks, alpha=cfg.mask_alpha, draw_bbox=cfg.draw_bbox)
            )
            t3 = time.perf_counter()
            infer_ms.append((t1 - t0) * 1000.0)
            pre_ms.append((t2 - t1) * 1000.0)
            post_ms.append((t3 - t2) * 1000.0)
            per_frame_dt.append(t3 - t0)
            t0 = t3
            progress.update(1)
        progress.close()
        _report_timing(cfg, tracker, meta, prompts, per_frame_dt, pre_ms, infer_ms, post_ms)
        write_video(rendered, cfg.output_path, fps=meta.fps, size=(meta.width, meta.height))
        return Path(cfg.output_path).resolve()
    finally:
        tracker.reset()
        if not cfg.keep_frames and cfg.frames_cache is None:
            shutil.rmtree(cache_root, ignore_errors=True)


def grab_first_frame(video_path: str | Path) -> np.ndarray:
    return read_first_frame(video_path)


def grab_first_frame_dir(frames_dir: str | Path, pattern: str = "*.tif*") -> np.ndarray:
    return read_first_frame_dir(frames_dir, pattern)
