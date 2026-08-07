from __future__ import annotations

from contextlib import contextmanager
import shutil
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .io_utils import (
    extract_frames,
    frames_metadata,
    list_frame_files,
    decode_frame,
    load_frame_rgb8,
    open_video_writer,
    to_rgb8,
    read_first_frame,
    read_first_frame_dir,
)
from .io_utils.video import read_metadata
from .metrics import format_benchmark_note, fps_summary, write_latency_chart, write_stage_chart
from .prompts import PromptSet
from .realtime import RealtimeSource
from .trackers import TrackingResult, VideoTracker
from .visualize import overlay_masks


class _LazyFrames:
    """Decode, crop, resize and normalise each source frame on demand.

    EdgeTAM's `init_state` does all of that for the whole clip up front, so it
    lands in setup and never appears in a frame time. A camera does not work
    that way: each frame arrives raw and has to be decoded and downsampled to
    the model's input size before anything else can happen. Swapping this in
    after `prepare()` moves that work into the tracking loop, where a real-time
    frame budget has to account for it.

    Deliberately not upstream's `_load_img_as_tensor`, which this replaced:
    that one resizes with PIL and writes `uint8 / 255.0`, which NumPy promotes
    to float64 -- 25 MB per frame at 1024 for the conversion and 25 MB again
    for the normalisation. Nothing in the model needs either. cv2 resizes the
    same frame ~30x faster and staying in float32 halves the memory traffic
    twice over, which on a Jetson (CPU and GPU on one memory bus) is the whole
    cost. See docs/EXPERIMENT_LOG.md 4.1.1 for the step-by-step measurement.

    Reading the source frames rather than the JPG cache also skips a lossy
    round-trip: the cache exists because `init_state` only loads JPGs from a
    directory, and its contents are thrown away the moment this is installed.
    """

    def __init__(self, paths, image_size, timer: list[float], view=None,
                 read_timer: list[float] | None = None) -> None:
        import torch

        self.paths = list(paths)
        self.image_size = image_size
        self.timer = timer
        # Reading a frame off disk is timed separately because a deployment fed
        # by a camera or a network link does not do it at all -- it is the one
        # part of `pre` that is an artefact of benchmarking against files.
        self.read_timer = read_timer if read_timer is not None else []
        self.view = view or (lambda img: img)
        # load_video_frames' own normalisation constants.
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        import torch

        start = time.perf_counter()
        img = decode_frame(self.paths[index])
        read_done = time.perf_counter()
        self.read_timer.append(read_done - start)

        # Crop and resize first, then convert depth and channels. On a 16-bit
        # mono frame `to_rgb8` is a full-frame min/max pass plus a 3x data
        # expansion; doing it at the model's input size instead of the
        # sensor's is the same arithmetic over far fewer pixels. For an 8-bit
        # source the order makes no difference to the result at all.
        img = self.view(img)
        size = self.image_size
        if img.shape[0] != size or img.shape[1] != size:
            # INTER_AREA averages over the pixels it discards; on a downscale
            # that is the difference between a resampled frame and an aliased
            # one. It degenerates to bilinear when upscaling, so pick by which
            # way this frame is going.
            shrinking = size < img.shape[0] or size < img.shape[1]
            img = cv2.resize(img, (size, size),
                             interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)
        rgb = to_rgb8(img)
        out = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32).div_(255.0)
        out = out.sub_(self.mean).div_(self.std)
        self.timer.append(time.perf_counter() - start)
        return out


def _install_realtime_preprocess(tracker, paths, timer: list[float], view=None,
                                 read_timer: list[float] | None = None) -> bool:
    """Route the tracker's frame source through `_LazyFrames`. True if it took."""
    state = getattr(tracker, "_state", None)
    predictor = getattr(tracker, "_predictor", None)
    if state is None or predictor is None or "images" not in state:
        return False
    paths = list(paths)
    if len(paths) != len(state["images"]):
        return False
    state["images"] = _LazyFrames(paths, predictor.image_size, timer, view, read_timer)
    return True


class _LazyVideoFrames:
    """`_LazyFrames`'s counterpart for direct-mp4 mode: decode+resize on demand
    through decord's random-access indexing instead of `load_video_frames_from_video_file`'s
    bulk `for frame in VideoReader(...)` loop, which -- like the JPG path's bulk
    load -- runs entirely inside `init_state` and would otherwise leave mp4-mode
    with no per-frame preprocessing cost to measure at all.

    Mirrors that function's own per-frame transform (permute, /255, normalise)
    exactly, so this changes when the work happens, not what it computes.
    """

    def __init__(self, video_path, image_size, timer: list[float]) -> None:
        import decord
        import torch

        decord.bridge.set_bridge("torch")
        self.reader = decord.VideoReader(str(video_path), width=image_size, height=image_size)
        self.timer = timer
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, index: int):
        import torch

        start = time.perf_counter()
        frame = self.reader[index]  # HWC uint8, already resized by VideoReader's width/height
        img = frame.permute(2, 0, 1).to(torch.float32).div_(255.0)
        img = img.sub_(self.mean).div_(self.std)
        self.timer.append(time.perf_counter() - start)
        return img


def _install_realtime_preprocess_mp4(tracker, video_path, timer: list[float]) -> bool:
    """`_install_realtime_preprocess`'s counterpart for direct-mp4 mode."""
    state = getattr(tracker, "_state", None)
    predictor = getattr(tracker, "_predictor", None)
    if state is None or predictor is None or "images" not in state:
        return False
    try:
        lazy = _LazyVideoFrames(video_path, predictor.image_size, timer)
    except ImportError:
        return False
    if len(lazy) != len(state["images"]):
        return False
    state["images"] = lazy
    return True


@contextmanager
def _postprocess_timer(tracker):
    """Accumulate seconds spent resizing masks back to source resolution.

    That resize is real per-frame work a deployment still pays -- the model
    emits masks at its own input size and the consumer wants them at the
    camera's -- so it belongs in the frame budget, unlike drawing an overlay.
    """
    predictor = getattr(tracker, "_predictor", None)
    spent: list[float] = []
    if predictor is None or not hasattr(predictor, "_get_orig_video_res_output"):
        yield spent
        return
    original = predictor._get_orig_video_res_output
    was_instance_attr = "_get_orig_video_res_output" in vars(predictor)

    def timed(*args, **kwargs):
        start = time.perf_counter()
        out = original(*args, **kwargs)
        spent.append(time.perf_counter() - start)
        return out

    predictor._get_orig_video_res_output = timed
    try:
        yield spent
    finally:
        if was_instance_attr:
            predictor._get_orig_video_res_output = original
        else:
            vars(predictor).pop("_get_orig_video_res_output", None)


def _split_frame(total_s, preprocess_s, post_s, read_s, wait_s, pre_ms, infer_ms,
                 post_ms, read_ms) -> float:
    """Attribute one frame's elapsed time, and return the budget in seconds.

    `preprocess_s`, `read_s` and `post_s` are appended to by the instrumented
    frame loader and the postprocess wrapper during that frame; whatever is
    left of the elapsed time is the model. Each can log more than once per
    frame (or not at all, on a frame that reuses a cached feature), so drain
    everything accumulated since the previous frame rather than assuming one
    entry each.

    Opening the frame file comes back out. It is real time that really passed,
    but it is time no deployment spends: a camera or a network link hands the
    frame over already in memory, and there is no file to read. Leaving it in
    would make the budget a measure of the benchmark's storage rather than of
    the pipeline -- which is exactly what it was doing, at ~36 ms of a ~40 ms
    `pre`. It is still measured and still reported, like the overlay and the
    encode; it is just not charged to the frame.

    Idling on a real-time source (`wait_s`) comes back out for the same reason,
    and it matters more: a model twice as fast as the camera spends half of
    every frame doing nothing, and charging that to the budget would report it
    as exactly as fast as the camera.

    `read_s` empty (mp4 mode, or a tracker with no instrumented loader) and
    `wait_s` empty (no frame skipping, or a model that never gets ahead) mean
    nothing is subtracted and this behaves as it always did.
    """
    read = sum(read_s) * 1000.0
    read_s.clear()
    wait = sum(wait_s) * 1000.0
    wait_s.clear()
    pre = sum(preprocess_s) * 1000.0 - read  # the loader timed read + transform
    preprocess_s.clear()
    post = sum(post_s) * 1000.0
    post_s.clear()
    budget = total_s * 1000.0 - read - wait
    read_ms.append(read)
    pre_ms.append(max(pre, 0.0))
    post_ms.append(post)
    infer_ms.append(max(budget - pre - post, 0.0))
    return budget / 1000.0


def _center_crop_window(width: int, height: int, size: int):
    """`(x0, y0, w, h)` of a centred `size`x`size` window, clamped to the frame."""
    w, h = min(size, width), min(size, height)
    return (width - w) // 2, (height - h) // 2, w, h


def _shift_prompts(prompts: PromptSet, dx: int, dy: int) -> PromptSet:
    """Move prompts from full-frame into crop coordinates."""
    return PromptSet(
        boxes=[replace(b, xyxy=(b.xyxy[0] - dx, b.xyxy[1] - dy,
                                b.xyxy[2] - dx, b.xyxy[3] - dy))
               for b in prompts.boxes],
        points=[replace(p, xy=(p.xy[0] - dx, p.xy[1] - dy)) for p in prompts.points],
    )


def _written_path(emit, cfg: "PipelineConfig") -> Path | None:
    """Where the video actually landed, not just what was asked for.

    `open_video_writer`'s fallback chain can swap the container (mp4v -> MJPG
    in .avi) when the requested codec's writer will not open on this machine,
    so the two can differ. `getattr` covers callers standing in a bare
    `write(frame)` callable with no `.path` -- tests, mainly -- by falling
    back to the requested path.
    """
    if emit is None:
        return None
    return getattr(emit, "path", Path(cfg.output_path)).resolve()


def _median(values: list[float], warmup: int) -> float:
    kept = sorted(values[warmup:] or values)
    return kept[len(kept) // 2] if kept else 0.0


@contextmanager
def _frame_stream(tracker, cfg: "PipelineConfig", prompts: PromptSet, frame_count: int,
                  wait_s: list[float], preprocess_s: list[float], read_s: list[float]):
    """Yield `(results, source)`: the tracker's output, and what paced it.

    Without `realtime_fps` that is `propagate()` over the whole clip and
    `source` is None -- every frame reaches the model because nothing is
    competing with it for time. With it, `cfg.fps_warmup` frames run first, at
    full speed, before the real-time clock starts.

    That ordering matters on a short clip. CUDA graph capture and kernel
    autotuning happen on a model's first few calls regardless of backend, and
    if the clock is already running while that happens, the one-time cost gets
    charged against it: a model that takes two seconds to reach full speed,
    racing a 30 FPS source for those two seconds, loses most of a short clip to
    warm-up rather than to steady-state slowness, and the drop count and
    staleness numbers this function's caller reports end up describing the
    warm-up instead of the thing being measured. A real deployment warms up
    once before its camera feed is judged in real time; running the warm-up
    frames untimed first, then starting the source after them, makes the
    measurement match that.

    `preprocess_s`/`read_s` are cleared right after: `_install_realtime_preprocess`
    patches them in before this function is ever called and they keep
    recording regardless of who is asking, so without this the warm-up frames'
    decode time would land in the first *measured* frame's numbers instead of
    nowhere.

    Starting at the first prompted frame -- for the warm-up prefix and the
    real-time portion after it alike -- mirrors `propagate()`, which begins at
    the earliest conditioning frame rather than at zero.
    """
    prompted = [p.frame_idx for p in prompts.boxes] + [p.frame_idx for p in prompts.points]
    start = min(prompted, default=0)
    if cfg.realtime_fps is None:
        yield tracker.propagate(), None
        return

    warmup = max(0, min(cfg.fps_warmup, frame_count - start - 1))
    if warmup:
        for _ in tracker.propagate_frames(range(start, start + warmup)):
            pass  # priming the model and its memory bank; deliberately not measured
        preprocess_s.clear()
        read_s.clear()
    with RealtimeSource(frame_count, cfg.realtime_fps, start=start + warmup,
                        wait_s=wait_s) as source:
        yield tracker.propagate_frames(source), source


def _emitted(previous: TrackingResult | None, result: TrackingResult):
    """`(frame_idx, masks)` pairs the video should show for one tracked frame.

    Just the frame itself, unless frames were dropped since the last result:
    those are written too, carrying the masks that were still current while
    they arrived. That is what a consumer downstream of the tracker actually
    sees -- the mask holds, then jumps -- and it keeps the mp4 the length of
    the clip. Writing only tracked frames instead would produce a video that
    plays back at however many times real speed the skipping saved.
    """
    if previous is not None:
        for idx in range(previous.frame_idx + 1, result.frame_idx):
            yield idx, previous.masks
    yield result.frame_idx, result.masks


@contextmanager
def _video_sink(cfg: "PipelineConfig", meta):
    """Yield a `write(frame_rgb)` callable, or None when no video was asked for.

    Either way the frame budget is the same: overlay and encoding sit outside
    it. None just skips the work entirely, for runs that want no mp4 at all.
    """
    if cfg.output_path is None:
        yield None
        return
    with open_video_writer(cfg.output_path, meta.fps, (meta.width, meta.height)) as emit:
        yield emit


@dataclass
class PipelineConfig:
    # None = produce no video. Overlay and encoding are excluded from the
    # frame budget either way; this skips doing them at all.
    output_path: Path | None
    # Exactly one input source must be set:
    #   video_path  -> an .mp4 (or any cv2-readable video) file, OR
    #   frames_dir  -> a directory of pre-extracted frames (frame_000000.tiff, ...).
    video_path: Path | None = None
    frames_dir: Path | None = None
    # Glob used to pick up frames inside frames_dir (frames mode only).
    frame_pattern: str = "*.tif*"
    # Take a centred NxN window instead of the whole frame (frames mode only).
    # The crop becomes the source: masks come back at its size, the overlay is
    # drawn on it, and prompts are shifted into its coordinates. At N equal to
    # the model input this removes the resize from preprocessing entirely.
    center_crop: int | None = None
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
    # Per-frame breakdown: preprocess (decode + resize to model input), model
    # inference, postprocess (masks back to source resolution).
    stage_chart: Path | None = None
    # Deliver frames on a clock at this rate instead of on demand, and drop the
    # ones that arrive while the model is busy (see src/realtime.py). Set it to
    # the rate the deployment has to sustain -- usually the source's own fps.
    # None = off: every frame is tracked, however long that takes.
    realtime_fps: float | None = None
    # Per-frame ceiling in ms to judge the run against, and count the frames
    # that went over. Independent of `realtime_fps` on purpose: a 30 FPS source
    # gives every frame a 33.3 ms slot, but the target being chased may be
    # tighter (or looser) than the slot. Defaults to the slot when only
    # `realtime_fps` is set, and to no verdict when neither is.
    deadline_ms: float | None = None


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
    encode_ms: list[float] | None = None,
    read_ms: list[float] | None = None,
    source: RealtimeSource | None = None,
) -> None:
    """Print FPS stats (with warm-up exclusion) and optionally write charts."""
    if not per_frame_dt:
        return
    # An explicit ceiling wins over the source's slot: "every frame under 35 ms"
    # and "keep up with 30 FPS" are different targets, and the tighter one is
    # usually the one being chased.
    deadline = cfg.deadline_ms or (1000.0 / cfg.realtime_fps if cfg.realtime_fps else None)
    # On a real-time source, `_frame_stream` already ran `cfg.fps_warmup` frames
    # before the clock started and never put them in `per_frame_dt` at all --
    # applying the same exclusion again here would strip that many *more*
    # frames off an already-warmup-free list, which on a short, heavily-skipped
    # run can collapse the whole report to a single sample.
    effective_warmup = 0 if cfg.realtime_fps is not None else cfg.fps_warmup
    stats = fps_summary(per_frame_dt, warmup=effective_warmup, deadline_ms=deadline)
    print(f"[pipeline] tracked {stats['frames']} frames in {stats['total_s']:.2f}s "
          f"({stats['avg_fps_all']:.1f} FPS all)")
    if stats["warmup"] > 0:
        print(f"[pipeline] avg {stats['avg_fps_post_warmup']:.1f} FPS over "
              f"{stats['kept_frames']} frames (excluded first {stats['warmup']} warm-up)")

    # Whether a frame rate holds is a question about the slowest frames, not the
    # average one -- so print the tail next to it rather than only drawing it on
    # a chart nobody parses.
    tail = [f"p50 {stats['p50_ms']:.1f}", f"p95 {stats['p95_ms']:.1f}",
            f"p99 {stats['p99_ms']:.1f}"]
    # Below ~1000 frames p99.9 is the max by construction, so printing both
    # would dress one number up as two.
    if stats["p999_meaningful"]:
        tail.append(f"p99.9 {stats['p999_ms']:.1f}")
    tail.append(f"max {stats['max_ms']:.1f} ms")
    print("[pipeline] per-frame budget: " + " · ".join(tail))
    if deadline is not None:
        share = 100.0 * stats["missed"] / stats["kept_frames"] if stats["kept_frames"] else 0.0
        verdict = "met" if stats["missed"] == 0 else f"{stats['missed']} over ({share:.1f}%)"
        source_of = ("ceiling" if cfg.deadline_ms
                     else f"{cfg.realtime_fps:.1f} FPS slot")
        print(f"[pipeline] {deadline:.1f} ms {source_of}: {verdict}")

    # The FPS above is how fast the model ran. On a real-time source that is no
    # longer the same question as how much of the clip it saw, so say both.
    if source is not None:
        offered = source.delivered + source.dropped
        share = 100.0 * source.dropped / offered if offered else 0.0
        print(f"[pipeline] real-time source at {source.fps:.1f} FPS: tracked "
              f"{source.delivered} of {offered} frames, dropped {source.dropped} "
              f"({share:.1f}%) as stale")
        # The number that verifies the mechanism, not just describes it: a
        # one-slot queue's item is always the most recent one produced before
        # pickup, so its age there is bounded by one source period no matter
        # how far behind the model's throughput is -- only the drop rate above
        # changes. This exceeding the period on a real run is the one thing
        # that would mean the queue itself is misbehaving, not just busy.
        lag_ms = sorted(s * 1000.0 for s in source.lag_s)
        period_ms = 1000.0 / source.fps
        if lag_ms:
            p50 = lag_ms[len(lag_ms) // 2]
            p99 = lag_ms[min(len(lag_ms) - 1, int(0.99 * len(lag_ms)))]
            worst = lag_ms[-1]
            verdict = "bounded as expected" if worst <= period_ms + 1.0 else \
                "OVER the source period -- the queue is not behaving as a one-slot mailbox"
            print(f"[pipeline] frame staleness at pickup: p50 {p50:.1f} · "
                  f"p99 {p99:.1f} · max {worst:.1f} ms (source period "
                  f"{period_ms:.1f} ms): {verdict}")

    # Otherwise these three only exist in the stage chart's title, which makes
    # the split unreadable without opening a PNG and unparseable by a caller.
    if infer_ms:
        w = min(effective_warmup, max(len(infer_ms) - 1, 0))
        line = (f"[pipeline] median per frame: pre {_median(pre_ms, w):.1f} + "
                f"inference {_median(infer_ms, w):.1f} + "
                f"post {_median(post_ms, w):.1f} ms")
        print(line)
        # Opening the frame file is excluded from everything above, the same
        # way the overlay and the encode are. Say what it cost and what the
        # budget would be with it, so nothing is hidden by the choice.
        read = _median(read_ms, w) if read_ms else 0.0
        if read > 0:
            with_io = stats["avg_fps_post_warmup"]
            budget = _median(pre_ms, w) + _median(infer_ms, w) + _median(post_ms, w)
            print(f"[pipeline] reading the frame file: {read:.1f} ms/frame, excluded "
                  "-- a camera or network link hands the frame over in memory, "
                  "there is no file to open")
            if budget + read > 0:
                print(f"[pipeline]   budget {budget:.1f} ms ({with_io:.1f} FPS); "
                      f"with file I/O it would be {budget + read:.1f} ms "
                      f"({1000.0 / (budget + read):.1f} FPS)")

    # The figure above is the deployable frame budget: preprocess + model +
    # mask postprocess. Overlay and encoding are excluded from it by
    # construction, but they are still measured, so say what they cost rather
    # than leaving the gap against wall-clock time unexplained.
    if encode_ms and cfg.output_path is not None:
        w = min(stats["warmup"], max(len(encode_ms) - 1, 0))
        kept = encode_ms[w:] or encode_ms
        if kept:
            overlay = sum(kept) / len(kept)
            print(f"[pipeline] overlay + mp4 encoding: {overlay:.1f} ms/frame "
                  "(not counted above -- demo output, not tracking)")

    if cfg.fps_chart is None and cfg.stage_chart is None:
        return
    note = _benchmark_note(tracker, meta, prompts)
    label = getattr(tracker, "name", None)
    if cfg.fps_chart is not None:
        per_frame_ms = [dt * 1000.0 for dt in per_frame_dt]
        out = write_latency_chart(per_frame_ms, cfg.fps_chart, warmup=effective_warmup,
                                   note=note, label=label)
        if out:
            print(f"[pipeline] wrote latency chart -> {out}")
    if cfg.stage_chart is not None:
        out = write_stage_chart(pre_ms, infer_ms, post_ms, cfg.stage_chart,
                                 warmup=effective_warmup, note=note, label=label,
                                 encode_ms=encode_ms)
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


def run(tracker: VideoTracker, prompts: PromptSet, cfg: PipelineConfig) -> Path | None:
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

    # Cropping here, in the pass that already reads every frame, means the crop
    # is what lands in the JPG cache -- so it is the source for the model, the
    # mask resize and the overlay alike, with no second decode.
    view = lambda rgb: rgb  # noqa: E731
    if cfg.center_crop:
        x0, y0, cw, ch = _center_crop_window(meta.width, meta.height, cfg.center_crop)
        view = lambda rgb: rgb[y0:y0 + ch, x0:x0 + cw]  # noqa: E731
        meta = replace(meta, width=cw, height=ch)
        prompts = _shift_prompts(prompts, x0, y0)
        print(f"[pipeline] centre crop {cw}x{ch} at ({x0}, {y0})")

    cache_root = Path(cfg.frames_cache) if cfg.frames_cache else Path(tempfile.mkdtemp(prefix="frames_"))
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        for idx, src in enumerate(tqdm(frame_files, desc="transcoding", unit="frame")):
            rgb = view(load_frame_rgb8(src))
            cv2.imwrite(str(cache_root / f"{idx:05d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        tracker.prepare(cache_root)
        tracker.set_prompts(prompts)

        per_frame_dt: list[float] = []
        pre_ms: list[float] = []
        infer_ms: list[float] = []
        post_ms: list[float] = []
        encode_ms: list[float] = []
        read_ms: list[float] = []
        preprocess_s: list[float] = []
        read_s: list[float] = []
        wait_s: list[float] = []
        # The source frames, not the JPG cache: the cache only existed to get
        # `init_state` through its bulk load, and re-decoding it per frame
        # would add a lossy compression round-trip to every measurement.
        _install_realtime_preprocess(tracker, frame_files, preprocess_s, view, read_s)

        progress = tqdm(total=len(frame_files), desc="tracking [frames]", unit="frame")
        previous = None
        with _frame_stream(tracker, cfg, prompts, len(frame_files), wait_s,
                           preprocess_s, read_s) as (results, source), \
                _video_sink(cfg, meta) as emit, _postprocess_timer(tracker) as post_s:
            t0 = time.perf_counter()
            for result in results:
                t1 = time.perf_counter()
                per_frame_dt.append(_split_frame(
                    t1 - t0, preprocess_s, post_s, read_s, wait_s,
                    pre_ms, infer_ms, post_ms, read_ms))
                if emit is not None:
                    # Overlay and encoding are how a demo video gets made, not
                    # part of tracking; timed separately, outside the budget.
                    e0 = time.perf_counter()
                    for idx, masks in _emitted(previous, result):
                        rgb = view(load_frame_rgb8(frame_files[idx]))
                        emit(overlay_masks(rgb, masks, alpha=cfg.mask_alpha,
                                           draw_bbox=cfg.draw_bbox))
                    encode_ms.append((time.perf_counter() - e0) * 1000.0)
                previous = result
                t0 = time.perf_counter()
                progress.update(1)
        progress.close()
        _report_timing(cfg, tracker, meta, prompts, per_frame_dt, pre_ms, infer_ms,
                       post_ms, encode_ms, read_ms, source)
        return _written_path(emit, cfg)
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

    per_frame_dt: list[float] = []
    pre_ms: list[float] = []
    infer_ms: list[float] = []
    post_ms: list[float] = []
    encode_ms: list[float] = []
    # decord decodes and resizes in one indexing call, so the file read cannot
    # be separated from the transform here; nothing is subtracted in mp4 mode.
    read_ms: list[float] = []
    read_s: list[float] = []
    wait_s: list[float] = []
    preprocess_s: list[float] = []
    if not _install_realtime_preprocess_mp4(tracker, video_path, preprocess_s):
        print("[pipeline] WARNING: could not install per-frame preprocessing timing "
              "for mp4 mode -- decode+resize cost will be folded into 'inference' "
              "rather than reported as 'pre'. Total frame time is still correct.")
    cursor = 0
    drained = False
    previous = None
    progress = tqdm(total=meta.frame_count, desc="tracking [mp4]", unit="frame")
    try:
        with _frame_stream(tracker, cfg, prompts, meta.frame_count, wait_s,
                           preprocess_s, read_s) as (results, source), \
                _video_sink(cfg, meta) as emit, _postprocess_timer(tracker) as post_s:
            t0 = time.perf_counter()
            for result in results:
                t1 = time.perf_counter()
                per_frame_dt.append(_split_frame(
                    t1 - t0, preprocess_s, post_s, read_s, wait_s,
                    pre_ms, infer_ms, post_ms, read_ms))
                if emit is not None:
                    e0 = time.perf_counter()
                    # Sequential read; `_emitted` yields indices in order.
                    for idx, masks in _emitted(previous, result):
                        while cursor < idx:
                            cap.read()
                            cursor += 1
                        ok, bgr = cap.read()
                        if not ok:
                            drained = True
                            break
                        cursor += 1
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        emit(overlay_masks(rgb, masks, alpha=cfg.mask_alpha,
                                           draw_bbox=cfg.draw_bbox))
                    encode_ms.append((time.perf_counter() - e0) * 1000.0)
                    if drained:
                        break
                previous = result
                t0 = time.perf_counter()
                progress.update(1)
    finally:
        progress.close()
        cap.release()
        tracker.reset()

    _report_timing(cfg, tracker, meta, prompts, per_frame_dt, pre_ms, infer_ms,
                       post_ms, encode_ms, read_ms, source)
    return _written_path(emit, cfg)


def _run_jpg(tracker, prompts, cfg, video_path):
    """Legacy path: dump JPGs to a temp dir then track."""
    cache_root = Path(cfg.frames_cache) if cfg.frames_cache else Path(tempfile.mkdtemp(prefix="frames_"))
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        meta = extract_frames(video_path, cache_root)
        tracker.prepare(cache_root)
        tracker.set_prompts(prompts)

        frame_files = sorted(cache_root.glob("*.jpg"))
        per_frame_dt: list[float] = []
        pre_ms: list[float] = []
        infer_ms: list[float] = []
        post_ms: list[float] = []
        encode_ms: list[float] = []
        read_ms: list[float] = []
        preprocess_s: list[float] = []
        read_s: list[float] = []
        wait_s: list[float] = []
        _install_realtime_preprocess(tracker, frame_files, preprocess_s,
                                     read_timer=read_s)
        progress = tqdm(total=len(frame_files), desc="tracking [jpg]", unit="frame")
        previous = None
        with _frame_stream(tracker, cfg, prompts, len(frame_files), wait_s,
                           preprocess_s, read_s) as (results, source), \
                _video_sink(cfg, meta) as emit, _postprocess_timer(tracker) as post_s:
            t0 = time.perf_counter()
            for result in results:
                t1 = time.perf_counter()
                per_frame_dt.append(_split_frame(
                    t1 - t0, preprocess_s, post_s, read_s, wait_s,
                    pre_ms, infer_ms, post_ms, read_ms))
                if emit is not None:
                    e0 = time.perf_counter()
                    for idx, masks in _emitted(previous, result):
                        bgr = cv2.imread(str(frame_files[idx]))
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        emit(overlay_masks(rgb, masks, alpha=cfg.mask_alpha,
                                           draw_bbox=cfg.draw_bbox))
                    encode_ms.append((time.perf_counter() - e0) * 1000.0)
                previous = result
                t0 = time.perf_counter()
                progress.update(1)
        progress.close()
        _report_timing(cfg, tracker, meta, prompts, per_frame_dt, pre_ms, infer_ms,
                       post_ms, encode_ms, read_ms, source)
        return _written_path(emit, cfg)
    finally:
        tracker.reset()
        if not cfg.keep_frames and cfg.frames_cache is None:
            shutil.rmtree(cache_root, ignore_errors=True)


def grab_first_frame(video_path: str | Path) -> np.ndarray:
    return read_first_frame(video_path)


def grab_first_frame_dir(frames_dir: str | Path, pattern: str = "*.tif*") -> np.ndarray:
    return read_first_frame_dir(frames_dir, pattern)
