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
from .trackers import VideoTracker
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


def _split_frame(total_s, preprocess_s, post_s, read_s, pre_ms, infer_ms, post_ms,
                 read_ms) -> float:
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

    `read_s` empty (mp4 mode, or a tracker with no instrumented loader) means
    nothing is subtracted and this behaves as it always did.
    """
    read = sum(read_s) * 1000.0
    read_s.clear()
    pre = sum(preprocess_s) * 1000.0 - read  # the loader timed read + transform
    preprocess_s.clear()
    post = sum(post_s) * 1000.0
    post_s.clear()
    budget = total_s * 1000.0 - read
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


def _decimate_prompts(prompts: PromptSet, stride: int) -> PromptSet:
    """Move prompts from source-frame onto tracked-frame numbering.

    Source frame `i` is inferred frame `i // stride`. A prompt on a frame that
    is never inferred snaps back to the last one that is -- the nearest frame
    the model will actually encode -- and says so, because a prompt silently
    landing on a different image than the user drew it on is the one way frame
    skipping can be wrong rather than merely coarser.
    """
    if stride <= 1:
        return prompts
    missed = sorted({p.frame_idx for p in (*prompts.boxes, *prompts.points)
                     if p.frame_idx % stride})
    if missed:
        print(f"[pipeline] frame skip {stride}: prompt frame(s) {missed} are not "
              f"inferred; snapped back to {[i - i % stride for i in missed]}")
    return PromptSet(
        boxes=[replace(b, frame_idx=b.frame_idx // stride) for b in prompts.boxes],
        points=[replace(p, frame_idx=p.frame_idx // stride) for p in prompts.points],
    )


def _apply_frame_skip(frame_files: list[Path], prompts: PromptSet,
                      stride: int) -> tuple[list[Path], PromptSet]:
    """Keep one source frame in `stride`; return the clip the model will see.

    This is the inference half of frame skipping. The tracker is handed a
    shorter clip and never learns frames were dropped: `init_state` counts
    these frames, `propagate_in_video` iterates over them, the memory bank
    stores them, and `_LazyFrames` decodes them. A skipped frame is not
    inferred cheaply -- it is not inferred at all.

    The source clip keeps its full length regardless: every frame still gets a
    mask, the skipped ones get the last one the model produced (`_covered_frames`
    is the other half). So this does not shorten the video or thin the output,
    it halves the inferences behind it.

    What that buys is deadline, not speed: a frame still costs what it cost,
    but one model step per `stride` source frames means a 30 fps camera allows
    `stride` x 33.3 ms for it. It costs temporal resolution in exchange --
    consecutive inferred frames are `stride` source frames apart, so apparent
    motion between them is `stride` times larger, the memory bank spans
    `stride` times more real time, and a held mask is up to `stride - 1`
    frames stale.
    """
    if stride <= 1:
        return frame_files, prompts
    kept = frame_files[::stride]
    print(f"[pipeline] frame skip {stride}: inferring {len(kept)} of "
          f"{len(frame_files)} frames; the other {len(frame_files) - len(kept)} "
          "hold the previous mask")
    return kept, _decimate_prompts(prompts, stride)


def _covered_frames(tracked_idx: int, stride: int, total: int) -> range:
    """Source frames that carry inferred frame `tracked_idx`'s masks.

    The inferred frame itself, then the skipped ones that follow it. Holding
    the mask forwards -- never backwards -- is not a rendering convenience: a
    mask cannot exist before the inference that produced it, so this is the
    only ordering a real-time consumer could actually see. At `stride` 1 the
    range is the single frame, i.e. exactly what the pipeline did before.
    """
    start = tracked_idx * stride
    return range(start, min(start + stride, total))


def _spread_over_source(values: list[float], stride: int, total: int,
                        share: bool = True) -> list[float]:
    """Re-index a per-inference series onto the source timeline.

    Every entry becomes ms per *source* frame, so a skipped run's chart has the
    same length, the same x axis and the same unit as an unskipped one and the
    two can be read against each other -- and against the source frame period,
    which is the line the whole pipeline has to stay under.

    `share` is which cadence the measured work runs at. Inference happens once
    per `stride` frames, so its cost is divided over the frames it carries
    (a 60 ms inference covering 2 frames is 30 ms of each frame's budget).
    The overlay and encode happen once per written frame and are already per
    frame, so they repeat instead.

    What this view cannot show is the tail: a 60 ms spike reads as 30 ms here.
    That is why `_report_timing` prints the slowest single inference against
    its own deadline separately -- the chart answers "is it keeping up", the
    printed line answers "did any one frame miss".
    """
    if stride <= 1:
        return values
    out: list[float] = []
    for idx, value in enumerate(values):
        span = len(_covered_frames(idx, stride, total))
        if span <= 0:
            break
        out.extend([value / span if share else value] * span)
    return out


def _source_frame_count(per_frame_dt: list[float], stride: int, meta) -> int:
    """How many source frames the run covered.

    From `meta` when it is consistent with the inference count, since the last
    group can be partial; otherwise from the inferences themselves, so a video
    container lying about its frame count cannot distort the chart.
    """
    n = len(per_frame_dt)
    total = int(getattr(meta, "frame_count", 0) or 0)
    if (n - 1) * stride < total <= n * stride:
        return total
    return n * stride


def _tracked_cache(cache_root: Path, keep: list[Path]) -> Path:
    """A directory holding just `keep`, renumbered 00000.jpg, 00001.jpg, ...

    `init_state` is handed a directory, not a list, so in jpg mode the frames
    the model must not see have to be out of it -- while the overlay still needs
    every one of them to render the held frames. Hence a second directory of
    hard links rather than deleting or moving anything: an inode each, no
    pixels copied. Falls back to a copy on a filesystem that refuses the link.
    """
    out = cache_root / "tracked"
    out.mkdir(exist_ok=True)
    for idx, src in enumerate(keep):
        dst = out / f"{idx:05d}.jpg"
        if dst.exists():
            dst.unlink()
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    return out


def _median(values: list[float], warmup: int) -> float:
    kept = sorted(values[warmup:] or values)
    return kept[len(kept) // 2] if kept else 0.0


@contextmanager
def _video_sink(cfg: "PipelineConfig", meta):
    """Yield a `write(frame_rgb)` callable, or None when no video was asked for.

    Either way the frame budget is the same: overlay and encoding sit outside
    it. None just skips the work entirely, for runs that want no mp4 at all.
    """
    if cfg.output_path is None:
        yield None
        return
    # Source frame rate even under a frame skip: every source frame is written,
    # the skipped ones carrying the last mask the model produced. The output is
    # the same clip either way, so a skipped run can be played against an
    # unskipped one frame for frame.
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
    # Infer one source frame in `frame_stride` and hold that mask over the rest
    # (1 = infer every frame). Every source frame still gets a mask and the
    # output clip keeps its full length; what halves at stride 2 is the number
    # of inferences behind it. So this does not make a frame cheaper -- it gives
    # each inferred frame `frame_stride` frame periods to finish in, at the cost
    # of `frame_stride`x the motion between the frames the memory bank sees.
    frame_stride: int = 1
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

    # Every number above counts inferred frames, which under a skip is not what
    # the camera delivers or what the output contains. State the source rate
    # too, and state the deadline explicitly: that is the only thing skipping
    # actually changes.
    if cfg.frame_stride > 1:
        # The last inference carries a full group only when the clip divides
        # evenly, so count the frames rather than multiplying by the stride --
        # 42 inferences over an 83-frame clip cover 83, not 84.
        covered = _source_frame_count(per_frame_dt, cfg.frame_stride, meta)
        print(f"[pipeline] frame skip {cfg.frame_stride}: {stats['frames']} inferences "
              f"carry {covered} source frames "
              f"({stats['avg_fps_post_warmup'] * cfg.frame_stride:.1f} source FPS)")

    # Whether the run met its deadline, stated for every run and not only for a
    # skipped one: an average hides the frame that missed, and a skipped run
    # cannot be compared against a baseline that never said what its own worst
    # frame cost. `frame_stride` frame periods is what one inference is allowed
    # -- at stride 1 that is simply the frame period.
    if meta is not None and meta.fps > 0 and per_frame_dt:
        budget = 1000.0 * cfg.frame_stride / meta.fps
        w = min(cfg.fps_warmup, max(len(per_frame_dt) - 1, 0))
        worst = max(per_frame_dt[w:] or per_frame_dt) * 1000.0
        verdict = "within" if worst <= budget else "OVER"
        if cfg.frame_stride > 1:
            print(f"[pipeline]   a {meta.fps:.1f} fps source gives each inference "
                  f"{budget:.1f} ms (was {budget / cfg.frame_stride:.1f} ms) -- "
                  "a frame does not get cheaper, its deadline moves")
            # The charts plot the amortised cost, where a spike is divided by
            # the frames it carries. That is the right view of "is it keeping
            # up" and the wrong one for "did any single frame miss", so the
            # worst inference is stated raw, against the deadline it had.
            print(f"[pipeline]   slowest inference {worst:.1f} ms -- {verdict} its "
                  f"{budget:.1f} ms; per source frame that is "
                  f"{worst / cfg.frame_stride:.1f} of {1000.0 / meta.fps:.1f} ms")
        else:
            print(f"[pipeline] slowest frame {worst:.1f} ms -- {verdict} the "
                  f"{budget:.1f} ms a {meta.fps:.1f} fps source allows")

    # Otherwise these three only exist in the stage chart's title, which makes
    # the split unreadable without opening a PNG and unparseable by a caller.
    if infer_ms:
        w = min(cfg.fps_warmup, max(len(infer_ms) - 1, 0))
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

    # Both charts are drawn on the source timeline, not the inference one: 500
    # source frames stay 500 points whether they took 500 inferences or 250, so
    # a skipped run and a baseline are the same plot of the same clip and can be
    # laid over each other. At stride 1 this is the identity.
    stride = cfg.frame_stride
    total = _source_frame_count(per_frame_dt, stride, meta)
    spread = lambda v, share=True: _spread_over_source(v, stride, total, share)  # noqa: E731
    chart_warmup = cfg.fps_warmup * stride
    if stride > 1:
        note += (f"  ·  frame skip {stride}: per source frame, each inference "
                 f"shared over the {stride} frames it carries")

    if cfg.fps_chart is not None:
        per_frame_ms = spread([dt * 1000.0 for dt in per_frame_dt])
        out = write_latency_chart(per_frame_ms, cfg.fps_chart, warmup=chart_warmup,
                                   note=note, label=label)
        if out:
            print(f"[pipeline] wrote latency chart -> {out}")
    if cfg.stage_chart is not None:
        out = write_stage_chart(spread(pre_ms), spread(infer_ms), spread(post_ms),
                                 cfg.stage_chart, warmup=chart_warmup, note=note,
                                 label=label,
                                 encode_ms=spread(encode_ms, share=False) if encode_ms
                                 else encode_ms)
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
        if cfg.frame_stride > 1:
            raise ValueError("video_mode=mp4 hands the whole file to decord, which "
                             "leaves no frame list to decimate; frame skipping needs "
                             "video_mode=jpg (or --frames-dir).")
        return "mp4"
    # auto
    if suffix == ".mp4" and _decord_available():
        if cfg.frame_stride == 1:
            return "mp4"
        print("[pipeline] frame skip: falling back to jpg mode -- the direct-mp4 "
              "path hands the whole file to decord and cannot be decimated.")
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
    if cfg.frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1 (got {cfg.frame_stride})")

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
    # Two lists from here on, and the difference is the feature: `tracked` is
    # what the model is given -- the JPG cache is transcoded from it, so it is
    # also what `init_state` counts and what the lazy loader decodes -- while
    # `frame_files` stays the whole source, because every frame still gets a
    # mask and lands in the output.
    tracked, prompts = _apply_frame_skip(frame_files, prompts, cfg.frame_stride)

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
        for idx, src in enumerate(tqdm(tracked, desc="transcoding", unit="frame")):
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
        # The source frames, not the JPG cache: the cache only existed to get
        # `init_state` through its bulk load, and re-decoding it per frame
        # would add a lossy compression round-trip to every measurement.
        _install_realtime_preprocess(tracker, tracked, preprocess_s, view, read_s)

        progress = tqdm(total=len(tracked), desc="tracking [frames]", unit="frame")
        with _video_sink(cfg, meta) as emit, _postprocess_timer(tracker) as post_s:
            t0 = time.perf_counter()
            for result in tracker.propagate():
                t1 = time.perf_counter()
                per_frame_dt.append(_split_frame(
                    t1 - t0, preprocess_s, post_s, read_s,
                    pre_ms, infer_ms, post_ms, read_ms))
                if emit is not None:
                    # Overlay and encoding are how a demo video gets made, not
                    # part of tracking; timed separately, outside the budget.
                    # Under a skip this writes the inferred frame and the ones
                    # holding its mask, so the cost stays per written frame.
                    e0 = time.perf_counter()
                    span = _covered_frames(result.frame_idx, cfg.frame_stride,
                                           len(frame_files))
                    for src_idx in span:
                        rgb = view(load_frame_rgb8(frame_files[src_idx]))
                        emit(overlay_masks(rgb, result.masks, alpha=cfg.mask_alpha,
                                           draw_bbox=cfg.draw_bbox))
                    encode_ms.append((time.perf_counter() - e0) * 1000.0 / max(len(span), 1))
                t0 = time.perf_counter()
                progress.update(1)
        progress.close()
        _report_timing(cfg, tracker, meta, prompts, per_frame_dt, pre_ms, infer_ms,
                       post_ms, encode_ms, read_ms)
        return Path(cfg.output_path).resolve() if cfg.output_path else None
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
    preprocess_s: list[float] = []
    if not _install_realtime_preprocess_mp4(tracker, video_path, preprocess_s):
        print("[pipeline] WARNING: could not install per-frame preprocessing timing "
              "for mp4 mode -- decode+resize cost will be folded into 'inference' "
              "rather than reported as 'pre'. Total frame time is still correct.")
    cursor = 0
    progress = tqdm(total=meta.frame_count, desc="tracking [mp4]", unit="frame")
    try:
        with _video_sink(cfg, meta) as emit, _postprocess_timer(tracker) as post_s:
            t0 = time.perf_counter()
            for result in tracker.propagate():
                t1 = time.perf_counter()
                per_frame_dt.append(_split_frame(
                    t1 - t0, preprocess_s, post_s, read_s,
                    pre_ms, infer_ms, post_ms, read_ms))
                if emit is not None:
                    e0 = time.perf_counter()
                    # Sequential read; propagate yields frames in order.
                    while cursor < result.frame_idx:
                        cap.read()
                        cursor += 1
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    cursor += 1
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    emit(overlay_masks(rgb, result.masks, alpha=cfg.mask_alpha,
                                       draw_bbox=cfg.draw_bbox))
                    encode_ms.append((time.perf_counter() - e0) * 1000.0)
                t0 = time.perf_counter()
                progress.update(1)
    finally:
        progress.close()
        cap.release()
        tracker.reset()

    _report_timing(cfg, tracker, meta, prompts, per_frame_dt, pre_ms, infer_ms,
                       post_ms, encode_ms, read_ms)
    return Path(cfg.output_path).resolve() if cfg.output_path else None


def _run_jpg(tracker, prompts, cfg, video_path):
    """Legacy path: dump JPGs to a temp dir then track."""
    cache_root = Path(cfg.frames_cache) if cfg.frames_cache else Path(tempfile.mkdtemp(prefix="frames_"))
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        meta = extract_frames(video_path, cache_root)
        frame_files = sorted(cache_root.glob("*.jpg"))
        tracked, prompts = _apply_frame_skip(frame_files, prompts, cfg.frame_stride)
        # The model gets a directory of only the inferred frames; the overlay
        # keeps reading the full cache, which is why they are separate dirs.
        tracker.prepare(_tracked_cache(cache_root, tracked)
                        if cfg.frame_stride > 1 else cache_root)
        tracker.set_prompts(prompts)

        per_frame_dt: list[float] = []
        pre_ms: list[float] = []
        infer_ms: list[float] = []
        post_ms: list[float] = []
        encode_ms: list[float] = []
        read_ms: list[float] = []
        preprocess_s: list[float] = []
        read_s: list[float] = []
        _install_realtime_preprocess(tracker, tracked, preprocess_s,
                                     read_timer=read_s)
        progress = tqdm(total=len(tracked), desc="tracking [jpg]", unit="frame")
        with _video_sink(cfg, meta) as emit, _postprocess_timer(tracker) as post_s:
            t0 = time.perf_counter()
            for result in tracker.propagate():
                t1 = time.perf_counter()
                per_frame_dt.append(_split_frame(
                    t1 - t0, preprocess_s, post_s, read_s,
                    pre_ms, infer_ms, post_ms, read_ms))
                if emit is not None:
                    e0 = time.perf_counter()
                    span = _covered_frames(result.frame_idx, cfg.frame_stride,
                                           len(frame_files))
                    for src_idx in span:
                        bgr = cv2.imread(str(frame_files[src_idx]))
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        emit(overlay_masks(rgb, result.masks, alpha=cfg.mask_alpha,
                                           draw_bbox=cfg.draw_bbox))
                    encode_ms.append((time.perf_counter() - e0) * 1000.0 / max(len(span), 1))
                t0 = time.perf_counter()
                progress.update(1)
        progress.close()
        _report_timing(cfg, tracker, meta, prompts, per_frame_dt, pre_ms, infer_ms,
                       post_ms, encode_ms, read_ms)
        return Path(cfg.output_path).resolve() if cfg.output_path else None
    finally:
        tracker.reset()
        if not cfg.keep_frames and cfg.frames_cache is None:
            shutil.rmtree(cache_root, ignore_errors=True)


def grab_first_frame(video_path: str | Path) -> np.ndarray:
    return read_first_frame(video_path)


def grab_first_frame_dir(frames_dir: str | Path, pattern: str = "*.tif*") -> np.ndarray:
    return read_first_frame_dir(frames_dir, pattern)
