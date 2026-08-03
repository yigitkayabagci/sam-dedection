#!/usr/bin/env python3
"""Benchmark tracking throughput: propagate FPS and per-module inference ms.

Runs the real tracking loop (`tracker.propagate()`) over a synthetic clip and
reports what you actually care about on an Orin:

  * frames per second, warm-up excluded
  * per-frame latency percentiles (p50/p90/p99) -- the tail is what drops
    frames on a live pipeline, and an average hides it
  * where the milliseconds go, module by module

By default it runs the stock PyTorch backend and the TensorRT one back to back
on the same frames, so the speedup is measured rather than inferred.

The breakdown comes from a second, separately timed pass: measuring per-module
time needs a `cudaStreamSynchronize` after every module, which itself costs
time. Mixing that into the throughput number would understate your real FPS.

Usage on the Orin:
    python tools/benchmark_tracking.py --frames 500
    python tools/benchmark_tracking.py --frames 500 --objects 2
    python tools/benchmark_tracking.py --frames 500 --tracker edgetam_trt

    # Latency chart (ms, max/min/avg, warm-up shaded) per backend:
    python tools/benchmark_tracking.py --frames 500 --fps-chart outputs/speed.png
    # -> outputs/speed_edgetam.png, outputs/speed_edgetam_trt.png
"""
from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cli import _load_backend_config, _resolve_paths  # noqa: E402
from src.metrics import format_benchmark_note, write_latency_chart  # noqa: E402
from src.prompts import BoxPrompt, PromptSet  # noqa: E402
from src.trackers import build_tracker  # noqa: E402

RESOLVE_KEYS = (
    "checkpoint",
    "engine_path",
    "image_encoder_engine",
    "memory_attention_engine",
    "memory_encoder_engine",
    "sam_head_engine",
)


# --------------------------------------------------------------------------
# Synthetic clip
# --------------------------------------------------------------------------


def write_synthetic_clip(
    outdir: Path, frames: int, width: int, height: int, objects: int,
    radius: int | None = None,
):
    """A textured background with `objects` blobs on smooth trajectories.

    Smooth motion matters: a clip where the target teleports would make the
    memory bank useless and is not what you will run in production. Returns the
    box prompts for frame 0.

    `radius` defaults to a size proportional to the frame (large, easy target);
    pass a smaller fixed value to stress-test tracking on a small object instead.
    """
    import cv2

    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    background = rng.integers(30, 90, size=(height, width, 3), dtype=np.uint8)
    background = cv2.GaussianBlur(background, (0, 0), 3)

    radius = max(4, radius) if radius is not None else max(12, min(width, height) // 12)
    colors = [(230, 60, 60), (60, 220, 90), (70, 120, 240), (240, 200, 60)]
    starts = [
        (width // 4 + i * width // (2 * max(objects, 1)), height // 2)
        for i in range(objects)
    ]

    def centre(i, t):
        x0, y0 = starts[i]
        phase = t / max(frames - 1, 1)
        x = x0 + int(0.30 * width * np.sin(2 * np.pi * (phase + 0.13 * i)))
        y = y0 + int(0.18 * height * np.cos(2 * np.pi * (phase + 0.29 * i)))
        return (
            int(np.clip(x, radius + 2, width - radius - 2)),
            int(np.clip(y, radius + 2, height - radius - 2)),
        )

    for t in range(frames):
        frame = background.copy()
        for i in range(objects):
            cx, cy = centre(i, t)
            cv2.circle(frame, (cx, cy), radius, colors[i % len(colors)], -1)
            cv2.circle(frame, (cx, cy), radius // 2, (250, 250, 250), 2)
        cv2.imwrite(str(outdir / f"{t:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

    boxes = []
    for i in range(objects):
        cx, cy = centre(i, 0)
        boxes.append(
            BoxPrompt(
                obj_id=i + 1,
                frame_idx=0,
                xyxy=(
                    float(cx - radius),
                    float(cy - radius),
                    float(cx + radius),
                    float(cy + radius),
                ),
            )
        )
    return PromptSet(boxes=boxes)


# --------------------------------------------------------------------------
# Per-module instrumentation
# --------------------------------------------------------------------------


@contextlib.contextmanager
def module_timers(tracker):
    """Wrap the four hot modules with synchronized timers, then restore them.

    Wrapping happens after `prepare()`, so this measures whatever the tracker
    actually installed -- a TensorRT engine or the PyTorch original. Same probe
    for both backends means the numbers are comparable.
    """
    import torch

    predictor = tracker._predictor
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)

    def wrap(label, fn):
        def timed(*args, **kwargs):
            sync()
            start = time.perf_counter()
            out = fn(*args, **kwargs)
            sync()
            totals[label] = totals.get(label, 0.0) + (time.perf_counter() - start)
            counts[label] = counts.get(label, 0) + 1
            return out

        return timed

    targets = [
        ("image_encoder", predictor.image_encoder, "forward"),
        ("memory_attention", predictor.memory_attention, "forward"),
        ("memory_encoder", predictor.memory_encoder, "forward"),
        ("sam_head", predictor, "_forward_sam_heads"),
    ]
    saved = []
    for label, owner, attr in targets:
        original = getattr(owner, attr)
        saved.append((owner, attr, original, attr in vars(owner)))
        setattr(owner, attr, wrap(label, original))
    try:
        yield totals, counts
    finally:
        for owner, attr, original, was_instance_attr in saved:
            if was_instance_attr:
                setattr(owner, attr, original)
            else:
                # It was a class method reached through the instance; drop the
                # instance attribute rather than pinning a bound copy.
                vars(owner).pop(attr, None)


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------


def timed_propagate(tracker, frames_dir, prompts, limit=None):
    """(setup ms, [per-frame ms]) for one propagate run.

    Setup covers model build, engine load, CUDA graph capture and decoding the
    clip onto the GPU -- one-off costs that must not land in the FPS number.
    """
    setup_start = time.perf_counter()
    tracker.prepare(frames_dir)
    tracker.set_prompts(prompts)
    setup_ms = (time.perf_counter() - setup_start) * 1e3
    per_frame = []
    previous = time.perf_counter()
    for count, _ in enumerate(tracker.propagate(), start=1):
        now = time.perf_counter()
        per_frame.append((now - previous) * 1e3)
        previous = now
        if limit is not None and count >= limit:
            break
    tracker.reset()
    return setup_ms, per_frame


def summarise(per_frame, setup_ms, warmup, totals, counts, breakdown_frames):
    kept = per_frame[warmup:] or per_frame
    kept_sorted = sorted(kept)

    def pct(p):
        return kept_sorted[min(len(kept_sorted) - 1, int(p * len(kept_sorted)))]

    mean = sum(kept) / len(kept)
    print(f"  frames              {len(per_frame)} ({warmup} excluded as warm-up)")
    print(f"  propagate           {mean:7.2f} ms/frame   ->  {1000.0 / mean:6.2f} FPS")
    print(f"  latency p50/p90/p99 {pct(0.50):7.2f} /{pct(0.90):7.2f} /{pct(0.99):7.2f} ms")
    print(f"  setup (one-off)     {setup_ms:7.0f} ms (model build, engine load + graph "
          "capture, clip decode)")

    if totals:
        print(f"  per-module inference ({breakdown_frames} frames, synchronized pass)")
        accounted = 0.0
        for label in ("image_encoder", "memory_attention", "memory_encoder", "sam_head"):
            if label not in totals:
                continue
            ms = totals[label] / breakdown_frames * 1e3
            accounted += ms
            calls = counts[label] / breakdown_frames
            print(f"    {label:<18}{ms:7.2f} ms   ({calls:.2f} calls/frame)")
        print(f"    {'--- accounted':<18}{accounted:7.2f} ms")
        print(
            f"    {'other':<18}{max(mean - accounted, 0.0):7.2f} ms   "
            "(memory bookkeeping, mask resize + download, frame indexing)"
        )
    return {"ms": mean, "fps": 1000.0 / mean}


def run_backend(tracker_name, config, frames_dir, prompts, args, chart_tag=None):
    cfg = _resolve_paths(_load_backend_config(tracker_name, config), RESOLVE_KEYS)
    if args.offload_video:
        cfg["offload_video_to_cpu"] = True
    if args.precision:
        cfg["precision"] = args.precision
    if args.no_cuda_graph:
        cfg["use_cuda_graph"] = False
    graphs = cfg.get("use_cuda_graph")
    suffix = ""
    if tracker_name.endswith("_trt"):
        suffix = f", cuda graphs {'on' if graphs is not False else 'off'}"
    label = (f"{tracker_name}  ({cfg.get('precision', '?')}, "
             f"{Path(config).name if config else 'default config'}{suffix})")
    print(f"\n{'=' * 70}\n>> {label}\n{'=' * 70}")

    tracker = build_tracker(tracker_name, **cfg)
    setup_ms, per_frame = timed_propagate(tracker, frames_dir, prompts)

    if args.fps_chart:
        width, height = (int(v) for v in args.size.lower().split("x"))
        model_size = getattr(getattr(tracker, "_predictor", None), "image_size", None)
        note = format_benchmark_note(
            tracker_name=tracker_name, precision=cfg.get("precision"),
            source_size=(width, height), model_size=model_size, batch=args.objects,
        )
        base = Path(args.fps_chart)
        out_path = base.with_name(f"{base.stem}_{chart_tag or tracker_name}{base.suffix}")
        out = write_latency_chart(per_frame, out_path, warmup=args.warmup,
                                   note=note, label=tracker_name)
        if out:
            print(f"  wrote latency chart -> {out}")

    totals = counts = None
    breakdown_frames = min(args.breakdown_frames, len(per_frame))
    if breakdown_frames > 0:
        with module_timers(tracker) as (totals, counts):
            timed_propagate(tracker, frames_dir, prompts, limit=breakdown_frames)
    tracker.reset()
    result = summarise(per_frame, setup_ms, args.warmup, totals, counts, breakdown_frames)
    result["name"] = label
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--frames", type=int, default=500)
    p.add_argument("--size", default="1280x720", help="Source frame size, WxH. The "
                   "model resizes to its own image_size regardless; this only "
                   "affects JPEG decode and the mask download.")
    p.add_argument("--objects", type=int, default=1)
    p.add_argument("--radius", type=int, default=None,
                   help="Target radius in pixels (default: proportional to frame "
                        "size, ~60px at 720p). Pass a small fixed value to stress-"
                        "test tracking on a small object.")
    p.add_argument("--warmup", type=int, default=20,
                   help="Frames excluded from the FPS average.")
    p.add_argument("--tracker", default=None,
                   help="Benchmark one backend. Default: edgetam then edgetam_trt.")
    p.add_argument("--config", default=None,
                   help="Backend YAML for --tracker, or for edgetam_trt in the "
                        "default two-backend run.")
    p.add_argument("--baseline-config", default=None,
                   help="Backend YAML for the stock PyTorch side of the default "
                        "two-backend run. Needed when the two must match on "
                        "something the default YAMLs disagree on -- comparing at "
                        "512, both sides have to run at 512 or the result "
                        "measures the resolution change, not TensorRT.")
    p.add_argument("--precision", default=None,
                   help="Override the config's precision (float16 | bfloat16 | float32).")
    p.add_argument("--breakdown-frames", type=int, default=100,
                   help="Frames for the synchronized per-module pass. 0 to skip.")
    p.add_argument("--no-cuda-graph", action="store_true",
                   help="Force use_cuda_graph=false. Pair with a normal run to "
                        "measure what CUDA graph replay is worth end to end; "
                        "TensorRT itself is unaffected either way.")
    p.add_argument("--cuda-graph-ab", action="store_true",
                   help="Run edgetam_trt twice, with and without CUDA graphs, "
                        "and report the difference. Implies --tracker edgetam_trt.")
    p.add_argument("--offload-video", action="store_true",
                   help="Keep decoded frames on the CPU. EdgeTAM otherwise holds "
                        "every frame on the GPU as fp32.")
    p.add_argument("--fps-chart", default=None,
                   help="Save a per-frame latency chart (ms, with max/min/avg, "
                        "warm-up shaded) for each backend. One PNG per backend, "
                        "named <path stem>_<backend>.png (needs matplotlib).")
    p.add_argument("--keep-frames", action="store_true")
    args = p.parse_args(argv)

    width, height = (int(v) for v in args.size.lower().split("x"))

    try:
        import torch

        if torch.cuda.is_available():
            print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
        else:
            print(f"device: CPU (torch {torch.__version__}) -- timings are not "
                  "representative of an Orin")
    except ImportError:
        pass

    tmp = Path(tempfile.mkdtemp(prefix="bench_frames_"))
    frames_dir = tmp
    print(f"generating {args.frames} synthetic {width}x{height} frames "
          f"with {args.objects} target(s) -> {frames_dir}")
    prompts = write_synthetic_clip(
        frames_dir, args.frames, width, height, args.objects, radius=args.radius
    )

    if not args.offload_video:
        # EdgeTAM preloads the whole clip to the GPU as fp32 at model resolution.
        print(
            f"note: EdgeTAM will hold all {args.frames} frames on the GPU "
            f"(~{args.frames * 3 * 1024 * 1024 * 4 / 1e9:.1f} GB at 1024x1024 fp32). "
            "Pass --offload-video if that is too much."
        )

    try:
        if args.cuda_graph_ab:
            # Same backend twice; only the graph toggle differs.
            backends = [("edgetam_trt", args.config), ("edgetam_trt", args.config)]
            graph_flags = [False, True]
        else:
            backends = (
                [(args.tracker, args.config)]
                if args.tracker
                else [("edgetam", args.baseline_config), ("edgetam_trt", args.config)]
            )
            graph_flags = [args.no_cuda_graph] * len(backends)

        results = []
        for (tracker_name, config), no_graph in zip(backends, graph_flags):
            args.no_cuda_graph = no_graph
            # cuda-graph-ab runs the same tracker_name twice; tag the chart
            # filenames by graph state too, or the second run overwrites the first.
            chart_tag = tracker_name
            if args.cuda_graph_ab:
                chart_tag += "_graphoff" if no_graph else "_graphon"
            try:
                results.append(
                    run_backend(tracker_name, config, frames_dir, prompts, args, chart_tag)
                )
            except Exception as exc:
                print(f"\n!! {tracker_name} failed: {type(exc).__name__}: {exc}")
                if args.tracker or args.cuda_graph_ab:
                    raise

        if len(results) > 1:
            print(f"\n{'=' * 70}\n>> comparison")
            base = results[0]
            for r in results:
                speedup = f"   ({base['ms'] / r['ms']:.2f}x)" if r is not base else ""
                print(f"  {r['name']:<48}{r['ms']:7.2f} ms  {r['fps']:6.2f} FPS{speedup}")
    finally:
        if tmp is not None and not args.keep_frames:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
