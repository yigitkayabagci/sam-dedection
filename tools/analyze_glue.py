#!/usr/bin/env python3
"""Break down the per-frame time that is *not* inside one of the four engines.

`benchmark_tracking.py` reports a single "other" line -- on an Orin, ~2.5 ms of
a ~26 ms frame. The obvious next question is whether fusing the four engines
into one would remove it. Most of it, it turns out, is not inter-module glue at
all, and a fused engine could not touch it.

This tool times the per-frame Python by category:

  bank assembly    reading the memory bank (a Python dict keyed by frame index)
                   and packing it into the engine's fixed-size input buffer
  bank write-back  storing this frame's outputs back into that dict
  mask postprocess resizing masks to video resolution and moving them to the
                   storage device
  launch gap       whatever is left between one engine returning and the next
                   being launched -- this, and only this, is what fusing the
                   engines into a single graph could remove

Run it after check_trt_parity.py, on the machine the engines were built for:

    python tools/analyze_glue.py --frames 200
    python tools/analyze_glue.py --frames 200 --config configs/edgetam_trt.yaml
"""
from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cli import _load_backend_config, _resolve_paths  # noqa: E402
from src.trackers import build_tracker  # noqa: E402
from tools.benchmark_tracking import RESOLVE_KEYS, write_synthetic_clip  # noqa: E402


@contextlib.contextmanager
def instrument(tracker):
    """Wrap the per-frame call sites and accumulate seconds per category.

    Every wrapper synchronizes, so these numbers are not additive with the
    unsynchronized FPS figure -- they are a proportional breakdown, not a
    replacement for it.
    """
    import torch

    predictor = tracker._predictor
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)

    def timed(label, fn):
        def wrapper(*args, **kwargs):
            sync()
            start = time.perf_counter()
            out = fn(*args, **kwargs)
            sync()
            totals[label] = totals.get(label, 0.0) + (time.perf_counter() - start)
            counts[label] = counts.get(label, 0) + 1
            return out

        return wrapper

    # `_prepare_memory_conditioned_features` is bank assembly *plus* the
    # memory_attention call it makes; subtracting the module leaves the glue.
    targets = [
        ("engine:image_encoder", predictor.image_encoder, "forward"),
        ("engine:memory_attention", predictor.memory_attention, "forward"),
        ("engine:memory_encoder", predictor.memory_encoder, "forward"),
        ("engine:sam_head", predictor, "_forward_sam_heads"),
        ("span:prepare_memory", predictor, "_prepare_memory_conditioned_features"),
        ("span:encode_memory", predictor, "_encode_memory_in_output"),
        ("postprocess", predictor, "_get_orig_video_res_output"),
        ("frame:track_step", predictor, "track_step"),
        ("frame:total", predictor, "_run_single_frame_inference"),
    ]
    saved = []
    for label, owner, attr in targets:
        original = getattr(owner, attr)
        saved.append((owner, attr, original, attr in vars(owner)))
        setattr(owner, attr, timed(label, original))
    try:
        yield totals, counts
    finally:
        for owner, attr, original, was_instance_attr in saved:
            if was_instance_attr:
                setattr(owner, attr, original)
            else:
                vars(owner).pop(attr, None)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--size", default="1280x720")
    p.add_argument("--objects", type=int, default=1)
    p.add_argument("--radius", type=int, default=None)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--tracker", default="edgetam_trt")
    p.add_argument("--config", default="configs/edgetam_trt.yaml")
    p.add_argument("--offload-video", action="store_true")
    args = p.parse_args(argv)

    width, height = (int(v) for v in args.size.lower().split("x"))
    cfg = _resolve_paths(_load_backend_config(args.tracker, args.config), RESOLVE_KEYS)
    if args.offload_video:
        cfg["offload_video_to_cpu"] = True

    tmp = Path(tempfile.mkdtemp(prefix="glue_"))
    try:
        print(f">> {args.frames} synthetic frames at {width}x{height}")
        prompts = write_synthetic_clip(
            tmp, args.frames, width, height, args.objects, radius=args.radius
        )
        tracker = build_tracker(args.tracker, **cfg)
        tracker.prepare(tmp)
        tracker.set_prompts(prompts)

        # Warm up outside the instrumented region.
        stream = tracker.propagate()
        for i, _ in enumerate(stream, start=1):
            if i >= args.warmup:
                break

        with instrument(tracker) as (totals, counts):
            frames = 0
            for _ in stream:
                frames += 1
        tracker.reset()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if not frames:
        raise SystemExit(
            "No frames measured. Raise --frames above --warmup."
        )

    def ms(label):
        return totals.get(label, 0.0) / frames * 1e3

    engines = {k: ms(k) for k in totals if k.startswith("engine:")}
    engine_total = sum(engines.values())
    # The two spans each contain one engine call; what is left is the Python
    # that assembles that engine's input or files away its output.
    bank_assembly = max(ms("span:prepare_memory") - ms("engine:memory_attention"), 0.0)
    bank_writeback = max(ms("span:encode_memory") - ms("engine:memory_encoder"), 0.0)
    postprocess = ms("postprocess")
    track_step = ms("frame:track_step")
    frame_total = ms("frame:total")

    # The call tree, which the arithmetic below has to respect:
    #   _run_single_frame_inference          <- frame:total
    #     _get_image_feature                   -> image_encoder (NOT in track_step)
    #     track_step                         <- frame:track_step
    #       _prepare_memory_conditioned_...  <- span:prepare_memory -> memory_attention
    #       _forward_sam_heads                 -> sam_head
    #       _encode_memory_in_output         <- span:encode_memory -> memory_encoder
    #     storage bookkeeping
    #   _get_orig_video_res_output           <- postprocess (outside frame:total)
    in_track_step = (
        ms("engine:memory_attention") + ms("engine:sam_head") + ms("engine:memory_encoder")
    )
    launch_gap = max(track_step - in_track_step - bank_assembly - bank_writeback, 0.0)
    # What is left of the frame once track_step and the image encoder are out:
    # the cached-feature lookup and the per-frame state bookkeeping.
    outside = max(frame_total - track_step - ms("engine:image_encoder"), 0.0)

    print(f"\n{'=' * 66}")
    print(f"per-frame breakdown  ({frames} frames, synchronized)")
    print("=" * 66)
    for label in ("engine:image_encoder", "engine:memory_attention",
                  "engine:memory_encoder", "engine:sam_head"):
        if label in engines:
            print(f"  {label.split(':')[1]:<22}{engines[label]:8.3f} ms")
    print(f"  {'--- engines':<22}{engine_total:8.3f} ms")
    print()
    print(f"  {'bank assembly':<22}{bank_assembly:8.3f} ms   dict -> engine input buffer")
    print(f"  {'bank write-back':<22}{bank_writeback:8.3f} ms   engine output -> dict")
    print(f"  {'launch gap':<22}{launch_gap:8.3f} ms   between engine calls")
    print(f"  {'mask postprocess':<22}{postprocess:8.3f} ms   resize to video res + move")
    print(f"  {'other frame work':<22}{outside:8.3f} ms   feature lookup, state bookkeeping")
    print(f"  {'--- frame total':<22}{frame_total + postprocess:8.3f} ms")
    accounted = engine_total + bank_assembly + bank_writeback + launch_gap + postprocess + outside
    drift = abs(accounted - (frame_total + postprocess))
    if drift > 0.05 * max(frame_total, 1e-9):
        print(f"  [!] categories sum to {accounted:.3f} ms, {drift:.3f} ms off the "
              "measured total -- treat the split as approximate")

    glue = bank_assembly + bank_writeback + launch_gap + postprocess + outside
    total = engine_total + glue
    if total > 0:
        print(f"\n  engines are {engine_total / total * 100:.1f}% of the frame; "
              f"everything else is {glue / total * 100:.1f}%")
    print(
        f"\n  A single fused engine could remove at most the launch gap "
        f"({launch_gap:.3f} ms,\n"
        f"  {launch_gap / total * 100:.1f}% of the frame). Bank assembly and write-back "
        "read and write a\n"
        "  Python dict keyed by frame index, so they survive any fusion; mask\n"
        "  postprocess happens after the model and is not part of it either."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
