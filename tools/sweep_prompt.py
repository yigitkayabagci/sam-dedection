#!/usr/bin/env python3
"""Measure what the prompt actually costs: target size, and object count.

Two claims worth having a number behind rather than an argument:

  target size does not change the frame time
      The model runs at a fixed input resolution -- 1024x1024 by default -- so
      a 12 px blob and a 120 px blob produce identically shaped tensors and
      identical work. If a sweep over --radius comes out flat, that is the
      evidence; if it does not, something depends on the content and is worth
      finding.

  object count does
      Objects are the batch dimension. Two targets is two rows through the
      memory attention and the mask decoder, so this one should rise -- the
      question is how steeply, and whether the TensorRT engines' optimisation
      profile (--max-batch) still covers it.

Each configuration is a separate tracking run over a freshly generated clip, so
they share nothing but the code.

Usage:
    python tools/sweep_prompt.py --radii 12,30,60,120 --objects 1,2,3
    python tools/sweep_prompt.py --radii 12,60 --objects 1 --frames 300 \\
        --out results/1024/05_sweep
"""
from __future__ import annotations

import argparse
import json
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


def run_one(cfg: dict, tracker_name: str, frames_dir: Path, prompts, warmup: int) -> dict:
    """One tracking pass; returns ms/frame statistics past warm-up."""
    tracker = build_tracker(tracker_name, **cfg)
    try:
        tracker.prepare(frames_dir)
        tracker.set_prompts(prompts)
        per_frame = []
        previous = time.perf_counter()
        for _ in tracker.propagate():
            now = time.perf_counter()
            per_frame.append((now - previous) * 1e3)
            previous = now
    finally:
        tracker.reset()
        del tracker
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    kept = sorted(per_frame[warmup:] or per_frame)
    if not kept:
        return {}
    return {
        "frames": len(per_frame),
        "mean_ms": sum(kept) / len(kept),
        "p50_ms": kept[len(kept) // 2],
        "p99_ms": kept[min(len(kept) - 1, int(0.99 * len(kept)))],
        "fps": 1000.0 * len(kept) / sum(kept),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--radii", default="12,30,60,120",
                   help="Comma-separated target radii in px.")
    p.add_argument("--objects", default="1,2,3",
                   help="Comma-separated object counts.")
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--size", default="1280x720")
    p.add_argument("--tracker", default="edgetam_trt")
    p.add_argument("--config", default="configs/edgetam_trt.yaml")
    p.add_argument("--offload-video", action="store_true")
    p.add_argument("--out", default=None, help="Path stem for .json/.md output.")
    args = p.parse_args(argv)

    radii = [int(v) for v in args.radii.split(",") if v.strip()]
    objects = [int(v) for v in args.objects.split(",") if v.strip()]
    width, height = (int(v) for v in args.size.lower().split("x"))

    base = _resolve_paths(_load_backend_config(args.tracker, args.config), RESOLVE_KEYS)
    if args.offload_video:
        base["offload_video_to_cpu"] = True

    results = []
    # Radius sweep at one object; then object sweep at one radius. A full grid
    # multiplies runs for a cross-term neither claim depends on.
    plan = [("radius", r, objects[0]) for r in radii]
    plan += [("objects", radii[0], n) for n in objects if n != objects[0]]

    for axis, radius, count in plan:
        tmp = Path(tempfile.mkdtemp(prefix="sweep_"))
        try:
            prompts = write_synthetic_clip(
                tmp, args.frames, width, height, count, radius=radius
            )
            print(f"\n>> radius={radius}px  objects={count}")
            stats = run_one(base, args.tracker, tmp, prompts, args.warmup)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if not stats:
            print("   no frames measured")
            continue
        stats.update(axis=axis, radius=radius, objects=count)
        results.append(stats)
        print(f"   {stats['mean_ms']:7.2f} ms/frame   {stats['fps']:6.2f} FPS   "
              f"p99 {stats['p99_ms']:.2f} ms")

    if not results:
        raise SystemExit("Nothing measured.")

    lines = [
        f"# Prompt sweep — {args.tracker}",
        "",
        f"{args.frames} frames at {args.size}, {args.warmup} warm-up excluded, "
        f"config `{args.config}`",
        "",
        "## Target size (objects fixed)",
        "",
        "| radius px | ms/frame | FPS | p99 ms |",
        "|---|---|---|---|",
    ]
    size_rows = [r for r in results if r["axis"] == "radius"]
    for r in size_rows:
        lines.append(f"| {r['radius']} | {r['mean_ms']:.2f} | {r['fps']:.2f} | "
                     f"{r['p99_ms']:.2f} |")
    if len(size_rows) > 1:
        lo = min(r["mean_ms"] for r in size_rows)
        hi = max(r["mean_ms"] for r in size_rows)
        spread = (hi - lo) / lo * 100.0
        measured = min(r["frames"] for r in size_rows) - args.warmup
        verdict = (
            "Flat, as expected: the model resizes every frame to a fixed input, "
            "so a target's size on screen never reaches it."
            if spread < 10 else
            "Not flat. Either something in the per-frame path depends on how much "
            "of the frame the mask covers, or the run-to-run noise floor is this "
            "wide — re-run with more frames, and with clocks pinned "
            "(`sudo jetson_clocks`), before concluding the former."
        )
        lines += [
            "",
            f"Spread across target sizes: **{spread:.1f}%** ({lo:.2f}–{hi:.2f} ms), "
            f"over {measured} measured frames per point. {verdict}",
        ]

    count_rows = [r for r in results if r["axis"] == "objects"]
    if count_rows:
        first = size_rows[0] if size_rows else None
        lines += [
            "",
            "## Object count (target size fixed)",
            "",
            "| objects | ms/frame | FPS | p99 ms | vs 1 object |",
            "|---|---|---|---|---|",
        ]
        rows = ([first] if first else []) + count_rows
        base_ms = rows[0]["mean_ms"] if rows else None
        for r in rows:
            ratio = f"{r['mean_ms'] / base_ms:.2f}x" if base_ms else "-"
            lines.append(f"| {r['objects']} | {r['mean_ms']:.2f} | {r['fps']:.2f} | "
                         f"{r['p99_ms']:.2f} | {ratio} |")
        lines += [
            "",
            "Objects are the batch dimension, so this one is expected to rise. "
            "Sub-linear growth means the batch is filling otherwise idle GPU "
            "width; linear means it is already saturated.",
        ]

    markdown = "\n".join(lines) + "\n"
    print("\n" + markdown)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".json").write_text(json.dumps(results, indent=2) + "\n")
        out.with_suffix(".md").write_text(markdown)
        print(f">> wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
