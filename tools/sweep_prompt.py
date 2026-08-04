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
    # Full grid: the two effects are expected to be independent (size changes
    # nothing, count scales with the batch), and a grid is what shows whether
    # that independence actually holds rather than assuming it.
    plan = [(r, n) for r in radii for n in objects]
    print(f">> {len(plan)} runs: {len(radii)} target sizes x {len(objects)} object counts")

    for radius, count in plan:
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
        stats.update(radius=radius, objects=count)
        results.append(stats)
        print(f"   {stats['mean_ms']:7.2f} ms/frame   {stats['fps']:6.2f} FPS   "
              f"p99 {stats['p99_ms']:.2f} ms")

    if not results:
        raise SystemExit("Nothing measured.")

    grid = {(r["radius"], r["objects"]): r for r in results}
    measured = min(r["frames"] for r in results) - args.warmup

    lines = [
        f"# Prompt sweep — {args.tracker}",
        "",
        f"{args.frames} frames at {args.size}, {args.warmup} warm-up excluded, "
        f"{measured} measured per point, config `{args.config}`",
        "",
        "## ms per frame",
        "",
        "| target radius \\ objects | " + " | ".join(str(n) for n in objects) + " |",
        "|---" * (len(objects) + 1) + "|",
    ]
    for r in radii:
        cells = []
        for n in objects:
            cell = grid.get((r, n))
            cells.append(f"{cell['mean_ms']:.2f}" if cell else "-")
        lines.append(f"| **{r} px** | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## p99 ms per frame",
        "",
        "| target radius \\ objects | " + " | ".join(str(n) for n in objects) + " |",
        "|---" * (len(objects) + 1) + "|",
    ]
    for r in radii:
        cells = []
        for n in objects:
            cell = grid.get((r, n))
            cells.append(f"{cell['p99_ms']:.2f}" if cell else "-")
        lines.append(f"| **{r} px** | " + " | ".join(cells) + " |")

    # --- what the grid says --------------------------------------------
    lines += ["", "## Reading", ""]

    # Target size: spread down each column, worst case across columns.
    spreads = []
    for n in objects:
        col = [grid[(r, n)]["mean_ms"] for r in radii if (r, n) in grid]
        if len(col) > 1:
            spreads.append((max(col) - min(col)) / min(col) * 100.0)
    if spreads:
        worst = max(spreads)
        lines.append(
            f"**Target size: {worst:.1f}% spread** (worst column). "
            + ("Flat, as expected — the model resizes every frame to a fixed "
               "input, so how large the target appears on screen never reaches "
               "it. A 12 px blob and a 120 px blob are identical work."
               if worst < 10 else
               "Not flat. Either something in the per-frame path depends on how "
               "much of the frame the mask covers, or the run-to-run noise floor "
               "is this wide — re-run with more frames and pinned clocks "
               "(`sudo jetson_clocks`) before concluding the former.")
        )

    # Object count: scaling along each row, relative to the smallest count.
    base_n = objects[0]
    ratios = []
    for r in radii:
        base = grid.get((r, base_n))
        top = grid.get((r, objects[-1]))
        if base and top and base["mean_ms"] > 0:
            ratios.append(top["mean_ms"] / base["mean_ms"])
    if ratios and len(objects) > 1:
        mean_ratio = sum(ratios) / len(ratios)
        span = objects[-1] / base_n
        lines += [
            "",
            f"**Object count: {mean_ratio:.2f}× going from {base_n} to "
            f"{objects[-1]} objects** (averaged over target sizes), against "
            f"{span:.0f}× the work. "
            + ("Sub-linear: the extra objects are filling GPU width that one "
               "object left idle, so they are close to free."
               if mean_ratio < span * 0.7 else
               "Close to linear: the batch is already saturating the GPU, so "
               "each additional object costs about what the first one did."),
            "",
            "Objects are the batch dimension — a second target is a second row "
            "through the memory attention and the mask decoder. Engines must "
            "have been built with `--max-batch` at least this high, or the "
            "tracker falls back to PyTorch and the timing means something else.",
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
