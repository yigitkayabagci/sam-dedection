#!/usr/bin/env python3
"""Track one clip twice, two backends, and compare the masks frame by frame.

This is the accuracy half of the experiment. `check_trt_parity.py` answers
"does each engine reproduce its module?" one module at a time, with a fresh
random input each call. That is necessary but not sufficient: EdgeTAM feeds its
own output back through the memory bank, so a per-module error that looks
negligible in isolation can accumulate over a few hundred frames. The only way
to rule that out is to run the whole loop and compare the masks it produces.

Mask IoU is the metric because it is the thing you ship. A logit that moves by
1e-3 does not matter; a logit that crosses zero does, and IoU counts exactly
those pixels.

The reference backend runs first and its masks are kept bit-packed (1 bit per
pixel), so the two models never sit on the GPU at the same time and a 500-frame
720p clip costs ~57 MB of host RAM per object instead of ~460 MB.

Usage on the Orin:
    # the headline number: does TensorRT change the masks?
    python tools/compare_backends.py --frames 500

    # strictest version -- fp32 PyTorch as ground truth
    python tools/compare_backends.py --frames 500 --reference-precision float32

    # on your own footage
    python tools/compare_backends.py --frames-dir data/frames --prompt-file p.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cli import _load_backend_config, _resolve_paths  # noqa: E402
from src.prompts.file_source import load_prompts  # noqa: E402
from src.trackers import build_tracker  # noqa: E402
from tools.benchmark_tracking import RESOLVE_KEYS, write_synthetic_clip  # noqa: E402

ENGINE_MODULES = ("image_encoder", "memory_attention", "memory_encoder", "sam_head")


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boolean masks.

    Two empty masks are identical, so they score 1.0 -- an occluded object that
    both backends agree is absent is agreement, not a failure. Scoring it 0
    would bury a genuinely good run under the frames where the target is hidden.
    """
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return 1.0 if union == 0 else intersection / union


def run_backend(
    name: str,
    config: str | None,
    frames_dir: Path,
    prompts,
    overrides: dict,
    require_engines: bool = False,
):
    """Yield (frame_idx, {obj_id: mask}) for a full propagate pass."""
    cfg = _resolve_paths(_load_backend_config(name, config), RESOLVE_KEYS)
    cfg.update(overrides)
    label = f"{name} ({cfg.get('precision', '?')})"
    print(f">> {label}")
    tracker = build_tracker(name, **cfg)
    tracker.prepare(frames_dir)
    tracker.set_prompts(prompts)

    # A backend that quietly fell back to PyTorch would score a perfect 1.0000
    # against PyTorch and prove nothing at all. Refuse to report that as a
    # result -- it is the one failure mode of this test that looks like a pass.
    if require_engines:
        loaded = sorted(getattr(tracker, "_engines", {}) or {})
        missing = [m for m in ENGINE_MODULES if m not in loaded]
        if missing:
            tracker.reset()
            raise SystemExit(
                f"{name} is running {', '.join(missing)} on PyTorch, not TensorRT.\n"
                "Comparing that against PyTorch would trivially score 1.0000 and\n"
                "tell you nothing. Build the missing engines first:\n"
                f"  python tools/build_trt_engines.py --outdir models/ "
                f"--module {missing[0]}\n"
                "Pass --allow-partial if you are deliberately testing a subset."
            )
    try:
        for result in tracker.propagate():
            yield result.frame_idx, result.masks
    finally:
        tracker.reset()
        del tracker
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def collect_packed(name, config, frames_dir, prompts, overrides, require_engines=False):
    """Run a backend and keep its masks bit-packed, one entry per frame."""
    packed = []
    for frame_idx, masks in run_backend(
        name, config, frames_dir, prompts, overrides, require_engines
    ):
        packed.append(
            (
                frame_idx,
                {oid: (np.packbits(m), m.shape) for oid, m in masks.items()},
            )
        )
    return packed


def unpack(entry):
    bits, shape = entry
    return np.unpackbits(bits, count=int(np.prod(shape))).astype(bool).reshape(shape)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--frames", type=int, default=500)
    p.add_argument("--size", default="1280x720")
    p.add_argument("--objects", type=int, default=1)
    p.add_argument(
        "--frames-dir",
        default=None,
        help="Track real footage instead of a synthetic clip. Needs --prompt-file.",
    )
    p.add_argument("--prompt-file", default=None, help="JSON prompts for --frames-dir.")
    p.add_argument("--reference", default="edgetam", help="Backend treated as correct.")
    p.add_argument("--reference-config", default=None)
    p.add_argument(
        "--reference-precision",
        default=None,
        help="Override the reference's precision. float32 is the strictest "
        "ground truth; leaving it alone compares against the fp16 PyTorch path "
        "you were already running.",
    )
    p.add_argument("--candidate", default="edgetam_trt", help="Backend under test.")
    p.add_argument("--candidate-config", default="configs/edgetam_trt.yaml")
    p.add_argument(
        "--warn-below",
        type=float,
        default=0.99,
        help="Report every frame whose IoU falls under this (default 0.99).",
    )
    p.add_argument(
        "--fail-below",
        type=float,
        default=0.0,
        help="Exit non-zero if mean IoU drops under this. 0 disables (default).",
    )
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Let the candidate run some modules on PyTorch. Off by default: a "
        "full fallback would score a meaningless 1.0000 against PyTorch.",
    )
    p.add_argument("--json", default=None, help="Write the per-frame IoUs here.")
    p.add_argument("--offload-video", action="store_true")
    p.add_argument("--keep-frames", action="store_true")
    args = p.parse_args(argv)

    overrides = {"offload_video_to_cpu": True} if args.offload_video else {}

    tmp = None
    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
        if not args.prompt_file:
            raise SystemExit("--frames-dir needs --prompt-file.")
        prompts = load_prompts(args.prompt_file)
    else:
        width, height = (int(v) for v in args.size.lower().split("x"))
        tmp = Path(tempfile.mkdtemp(prefix="compare_backends_"))
        frames_dir = tmp / "frames"
        print(f">> Writing {args.frames} synthetic frames at {width}x{height}")
        prompts = write_synthetic_clip(frames_dir, args.frames, width, height, args.objects)

    try:
        ref_overrides = dict(overrides)
        if args.reference_precision:
            ref_overrides["precision"] = args.reference_precision
        reference = collect_packed(
            args.reference, args.reference_config, frames_dir, prompts, ref_overrides
        )

        per_frame = []
        candidate_frames = 0
        for frame_idx, masks in run_backend(
            args.candidate,
            args.candidate_config,
            frames_dir,
            prompts,
            overrides,
            require_engines=not args.allow_partial,
        ):
            if candidate_frames >= len(reference):
                break
            ref_idx, ref_masks = reference[candidate_frames]
            candidate_frames += 1
            if ref_idx != frame_idx:
                raise SystemExit(
                    f"Backends disagree on frame order ({ref_idx} vs {frame_idx})."
                )
            if set(ref_masks) != set(masks):
                raise SystemExit(
                    f"frame {frame_idx}: object ids differ "
                    f"({sorted(ref_masks)} vs {sorted(masks)})."
                )
            for obj_id in sorted(masks):
                per_frame.append(
                    {
                        "frame": frame_idx,
                        "obj_id": obj_id,
                        "iou": iou(unpack(ref_masks[obj_id]), masks[obj_id]),
                    }
                )
    finally:
        if tmp is not None and not args.keep_frames:
            shutil.rmtree(tmp, ignore_errors=True)

    if not per_frame:
        raise SystemExit("No frames compared.")
    if candidate_frames != len(reference):
        print(
            f"   WARN: compared {candidate_frames} of {len(reference)} reference frames"
        )

    scores = np.array([r["iou"] for r in per_frame])
    below = [r for r in per_frame if r["iou"] < args.warn_below]
    worst = min(per_frame, key=lambda r: r["iou"])

    print(f"\n{'=' * 62}")
    print(f"mask IoU: {args.candidate} vs {args.reference}")
    print("=" * 62)
    print(f"  comparisons        {len(scores)} ({candidate_frames} frames x "
          f"{len(scores) // max(candidate_frames, 1)} objects)")
    print(f"  mean IoU           {scores.mean():.4f}")
    print(f"  min  IoU           {scores.min():.4f}  (frame {worst['frame']}, "
          f"obj {worst['obj_id']})")
    print(f"  p01  IoU           {np.percentile(scores, 1):.4f}")
    print(f"  below {args.warn_below:.2f}         {len(below)} of {len(scores)}")
    for record in below[:10]:
        print(f"      frame {record['frame']:>5} obj {record['obj_id']}  "
              f"IoU {record['iou']:.4f}")
    if len(below) > 10:
        print(f"      ... and {len(below) - 10} more")
    print(
        "\n  A mean at 1.0000 with nothing below the threshold means TensorRT\n"
        "  reproduced the reference masks pixel for pixel over the whole clip,\n"
        "  memory feedback included."
    )

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "reference": args.reference,
                    "reference_precision": args.reference_precision,
                    "candidate": args.candidate,
                    "frames": candidate_frames,
                    "mean_iou": float(scores.mean()),
                    "min_iou": float(scores.min()),
                    "below_threshold": len(below),
                    "threshold": args.warn_below,
                    "per_frame": per_frame,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\n  wrote {out}")

    if args.fail_below and scores.mean() < args.fail_below:
        raise SystemExit(
            f"mean IoU {scores.mean():.4f} is below --fail-below {args.fail_below}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
