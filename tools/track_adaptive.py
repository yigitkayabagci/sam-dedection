#!/usr/bin/env python3
"""Track with a window that follows the target and sizes itself to it.

The Orin runs a 512x512 frame in about 10 ms and roughly 20-30 ms is
affordable. On Anti-UAV410 most targets are a handful of pixels, so the most
valuable thing to spend that margin on is not a bigger model -- it is showing
the model a crop in which the target is not tiny.

Segments, not per-frame adjustment
----------------------------------
SAM 2 stores memory features in *input* coordinates. Move the crop and every
stored memory is in a different frame of reference from the features reading
it. So the window is treated as a **segment boundary**: it holds still (a size
ladder, a position grid, and a margin before either is touched), and when it
finally has to move, tracking restarts at the new window with a fresh memory
bank, re-prompted from the last mask.

That is why this is a driver rather than a patch inside the tracker: a segment
is exactly one `prepare` / `set_prompts` / `propagate` cycle, so nothing about
the `VideoTracker` interface has to change, and the coordinate change is
explicit instead of silently approximate.

The cost is one re-prompt per boundary. With hysteresis those are rare -- if
they are not, the printed reasons say which knob is too tight.

Usage:
    python tools/track_adaptive.py --data data/anti-uav410 --split val \\
        --tracker edgetam --config configs/edgetam_samurai_512.yaml --limit 10
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.accuracy import box_from_mask, report, score_sequence  # noqa: E402
from src.prompts import BoxPrompt, PromptSet  # noqa: E402
from src.trackers import build_tracker  # noqa: E402
from src.trackers.adaptive import AdaptiveConfig, AdaptiveWindow  # noqa: E402
from src.training.antiuav import frame_shape, list_sequences  # noqa: E402


def _write_segment(frames, rect, size, cache: Path) -> Path:
    """Crop and resize one segment's frames into a directory the tracker reads.

    EdgeTAM's `init_state` only loads a folder of JPEGs, so a segment has to be
    materialised. That is real work -- and it is the honest cost of this
    approach, which is why the segment count is reported alongside the scores.
    """
    import cv2

    x0, y0, side = rect
    if cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True)
    for index, path in enumerate(frames):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        crop = img[y0:y0 + side, x0:x0 + side]
        if crop.shape[:2] != (size, size):
            shrinking = size < crop.shape[0]
            crop = cv2.resize(crop, (size, size),
                              interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)
        cv2.imwrite(str(cache / f"{index}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return cache


def track_adaptive(tracker, sequence, cfg: AdaptiveConfig, cache: Path,
                   segment_max: int = 400):
    """`(predicted boxes, ground truth, segment reasons)`, in frame coordinates.

    `segment_max` bounds how many frames are materialised at a time. Running
    out of them forces a re-anchor that the window policy did not ask for, so
    those are recorded as `chunk` and reported separately -- they are an
    artefact of this offline harness, not of the method. A deployment gets its
    frames one at a time and crops each on arrival for nothing.
    """
    height, width = frame_shape(sequence.frames[0])
    window = AdaptiveWindow((width, height), cfg)
    gt = sequence.labels.boxes
    visible = sequence.labels.visible_indices()
    if visible.size == 0:
        raise ValueError(f"{sequence.name}: no annotated frame to prompt from.")

    start = int(visible[0])
    pred = np.full_like(gt, np.nan)
    anchor = gt[start]
    window.move_to(anchor)
    reasons: list[str] = []
    frame = start

    while frame < len(sequence.frames):
        chunk = sequence.frames[frame:frame + segment_max]
        segment = _write_segment(chunk, window.rect, cfg.size, cache)
        tracker.prepare(segment)
        tracker.set_prompts(PromptSet(boxes=[BoxPrompt(
            obj_id=1, frame_idx=0,
            xyxy=tuple(float(v) for v in window.to_window(anchor)))]))

        stop_at = None
        for result in tracker.propagate():
            index = frame + result.frame_idx
            if index >= len(pred):
                break
            mask = result.masks.get(1)
            local = box_from_mask(mask) if mask is not None else np.full(4, np.nan)
            if not np.isnan(local).any():
                pred[index] = window.to_frame(local)
                anchor = pred[index]
            # The window may only move at a segment boundary, so a decision to
            # move ends this segment rather than taking effect inside it.
            reason = window.needs_move(pred[index])
            if reason is not None and index > frame:
                reasons.append(f"{index}:{reason}")
                stop_at = index
                break
        tracker.reset()

        if stop_at is not None:
            window.move_to(anchor)
            frame = stop_at
            continue
        frame += len(chunk)
        if frame < len(sequence.frames):
            reasons.append(f"{frame}:chunk")

    return pred[start:], gt[start:], reasons


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", required=True, help="Anti-UAV410 root.")
    p.add_argument("--split", default="val")
    p.add_argument("--tracker", default="edgetam")
    p.add_argument("--config", default="configs/edgetam_512.yaml")
    p.add_argument("--size", type=int, default=512, help="Model input size.")
    p.add_argument("--target-fraction", type=float, default=0.08,
                   help="Share of the window the target should occupy, by its "
                        "longer side. Smaller = more zoom = more segments.")
    p.add_argument("--min-window", type=int, default=128)
    p.add_argument("--scale-step", type=float, default=1.5,
                   help="Ratio between window sizes. Larger = fewer resizes.")
    p.add_argument("--margin", type=float, default=0.2,
                   help="How close to an edge the target may come, as a "
                        "fraction of the window.")
    p.add_argument("--segment-max", type=int, default=400,
                   help="Frames materialised per segment. Running out forces a "
                        "re-anchor the window policy did not ask for; those are "
                        "reported as `chunk` and are a harness artefact.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)

    cfg = AdaptiveConfig(size=args.size, target_fraction=args.target_fraction,
                         min_window=args.min_window, scale_step=args.scale_step,
                         margin=args.margin)
    backend = yaml.safe_load(Path(args.config).read_text())
    for key in ("checkpoint", "lora_adapter", "image_encoder_engine",
                "memory_attention_engine", "memory_encoder_engine",
                "sam_head_engine", "engine_path"):
        if backend.get(key) and not Path(backend[key]).is_absolute():
            backend[key] = str(ROOT / backend[key])
    tracker = build_tracker(args.tracker, **backend)

    sequences = list_sequences(args.data, args.split)[:args.limit]
    cache = Path(tempfile.mkdtemp(prefix="adaptive_"))
    scores, segments = [], {}
    try:
        for sequence in sequences:
            pred, gt, reasons = track_adaptive(tracker, sequence, cfg, cache / "seg",
                                               args.segment_max)
            score = score_sequence(sequence.name, pred, gt)
            scores.append(score)
            segments[sequence.name] = reasons
            print(f"{sequence.name:<28} SA {score.state_accuracy:.4f}  "
                  f"{len(reasons) + 1:>3} segment(s)  {score.dropouts.summary()}")
    finally:
        shutil.rmtree(cache, ignore_errors=True)

    print(f"\n{report(scores)}")
    moves = [r.split(":")[1] for rs in segments.values() for r in rs]
    if moves:
        frames = sum(s.frames for s in scores)
        real = moves.count("edge") + moves.count("size")
        print(f"\nwindow moved {real} time(s) in {frames} frames "
              f"({moves.count('edge')} on `edge`, {moves.count('size')} on `size`; "
              f"{moves.count('chunk')} forced by --segment-max, a harness artefact). "
              "Each real move costs a memory re-anchor -- more than a few per "
              "hundred frames means --margin or --scale-step is too tight.")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "config": vars(args),
            "sequences": [
                {"name": s.name, "frames": s.frames,
                 "state_accuracy": s.state_accuracy, "success_auc": s.success_auc,
                 "dropout_lengths": list(s.dropouts.lengths),
                 "segments": segments[s.name]}
                for s in scores
            ],
        }, indent=2) + "\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
