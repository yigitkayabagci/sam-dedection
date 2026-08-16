#!/usr/bin/env python3
"""Track Anti-UAV410 sequences and score them against the ground truth.

This is the one measurement every later decision is made against: did the
fine-tune help, did INT8 cost anything, did SAMURAI recover the target. Running
it through the same `VideoTracker` interface as `cli.py` means the thing being
scored is the thing that will be deployed -- backend, precision, engines and
all -- rather than a reimplementation of it.

Three numbers come back per sequence:

    state accuracy   the Anti-UAV metric: overlap while the target is visible,
                     credit for correctly reporting that it is gone.
    success AUC      average overlap over visible frames, for comparison with
                     published Anti-UAV410 results.
    dropouts         how often a visible target was lost, and for how long.
                     A mean hides the difference between one 60-frame loss and
                     sixty 1-frame ones, and those are different bugs.

Two input modes, matching what the deployment offers (`tools/run_records.py`):

    full   the whole 640x512 frame resized to the model input
    crop   a fixed 512x512 window -- native pixels, no resize at all

`crop` transcodes into a temporary directory because EdgeTAM's `init_state`
only reads a folder of JPEGs; `full` points straight at the sequence and costs
nothing.

Usage:
    python tools/eval_antiuav.py --data data/anti-uav410 --split val \\
        --tracker edgetam_trt --config configs/edgetam_trt_512.yaml --limit 20
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
from src.training.antiuav import crop_window, frame_shape, list_sequences  # noqa: E402


def _prepare_crop(sequence, size: int, cache: Path) -> tuple[Path, np.ndarray]:
    """Transcode a fixed `size` window of every frame, and shift the labels.

    The window is placed once, on the whole sequence, so that the coordinate
    frame the memory bank stores features in never moves. Sequences whose
    target leaves that window are exactly the failure this mode has -- they are
    scored, not skipped, because losing the target is the cost being measured.
    """
    import cv2

    height, width = frame_shape(sequence.frames[0])
    origin = crop_window(sequence.labels.boxes, (width, height), size)
    if origin is None:
        # No fixed window holds the whole track. Centre it on the first frame
        # and let the score record what that costs.
        first = sequence.labels.boxes[sequence.labels.visible_indices()[0]]
        origin = crop_window(first[None], (width, height), size)
    x0, y0 = origin

    cache.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(sequence.frames):
        img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(str(cache / f"{i}.jpg"), img[y0:y0 + size, x0:x0 + size],
                    [cv2.IMWRITE_JPEG_QUALITY, 100])

    boxes = sequence.labels.boxes.copy()
    boxes[:, 0::2] -= x0
    boxes[:, 1::2] -= y0
    return cache, boxes


def track_sequence(tracker, sequence, mode: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    """`(predicted boxes, ground-truth boxes)`, both NaN where absent.

    A predicted box is the tight box around the mask. When EdgeTAM's object
    score goes negative the mask comes back empty, which becomes NaN -- so
    "the tracker says the target is gone" is expressed the same way the labels
    express it, and the metric can compare them directly.
    """
    cache = None
    try:
        if mode == "crop":
            cache = Path(tempfile.mkdtemp(prefix="antiuav_"))
            frames_dir, gt = _prepare_crop(sequence, size, cache)
        else:
            frames_dir, gt = sequence.frames[0].parent, sequence.labels.boxes

        visible = sequence.labels.visible_indices()
        if visible.size == 0:
            raise ValueError(f"{sequence.name}: no annotated frame to prompt from.")
        start = int(visible[0])

        tracker.prepare(frames_dir)
        tracker.set_prompts(PromptSet(
            boxes=[BoxPrompt(obj_id=1, frame_idx=start,
                             xyxy=tuple(float(v) for v in gt[start]))]
        ))

        pred = np.full_like(gt, np.nan)
        for result in tracker.propagate():
            mask = result.masks.get(1)
            if mask is not None and result.frame_idx < len(pred):
                pred[result.frame_idx] = box_from_mask(mask)

        # Frames before the prompt were never tracked; scoring them as misses
        # would penalise the backend for something it was never asked to do.
        return pred[start:], gt[start:]
    finally:
        tracker.reset()
        if cache is not None:
            shutil.rmtree(cache, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", required=True, help="Anti-UAV410 root (holds train/ val/ test/).")
    p.add_argument("--split", default="val")
    p.add_argument("--tracker", default="edgetam")
    p.add_argument("--config", default=None, help="Backend YAML (default configs/<tracker>.yaml).")
    p.add_argument("--mode", choices=("full", "crop"), default="crop",
                   help="full = whole frame resized; crop = fixed 512 window, native pixels.")
    p.add_argument("--size", type=int, default=512, help="Model input / crop size.")
    p.add_argument("--limit", type=int, default=None, help="Score only the first N sequences.")
    p.add_argument("--json", default=None, help="Also write per-sequence scores here.")
    args = p.parse_args(argv)

    cfg_path = Path(args.config) if args.config else ROOT / f"configs/{args.tracker}.yaml"
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    for key in ("checkpoint", "image_encoder_engine", "memory_attention_engine",
                "memory_encoder_engine", "sam_head_engine", "engine_path"):
        if cfg.get(key):
            cfg[key] = str(ROOT / cfg[key]) if not Path(cfg[key]).is_absolute() else cfg[key]
    tracker = build_tracker(args.tracker, **cfg)

    sequences = list_sequences(args.data, args.split)[:args.limit]
    print(f"{len(sequences)} sequence(s), {args.tracker} @ {cfg_path.name}, mode={args.mode}\n")

    scores, failed = [], []
    for sequence in sequences:
        try:
            pred, gt = track_sequence(tracker, sequence, args.mode, args.size)
        except Exception as exc:  # one bad sequence must not lose the whole run
            print(f"!! {sequence.name}: {exc}")
            failed.append(sequence.name)
            continue
        score = score_sequence(sequence.name, pred, gt)
        scores.append(score)
        print(f"{sequence.name:<28} SA {score.state_accuracy:.4f}  "
              f"AUC {score.success_auc:.4f}  {score.dropouts.summary()}")

    print(f"\n{report(scores)}")
    if failed:
        print(f"\n**Did not complete: {', '.join(failed)}**")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "tracker": args.tracker, "config": str(cfg_path), "mode": args.mode,
            "sequences": [
                {"name": s.name, "frames": s.frames,
                 "state_accuracy": s.state_accuracy, "success_auc": s.success_auc,
                 "dropout_lengths": list(s.dropouts.lengths)}
                for s in scores
            ],
            "failed": failed,
        }, indent=2) + "\n")
        print(f"\nwrote {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
