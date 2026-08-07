#!/usr/bin/env python3
"""Render what dropping stale frames looks like, next to what buffering looks like.

Both policies run the same tracker at the same speed on the same clip. The only
difference is which frame the tracker is handed when it comes free:

    buffered     the next frame in line -- an unbounded queue, nothing dropped
    drop-stale   whichever frame is current -- a one-slot queue (src/realtime.py)

The output is one side-by-side mp4 at the source's own frame rate, so each
instant of playback shows both answers to the same question: where is the
target *now*. Buffering answers it later and later; dropping answers it in
jumps but never late. The dashed circle is where the target actually is.

This is a picture of the delivery policy, not a measurement of EdgeTAM: the
"tracker" here is exact by construction and its cost is a fixed `--work-ms`.
Numbers come from `cli.py --realtime-fps` on real weights; this is for seeing
the mechanism (and for the thesis figure that explains it).

Usage:
    python tools/demo_frame_skip.py --out outputs/frame_skip_demo.mp4
    python tools/demo_frame_skip.py --frames 150 --fps 30 --work-ms 80
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import open_video_writer  # noqa: E402
from src.visualize import overlay_masks  # noqa: E402
from tools.benchmark_tracking import write_synthetic_clip  # noqa: E402


def blob_mask(rgb: np.ndarray) -> np.ndarray:
    """The target's pixels, exactly. A perfect tracker, so the only thing the
    video can be showing is the delivery policy.

    By saturation rather than by colour: the clip draws its target fully
    saturated on a desaturated grey background (median 12 against the blob's
    ~190), and thresholding there works whichever palette entry the generator
    picked. The dilate closes the white ring the generator draws inside the
    blob, which is unsaturated and would otherwise punch a hole in the mask.
    """
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
    solid = cv2.dilate((sat > 120).astype(np.uint8), np.ones((5, 5), np.uint8))
    return cv2.erode(solid, np.ones((5, 5), np.uint8)).astype(bool)


def completed_frame_at(now_s: float, work_s: float, period_s: float, policy: str) -> int | None:
    """Which frame's masks are on screen at `now_s`, under `policy`.

    Both policies run the same tracker: it is busy for `work_s` per frame and
    picks its next frame the moment it finishes. What differs is the pick.

    buffered -- frames queue up untouched, so the tracker walks them in order
    and finishes frame k at `(k + 1) * work_s`. The newest finished frame is
    therefore `now_s / work_s - 1`, and since the clip has moved on to
    `now_s / period_s` by then, the gap between the two grows for as long as
    the run lasts. That is the failure mode: not wrong masks, late ones.

    drop-stale -- the queue holds one frame and a new arrival replaces it, so
    the tracker always picks up whatever is current. Starting at frame 0, each
    pick is the frame that had just arrived when the previous one finished.
    """
    if policy == "buffered":
        done = int(now_s / work_s) - 1
        return done if done >= 0 else None
    shown, finishes_at = None, 0.0
    while finishes_at <= now_s:
        shown = int(finishes_at / period_s)  # the frame current at pick-up time
        finishes_at += work_s
    return shown


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", default="outputs/frame_skip_demo.mp4")
    p.add_argument("--frames", type=int, default=150, help="Source frames to render.")
    p.add_argument("--fps", type=float, default=30.0, help="Source frame rate.")
    p.add_argument("--work-ms", type=float, default=80.0,
                   help="What one frame costs the tracker. Above 1000/fps it "
                        "cannot keep up, which is the case worth watching.")
    p.add_argument("--size", default="640x360", help="WxH.")
    args = p.parse_args(argv)

    width, height = (int(v) for v in args.size.lower().split("x"))
    period_s, work_s = 1.0 / args.fps, args.work_ms / 1000.0
    if work_s <= period_s:
        print(f"[demo] {args.work_ms:.0f} ms/frame keeps up with {args.fps:g} FPS "
              f"({1000 / args.fps:.1f} ms/frame) -- nothing will be dropped and "
              "both panels will look the same. Raise --work-ms to see the split.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp)
        write_synthetic_clip(clip, args.frames, width, height, objects=1)
        frames = [cv2.cvtColor(cv2.imread(str(clip / f"{t:05d}.jpg")), cv2.COLOR_BGR2RGB)
                  for t in range(args.frames)]
        masks = [blob_mask(f) for f in frames]

        lag = {"buffered": [], "drop-stale": []}
        with open_video_writer(out_path, args.fps, (width * 2, height)) as emit:
            for t, frame in enumerate(frames):
                panels = []
                for policy in ("buffered", "drop-stale"):
                    shown = completed_frame_at(t * period_s, work_s, period_s, policy)
                    panel = frame.copy()
                    behind = 0
                    if shown is not None:
                        behind = t - shown
                        lag[policy].append(behind)
                        # obj_id 2 is the palette's green: the target itself is
                        # blue, and a mask the same colour as the thing it is
                        # supposed to be tracking would hide the whole point.
                        panel = overlay_masks(panel, {2: masks[shown]}, alpha=0.55)
                    # No separate truth marker needed -- the blob is the truth,
                    # and the gap between it and the green box is the lag.
                    cv2.putText(panel, f"{policy}   {behind} frames behind",
                                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    panels.append(panel)
                emit(np.hstack(panels))

    print(f"[demo] {args.frames} frames at {args.fps:g} FPS, tracker at "
          f"{args.work_ms:.0f} ms/frame ({1000 / args.work_ms:.1f} FPS)")
    for policy, values in lag.items():
        print(f"[demo]   {policy:11s} lag: median {sorted(values)[len(values) // 2]:3d} "
              f"· worst {max(values):3d} frames")
    print(f"[demo] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
