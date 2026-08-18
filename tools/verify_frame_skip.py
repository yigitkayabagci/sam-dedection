#!/usr/bin/env python3
"""Prove from the output video alone that frame skipping did what it claims.

A skipped run and an unskipped one produce videos that look the same, which is
the point of the technique and also why it is hard to believe: nothing on
screen says the model ran half as often. `run.txt` says so, but a number in a
log is not evidence you can hand someone -- so this reads it back out of the
mp4 itself.

**How.** The overlay is the only saturated colour in a thermal (grayscale)
frame, so it can be separated on colour and reduced to its centroid. The
centroid, not the box corners: corners are a min/max over the noisiest pixels
in the frame and a single stray pixel moves them, while a centroid averages
over the whole mask and barely notices mp4 compression.

**The test.** Under `--frame-skip N` the overlay must be *frozen* inside each
group of N frames and jump between groups, because frames 1..N-1 of a group
carry frame 0's mask, held. So compare two displacements in the same video:

    within  = centroid movement inside a group   -> must be ~0 under a skip
    across  = centroid movement between groups   -> the target's real motion

`across / within` is the signal, and it is self-normalising: whatever the
compression noise is, it is in both numbers. An unskipped run has no group
structure, so its two numbers are the same and the ratio is ~1.

**When it cannot answer.** If the target does not move, everything is frozen
in both videos and no measurement can tell a held mask from a fresh one. That
is not a failure of the pipeline, it is a clip with no information in it -- so
this reports the motion it found and says INCONCLUSIVE rather than guessing.

Usage:
    python tools/verify_frame_skip.py frame_output/record_024/full1024_skip2/tracked.mp4 --stride 2
    # with the unskipped run for contrast (recommended)
    python tools/verify_frame_skip.py .../full1024_skip2/tracked.mp4 --stride 2 \
        --baseline frame_output/record_024/full1024/tracked.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def overlay_centroids(video: Path, saturation: int = 40) -> list[tuple[float, float] | None]:
    """Centre of the drawn overlay in each frame, or None where there is none.

    `saturation` is how far a pixel's channels must disagree to be overlay
    rather than source. A grayscale source has zero spread, so anything above
    the compression's own colour noise works; raise it on a colour source.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"could not open {video}")
    out: list[tuple[float, float] | None] = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            img = bgr.astype(np.int16)
            sel = (img.max(axis=2) - img.min(axis=2)) > saturation
            if sel.sum() < 4:
                out.append(None)
                continue
            ys, xs = np.nonzero(sel)
            out.append((float(xs.mean()), float(ys.mean())))
    finally:
        cap.release()
    return out


def _step(a, b) -> float | None:
    if a is None or b is None:
        return None
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def group_motion(centroids: list, stride: int) -> tuple[list[float], list[float]]:
    """(movement inside a group, movement between groups), in pixels.

    Both are per one frame of separation, so they are directly comparable: the
    across-group step spans `stride` frames and is divided by it.
    """
    within, across = [], []
    for idx in range(len(centroids) - 1):
        step = _step(centroids[idx], centroids[idx + 1])
        if step is None:
            continue
        # A group boundary sits where the next frame starts a new group.
        if (idx + 1) % stride == 0:
            continue
        within.append(step)
    for start in range(0, len(centroids) - stride, stride):
        step = _step(centroids[start], centroids[start + stride])
        if step is not None:
            across.append(step / stride)
    return within, across


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def describe(name: str, video: Path, stride: int, saturation: int) -> tuple[float, float]:
    centroids = overlay_centroids(video, saturation)
    found = sum(c is not None for c in centroids)
    within, across = group_motion(centroids, stride)
    w, a = median(within), median(across)
    print(f"{name}: {len(centroids)} frames, overlay found in {found}")
    print(f"   inside a group of {stride}: {w:.2f} px/frame     "
          f"between groups: {a:.2f} px/frame")
    return w, a


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path, help="tracked.mp4 from the skipped run.")
    p.add_argument("--stride", type=int, default=2, help="The N passed to --frame-skip.")
    p.add_argument("--baseline", type=Path, default=None,
                   help="tracked.mp4 from the unskipped run of the same clip.")
    p.add_argument("--saturation", type=int, default=40,
                   help="How far a pixel's channels must disagree to count as overlay "
                        "rather than grayscale source (default 40).")
    p.add_argument("--noise", type=float, default=0.30,
                   help="Centroid movement below this many pixels per frame counts as "
                        "frozen -- mp4 compression alone moves it by well under a "
                        "tenth of a pixel once averaged over a mask (default 0.30).")
    args = p.parse_args(argv)

    if args.stride < 2:
        raise SystemExit("--stride must be 2 or more; there is nothing to verify at 1.")

    within, across = describe("skipped ", args.video, args.stride, args.saturation)
    if args.baseline is not None:
        print()
        describe("baseline", args.baseline, args.stride, args.saturation)
    print()

    if not np.isfinite(within) or not np.isfinite(across):
        print("NO OVERLAY: nothing coloured found in the video. If the source is not "
              "grayscale, raise --saturation.")
        return 1
    if across < args.noise:
        print(f"INCONCLUSIVE: the target moves {across:.2f} px/frame in this clip, which "
              f"is below the {args.noise:.2f} px noise floor. A held mask and a fresh "
              "one are identical when nothing moves, so no measurement on this clip can "
              "tell them apart. Re-run the check on a record where the target moves.")
        return 1
    if within < args.noise:
        # A ratio against a within-group step of zero is a number with no
        # meaning; say "frozen" where that is what was measured.
        contrast = f", {across / within:.0f}x apart" if within > 0.01 else ""
        print(f"CONFIRMED: the overlay is frozen inside a group of {args.stride} "
              f"({within:.2f} px/frame) and moves {across:.2f} px/frame between "
              f"groups{contrast}. That is the mask being held over the frames the "
              "model never saw, read off the video rather than the log.")
        return 0
    print(f"NOT HOLDING: the overlay moves {within:.2f} px/frame inside a group, which "
          f"is not frozen -- against {across:.2f} px/frame between groups. Either this "
          "run did not use --frame-skip, or this is not the skipped video. Check "
          "`grep 'frame skip' run.txt` in the same folder.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
