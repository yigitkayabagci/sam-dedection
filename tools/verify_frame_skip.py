#!/usr/bin/env python3
"""Prove from the output video alone that frame skipping did what it claims.

A skipped run and an unskipped one produce videos that look the same, which is
the point of the technique and also why it is hard to believe: nothing on
screen says the model ran half as often. `run.txt` says so, but a number in a
log is not evidence you can hand someone -- so this reads the evidence back out
of the mp4 itself.

The overlay is the only saturated colour in a thermal (grayscale) frame, so the
tracked box can be recovered from any frame by separating it on colour. Under
`--frame-skip 2` the box must then be a staircase: frames 0 and 1 carry the
same box, 2 and 3 carry the same box, and so on -- because frame 1's mask *is*
frame 0's, held. An unskipped run of the same clip has no such structure; its
box moves with the target every frame.

Usage:
    python tools/verify_frame_skip.py frame_output/record_027/full512_skip2/tracked.mp4 --stride 2
    # with the baseline for contrast (recommended -- see the note on a static target)
    python tools/verify_frame_skip.py .../full512_skip2/tracked.mp4 --stride 2 \
        --baseline frame_output/record_027/full512/tracked.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def overlay_boxes(video: Path, saturation: int = 40) -> list[tuple | None]:
    """The tracked box in each frame, found by colour, or None where absent.

    The overlay is drawn in a fixed palette colour and the source is grayscale,
    so any pixel whose channels disagree by more than `saturation` belongs to
    the overlay and nothing else does. On a colour source raise the threshold
    (or trust the box-drawing colour only).
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"could not open {video}")
    out: list[tuple | None] = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            img = bgr.astype(np.int16)
            spread = img.max(axis=2) - img.min(axis=2)
            ys, xs = np.where(spread > saturation)
            out.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                       if len(xs) else None)
    finally:
        cap.release()
    return out


def held_pairs(boxes: list[tuple | None], stride: int, tolerance: int = 1) -> tuple[int, int]:
    """(pairs whose box is unchanged, pairs compared) within each stride group.

    A tolerance of a pixel absorbs mp4 compression moving an edge; it cannot
    absorb a target that actually moved, which is a box shifting by its own
    motion over a whole frame period.
    """
    same = total = 0
    for start in range(0, len(boxes), stride):
        first = boxes[start]
        for other in boxes[start + 1:start + stride]:
            if first is None or other is None:
                continue
            total += 1
            same += all(abs(a - b) <= tolerance for a, b in zip(first, other))
    return same, total


def report(name: str, video: Path, stride: int, saturation: int) -> float:
    boxes = overlay_boxes(video, saturation)
    found = sum(b is not None for b in boxes)
    same, total = held_pairs(boxes, stride)
    share = same / total if total else 0.0
    print(f"{name}: {len(boxes)} frames, box found in {found}")
    if total:
        print(f"   {same}/{total} ({share:.0%}) of frames inside a group of {stride} "
              "carry an unchanged box")
    return share


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path, help="tracked.mp4 from the skipped run.")
    p.add_argument("--stride", type=int, default=2, help="The N passed to --frame-skip.")
    p.add_argument("--baseline", type=Path, default=None,
                   help="tracked.mp4 from the unskipped run of the same clip. Without "
                        "it a target that never moves looks exactly like a held mask, "
                        "and the check cannot tell them apart.")
    p.add_argument("--saturation", type=int, default=40,
                   help="How far a pixel's channels must disagree to count as overlay "
                        "rather than grayscale source (default 40).")
    args = p.parse_args(argv)

    skipped = report("skipped ", args.video, args.stride, args.saturation)
    base = None
    if args.baseline is not None:
        base = report("baseline", args.baseline, args.stride, args.saturation)

    print()
    if skipped < 0.9:
        print(f"NOT HOLDING: only {skipped:.0%} of the skipped frames repeat their "
              "group's box. Either the run did not use --frame-skip, or this is not "
              "the skipped video.")
        return 1
    if base is None:
        print(f"HOLDING: {skipped:.0%} of frames repeat their group's box, which is "
              "the mask being held over the frames the model never saw. Pass "
              "--baseline to rule out a target that simply does not move.")
        return 0
    if base > 0.9:
        print(f"INCONCLUSIVE: the baseline also repeats its box {base:.0%} of the "
              "time, so the target barely moves in this clip and a held mask is "
              "indistinguishable from a fresh one. Try a clip with more motion.")
        return 1
    print(f"CONFIRMED: {skipped:.0%} held in the skipped run against {base:.0%} in "
          f"the baseline. The box moves every frame when every frame is inferred, "
          f"and every {args.stride} frames when one in {args.stride} is -- which is "
          "the skip, visible in the output rather than only in the log.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
