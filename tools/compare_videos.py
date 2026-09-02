#!/usr/bin/env python3
"""Tile several `run_records.py` runs of one clip into a single comparison video.

`run_records.py` writes one `tracked.mp4` per arm, in its own folder, off one
saved prompt -- same target, same frames, one variable. That is what makes them
comparable and it is also what makes them awkward to show: the difference
between two checkpoints is a handful of frames somewhere in the middle of two
files nobody is going to scrub through in parallel.

This puts them in one frame, labelled, so the difference is where the eye
already is. It reads the mp4s `run_records` wrote and nothing else -- no model,
no engine, no CUDA -- so it runs anywhere the clips are, including a laptop
with the results copied onto it.

    python tools/compare_videos.py \\
        --record frame_output/records96_768_final/record_s96 \\
        --mode full768 \\
        --arms stock,aerial_stable,aerial_stable_woc_final,aerial_stable_rgb

An arm is a `--weights` name and resolves to the folder `run_records` gave it:
`stock` keeps the bare `<mode>/`, everything else is `<mode>_<arm>/`. A folder
name works too, which is how a `--policy` rung joins the grid:

    --arms full768,full768_aerial_stable,full768_aerial_stable_guard

`--stills N` also writes N PNGs of the same grid at evenly spaced frames. A
still is what goes in a slide; the video is what plays behind it.

WHAT THIS CANNOT TELL YOU. These recordings carry no drawn answer, so nothing
here is an IoU and no pane is automatically the winner -- read it beside the
`SUMMARY.md` the same run wrote. An arm trained on another modality (the RGB
model on a thermal clip) is in the grid because it was asked for, not because
its pane is evidence about the thermal ones.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# A label band that scales with the pane, so a 768-wide grid and a 1920-wide one
# read the same. Fractions of pane width, measured to stay legible projected.
BAND_FRACTION = 0.075          # band height, as a fraction of pane height
TEXT_FRACTION = 0.55           # text height within the band
MIN_BAND = 22


def folder_for(mode: str, arm: str) -> str:
    """The folder `run_records.py` wrote for this arm, mirroring its `folder()`.

    A plain `stock` run keeps the bare `<mode>` name there, so it has to keep it
    here -- looking for `full768_stock/` would find nothing and report a missing
    run for the one arm that is always present. An arm that already names a
    folder is passed through, which is how a policy rung (`full768_x_guard`)
    joins a grid of plain ones.
    """
    if arm == mode or arm.startswith(mode + "_"):
        return arm
    return mode if arm == "stock" else f"{mode}_{arm}"


def grid_shape(count: int, cols: int | None = None) -> tuple[int, int]:
    """`(rows, cols)` for `count` panes: one row up to three, then two rows.

    Two arms belong side by side and three belong in a row -- that is the shape
    of the question being asked. Four in a row on a 16:9 slide leaves each pane
    too small to see a mask edge, which is the whole reason for the picture.
    """
    if count < 1:
        raise ValueError("nothing to tile")
    if cols:
        return math.ceil(count / cols), cols
    if count <= 3:
        return 1, count
    return 2, math.ceil(count / 2)


def band_height(pane_h: int) -> int:
    """The caption band above one pane. Named so the sheet size is known before
    a frame is read -- the writer needs it up front."""
    return max(MIN_BAND, int(pane_h * BAND_FRACTION))


def _label(pane, text: str, cv2, np):
    """`pane` with a caption band above it, returned as a new array."""
    h, w = pane.shape[:2]
    band = band_height(h)
    out = np.zeros((h + band, w, 3), dtype=pane.dtype)
    out[band:] = pane
    scale = cv2.getFontScaleFromHeight(
        cv2.FONT_HERSHEY_SIMPLEX, max(10, int(band * TEXT_FRACTION)), 1)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(out, text, ((w - tw) // 2, (band + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _tile(panes, rows: int, cols: int, np):
    """Panes (already labelled and equal-sized) into one image, row by row."""
    h, w = panes[0].shape[:2]
    sheet = np.zeros((rows * h, cols * w, 3), dtype=panes[0].dtype)
    for i, pane in enumerate(panes):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = pane
    return sheet


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--record", required=True,
                   help="The record directory run_records wrote, i.e. "
                        "<out>/<tag>/<record name>/ -- the one holding "
                        "full768/, full768_aerial_stable/ and so on.")
    p.add_argument("--mode", default="full768",
                   help="Which input configuration to compare. All panes come "
                        "from one mode because a crop and a resize are different "
                        "pictures, and the grid is for varying the model.")
    p.add_argument("--arms", required=True,
                   help="Comma-separated --weights names, in the order they "
                        "should appear. A folder name is taken as given.")
    p.add_argument("--out", default=None,
                   help="Output mp4 (default: <record>/compare_<mode>.mp4).")
    p.add_argument("--cols", type=int, default=None,
                   help="Force the column count (default: one row up to three "
                        "arms, two rows above that).")
    p.add_argument("--width", type=int, default=1920,
                   help="Width of the finished grid, panes scaled to fit "
                        "(default 1920). Four 1280x768 panes untouched are "
                        "2560 across, which no slide wants.")
    p.add_argument("--fps", type=float, default=None,
                   help="Override the fps read from the first arm's mp4.")
    p.add_argument("--stills", type=int, default=0,
                   help="Also write this many PNGs of the grid, at evenly "
                        "spaced frames. These are what go in the slides.")
    args = p.parse_args(argv)

    import cv2
    import numpy as np

    from src.io_utils import open_video_writer

    record = Path(args.record).expanduser().resolve()
    if not record.is_dir():
        raise SystemExit(f"{record} is not a directory.")

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    videos = []
    for arm in arms:
        path = record / folder_for(args.mode, arm) / "tracked.mp4"
        if not path.is_file():
            have = sorted(d.name for d in record.iterdir()
                          if d.is_dir() and (d / "tracked.mp4").is_file())
            listing = ", ".join(have) if have else "(no run with a tracked.mp4)"
            raise SystemExit(
                f"no run for --arms {arm}: {path} is not there.\n"
                f"{record} holds: {listing}\n"
                f"An arm that did not finish has no mp4 -- read its run.txt.")
        videos.append(path)

    caps = [cv2.VideoCapture(str(v)) for v in videos]
    try:
        sizes = {(int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))) for c in caps}
        if len(sizes) > 1:
            raise SystemExit(
                f"the arms are not one frame size ({sorted(sizes)}), so they are "
                f"not one mode. Compare within a mode: a crop and a resize are "
                f"different pictures of the same clip and tiling them would "
                f"invite reading the difference between them as a result.")
        counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
        frames = min(counts)
        if len(set(counts)) > 1:
            print(f"!! the arms are {counts} frames long; using the shortest "
                  f"({frames}). A short arm is a run that stopped early.")
        if frames < 1:
            raise SystemExit("the arms hold no frames.")

        rows, cols = grid_shape(len(caps), args.cols)
        src_w, src_h = sizes.pop()
        pane_w = max(2, (args.width // cols) // 2 * 2)
        pane_h = max(2, int(src_h * pane_w / src_w) // 2 * 2)
        band = band_height(pane_h)
        sheet_size = (cols * pane_w, rows * (pane_h + band))
        fps = args.fps or (caps[0].get(cv2.CAP_PROP_FPS) or 30.0)

        out_path = Path(args.out) if args.out else record / f"compare_{args.mode}.mp4"
        still_at = {round((i + 1) * frames / (args.stills + 1))
                    for i in range(max(args.stills, 0))}

        print(f">> {len(caps)} arms, {frames} frames, {rows}x{cols} grid, "
              f"{sheet_size[0]}x{sheet_size[1]}, {fps:g} fps")
        for arm, video in zip(arms, videos):
            print(f"   {arm:<32} {video.parent.name}/")

        with open_video_writer(out_path, fps, sheet_size) as write:
            for idx in range(frames):
                panes = []
                for cap, arm in zip(caps, arms):
                    ok, bgr = cap.read()
                    if not ok:
                        bgr = np.zeros((src_h, src_w, 3), dtype=np.uint8)
                    pane = cv2.resize(bgr, (pane_w, pane_h),
                                      interpolation=cv2.INTER_AREA)
                    panes.append(_label(pane, arm, cv2, np))
                while len(panes) < rows * cols:   # a ragged last row stays black
                    panes.append(np.zeros_like(panes[0]))
                sheet = _tile(panes, rows, cols, np)
                write(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB))
                if idx in still_at:
                    still = out_path.with_name(f"{out_path.stem}_f{idx:05d}.png")
                    cv2.imwrite(str(still), sheet)
                    print(f"   still: {still.name}")
    finally:
        for cap in caps:
            cap.release()

    print(f"\n>> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
