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


def trim_span(frames: int, fps: float, start: float,
              seconds: float | None) -> tuple[int, int]:
    """`(first frame, how many)` for `--start` / `--seconds`, in frames.

    Separated from the reading loop because the arithmetic is where a trim goes
    wrong quietly: a `--start` past the end would otherwise write an empty
    video, and a `--seconds` longer than what is left would silently mean "the
    rest" without saying so. Both are decided here, before an arm is opened.
    """
    first = max(0, round(start * fps))
    if first >= frames:
        raise SystemExit(
            f"--start {start:g}s is {first} frames into a clip that is "
            f"{frames} frames ({frames / fps:.1f}s) long.")
    keep = frames - first
    if seconds is not None:
        keep = min(keep, max(1, round(seconds * fps)))
    return first, keep


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


def _fit(frame, pane_w: int, pane_h: int, cv2, np):
    """`frame` scaled to fit `pane_w`x`pane_h` and centred, never stretched.

    Aspect is preserved because the whole point of putting a crop beside a full
    frame is the field of view, and stretching one to the other's shape is
    exactly the thing that would hide it.
    """
    h, w = frame.shape[:2]
    scale = min(pane_w / w, pane_h / h)
    new_w, new_h = max(2, int(w * scale)), max(2, int(h * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if (new_w, new_h) == (pane_w, pane_h):
        return resized
    out = np.zeros((pane_h, pane_w, 3), dtype=frame.dtype)
    y0, x0 = (pane_h - new_h) // 2, (pane_w - new_w) // 2
    out[y0:y0 + new_h, x0:x0 + new_w] = resized
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
    p.add_argument("--labels", default=None,
                   help="Comma-separated captions, one per arm and in the same "
                        "order. Default: the arm names. A run name and a caption "
                        "answer to different readers -- `aerial_stable_woc_final` "
                        "identifies which checkpoint ran, `finetune` says what "
                        "the slide is claiming -- and a projected pane has room "
                        "for the second. The folder each pane came from is still "
                        "printed here and still in provenance.json, so renaming "
                        "the caption never loses which weights produced it.")
    p.add_argument("--out", default=None,
                   help="Output mp4 (default: <record>/compare_<mode>.mp4).")
    p.add_argument("--cols", type=int, default=None,
                   help="Force the column count (default: one row up to three "
                        "arms, two rows above that).")
    p.add_argument("--width", type=int, default=1920,
                   help="Width of the finished grid, panes scaled to fit "
                        "(default 1920). Four 1280x768 panes untouched are "
                        "2560 across, which no slide wants.")
    p.add_argument("--no-captions", dest="captions_on", action="store_false",
                   help="Tile the panes bare, with no caption band. For a slide "
                        "that carries its own labels underneath: a caption burnt "
                        "into the frame cannot be restyled, translated or moved, "
                        "and two of them saying the same thing is worse than "
                        "none. The order of --arms is still the order of the "
                        "panes, and the tool prints which folder each one came "
                        "from, so the mapping is recoverable from the terminal.")
    p.add_argument("--fit", action="store_true",
                   help="Pad arms of different frame sizes into a common pane "
                        "instead of refusing them. Off by default because the "
                        "usual reason two arms differ is that they are two modes, "
                        "and tiling a crop beside a resize invites reading the "
                        "framing as a result. Turn it on when the framing IS the "
                        "result -- 'this is what the 768 crop sees, this is what "
                        "the whole frame squeezed into 768 sees'. Each pane keeps "
                        "its own aspect ratio and is centred on black; nothing is "
                        "stretched, so what the bars show is field of view.")
    p.add_argument("--start", type=float, default=0.0,
                   help="Skip this many seconds of every arm before tiling. The "
                        "arms are seeked together, so they stay frame-aligned -- "
                        "which is the only thing that makes the picture a "
                        "comparison rather than two clips playing near each "
                        "other.")
    p.add_argument("--seconds", type=float, default=None,
                   help="Length to keep, in seconds (default: to the end). A "
                        "slide gets thirty seconds of attention, not four "
                        "minutes. Stills are still named by their frame number "
                        "in the original clip, so a trimmed grid can be traced "
                        "back to the run it came from.")
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
    captions = ([c.strip() for c in args.labels.split(",")]
                if args.labels else list(arms))
    if len(captions) != len(arms):
        raise SystemExit(
            f"--labels has {len(captions)} captions for {len(arms)} arms. They "
            f"are matched by position, so a short list would caption the wrong "
            f"pane rather than leave one blank.")
    videos = []
    for arm in arms:
        # A directory that exists under --record is taken as given. That is what
        # lets a grid cross modes (`--arms full768,crop768`), which the mode rule
        # below could not express: it would read `crop768` as a weights name and
        # look for `full768_crop768/`.
        named = record / arm
        folder = arm if named.is_dir() else folder_for(args.mode, arm)
        path = record / folder / "tracked.mp4"
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
        if len(sizes) > 1 and not args.fit:
            raise SystemExit(
                f"the arms are not one frame size ({sorted(sizes)}), so they are "
                f"not one mode. Compare within a mode: a crop and a resize are "
                f"different pictures of the same clip and tiling them would "
                f"invite reading the difference between them as a result.\n"
                f"Pass --fit when that difference is the point -- it pads each "
                f"pane rather than stretching it, so the bars show field of view.")
        counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
        frames = min(counts)
        if len(set(counts)) > 1:
            print(f"!! the arms are {counts} frames long; using the shortest "
                  f"({frames}). A short arm is a run that stopped early.")
        if frames < 1:
            raise SystemExit("the arms hold no frames.")

        rows, cols = grid_shape(len(caps), args.cols)
        each = [(int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))) for c in caps]
        pane_w = max(2, (args.width // cols) // 2 * 2)
        # Width-fit every arm, then take the tallest result as the common pane:
        # a shorter one is padded rather than stretched, so a 16:9 arm beside a
        # square one loses no pixels and gains no aspect it did not have.
        pane_h = max(2, max(int(h * pane_w / w) for w, h in each) // 2 * 2)
        band = band_height(pane_h) if args.captions_on else 0
        sheet_size = (cols * pane_w, rows * (pane_h + band))
        fps = args.fps or (caps[0].get(cv2.CAP_PROP_FPS) or 30.0)

        # Trimmed after the arms agree on a length, so `--start` past the end
        # of the shortest arm is caught here rather than as an empty video.
        first, keep = trim_span(frames, fps, args.start, args.seconds)
        if first or keep != frames:
            print(f">> trimmed to {keep} frames ({keep / fps:.1f}s) from frame "
                  f"{first} ({first / fps:.1f}s)")

        out_path = Path(args.out) if args.out else record / f"compare_{args.mode}.mp4"
        still_at = {round((i + 1) * keep / (args.stills + 1))
                    for i in range(max(args.stills, 0))}

        print(f">> {len(caps)} arms, {keep} frames, {rows}x{cols} grid, "
              f"{sheet_size[0]}x{sheet_size[1]}, {fps:g} fps")
        for caption, video in zip(captions, videos):
            print(f"   {caption:<32} {video.parent.name}/")

        # Read and discard rather than seek: mp4 seeking lands on a keyframe,
        # and an arm that started one frame off the others would be a
        # comparison of two different moments.
        for _ in range(first):
            for cap in caps:
                cap.read()

        with open_video_writer(out_path, fps, sheet_size) as write:
            for idx in range(keep):
                panes = []
                for cap, caption, (sw, sh) in zip(caps, captions, each):
                    ok, bgr = cap.read()
                    if not ok:
                        bgr = np.zeros((sh, sw, 3), dtype=np.uint8)
                    pane = _fit(bgr, pane_w, pane_h, cv2, np)
                    panes.append(_label(pane, caption, cv2, np)
                                 if args.captions_on else pane)
                while len(panes) < rows * cols:   # a ragged last row stays black
                    panes.append(np.zeros_like(panes[0]))
                sheet = _tile(panes, rows, cols, np)
                write(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB))
                if idx in still_at:
                    # Numbered in the original clip, not in the trim, so a still
                    # can be found again in the run it came from.
                    still = out_path.with_name(
                        f"{out_path.stem}_f{first + idx:05d}.png")
                    cv2.imwrite(str(still), sheet)
                    print(f"   still: {still.name}")
    finally:
        for cap in caps:
            cap.release()

    print(f"\n>> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
