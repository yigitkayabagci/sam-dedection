#!/usr/bin/env python3
"""Watch one sequence run: N consecutive masked frames, image beside mask.

`inspect_stage_c.py` answers a different question. It samples *around the
disappearances* -- twelve tiles taken from four places in a sequence -- because
what it is there to show is the transition. That sheet cannot tell you whether
the masks hold together in between: whether frame 41's mask is the same object
as frame 40's, or whether the store drifted onto a rooftop for twenty frames
and came back. Only a run of neighbours, in order, shows that.

So this takes the **longest run of consecutive masked frames** in one sequence
and renders it: left the frame, right the same frame with its mask and the
mask's bounding box, captioned with the area. Watch it as an MP4; read it as a
contact sheet if the MP4 cannot be written.

**"Consecutive" means consecutive *extracted* frames, not consecutive seconds
of video.** VTUAV-VIS is extracted with `--frames masked`, which keeps only the
frames that have a drawn mask -- normally every 30th source frame. 100
consecutive items here are therefore about 100 seconds of a 30 fps recording
seen one frame per second, not 3.3 seconds of continuous motion. The object
should still be the same object and should still move smoothly-ish across the
run; it will not look like video. Runs are found in the store's own frame
indices -- position within `Sequence.frames` -- and never in filename order,
because the filenames carry the source frame numbers and those jump by 30.

**The percentile stretch is for looking, not for measuring.** A 16-bit thermal
frame and a dark 8-bit one are both black under a plain cast, so both halves
are stretched 2/98 before drawing. Training does not see these pixels;
`aerial.load_image` normalises differently. No number printed off the stretched
image would mean anything -- the areas printed come from the mask itself.

The data comes from `vtuav_vis_sequences`, the same reader Stage C trains
from, so what is previewed is what the loss sees.

    python tools/preview_sequence.py --root /content/data/VTUAV_VIS_aerial_stage_c \
        --modality rgb --count 100 --out /content/preview

With no `--sequence` it previews the sequence with the most masked frames.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence as SequenceABC
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HALF = 640                  # each half of a panel, in pixels wide
CAPTION = 30                # the caption strip under a panel, in pixels tall
SHEET_WIDTH = 520           # a whole panel's width in the contact sheet
SHEET_COLUMNS = 5
# Colours are BGR: everything here is handed straight to cv2, which writes BGR.
MASK = (60, 190, 250)       # the mask overlay, and its box
INK = (255, 255, 255)
SHADOW = (0, 0, 0)
EMPTY = (235, 70, 70)       # a frame whose stored mask has no pixels in it
# COCO's absolute-area buckets, used here only as a label to read the run by.
# The user's interest is small and medium targets; these say which is which
# without pretending to be a measurement of anything.
SMALL = 32 * 32
MEDIUM = 96 * 96


def longest_run(indices: SequenceABC[int] | Mapping[int, object],
                count: int = 100) -> np.ndarray:
    """The longest stretch of consecutive frame indices, trimmed to `count`.

    `indices` is the store's keys: the positions in `Sequence.frames` that
    carry a mask. A run is a maximal set of consecutive integers among them,
    so a single missing index breaks one run into two -- which is the point,
    since a preview that silently jumped a gap would show a discontinuity the
    data does not have. The longest run wins even when a shorter one starts
    earlier; ties go to the earlier run.
    """
    keys = sorted(int(i) for i in indices)
    if not keys:
        return np.zeros(0, dtype=int)
    runs: list[list[int]] = [[keys[0]]]
    for key in keys[1:]:
        if key == runs[-1][-1] + 1:
            runs[-1].append(key)
        else:
            runs.append([key])
    best = max(runs, key=len)          # max() keeps the first of equal lengths
    return np.asarray(best[:max(int(count), 1)], dtype=int)


def pick_sequence(sequences: SequenceABC, stores: Mapping[str, Mapping]):
    """The sequence with the most masked frames -- the one worth watching.

    Taking the first by name usually takes a sequence with a handful of masks
    in it, which shows nothing about whether the masks hold together.
    """
    ranked = sorted(sequences, key=lambda s: (-len(stores.get(s.name, {})),
                                              s.name))
    return ranked[0] if ranked else None


def align(mask, shape: tuple[int, int]) -> np.ndarray:
    """A mask as a boolean array the size of the frame, nearest-neighbour.

    Stores are allowed to hold masks at another resolution. Areas are only
    comparable once they are in frame pixels, so the resize happens before
    anything is counted, not after.
    """
    mask = np.asarray(mask).astype(bool)
    if mask.ndim == 3:
        mask = mask[..., 0]
    height, width = int(shape[0]), int(shape[1])
    if mask.shape == (height, width) or mask.size == 0:
        return mask
    rows = np.clip(np.arange(height) * mask.shape[0] // max(height, 1),
                   0, mask.shape[0] - 1)
    columns = np.clip(np.arange(width) * mask.shape[1] // max(width, 1),
                      0, mask.shape[1] - 1)
    return mask[rows][:, columns]


def mask_stats(mask, shape: tuple[int, int]) -> dict:
    """Area in frame pixels, area as a share of the whole frame, and empty.

    The share is of the **frame**, not of a crop or of the mask's own bounding
    box: it is the number that says whether this is a small target, and a
    share of the box would be near 1 for every target ever drawn.
    """
    height, width = int(shape[0]), int(shape[1])
    frame_area = max(height * width, 1)
    if mask is None:
        return {"area": 0, "share": 0.0, "empty": True, "box": None}
    area_mask = align(mask, (height, width))
    area = int(np.count_nonzero(area_mask))
    box = None
    if area:
        rows, columns = np.nonzero(area_mask)
        box = (int(columns.min()), int(rows.min()),
               int(columns.max()) + 1, int(rows.max()) + 1)
    return {"area": area, "share": area / frame_area, "empty": area == 0,
            "box": box}


def stretch(frame: np.ndarray) -> np.ndarray:
    """An 8-bit BGR view of a frame, percentile-stretched so it is visible.

    Unlike `inspect_stage_c.stretch` this keeps colour: VTUAV-VIS's RGB half is
    previewed here too, and judging whether a mask sits on the right object is
    much easier in colour. The percentiles are taken over all channels at once
    so the colour balance survives. For looking only -- see the module
    docstring.
    """
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[2] == 1:
        frame = frame[..., 0]
    lo, hi = np.percentile(frame, [2, 98])
    scaled = np.clip((frame.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    out = (scaled * 255).astype(np.uint8)
    return out if out.ndim == 3 else np.dstack([out] * 3)


def _label(canvas: np.ndarray, text: str, origin: tuple[int, int],
           colour=INK) -> None:
    import cv2

    for shade, thickness in ((SHADOW, 3), (colour, 1)):
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    shade, thickness, cv2.LINE_AA)


def panel(frame: np.ndarray, mask, index: int, half: int = HALF
          ) -> tuple[np.ndarray, dict]:
    """One side-by-side pair: the frame, then the frame with its own mask.

    Returns the panel and the stats it was captioned with, so a caller never
    has to recompute the area it is looking at.
    """
    import cv2

    plain = stretch(frame)
    height, width = plain.shape[:2]
    stats = mask_stats(mask, (height, width))
    drawn = plain.copy()
    if not stats["empty"]:
        area_mask = align(mask, (height, width))
        drawn[area_mask] = (0.45 * drawn[area_mask]
                            + 0.55 * np.array(MASK)).astype(np.uint8)
        # The box is padded outward and scaled with the frame: a 20-pixel
        # target's box drawn tight and one pixel thick disappears in the
        # downscale, and a box drawn *over* the mask hides the thing being
        # judged. Small targets are the case this viewer exists for.
        thickness = max(1, int(round(width / HALF)))
        pad = thickness + 2
        x0, y0, x1, y1 = stats["box"]
        cv2.rectangle(drawn, (max(x0 - pad, 0), max(y0 - pad, 0)),
                      (min(x1 - 1 + pad, width - 1),
                       min(y1 - 1 + pad, height - 1)), MASK, thickness)

    scale = half / max(width, 1)
    size = (half, max(int(round(height * scale)), 1))
    plain = cv2.resize(plain, size, interpolation=(
        cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR))
    drawn = cv2.resize(drawn, size, interpolation=(
        cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR))
    body = np.hstack([plain, drawn])
    strip = np.zeros((CAPTION, body.shape[1], 3), np.uint8)
    canvas = np.vstack([body, strip])
    _label(canvas, f"#{index}   {width}x{height}",
           (8, canvas.shape[0] - 9))
    right = (f"{stats['area']} px   {stats['share'] * 100:.3f}% of frame"
             if not stats["empty"] else "MASK EMPTY")
    _label(canvas, right, (half + 8, canvas.shape[0] - 9),
           INK if not stats["empty"] else EMPTY)
    cv2.line(canvas, (half, 0), (half, body.shape[0]), (40, 40, 40), 1)
    return canvas, stats


def contact_sheet(panels: SequenceABC[np.ndarray],
                  columns: int = SHEET_COLUMNS) -> np.ndarray:
    """Every panel of the run in one PNG, in order, left to right.

    The fallback for a machine whose OpenCV cannot write an MP4 -- a headless
    Colab image without the codec, most often. It is worse for judging motion
    and better for comparing two frames far apart, so it is written either way.
    """
    import cv2

    tiles = []
    for image in panels:
        scale = SHEET_WIDTH / max(image.shape[1], 1)
        tiles.append(cv2.resize(
            image, (SHEET_WIDTH, max(int(round(image.shape[0] * scale)), 1)),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR))
    if not tiles:
        return np.zeros((1, 1, 3), np.uint8)
    height = max(t.shape[0] for t in tiles)
    rows = []
    for start in range(0, len(tiles), columns):
        row = np.zeros((height, SHEET_WIDTH * columns, 3), np.uint8)
        for column, tile in enumerate(tiles[start:start + columns]):
            row[:tile.shape[0],
                column * SHEET_WIDTH:column * SHEET_WIDTH + tile.shape[1]] = tile
        rows.append(row)
    return np.vstack(rows)


def write_video(path: Path, panels: SequenceABC[np.ndarray], fps: float
                ) -> bool:
    """Write the run as an mp4v MP4; say so and give up if OpenCV cannot.

    Returns whether the file was written. A missing codec is not an error
    worth stopping for: the contact sheet still carries the same frames.
    """
    import cv2

    if not panels:
        return False
    height, width = panels[0].shape[:2]
    try:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(fps), (width, height))
    except Exception as error:                        # pragma: no cover
        print(f"!! VideoWriter raised ({error}); writing the PNG only.")
        return False
    if not writer.isOpened():
        print(f"!! OpenCV could not open an mp4v writer for {path}; "
              "writing the PNG only.")
        return False
    for image in panels:
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height))
        writer.write(image)
    writer.release()
    return path.is_file() and path.stat().st_size > 0


def preview(sequence, store: Mapping[int, np.ndarray], out: Path,
            count: int = 100, fps: float = 6.0, quiet: bool = False) -> dict:
    """Render the longest consecutive masked run of one sequence.

    `out` may be a directory or an `.mp4` path; either way an MP4 and a PNG of
    the same stem are written beside each other.
    """
    import cv2

    out = Path(out)
    if out.suffix.lower() in (".mp4", ".png"):
        out.parent.mkdir(parents=True, exist_ok=True)
        stem = out.with_suffix("")
    else:
        out.mkdir(parents=True, exist_ok=True)
        stem = out / sequence.name
    indices = longest_run(store, count)

    panels, rows = [], []
    for index in indices:
        index = int(index)
        frame = cv2.imread(str(sequence.frames[index]), cv2.IMREAD_UNCHANGED)
        if frame is None:
            continue
        mask = store.get(index) if store is not None else None
        image, stats = panel(frame, mask, index)
        panels.append(image)
        rows.append({"index": index, "area": stats["area"],
                     "share": stats["share"], "empty": stats["empty"]})

    sheet_path = stem.with_suffix(".png")
    video_path = stem.with_suffix(".mp4")
    sheet = contact_sheet(panels)
    if sheet.size > 3:
        cv2.imwrite(str(sheet_path), sheet)
    wrote_video = write_video(video_path, panels, fps)

    report = {
        "sequence": sequence.name,
        "masked_frames": len(store) if store is not None else 0,
        "run": [int(i) for i in indices],
        "video": str(video_path) if wrote_video else None,
        "sheet": str(sheet_path) if sheet.size > 3 else None,
        "frames": rows,
    }
    if not quiet:
        _print_report(report)
    return report


def _print_report(report: dict) -> None:
    """The per-frame table. The share column is the one to read."""
    rows = report["frames"]
    print(f"\nsequence      {report['sequence']}")
    print(f"masked frames {report['masked_frames']}")
    if report["run"]:
        print(f"longest run   {len(report['run'])} consecutive extracted "
              f"frames, {report['run'][0]}..{report['run'][-1]}")
    print("consecutive = consecutive EXTRACTED frames; VTUAV-VIS keeps about "
          "every 30th\n              source frame, so this is not 100 "
          "contiguous seconds of video.")
    print(f"\n{'frame':>7}{'area px':>10}{'SHARE OF FRAME':>18}"
          f"{'size':>9}   mask")
    for row in rows:
        share = row["share"] * 100
        bucket = ("--" if row["empty"] else
                  "small" if row["area"] < SMALL else
                  "medium" if row["area"] < MEDIUM else "large")
        print(f"{row['index']:>7}{row['area']:>10}{share:>17.4f}%"
              f"{bucket:>9}   {'EMPTY' if row['empty'] else 'ok'}")
    shares = [row["share"] * 100 for row in rows if not row["empty"]]
    if shares:
        print(f"\nshare of frame: min {min(shares):.4f}%  "
              f"median {float(np.median(shares)):.4f}%  "
              f"max {max(shares):.4f}%")
        areas = [row["area"] for row in rows if not row["empty"]]
        small = sum(1 for a in areas if a < SMALL)
        medium = sum(1 for a in areas if SMALL <= a < MEDIUM)
        print(f"small (<{SMALL} px) {small}   "
              f"medium (<{MEDIUM} px) {medium}   "
              f"large {len(areas) - small - medium}"
              "   [COCO's area buckets, as a label to read the run by]")
    empties = sum(1 for row in rows if row["empty"])
    if empties:
        print(f"!! {empties} of {len(rows)} frames carry an empty mask: the "
              "store has an entry there\n   and the loss will be trained on "
              "no foreground at all.")
    print(f"\nvideo  {report['video'] or '(not written)'}")
    print(f"sheet  {report['sheet'] or '(not written)'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True,
                        help="The VTUAV-VIS root notebook 31 staged, the one "
                             "with <sequence>/mask/<modality>/ under it.")
    parser.add_argument("--sequence",
                        help="Sequence name, with or without the vtuav_vis__ "
                             "prefix. Omitted, the sequence with the most "
                             "masked frames is previewed.")
    parser.add_argument("--modality", choices=("ir", "rgb"), default="ir",
                        help="Which half to read. `mask/ir` and `mask/rgb` "
                             "hold different masks, so this picks the pixels "
                             "and the masks together.")
    parser.add_argument("--count", type=int, default=100,
                        help="How many consecutive masked frames to render.")
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--out", type=Path, required=True,
                        help="A directory, or an .mp4 path. Both an MP4 and a "
                             "PNG contact sheet of that stem are written.")
    args = parser.parse_args()

    from src.training.aerial_video import vtuav_vis_sequences

    sequences, stores = vtuav_vis_sequences(args.root, modality=args.modality)
    if args.sequence:
        wanted = {args.sequence, f"vtuav_vis__{args.sequence}"}
        chosen = next((s for s in sequences if s.name in wanted), None)
        if chosen is None:
            names = ", ".join(sorted(s.name for s in sequences)[:12])
            raise SystemExit(f"{args.sequence}: no such sequence under "
                             f"{args.root}. Available: {names}")
    else:
        chosen = pick_sequence(sequences, stores)
    if chosen is None:
        raise SystemExit(f"{args.root}: nothing to preview.")

    report = preview(chosen, stores.get(chosen.name, {}), args.out,
                     count=args.count, fps=args.fps)
    if report["sheet"]:
        Path(report["sheet"]).with_suffix(".json").write_text(
            json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
