#!/usr/bin/env python3
"""Look at what Stage C actually trains on, one sequence at a time.

Stage C is the only stage with video in it, and video is the only place the
failure this project cares about appears -- a target leaves the frame, comes
back, and the tracker does or does not pick it up again. A table of frame
counts cannot show that. This writes contact sheets: a strip of real frames
with the box, the mask if the run has one, and the `exist` flag drawn on each,
so a disappearance can be *seen*.

**It samples around the disappearances on purpose.** A uniform stride over a
1 500-frame sequence spends almost all of its tiles on ordinary tracking and
usually misses every transition. `windows_around_gaps` finds the frames where
`exist` flips and centres a window on each; sequences with no flip fall back to
a uniform sample so a source without disappearances is still inspectable.

**The masks drawn are the ones the loss sees, not a re-derivation.** Stage C's
supervision comes from `STORES[sequence.name][frame_index]` -- VTUAV-VIS's own
instance masks and notebook 17/25's accepted teacher masks -- and a frame with
no entry there trains on box projection instead. Both are drawn differently, so
"this frame has a real mask" and "this frame has only a box" are distinguishable
at a glance. That distinction is the point: a run can look fully supervised in
the frame count and be box-only in the frames that matter.

From a notebook that already has them assembled:

    from tools.inspect_stage_c import render
    render(sequences, STORES, Path(MIRROR) / "inspect")

Standalone, rebuilding the same sequences from their roots:

    python tools/inspect_stage_c.py --vtuav /content/data/VTUAV_aerial_stage_c \
        --vtuav-vis /content/data/VTUAV_VIS_aerial_stage_c \
        --birdsai /content/data/BIRDSAI --out /content/inspect
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence as SequenceABC
from pathlib import Path

import numpy as np

TILE = 320                  # each frame's side in the sheet, in pixels
COLUMNS = 6
PRESENT = (90, 220, 90)     # box colour where `exist` is True
MASKED = (250, 190, 60)     # the frame carries a real mask
ABSENT = (70, 70, 235)      # `exist` is False -- nothing to find in this frame


def windows_around_gaps(exist: np.ndarray, span: int, count: int
                        ) -> list[np.ndarray]:
    """Frame indices to draw: one window per `exist` transition, then filler.

    A transition is where the flag changes, which is both the disappearance and
    the return -- the two frames a memory bank has to get right. Windows are
    centred on it and clipped to the sequence, and a sequence whose flag never
    changes gets a uniform sample instead so it is not silently skipped.
    """
    exist = np.asarray(exist).astype(bool)
    total = int(exist.shape[0])
    if total == 0:
        return []
    span = max(int(span), 1)
    flips = list(np.flatnonzero(exist[1:] != exist[:-1]) + 1)

    windows: list[np.ndarray] = []
    for flip in flips[:count]:
        start = int(np.clip(flip - span // 2, 0, max(total - span, 0)))
        windows.append(np.arange(start, min(start + span, total)))
    if len(windows) < count:
        # Filler is uniform over the whole sequence, not more of the same
        # neighbourhood: what it is there to show is the ordinary case.
        stride = max(total // max(span, 1), 1)
        uniform = np.arange(0, total, stride)[:span]
        if uniform.size:
            windows.append(uniform)
    return windows


def stretch(frame: np.ndarray) -> np.ndarray:
    """A thermal frame as 8-bit RGB, percentile-stretched so it is visible.

    16-bit thermal and a dark 8-bit frame both look black under a plain cast.
    The 2/98 stretch is for *looking*, and it is not what training sees --
    `aerial.load_image` normalises differently. Nothing here is a measurement.
    """
    if frame.ndim == 3:
        frame = frame[..., 0] if frame.shape[2] == 1 else frame.mean(axis=2)
    lo, hi = np.percentile(frame, [2, 98])
    scaled = np.clip((frame.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    grey = (scaled * 255).astype(np.uint8)
    return np.dstack([grey] * 3)


def draw(frame: np.ndarray, box, mask, present: bool) -> np.ndarray:
    """One tile: the frame, its box, its mask, and what kind of frame it is."""
    import cv2

    canvas = stretch(frame)
    height, width = canvas.shape[:2]
    if mask is not None and np.any(mask):
        area = np.asarray(mask).astype(bool)
        if area.shape != canvas.shape[:2]:
            area = cv2.resize(area.astype(np.uint8), (width, height),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        canvas[area] = (0.45 * canvas[area] + 0.55 * np.array(MASKED)).astype(np.uint8)
    if present and box is not None and np.all(np.isfinite(box)):
        x0, y0, x1, y1 = (int(round(float(v))) for v in box)
        cv2.rectangle(canvas, (x0, y0), (x1, y1),
                      MASKED if mask is not None else PRESENT, 2)
    scale = TILE / max(height, width, 1)
    canvas = cv2.resize(canvas, (int(width * scale), int(height * scale)),
                        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    if not present:
        cv2.rectangle(canvas, (0, 0),
                      (canvas.shape[1] - 1, canvas.shape[0] - 1), ABSENT, 4)
    return canvas


def sheet(sequence, store: Mapping[int, np.ndarray], indices: np.ndarray
          ) -> np.ndarray:
    """A grid of tiles, captioned with what each frame is."""
    import cv2

    tiles = []
    for index in indices:
        index = int(index)
        frame = cv2.imread(str(sequence.frames[index]), cv2.IMREAD_UNCHANGED)
        if frame is None:
            continue
        mask = store.get(index) if store else None
        present = bool(sequence.labels.exist[index])
        tile = draw(frame, sequence.labels.boxes[index], mask, present)
        label = (f"{index}  " + ("mask" if mask is not None else
                                 "box" if present else "YOK"))
        cv2.putText(tile, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(tile, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    MASKED if mask is not None else
                    PRESENT if present else ABSENT, 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return np.zeros((1, 1, 3), np.uint8)

    height = max(t.shape[0] for t in tiles)
    width = max(t.shape[1] for t in tiles)
    rows = []
    for start in range(0, len(tiles), COLUMNS):
        row = np.zeros((height, width * COLUMNS, 3), np.uint8)
        for column, tile in enumerate(tiles[start:start + COLUMNS]):
            row[:tile.shape[0], column * width:column * width + tile.shape[1]] = tile
        rows.append(row)
    return np.vstack(rows)


def render(sequences: SequenceABC, stores: Mapping[str, Mapping],
           out: Path, per_source: int = 4, span: int = 12,
           windows: int = 2, quiet: bool = False) -> dict:
    """Contact sheets for `per_source` sequences of every source, plus an index."""
    import cv2

    from src.training.aerial_video import source_name

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    by_source: dict[str, list] = {}
    for sequence in sequences:
        by_source.setdefault(source_name(sequence), []).append(sequence)

    report: dict[str, list] = {}
    for source, rows in sorted(by_source.items()):
        # The sequences with the most transitions first: a source's value here
        # is the disappearances it contains, and taking the first few by name
        # would usually take the ones with none.
        ranked = sorted(
            rows, key=lambda s: -int(np.count_nonzero(
                np.asarray(s.labels.exist)[1:] != np.asarray(s.labels.exist)[:-1])))
        folder = out / source
        folder.mkdir(parents=True, exist_ok=True)
        for sequence in ranked[:per_source]:
            store = stores.get(sequence.name, {})
            picked = windows_around_gaps(sequence.labels.exist, span, windows)
            for number, indices in enumerate(picked):
                image = sheet(sequence, store, indices)
                if image.size <= 3:
                    continue
                path = folder / f"{sequence.name}_{number:02d}.png"
                cv2.imwrite(str(path), image[..., ::-1])
                report.setdefault(source, []).append({
                    "sequence": sequence.name, "sheet": str(path),
                    "frames": len(sequence),
                    "visible": int(np.count_nonzero(sequence.labels.exist)),
                    "masks": len(store),
                    "shown": [int(i) for i in indices],
                })
            if not quiet:
                print(f"   {sequence.name:<44}{len(sequence):>6} frames"
                      f"{len(store):>7} masks")
    # Where the disappearance supervision actually comes from. Only
    # `vtuav_sequences` reads a real `exist` column; `vtuav_vis_sequences` sets
    # `exist=np.ones` because every extracted frame is a masked one, and
    # `birdsai_sequences` splits a track on its gaps instead of marking them.
    # So a source with no absent frames trains the object-score head on nothing
    # -- which is worth seeing beside the sheets rather than inferred from
    # three function bodies.
    absence = {}
    for source, rows in sorted(by_source.items()):
        frames = sum(len(s) for s in rows)
        visible = sum(int(np.count_nonzero(s.labels.exist)) for s in rows)
        absence[source] = {"sequences": len(rows), "frames": frames,
                           "absent": frames - visible}
    (out / "index.json").write_text(
        json.dumps({"sheets": report, "absence": absence}, indent=2) + "\n")
    if not quiet:
        print(f"\n{'source':<14}{'seqs':>7}{'frames':>10}{'absent':>10}")
        for source, row in absence.items():
            print(f"{source:<14}{row['sequences']:>7}{row['frames']:>10}"
                  f"{row['absent']:>10}")
        if not any(row["absent"] for row in absence.values()):
            print("!! no source here has an absent frame: nothing trains the "
                  "object-score head,\n   which is the head that decides "
                  "whether the target is there at all.")
        total = sum(len(v) for v in report.values())
        print(f"\n{total} sheets -> {out}")
        print("   green box = box only (the loss uses box projection here)")
        print("   yellow    = a real mask, which is what the mask loss needs")
        print("   red frame = `exist` is False; there is nothing to find")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vtuav", type=Path)
    parser.add_argument("--vtuav-vis", type=Path)
    parser.add_argument("--birdsai", type=Path)
    parser.add_argument("--pool", type=Path,
                        help="The teacher-mask pool root, so VTUAV's accepted "
                             "17/25 masks are drawn as masks and not as boxes.")
    parser.add_argument("--min-box-iou", type=float, default=0.80)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=4)
    parser.add_argument("--span", type=int, default=12)
    parser.add_argument("--windows", type=int, default=2)
    args = parser.parse_args()

    from src.training.aerial_video import (birdsai_sequences, empty_stores,
                                           pool_sequence_stores,
                                           vtuav_sequences,
                                           vtuav_vis_sequences)

    sequences, stores = [], {}
    vtuav = []
    if args.vtuav:
        vtuav = vtuav_sequences(args.vtuav, modality="ir")
        sequences += vtuav
    if args.vtuav_vis:
        vis, vis_stores = vtuav_vis_sequences(args.vtuav_vis, modality="ir")
        sequences += vis
        stores.update(vis_stores)
    if args.birdsai:
        sequences += birdsai_sequences(args.birdsai, split="TrainReal")
    if not sequences:
        raise SystemExit(
            "nothing to inspect -- pass at least one of --vtuav, --vtuav-vis "
            "or --birdsai, pointing at the directories notebook 31 staged.")

    merged = empty_stores(sequences)
    if args.pool and vtuav:
        merged.update(pool_sequence_stores(
            args.pool, vtuav, {"vtuav_thermal", "vtuav_lt_thermal"},
            min_box_iou=args.min_box_iou))
    merged.update(stores)
    render(sequences, merged, args.out, per_source=args.per_source,
           span=args.span, windows=args.windows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
