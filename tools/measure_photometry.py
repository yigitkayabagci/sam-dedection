#!/usr/bin/env python3
"""Write `photometry.json` for a run that was made without `--photometry`.

The measurement does not need the tracker: it is a property of the frames the
encoder was given, and those can be re-read afterwards. So a run already on
disk can be diagnosed without paying for it again --

    python3 tools/measure_photometry.py <run folder> --records <the frames>

-- and then `tools/diagnose_break.py <run folder>` has both halves it needs.

It reproduces the pipeline's own pre-encoder path exactly, in this order:
decode unchanged, centre crop (read back from the run's `provenance.json`, so
it is the crop that run used and not one typed again here), resize to the
model's input size with INTER_AREA, then `to_rgb8` -- which on a 16-bit source
is itself a per-frame min/max stretch. Measuring before that chain would
describe the sensor; measuring after it describes what the encoder saw, and
what the encoder saw is the question.

Two numbers per frame, both in grey levels:

    span    the 1st-to-99th percentile width. Small means the encoder was
            handed a tenth of the range it was trained on.
    drift   how far this frame's range has moved from the median of the frames
            the memory bank is holding. Large means the bank's contents were
            encoded under another exposure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def frames_of(folder: Path, pattern: str) -> list[Path]:
    from src.io_utils import list_frame_files

    return [Path(f) for f in list_frame_files(folder, pattern)]


def measure(records: Path, pattern: str, crop: int | None, size: int,
            limit: int = 0) -> list[dict]:
    import cv2

    from src.io_utils import decode_frame, to_rgb8
    from src.pipeline import _center_crop_window, stretch_range

    files = frames_of(records, pattern)
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"No frames matching {pattern!r} in {records}/.")

    window = None
    rows = []
    for index, path in enumerate(files):
        img = decode_frame(path)
        if crop and window is None:
            height, width = img.shape[:2]
            window = _center_crop_window(width, height, crop)
            print(f"[photometry] centre crop {window[2]}x{window[3]} at "
                  f"({window[0]}, {window[1]})")
        if window is not None:
            x0, y0, w, h = window
            img = img[y0:y0 + h, x0:x0 + w]
        if img.shape[0] != size or img.shape[1] != size:
            shrinking = size < img.shape[0] or size < img.shape[1]
            img = cv2.resize(img, (size, size),
                             interpolation=cv2.INTER_AREA if shrinking
                             else cv2.INTER_LINEAR)
        # floor 0: measure, change nothing. The report is the frame's own
        # photometry either way.
        _, report = stretch_range(to_rgb8(img), 0)
        rows.append({"frame": index, **report})
        if index and index % 500 == 0:
            print(f"[photometry] {index}/{len(files)} frames")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path,
                        help="A run folder written by run_records.py.")
    parser.add_argument("--records", type=Path, default=None,
                        help="The frame directory that run read. Defaults to "
                             "the --frames-dir in the folder's provenance.json.")
    parser.add_argument("--pattern", default=None,
                        help="Frame glob. Defaults to the run's own.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Measure only the first N frames (a quick look).")
    args = parser.parse_args(argv)

    provenance = json.loads((args.folder / "provenance.json").read_text())
    command = provenance.get("command", [])

    def flag(name, fallback=None):
        return (command[command.index(name) + 1]
                if name in command and command.index(name) + 1 < len(command)
                else fallback)

    records = args.records or Path(flag("--frames-dir", ""))
    pattern = args.pattern or flag("--frame-pattern", "*.tif*")
    crop = provenance.get("center_crop")
    size = int(provenance.get("image_size", 1024))
    if not records or not Path(records).is_dir():
        raise SystemExit(
            f"{records!r} is not a directory. The run's provenance.json says "
            f"its frames were at {flag('--frames-dir')!r}; pass --records if "
            f"they have moved.")

    print(f"[photometry] {records} ({pattern}), crop {crop}, model input {size}")
    rows = measure(Path(records), pattern, crop, size, args.limit)

    from src.pipeline import PipelineConfig, _write_prefilter

    out = args.folder / "photometry.json"
    _write_prefilter(PipelineConfig(output_path=None, prefilter=0,
                                    prefilter_log=out), rows)
    print(f"[photometry] now run:  python3 tools/diagnose_break.py {args.folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
