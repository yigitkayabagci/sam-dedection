#!/usr/bin/env python3
"""Turn a record's raw thermal frames into 8-bit images the tracker can read.

`frames/<record>/vis/` holds one recording as the sensor wrote it: 16-bit mono
TIFFs, usually; sometimes a headerless `.raw` dump. Nothing downstream reads
those well. `to_rgb8` stretches each frame between its own min and max, so one
hot pixel or one dead one decides the contrast of the whole frame, and the
result flickers from frame to frame. This writes `<record>/prep/` beside `vis/`
-- one 8-bit PNG per source frame, same file names -- so the tracker, the
notebooks and a plain image viewer all see the same picture.

    python tools/prep_thermal.py /media/sedatc/SSD_disk/yigit/frames/record_s40
    python tools/prep_thermal.py /media/sedatc/SSD_disk/yigit/frames    # every record

**The base step is a percentile clip.** The 1st and 99th percentile of the
frame become 0 and 255, and everything outside is saturated. Two percent of
the pixels are given up on purpose: those are the hot and dead pixels and the
one sun-lit roof, which is exactly what a min/max stretch spends its range on.
`--lo` / `--hi` move the cut.

**Where it goes past that, and why each step is there:**

`--scope ema` (default)  The clip window follows the sequence instead of
    being re-picked on every frame. Each frame's own percentiles are blended
    into a running pair (`--ema 0.9` keeps 90 % of the previous window, a time
    constant of about ten frames), so a target crossing a hot patch does not
    make the whole frame breathe. This is the "percentile normalisation +
    temporal EMA" arm `docs/thermal_contrast_tracking_gelisim_raporu_tr.md`
    §12 lists as the thing to try instead of per-frame contrast, and it is
    the reason this is a script rather than a `cv2.normalize` call. Two
    guards keep the smoothing from becoming the problem. A plain EMA lags a
    steady drift by `ema / (1 - ema)` frames' worth of it -- measured on a
    synthetic ramp of 4 % of the window per frame, that put 27 % of every
    frame into saturation -- so the window is never let more than `--slack`
    of its width away from the frame's own percentiles (0.1: the lag is
    bounded, the saturation with it, and a stable scene never touches the
    bound). And a jump larger than `--reset` of the width (a scene cut, a
    camera re-calibration -- thermal cores do this every few minutes) snaps
    the window to the new frame rather than fading over for ten frames; the
    report says on which frames that happened.
`--scope frame`  Every frame on its own percentiles. What the request asked
    for literally; kept as the comparison arm.
`--scope global`  One window for the whole record, from the histogram of every
    frame (two passes over the files). No flicker at all, and no adaptation
    either: a record that pans from cold sky to warm ground spends its range
    on both.
`--median 3`  A median filter on the raw values before anything else, for
    fixed-pattern speckle. Off by default because it blurs a five-pixel target
    as readily as it removes a hot pixel; the percentile clip already keeps
    isolated pixels from setting the window.
`--clahe 2.0`  Contrast-limited adaptive histogram equalisation on the 8-bit
    result. It lifts local contrast where the scene is flat, which is the
    thermal failure mode, but it is off by default: the encoder was trained
    on plain frames, and the same document is explicit that CLAHE goes into
    an A/B arm rather than into the deployment path. With `--clahe-grid 1` it
    is a global plateau equalisation, the AGC most thermal cores ship.
`--gamma`, `--invert`  A transfer curve and a polarity flip. White-hot and
    black-hot are the same scene, and the pools have both.

Every frame's raw percentiles, the window it was actually mapped through, and
how much of it saturated go to `prep/prep.json`, with the parameters, so a
number read off a `prep/` folder months later can be traced to the raw values
that produced it. `--measure` writes the report and nothing else.
`prep/preview.png` puts the current min/max stretch beside the result on a
few frames, so the difference is looked at rather than argued.

The percentiles come off a histogram (`cv2.calcHist`, 65 536 bins on 16-bit
data) rather than `np.percentile`, for the reason `src/pipeline.photometry`
gives: on a 1280x768 uint16 frame the histogram pair is ~1 ms and the sort
~30 ms. Timed on synthetic frames on the machine this was written on; the
number in `prep.json` is the one that counts.

A headerless dump needs its shape: `--raw-shape 1280 768 --raw-dtype '<u2'`.
When the file size is exactly one of the usual thermal sizes at 16 bits it is
inferred and printed; otherwise the script stops and says what it would need.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Files decoded by an image codec. Everything else that is not `.npy` is read
# as a bare dump and needs `RawSpec`.
IMAGE_SUFFIXES = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
SOURCE_PATTERNS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.bmp",
                   "*.npy", "*.raw", "*.bin")
# Frame sizes a headerless 16-bit dump is matched against, by byte count. The
# first is the recordings' own size (CLAUDE.md); the rest are the common cores.
KNOWN_SHAPES = ((1280, 768), (1280, 1024), (2048, 1536), (1024, 768), (640, 512),
                (640, 480), (336, 256), (320, 256), (320, 240), (160, 120))
SCOPES = ("ema", "frame", "global")
FLOAT_SAMPLES = 4096        # pixels a float frame contributes to a global window
PREVIEW_HALF = 480          # width of each half of a preview row, in pixels


@dataclass(frozen=True)
class RawSpec:
    """How to read a headerless dump: `width` x `height` values of `dtype`,
    starting `offset` bytes in. With no offset the header, if any, is assumed
    to come first and be whatever the file holds beyond the frame."""

    width: int
    height: int
    dtype: str = "<u2"
    offset: int | None = None


@dataclass(frozen=True)
class Params:
    """What the mapping does to a frame. Percentiles are in percent."""

    lo: float = 1.0
    hi: float = 99.0
    scope: str = "ema"
    ema: float = 0.9
    slack: float = 0.1
    reset: float = 0.5
    min_span: float = 0.0
    gamma: float = 1.0
    invert: bool = False
    clahe: float = 0.0
    clahe_grid: int = 8
    median: int = 0

    def __post_init__(self):
        if not 0 <= self.lo < self.hi <= 100:
            raise ValueError(f"percentiles must satisfy 0 <= lo < hi <= 100, got {self.lo}, {self.hi}")
        if self.scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}, got {self.scope!r}")
        if not 0 <= self.ema < 1:
            raise ValueError(f"--ema is the weight kept from the previous window, 0 <= ema < 1; got {self.ema}")
        if self.slack < 0 or self.reset <= 0:
            raise ValueError("--slack is a non-negative share of the window and --reset a positive one")
        if self.median not in (0, 3, 5):
            raise ValueError(f"--median is 0, 3 or 5 (what cv2 filters on 16-bit data); got {self.median}")
        if self.gamma <= 0:
            raise ValueError(f"--gamma must be positive, got {self.gamma}")
        if self.clahe < 0 or self.clahe_grid < 1:
            raise ValueError("--clahe is a non-negative clip limit and --clahe-grid a positive tile count")


# --------------------------------------------------------------------------- #
# Finding the frames
# --------------------------------------------------------------------------- #

def has_frames(folder: Path) -> bool:
    return any(any(True for _ in folder.glob(pattern)) for pattern in SOURCE_PATTERNS)


def find_records(path: str | Path, src_name: str = "vis") -> list[tuple[Path, Path]]:
    """`(record, source)` pairs for a path that is a record, its `vis/`, a
    folder of frames, or a root holding many records.

    The output folder is always `<record>/<out-name>/`, so what has to be
    settled here is which folder is the record. A folder with frames straight
    in it is its own record, and `prep/` goes inside it.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"[prep] {path} is not a directory.")
    if (path / src_name).is_dir():
        return [(path, path / src_name)]
    if path.name == src_name:
        return [(path.parent, path)]
    if has_frames(path):
        return [(path, path)]
    found = sorted((child, child / src_name) for child in path.iterdir()
                   if child.is_dir() and (child / src_name).is_dir())
    if found:
        return found
    raise SystemExit(f"[prep] {path}: no '{src_name}/' in it, no frames in it, and no "
                     f"'<record>/{src_name}/' below it.")


def source_files(folder: Path, pattern: str) -> tuple[list[Path], str]:
    """The frames in `folder`, sorted by name, and the pattern that found them.

    `--pattern` is tried first; when it matches nothing the other suffixes
    are tried in turn and the one used is printed, so a folder of PNGs does
    not fail in a way that reads like an empty directory.
    """
    files = sorted(p for p in folder.glob(pattern) if p.is_file())
    if files:
        return files, pattern
    for candidate in SOURCE_PATTERNS:
        if candidate == pattern:
            continue
        files = sorted(p for p in folder.glob(candidate) if p.is_file())
        if files:
            print(f"[prep] nothing matches {pattern!r} in {folder}; using {candidate!r} "
                  f"({len(files)} files)")
            return files, candidate
    raise SystemExit(f"[prep] no frames in {folder}: nothing matches {pattern!r} or any of "
                     f"{', '.join(SOURCE_PATTERNS)}.")


# --------------------------------------------------------------------------- #
# Reading a frame
# --------------------------------------------------------------------------- #

def infer_raw_spec(path: Path, dtype: str = "<u2") -> RawSpec | None:
    """A `RawSpec` for a dump whose size is exactly one known frame, else None."""
    size = path.stat().st_size
    itemsize = np.dtype(dtype).itemsize
    fits = [(w, h) for w, h in KNOWN_SHAPES if w * h * itemsize == size]
    if len(fits) != 1:
        return None
    return RawSpec(fits[0][0], fits[0][1], dtype, 0)


def read_dump(path: Path, spec: RawSpec) -> np.ndarray:
    dt = np.dtype(spec.dtype)
    count = spec.width * spec.height
    need = count * dt.itemsize
    size = path.stat().st_size
    offset = size - need if spec.offset is None else spec.offset
    if offset < 0 or size < offset + need:
        raise ValueError(f"{path.name}: {size} bytes cannot hold {spec.width}x{spec.height} "
                         f"{dt.name} ({need} bytes) at offset {offset}")
    data = np.fromfile(path, dtype=dt, count=count, offset=offset)
    if not dt.isnative:
        data = data.astype(dt.newbyteorder("="))
    return data.reshape(spec.height, spec.width)


def as_mono(img: np.ndarray) -> np.ndarray:
    """One channel. A replicated three-channel thermal frame comes out as its
    grey; a real colour image is converted, which is the best a thermal
    pipeline can do with it."""
    import cv2

    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 1:
        return img[..., 0]
    if img.ndim == 3 and img.shape[2] in (3, 4):
        return cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_BGR2GRAY)
    raise ValueError(f"cannot read a {img.shape} array as a frame")


def read_frame(path: Path, raw: RawSpec | None = None) -> np.ndarray:
    """A frame as a 2-D array in its own dtype: image files through cv2,
    `.npy` through numpy, anything else as a dump described by `raw`."""
    import cv2

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return as_mono(np.load(path))
    if suffix in IMAGE_SUFFIXES:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"could not decode {path}")
        return as_mono(img)
    if raw is None:
        raise SystemExit(f"[prep] {path.name} is a bare dump and its size matches no known frame: "
                         f"pass --raw-shape W H (and --raw-dtype, default '<u2').")
    return read_dump(path, raw)


# --------------------------------------------------------------------------- #
# Percentiles and windows
# --------------------------------------------------------------------------- #

def _levels(img: np.ndarray) -> int | None:
    return {np.dtype(np.uint8): 256, np.dtype(np.uint16): 65536}.get(img.dtype)


def _from_counts(counts: np.ndarray, qs: Sequence[float]) -> list[float]:
    """The order statistics for fractions `qs` from a histogram: the lowest
    level at which the running count reaches `q` of the pixels."""
    total = float(counts.sum())
    if total <= 0:
        return [0.0 for _ in qs]
    running = np.cumsum(counts)
    return [float(np.searchsorted(running, max(q * total, 1.0))) for q in qs]


def percentiles_of(img: np.ndarray, qs: Sequence[float]) -> list[float]:
    """Percentiles as fractions in [0, 1]. Histogram for 8/16-bit data, a
    sort for anything else."""
    import cv2

    levels = _levels(img)
    if levels is not None:
        counts = cv2.calcHist([np.ascontiguousarray(img)], [0], None, [levels], [0, levels]).ravel()
        return _from_counts(counts, qs)
    return [float(v) for v in np.percentile(img.ravel(), [q * 100 for q in qs], method="lower")]


class Histogram:
    """Every frame's values, merged, for `--scope global`. Integer data adds
    up its histograms; float data keeps a fixed sample of each frame so the
    memory stays bounded whatever the record's length."""

    def __init__(self):
        self.counts = None
        self.samples: list[np.ndarray] = []

    def add(self, img: np.ndarray) -> None:
        import cv2

        levels = _levels(img)
        if levels is None:
            flat = img.ravel()
            self.samples.append(flat[:: max(1, flat.size // FLOAT_SAMPLES)].astype(np.float64))
            return
        counts = cv2.calcHist([np.ascontiguousarray(img)], [0], None, [levels], [0, levels]).ravel()
        self.counts = counts if self.counts is None else self.counts + counts

    def percentiles(self, qs: Sequence[float]) -> list[float]:
        if self.counts is not None:
            return _from_counts(self.counts, qs)
        if not self.samples:
            return [0.0 for _ in qs]
        pool = np.concatenate(self.samples)
        return [float(v) for v in np.percentile(pool, [q * 100 for q in qs], method="lower")]


class Window:
    """The clip window a sequence is mapped through, and how it follows it.

    `update` blends each frame's percentiles into the running pair with weight
    `ema` on the old window. Either edge is then held within `slack` of the
    frame's own width from the frame's own value, so a steady drift cannot
    open a lag that saturates one end -- the bound only engages when the EMA
    has fallen that far behind, so a stable scene is smoothed exactly as an
    EMA would, and `|hi - raw_hi| <= slack * raw_span` holds on every row of
    the report.
    An edge that moved by more than `reset` of the width snaps the whole
    window to the frame, which is what a scene cut or a core re-calibration
    needs. `ema=0` is per-frame.
    """

    def __init__(self, ema: float, reset: float, slack: float = 0.1):
        self.ema = ema
        self.reset = reset
        self.slack = slack
        self.lo: float | None = None
        self.hi: float | None = None

    def update(self, lo: float, hi: float) -> tuple[float, float, bool]:
        if self.lo is None or self.ema <= 0:
            self.lo, self.hi = lo, hi
            return lo, hi, False
        width = max(self.hi - self.lo, 1.0)
        if abs(lo - self.lo) > self.reset * width or abs(hi - self.hi) > self.reset * width:
            self.lo, self.hi = lo, hi
            return lo, hi, True
        room = self.slack * max(hi - lo, 1.0)
        self.lo = min(max(self.ema * self.lo + (1 - self.ema) * lo, lo - room), lo + room)
        self.hi = min(max(self.ema * self.hi + (1 - self.ema) * hi, hi - room), hi + room)
        return self.lo, self.hi, False


def widen(lo: float, hi: float, min_span: float) -> tuple[float, float]:
    """A window at least `min_span` wide, so a flat frame (lens cap, uniform
    sky) is not stretched into pure noise."""
    if hi - lo >= min_span:
        return lo, hi
    centre = (lo + hi) / 2
    return centre - min_span / 2, centre + min_span / 2


# --------------------------------------------------------------------------- #
# The mapping itself
# --------------------------------------------------------------------------- #

def lookup_table(lo: float, hi: float, params: Params, levels: int) -> np.ndarray:
    x = np.arange(levels, dtype=np.float32)
    y = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    if params.gamma != 1.0:
        y = y ** params.gamma
    if params.invert:
        y = 1.0 - y
    return np.rint(y * 255.0).astype(np.uint8)


def to_uint8(img: np.ndarray, lo: float, hi: float, params: Params) -> np.ndarray:
    """`img` through the window `[lo, hi]` -> 0..255. A lookup table for 8-
    and 16-bit data (`cv2.LUT` for 8, a take for 16), arithmetic otherwise."""
    import cv2

    if img.dtype == np.uint8:
        return cv2.LUT(np.ascontiguousarray(img), lookup_table(lo, hi, params, 256))
    if img.dtype == np.uint16:
        return lookup_table(lo, hi, params, 65536)[img]
    y = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    if params.gamma != 1.0:
        y = y ** params.gamma
    if params.invert:
        y = 1.0 - y
    return np.rint(y * 255.0).astype(np.uint8)


def denoise(img: np.ndarray, ksize: int) -> np.ndarray:
    import cv2

    if img.dtype not in (np.uint8, np.uint16, np.float32):
        img = img.astype(np.float32)
    return cv2.medianBlur(np.ascontiguousarray(img), ksize)


def output_stats(out8: np.ndarray) -> dict:
    """Where the 8-bit result landed: its 1/99 percentiles and the share of
    pixels pinned at either end."""
    import cv2

    counts = cv2.calcHist([out8], [0], None, [256], [0, 256]).ravel()
    total = float(counts.sum()) or 1.0
    p1, p99 = _from_counts(counts, (0.01, 0.99))
    return {"out_p1": p1, "out_p99": p99, "out_span": round(p99 - p1, 1),
            "saturated": round(float(counts[0] + counts[255]) / total, 4)}


# --------------------------------------------------------------------------- #
# Running a record
# --------------------------------------------------------------------------- #

def prefetch(pool: ThreadPoolExecutor, fn: Callable, items: Sequence, depth: int) -> Iterator:
    """`(item, fn(item))` in order, with up to `depth` calls in flight. cv2
    releases the GIL in `imread`, so the decode runs beside the mapping, but
    submitting every file at once would hold every decoded frame in memory."""
    queue: deque = deque()
    it = iter(items)
    for item in it:
        queue.append((item, pool.submit(fn, item)))
        if len(queue) >= depth:
            break
    while queue:
        item, future = queue.popleft()
        try:
            nxt = next(it)
        except StopIteration:
            pass
        else:
            queue.append((nxt, pool.submit(fn, nxt)))
        yield item, future.result()


def write_image(dst: Path, out8: np.ndarray, ext: str, quality: int) -> None:
    import cv2

    flags: list[int] = []
    if ext in ("jpg", "jpeg"):
        flags = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    elif ext == "png":
        flags = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    if not cv2.imwrite(str(dst), out8, flags):
        raise OSError(f"could not write {dst}")


def already_done(out: Path, params: Params, files: Sequence[Path], ext: str, limit: int) -> bool:
    """True when `out/prep.json` describes exactly this run and every output
    is on disk, so a re-run without `--overwrite` is a no-op."""
    report = out / "prep.json"
    if not report.is_file():
        return False
    try:
        body = json.loads(report.read_text())
    except (OSError, ValueError):
        return False
    if body.get("params") != asdict(params) or body.get("ext") != ext or body.get("limit", 0) != limit:
        return False
    if body.get("frames") != len(files):
        return False
    return all((out / f"{p.stem}.{ext}").is_file() for p in files)


def _preview_indices(count: int, wanted: int) -> set[int]:
    if wanted <= 0 or count == 0:
        return set()
    return {int(i) for i in np.linspace(0, count - 1, min(wanted, count)).round()}


def _preview_row(img: np.ndarray, out8: np.ndarray, name: str, lo: float, hi: float) -> np.ndarray:
    """Left: today's path, a min/max stretch of the raw frame. Right: the
    result. Captioned with the raw window the right half was mapped through."""
    import cv2

    naive = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    scale = PREVIEW_HALF / max(img.shape[1], 1)
    size = (PREVIEW_HALF, max(int(round(img.shape[0] * scale)), 1))
    halves = [cv2.resize(x, size, interpolation=cv2.INTER_AREA) for x in (naive, out8)]
    row = cv2.cvtColor(np.concatenate(halves, axis=1), cv2.COLOR_GRAY2BGR)
    strip = np.zeros((28, row.shape[1], 3), dtype=np.uint8)
    caption = f"{name}   min/max  |  clip [{lo:.0f}, {hi:.0f}]"
    cv2.putText(strip, caption, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([row, strip], axis=0)


def prep_record(record: Path, source: Path, out: Path, params: Params, *, pattern: str = "*.tif*",
                raw: RawSpec | None = None, ext: str = "png", quality: int = 95, limit: int = 0,
                overwrite: bool = False, workers: int | None = None, preview: int = 6,
                measure: bool = False, progress: bool = True) -> dict:
    """Map every frame of one record and write the report. Returns the report."""
    import cv2
    from tqdm import tqdm

    files, pattern = source_files(source, pattern)
    if limit:
        files = files[:limit]
    if raw is None and files[0].suffix.lower() not in IMAGE_SUFFIXES + (".npy",):
        raw = infer_raw_spec(files[0])
        if raw is not None:
            print(f"[prep] {files[0].name}: {files[0].stat().st_size} bytes is exactly "
                  f"{raw.width}x{raw.height} uint16, reading every dump that way")
    if not measure and not overwrite and already_done(out, params, files, ext, limit):
        print(f"[prep] {record.name}: {out} already holds these {len(files)} frames with these "
              f"parameters; pass --overwrite to redo them")
        return json.loads((out / "prep.json").read_text())

    workers = workers or min(8, os.cpu_count() or 1)
    depth = max(2, workers * 2)
    qs = (params.lo / 100.0, params.hi / 100.0)
    reader = lambda p: read_frame(p, raw)  # noqa: E731
    label = f"{record.name}/{source.name}" if source != record else record.name
    started = time.perf_counter()

    fixed: tuple[float, float] | None = None
    if params.scope == "global":
        merged = Histogram()
        with ThreadPoolExecutor(workers) as pool:
            for _, img in tqdm(prefetch(pool, reader, files, depth), total=len(files),
                               desc=f"[prep] {label} pass 1/2 histogram", unit="frame",
                               disable=not progress):
                merged.add(denoise(img, params.median) if params.median else img)
        lo, hi = merged.percentiles(qs)
        fixed = widen(lo, hi, params.min_span)

    if not measure:
        out.mkdir(parents=True, exist_ok=True)
    window = Window(params.ema if params.scope == "ema" else 0.0, params.reset, params.slack)
    clahe = (cv2.createCLAHE(clipLimit=params.clahe, tileGridSize=(params.clahe_grid,) * 2)
             if params.clahe > 0 else None)
    wanted = _preview_indices(len(files), 0 if measure else preview)
    panels: list[np.ndarray] = []
    rows: list[dict] = []
    shape: tuple[int, int] | None = None
    dtype: str | None = None
    writes: deque = deque()

    with ThreadPoolExecutor(workers) as pool:
        desc = f"[prep] {label}" + (" pass 2/2 write" if fixed else "")
        for index, (path, img) in enumerate(tqdm(prefetch(pool, reader, files, depth),
                                                 total=len(files), desc=desc, unit="frame",
                                                 disable=not progress)):
            if params.median:
                img = denoise(img, params.median)
            if shape is None:
                shape, dtype = (int(img.shape[0]), int(img.shape[1])), str(img.dtype)
            lo_f, hi_f = percentiles_of(img, qs)
            if fixed is not None:
                lo, hi, reset = fixed[0], fixed[1], False
            else:
                lo, hi, reset = window.update(lo_f, hi_f)
                lo, hi = widen(lo, hi, params.min_span)
            out8 = to_uint8(img, lo, hi, params)
            if clahe is not None:
                out8 = clahe.apply(out8)
            row = {"frame": path.name, "raw_lo": lo_f, "raw_hi": hi_f, "raw_span": round(hi_f - lo_f, 1),
                   "lo": round(lo, 2), "hi": round(hi, 2), "reset": reset, **output_stats(out8)}
            rows.append(row)
            if not measure:
                writes.append(pool.submit(write_image, out / f"{path.stem}.{ext}", out8, ext, quality))
                while len(writes) > depth:
                    writes.popleft().result()
            if index in wanted:
                panels.append(_preview_row(img, out8, path.name, lo, hi))
        for future in writes:
            future.result()

    seconds = time.perf_counter() - started
    raw_spans = np.array([r["raw_span"] for r in rows], dtype=np.float64)
    report = {
        "record": str(record), "source": str(source), "out": str(out), "pattern": pattern,
        "reader": "dump" if raw is not None and files[0].suffix.lower() not in IMAGE_SUFFIXES + (".npy",)
        else "image",
        "raw": asdict(raw) if raw is not None else None,
        "params": asdict(params), "ext": ext, "limit": limit, "measure_only": measure,
        "frames": len(rows), "shape": list(shape) if shape else None, "dtype": dtype,
        "summary": {
            "raw_lo_min": float(min(r["raw_lo"] for r in rows)),
            "raw_hi_max": float(max(r["raw_hi"] for r in rows)),
            "raw_span_median": float(np.median(raw_spans)),
            "raw_span_min": float(raw_spans.min()),
            "raw_span_max": float(raw_spans.max()),
            "window": [fixed[0], fixed[1]] if fixed else None,
            "out_span_median": float(np.median([r["out_span"] for r in rows])),
            "saturated_median": float(np.median([r["saturated"] for r in rows])),
            "saturated_max": float(max(r["saturated"] for r in rows)),
            "resets": int(sum(r["reset"] for r in rows)),
            "reset_frames": [r["frame"] for r in rows if r["reset"]][:50],
            "seconds": round(seconds, 2),
            "ms_per_frame": round(1000.0 * seconds / max(len(rows), 1), 2),
        },
        "rows": rows,
    }
    if not measure:
        (out / "prep.json").write_text(json.dumps(report, indent=1))
        if panels:
            cv2.imwrite(str(out / "preview.png"), np.concatenate(panels, axis=0))
    return report


def describe(report: dict) -> str:
    s = report["summary"]
    shape = report["shape"] or ["?", "?"]
    lines = [
        f"[prep] {Path(report['record']).name}: {report['frames']} frames, {shape[1]}x{shape[0]} "
        f"{report['dtype']}, read as {report['reader']}, {s['ms_per_frame']} ms/frame",
        f"[prep]   raw {report['params']['lo']:g}-{report['params']['hi']:g} % window: "
        f"levels {s['raw_lo_min']:.0f}..{s['raw_hi_max']:.0f} over the record, span median "
        f"{s['raw_span_median']:.0f} (min {s['raw_span_min']:.0f}, max {s['raw_span_max']:.0f})",
        f"[prep]   output: 1-99 % span median {s['out_span_median']:.0f} of 255, saturated median "
        f"{100 * s['saturated_median']:.2f} % (max {100 * s['saturated_max']:.2f} %)",
    ]
    if report["params"]["scope"] == "ema":
        where = ", ".join(s["reset_frames"][:5]) + (" ..." if s["resets"] > 5 else "")
        lines.append(f"[prep]   window resets: {s['resets']}" + (f" ({where})" if s["resets"] else ""))
    elif s.get("window"):
        lines.append(f"[prep]   one window for the record: [{s['window'][0]:.0f}, {s['window'][1]:.0f}]")
    if report["measure_only"]:
        lines.append("[prep]   --measure: nothing written")
    else:
        lines.append(f"[prep]   -> {report['out']}  (prep.json, preview.png)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", nargs="+", help="A record, its vis/ folder, a folder of frames, "
                                          "or a root of records.")
    p.add_argument("--src-name", default="vis", help="The raw folder's name inside a record.")
    p.add_argument("--out-name", default="prep", help="The output folder's name inside a record.")
    p.add_argument("--out", default=None, help="Write here instead of <record>/<out-name>/ "
                                               "(with several records: <out>/<record name>/).")
    p.add_argument("--pattern", default="*.tif*", help="Frame glob inside the raw folder.")
    p.add_argument("--lo", type=float, default=1.0, help="Lower percentile, in percent (0.01 is read as 1).")
    p.add_argument("--hi", type=float, default=99.0, help="Upper percentile, in percent (0.99 is read as 99).")
    p.add_argument("--scope", choices=SCOPES, default="ema",
                   help="ema: window follows the sequence; frame: each frame alone; global: one window.")
    p.add_argument("--ema", type=float, default=0.9, help="Weight kept from the previous window per frame.")
    p.add_argument("--slack", type=float, default=0.1,
                   help="How far, as a share of its width, the window may lag a frame's own percentiles.")
    p.add_argument("--reset", type=float, default=0.5,
                   help="Snap the window to the frame when an edge moves more than this share of its width.")
    p.add_argument("--min-span", type=float, default=0.0, help="Narrowest window, in raw levels.")
    p.add_argument("--gamma", type=float, default=1.0, help="Transfer exponent on the 0..1 result.")
    p.add_argument("--invert", action="store_true", help="Flip polarity (white-hot <-> black-hot).")
    p.add_argument("--clahe", type=float, default=0.0, help="CLAHE clip limit on the result; 0 = off.")
    p.add_argument("--clahe-grid", type=int, default=8, help="CLAHE tiles per side; 1 = global.")
    p.add_argument("--median", type=int, default=0, choices=(0, 3, 5), help="Median filter on the raw values.")
    p.add_argument("--raw-shape", type=int, nargs=2, metavar=("W", "H"), default=None,
                   help="Frame size of a headerless dump.")
    p.add_argument("--raw-dtype", default="<u2", help="numpy dtype of a dump ('<u2', '>u2', 'u1', 'f4').")
    p.add_argument("--raw-offset", type=int, default=None, help="Header bytes before the frame in a dump.")
    p.add_argument("--ext", default="png", help="Output format: png (lossless, default), jpg, bmp, tif.")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality when --ext jpg.")
    p.add_argument("--limit", type=int, default=0, help="Only the first N frames (a quick look).")
    p.add_argument("--overwrite", action="store_true", help="Redo a record already prepared this way.")
    p.add_argument("--workers", type=int, default=None, help="Decode/write threads (default: up to 8).")
    p.add_argument("--preview", type=int, default=6, help="Frames in preview.png; 0 = none.")
    p.add_argument("--measure", action="store_true", help="Report only, write nothing.")
    p.add_argument("--quiet", action="store_true", help="No progress bar.")
    return p.parse_args(argv)


def params_from(args: argparse.Namespace) -> Params:
    lo, hi = args.lo, args.hi
    if 0 < lo < 1 and 0 < hi <= 1:
        print(f"[prep] --lo {lo:g} --hi {hi:g} read as fractions: using {100 * lo:g} % and {100 * hi:g} %")
        lo, hi = 100 * lo, 100 * hi
    try:
        return Params(lo=lo, hi=hi, scope=args.scope, ema=args.ema, slack=args.slack, reset=args.reset,
                      min_span=args.min_span, gamma=args.gamma, invert=args.invert,
                      clahe=args.clahe, clahe_grid=args.clahe_grid, median=args.median)
    except ValueError as err:
        raise SystemExit(f"[prep] {err}") from None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    params = params_from(args)
    raw = (RawSpec(args.raw_shape[0], args.raw_shape[1], args.raw_dtype, args.raw_offset)
           if args.raw_shape else None)
    ext = args.ext.lower().lstrip(".")

    records: list[tuple[Path, Path]] = []
    for given in args.path:
        records.extend(find_records(given, args.src_name))
    if not records:
        raise SystemExit("[prep] nothing to do")
    explicit = Path(args.out).expanduser().resolve() if args.out else None
    print(f"[prep] {len(records)} record(s); scope {params.scope}, window {params.lo:g}-{params.hi:g} %"
          + (f", ema {params.ema:g}" if params.scope == "ema" else "")
          + (", clahe" if params.clahe else "") + (", invert" if params.invert else ""))

    for record, source in records:
        if explicit is None:
            out = record / args.out_name
        else:
            out = explicit if len(records) == 1 else explicit / record.name
        report = prep_record(record, source, out, params, pattern=args.pattern, raw=raw, ext=ext,
                             quality=args.quality, limit=args.limit, overwrite=args.overwrite,
                             workers=args.workers, preview=args.preview, measure=args.measure,
                             progress=not args.quiet)
        print(describe(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
