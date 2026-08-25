#!/usr/bin/env python3
"""Read AeroVIS out of its 12.6 GiB Drive zip without downloading the zip.

`docs/rgb_aerial_kaynaklar.md` §3 counted the archive's *entries*. That answered
"is the release real" but not the question stage B actually cares about: does
AeroVIS hand over the `(image, box prompt, mask)` triple, or only the mask? This
opens the annotations, decodes RLE against real frames, and measures it.

**Why ranged GET and not a download.** The archive is 13 563 857 996 B and the
only useful part for an inspection is `aero_vis.json` (332 MB) plus a handful of
JPEGs. Drive serves `Range:` requests with 206 -- so a file-like object over
ranged GETs lets `zipfile` seek the central directory and inflate single members,
and the whole inspection costs ~400 MB instead of 12.6 GiB. Two Drive quirks are
worked around here: `HEAD` answers `text/html` with `Content-Length: 0`, so the
size comes from a one-byte ranged GET's `Content-Range`; and any request can fail
transiently, so every fetch retries with backoff.

The measurement it exists to reproduce (2026-08, full annotation file):

    triples (box AND mask, same frame, same instance) : 1 378 603
    box without mask                                  : 149
    mask without box                                  : 0
    boxes are NOT the mask envelope                   : 3.8 % match exactly

That last line is the load-bearing one. AeroVIS ships the source datasets'
hand-drawn boxes, not boxes recovered from the masks it generated, so a box
prompt is an independent signal rather than the answer leaking into the prompt.

Usage:
    python tools/inspect_aerovis.py --stats
    python tools/inspect_aerovis.py --render out/ --per-source 2
    python tools/inspect_aerovis.py --json /local/aero_vis.json --stats  # already have it
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import random
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

DRIVE_ID = "1DMLagGZMPntrvxk5W0PsaIoybsE7WX56"
URL = (f"https://drive.usercontent.google.com/download"
       f"?id={DRIVE_ID}&export=download&confirm=t")
JSON_MEMBER = "AeroVIS/aero_vis.json"
FRAME_MEMBER = "AeroVIS/sequences/{}"

PALETTE = ((255, 80, 80), (80, 200, 255), (120, 255, 120), (255, 220, 60),
           (230, 110, 255), (255, 160, 50), (90, 255, 220), (200, 160, 255),
           (255, 120, 180), (160, 255, 60))


class RangedHTTPFile:
    """Seekable file over HTTP ranged GETs, enough of one for `zipfile`.

    `zipfile` only needs read/seek/tell/seekable, so this is a plain object
    rather than a `RawIOBase` -- wrapping a `RawIOBase` in `BufferedReader`
    would demand `readinto`, and the buffering here is already explicit.
    """

    def __init__(self, url: str = URL, chunk: int = 1 << 20) -> None:
        self.url, self._pos, self._chunk = url, 0, chunk
        self._buf, self._buf_start = b"", -1
        self.size = self._probe_size()

    def _get(self, start: int, end: int) -> bytes:
        """Bytes [start, end] inclusive, with backoff -- Drive fails at random."""
        end = min(end, self.size - 1)
        last: Exception | None = None
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except Exception as exc:            # noqa: BLE001 -- retried below
                last = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"ranged GET {start}-{end} failed") from last

    def _probe_size(self) -> int:
        """Drive's HEAD lies (text/html, length 0); Content-Range does not."""
        last: Exception | None = None
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    self.url, headers={"Range": "bytes=0-0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    rng = r.headers.get("Content-Range", "")
                    r.read()
                    if "/" in rng:
                        return int(rng.rsplit("/", 1)[1])
                    last = RuntimeError(f"no Content-Range: {rng!r}")
            except Exception as exc:            # noqa: BLE001 -- retried below
                last = exc
            time.sleep(2 ** attempt)
        raise RuntimeError("could not size the archive") from last

    # -- file protocol ----------------------------------------------------
    def readable(self) -> bool: return True
    def writable(self) -> bool: return False
    def seekable(self) -> bool: return True
    def tell(self) -> int: return self._pos
    def close(self) -> None: pass

    @property
    def closed(self) -> bool: return False

    def seek(self, off: int, whence: int = 0) -> int:
        self._pos = (off if whence == 0
                     else self._pos + off if whence == 1
                     else self.size + off)
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self._pos
        n = min(n, self.size - self._pos)
        if n <= 0:
            return b""
        if n > self._chunk:                     # bulk: straight through
            data = self._get(self._pos, self._pos + n - 1)
            self._pos += len(data)
            return data
        if not (self._buf_start <= self._pos
                and self._pos + n <= self._buf_start + len(self._buf)):
            self._buf = self._get(self._pos, self._pos + max(self._chunk, n) - 1)
            self._buf_start = self._pos
        off = self._pos - self._buf_start
        out = self._buf[off:off + n]
        self._pos += len(out)
        return out


def open_archive(url: str = URL) -> zipfile.ZipFile:
    return zipfile.ZipFile(RangedHTTPFile(url))


def load_annotations(zf: zipfile.ZipFile | None, cache: Path | None) -> dict:
    """Annotations from a local copy if there is one, else out of the archive."""
    if cache and cache.exists():
        return json.loads(cache.read_text())
    if zf is None:
        raise SystemExit("no --json cache and no archive access")
    info = zf.getinfo(JSON_MEMBER)
    print(f"  fetching {JSON_MEMBER}: {info.compress_size:,} B compressed "
          f"-> {info.file_size:,} B", file=sys.stderr)
    with zf.open(info) as f:
        raw = f.read()
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(raw)
    return json.loads(raw)


def _tracks(data: dict) -> list[dict]:
    """Real tracks: `iscrowd=1` entries are ignore regions with no boxes."""
    return [a for a in data["annotations"] if a.get("iscrowd") != 1]


def stats(data: dict, sample: int = 400) -> str:
    cats = {c["id"]: c["name"] for c in data["categories"]}
    vids = {v["id"]: v for v in data["videos"]}
    real = _tracks(data)
    crowd = len(data["annotations"]) - len(real)

    both = box_only = mask_only = 0
    per_src: collections.Counter[str] = collections.Counter()
    per_cat: collections.Counter[str] = collections.Counter()
    per_cat_tracks: collections.Counter[str] = collections.Counter()
    sides: list[float] = []
    per_frame: collections.Counter[tuple[int, int]] = collections.Counter()
    lengths: list[int] = []

    for ann in real:
        name = cats[ann["category_id"]]
        src = vids[ann["video_id"]]["source"]
        per_cat_tracks[name] += 1
        n = 0
        for fi, (box, seg) in enumerate(zip(ann["bboxes"], ann["segmentations"])):
            hb, hs = box is not None, seg is not None
            if hb and hs:
                both += 1
                n += 1
                per_src[src] += 1
                per_cat[name] += 1
                sides.append((float(box[2]) * float(box[3])) ** 0.5)
                per_frame[(ann["video_id"], fi)] += 1
            elif hb:
                box_only += 1
            elif hs:
                mask_only += 1
        lengths.append(n)

    side = np.asarray(sides)
    lens = np.asarray(lengths)
    counts = np.asarray(list(per_frame.values()))

    out: list[str] = []
    add = out.append
    add("# AeroVIS -- measured, not assumed")
    add("")
    add(f"videos {len(vids)} | frames {sum(v['length'] for v in vids.values()):,} "
        f"| tracks {len(real):,} | iscrowd=1 ignore regions {crowd}")
    add("")
    add("## The stage-B question: (image, box prompt, mask) triples")
    add(f"  triples (box AND mask) : {both:,}")
    add(f"  box without mask       : {box_only:,}")
    add(f"  mask without box       : {mask_only:,}")
    add(f"  labelled frames        : {len(counts):,}")
    add("")
    add("## Per source")
    for src, n in per_src.most_common():
        add(f"  {src:13s} {n:9,d}  ({100 * n / both:.1f} %)")
    add("")
    add("## Per category (triples / tracks)")
    for name, n in per_cat.most_common():
        add(f"  {name:11s} {n:9,d} / {per_cat_tracks[name]:5d}")
    add("")
    add("## Object scale, sqrt(w*h) px -- aerial means small")
    for q in (1, 5, 25, 50, 75, 95, 99):
        add(f"  p{q:<3d} {np.percentile(side, q):7.1f}")
    add(f"  COCO-small <32px {100 * (side < 32).mean():.1f} % | "
        f"32-96 {100 * ((side >= 32) & (side < 96)).mean():.1f} % | "
        f">=96 {100 * (side >= 96).mean():.1f} %")
    add("")
    add("## Instances per labelled frame")
    add(f"  median {np.median(counts):.0f} | mean {counts.mean():.1f} "
        f"| max {counts.max()}")
    add("")
    add("## Track length in labelled frames -- stage C material")
    add(f"  median {np.median(lens):.0f} | mean {lens.mean():.1f} "
        f"| max {lens.max()} | >=8 frames: {(lens >= 8).sum():,}/{len(lens):,}")
    add("")
    add(prompt_independence(data, sample=sample))
    return "\n".join(out)


def prompt_independence(data: dict, sample: int = 400, seed: int = 0) -> str:
    """Is the shipped box the mask's envelope, or an independent annotation?

    If AeroVIS had recovered boxes from its own masks, a box prompt would be a
    lossy restatement of the answer and box-prompted training would flatter
    itself. Comparing each box against the tight envelope of the decoded mask
    settles it from the file rather than from the paper.
    """
    from pycocotools import mask as maskutil

    rng = random.Random(seed)
    real = _tracks(data)
    picks: list[tuple[dict, int]] = []
    for ann in rng.sample(real, min(sample, len(real))):
        idx = [i for i, (b, s) in enumerate(zip(ann["bboxes"], ann["segmentations"]))
               if b and s]
        if idx:
            picks.append((ann, rng.choice(idx)))

    deltas: list[list[float]] = []
    fills: list[float] = []
    empty = area_matches = 0
    for ann, fi in picks:
        box = ann["bboxes"][fi]
        seg = ann["segmentations"][fi]
        m = maskutil.decode({"size": seg["size"], "counts": seg["counts"].encode()})
        if m.sum() == 0:
            empty += 1
            continue
        ys, xs = np.nonzero(m)
        tight = (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
        deltas.append([abs(tight[i] - box[i]) for i in range(4)])
        fills.append(m.sum() / (box[2] * box[3]))
        if abs(int(ann["areas"][fi]) - int(m.sum())) <= 1:
            area_matches += 1

    d = np.asarray(deltas)
    f = np.asarray(fills)
    n = len(d)
    return "\n".join([
        f"## Box vs mask envelope ({n} sampled instances, {empty} empty masks)",
        f"  mean |delta| x,y,w,h : {d.mean(0).round(2)}",
        f"  exact match on all 4 : {100 * (d.max(1) == 0).mean():.1f} %",
        f"  within 2 px          : {100 * (d.max(1) <= 2).mean():.1f} %",
        f"  mask/box fill        : median {100 * np.median(f):.1f} %",
        f"  `areas` == mask px   : {100 * area_matches / n:.1f} %",
        "",
        "  A low exact-match rate means the boxes are the source datasets' own",
        "  annotations, not envelopes of AeroVIS's masks -- so the box prompt is",
        "  independent of the target and stage B can train on it honestly.",
    ])


def render(data: dict, zf: zipfile.ZipFile, out_dir: Path,
           per_source: int = 2) -> list[Path]:
    """Overlay decoded masks and box prompts on real frames, for eyes not stats."""
    from PIL import Image, ImageDraw, ImageFont
    from pycocotools import mask as maskutil

    out_dir.mkdir(parents=True, exist_ok=True)
    cats = {c["id"]: c["name"] for c in data["categories"]}
    vids = {v["id"]: v for v in data["videos"]}
    by_video: dict[int, list[dict]] = collections.defaultdict(list)
    for ann in _tracks(data):
        by_video[ann["video_id"]].append(ann)

    ranked: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    for vid, video in vids.items():
        n = sum(1 for a in by_video[vid] for b in a["bboxes"] if b)
        ranked[video["source"]].append((n, vid))
    picks: list[int] = []
    for src, lst in ranked.items():
        lst.sort(reverse=True)
        step = max(1, len(lst) // max(1, per_source))
        picks += [vid for _, vid in lst[::step][:per_source]]

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    written: list[Path] = []
    for vid in picks:
        video = vids[vid]
        anns = by_video[vid]
        counts = [sum(1 for a in anns if a["bboxes"][f] and a["segmentations"][f])
                  for f in range(video["length"])]
        fi = int(np.argmax(counts))
        with zf.open(FRAME_MEMBER.format(video["file_names"][fi])) as fh:
            frame = Image.open(io.BytesIO(fh.read())).convert("RGB")

        arr = np.asarray(frame).astype(np.float32)
        over = arr.copy()
        drawn = []
        for k, ann in enumerate(anns):
            seg, box = ann["segmentations"][fi], ann["bboxes"][fi]
            if seg is None or box is None:
                continue
            m = maskutil.decode(
                {"size": seg["size"], "counts": seg["counts"].encode()}).astype(bool)
            col = PALETTE[k % len(PALETTE)]
            over[m] = over[m] * 0.45 + np.asarray(col, dtype=np.float32) * 0.55
            drawn.append((box, cats[ann["category_id"]], ann["track_id"], col))

        panel = Image.fromarray(over.astype(np.uint8))
        pen = ImageDraw.Draw(panel)
        for (x, y, w, h), name, tid, col in drawn:
            pen.rectangle([x, y, x + w, y + h], outline=col, width=2)
            pen.text((x, max(0, y - 17)), f"{name}#{tid}", fill=col, font=font)

        W, H = frame.size
        canvas = Image.new("RGB", (W * 2 + 12, H), (18, 18, 18))
        canvas.paste(frame, (0, 0))
        canvas.paste(panel, (W + 12, 0))
        cap = ImageDraw.Draw(canvas)
        cap.text((8, 8), f'{video["name"]} ({video["source"]}) frame {fi} -- raw',
                 fill=(255, 255, 255), font=font)
        cap.text((W + 20, 8), f"{len(drawn)} instances: mask + box prompt",
                 fill=(255, 255, 255), font=font)
        if canvas.width > 2400:
            canvas = canvas.resize(
                (2400, int(canvas.height * 2400 / canvas.width)), Image.LANCZOS)
        path = out_dir / f'{video["source"]}_{video["name"]}_f{fi}.jpg'
        canvas.save(path, quality=88)
        written.append(path)
        print(f"  {video['name']:8s} frame {fi:4d}: {len(drawn):3d} instances "
              f"-> {path}", file=sys.stderr)
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", type=Path, default=None,
                   help="Local aero_vis.json; fetched into this path if absent.")
    p.add_argument("--stats", action="store_true", help="Measure and print.")
    p.add_argument("--render", type=Path, default=None,
                   help="Directory for mask+box overlay panels.")
    p.add_argument("--per-source", type=int, default=2,
                   help="Sequences to render per source dataset.")
    p.add_argument("--sample", type=int, default=400,
                   help="Instances sampled for the box-vs-envelope check.")
    p.add_argument("--out", type=Path, default=None, help="Write stats here too.")
    args = p.parse_args(argv)

    if not args.stats and args.render is None:
        p.error("nothing to do: pass --stats and/or --render")

    need_archive = args.render is not None or not (args.json and args.json.exists())
    zf = open_archive() if need_archive else None
    if zf is not None:
        print(f"  archive: {len(zf.namelist()):,} entries", file=sys.stderr)

    data = load_annotations(zf, args.json)

    if args.stats:
        text = stats(data, sample=args.sample)
        print(text)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n")
    if args.render is not None:
        render(data, zf, args.render, per_source=args.per_source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
