#!/usr/bin/env python3
"""A Hugging Face dataset into the directory layout the training path reads.

SegFly is distributed as **parquet**, not as folders -- columns `image`,
`label`, `RGB_aligned`, `scene`, `altitude`, `modality` -- so a `DatasetSpec`'s
globs find nothing in it. This writes it out as PNGs:

    <dest>/images/<scene>_<altitude>_<n>.png    the modality being trained on
    <dest>/rgb/<scene>_<altitude>_<n>.png       its registered RGB half
    <dest>/labels/<scene>_<altitude>_<n>.png    the semantic map

which is exactly what `SPECS["segfly"]` globs, on purpose: the exporter and the
spec agree by construction rather than by someone keeping two guesses in step.

**Three things it does that a loop over `load_dataset` would not.**

*Streams by default.* SegFly's RGB half is 20 000 frames at 5472x3648; pulling
it to train on 640x512 thermal would spend a Colab session's disk on pixels
nothing reads. `--modality thermal` keeps only the rows that carry a thermal
image, and streaming means the rest is never downloaded rather than downloaded
and skipped.

*Keeps scene and altitude in the filename.* They are columns, and once the rows
become files the only place that survives is the name. SegFly spans 30/50 m and
the altitude axis is a real experimental variable here (report section 3.4) --
losing it at export would be losing it for good.

*Checks the labels are class ids and not a smear.* A segmentation map saved
through a lossy codec at any point comes back with values near the class ids
rather than on them, and every downstream count is then quietly wrong.
`--expect` compares what came out against the spec's palette and says which
values are unaccounted for.

    python tools/export_hf_dataset.py markus-42/SegFly --dest /content/data/SegFly \\
        --modality thermal --expect segfly
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.aerial import SPECS  # noqa: E402

COLUMNS = {"image": "images", "label": "labels", "RGB_aligned": "rgb"}


def stem(row: dict, index: int) -> str:
    """`<scene>_<altitude>_<n>` -- the pairing key and the metadata, together.

    Zero-padded so a lexical sort is also a numeric one, and identical across
    the three directories so `list_frames` pairs them by stem with no rule
    beyond "same name".
    """
    scene = str(row.get("scene") or "scene").replace("/", "-").replace(" ", "")
    altitude = str(row.get("altitude") or "").replace("/", "-").replace(" ", "")
    parts = [p for p in (scene, altitude) if p]
    return "_".join(parts + [f"{index:06d}"])


def write(image, path: Path) -> np.ndarray | None:
    """One PIL image to `path` as PNG, returning it as an array.

    PNG throughout, including for the photographic halves. A label map has to
    be lossless or its class ids stop being ids, and using one format for all
    three removes the chance of the wrong one being picked for the map.
    """
    if image is None:
        return None
    import PIL.Image

    array = np.array(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    PIL.Image.fromarray(array).save(path, format="PNG", optimize=False)
    return array


def export(dataset_id: str, dest: Path, modality: str | None, split: str,
           limit: int | None, streaming: bool, quiet: bool = False) -> dict:
    """Write the rows out; return counts and the label values seen."""
    from datasets import load_dataset

    rows = load_dataset(dataset_id, split=split, streaming=streaming)
    values: Counter = Counter()
    written = 0
    skipped = 0

    for index, row in enumerate(rows):
        if modality and str(row.get("modality", "")).lower() != modality.lower():
            skipped += 1
            continue
        name = stem(row, index)
        for column, folder in COLUMNS.items():
            array = write(row.get(column), dest / folder / f"{name}.png")
            if array is not None and column == "label":
                values.update(np.unique(array).tolist())
        written += 1
        if not quiet and written % 500 == 0:
            print(f"  {written} rows written ({skipped} skipped)")
        if limit is not None and written >= limit:
            break

    return {"written": written, "skipped": skipped,
            "values": {int(v): int(n) for v, n in sorted(values.items())}}


def verify(values: dict[int, int], spec_name: str | None) -> str:
    """The label values that came out, against the palette they should be.

    Reports the unexpected ones rather than a pass/fail, because the two ways
    this goes wrong look different: a handful of stray values is a lossy save
    somewhere in the chain, and a completely disjoint set is the wrong palette
    in the spec.
    """
    lines = [f"label values found: {sorted(values)}"]
    if spec_name is None:
        return "\n".join(lines)

    known = set(SPECS[spec_name].classes.values())
    unexpected = sorted(set(values) - known)
    missing = sorted(known - set(values))
    lines.append(f"{spec_name} palette:    {sorted(known)}")
    if unexpected:
        lines.append(
            f"!! {len(unexpected)} value(s) not in the palette: {unexpected[:20]}"
            + ("" if len(unexpected) <= 20 else " ...")
            + "\n   A few strays next to real ids means something in the chain "
              "re-encoded the map lossily. A disjoint set means the spec's "
              "palette is wrong.")
    if missing:
        lines.append(f"   {len(missing)} palette value(s) absent from this "
                     f"export: {missing} -- rare classes, or a partial export.")
    if not unexpected:
        lines.append("every value is a class id in the palette")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", help="Hub id, e.g. markus-42/SegFly.")
    p.add_argument("--dest", required=True, help="Local directory to write into.")
    p.add_argument("--split", default="train")
    p.add_argument("--modality", default=None,
                   help="Keep only rows whose `modality` column matches, e.g. "
                        "thermal. Without it every row is written, which for "
                        "SegFly means 20 000 frames at 5472x3648.")
    p.add_argument("--limit", type=int, default=None, help="Stop after N rows.")
    p.add_argument("--expect", default=None, choices=sorted(SPECS),
                   help="Check the label values against this spec's palette.")
    p.add_argument("--download", action="store_true",
                   help="Download the whole dataset first instead of streaming "
                        "it. Faster per row, and needs the disk for all of it.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    dest = Path(args.dest).expanduser()
    print(f">> {args.dataset} [{args.split}] -> {dest}"
          + (f", modality={args.modality}" if args.modality else ""))
    result = export(args.dataset, dest, args.modality, args.split, args.limit,
                    streaming=not args.download, quiet=args.quiet)

    print(f"\n{result['written']} rows written, {result['skipped']} skipped")
    print(verify(result["values"], args.expect))
    print(f"\nNow train on it with:\n"
          f"    --dataset {args.expect or '<spec>'}:{dest}:thermal:components:train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
