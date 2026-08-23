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

**Why this reads parquet directly instead of streaming.** SegFly is 178 GiB in
761 shards, and only **15 007 of its 35 613 rows are thermal**. Measured by
reading every shard's `modality` column: a thermal row is ~1.4 MB (640x512), an
RGB row is ~11 MB (4000x3000), and the two are sorted into different shards --
311 hold thermal rows only, 434 hold RGB rows only, 16 are mixed. So the shards
worth having are 327 of 761, and 22 GiB of 178.

`load_dataset(streaming=True)` filters *after* the bytes arrive. It walks the
shards in order and the first 57 are pure RGB, so a `--modality thermal` export
downloads ~25 GiB before it can write its first row -- and all 178 GiB before
it writes its last, to keep 22 GiB of it. That is a seven-hour cell, measured,
and the seven hours are almost entirely spent on rows that get skipped.

Parquet is columnar and the modality is one tiny column, so the same question
can be asked for a few hundred KB per shard instead of a few hundred MB:

1. read `modality` from every shard's footer, in parallel, and cache the answer
2. download **only** the shards that hold a matching row
3. read only the wanted columns, decode only the wanted rows, then delete the
   shard before fetching the next one

Peak disk is one shard plus the output, and a dying runtime resumes at the
shard it died on rather than at row zero.

`--stream` restores the old behaviour for a dataset this cannot plan.

    python tools/export_hf_dataset.py markus-42/SegFly --dest /content/data/SegFly \\
        --modality thermal --expect segfly
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.aerial import SPECS  # noqa: E402

COLUMNS = {"image": "images", "label": "labels", "RGB_aligned": "rgb"}
META = ("scene", "altitude", "modality")


def folders_for(modality: str | None) -> dict[str, str]:
    """Where each column lands, given which modality is being exported.

    On an RGB export the `image` column *is* the RGB frame, so it goes to
    `rgb/` -- that is the directory `SPECS[...].rgb` globs, and it is what
    lets the same spec read a thermal export (`images/` + `labels/`) and an
    RGB one (`rgb/` + `labels/`) with nothing but the `--dataset` modality
    field changing between them.
    """
    if modality and modality.lower() == "rgb":
        return {**COLUMNS, "image": "rgb"}
    return dict(COLUMNS)


def stem(row: dict, index: int) -> str:
    """`<scene>_<altitude>_<n>` -- the pairing key and the metadata, together.

    Zero-padded so a lexical sort is also a numeric one, and identical across
    the three directories so `list_frames` pairs them by stem with no rule
    beyond "same name".

    `index` is the row's position in the **whole split**, not in its shard, so
    the name a row gets does not depend on which shards were selected. An
    export narrowed to one modality and one widened later agree on names, and
    re-running after a crash overwrites rather than duplicates.
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


def dump(cell, stub: Path) -> Path | None:
    """One raw parquet Image cell to disk **unchanged** -- no decode, no PNG.

    The passthrough path, and it exists for one measured reason: SegFly's RGB
    frames are 4000x3000 JPEGs of ~11 MB that re-encode to ~27 MB of PNG, so
    a PNG export of the RGB slice costs 2.5x the disk and the decode time to
    make every file *worse*. The publisher's bytes are kept byte for byte,
    under the original format's suffix.

    Never used for label maps -- those must be decoded anyway to count their
    values, and must stay PNG to stay lossless.
    """
    if not cell or cell.get("bytes") is None:
        return None
    suffix = Path(cell.get("path") or "").suffix.lower() or ".jpg"
    out = stub.with_suffix(suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(cell["bytes"])
    return out


def decode(cell) -> "PIL.Image.Image | None":                    # noqa: F821
    """A raw parquet `{bytes, path}` Image cell to PIL, or None if absent.

    The bytes are whatever the publisher stored -- SegFly's thermal half is
    MPO, a multi-picture JPEG, and PIL reads its primary frame. Decoding here
    rather than through `datasets` is what lets the caller skip a row without
    ever touching its pixels.
    """
    if not cell or cell.get("bytes") is None:
        return None
    import io

    import PIL.Image

    return PIL.Image.open(io.BytesIO(cell["bytes"]))


# --------------------------------------------------------------------------
# Planning: which shards hold the rows we want
# --------------------------------------------------------------------------


def shard_paths(dataset_id: str, split: str) -> list[str]:
    """Every parquet shard of `split`, in the order that defines row indices."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    found = fs.glob(f"datasets/{dataset_id}/**/{split}-*.parquet")
    if not found:                       # single-file splits, or a flat repo
        found = fs.glob(f"datasets/{dataset_id}/**/*{split}*.parquet")
    if not found:
        raise SystemExit(
            f"{dataset_id}: no parquet shard matched {split!r}. Run with "
            f"--stream to fall back to load_dataset.")
    return sorted(found)


def plan(dataset_id: str, split: str, modality: str | None, cache: Path,
         workers: int = 16, quiet: bool = False) -> list[dict]:
    """Per shard: its row offset, and which of its rows match `modality`.

    Reads one string column out of each shard's footer -- a few hundred KB
    against the few hundred MB the same answer costs when the filter runs after
    the download. The result is cached because it is the one part of the export
    that a crashed runtime should never have to pay for twice.

    **Not derived from shard size, and not assumed contiguous.** Both are
    tempting on SegFly (its thermal shards are 55-90 MiB against 300-510 MiB
    for the RGB ones) and both are guesses about a repacking nobody promised.
    The column is cheap enough that there is no reason to guess.
    """
    if cache.is_file():
        saved = json.loads(cache.read_text())
        if saved.get("dataset") == dataset_id and saved.get("split") == split \
                and saved.get("modality") == modality:
            if not quiet:
                print(f"   shard plan reused from {cache.name}")
            return saved["shards"]

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    paths = shard_paths(dataset_id, split)
    if not quiet:
        print(f"   planning {len(paths)} shards "
              f"({'modality=' + modality if modality else 'all rows'}) ...")

    def probe(path: str) -> tuple[str, int, list[int]]:
        with fs.open(path, "rb") as handle:
            handle_pq = pq.ParquetFile(handle)
            if not modality:                  # every row wanted; footer only
                rows = handle_pq.metadata.num_rows
                return path, rows, list(range(rows))
            values = handle_pq.read(columns=["modality"]) \
                              .column("modality").to_pylist()
        keep = [i for i, v in enumerate(values)
                if str(v or "").lower() == modality.lower()]
        return path, len(values), keep

    with ThreadPoolExecutor(workers) as pool:
        probed = {p: (n, k) for p, n, k in pool.map(probe, paths)}

    shards, offset = [], 0
    for path in paths:
        rows, keep = probed[path]
        shards.append({"path": path, "offset": offset, "rows": rows,
                       "keep": keep})
        offset += rows

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"dataset": dataset_id, "split": split,
                                 "modality": modality, "shards": shards}))
    if not quiet:
        wanted = sum(len(s["keep"]) for s in shards)
        hot = sum(1 for s in shards if s["keep"])
        print(f"   {wanted} of {offset} rows are wanted, in {hot} of "
              f"{len(paths)} shards -- the other {len(paths) - hot} are never "
              f"downloaded")
    return shards


def _fast_transfer() -> None:
    """Turn on `hf_transfer` if it is installed, and stay quiet if it is not.

    It downloads one file over several connections, which on a Colab runtime is
    the difference between saturating the link and not. The env var must not be
    set without the package -- `huggingface_hub` then raises instead of falling
    back -- so this checks rather than hopes.
    """
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"):
        return
    try:
        import hf_transfer  # noqa: F401
    except ImportError:
        return
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def pull(dataset_id: str, path: str, into: Path) -> Path:
    """One shard onto local disk, resumably, returning where it landed.

    `local_dir` keeps it a real file instead of a symlink into the hub cache,
    so the caller can delete it the moment it has been read and hold peak disk
    at one shard.
    """
    from huggingface_hub import hf_hub_download

    _fast_transfer()

    # `datasets/<owner>/<name>/<path in repo>` -> `<path in repo>`
    inside = path.split(f"datasets/{dataset_id}/", 1)[-1]
    local = hf_hub_download(dataset_id, inside, repo_type="dataset",
                            local_dir=into)
    return Path(local)


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------


def export_parquet(dataset_id: str, dest: Path, modality: str | None,
                   split: str, limit: int | None, columns: tuple[str, ...],
                   work: Path | None = None, quiet: bool = False,
                   passthrough: bool = False, spread: bool = False) -> dict:
    """Shard by shard: download, write the matching rows, delete, next.

    `spread` changes what `limit` means. Without it, `limit` takes the *first*
    N matching rows -- minimal download, but the rows come from whichever
    scene the repository happens to open with. With it, the N rows are taken
    from **evenly spaced shards** across everything that matched, so a slice
    of SegFly's RGB half samples all of its scenes and altitudes instead of
    3 000 consecutive frames of one field -- while still downloading only the
    shards the slice actually touches (~1/7th of them for 3 000 of 20 606).
    Shard-level rather than row-level on purpose: a row-level stride lands in
    nearly every shard and quietly re-downloads the lot.
    """
    import pyarrow.parquet as pq

    work = work or dest / "_parquet"
    shards = plan(dataset_id, split, modality, dest / "shard_plan.json",
                  quiet=quiet)
    hot = [s for s in shards if s["keep"]]
    if limit and spread and hot:
        per = max(1, sum(len(s["keep"]) for s in hot) // len(hot))
        needed = min(len(hot), max(1, -(-limit // per)))
        step = len(hot) / needed
        hot = [hot[int(i * step)] for i in range(needed)]
        if not quiet:
            print(f"   spread: {needed} evenly spaced shards cover the "
                  f"{limit}-row slice")

    values: Counter = Counter()
    written = 0
    read = list(columns) + list(META)
    folders = folders_for(modality)

    for n, shard in enumerate(hot, 1):
        local = pull(dataset_id, shard["path"], work)
        try:
            table = pq.ParquetFile(local).read(columns=read)
            for i in shard["keep"]:
                row = {k: table.column(k)[i].as_py() for k in read}
                name = stem(row, shard["offset"] + i)
                for column in columns:
                    folder = dest / folders[column]
                    if passthrough and column != "label":
                        dump(row.get(column), folder / name)
                        continue
                    array = write(decode(row.get(column)),
                                  folder / f"{name}.png")
                    if array is not None and column == "label":
                        values.update(np.unique(array).tolist())
                written += 1
                if limit is not None and written >= limit:
                    break
        finally:
            local.unlink(missing_ok=True)
        if not quiet:
            print(f"  shard {n}/{len(hot)}: {written} rows written")
        if limit is not None and written >= limit:
            break

    shutil.rmtree(work, ignore_errors=True)
    # Everything in the split that was not written: filtered rows, rows in
    # shards never downloaded, and -- under `spread` or a mid-shard `limit` --
    # matching rows deliberately left behind. One definition, no bookkeeping.
    skipped = sum(s["rows"] for s in shards) - written
    return {"written": written, "skipped": skipped,
            "values": {int(v): int(n) for v, n in sorted(values.items())}}


def export(dataset_id: str, dest: Path, modality: str | None, split: str,
           limit: int | None, streaming: bool, quiet: bool = False,
           columns: tuple[str, ...] = tuple(COLUMNS),
           passthrough: bool = False, spread: bool = False) -> dict:
    """Write the rows out; return counts and the label values seen.

    `streaming=True` is the `load_dataset` fallback -- correct on any dataset
    and, on one where most rows are filtered out, ruinously slow. See the
    module docstring. It decodes through `datasets`, so `passthrough` and
    `spread` only exist on the parquet path.
    """
    if not streaming:
        return export_parquet(dataset_id, dest, modality, split, limit,
                              columns, quiet=quiet,
                              passthrough=passthrough, spread=spread)

    from datasets import load_dataset

    rows = load_dataset(dataset_id, split=split, streaming=True)
    values: Counter = Counter()
    written = 0
    skipped = 0

    for index, row in enumerate(rows):
        if modality and str(row.get("modality", "")).lower() != modality.lower():
            skipped += 1
            continue
        name = stem(row, index)
        for column in columns:
            array = write(row.get(column),
                          dest / folders_for(modality)[column] / f"{name}.png")
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

    spec = SPECS[spec_name]
    known = set(spec.classes.values())
    ignored = sorted((set(values) - known) & set(spec.ignore))
    unexpected = sorted(set(values) - known - set(spec.ignore))
    missing = sorted(known - set(values))
    lines.append(f"{spec_name} palette:    {sorted(known)}")
    if ignored:
        # Not an error and not a mystery: ids the spec knows about and drops
        # -- for SegFly, stray pixels the publisher's own class remapping
        # missed. Reported so a new id showing up here is noticed, silently
        # accepted so it never reads as a broken download.
        lines.append(f"   {len(ignored)} leftover id(s) the spec ignores: "
                     f"{ignored}")
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
                        "SegFly means 20 606 frames at 4000x3000.")
    p.add_argument("--limit", type=int, default=None, help="Stop after N rows.")
    p.add_argument("--expect", default=None, choices=sorted(SPECS),
                   help="Check the label values against this spec's palette.")
    p.add_argument("--no-rgb", action="store_true",
                   help="Skip the `RGB_aligned` column. It is read only by "
                        "stage-A distillation, and none of the notebooks "
                        "distil from SegFly -- so for them it is ~10 GB of "
                        "PNGs nothing opens.")
    p.add_argument("--passthrough", action="store_true",
                   help="Write photographic columns as the publisher's own "
                        "bytes (JPEG stays JPEG) instead of re-encoding to "
                        "PNG. Labels are always PNG. For SegFly's RGB slice "
                        "this is 11 MB a frame instead of 27.")
    p.add_argument("--spread", action="store_true",
                   help="With --limit: take the rows from evenly spaced "
                        "shards across the whole match, so a slice samples "
                        "every scene instead of the first one.")
    p.add_argument("--stream", action="store_true",
                   help="Fall back to load_dataset streaming instead of "
                        "reading the parquet shards directly. Correct "
                        "anywhere; on a filtered export, far slower.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    columns = tuple(c for c in COLUMNS
                    if not (args.no_rgb and c == "RGB_aligned"))
    dest = Path(args.dest).expanduser()
    print(f">> {args.dataset} [{args.split}] -> {dest}"
          + (f", modality={args.modality}" if args.modality else "")
          + (", without RGB_aligned" if args.no_rgb else ""))
    result = export(args.dataset, dest, args.modality, args.split, args.limit,
                    streaming=args.stream, quiet=args.quiet, columns=columns,
                    passthrough=args.passthrough, spread=args.spread)

    print(f"\n{result['written']} rows written, {result['skipped']} skipped")
    print(verify(result["values"], args.expect))
    print(f"\nNow train on it with:\n"
          f"    --dataset {args.expect or '<spec>'}:{dest}:thermal:components:train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
