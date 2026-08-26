"""A staged mask pool, read back as stage B's training data.

`pool.py` writes one directory per frame -- `record.json` beside a run-length
`pseudo_masks.npz` -- and notebooks 13-18 zip those directories to Drive. That
is where the supply line stopped: `docs/mask_pool_plan.md` says wiring the
pools into stage B's `DATASETS` is "a deliberate follow-up", and this module is
it. It turns a pool directory into the `FrameIndex` list `sample_windows`,
`split_index` and `image_loop` already consume, so a pool trains through
exactly the same loop, losses, schedule and evaluation as Kust4K's decomposed
semantic maps -- which is the only reason a pool run's number can be set beside
a `07`-`11` number at all.

**The pixels are not in the pool.** A store is a few kilobytes of run lengths;
the frame it describes is on the dataset's own disk. So every entry point here
takes an `images_root` and re-roots the absolute path the harvest runtime
recorded (`/content/data/HIT_UAV/...`) onto wherever the images are now. The
re-rooting is done by *trying* suffixes against the filesystem rather than by
string surgery: the first frame decides how many leading components to drop and
every frame after it reuses that answer, so a pool harvested in Colab trains on
a workstation with one setting and no editing of thirty thousand records.

**Three things the pool already decided, and this module does not re-decide.**

`Instance.label` is the box's row number in its frame, which is the key
`label_pool` filed the mask under -- so recovering a mask is a lookup, with no
box matching and nothing to drift. Rejected boxes are absent from the store and
are never indexed; a store the index disagrees with raises rather than trains.
And a mask's coordinates are the *inset* frame's wherever the reader carried an
inset (DroneVehicle's 100 px white band), so the inset travels on the synthetic
spec's `border` and `aerial.image_origin` applies it at the one place that
reads pixels.

What this module *does* decide is which indexed instances are worth training
on, and it applies `aerial.InstanceGates` for it -- the same four gates, with
the same per-gate reject counts, that the semantic path applies to a connected
component. A teacher mask that survived `labels.Gates` can still be four pixels
across, and a four-pixel target is not a prompt anyone would give.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .aerial import (ROLES, DatasetSpec, Frame, FrameIndex, Instance,
                     InstanceGates, Source, reject_reason)
from .labels import MASK_STORE
from .pool import RECORD_FILE

PALETTE_SOURCE = (
    "the pool's own record.json -- each instance carries the class name the "
    "box reader read out of the dataset's annotation file, so there is no "
    "palette to guess. Ids here are positions in the sorted set of names "
    "actually present in this pool, assigned once when the index is built."
)


# --------------------------------------------------------------------------
# Finding the pixels again
# --------------------------------------------------------------------------


class Relocator:
    """Re-roots the absolute image paths a harvest runtime wrote.

    A record says `/content/data/HIT_UAV/normal_json/train/0_01.jpg` because
    that is where the file was when the teacher looked at it. Training happens
    somewhere else, so the path has to be rebuilt against a local root -- and
    the honest way to do that is to ask the filesystem, because nothing in the
    record says how many leading components belong to the old root.

    The first hit fixes `depth` and every frame after it is one `is_file`
    check, so a pool of thirty thousand records costs thirty thousand stats
    rather than thirty thousand searches. `misses` counts what could not be
    found at all, which is the number that tells you the images were never
    downloaded rather than that the pool is empty.
    """

    def __init__(self, images_root: str | Path | None) -> None:
        self.root = Path(images_root) if images_root else None
        self.depth: int | None = None
        self.misses = 0

    def __call__(self, recorded: str | Path) -> Path | None:
        path = Path(recorded)
        if self.root is None:
            return path if path.is_file() else self._miss()

        parts = path.parts
        if self.depth is not None:
            candidate = self.root.joinpath(*parts[self.depth:])
            if candidate.is_file():
                return candidate
        # Deepest suffix first: a shallow one can match the wrong file when the
        # local root happens to repeat a directory name the old root also had.
        for depth in range(len(parts) - 1, -1, -1):
            candidate = self.root.joinpath(*parts[depth:])
            if candidate.is_file():
                self.depth = depth
                return candidate
        return self._miss()

    def _miss(self) -> None:
        self.misses += 1
        return None


# --------------------------------------------------------------------------
# What is on disk
# --------------------------------------------------------------------------


def store_areas(store: str | Path) -> dict[int, int]:
    """`{instance key: lit pixels}` off a run-length store, without decoding it.

    The index needs an area per instance -- it is what `InstanceGates` gates on
    and what `Instance.fill` divides -- and decoding a full-frame boolean mask
    to count its True pixels is 330 KB of allocation for a number the runs
    already hold. `rle_encode` starts every frame with a False run, so the lit
    pixels are the odd-indexed entries and the whole file is a sum over a
    slice.
    """
    with np.load(store) as data:
        runs, offsets, frames = data["runs"], data["offsets"], data["frames"]
        return {int(frame): int(runs[offsets[i]:offsets[i + 1]][1::2].sum())
                for i, frame in enumerate(frames)}


def _image_shape(path: Path) -> tuple[int, int] | None:
    """`(height, width)` from the file header, or None if nothing read it."""
    try:
        from PIL import Image

        with Image.open(path) as handle:
            width, height = handle.size
        return int(height), int(width)
    except ImportError:
        pass
    except Exception:                        # noqa: BLE001 - unreadable file
        return None
    try:
        import cv2

        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return None if raw is None else (int(raw.shape[0]), int(raw.shape[1]))
    except Exception:                        # noqa: BLE001 - unreadable file
        return None


def pool_datasets(root: str | Path) -> list[str]:
    """The pool names under `root` -- the directories that hold records.

    A staged Drive folder can hold several (`kust4k_rgb`, `kust4k_thermal`,
    `kust4k_broken_*`), and which of them a run wants is a decision, not a
    default. Named by their path relative to `root` so a pool nested one level
    deeper -- what `archive_to` produces when a zip carries its parent -- is
    still found and still named unambiguously.
    """
    root = Path(root)
    names = {record.parent.relative_to(root).parts[0]
             for record in root.rglob(RECORD_FILE)
             if record.parent != root}
    return sorted(names)


def discover_pools(root: str | Path, depth: int = 6) -> dict[str, Path]:
    """`{pool name: the directory that holds it}` under `root`.

    `pool_datasets` reads the *directory* name, which is right when a pool was
    unzipped where it was written and wrong the moment an archive carried its
    parent along: the folder then reads as `pool`, one name for four datasets.
    Every record already states which pool it belongs to, so this asks the
    records and then finds the ancestor directory with that name -- exact, and
    unaffected by how deep an archive nested things.

    Bounded to `depth` levels of glob rather than an `rglob`, because the root
    this is usually pointed at is a mounted Drive folder holding tens of
    thousands of small files and a full walk of one takes minutes. It stops at
    the first depth that finds anything, so a pool one level down costs one
    listing rather than six.
    """
    root = Path(root)
    found: dict[str, Path] = {}
    for level in range(1, max(depth, 1) + 1):
        for record in root.glob("/".join(["*"] * level + [RECORD_FILE])):
            try:
                name = str(json.loads(record.read_text()).get("dataset") or "")
            except (OSError, ValueError):
                continue
            if not name or name in found:
                continue
            owner = next((p for p in record.parents if p.name == name), None)
            found[name] = owner or record.parent
        if found:
            break
    return dict(sorted(found.items()))


def group_records(root: str | Path, workers: int = 8) -> dict[str, list[Path]]:
    """`{pool name: its record files}` under `root`, grouped by what they say.

    The layout-independent answer, and the one a local pool root should use.
    `pool_datasets` and `discover_pools` both infer a pool from a *directory*,
    which works only when the archive that staged it happened to carry the pool
    name at the top -- and the archives on a real Drive do not agree about
    that: one harvest zipped per split (`train.zip`), another per chunk
    (`000000.zip`), a third per pool. Every record already states which pool it
    belongs to, so grouping on that field is right whatever the folders look
    like, and it costs one JSON read per frame on local disk.

    Use `discover_pools` for a **mounted** root instead: it is depth-bounded
    because an `rglob` over Drive's FUSE mount takes minutes.
    """
    from concurrent.futures import ThreadPoolExecutor

    records = sorted(Path(root).rglob(RECORD_FILE))

    def named(path: Path) -> tuple[str, Path] | None:
        try:
            name = str(json.loads(path.read_text()).get("dataset") or "")
        except (OSError, ValueError):
            return None
        return (name, path) if name else None

    grouped: dict[str, list[Path]] = {}
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        for row in pool.map(named, records):
            if row is not None:
                grouped.setdefault(row[0], []).append(row[1])
    return {name: sorted(paths) for name, paths in sorted(grouped.items())}


def link_pool(records: SequenceABC[Path], target: str | Path) -> Path:
    """Gather one pool's frames under `target`, by hard link where possible.

    `group_records` can say which frames belong to which pool whatever the
    folders look like, but two things downstream still want a real directory:
    `--pool` on the two CLIs, and the cache filename, which is the pool
    directory's basename and has to be unique per pool. Rebuilding the tree
    the harvest wrote -- `<target>/<key>/{record.json,pseudo_masks.npz}` --
    gives both, and a hard link costs an inode rather than a copy of the data.

    Falls back to copying when the link cannot be made (a different device,
    a filesystem without links). Existing frames are left alone, so a second
    call after a partial one finishes it rather than repeating it.
    """
    import os
    import shutil

    target = Path(target)
    for record in records:
        try:
            key = str(json.loads(record.read_text()).get("key") or "")
        except (OSError, ValueError):
            continue
        parts = [p for p in Path(key or record.parent.name).parts
                 if p not in ("..", "/", "")]
        folder = target.joinpath(*parts) if parts else target / record.parent.name
        folder.mkdir(parents=True, exist_ok=True)
        for name in (RECORD_FILE, MASK_STORE):
            source, destination = record.parent / name, folder / name
            if destination.exists() or not source.is_file():
                continue
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
    return target


@dataclass(frozen=True)
class PoolFrame:
    """One record, resolved against the local disk. `None` fields are misses."""

    key: str
    image: Path
    store: Path
    shape: tuple[int, int]                 # (height, width) the record claims
    border: int
    rows: tuple[tuple[int, str, tuple[float, float, float, float], int], ...]


def _read_frame(record_path: Path, relocate: Relocator,
                areas_of=store_areas) -> PoolFrame | str:
    """One `record.json` + its store, or the name of what went wrong."""
    record = json.loads(record_path.read_text())
    store = record_path.parent / MASK_STORE
    if not store.is_file():
        return "no_store"
    image = relocate(record["image"])
    if image is None:
        return "no_image"

    height, width = (int(v) for v in record["shape"])
    on_disk = _image_shape(image)
    if on_disk is None:
        return "unreadable_image"
    border = (on_disk[0] - height) // 2
    if border < 0 or on_disk != (height + 2 * border, width + 2 * border):
        # The frame on disk is not the frame the teacher saw. Stamping the
        # stored mask onto it would be silently wrong at every pixel, which is
        # the one failure worth refusing outright.
        return "shape_mismatch"

    areas = areas_of(store)
    rows = []
    for instance in record["instances"]:
        if instance["verdict"] is not None:
            continue
        key = int(instance["i"])
        if key not in areas:
            return "store_disagrees"
        rows.append((key, str(instance["class"]),
                     tuple(float(v) for v in instance["box"]), areas[key]))
    if not rows:
        return "no_accepted"
    return PoolFrame(key=str(record["key"]), image=image, store=store,
                     shape=(height, width), border=border, rows=tuple(rows))


# --------------------------------------------------------------------------
# The index
# --------------------------------------------------------------------------


def pool_spec(name: str, classes: SequenceABC[str], border: int = 0) -> DatasetSpec:
    """A `DatasetSpec` standing in for a pool, so the rest of the stack agrees.

    Everything downstream reaches through `Source.spec` -- `eval_instances`
    qualifies a class as `spec.name/spec.name_of(id)`, `summarise` groups by
    it, `image_origin` reads `border`. None of them need an annotation map, so
    the glob fields stay empty and the class table is built from the names the
    pool's own records carry rather than from a palette anybody wrote down.
    """
    table = {name_: index for index, name_ in enumerate(sorted(set(classes)))}
    return DatasetSpec(
        name=f"pool/{name}", masks="", classes=table,
        things=tuple(sorted(table)), border=int(border),
        palette_source=PALETTE_SOURCE)


def index_pool(
    pool_dir: str | Path,
    images_root: str | Path | None = None,
    modality: str = "thermal",
    role: str = "all",
    gates: InstanceGates | None = None,
    name: str | None = None,
    limit: int | None = None,
    workers: int = 8,
    progress=None,
    records: SequenceABC[Path] | None = None,
) -> list[FrameIndex]:
    """Every accepted instance in one pool, as the index stage B samples from.

    `pool_dir` is one dataset's directory (the one holding `record.json` files
    under it), `images_root` is where that dataset's frames live now.
    `modality` decides only whether the frames are read as one channel replicated
    three ways or as colour, and `role` is `aerial`'s: a pool is teacher output,
    so `train` -- never scored on -- is the honest default for anything whose
    masks nobody drew.

    Frames are dropped, with a count, when their store is missing, their image
    cannot be found or read, the file on disk is a different size from the one
    the teacher saw, or every box in them was rejected at harvest. The counts
    ride on each entry's `rejects` so `summarise` prints them.
    """
    from concurrent.futures import ThreadPoolExecutor

    pool_dir = Path(pool_dir)
    records = (sorted(pool_dir.rglob(RECORD_FILE)) if records is None
               else sorted(records))
    if limit is not None:
        records = records[:limit]
    if not records:
        raise FileNotFoundError(
            f"{pool_dir}: no {RECORD_FILE} anywhere underneath. Either the "
            f"pool zip was never unpacked here, or this is the folder above "
            f"the pools -- try one of {pool_datasets(pool_dir.parent) or '[]'}.")

    relocate = Relocator(images_root)
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        stream = pool.map(lambda r: _read_frame(r, relocate), records)
        if progress is not None:
            stream = progress(stream, total=len(records), desc=pool_dir.name)
        results = list(stream)

    frames = [r for r in results if isinstance(r, PoolFrame)]
    skipped: dict[str, int] = {}
    for result in results:
        if isinstance(result, str):
            skipped[result] = skipped.get(result, 0) + 1
    if not frames:
        raise ValueError(
            f"{pool_dir}: {len(records)} records and not one usable frame "
            f"({skipped}). `no_image` means --images points somewhere the "
            f"frames are not; `shape_mismatch` means it points at a different "
            f"copy of them.")

    borders = {f.border for f in frames}
    if len(borders) > 1:
        raise ValueError(
            f"{pool_dir}: frames disagree about the inset the harvest cut "
            f"({sorted(borders)}). A pool is one dataset read one way; two "
            f"insets mean two datasets were written into one directory.")

    spec = pool_spec(name or pool_dir.name,
                     [row[1] for f in frames for row in f.rows],
                     border=borders.pop())
    source = Source(spec=spec, gates=gates or InstanceGates(), mode="pool",
                    gray=modality == "thermal", role=role)

    index: list[FrameIndex] = []
    for frame in frames:
        height, width = frame.shape
        rejects = dict(skipped) if not index else {}
        instances = []
        for label, class_name, box, area in frame.rows:
            instance = Instance(label=label, class_id=spec.classes[class_name],
                                box=box, area=area)
            reason = reject_reason(instance, float(height * width), source.gates)
            if reason is None:
                instances.append(instance)
            else:
                rejects[reason] = rejects.get(reason, 0) + 1
        if not instances:
            continue
        index.append(FrameIndex(
            frame=Frame(name=frame.key, image=frame.image, mask=frame.store),
            size=(width, height), instances=tuple(instances), rejects=rejects,
            source=source))
    if not index:
        raise ValueError(
            f"{pool_dir}: every accepted mask failed `InstanceGates` "
            f"{source.gates}. These are teacher masks on detection boxes, so "
            f"the usual cause is `min_area` against a set of very small "
            f"targets.")
    return index


@dataclass(frozen=True)
class PoolRequest:
    """One `--pool` argument, parsed. See `parse_pool` for the syntax."""

    pool: Path
    images: Path | None
    modality: str = "thermal"
    role: str = "train"
    gates: InstanceGates | None = None
    name: str | None = None

    @property
    def label(self) -> str:
        return f"pool:{self.name or self.pool.name}:{self.modality}"

    @property
    def cache_name(self) -> str:
        """A filename for this pool's cached index -- unique per pool + modality."""
        return self.label.replace(":", "_").replace("/", "_")


POOL_MODALITIES = ("thermal", "rgb")


def parse_pool(argument: str, gates: InstanceGates | None = None) -> PoolRequest:
    """`"pool_dir[:images_root[:modality[:role]]]"` -- one repeatable flag.

    Positional with defaults falling off the end, the same shape
    `datasets.parse` takes and for the same reason: a flag that needed four
    more beside it would not be repeatable, and every field after the pool has
    an obvious default. `images_root` may be left empty when the frames still
    sit where the harvest runtime left them, which on a Colab-to-Colab run is
    the common case.

    `role` defaults to **train**, not `all`. A pool's masks are a teacher's
    guess gated four ways; scoring on them measures the teacher as much as the
    model, exactly as `split_index` argues for reconstructed semantic sets. Ask
    for `all` deliberately when the pool is the only thing a run has to grade
    on, and read the number knowing what it is.
    """
    parts = argument.split(":")
    if not parts[0]:
        raise ValueError(
            f"--pool {argument!r} should be pool_dir[:images_root[:modality"
            f"[:role]]], e.g. /content/pool/hituav_thermal:/content/data/HIT_UAV")

    pool = Path(parts[0])
    images = Path(parts[1]) if len(parts) > 1 and parts[1] else None
    modality = parts[2] if len(parts) > 2 and parts[2] else "thermal"
    role = parts[3] if len(parts) > 3 and parts[3] else "train"

    if modality not in POOL_MODALITIES:
        raise ValueError(f"modality must be one of {POOL_MODALITIES}, "
                         f"got {modality!r}")
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    return PoolRequest(pool=pool, images=images, modality=modality, role=role,
                       gates=gates, name=pool.name)


def describe_pools(requests: SequenceABC[PoolRequest]) -> str:
    """One line per pool, so a mixed run says what it is made of."""
    lines = ["| pool | modality | role | images |", "|---|---|---|---|"]
    for request in requests:
        lines.append(f"| `{request.pool}` | {request.modality} | "
                     f"`{request.role}` | `{request.images or 'as recorded'}` |")
    return "\n".join(lines)


def index_pools(requests: Iterable[PoolRequest], cache_dir: str | Path | None = None,
                workers: int = 8, progress=None, log=print) -> list[FrameIndex]:
    """Several pools' indexes, concatenated -- the flag order is the order."""
    index: list[FrameIndex] = []
    for request in requests:
        cache = (Path(cache_dir) / f"{request.cache_name}.json"
                 if cache_dir else None)
        if cache is not None and cache.is_file():
            part = load_pool_index(cache, gates=request.gates, role=request.role)
            log(f"  {request.label}: index reused from {cache}")
        else:
            part = index_pool(request.pool, request.images, request.modality,
                              request.role, request.gates, request.name,
                              workers=workers, progress=progress)
            if cache is not None:
                save_pool_index(cache, part)
                log(f"  {request.label}: index written to {cache}")
        index.extend(part)
    return index


# --------------------------------------------------------------------------
# Caching an index
# --------------------------------------------------------------------------


def save_pool_index(path: str | Path, index: SequenceABC[FrameIndex]) -> Path:
    """Write a pool index, class table included.

    `aerial.save_index` stamps an index with its spec *name* and mode, which is
    all a spec in `SPECS` needs -- the rest of it is in the repo. A pool's spec
    is built from the pool, so the class table has to travel with the file or a
    reused index would report `class 3` where the first run reported `car`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = next((e.source for e in index if e.source), None)
    if stamp is None:
        raise ValueError("a pool index without a Source cannot be cached")
    path.write_text(json.dumps({
        "spec": {"name": stamp.spec.name, "classes": stamp.spec.classes,
                 "border": stamp.spec.border},
        "modality": "thermal" if stamp.gray else "rgb",
        "role": stamp.role,
        "frames": [{"name": e.frame.name, "image": str(e.frame.image),
                    "store": str(e.frame.mask), "size": list(e.size),
                    "rejects": e.rejects,
                    "instances": [[i.label, i.class_id, *i.box, i.area]
                                  for i in e.instances]}
                   for e in index],
    }) + "\n")
    return path


def load_pool_index(path: str | Path, gates: InstanceGates | None = None,
                    role: str | None = None) -> list[FrameIndex]:
    """Read one back. `gates` and `role` may be overridden; the classes may not."""
    payload = json.loads(Path(path).read_text())
    stamp = payload["spec"]
    spec = DatasetSpec(name=stamp["name"], masks="",
                       classes={k: int(v) for k, v in stamp["classes"].items()},
                       things=tuple(sorted(stamp["classes"])),
                       border=int(stamp.get("border", 0)),
                       palette_source=PALETTE_SOURCE)
    source = Source(spec=spec, gates=gates or InstanceGates(), mode="pool",
                    gray=payload["modality"] == "thermal",
                    role=role or payload["role"])
    return [FrameIndex(
        frame=Frame(name=e["name"], image=Path(e["image"]),
                    mask=Path(e["store"])),
        size=(int(e["size"][0]), int(e["size"][1])), rejects=e["rejects"],
        instances=tuple(Instance(label=int(r[0]), class_id=int(r[1]),
                                 box=(r[2], r[3], r[4], r[5]), area=int(r[6]))
                        for r in e["instances"]),
        source=source,
    ) for e in payload["frames"]]


__all__ = ["PALETTE_SOURCE", "POOL_MODALITIES", "PoolFrame", "PoolRequest",
           "Relocator", "describe_pools", "discover_pools", "group_records",
           "index_pool", "index_pools", "link_pool",
           "load_pool_index", "parse_pool", "pool_datasets", "pool_spec",
           "save_pool_index", "store_areas"]
