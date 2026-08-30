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

import hashlib
import itertools
import json
import threading
import zipfile
from collections.abc import Iterable, Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .aerial import (ROLES, DatasetSpec, Frame, FrameIndex, Instance,
                     InstanceGates, Source, group_of, reject_reason)
from .labels import MASK_STORE
from .pool import RECORD_FILE

# Why a frame never reached the index. Kept apart from `InstanceGates`' names
# so a caller can tell "these frames were not usable" from "these instances
# were not worth training on" -- the first is usually a missing download and
# the second is the gates doing their job.
SKIP_REASONS = ("no_store", "no_image", "unreadable_image", "shape_mismatch",
                "store_disagrees", "record_schema", "no_accepted")

# How much of a file `_same_frames` reads before calling two copies the same.
# Size plus the first megabyte separates a duplicate from a re-encode -- two
# JPEGs of one scene differ in their first table -- without reading SegFly's
# 11 MB frames end to end for a decision about a directory name.
PREFIX_BYTES = 1 << 20

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

    **Stripping is not enough when the tree gained a level.** Suffix matching
    can only drop leading components, so a record written as
    `/content/data/HIT_UAV/normal_json/train/0_01.jpg` misses every candidate
    once the archive unpacks one folder deeper
    (`HIT_UAV/HIT-UAV-...-main/normal_json/train/0_01.jpg`) -- and misses it
    for all 2 866 frames, which reads exactly like a dataset that was never
    downloaded. So a miss falls back to an index of the root by file name,
    built once and shared by every frame after it. Where a name repeats --
    DroneVehicle keeps `trainimg/04991.jpg` beside `trainimgr/04991.jpg`, one
    per modality -- the candidate sharing the longest tail with the recorded
    path wins, and a tie is left unresolved rather than guessed: picking the
    wrong modality's frame would train thermal masks on RGB pixels.
    `found_by_name` counts what came back this way, because a run that needed
    it is a run whose images moved.
    """

    def __init__(self, images_root: str | Path | None,
                 by_name: bool = True) -> None:
        self.root = Path(images_root) if images_root else None
        self.depth: int | None = None
        self.misses = 0
        self.found_by_name = 0
        self.ambiguous = 0
        self._by_name = by_name
        self._names: dict[str, list[Path]] | None = None
        self._lock = threading.Lock()

    def direct(self, relative: str | Path | None) -> Path | None:
        """`root / relative`, when the record carried an archive-relative path.

        `aerovis.write_pool` writes `image_rel` beside `image` for exactly this
        reason -- its stores are a few hundred megabytes and its frames are
        12.6 GiB, so the pool reaches Drive without them and a training run
        re-points at its own copy. Where a record says where the frame sits
        *inside its archive*, that is an answer rather than a search, and it
        cannot pick the wrong file the way a suffix match can.
        """
        if not relative or self.root is None:
            return None
        candidate = self.root / str(relative)
        return candidate if candidate.is_file() else None

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
        return self._named(parts) or self._miss()

    def _named(self, parts: tuple[str, ...]) -> Path | None:
        """The file under the root that carries this name, when one does.

        The index is built on the first miss rather than up front, so a pool
        whose paths all resolve by stripping never pays for the walk.
        """
        if not self._by_name or self.root is None or not parts:
            return None
        with self._lock:
            if self._names is None:
                self._names = _index_by_name(self.root)
        candidates = self._names.get(parts[-1], ())
        if not candidates:
            return None
        if len(candidates) > 1:
            ranked = sorted(candidates, key=lambda p: -_shared_tail(p.parts, parts))
            if _shared_tail(ranked[0].parts, parts) \
                    == _shared_tail(ranked[1].parts, parts):
                self.ambiguous += 1
                return None
            candidates = ranked
        self.found_by_name += 1
        return candidates[0]

    def _miss(self) -> None:
        self.misses += 1
        return None


def _index_by_name(root: Path) -> dict[str, list[Path]]:
    """`{file name: paths}` for everything under `root`, walked once."""
    names: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file():
            names.setdefault(path.name, []).append(path)
    return names


def _shared_tail(left: SequenceABC[str], right: SequenceABC[str]) -> int:
    """How many trailing components two paths have in common."""
    shared = 0
    for one, other in zip(reversed(left), reversed(right)):
        if one != other:
            break
        shared += 1
    return shared


def resolve_images_root(records: Iterable[str | Path],
                        images_root: str | Path | None,
                        search_root: str | Path | None = None,
                        probes: int = 5) -> Path | None:
    """The root a pool's recorded paths were written against, read off the disk.

    `Relocator` handles a tree that gained or lost levels, one frame at a time,
    and refuses a name that is ambiguous under the root it was given -- which is
    the right call per frame and the wrong one for a *whole pool*: HIT-UAV's
    archive re-packs itself, so `normal_json/train/0_01.jpg` exists twice under
    `/content/data/HIT_UAV` and every frame of the pool ties. Naming the inner
    tree once settles all of them, and this works that name out instead of
    asking a settings cell to hard-code a path that depends on which mirror the
    pool was harvested from and how the archive happened to unpack.

    Probes are spread over the pool rather than taken off the front, and each
    one's file name is looked up under `search_root` (the configured root by
    default). Every hit proposes the root sitting in front of the tail it shares
    with the recorded path, and the proposals are ranked by:

    1. how much of the recorded path they agree with -- `normal_json/train/f`
       shares two components with a record naming `hit-uav/images/train/f`
       where `JPEGImages/f` shares one, and the deeper agreement is the better
       claim, which is `Relocator`'s rule too;
    2. how many probes they actually re-root, which tells the tree holding the
       frames from a wrapper holding one stray copy;
    3. depth, since a shallow root only reaches those frames through the deep
       one.

    Two answers are deliberately *not* a re-rooting. Nothing beating the
    configured root returns that root unchanged: frames that missed under a
    root which resolves the others are missing, not misplaced, and widening the
    root would hide a half-finished download behind a search. And roots that
    tie while holding *different pixels* return `None` rather than a guess --
    four copies of one archive are one answer, DroneVehicle's two modalities
    are not. `None` also means no frame of the pool is anywhere underneath,
    which is the missing-download case and a fact to report.
    """
    # A root can still carry the glob it was configured with, when nothing
    # matched it -- which is what a pattern resolved before the download looks
    # like. Its solid prefix is a real directory and the right place to search.
    parts = Path(images_root).parts if images_root else ()
    solid = list(itertools.takewhile(lambda part: "*" not in part, parts))
    configured = Path(*parts) if parts and len(solid) == len(parts) else None
    base = Path(search_root) if search_root else (
        Path(*solid) if solid else configured)
    if base is None or not base.is_dir():
        return None

    every = list(records)
    spread_out = every[::max(len(every) // max(probes, 1), 1)][:max(probes, 1)]
    wanted: list[Path] = []
    for record_path in spread_out:
        try:
            wanted.append(Path(json.loads(
                Path(record_path).read_text())["image"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if not wanted:
        return None

    def resolves(root: Path) -> int:
        relocate = Relocator(root, by_name=False)
        return sum(1 for recorded in wanted if relocate(recorded) is not None)

    standing = resolves(configured) if configured is not None else -1
    if standing == len(wanted):
        return configured                    # the configured root already works

    names = _index_by_name(base)
    proposed: dict[Path, int] = {}
    for recorded in wanted:
        for path in names.get(recorded.name, ()):
            shared = max(_shared_tail(path.parts, recorded.parts), 1)
            root = Path(*path.parts[:len(path.parts) - shared])
            proposed[root] = max(proposed.get(root, 0), shared)
    if not proposed:
        return None
    # Agreement with the recorded path first -- `normal_json/train/f` against a
    # record naming `hit-uav/images/train/f` shares two components where
    # `JPEGImages/f` shares one, and the deeper agreement is the better claim,
    # which is `Relocator`'s rule too. Then how many probes the root actually
    # re-roots, which separates the tree holding the frames from a wrapper
    # holding one stray copy; then the deeper root, since a shallow one only
    # reaches those frames through it.
    rank = {root: (shared, resolves(root), len(root.parts))
            for root, shared in proposed.items()}
    ranked = sorted(rank, key=lambda root: (tuple(-v for v in rank[root]),
                                            str(root)))
    best = [root for root in ranked if rank[root] == rank[ranked[0]]]
    if rank[best[0]][1] <= standing:
        # Nothing found more frames than the root the run was configured with,
        # so the frames that missed are missing rather than misplaced. Widening
        # the root would hide a half-finished download behind a search, and a
        # root one level up can match a *different* dataset's frame of the same
        # name -- the pools are full of `000001.jpg`.
        return configured
    if len(best) > 1 and not _same_frames(best, wanted):
        return None
    return best[0]


def _same_frames(roots: SequenceABC[Path], wanted: SequenceABC[Path]) -> bool:
    """Whether every root re-roots the probes onto byte-identical files.

    Several roots resolving the same number of frames is not by itself a
    problem: HIT-UAV's archive ships each frame four times (`normal_json/`,
    `rotate_json/` and two `JPEGImages/` trees) because the annotations come in
    four formats, and three of those copies are the same bytes -- so refusing
    the pool over the ambiguity, which is what a per-frame reader must do,
    throws away 2 866 frames to protect nothing.

    Where the copies differ, the refusal is the right answer and stays:
    DroneVehicle keeps `trainimg/04991.jpg` beside `trainimgr/04991.jpg`, one
    per modality, and a run that picked the wrong one would train thermal masks
    on RGB pixels and never say so.
    """
    for recorded in wanted:
        digests = set()
        for root in roots:
            found = Relocator(root, by_name=False)(recorded)
            if found is None:
                return False
            try:
                with found.open("rb") as handle:
                    head = handle.read(PREFIX_BYTES)
                digests.add((found.stat().st_size,
                             hashlib.md5(head).hexdigest()))
            except OSError:
                return False
        if len(digests) > 1:
            return False
    return True


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


def acceptance(records: SequenceABC[Path]) -> dict:
    """What a harvest actually produced: boxes attempted, kept, and why not.

    The number nobody was reading, and the one that decides whether a pool is
    worth its download. A rejected box writes **nothing** to the store, so a
    pool of forty thousand frames whose teacher refused nine tenths of them is
    a pool of four thousand masks -- and from the outside it looks the same as
    a pool of forty thousand, because the frame directories are all there.

    `rejected` is keyed by the gate that stopped the box, and the order
    matters: `reject_reason` returns at the **first** gate that fails, so a
    box counted under `teacher_iou` was never measured against its own
    annotation. A pool whose rejects pile up there has not been shown to carry
    bad masks; it has been shown to carry masks the teacher was unsure of.
    """
    counts = {"frames": 0, "attempted": 0, "accepted": 0}
    rejected: dict[str, int] = {}
    classes: dict[str, int] = {}
    teachers: set[str] = set()
    for path in records:
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        counts["frames"] += 1
        teachers.add(str(record.get("teacher", "?")))
        for instance in record.get("instances", ()):
            counts["attempted"] += 1
            verdict = instance.get("verdict")
            if verdict is None:
                counts["accepted"] += 1
                name = str(instance.get("class", "?"))
                classes[name] = classes.get(name, 0) + 1
            else:
                rejected[str(verdict)] = rejected.get(str(verdict), 0) + 1
    return {**counts,
            "rate": counts["accepted"] / counts["attempted"]
                    if counts["attempted"] else 0.0,
            "rejected": dict(sorted(rejected.items(), key=lambda kv: -kv[1])),
            "accepted_by_class": dict(sorted(classes.items(),
                                             key=lambda kv: -kv[1])),
            "teachers": sorted(teachers)}


def spread(index: SequenceABC[FrameIndex], limit: int | None,
           seed: int = 0) -> list[FrameIndex]:
    """`limit` entries, chosen to sit as far apart as the pool allows.

    A cap is only useful if what it keeps is a *sample*, and a uniform draw
    over frames is not one here. AeroVIS is sequences with a median track of
    117 frames, so 10 000 drawn uniformly out of 39 943 is roughly a quarter
    of every sequence -- and a quarter of a sequence is mostly neighbouring
    frames, which are the same picture. The batch would be large and its
    variety would not.

    So the cap is spent in two passes. **Across** sequences first, round-robin
    from a seeded order, so a hundred sequences each give roughly the same
    number and no sequence is dropped while another gives thousands.
    **Within** a sequence second, at an even stride rather than at random,
    because evenly spaced frames are the furthest apart the sequence can
    offer and random ones cluster by chance.

    Ungrouped pools -- detection sets whose frames have no sequence in their
    key -- fall out as one frame per group, which makes the round-robin a
    plain stride over the whole pool. That is the right answer there too.
    """
    if limit is None or limit >= len(index):
        return list(index)
    if limit <= 0:
        return []

    groups: dict[str, list[FrameIndex]] = {}
    for entry in index:
        groups.setdefault(group_of(entry), []).append(entry)
    names = sorted(groups)
    for name in names:
        groups[name].sort(key=lambda entry: entry.frame.name)

    # Seeded so the frames left over when `limit` does not divide evenly do
    # not always land on the alphabetically first sequences.
    order = [int(i) for i in np.random.default_rng(seed).permutation(len(names))]
    counts = dict.fromkeys(names, 0)
    remaining = limit
    while remaining > 0:
        moved = False
        for position in order:
            name = names[position]
            if counts[name] >= len(groups[name]):
                continue
            counts[name] += 1
            remaining -= 1
            moved = True
            if remaining == 0:
                break
        if not moved:                       # every group is exhausted
            break

    chosen: list[FrameIndex] = []
    for name in names:
        take, members = counts[name], groups[name]
        if not take:
            continue
        step = len(members) / take
        chosen += [members[min(int(i * step), len(members) - 1)]
                   for i in range(take)]
    return sorted(chosen, key=lambda entry: entry.frame.name)


def frame_keys(index: SequenceABC[FrameIndex]) -> set[str]:
    """The frames an index covers, as keys comparable across two readers.

    A pool names a frame by the key its harvest recorded and a semantic set
    names it by `list_frames`' own key, and the two agree on the stem and
    nothing else -- one may carry a split directory, the other a sequence. The
    stem, lowercased, is what both always have.
    """
    return {Path(entry.frame.name).stem.lower() for entry in index}


def exclude_frames(index: SequenceABC[FrameIndex],
                   keys: SequenceABC[str] | set[str]) -> list[FrameIndex]:
    """`index` without the frames in `keys` -- the anti-leak filter.

    A pool harvested from a dataset that is *also* the run's held-out grade is
    the same pixels twice: the pool trains on them and the grade scores on
    them, and `split_index` cannot prevent it because the two are separate
    sources with separate permutations. Dropping the graded frames from the
    pool costs the pool its val/test share and keeps every training frame it
    has, which is a far better trade than dropping the pool or the grade.
    """
    wanted = {str(k).lower() for k in keys}
    return [entry for entry in index
            if Path(entry.frame.name).stem.lower() not in wanted]


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


def _example_image(record_path: Path) -> str:
    """The image path one record asks for, for an error that names it."""
    try:
        return str(json.loads(record_path.read_text()).get("image", "?"))
    except Exception:                        # noqa: BLE001 - diagnostics only
        return "?"


def _read_frame(record_path: Path, relocate: Relocator,
                areas_of=store_areas,
                min_box_iou: float = 0.0) -> PoolFrame | str:
    """One `record.json` + its store, or the name of what went wrong.

    `min_box_iou` is a **second, stricter** cut on the gate the harvest already
    applied. The harvest writes every instance's four gate readings beside its
    verdict, so tightening the box-IoU threshold after the fact is a pass over
    the records rather than another run of the teacher -- which on a pool of a
    hundred thousand frames is the difference between a minute and a day. It
    can only ever remove instances: a harvest run at 0.6 has no record of what
    0.5 would have kept. An instance from an older harvest, written before the
    readings were stored, is kept rather than dropped -- silently discarding a
    pool because its records are a version behind is the worse failure.
    """
    record = json.loads(record_path.read_text())
    store = record_path.parent / MASK_STORE
    if not store.is_file():
        return "no_store"
    image = relocate.direct(record.get("image_rel")) or relocate(record["image"])
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
        try:
            key = int(instance["i"])
            if "verdict" in instance:
                if instance["verdict"] is not None:
                    continue
                measured = instance.get("box_iou")
                if (min_box_iou > 0 and measured is not None
                        and float(measured) < min_box_iou):
                    continue
                if key not in areas:
                    return "store_disagrees"
            elif key not in areas:
                # A harvest that wrote no verdicts leaves the store as the only
                # statement of which boxes the gates kept: every writer here
                # stores the accepted masks and only those, so a box with no
                # mask is a rejected box, not a store that disagrees.
                continue
            rows.append((key, str(instance["class"]),
                         tuple(float(v) for v in instance["box"]), areas[key]))
        except (KeyError, TypeError, ValueError):
            # A record shaped by some other harvest. Naming it is worth more
            # than a traceback that ends a half-hour of indexing.
            return "record_schema"
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


def wanted_frames(records: SequenceABC[Path]) -> dict[str, set[str]]:
    """`{basename: {recorded path}}` over a pool's records -- what to extract.

    A pool names every frame it needs and nothing else, so it is a
    manifest. Keying by basename first makes the membership test against an
    archive's name list one dict hit per member rather than a scan.
    """
    wanted: dict[str, set[str]] = {}
    for record_path in records:
        recorded = json.loads(record_path.read_text()).get("image")
        if recorded:
            posix = Path(recorded).as_posix()
            wanted.setdefault(Path(posix).name, set()).add(posix)
    return wanted


def extract_frames(records: SequenceABC[Path], archives: SequenceABC[Path],
                   target: str | Path, progress=None) -> dict:
    """Only the frames `records` asks for, out of `archives`, into `target`.

    VTUAV's tracking split is 214 GiB across fifteen archives and annotates
    every tenth frame, so a pool built from it wants a few percent of what the
    archives hold. Extracting all of it to reach that few percent is a disk
    bill nobody has to pay: the pool *is* the manifest, and a zip's central
    directory can be read without inflating a byte.

    A member is taken when a recorded path ends with it -- exact, and it cannot
    pick the wrong modality the way a basename match could, because the
    recorded path carries `.../ir/000000.jpg` and so does the member. Members
    already on disk with a non-zero size are skipped, which makes a second run
    the resume path.
    """
    target = Path(target)
    wanted = wanted_frames(records)
    report = {"asked": sum(len(v) for v in wanted.values()), "taken": 0,
              "already": 0, "unreadable": [], "by_archive": {}, "missing": 0}
    still = {name: set(paths) for name, paths in wanted.items()}
    stream = archives if progress is None else progress(
        archives, total=len(archives), desc="archives")
    for archive in stream:
        took = 0
        with zipfile.ZipFile(archive) as handle:
            for member in handle.namelist():
                if member.endswith("/"):
                    continue
                base = member.rsplit("/", 1)[-1]
                candidates = still.get(base)
                if not candidates:
                    continue
                hit = next((c for c in candidates
                            if c.endswith("/" + member) or c == member), None)
                if hit is None:
                    continue
                candidates.discard(hit)
                landing = target / member
                if landing.is_file() and landing.stat().st_size:
                    report["already"] += 1
                    took += 1
                    continue
                try:
                    handle.extract(member, target)
                    report["taken"] += 1
                    took += 1
                except Exception:
                    report["unreadable"].append(member)
        report["by_archive"][Path(archive).name] = took
    report["missing"] = sum(len(v) for v in still.values())
    return report


def why_no_image(pool_dir: str | Path, images_root: str | Path | None,
                 sample: int = 3) -> dict:
    """Why a pool's frames could not be found, in terms of the two paths.

    `no_image` on every record has three causes and they need different fixes:
    the root is not there at all (the archive was never fetched), the root is
    there but holds a different part of the set (a thermal export standing in
    for an RGB one), or the frames are there under a name the recorded path
    does not end with. Guessing between them costs a re-index each time, so
    this reads one record and the filesystem and says which it is.
    """
    pool_dir = Path(pool_dir)
    records = sorted(pool_dir.rglob(RECORD_FILE))[:max(sample, 1)]
    report: dict = {"pool": str(pool_dir), "records": len(records),
                    "root": str(images_root) if images_root else None,
                    "recorded": [], "verdict": "", "root_exists": False,
                    "files_under_root": 0, "extensions": {}}
    if not records:
        report["verdict"] = "no records here at all -- the pool zip is missing"
        return report
    for record_path in records:
        record = json.loads(record_path.read_text())
        report["recorded"].append({"image": record.get("image"),
                                   "image_rel": record.get("image_rel")})
    root = Path(images_root) if images_root else None
    if root is None:
        report["verdict"] = "no images root given -- pass one on the --pool flag"
        return report
    report["root_exists"] = root.is_dir()
    if not report["root_exists"]:
        report["verdict"] = (f"{root} does not exist -- the archive was never "
                             f"fetched or went somewhere else")
        return report
    counted: dict[str, int] = {}
    total = 0
    for found in root.rglob("*"):
        if found.is_file():
            total += 1
            counted[found.suffix.lower()] = counted.get(found.suffix.lower(), 0) + 1
            if total >= 200_000:
                break
    report["files_under_root"] = total
    report["extensions"] = dict(sorted(counted.items(), key=lambda kv: -kv[1])[:6])
    if not total:
        report["verdict"] = f"{root} is empty -- the archive unpacked nowhere"
        return report
    stem = Path(report["recorded"][0]["image"] or "").name
    matches = [str(m) for m in root.rglob(stem)][:3] if stem else []
    report["same_name_here"] = matches
    if matches:
        report["verdict"] = (
            f"the frame is here as {matches[0]} but its recorded path does not "
            f"end with the same components -- the root is one level off")
    else:
        report["verdict"] = (
            f"{root} holds {total} files and none is named {stem}. This root "
            f"is a different part of the set than the pool was harvested from")
    return report


def index_pool(
    pool_dir: str | Path,
    images_root: str | Path | None = None,
    modality: str = "thermal",
    role: str = "all",
    gates: InstanceGates | None = None,
    name: str | None = None,
    limit: int | None = None,
    workers: int = 8,
    min_box_iou: float = 0.0,
    progress=None,
    records: SequenceABC[Path] | None = None,
    report=None,
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

    `min_box_iou` re-cuts the harvest's own box-IoU gate, upwards only. The
    records carry every gate reading, not just the verdict they produced, so
    "actually, only keep masks whose box agrees with the annotation at 0.7"
    costs a pass over the index instead of a second run of the teacher. A frame
    left with no instance under the stricter cut is dropped as `no_accepted`,
    the same as one the harvest itself emptied.

    `report` is called with one line of prose when the frames were not where
    the records said -- a relocation that needed the by-name fallback, or one
    that gave up. Nothing else here prints, so a caller that wants silence
    passes nothing and reads the counts instead.
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
        stream = pool.map(
            lambda r: _read_frame(r, relocate, min_box_iou=min_box_iou),
            records)
        if progress is not None:
            stream = progress(stream, total=len(records), desc=pool_dir.name)
        results = list(stream)

    frames = [r for r in results if isinstance(r, PoolFrame)]
    skipped: dict[str, int] = {}
    for result in results:
        if isinstance(result, str):
            skipped[result] = skipped.get(result, 0) + 1
    if relocate.found_by_name and report is not None:
        report(f"   {pool_dir.name}: {relocate.found_by_name} frame(s) found by "
               f"name under {images_root} -- the tree moved since the harvest, "
               f"so the recorded paths no longer strip down to it.")
    if relocate.ambiguous and report is not None:
        report(f"   {pool_dir.name}: {relocate.ambiguous} frame(s) skipped "
               f"because two files under {images_root} carry that name and "
               f"nothing in the record picks one. DroneVehicle's two "
               f"modalities do this -- point --images at one half.")
    if not frames:
        raise ValueError(
            f"{pool_dir}: {len(records)} records and not one usable frame "
            f"({skipped}). `no_image` means --images points somewhere the "
            f"frames are not; `shape_mismatch` means it points at a different "
            f"copy of them. The first record wants "
            f"{_example_image(records[0])!r} and nothing of that name is under "
            f"{images_root}.")

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
           "Relocator", "SKIP_REASONS", "acceptance", "describe_pools",
           "discover_pools",
           "exclude_frames", "frame_keys", "group_records",
           "index_pool", "index_pools", "link_pool", "spread",
           "extract_frames", "wanted_frames", "why_no_image",
           "load_pool_index", "parse_pool", "pool_datasets", "pool_spec",
           "save_pool_index", "store_areas"]
