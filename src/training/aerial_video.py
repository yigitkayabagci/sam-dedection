"""Aerial thermal tracking datasets in the clip-loop's common format.

Stage B already learns aerial thermal appearance from still-image pools.  The
video path needs a different contract: ordered frames, one identity, one box
per frame, and whole flights kept on one side of the train/validation split.

Two public datasets provide that without reversing the deployment viewpoint:

* VTUAV: the thermal camera is on the UAV and every sequence follows one
  target.  The lightweight extraction keeps only the annotated frames (usually
  every tenth source frame); those files and box rows still align by position.
* BIRDSAI: night-time aerial TIR with MOT rows.  One video can contain many
  identities, so each clean contiguous track run becomes one training
  sequence.  Interpolated ``noise=1`` rows split a run rather than becoming a
  false negative, and all identities from one flight stay in the same split.

Both are adapted to :class:`src.training.antiuav.Sequence`; the clip loader and
loss therefore do not gain a dataset-specific branch.  The VTUAV VIS release's
sparse drawn masks and the image-teacher pools harvested by notebooks 17/25
can be attached to the same sequences.  A frame with no accepted mask still
falls back to box projection, rather than being treated as empty foreground.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence as SequenceABC
from functools import lru_cache
from pathlib import Path

import numpy as np

from .antiuav import Sequence, SequenceLabels, frame_shape, sample_clips
from .labels import MASK_STORE, open_masks
from .masklets import read_boxes
from .pool import RECORD_FILE


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def _frame_key(path: Path) -> tuple[int, int, str]:
    stem = path.stem
    tail = stem.rsplit("_", 1)[-1]
    return (0, int(tail), "") if tail.isdigit() else (1, 0, stem)


def source_name(sequence_or_name: Sequence | str) -> str:
    """The stable source prefix embedded in an adapted sequence name."""
    name = str(getattr(sequence_or_name, "name", sequence_or_name))
    return name.split("__", 1)[0]


def flight_name(sequence_or_name: Sequence | str) -> str:
    """Dataset + physical flight, excluding BIRDSAI track/run identifiers."""
    name = str(getattr(sequence_or_name, "name", sequence_or_name))
    parts = name.split("__")
    return "__".join(parts[:2]) if len(parts) > 1 else name


def vtuav_sequences(
    root: str | Path,
    modality: str = "ir",
    prefix: str = "vtuav",
) -> list[Sequence]:
    """VTUAV tracking extractions as single-object sequences.

    ``tools.fetch_datasets.extract(..., frames='tracked_ir')`` leaves frames
    0, 10, 20, ... beside one box row per retained file.  Their *positions*
    align even though their stems are strided, which is precisely the temporal
    sampling intended here.  An absent long-term row remains in place and
    supervises the object score rather than being dropped.
    """
    if modality not in ("ir", "rgb"):
        raise ValueError(f"modality must be ir or rgb, got {modality!r}")
    root = Path(root)
    sequences: list[Sequence] = []
    for box_file in sorted(root.rglob(f"{modality}.txt")):
        folder = box_file.parent / modality
        if not folder.is_dir():
            continue
        frames = tuple(sorted(
            (path for path in folder.iterdir()
             if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
            key=_frame_key,
        ))
        if not frames:
            continue
        boxes, exist = read_boxes(box_file)
        count = min(len(frames), len(boxes))
        if count < 2:
            continue
        sequences.append(Sequence(
            name=f"{prefix}__{box_file.parent.name}",
            split="unsplit",
            frames=frames[:count],
            labels=SequenceLabels(exist=exist[:count], boxes=boxes[:count]),
        ))
    if not sequences:
        raise FileNotFoundError(
            f"{root}: no VTUAV sequence with {modality}/ and {modality}.txt. "
            "Extract tracking archives with --frames tracked_ir first."
        )
    return sequences


@lru_cache(maxsize=128)
def _read_image_mask(path: str) -> np.ndarray:
    """Read one drawn mask lazily; a small cache helps overlapping clips."""
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


@lru_cache(maxsize=128)
def _read_pool_mask(path: str, key: int) -> np.ndarray:
    """Decode one accepted teacher mask lazily from a per-frame RLE store."""
    return open_masks(path)[int(key)]


class ImageMaskStore(Mapping[int, np.ndarray]):
    """Sparse frame-index to PNG masks from VTUAV VIS."""

    def __init__(self, paths: Mapping[int, str | Path]) -> None:
        self.paths = {int(index): str(path) for index, path in paths.items()}

    def __getitem__(self, index: int) -> np.ndarray:
        return _read_image_mask(self.paths[int(index)])

    def __iter__(self):
        return iter(self.paths)

    def __len__(self) -> int:
        return len(self.paths)


class PoolMaskStore(Mapping[int, np.ndarray]):
    """Sparse frame-index to one instance in an image-teacher pool store."""

    def __init__(self, entries: Mapping[int, tuple[str | Path, int]]) -> None:
        self.entries = {
            int(index): (str(path), int(key))
            for index, (path, key) in entries.items()
        }

    def __getitem__(self, index: int) -> np.ndarray:
        path, key = self.entries[int(index)]
        return _read_pool_mask(path, key)

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def _box_from_mask(path: Path) -> np.ndarray | None:
    mask = _read_image_mask(str(path))
    yy, xx = np.nonzero(mask)
    if not xx.size:
        return None
    return np.asarray(
        [xx.min(), yy.min(), xx.max() + 1, yy.max() + 1], dtype=np.float64)


def vtuav_vis_sequences(
    root: str | Path,
    modality: str = "ir",
    prefix: str = "vtuav_vis",
) -> tuple[list[Sequence], dict[str, ImageMaskStore]]:
    """VTUAV VIS mask frames as sparse, long-horizon video sequences.

    ``fetch_datasets.py vtuav_vis --frames masked`` extracts only frames with
    a drawn mask.  Consecutive items here are therefore normally 30 source
    frames apart: deliberately useful long-horizon supervision, but not a
    substitute for the denser tracking split.  Boxes are derived from the
    masks themselves so an all-frame ``*.txt`` can never be misaligned with a
    masked-only extraction.
    """
    if modality not in ("ir", "rgb"):
        raise ValueError(f"modality must be ir or rgb, got {modality!r}")
    root = Path(root)
    sequences: list[Sequence] = []
    stores: dict[str, ImageMaskStore] = {}
    for mask_dir in sorted(root.rglob(f"mask/{modality}")):
        if not mask_dir.is_dir():
            continue
        sequence_dir = mask_dir.parent.parent
        image_dir = sequence_dir / modality
        images = {
            path.stem: path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        } if image_dir.is_dir() else {}
        rows: list[tuple[Path, Path, np.ndarray]] = []
        for mask_path in sorted(
            (path for path in mask_dir.iterdir()
             if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
            key=_frame_key,
        ):
            image_path = images.get(mask_path.stem)
            box = _box_from_mask(mask_path)
            if image_path is not None and box is not None:
                rows.append((image_path, mask_path, box))
        if len(rows) < 2:
            continue
        # The part goes in the name whenever the archives were unpacked into
        # directories of their own. VTUAV-VIS names a sequence by target kind
        # and a counter that restarts per archive, so `train_003` is a train
        # sequence inside `test_001.zip` *and* the name of a training archive;
        # two of them under one prefix would be one key in the store dict and
        # one of the two would silently replace the other. A flat extraction
        # keeps the old names, so nothing already measured moves.
        part = sequence_dir.parent
        stem = (f"{part.name}_{sequence_dir.name}"
                if part != root and part.parent == root else sequence_dir.name)
        name = f"{prefix}__{stem}"
        sequences.append(Sequence(
            name=name,
            split="unsplit",
            frames=tuple(image_path for image_path, _, _ in rows),
            labels=SequenceLabels(
                exist=np.ones(len(rows), dtype=bool),
                boxes=np.stack([box for _, _, box in rows]),
            ),
        ))
        stores[name] = ImageMaskStore({
            index: mask_path for index, (_, mask_path, _) in enumerate(rows)
        })
    if not sequences:
        raise FileNotFoundError(
            f"{root}: no VTUAV VIS sequence with {modality}/ and "
            f"mask/{modality}/. Extract a VIS archive with --frames masked."
        )
    return sequences, stores


def pool_sequence_stores(
    root: str | Path,
    sequences: SequenceABC[Sequence],
    datasets: set[str] | None = None,
    min_box_iou: float = 0.0,
) -> dict[str, PoolMaskStore]:
    """Attach per-frame image-teacher pool masks to video sequences.

    Pool records key VTUAV frames as ``<sequence>/<stem>`` while Stage C keys
    them by their position in a :class:`Sequence`.  This function performs
    that join once.  Only accepted instances are attached; ``min_box_iou`` can
    repeat Stage B's stricter post-harvest cut when that reading is present.
    Rejected or absent frames remain missing and the clip loss uses its
    box/object-score terms.
    """
    by_frame: dict[tuple[str, str], tuple[str, int]] = {}
    for sequence in sequences:
        parts = sequence.name.split("__")
        if len(parts) < 2 or source_name(sequence) != "vtuav":
            continue
        flight = parts[1]
        for index, frame in enumerate(sequence.frames):
            by_frame[(flight, frame.stem)] = (sequence.name, index)

    attached: dict[str, dict[int, tuple[Path, int]]] = defaultdict(dict)
    for record_path in sorted(Path(root).rglob(RECORD_FILE)):
        try:
            record = json.loads(record_path.read_text())
        except (OSError, ValueError):
            continue
        if datasets is not None and str(record.get("dataset")) not in datasets:
            continue
        key_parts = Path(str(record.get("key", ""))).parts
        if len(key_parts) < 2:
            continue
        stem = Path(key_parts[-1]).stem
        match = None
        for flight in reversed(key_parts[:-1]):
            match = by_frame.get((flight, stem))
            if match is not None:
                break
        if match is None:
            image = Path(str(record.get("image", "")))
            flight = image.parent.parent.name if image.parent.name in ("ir", "rgb") else ""
            match = by_frame.get((flight, image.stem))
        if match is None:
            continue
        accepted = []
        for row in record.get("instances", ()):
            if row.get("verdict") is not None or "i" not in row:
                continue
            measured = row.get("box_iou")
            if (min_box_iou > 0 and measured is not None
                    and float(measured) < min_box_iou):
                continue
            accepted.append(int(row["i"]))
        store_path = record_path.parent / MASK_STORE
        if not accepted or not store_path.is_file():
            continue
        sequence_name, index = match
        attached[sequence_name][index] = (store_path, accepted[0])
    return {name: PoolMaskStore(rows) for name, rows in attached.items()}


def _birdsai_frame_positions(folder: Path) -> tuple[list[Path], dict[int, int]]:
    ordered = sorted(
        (path for path in folder.iterdir()
         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=_frame_key,
    ) if folder.is_dir() else []
    positions: dict[int, int] = {}
    for position, path in enumerate(ordered):
        tail = path.stem.rsplit("_", 1)[-1]
        if tail.isdigit():
            positions.setdefault(int(tail), position)
    return ordered, positions


def _contiguous(items: list[tuple[int, np.ndarray]]) -> list[list[tuple[int, np.ndarray]]]:
    runs: list[list[tuple[int, np.ndarray]]] = []
    current: list[tuple[int, np.ndarray]] = []
    for item in sorted(items, key=lambda row: row[0]):
        if current and item[0] != current[-1][0] + 1:
            runs.append(current)
            current = []
        current.append(item)
    if current:
        runs.append(current)
    return runs


def birdsai_sequences(
    root: str | Path,
    split: str = "TrainReal",
    drop_noisy: bool = True,
    drop_occluded: bool = False,
    min_run: int = 8,
    prefix: str = "birdsai",
) -> list[Sequence]:
    """BIRDSAI MOT annotations as one sequence per clean identity run.

    A noisy/interpolated row is unknown supervision, not evidence that the
    target disappeared.  Dropping it from a full-length label vector would
    teach object-score=0, so it instead breaks the run.  The same applies to a
    missing annotation.  Occlusion rows stay by default: surviving occlusion is
    one of the reasons to train the memory path at all.
    """
    root = Path(root)
    annotation_dirs = [p for p in sorted(root.rglob(f"{split}/annotations"))
                       if p.is_dir()]
    if not annotation_dirs:
        raise FileNotFoundError(
            f"{root}: no {split}/annotations; fetch the BIRDSAI real split first."
        )

    sequences: list[Sequence] = []
    for annotation_dir in annotation_dirs:
        images_root = annotation_dir.parent / "images"
        for csv_path in sorted(annotation_dir.glob("*.csv")):
            ordered, by_number = _birdsai_frame_positions(images_root / csv_path.stem)
            if not ordered:
                continue
            tracks: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
            with csv_path.open(newline="") as stream:
                for cells in csv.reader(stream):
                    if len(cells) < 10:
                        continue
                    try:
                        number, track_id = int(float(cells[0])), int(float(cells[1]))
                        x, y, width, height = (float(value) for value in cells[2:6])
                        occluded, noisy = int(float(cells[8])), int(float(cells[9]))
                    except (TypeError, ValueError):
                        continue
                    if width <= 0 or height <= 0:
                        continue
                    if (drop_noisy and noisy) or (drop_occluded and occluded):
                        continue
                    position = by_number.get(number)
                    if position is None and 0 <= number < len(ordered):
                        position = number
                    if position is None:
                        continue
                    tracks[track_id].append((
                        position,
                        np.asarray([x, y, x + width, y + height], dtype=np.float64),
                    ))

            for track_id, rows in sorted(tracks.items()):
                for run_index, run in enumerate(_contiguous(rows)):
                    if len(run) < max(int(min_run), 2):
                        continue
                    positions = [position for position, _ in run]
                    boxes = np.stack([box for _, box in run])
                    sequences.append(Sequence(
                        name=(f"{prefix}__{csv_path.stem}__track{track_id}"
                              f"__run{run_index}"),
                        split="unsplit",
                        frames=tuple(ordered[position] for position in positions),
                        labels=SequenceLabels(
                            exist=np.ones(len(run), dtype=bool), boxes=boxes),
                    ))
    if not sequences:
        raise ValueError(
            f"{root}: {split} was found but no clean track run reached {min_run} frames."
        )
    return sequences


def split_flights(
    sequences: SequenceABC[Sequence],
    seed: int = 0,
    val_fraction: float = 0.2,
    test_fraction: float = 0.1,
    hold_out: SequenceABC[Sequence] | None = None,
) -> dict[str, list[Sequence]]:
    """Whole-flight, per-source train/val/test split.

    BIRDSAI identities from the same physical video share pixels and must not
    straddle splits.  Sources are allocated separately so a small source does
    not accidentally disappear from validation behind a much larger one.

    **`hold_out` is a test set that was decided elsewhere**, and it replaces
    the sampled one rather than joining it. The case is a dataset that ships
    its own held-out split: VTUAV-VIS's `test_00x` archives are the authors'
    choice of which sequences nobody trains on, and a hash-ordered sample over
    everything would put some of them in `train` and some training sequences in
    `test` -- a grade that measures the sampler as much as the model. Passing
    them here keeps `test_fraction` out of it: the fractions then divide only
    what is left, and `test_fraction=0` is the honest setting beside a
    `hold_out`.

    Nothing checks that the two sets are disjoint, because nothing here can:
    the sequences carry no provenance beyond their names. `Part.into` and the
    part-qualified names from `vtuav_vis_sequences` are what make two archives'
    identically-named sequences distinguishable in the first place.
    """
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val/test fractions must be non-negative and sum below one.")

    by_source: dict[str, dict[str, list[Sequence]]] = defaultdict(lambda: defaultdict(list))
    for sequence in sequences:
        by_source[source_name(sequence)][flight_name(sequence)].append(sequence)

    result = {"train": [], "val": [], "test": []}
    for source, flights in sorted(by_source.items()):
        names = sorted(flights, key=lambda name: hashlib.sha256(
            f"{seed}:{source}:{name}".encode()).digest())
        count = len(names)
        n_test = min(count - 1, max(1, round(count * test_fraction))) if test_fraction else 0
        remaining = count - n_test
        n_val = min(remaining - 1, max(1, round(count * val_fraction))) if val_fraction else 0
        allocation = {
            "test": names[:n_test],
            "val": names[n_test:n_test + n_val],
            "train": names[n_test + n_val:],
        }
        if not allocation["train"]:
            raise ValueError(f"{source}: {count} flight(s) cannot form a non-empty train split.")
        for part, flight_names in allocation.items():
            for name in flight_names:
                result[part].extend(flights[name])
    if hold_out is not None:
        # Replaces rather than joins: a sampled test set beside a given one
        # would grade on a mixture, and which half a number came from would
        # not be recoverable from the number.
        result["test"] = list(hold_out)
    return result


def video_clips(
    sequences: SequenceABC[Sequence],
    length: int = 8,
    stride: int = 1,
    size: int = 512,
    min_visible: int = 2,
    jitter: int = 0,
    seed: int | None = None,
) -> list:
    """Sample clips while deriving each sequence's own source resolution."""
    clips = []
    for offset, sequence in enumerate(sequences):
        height, width = frame_shape(sequence.frames[0])
        clips.extend(sample_clips(
            [sequence], length=length, stride=stride, size=size,
            frame_size=(width, height), min_visible=min_visible,
            jitter=jitter, seed=None if seed is None else seed + offset,
        ))
    return clips


def weighted_clip_sample(
    clips: SequenceABC,
    weights: Mapping[str, float],
    total: int,
    seed: int = 0,
) -> list:
    """A reproducible source-balanced clip pool, sampling with replacement."""
    groups: dict[str, list] = defaultdict(list)
    for clip in clips:
        groups[source_name(clip.sequence)].append(clip)
    active = {name: float(weight) for name, weight in weights.items()
              if weight > 0 and groups.get(name)}
    if not active:
        raise ValueError(f"none of the weighted sources are present: {sorted(groups)}")
    norm = sum(active.values())
    rng = np.random.default_rng(seed)
    names = sorted(active)
    counts = {name: int(np.floor(total * active[name] / norm)) for name in names}
    for name in names[:max(0, total - sum(counts.values()))]:
        counts[name] += 1
    out = []
    for name in names:
        group = groups[name]
        picks = rng.integers(0, len(group), size=counts[name])
        out.extend(group[int(index)] for index in picks)
    rng.shuffle(out)
    return out


def empty_stores(sequences: SequenceABC[Sequence]) -> dict[str, dict]:
    """Box-only Stage-C supervision in the mapping shape ``prefetch`` wants."""
    return {sequence.name: {} for sequence in sequences}
