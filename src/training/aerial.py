"""Static aerial / RGB-T segmentation sets, read as **prompted instances**.

`antiuav.py`'s sibling. That one reads video with one box per frame; this one
reads single images with a dense semantic map, and the difference between them
is the whole reason this module exists.

**The label these datasets ship is the wrong target, and the fix is not a
different dataset.** Kust4K, SegFly and Caltech Aerial RGB-T are *semantic*
segmentation: pixel to class, so "every car is car". EdgeTAM needs the opposite
-- *instance* discrimination: this car, not the one parked beside it. Training a
promptable segmenter on a semantic map actively pulls two adjacent cars'
features together, which is the failure the tracker then inherits.

SAM 2 never sees a semantic map either. Its recipe samples up to 64 individual
masks per image and prompts each one separately, dropping masks that cover more
than ~90 % of the frame. So the fix is to produce that shape of target from the
data we have:

1. **Decompose the semantic map into connected components, one class at a
   time.** Per class, not over the union: a car touching a pedestrian would
   otherwise merge into one component spanning two classes, which is worse than
   the problem being solved.
2. **Keep only the "thing" classes.** Kust4K labels eight, of which four are
   trackable objects (motorcycle, car, truck, person). Road, building and
   vegetation are "stuff" -- they have no instances and are not tracking
   targets, so a component of them is not a training example.
3. **Gate what survives** (`InstanceGates`). Too small to be a target, too
   large to be an object, a sliver whose bounding box it barely fills -- each
   is a reason to distrust the component, and `decompose` reports which gate
   rejected what rather than silently dropping it. That report *is* the answer
   to the open question this whole stage rests on: does connected-component
   decomposition yield clean instances, or do two parked cars fuse into one?

**One window per image, several prompts inside it.** The encoder is the
expensive part -- 38.7 GFLOP at 512 -- and it does not depend on the prompt, so
an image is encoded once and every instance inside the window is prompted
against those same features. That is both cheaper and a better training signal:
the batch contains several objects that share a scene, which is exactly the
discrimination the loss then has to make.

**Windows are native pixels wherever the frame allows it.** A 512 window on a
640x512 thermal frame is the deployment's `crop512` input with no resampling at
all, and `crop_window`/`map_boxes` are shared with `antiuav.py` so the geometry
cannot drift between the video path and this one.

Pure Python and numpy at module level -- the decomposition, the gates and the
window geometry are testable with no OpenCV, no EdgeTAM and no GPU. cv2 and
torch are imported inside the three functions that touch pixels.
"""
from __future__ import annotations

import json
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .antiuav import MEAN, STD, crop_window, map_boxes

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
INDEX_FILE = "instances.json"


# --------------------------------------------------------------------------
# What a dataset looks like on disk
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """Where the files are, what the pixel values mean, and which are things.

    Declarative on purpose. Every one of these sets ships a slightly different
    directory layout and a different palette, and the alternative to a spec is
    a reader per dataset with the same logic copied into each. `probe_classes`
    reads the values actually present so a spec can be checked against the
    download rather than trusted.

    `thermal` and `rgb` are both globs because the RGB-T sets ship registered
    pairs: the thermal half is what the encoder is trained on, and the RGB half
    is the teacher's input in the distillation stage (`src/training/distill.py`).
    """

    name: str
    masks: str                              # glob, relative to the root
    classes: dict[str, int]                 # class name -> value in the map
    things: tuple[str, ...]                 # which of them are trackable
    thermal: str | None = None              # glob, or None if the set has none
    rgb: str | None = None
    strip: tuple[str, ...] = ()             # suffixes cut from a mask stem
    ignore: tuple[int, ...] = ()            # values that are neither thing nor stuff

    def glob(self, modality: str) -> str:
        """The image glob for `modality`, or a readable failure."""
        available = {"thermal": self.thermal, "rgb": self.rgb}
        if modality not in available:
            raise ValueError(f"modality must be thermal or rgb, got {modality!r}")
        pattern = available[modality]
        if pattern is None:
            raise ValueError(f"{self.name} has no {modality} images.")
        return pattern

    @property
    def thing_ids(self) -> tuple[int, ...]:
        missing = [t for t in self.things if t not in self.classes]
        if missing:
            raise KeyError(f"{self.name}: things not in classes: {missing}")
        return tuple(self.classes[name] for name in self.things)

    def name_of(self, value: int) -> str:
        for name, v in self.classes.items():
            if v == value:
                return name
        return f"class {value}"


# The palettes below are the ones the papers document. They are *assumptions
# about a download*, which is a different thing from a fact -- run
# `probe_classes` on the extracted masks before training and fix the spec with
# `dataclasses.replace` if the values differ. The notebook does exactly that.
SPECS: dict[str, DatasetSpec] = {
    # 4 024 registered RGB-TIR pairs at 640x512, 8 classes (Sci. Data 2025).
    # 640x512 is the project's native resolution: a 512 window is a crop, not a
    # resize, and it is the same input the Orin deployment feeds the model.
    "kust4k": DatasetSpec(
        name="kust4k",
        thermal="**/tir/*.png",
        rgb="**/rgb/*.png",
        masks="**/label/*.png",
        classes={"background": 0, "road": 1, "building": 2, "vegetation": 3,
                 "motorcycle": 4, "car": 5, "truck": 6, "person": 7},
        things=("motorcycle", "car", "truck", "person"),
    ),
    # 15 007 registered thermal frames at 640x512 alongside 4000x3000 RGB,
    # 15 classes (ECCV 2026). The registered pair is the 640x512 one.
    "segfly": DatasetSpec(
        name="segfly",
        thermal="**/thermal/*.png",
        rgb="**/rgb/*.jpg",
        masks="**/labels/*.png",
        classes={"background": 0, "road": 1, "building": 2, "vegetation": 3,
                 "water": 4, "car": 5, "truck": 6, "bus": 7, "motorcycle": 8,
                 "bicycle": 9, "person": 10, "pole": 11, "fence": 12,
                 "solar_panel": 13, "container": 14},
        things=("car", "truck", "bus", "motorcycle", "bicycle", "person"),
    ),
    # 500 sequences / ~1.7 M registered 1920x1080 RGB-T pairs (CVPR 2022).
    #
    # **A stage-A dataset, not a stage-B one.** Its annotation is a tracking
    # box, and only a 100-video subset carries masks -- so `list_frames` will
    # find almost nothing here and `list_pairs` will find more registered pairs
    # than every other set on the list combined. That is exactly the shape
    # modality distillation wants, because distillation reads no labels at all.
    #
    # Two knobs matter more here than anywhere else, both because the frames
    # are video at 1920x1080: `--crop 0.474` keeps native pixels instead of
    # squeezing a 16:9 frame into a square, and `--pairs` samples across the
    # set rather than truncating it (the first 5 000 pairs are two flights).
    "vtuav": DatasetSpec(
        name="vtuav",
        thermal="**/ir/*.jpg",
        rgb="**/rgb/*.jpg",
        masks="**/mask/*.png",
        classes={"background": 0, "target": 255},
        things=("target",),
    ),
    # FLIR ADK 640x512 with hardware-synchronised RGB, 4 195 dense masks
    # (ECCV 2024). Field-focused classes -- no small vehicles -- so this is a
    # generalisation check, not a source of instances.
    "caltech": DatasetSpec(
        name="caltech",
        thermal="**/thermal/*.png",
        rgb="**/rgb/*.png",
        masks="**/masks/*.png",
        classes={"background": 0, "sky": 1, "water": 2, "vegetation": 3,
                 "terrain": 4, "rock": 5, "structure": 6, "vehicle": 7,
                 "person": 8},
        things=("vehicle", "person"),
    ),
}


@dataclass(frozen=True)
class Frame:
    """One image, its semantic map, and its registered twin if there is one."""

    name: str
    image: Path
    mask: Path
    pair: Path | None = None


def describe_layout(root: str | Path, limit: int = 30) -> str:
    """Every directory under `root` that holds images, with counts and a stem.

    The first thing to look at after a download, and the answer to the most
    common way this stage fails. A `DatasetSpec`'s globs are written against
    what a paper says the archive contains; archives are repacked, renamed and
    nested inside an extra folder all the time, and the symptom is a
    `FileNotFoundError` that says what was *not* found rather than what is
    there. This says what is there, in the shape a glob is written in.
    """
    root = Path(root)
    directories: dict[Path, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            directories.setdefault(path.parent, []).append(path)

    if not directories:
        return f"{root}: no image files anywhere underneath it."

    lines = [f"{root}: {len(directories)} directories hold images",
             "", "| glob to write | files | example stem |", "|---|---:|---|"]
    for directory in sorted(directories, key=lambda d: -len(directories[d]))[:limit]:
        files = sorted(directories[directory])
        relative = directory.relative_to(root)
        suffixes = sorted({p.suffix.lower() for p in files})
        for suffix in suffixes:
            count = sum(1 for p in files if p.suffix.lower() == suffix)
            example = next(p for p in files if p.suffix.lower() == suffix)
            lines.append(f"| `{relative.as_posix()}/*{suffix}` | {count} | "
                         f"`{example.stem}` |")
    if len(directories) > limit:
        lines.append(f"| … and {len(directories) - limit} more | | |")
    return "\n".join(lines)


def _stem(path: Path, strip: SequenceABC[str]) -> str:
    stem = path.stem
    for suffix in strip:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def list_frames(root: str | Path, spec: DatasetSpec,
                modality: str = "thermal") -> list[Frame]:
    """Pair every image with its semantic map, by stem, sorted by name.

    An image with no map, or a map with no image, is dropped rather than
    guessed at -- a half-download is a common enough failure that inventing a
    pairing for it would show up as a mysteriously weak run instead of a
    missing file.
    """
    root = Path(root)
    images = {_stem(p, spec.strip): p for p in sorted(root.glob(spec.glob(modality)))
              if p.suffix.lower() in IMAGE_SUFFIXES}
    masks = {_stem(p, spec.strip): p for p in sorted(root.glob(spec.masks))
             if p.suffix.lower() in IMAGE_SUFFIXES}
    pairs: dict[str, Path] = {}
    other = "rgb" if modality == "thermal" else "thermal"
    if getattr(spec, other) is not None:
        pairs = {_stem(p, spec.strip): p for p in sorted(root.glob(spec.glob(other)))
                 if p.suffix.lower() in IMAGE_SUFFIXES}

    shared = sorted(set(images) & set(masks))
    if not shared:
        raise FileNotFoundError(
            f"{root}: no image/mask pairs for {spec.name} ({modality}).\n"
            f"Found {len(images)} images under {spec.glob(modality)!r} and "
            f"{len(masks)} masks under {spec.masks!r}"
            + (", and their stems do not match -- see DatasetSpec.strip."
               if images and masks else ".")
            + f"\n\nWhat is actually there:\n\n{describe_layout(root)}\n\n"
            f"Fix it without editing the repo:\n"
            f"    from dataclasses import replace\n"
            f"    spec = replace(spec, thermal=..., rgb=..., masks=..., strip=(...))")
    return [Frame(name=name, image=images[name], mask=masks[name],
                  pair=pairs.get(name)) for name in shared]


def list_pairs(root: str | Path, spec: DatasetSpec) -> list[tuple[Path, Path]]:
    """Every registered `(thermal, rgb)` pair, **whether or not it is labelled**.

    Separate from `list_frames` because the difference matters more than the
    duplication costs. Modality distillation (`src/training/distill.py`) needs
    no labels at all, and these sets carry far more registered pairs than
    annotated ones -- MVUAV is 53 828 frames of which 2 183 are labelled. Going
    through `list_frames` would throw 96 % of the pretraining data away for a
    label the pretraining objective never looks at.
    """
    root = Path(root)
    thermal = {_stem(p, spec.strip): p for p in sorted(root.glob(spec.glob("thermal")))
               if p.suffix.lower() in IMAGE_SUFFIXES}
    rgb = {_stem(p, spec.strip): p for p in sorted(root.glob(spec.glob("rgb")))
           if p.suffix.lower() in IMAGE_SUFFIXES}
    shared = sorted(set(thermal) & set(rgb))
    if not shared:
        raise FileNotFoundError(
            f"{root}: no registered thermal/RGB pairs for {spec.name}. Found "
            f"{len(thermal)} thermal and {len(rgb)} RGB images; the stems have "
            f"to match.")
    return [(thermal[name], rgb[name]) for name in shared]


def split_frames(frames: SequenceABC[Frame], fractions=(0.8, 0.1, 0.1),
                 seed: int = 0) -> dict[str, list[Frame]]:
    """train / val / test, by name hash rather than by position.

    These sets ship no official split. Splitting on a shuffled *name* order and
    not on the file order matters: consecutive frames of one flight sit next to
    each other on disk, so a positional split puts near-duplicate images on
    both sides of it and the held-out number stops meaning anything.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    order = np.random.default_rng(seed).permutation(len(frames))
    cuts = np.cumsum([int(round(f * len(frames))) for f in fractions[:-1]])
    parts = np.split(order, cuts)
    return {name: [frames[int(i)] for i in sorted(part)]
            for name, part in zip(("train", "val", "test"), parts)}


# --------------------------------------------------------------------------
# Semantic map -> instances
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class InstanceGates:
    """Four reasons a connected component is not a usable training instance.

    `min_area`   a component of a handful of pixels is annotation noise or an
                 antialiasing fringe, and its box is not a prompt anyone would
                 give.
    `min_side`   the same in the other direction: a 200-pixel component one
                 pixel wide is a boundary artefact, not an object.
    `max_area`   SAM 2's own rule, as a fraction of the *window*: a mask
                 covering nearly everything is "stuff" that slipped through the
                 class filter, and it teaches the decoder to answer any prompt
                 with the whole frame.
    `fill`       area over bounding-box area. A real vehicle from above fills
                 most of its box; a component at 0.15 is either a diagonal
                 sliver or -- the case that matters -- **two objects joined by
                 a thin bridge of pixels**, which is exactly the failure this
                 whole decomposition can produce.
    """

    min_area: int = 48
    min_side: int = 4
    max_area: float = 0.9
    fill: float = 0.25


@dataclass(frozen=True)
class Instance:
    """One component of one thing class, in frame coordinates.

    `label` is its value in the component image `decompose` returns, which is
    how the loader recovers the exact same mask without re-deciding anything:
    the index pass and the training pass run the same function on the same
    pixels, so the labels agree by construction rather than by matching boxes.
    """

    label: int
    class_id: int
    box: tuple[float, float, float, float]     # xyxy, frame coordinates
    area: int

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def fill(self) -> float:
        return self.area / max(self.width * self.height, 1e-6)

    def inside(self, origin: tuple[int, int], window: tuple[int, int]) -> bool:
        """True when the whole box sits within `window` at `origin`.

        Whole, not overlapping: a car cut by the window edge is a truncated
        target whose prompt claims an extent the pixels do not have. Dropping
        it costs one instance and keeps every remaining target honest.
        """
        x0, y0 = origin
        return (self.box[0] >= x0 and self.box[1] >= y0
                and self.box[2] <= x0 + window[0] and self.box[3] <= y0 + window[1])


def split_bridges(binary: np.ndarray, ratio: float = 0.55) -> np.ndarray:
    """Watershed on the distance transform: `[H, W] uint8` -> labelled `int32`.

    The one cheap repair for the failure `fill` detects. Two objects joined by
    a **thin bridge of pixels** are one connected component, but they are two
    peaks in the distance transform with a valley between them, so seeding at
    the peaks and flooding outwards separates them. `ratio` is where a peak
    starts: the fraction of each component's *own* maximum distance, per
    component rather than globally, so a large object does not set the
    threshold for a small one beside it.

    **What it cannot do, stated plainly:** two rectangles abutting along a full
    edge tile into a larger rectangle whose distance transform has no valley at
    all. Nothing about mask geometry can separate those -- only something that
    looks at the *pixels* can, which is the SAM-2-as-splitter route in
    `docs/encoder_mimari.md`. So this is a partial rescue, and the honest way
    to read it is against the `fill` reject count it is meant to reduce.
    """
    import cv2

    binary = (np.asarray(binary) > 0).astype(np.uint8)
    count, labelled, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros(binary.shape, dtype=np.int32)
    next_label = 1

    for index in range(1, count):
        piece = (labelled == index).astype(np.uint8)
        distance = cv2.distanceTransform(piece, cv2.DIST_L2, 3)
        peak = float(distance.max())
        cores = (distance >= ratio * peak).astype(np.uint8) * piece
        seeds, markers = cv2.connectedComponents(cores, connectivity=8)
        if seeds <= 2:
            out[piece > 0] = next_label       # one peak: nothing to split
            next_label += 1
            continue

        # cv2.watershed floods from labelled markers into the 0-valued unknown
        # band, over an 8-bit 3-channel image. The distance transform inverted
        # is that image: flooding downhill from each peak meets at the valley,
        # which is where the two objects join.
        markers = markers + 1
        markers[(piece > 0) & (cores == 0)] = 0
        markers[piece == 0] = 1               # background, so edges are found
        height = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX)
        relief = cv2.cvtColor((255 - height).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.watershed(relief, markers)

        for marker in range(2, seeds + 1):
            region = (markers == marker) & (piece > 0)
            if region.any():
                out[region] = next_label
                next_label += 1
        # Watershed marks its boundary pixels -1; they belong to neither side,
        # and leaving them out only thins each mask by a pixel.

    return out


def decompose(semantic: np.ndarray, spec: DatasetSpec,
              gates: InstanceGates = InstanceGates(),
              split: str = "none"
              ) -> tuple[np.ndarray, list[Instance], dict[str, int]]:
    """`(component image, instances, rejects)` for one semantic map.

    Components are found **per thing class** and then given globally unique
    labels, so a car touching a pedestrian stays two instances. Label 0 means
    "no instance here" -- stuff classes, ignored values and rejected components
    all collapse to it, because none of them is a target.

    `split="watershed"` additionally tries to separate objects joined by a thin
    bridge of pixels (`split_bridges`). It is off by default because it is a
    repair with its own failure mode -- it can over-split one long vehicle into
    two -- and the only honest way to choose is to compare the two on the
    `fill` reject count and on the panels the notebook draws.

    `rejects` counts which gate stopped what. It is the measurement the whole
    stage rests on: a class that rejects 40 % on `fill` is a class whose
    objects are being fused by the decomposition, and no amount of training
    fixes a bad target.
    """
    import cv2

    if split not in ("none", "watershed"):
        raise ValueError(f"split must be none or watershed, got {split!r}")

    semantic = np.asarray(semantic)
    if semantic.ndim == 3:
        semantic = semantic[..., 0]
    components = np.zeros(semantic.shape, dtype=np.int32)
    instances: list[Instance] = []
    rejects: dict[str, int] = {}
    frame_area = float(semantic.size)
    next_label = 1

    for class_id in spec.thing_ids:
        binary = (semantic == class_id).astype(np.uint8)
        if not binary.any():
            continue
        if split == "watershed":
            labelled = split_bridges(binary)
            count = int(labelled.max()) + 1
            stats = _stats_of(labelled, count)
        else:
            count, labelled, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8)
        for index in range(1, count):
            x, y, w, h, area = (int(stats[index, k]) for k in range(5))
            if area == 0:
                continue
            candidate = Instance(label=next_label, class_id=int(class_id),
                                 box=(float(x), float(y), float(x + w), float(y + h)),
                                 area=area)
            reason = reject_reason(candidate, frame_area, gates)
            if reason is not None:
                rejects[reason] = rejects.get(reason, 0) + 1
                continue
            components[labelled == index] = next_label
            instances.append(candidate)
            next_label += 1

    return components, instances, rejects


def _stats_of(labelled: np.ndarray, count: int) -> np.ndarray:
    """`connectedComponentsWithStats`-shaped rows for an already-labelled image.

    Same five columns in the same order (x, y, w, h, area) so `decompose` reads
    a watershed result and a plain component result through one code path --
    the alternative is two loops that have to be kept in step.
    """
    stats = np.zeros((max(count, 1), 5), dtype=np.int64)
    for index in range(1, count):
        rows, columns = np.nonzero(labelled == index)
        if rows.size == 0:
            continue
        stats[index] = (columns.min(), rows.min(),
                        columns.max() - columns.min() + 1,
                        rows.max() - rows.min() + 1, rows.size)
    return stats


def reject_reason(instance: Instance, frame_area: float,
                  gates: InstanceGates = InstanceGates()) -> str | None:
    """The first gate this component fails, or None if it passes them all.

    Which gate, not whether -- "38 % rejected" says the decomposition is weak,
    "38 % rejected on `fill`" says adjacent objects are being fused and the
    remedy is a different class filter, not a different learning rate.
    """
    if instance.area < gates.min_area:
        return "min_area"
    if min(instance.width, instance.height) < gates.min_side:
        return "min_side"
    if instance.area > gates.max_area * frame_area:
        return "max_area"
    if instance.fill < gates.fill:
        return "fill"
    return None


# --------------------------------------------------------------------------
# The index: every frame's instances, computed once
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameIndex:
    frame: Frame
    size: tuple[int, int]                 # (width, height) of the semantic map
    instances: tuple[Instance, ...]
    rejects: dict[str, int]


def read_mask(path: str | Path) -> np.ndarray:
    """A semantic map as a 2-D integer array, colour palettes included.

    Some of these sets ship a paletted PNG and some ship an RGB one. Reading
    unchanged and taking the first channel is right for the first and wrong for
    the second, so an RGB map is collapsed to a single value per pixel instead
    -- distinct colours stay distinct, and `probe_classes` then reports what is
    actually in the file.
    """
    import cv2

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Could not read semantic map: {path}")
    if raw.ndim == 2:
        return raw.astype(np.int32)
    if raw.shape[2] == 1:
        return raw[..., 0].astype(np.int32)
    channels = raw[..., :3].astype(np.int32)
    if (channels[..., 0] == channels[..., 1]).all() and \
            (channels[..., 1] == channels[..., 2]).all():
        return channels[..., 0]          # a grey map saved as three channels
    return (channels[..., 2] << 16) | (channels[..., 1] << 8) | channels[..., 0]


def probe_classes(frames: SequenceABC[Frame], limit: int = 64) -> dict[int, int]:
    """`{pixel value: how many of the sampled maps contain it}`.

    The one honest way to check a `DatasetSpec` against a download. A palette
    guessed from a paper and a palette in an archive disagree often enough that
    training on the guess produces a run with no instances at all and no error
    to explain it.
    """
    counts: dict[int, int] = {}
    for frame in list(frames)[:limit]:
        for value in np.unique(read_mask(frame.mask)):
            counts[int(value)] = counts.get(int(value), 0) + 1
    return dict(sorted(counts.items()))


def index_frames(frames: SequenceABC[Frame], spec: DatasetSpec,
                 gates: InstanceGates = InstanceGates(),
                 workers: int = 8, progress=None,
                 split: str = "none") -> list[FrameIndex]:
    """Decompose every frame's map once, so window sampling has something to aim at.

    This is a full pass over the semantic maps and nothing else -- no images are
    decoded. It is what makes the "are the instances clean?" question answerable
    before any GPU time is spent, and `save_index` keeps the answer across a
    runtime restart.
    """
    from concurrent.futures import ThreadPoolExecutor

    def one(frame: Frame) -> FrameIndex:
        semantic = read_mask(frame.mask)
        _, instances, rejects = decompose(semantic, spec, gates, split)
        return FrameIndex(frame=frame,
                          size=(int(semantic.shape[1]), int(semantic.shape[0])),
                          instances=tuple(instances), rejects=rejects)

    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        stream = pool.map(one, frames)
        if progress is not None:
            stream = progress(stream, total=len(frames), desc="indexing")
        return list(stream)


def save_index(path: str | Path, index: SequenceABC[FrameIndex]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"name": e.frame.name, "image": str(e.frame.image),
                "mask": str(e.frame.mask),
                "pair": str(e.frame.pair) if e.frame.pair else None,
                "size": list(e.size), "rejects": e.rejects,
                "instances": [[i.label, i.class_id, *i.box, i.area]
                              for i in e.instances]}
               for e in index]
    path.write_text(json.dumps(payload) + "\n")
    return path


def load_index(path: str | Path) -> list[FrameIndex]:
    payload = json.loads(Path(path).read_text())
    return [FrameIndex(
        frame=Frame(name=e["name"], image=Path(e["image"]), mask=Path(e["mask"]),
                    pair=Path(e["pair"]) if e["pair"] else None),
        size=(int(e["size"][0]), int(e["size"][1])),
        rejects=e["rejects"],
        instances=tuple(Instance(label=int(r[0]), class_id=int(r[1]),
                                 box=(r[2], r[3], r[4], r[5]), area=int(r[6]))
                        for r in e["instances"]),
    ) for e in payload]


def summarise(index: SequenceABC[FrameIndex], spec: DatasetSpec) -> str:
    """One markdown block: instances per class, their sizes, and what was rejected.

    Two numbers decide whether this stage is worth running at all -- how many
    instances the decomposition produced, and what share of components it threw
    away on `fill`. The second is the fusion warning: components rejected for
    filling a quarter of their own bounding box are usually two objects joined
    by a bridge of pixels.
    """
    per_class: dict[int, list[Instance]] = {}
    rejects: dict[str, int] = {}
    for entry in index:
        for instance in entry.instances:
            per_class.setdefault(instance.class_id, []).append(instance)
        for reason, count in entry.rejects.items():
            rejects[reason] = rejects.get(reason, 0) + count

    total = sum(len(v) for v in per_class.values())
    empty = sum(1 for e in index if not e.instances)
    lines = [
        f"{len(index)} frames, {total} instances kept, "
        f"{sum(rejects.values())} components rejected. "
        f"{empty} frame(s) hold no instance and are dropped.",
        "",
        "| class | instances | frames | median area px | median side px |",
        "|---|---:|---:|---:|---:|",
    ]
    for class_id, items in sorted(per_class.items(), key=lambda kv: -len(kv[1])):
        areas = np.array([i.area for i in items], dtype=np.float64)
        sides = np.array([max(i.width, i.height) for i in items], dtype=np.float64)
        frames = sum(1 for e in index if any(i.class_id == class_id for i in e.instances))
        lines.append(f"| {spec.name_of(class_id)} | {len(items)} | {frames} | "
                     f"{np.median(areas):.0f} | {np.median(sides):.0f} |")

    if rejects:
        lines += ["", "| gate that rejected | components |", "|---|---:|"]
        for reason, count in sorted(rejects.items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{reason}` | {count} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """A `size`x`size` window on one frame, and the instances inside it.

    The single-image counterpart of `antiuav.Clip`, down to the field names:
    `origin` + `window` is the source rectangle, `size` is what it is resized
    to, and `native` is True exactly when no resampling happens. Sharing the
    vocabulary is not cosmetic -- `crop_window` and `map_boxes` are the same
    functions the video path uses, so a window here and a window there cannot
    drift apart.
    """

    frame: Frame
    origin: tuple[int, int]
    window: tuple[int, int]
    size: int
    instances: tuple[Instance, ...]

    @property
    def native(self) -> bool:
        return self.window == (self.size, self.size)

    @property
    def boxes(self) -> np.ndarray:
        """The instances' boxes in the model's input coordinates."""
        raw = np.array([i.box for i in self.instances], dtype=np.float32)
        return map_boxes(raw, self.origin, self.window, self.size)


def windows_for(entry: FrameIndex, size: int = 512, per_image: int = 1,
                max_instances: int = 8, jitter: int = 0,
                rng: np.random.Generator | None = None) -> list[Sample]:
    """Up to `per_image` windows on one frame, each anchored on an instance.

    An anchor is picked at random and the window centred on it, which is the
    only sampling that works on both shapes of source this has to handle: a
    640x512 thermal frame where the window is nearly the whole image, and a
    4000x3000 aerial RGB frame where it is a thirtieth of it and a uniformly
    placed window would usually contain nothing.

    Every *other* indexed instance that falls entirely inside the window comes
    along for free. That is the point of the window: one encode, several
    prompts, several objects sharing a scene.
    """
    rng = rng or np.random.default_rng()
    if not entry.instances:
        return []

    fits = min(entry.size) >= size
    order = rng.permutation(len(entry.instances))[:max(per_image, 1)]
    samples: list[Sample] = []
    for i in order:
        anchor = entry.instances[int(i)]
        box = np.array([anchor.box], dtype=np.float32)
        # A frame smaller than the window has no crop to take -- `crop_window`
        # would clamp the origin to 0 and hand back a rectangle running off the
        # edge. That case is the whole-frame resize, same as an oversized
        # anchor.
        origin = crop_window(box, entry.size, size, jitter, rng) if fits else None
        if origin is None:
            # No crop contains the anchor, so fall back to resizing the whole
            # frame -- the same two input modes the deployment offers, and the
            # same fallback `antiuav._window_for` makes.
            origin, window = (0, 0), entry.size
        else:
            window = (size, size)

        inside = [x for x in entry.instances if x.inside(origin, window)]
        if not inside:
            continue
        if len(inside) > max_instances:
            # The anchor goes first and unconditionally -- it is why this
            # window exists, and a window whose anchor was shuffled out is a
            # window placed for no reason. The rest fill the remaining slots.
            others = [x for x in inside if x.label != anchor.label]
            picked = rng.permutation(len(others))[:max(max_instances - 1, 0)]
            inside = [anchor] + [others[int(j)] for j in sorted(picked)]
        samples.append(Sample(frame=entry.frame, origin=origin, window=window,
                              size=size, instances=tuple(inside)))
    return samples


def sample_windows(index: SequenceABC[FrameIndex], size: int = 512,
                   per_image: int = 1, max_instances: int = 8, jitter: int = 0,
                   seed: int | None = None) -> list[Sample]:
    """Every frame's windows, in one list -- the epoch's pool."""
    rng = np.random.default_rng(seed)
    samples: list[Sample] = []
    for entry in index:
        samples.extend(windows_for(entry, size, per_image, max_instances, jitter, rng))
    return samples


# --------------------------------------------------------------------------
# Pixels
# --------------------------------------------------------------------------


def load_image(path: str | Path, origin: tuple[int, int], window: tuple[int, int],
               size: int, gray: bool = True) -> np.ndarray:
    """One image's window as HxWx3 uint8 at the model's input size.

    `gray=True` for thermal, and it is not a formality: the checkpoint's stem
    takes three channels, so the single thermal channel is replicated rather
    than the stem rewritten -- a one-channel stem would change the ONNX graph
    and every engine built from it to save a rounding error of a 38.7 GFLOP
    encoder. `antiuav.load_window` makes the same choice for the same reason.
    """
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if gray:
        if img.ndim == 3:
            img = img[..., 0] if img.shape[2] == 1 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    x0, y0 = origin
    img = img[y0:y0 + window[1], x0:x0 + window[0]]
    if img.shape[:2] != (size, size):
        shrinking = size < img.shape[0] or size < img.shape[1]
        img = cv2.resize(img, (size, size),
                         interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)
    return np.repeat(img[:, :, None], 3, axis=2) if img.ndim == 2 else img


def normalise(images: np.ndarray, device: str = "cpu"):
    """uint8 HxWx3 (or NxHxWx3) -> the normalised float tensor the model takes.

    ImageNet statistics, matching `load_video_frames` and `src/pipeline.py`.
    Training on different statistics from the ones the tracker uses at
    inference would be a silent, permanent domain shift of our own making.
    """
    import torch

    array = images if images.ndim == 4 else images[None]
    out = torch.from_numpy(np.ascontiguousarray(array)).to(device)
    out = out.permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.as_tensor(MEAN, device=out.device)[:, None, None]
    std = torch.as_tensor(STD, device=out.device)[:, None, None]
    return out.sub_(mean).div_(std)


def sample_masks(sample: Sample, spec: DatasetSpec,
                 gates: InstanceGates = InstanceGates(),
                 split: str = "none") -> np.ndarray:
    """`[K, size, size]` boolean targets, one per instance, in input coordinates.

    The semantic map is decomposed again here rather than stored. `decompose`
    is deterministic, so the labels it produces now are the labels the index
    recorded, and recovering a mask is `components == instance.label` -- no
    box matching, no tolerance, no drift between what was indexed and what is
    trained on. The alternative, a mask store like the video path's, would be
    ~330 KB per instance for data that regenerates in milliseconds.

    `split` has to be the value the index was built with, which is why it
    travels on `ImageSplit` rather than being read from a flag in two places:
    decomposing differently here would renumber the labels and silently pair
    every instance with another instance's mask.
    """
    import cv2

    components, _, _ = decompose(read_mask(sample.frame.mask), spec, gates, split)
    x0, y0 = sample.origin
    width, height = sample.window
    crop = components[y0:y0 + height, x0:x0 + width]

    out = np.zeros((len(sample.instances), sample.size, sample.size), dtype=bool)
    for k, instance in enumerate(sample.instances):
        mask = (crop == instance.label).astype(np.uint8)
        if mask.shape[:2] != (sample.size, sample.size):
            # Nearest neighbour: an aerial car can be twenty pixels across, and
            # any interpolation of a boolean mask at that size invents or
            # erases them.
            mask = cv2.resize(mask, (sample.size, sample.size),
                              interpolation=cv2.INTER_NEAREST)
        out[k] = mask.astype(bool)
    return out


__all__ = [
    "DatasetSpec", "Frame", "FrameIndex", "Instance", "InstanceGates", "SPECS",
    "Sample", "decompose", "describe_layout", "index_frames", "list_frames", "list_pairs",
    "load_image", "load_index", "normalise", "probe_classes", "read_mask",
    "reject_reason", "replace", "sample_masks", "sample_windows", "save_index",
    "split_frames", "summarise", "windows_for",
]
