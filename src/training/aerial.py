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
2. **Keep only the "thing" classes.** Kust4K labels nine, of which four are
   trackable objects (motorcycle, car, truck, human). Road, building, tree and
   traffic facilities are "stuff" -- they have no instances and are not
   tracking targets, so a component of them is not a training example.
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
import warnings
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
    border: int = 0                         # pixels of padding to discard, per side

    def inset(self, height: int, width: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """`(origin, size)` of the real picture inside a padded frame.

        DroneVehicle ships every image, in **both** modalities, as 840x712 with
        a 100 px band of pure white on all four sides. Left in, that band is a
        third of the frame: the teacher spends a third of its patch tokens
        describing white, `shared_window` places crops that are partly margin,
        and the normalisation statistics are pulled toward 255. Taken out, what
        is left is 640x512 -- exactly this project's native resolution, so the
        crop is not a workaround but the frame the sensor actually produced.

        Applied when the image is read, so nothing is re-encoded and the JPEG
        is not decoded and saved a second time.
        """
        b = self.border
        if not b:
            return (0, 0), (width, height)
        if height <= 2 * b or width <= 2 * b:
            raise ValueError(
                f"{self.name}: border={b} leaves nothing of a {width}x{height} "
                f"image. The spec's border does not match this download.")
        return (b, b), (width - 2 * b, height - 2 * b)

    def glob(self, modality: str) -> str:
        """The image glob for `modality`, or a readable failure."""
        available = {"thermal": self.thermal, "rgb": self.rgb}
        if modality not in available:
            raise ValueError(f"modality must be thermal or rgb, got {modality!r}")
        pattern = available[modality]
        if pattern is None:
            raise ValueError(f"{self.name} has no {modality} images.")
        return pattern

    def mask_glob(self, modality: str) -> str:
        """The mask glob for `modality`.

        VTUAV annotates each modality separately -- `mask/ir` and `mask/rgb`,
        875 and 874 files -- so one pattern for both would glob the two into a
        single dict where they collide frame for frame. A `{modality}`
        placeholder is filled with the directory name that modality's *images*
        live in, which keeps the two halves from ever being mixed. Sets with
        one mask directory have no placeholder and are returned unchanged.
        """
        if "{modality}" not in self.masks:
            return self.masks
        folders = [p for p in self.glob(modality).split("/")[:-1] if "*" not in p]
        if not folders:
            raise ValueError(
                f"{self.name}: masks={self.masks!r} needs a modality directory, "
                f"but {modality} images are globbed as {self.glob(modality)!r}, "
                f"which names none.")
        return self.masks.format(modality=folders[-1])

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
    # 4 024 registered RGB-TIR pairs at 640x512 (2 514 day / 1 510 night),
    # 9 classes (Sci. Data 2025). 640x512 is the project's native resolution: a
    # 512 window is a crop, not a resize, and it is the same input the Orin
    # deployment feeds the model.
    #
    # **The palette is the authors' own `visual.py`, not a reading of the
    # paper**, because an earlier version of this spec was wrong from id 3 on:
    # it assumed a `vegetation` class at 3 that does not exist, which shifted
    # every id after it by one and left no room for id 8 at all. The effect was
    # the same class of bug as the SegFly one below -- `things` picked up id 6,
    # which is **tree**, and missed id 3, which is **motorcycle**. Two
    # independent checks agree on the list below: `get_palette()` in the
    # archive's `visual.py`, and the value frequencies in `Seg_annos.zip`
    # (id 6 appears in 36 of 41 sampled frames, which fits tree and not truck;
    # id 8 appears in 30, and the old palette had no id 8).
    #
    # The three archives are **flat** -- `00001D.png` at the top level of each,
    # no directories -- so `tools/fetch_datasets.py` extracts them into the
    # `tir/`, `rgb/` and `label/` folders these globs expect. The `D`/`N` in a
    # stem is day/night, and the same stem names the frame in all three.
    "kust4k": DatasetSpec(
        name="kust4k",
        thermal="**/tir/*.png",
        rgb="**/rgb/*.png",
        masks="**/label/*.png",
        classes={"unlabelled": 0, "road": 1, "building": 2, "motorcycle": 3,
                 "car": 4, "truck": 5, "tree": 6, "human": 7,
                 "traffic_facilities": 8},
        things=("motorcycle", "car", "truck", "human"),
        ignore=(0,),
    ),
    # >15 000 geometrically aligned RGB-T pairs at 640x512 thermal, plus
    # >20 000 RGB-only frames, over three altitudes (30/40/50 m). ECCV 2026.
    # Thermal exists for scenes 3, 4, 5 and 9 only.
    #
    # **The class ids have gaps and the thing classes are only two.** This is
    # the authors' own table, not a guess, and the difference matters: an
    # earlier version of this spec assumed contiguous ids and would have
    # trained on **grass, vegetation, tree and ground obstacle** as tracking
    # targets. Vehicle is 13 and Truck is 36; there is no person, bus,
    # motorcycle or bicycle class at all.
    #
    # Labels are 8-bit single-channel PNG whose pixel value *is* the class id.
    #
    # Distributed as a Hugging Face **parquet** dataset, not as directories:
    # columns `image`, `label`, `RGB_aligned`, `scene`, `altitude`, `modality`.
    # `tools/export_hf_dataset.py` writes it into the layout below.
    "segfly": DatasetSpec(
        name="segfly",
        thermal="**/images/*.png",
        rgb="**/rgb/*.png",
        masks="**/labels/*.png",
        classes={"unlabeled": 0, "road": 1, "walkway": 2, "dirt": 3,
                 "gravel": 4, "grass": 6, "vegetation": 7, "tree": 8,
                 "ground_obstacle": 9, "vehicle": 13, "water": 14,
                 "building": 16, "roof": 17, "parking_lot": 33,
                 "construction": 34, "truck": 36},
        things=("vehicle", "truck"),
        ignore=(0,),
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
        masks="**/mask/{modality}/*.png",
        classes={"background": 0, "target": 255},
        things=("target",),
    ),
    # VTUAV's video-instance-segmentation split: 100 of the 500 sequences,
    # RGB-T pairs at 1920x1080 with **per-frame instance masks**.
    #
    # **The only set on this list whose annotation is already what stage B
    # wants.** Everything `decompose` does in `components` mode -- filter to
    # thing classes, connected components, gate against fusion -- exists to
    # reconstruct exactly this, so here it is skipped: `mode="labels"` reads
    # each distinct non-zero value as one instance. Use it with
    #     Source(SPECS["vtuav_vis"], mode="labels", gray=True)
    #
    # Two caveats the authors' own paper implies and this cannot fix:
    # the two modalities are **not perfectly registered**, and not every frame
    # is hand-annotated -- some masks are propagated. That makes it excellent
    # stage-B data and something to be careful with in stage A, where a
    # misregistered pair means the teacher looked somewhere the student did not
    # (`distill_loss(..., tolerance=1)` is the remedy).
    #
    # **The globs below were read off the archive, not guessed.** An earlier
    # version of this comment said they were a guess and asked for
    # `describe_layout` to confirm; that has now been done by listing the
    # central directory of `training/train_001.zip` over range requests. What
    # it contains:
    #
    #     bike_009/rgb/000000.jpg        26 059 frames        1920x1080
    #     bike_009/ir/000000.jpg         26 059 frames        1920x1080
    #     bike_009/mask/rgb/000000.png      875 masks         1920x1080, {0,255}
    #     bike_009/mask/ir/000000.png       874 masks
    #     bike_009/rgb.txt               per-frame `x y w h` tracking boxes
    #
    # Three things in that listing shape the spec. **Masks are per modality**,
    # so the pattern below is `{modality}`-templated -- globbing both into one
    # dict would collide frame for frame. **Only every 30th frame is masked**
    # (1 749 masks against 52 118 images), which is why the same download is
    # worth far more to `list_pairs` than to `list_frames`. And the mask value
    # is **255, not 1** -- this spec said 1, which would have made every mask
    # read as empty and every loss compare a prediction against nothing.
    #
    # Foreground is 0.2-2.0 % of the frame across the sequences sampled. That
    # is the small-target regime this project exists for, and it is also why
    # `--crop` matters: resizing 1920x1080 to a square would put a 0.2 % target
    # below the patch grid.
    #
    # Two caveats the authors' own paper implies and this cannot fix: the two
    # modalities are **not perfectly registered**, and not every frame is
    # hand-annotated -- some masks are propagated. That makes it excellent
    # stage-B data and something to be careful with in stage A, where a
    # misregistered pair means the teacher looked somewhere the student did not
    # (`distill_loss(..., tolerance=1)` is the remedy).
    "vtuav_vis": DatasetSpec(
        name="vtuav_vis",
        thermal="**/ir/*.jpg",
        rgb="**/rgb/*.jpg",
        masks="**/mask/{modality}/*.png",
        classes={"background": 0, "target": 255},
        things=("target",),
    ),
    # FLIR ADK 640x512 with hardware-synchronised RGB, 4 195 dense masks
    # (ECCV 2024). Field-focused classes -- no small vehicles -- so this is a
    # generalisation check, not a source of instances.
    # 28 442 registered RGB-TIR pairs from a UAV, day and night, oriented-box
    # annotation (RA-L 2022). **A stage-A set only**: its labels are boxes in
    # XML, not masks, so `list_frames` finds nothing here and `list_pairs`
    # finds all of it -- which is exactly what modality distillation reads.
    #
    # Every image, in both modalities, is 840x712 with a **100 px band of pure
    # white on all four sides**; measured, not taken from the paper. Inside it
    # is 640x512, this project's native resolution. `border=100` discards the
    # band at read time -- see `DatasetSpec.inset`, and note that leaving it in
    # would spend a third of the teacher's patch tokens on white.
    #
    # The authors distribute it through Baidu; the Hugging Face mirror below
    # serves the same three archives over plain HTTPS, anonymously, with range
    # support. `masks` names the XML directory so the spec is honest about
    # having none -- the glob matches no image and `list_frames` says so.
    "dronevehicle": DatasetSpec(
        name="dronevehicle",
        thermal="**/*imgr/*.jpg",
        rgb="**/*img/*.jpg",
        masks="**/*label/*.xml",
        classes={"background": 0},
        things=(),
        border=100,
    ),
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


def _named_dirs(pattern: str) -> int:
    """How many directory levels a glob names between its prefix and the file.

    Counts every component except a leading `**` and the filename, **whether or
    not it contains a wildcard**. Counting only wildcard-free components was
    wrong the moment a dataset named its two modality directories `trainimg`
    and `trainimgr`: the spec has to glob them as `*img` and `*imgr` (the
    prefix changes per split), those count as zero fixed directories, and the
    two halves then reduce to different keys -- `train/trainimg/00001` against
    `train/trainimgr/00001` -- so `list_pairs` matches nothing at all and says
    the download has no registered pairs.
    """
    return sum(1 for part in pattern.split("/")[:-1] if part != "**")


def _key(path: Path, root: Path, pattern: str, strip: SequenceABC[str]) -> str:
    """The pairing key: what identifies the *frame*, not just the file.

    `path.stem` alone is not enough, and the way it fails is silent. VTUAV
    numbers frames per sequence, so `bike_009/rgb/000000.jpg` and
    `car_083/rgb/000000.jpg` share the stem `000000`. Keying a dict on the
    stem keeps one of the fourteen sequences in a training zip -- and since
    the image and mask dicts are built from *different* globs, the survivor on
    each side can be a different sequence, which pairs one flight's picture
    with another flight's mask. Nothing downstream can detect that; it just
    trains on nonsense.

    So the key is the relative path with the glob's own fixed directories
    taken out: `bike_009/rgb/000000.jpg` and `bike_009/mask/rgb/000000.png`
    both reduce to `bike_009/000000`, while a flat set like Kust4K
    (`tir/00001D.png`) still reduces to `00001D`.
    """
    relative = path.relative_to(root)
    keep = max(len(relative.parts) - _named_dirs(pattern) - 1, 0)
    return "/".join((*relative.parts[:keep], _stem(path, strip)))


def list_frames(root: str | Path, spec: DatasetSpec,
                modality: str = "thermal") -> list[Frame]:
    """Pair every image with its semantic map, by stem, sorted by name.

    An image with no map, or a map with no image, is dropped rather than
    guessed at -- a half-download is a common enough failure that inventing a
    pairing for it would show up as a mysteriously weak run instead of a
    missing file.
    """
    root = Path(root)
    mask_pattern = spec.mask_glob(modality)
    images = {_key(p, root, spec.glob(modality), spec.strip): p
              for p in sorted(root.glob(spec.glob(modality)))
              if p.suffix.lower() in IMAGE_SUFFIXES}
    masks = {_key(p, root, mask_pattern, spec.strip): p
             for p in sorted(root.glob(mask_pattern))
             if p.suffix.lower() in IMAGE_SUFFIXES}
    pairs: dict[str, Path] = {}
    other = "rgb" if modality == "thermal" else "thermal"
    if getattr(spec, other) is not None:
        pairs = {_key(p, root, spec.glob(other), spec.strip): p
                 for p in sorted(root.glob(spec.glob(other)))
                 if p.suffix.lower() in IMAGE_SUFFIXES}

    shared = sorted(set(images) & set(masks))
    if not shared:
        raise FileNotFoundError(
            f"{root}: no image/mask pairs for {spec.name} ({modality}).\n"
            f"Found {len(images)} images under {spec.glob(modality)!r} and "
            f"{len(masks)} masks under {mask_pattern!r}"
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
    thermal = {_key(p, root, spec.glob("thermal"), spec.strip): p
               for p in sorted(root.glob(spec.glob("thermal")))
               if p.suffix.lower() in IMAGE_SUFFIXES}
    rgb = {_key(p, root, spec.glob("rgb"), spec.strip): p
           for p in sorted(root.glob(spec.glob("rgb")))
           if p.suffix.lower() in IMAGE_SUFFIXES}
    shared = sorted(set(thermal) & set(rgb))
    if not shared:
        raise FileNotFoundError(
            f"{root}: no registered thermal/RGB pairs for {spec.name}. Found "
            f"{len(thermal)} thermal and {len(rgb)} RGB images; the stems have "
            f"to match.")
    return [(thermal[name], rgb[name]) for name in shared]


def _seed_for(name: str) -> int:
    """A stable per-source seed offset, derived from the name rather than a
    position in the list.

    This is what makes two runs on overlapping dataset lists comparable. The
    offset used to be the source's index in the sorted grouping, so VTUAV was
    the third source in a three-dataset run and the only one in a VTUAV-only
    run -- different seed, different permutation, **different held-out
    sequences**. Comparing the two test numbers would then be comparing two
    different test sets, which is not a comparison at all.

    Keyed on the name, a dataset's split depends only on that dataset and the
    caller's seed, so adding SegFly and Kust4K to a run leaves VTUAV's split
    exactly where it was. `blake2b` rather than `hash()` because the built-in
    is salted per process and would not survive a restart.
    """
    import hashlib

    digest = hashlib.blake2b(name.encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def group_of(item) -> str:
    """The sequence an item came from, or the item itself if it stands alone.

    `list_frames` names a frame by sequence and number -- `bike_009/000000` --
    so everything before the last `/` is the flight it was cut from. A frame
    from a set with no sequence directories (`00001D`) has no group and
    becomes its own, which is what makes grouping safe to apply everywhere.
    """
    name = getattr(getattr(item, "frame", item), "name", "")
    return name.rsplit("/", 1)[0] if "/" in name else name


def split_frames(items: SequenceABC, fractions=(0.8, 0.1, 0.1),
                 seed: int = 0, group=group_of) -> dict[str, list]:
    """train / val / test, cut between **sequences** rather than between frames.

    These sets ship no official split, and the file order is the wrong thing to
    cut: consecutive frames of one flight sit next to each other on disk, so a
    positional split puts near-duplicate images on both sides of the line and
    the held-out number stops meaning anything. A seeded permutation fixes the
    ordering half of that.

    It does not fix the other half, which is why the split is on groups.
    Permuting *frames* leaves two masks from one flight a second apart -- VTUAV
    annotates every 30th frame -- on opposite sides of the split, and they show
    the same target against the same background. The model has effectively seen
    the test frame, so the number comes out high for a reason that has nothing
    to do with how well it generalises. Holding out whole sequences is the
    difference between measuring generalisation and measuring memory.

    Sets with no sequence structure are unaffected: `group_of` makes every
    frame its own group and the result is the plain permutation. `group=None`
    asks for that explicitly.

    Works on anything -- `Frame`s or `FrameIndex` entries -- because it only
    ever looks at `.name`. For a run mixing datasets use `split_index`, which
    stratifies.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    if group is None:
        keys = [str(i) for i in range(len(items))]
    else:
        keys = [group(item) for item in items]

    groups: dict[str, list[int]] = {}
    for position, key in enumerate(keys):
        groups.setdefault(key, []).append(position)
    names = sorted(groups)

    # Fewer groups than parts cannot be split between them -- one sequence
    # would mean an empty val and test. Fall back to cutting frames and say so,
    # because a leaky split that runs beats a crash only if you know it leaked.
    if group is not None and len(names) < len(fractions) < len(items):
        warnings.warn(
            f"{len(names)} sequence(s) for a {len(fractions)}-way split: "
            f"falling back to splitting frames, so held-out frames share their "
            f"sequence with training ones and the score will read high. "
            f"Download another sequence to fix it.", stacklevel=2)
        return split_frames(items, fractions, seed, group=None)

    order = np.random.default_rng(seed).permutation(len(names))
    counts = _allocate(len(names), fractions)
    parts = np.split(order, np.cumsum(counts[:-1]))
    return {name: [items[i] for g in sorted(part) for i in groups[names[int(g)]]]
            for name, part in zip(("train", "val", "test"), parts)}


def _allocate(total: int, fractions: SequenceABC[float]) -> list[int]:
    """How many groups each split gets -- never starving one to rounding.

    Rounding each fraction independently is fine on thousands of frames and
    wrong on a handful of sequences: 6 sequences at (0.8, 0.1, 0.1) rounds to
    5, 1, 0 and the **test set is empty**, which surfaces later as a
    `nan` score rather than as an error here. Largest-remainder first, then
    every split with a non-zero fraction is guaranteed one group as long as
    there are enough to go round, taken from the largest split.
    """
    exact = [f * total for f in fractions]
    counts = [int(x) for x in exact]
    for index in sorted(range(len(exact)), key=lambda i: exact[i] - counts[i],
                        reverse=True)[: total - sum(counts)]:
        counts[index] += 1

    for index, fraction in enumerate(fractions):
        if fraction > 0 and counts[index] == 0:
            donor = max(range(len(counts)), key=lambda i: counts[i])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[index] += 1
    return counts


def split_index(index: SequenceABC[FrameIndex], fractions=(0.8, 0.1, 0.1),
                seed: int = 0) -> dict[str, list[FrameIndex]]:
    """`split_frames` per source, then concatenated -- a **stratified** split.

    Two reasons it cannot just permute the whole thing. Splitting proportionally
    within each dataset keeps a small one from landing almost entirely in
    `test` by chance, which on a 4 024-frame set beside a 100-sequence one is
    not a remote possibility. And it splits on `FrameIndex` entries rather than
    on frame names, because names collide across datasets: `000123.png` exists
    in most of them, and a name-keyed lookup would silently train on one
    dataset's frame while scoring another's.

    **`Source.role` decides which splits a dataset feeds at all**, and the
    reason is what a held-out number is supposed to mean. A dataset whose
    instances were *reconstructed* from a semantic map -- connected components,
    gates, the whole of `decompose` -- carries a target that is itself
    uncertain: where the decomposition fused two cars, its "ground truth" says
    one blob, and a model that correctly separates them is marked **wrong**.
    Scoring on that measures the decomposition as much as the model.

    So a set with real instance masks can be `all` (or `eval`) and a
    reconstructed one `train`, and then the test number is against annotation
    somebody actually drew. Training on both is still right -- more data, and a
    slightly noisy target is a normal thing to learn from. It is only the
    *measurement* that has to be clean.
    """
    grouped: dict[str, list[FrameIndex]] = {}
    for entry in index:
        grouped.setdefault(entry.source.name if entry.source else "?", []).append(entry)

    out: dict[str, list[FrameIndex]] = {"train": [], "val": [], "test": []}
    for name, entries in sorted(grouped.items()):
        offset = _seed_for(name)
        role = entries[0].source.role if entries[0].source else "all"
        if role == "train":
            out["train"].extend(entries)
            continue
        if role == "eval":
            # Everything this set has goes to val and test, in the ratio the
            # caller asked for between them.
            val, test = fractions[1], fractions[2]
            share = val / max(val + test, 1e-9)
            parts = split_frames(entries, (share, 1.0 - share, 0.0),
                                 seed + offset)
            out["val"].extend(parts["train"])
            out["test"].extend(parts["val"])
            continue
        # Seeded by name (see `_seed_for`), so two datasets of the same length
        # do not receive the identical permutation -- and so a set's own split
        # does not move when a different dataset joins the run.
        parts = split_frames(entries, fractions, seed + offset)
        for name in out:
            out[name].extend(parts[name])
    return out


# --------------------------------------------------------------------------
# Semantic map -> instances
# --------------------------------------------------------------------------


MODES = ("components", "watershed", "labels")

# Which splits a dataset feeds. `all` is the ordinary 80/10/10.
ROLES = ("all", "train", "eval")


@dataclass(frozen=True)
class Source:
    """Everything needed to turn one dataset's files into training targets.

    A *per-sample* bundle rather than a per-run setting, and that is the whole
    point: one encoder trained on one dataset is one sensor, one city and one
    set of annotation habits, which is not enough to move a trunk that carries
    general visual features. Mixing several datasets in a single batch is what
    makes the encoder see past any one of them -- so every `Sample` carries the
    rules for decoding *its own* frame, and a batch may hold VTUAV's instance
    masks beside Kust4K's decomposed semantic ones.

    It also removes a class of silent bug. The mode used to build an index and
    the mode used to load its masks have to agree exactly, or the component
    labels renumber and every instance is paired with another one's mask.
    Carrying them together makes disagreeing impossible rather than merely
    discouraged.
    """

    spec: DatasetSpec
    gates: "InstanceGates" = None            # filled in below; see __post_init__
    mode: str = "components"
    gray: bool = True                        # thermal replicates one channel
    role: str = "all"                        # which splits this feeds; see below

    def __post_init__(self) -> None:
        if self.gates is None:
            object.__setattr__(self, "gates", InstanceGates())
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {self.role!r}")

    @property
    def name(self) -> str:
        return f"{self.spec.name}/{self.mode}"


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
              mode: str = "components"
              ) -> tuple[np.ndarray, list[Instance], dict[str, int]]:
    """`(component image, instances, rejects)` for one annotation map.

    Three modes, because the datasets differ in what their masks already are:

    `components` (default) -- **the map is semantic**, so instances are found
        as connected components **per thing class**, then given globally unique
        labels. Per class and not over their union: a car touching a pedestrian
        would otherwise fuse into one component spanning two classes.

    `watershed` -- the same, plus an attempt to separate objects joined by a
        thin bridge of pixels (`split_bridges`). Off by default because it is a
        repair with its own failure mode -- it can over-split one long vehicle
        -- and the only honest way to choose is to compare the two on the
        `fill` reject count and on the panels the notebook draws.

    `labels` -- **the map already carries instances**, one value each, as
        VTUAV's video-instance-segmentation split does. Nothing is decomposed
        and no class filter applies: every non-background value *is* a target,
        which is the annotation this whole stage has been reconstructing. The
        gates still run, but read a `fill` rejection differently here -- it is
        a genuinely thin or diagonal object, not two objects fused, because
        there was no fusing step.

    Label 0 means "no instance here" in the returned image. `rejects` counts
    which gate stopped what.
    """
    import cv2

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    semantic = np.asarray(semantic)
    if semantic.ndim == 3:
        semantic = semantic[..., 0]
    components = np.zeros(semantic.shape, dtype=np.int32)
    instances: list[Instance] = []
    rejects: dict[str, int] = {}
    frame_area = float(semantic.size)
    next_label = 1

    def keep(labelled, count, stats, class_id):
        nonlocal next_label
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

    if mode == "labels":
        # Every distinct non-background value is one instance. Values are
        # remapped to a dense 1..N so the stats table below is indexable, and
        # `spec.ignore` plus 0 are the only things dropped.
        skip = {0, *spec.ignore}
        values = [int(v) for v in np.unique(semantic) if int(v) not in skip]
        dense = np.zeros(semantic.shape, dtype=np.int32)
        for index, value in enumerate(values, start=1):
            dense[semantic == value] = index
        class_id = spec.thing_ids[0] if spec.thing_ids else 0
        keep(dense, len(values) + 1, _stats_of(dense, len(values) + 1), class_id)
        return components, instances, rejects

    for class_id in spec.thing_ids:
        binary = (semantic == class_id).astype(np.uint8)
        if not binary.any():
            continue
        if mode == "watershed":
            labelled = split_bridges(binary)
            count = int(labelled.max()) + 1
            stats = _stats_of(labelled, count)
        else:
            count, labelled, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8)
        keep(labelled, count, stats, class_id)

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
    size: tuple[int, int]                 # (width, height) of the annotation map
    instances: tuple[Instance, ...]
    rejects: dict[str, int]
    source: Source | None = None


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


def index_frames(frames: SequenceABC[Frame], source: Source,
                 workers: int = 8, progress=None) -> list[FrameIndex]:
    """Decompose every frame's map once, so window sampling has something to aim at.

    A full pass over the annotation maps and nothing else -- no images are
    decoded. It is what makes the "are the instances clean?" question answerable
    before any GPU time is spent, and `save_index` keeps the answer across a
    runtime restart.
    """
    from concurrent.futures import ThreadPoolExecutor

    def one(frame: Frame) -> FrameIndex:
        semantic = read_mask(frame.mask)
        _, instances, rejects = decompose(semantic, source.spec, source.gates,
                                          source.mode)
        return FrameIndex(frame=frame,
                          size=(int(semantic.shape[1]), int(semantic.shape[0])),
                          instances=tuple(instances), rejects=rejects,
                          source=source)

    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        stream = pool.map(one, frames)
        if progress is not None:
            stream = progress(stream, total=len(frames), desc="indexing")
        return list(stream)


def save_index(path: str | Path, index: SequenceABC[FrameIndex]) -> Path:
    """Write an index, stamped with the source that produced it.

    The stamp is what `load_index` checks. An index built with one mode and
    loaded by a run decomposing with another renumbers every component label,
    so each instance would train against a different instance's mask -- and the
    loss would stay perfectly finite while it happened.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = next((e.source for e in index if e.source), None)
    payload = [{"name": e.frame.name, "image": str(e.frame.image),
                "mask": str(e.frame.mask),
                "pair": str(e.frame.pair) if e.frame.pair else None,
                "size": list(e.size), "rejects": e.rejects,
                "instances": [[i.label, i.class_id, *i.box, i.area]
                              for i in e.instances]}
               for e in index]
    path.write_text(json.dumps({
        "source": {"spec": stamp.spec.name if stamp else None,
                   "mode": stamp.mode if stamp else None},
        "frames": payload,
    }) + "\n")
    return path


def load_index(path: str | Path, source: Source | None = None) -> list[FrameIndex]:
    """Read an index back, refusing one built by a different source."""
    payload = json.loads(Path(path).read_text())
    stamp, rows = payload["source"], payload["frames"]
    if source is not None and stamp["spec"] is not None:
        if (stamp["spec"], stamp["mode"]) != (source.spec.name, source.mode):
            raise ValueError(
                f"{path} was built from {stamp['spec']}/{stamp['mode']} but this "
                f"run decomposes with {source.spec.name}/{source.mode}. Loading "
                f"it would renumber every component label and pair each instance "
                f"with another one's mask. Delete the cache or match the mode.")
    return [FrameIndex(
        frame=Frame(name=e["name"], image=Path(e["image"]), mask=Path(e["mask"]),
                    pair=Path(e["pair"]) if e["pair"] else None),
        size=(int(e["size"][0]), int(e["size"][1])),
        rejects=e["rejects"],
        instances=tuple(Instance(label=int(r[0]), class_id=int(r[1]),
                                 box=(r[2], r[3], r[4], r[5]), area=int(r[6]))
                        for r in e["instances"]),
        source=source,
    ) for e in rows]


def summarise(index: SequenceABC[FrameIndex], spec: DatasetSpec | None = None) -> str:
    """One markdown block: instances per class, their sizes, and what was rejected.

    Two numbers decide whether this stage is worth running at all -- how many
    instances the decomposition produced, and what share of components it threw
    away on `fill`. The second is the fusion warning: components rejected for
    filling a quarter of their own bounding box are usually two objects joined
    by a bridge of pixels.
    """
    # Keyed by (dataset, class) because a mixed index has several: `car` is 5
    # in Kust4K and something else elsewhere, and collapsing them would add two
    # unrelated distributions together.
    per_class: dict[tuple[str, int], list[Instance]] = {}
    rejects: dict[str, int] = {}
    naming: dict[str, DatasetSpec] = {}
    for entry in index:
        owner = entry.source.spec if entry.source else spec
        label = owner.name if owner else "?"
        naming[label] = owner
        for instance in entry.instances:
            per_class.setdefault((label, instance.class_id), []).append(instance)
        for reason, count in entry.rejects.items():
            rejects[reason] = rejects.get(reason, 0) + count

    total = sum(len(v) for v in per_class.values())
    empty = sum(1 for e in index if not e.instances)
    lines = [
        f"{len(index)} frames, {total} instances kept, "
        f"{sum(rejects.values())} components rejected. "
        f"{empty} frame(s) hold no instance and are dropped.",
        "",
        "| dataset | class | instances | frames | median area px | median side px |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (label, class_id), items in sorted(per_class.items(), key=lambda kv: -len(kv[1])):
        areas = np.array([i.area for i in items], dtype=np.float64)
        sides = np.array([max(i.width, i.height) for i in items], dtype=np.float64)
        frames = sum(1 for e in index
                     if (e.source.spec.name if e.source else label) == label
                     and any(i.class_id == class_id for i in e.instances))
        owner = naming.get(label)
        name = owner.name_of(class_id) if owner else str(class_id)
        lines.append(f"| {label} | {name} | {len(items)} | {frames} | "
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
    source: Source | None = None

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
                              size=size, instances=tuple(inside),
                              source=entry.source))
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


def sample_masks(sample: Sample) -> np.ndarray:
    """`[K, size, size]` boolean targets, one per instance, in input coordinates.

    The semantic map is decomposed again here rather than stored. `decompose`
    is deterministic, so the labels it produces now are the labels the index
    recorded, and recovering a mask is `components == instance.label` -- no
    box matching, no tolerance, no drift between what was indexed and what is
    trained on. The alternative, a mask store like the video path's, would be
    ~330 KB per instance for data that regenerates in milliseconds.

    The decoding rules come from `sample.source` rather than from an argument,
    which is what makes them impossible to disagree with the index: decomposing
    differently here would renumber the labels and silently pair every instance
    with another instance's mask. It is also what lets one batch mix datasets.
    """
    import cv2

    if sample.source is None:
        raise ValueError(
            "this Sample carries no Source, so there is no way to know how its "
            "mask should be decoded. Build windows from an index made by "
            "index_frames(frames, source).")
    source = sample.source
    components, _, _ = decompose(read_mask(sample.frame.mask), source.spec,
                                 source.gates, source.mode)
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
    "MODES", "ROLES", "Sample", "Source", "decompose", "describe_layout", "index_frames",
    "list_frames", "list_pairs", "split_bridges",
    "load_image", "load_index", "normalise", "probe_classes", "read_mask",
    "reject_reason", "replace", "sample_masks", "sample_windows", "save_index",
    "split_frames", "split_index", "summarise", "windows_for",
]
