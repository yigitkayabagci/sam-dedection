"""Box-annotated datasets, read into one shape the pool labeller consumes.

The mask-pool stage (`src/training/pool.py`) turns *detection* annotations --
one image, many boxes -- into teacher masks. The datasets that carry those
boxes each ship a different format: VisDrone is YOLO text next to the images,
HIT-UAV is one COCO json per split, DroneVehicle is VOC-ish XML whose boxes are
oriented polygons, RGBTDronePerson and VTUAV-det keep their COCO jsons *outside*
the image archive, and BIRDSAI is one MOT csv per sequence. This module reads
each into a `BoxFrame` and nothing else; everything downstream -- zoom crops,
gates, storage -- is format-blind.

Two decisions the readers share:

**Boxes stay normalised until the image is decoded.** YOLO coordinates are
fractions of an image size the label file does not state, and reading every
image header twice (once to size the boxes, once to label) doubles the I/O of
an indexing pass for nothing. `BoxFrame.resolved(shape)` converts at the moment
the pixels are in hand.

**Class names come from the dataset's own files wherever it has them.** COCO
carries a `categories` table and the reader uses it; the palettes this repo
once wrote from memory were wrong three times out of three
(`aerial.DatasetSpec.palette_source` is the scar tissue). YOLO text carries
only indices, so `VISDRONE_NAMES` exists -- with its source written down -- and
`class_histogram` is the probe that checks it against a download.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ElementTree
from collections.abc import Iterable, Sequence as SequenceABC
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# The ultralytics VisDrone2019-DET conversion, which is what the mirror this
# repo fetches (`banu4prasad/VisDrone-Dataset`) ships: the devkit's categories
# minus `ignored regions` (0) and `others` (11), re-indexed from zero. Source:
# the mirror's own `visdrone.yaml`, cross-checked against the VisDrone devkit's
# category list. `class_histogram` in the notebook is the check against the
# download; if the yaml in the download disagrees, trust the download.
VISDRONE_NAMES = ("pedestrian", "people", "bicycle", "car", "van", "truck",
                  "tricycle", "awning-tricycle", "bus", "motor")

# DroneVehicle's class names are not clean, and a census of all 35 980 train
# XMLs (both modalities, 2026-08) says exactly how dirty: `feright_car` 7 419
# and `feright car` 3 773 in the thermal half alone -- the same class under two
# spellings, so a histogram keyed on either loses a third of the freight cars.
# Plus one `feright` and one `truvk`. The five real classes are car, truck,
# bus, van, freight car; `*` (one box) is left alone on purpose so the probe
# still shows it rather than the reader quietly deciding what it meant.
DRONEVEHICLE_ALIASES = {"feright car": "feright_car",
                        "feright": "feright_car",
                        "truvk": "truck"}

# BIRDSAI's MOT csv carries two class columns, and the legend is the LILA
# dataset page's own ("Annotation format"), read there rather than inferred:
# `class` is 0 for animals and 1 for humans, `species` refines it. 3 and 4
# appear only in the real half, 5-8 only in the synthetic one.
BIRDSAI_SPECIES = {-1: "unknown", 0: "human", 1: "elephant", 2: "lion",
                   3: "giraffe", 4: "dog", 5: "crocodile", 6: "hippo",
                   7: "zebra", 8: "rhino"}


@dataclass(frozen=True)
class BoxFrame:
    """One image and its box annotations, in whatever coordinates the file had.

    `pair` is the registered twin in the other modality, when the dataset ships
    one -- DroneVehicle's RGB half for its thermal half. The pool labeller can
    then prompt the teacher on either side of the pair while storing the mask
    against `image`.

    `inset` is pixels of frame padding to discard per side (DroneVehicle's
    100 px white band). It is carried here rather than applied here because the
    boxes can only be shifted when the image is decoded and cropped in the same
    breath -- shifting one without the other is a silent half-pixel bug.
    """

    key: str                                   # unique within its dataset
    image: Path
    boxes: np.ndarray                          # [N, 4] xyxy, or cxcywh fractions
    classes: tuple[str, ...]
    pair: Path | None = None
    normalized: bool = False                   # True: boxes are YOLO fractions
    inset: int = 0

    def __post_init__(self) -> None:
        boxes = np.asarray(self.boxes, dtype=np.float64).reshape(-1, 4)
        object.__setattr__(self, "boxes", boxes)
        if len(self.classes) != len(boxes):
            raise ValueError(
                f"{self.key}: {len(boxes)} boxes but {len(self.classes)} classes")

    def resolved(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """`(boxes_xyxy, keep)` in pixel coordinates of the *inset* frame.

        `shape` is `(height, width)` of the decoded image **before** the inset
        is cut; the returned boxes are shifted into the cut frame. `keep` marks
        the boxes that survive: a box with no positive extent after clipping
        fell inside the discarded border (or was degenerate in the file), and
        prompting a teacher with it would ask for a mask of nothing.
        """
        height, width = int(shape[0]), int(shape[1])
        boxes = self.boxes.astype(np.float64).copy()
        if self.normalized:
            cx, cy = boxes[:, 0] * width, boxes[:, 1] * height
            half_w, half_h = boxes[:, 2] * width / 2, boxes[:, 3] * height / 2
            boxes = np.stack([cx - half_w, cy - half_h,
                              cx + half_w, cy + half_h], axis=1)
        if self.inset:
            boxes -= float(self.inset)
            height, width = height - 2 * self.inset, width - 2 * self.inset
        clipped = boxes.copy()
        clipped[:, 0::2] = np.clip(clipped[:, 0::2], 0, width)
        clipped[:, 1::2] = np.clip(clipped[:, 1::2], 0, height)
        keep = ((clipped[:, 2] - clipped[:, 0] >= 1)
                & (clipped[:, 3] - clipped[:, 1] >= 1)
                & np.isfinite(clipped).all(axis=1))
        return clipped, keep


# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------


def _images_by_stem(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(folder.iterdir())
            if p.suffix.lower() in IMAGE_SUFFIXES} if folder.is_dir() else {}


def yolo_frames(root: str | Path, names: SequenceABC[str] = VISDRONE_NAMES,
                images: str = "images", labels: str = "labels") -> list[BoxFrame]:
    """YOLO-layout detection data: `images/*.jpg` beside `labels/*.txt`.

    Boxes stay as the file's normalised `cx cy w h`; `resolved` converts them
    when the image is decoded. A class index past the end of `names` keeps its
    number as the name (`class 12`) rather than raising -- the histogram probe
    is where a surprise index should surface, not an exception half way
    through an indexing pass.
    """
    root = Path(root)
    stems = _images_by_stem(root / images)
    if not stems:
        raise FileNotFoundError(
            f"{root / images}: no images. YOLO layout wants {images}/ and "
            f"{labels}/ side by side under {root}.")

    frames = []
    for stem, image in stems.items():
        label = root / labels / f"{stem}.txt"
        boxes, classes = [], []
        if label.is_file():
            for line in label.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                index = int(float(parts[0]))
                boxes.append([float(v) for v in parts[1:5]])
                classes.append(names[index] if 0 <= index < len(names)
                               else f"class {index}")
        if boxes:
            frames.append(BoxFrame(
                key=stem, image=image,
                boxes=np.asarray(boxes, dtype=np.float64),
                classes=tuple(classes), normalized=True))
    if not frames:
        raise FileNotFoundError(
            f"{root}: images found but not one readable label under {labels}/.")
    return frames


def coco_frames(annotations: str | Path, images_root: str | Path,
                drop: tuple[str, ...] = ("dontcare", "ignore")) -> list[BoxFrame]:
    """One COCO json's boxes, `[x, y, w, h]` converted to xyxy.

    Class names come from the json's own `categories` table -- the one format
    here that carries its palette with it. `drop` removes the annotation
    classes that mark unlabelable regions rather than objects (HIT-UAV ships a
    `DontCare` class); matched case-insensitively so `DontCare` and `dontcare`
    both go.
    """
    annotations = Path(annotations)
    images_root = Path(images_root)
    data = json.loads(annotations.read_text())
    names = {c["id"]: str(c["name"]) for c in data.get("categories", [])}
    dropped = {n.lower() for n in drop}

    by_image: dict[int, list[tuple[list[float], str]]] = {}
    for entry in data.get("annotations", []):
        name = names.get(entry.get("category_id"), f"class {entry.get('category_id')}")
        if name.lower() in dropped:
            continue
        x, y, w, h = (float(v) for v in entry["bbox"])
        by_image.setdefault(int(entry["image_id"]), []).append(
            ([x, y, x + w, y + h], name))

    frames = []
    for info in data.get("images", []):
        rows = by_image.get(int(info["id"]))
        if not rows:
            continue
        image = images_root / info["file_name"]
        frames.append(BoxFrame(
            key=Path(info["file_name"]).stem, image=image,
            boxes=np.asarray([r[0] for r in rows], dtype=np.float64),
            classes=tuple(r[1] for r in rows)))
    if not frames:
        raise ValueError(f"{annotations}: no image ended up with a usable box.")
    return frames


def _envelope(obj: ElementTree.Element) -> list[float] | None:
    """The axis-aligned envelope of one VOC object, oriented or not.

    DroneVehicle annotates oriented boxes as `<polygon>` with `x1..y4`; some
    re-releases carry plain `<bndbox>` instead, and a few objects carry both.
    The envelope of the polygon is what a box prompt can express -- the
    orientation is real information, but SAM's prompt encoder has nowhere to
    put it, and the gates measure the *mask* against its own extent, so an
    envelope inflated by a rotated vehicle costs prompt tightness, not
    correctness.
    """
    polygon = obj.find("polygon")
    if polygon is not None:
        xs = [float(polygon.findtext(f"x{i}", "nan")) for i in range(1, 5)]
        ys = [float(polygon.findtext(f"y{i}", "nan")) for i in range(1, 5)]
        if np.isfinite(xs).all() and np.isfinite(ys).all():
            return [min(xs), min(ys), max(xs), max(ys)]
    bndbox = obj.find("bndbox")
    if bndbox is not None:
        values = [float(bndbox.findtext(k, "nan"))
                  for k in ("xmin", "ymin", "xmax", "ymax")]
        if np.isfinite(values).all():
            return values
    return None


def dronevehicle_frames(root: str | Path, modality: str = "thermal",
                        border: int = 100) -> list[BoxFrame]:
    """DroneVehicle's boxes, for whichever half the pool is being built on.

    Layout (read off the archives' central directories, see
    `tools/fetch_datasets.py`): `train/trainimg` RGB, `train/trainimgr`
    thermal, labels in `trainlabel` / `trainlabelr` -- the `r` suffix marks the
    thermal side throughout, and each modality is annotated separately.
    Missing thermal labels fall back to the RGB ones; the halves are
    registered, so the box still lands on the vehicle, and the per-modality
    gate in the pool loop is what catches the residual.

    `border` is the 100 px white band every frame ships on all four sides --
    carried on the frame as `inset`, applied when the image is decoded. It is
    real, not nominal: decoding four frames off the mirror and measuring gives
    840x712 with 99.3-99.9 % of the band at exactly 255 (the rest is JPEG
    ringing) and 640x512 left inside.

    Class names go through `DRONEVEHICLE_ALIASES` on the way out, because the
    archive spells freight car two ways. Everything else is the file's own.

    Two more things the census of all 35 980 train XMLs turned up, in case a
    caller is deciding how to read them. 7.2 % of objects are a plain
    `<bndbox>` rather than a `<polygon>`, so both branches of `_envelope`
    carry real traffic. And the two halves are annotated separately but not
    independently: 53 % of thermal boxes are byte-identical to an RGB one and
    another 36 % sit within 40 px of one, while 10.6 % have no RGB counterpart
    at all -- vehicles the thermal half can see and the visible half cannot.
    That last number is the dataset's reason to exist, and it is also why the
    thermal labels are the ones to harvest.
    """
    if modality not in ("thermal", "rgb"):
        raise ValueError(f"modality must be thermal or rgb, got {modality!r}")
    root = Path(root)
    suffix = "r" if modality == "thermal" else ""
    other = "" if modality == "thermal" else "r"

    frames: list[BoxFrame] = []
    for image_dir in sorted(root.rglob(f"*img{suffix}")):
        if not image_dir.is_dir() or not image_dir.name.endswith(f"img{suffix}"):
            continue
        prefix = image_dir.name[:-len(f"img{suffix}")]
        label_dir = image_dir.parent / f"{prefix}label{suffix}"
        if not label_dir.is_dir():
            label_dir = image_dir.parent / f"{prefix}label"
        pair_dir = image_dir.parent / f"{prefix}img{other}"

        for image in sorted(image_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = label_dir / f"{image.stem}.xml"
            if not label.is_file():
                continue
            try:
                objects = ElementTree.parse(label).getroot().iter("object")
            except ElementTree.ParseError:
                continue
            boxes, classes = [], []
            for obj in objects:
                envelope = _envelope(obj)
                if envelope is not None:
                    boxes.append(envelope)
                    name = (obj.findtext("name") or "object").strip()
                    classes.append(DRONEVEHICLE_ALIASES.get(name, name))
            if not boxes:
                continue
            pair = pair_dir / image.name
            frames.append(BoxFrame(
                key=f"{image_dir.parent.name}/{image.stem}", image=image,
                boxes=np.asarray(boxes, dtype=np.float64),
                classes=tuple(classes),
                pair=pair if pair.is_file() else None,
                inset=border))
    if not frames:
        raise FileNotFoundError(
            f"{root}: no DroneVehicle-shaped pairs (an `*img{suffix}/` folder "
            f"with an `*label{suffix or ''}/` of XMLs beside it).")
    return frames


def hituav_frames(root: str | Path, split: str = "train",
                  drop: tuple[str, ...] = ("dontcare", "ignore")) -> list[BoxFrame]:
    """HIT-UAV's standard-box annotations, wherever the archive unpacked.

    The GitHub archive nests everything under a
    `HIT-UAV-Infrared-Thermal-Dataset-main/` folder whose exact name is the
    branch's, so the COCO json is *found* (`normal_json/annotations/<split>.json`)
    rather than assumed at a fixed depth. `file_name` entries have shipped both
    bare and split-prefixed across releases; the images root is chosen by
    testing the first entry against the download, which is the only authority
    that cannot be stale.
    """
    root = Path(root)
    candidates = sorted(root.rglob(f"annotations/{split}.json"))
    if not candidates:
        raise FileNotFoundError(
            f"{root}: no annotations/{split}.json anywhere underneath -- is "
            f"this the extracted HIT-UAV archive?")
    annotations = candidates[0]
    base = annotations.parent.parent          # .../normal_json

    first = next(iter(json.loads(annotations.read_text()).get("images", [])), None)
    if first is None:
        raise ValueError(f"{annotations}: an annotation file with no images.")
    images_root = base
    if not (base / first["file_name"]).is_file() \
            and (base / split / first["file_name"]).is_file():
        images_root = base / split
    return coco_frames(annotations, images_root, drop=drop)


def _with_pair(frames: SequenceABC[BoxFrame], pair_root: Path) -> list[BoxFrame]:
    """Attach each frame's twin in the other modality, when the folder is there.

    Matched by stem rather than by full name so a mirror that re-encodes one
    half (`.jpg` beside `.png`) still pairs up; a frame whose twin is missing
    keeps `pair=None` and the pool falls back to prompting on itself.
    """
    twins = _images_by_stem(pair_root)
    if not twins:
        return list(frames)
    return [replace(frame, pair=twins.get(frame.image.stem)) for frame in frames]


def _find_json(root: Path, name: str, hint: str) -> Path:
    found = sorted(root.rglob(name))
    if not found:
        raise FileNotFoundError(
            f"{root}: no {name} anywhere underneath. It is a separate Drive "
            f"file, not a member of the image zip -- {hint}")
    return found[0]


def rgbtdroneperson_frames(root: str | Path, split: str = "train",
                           modality: str = "thermal",
                           drop: tuple[str, ...] = ("uncertain", "dontcare",
                                                    "ignore")) -> list[BoxFrame]:
    """RGBTDronePerson, whose boxes are drawn on the **thermal** half.

    Layout, read off the archive's central directory:
    `train/{annotation,thermal,visible}` and `val/{...}`, with the COCO jsons
    shipped separately (`tools/fetch_datasets.py` puts them in the dataset
    root). `split` is `train`, `val`, or `sub_train` -- the publisher's
    1 013-frame subset, whose images live in `train/` and which is the only
    split annotated on **both** modalities.

    Two things to know before prompting a teacher with these boxes. The
    targets are tiny: median sqrt(area) is 11-12 px and 93 % are under 16 px,
    so the pool's zoom crop is doing all the work. And the two modalities
    disagree: matching `sub_train_thermal.json` to `sub_train_visible.json`
    (same class, nearest centre) puts the median centre offset at 11.7 px --
    about one target diameter -- so `pair` is set for completeness but
    `--prompt pair` on this dataset prompts the teacher next to the target,
    not on it. Measured numbers and method: `docs/datasets.md`.

    `crowd` boxes are kept on purpose: they are group rectangles, the
    `component` gate rejects most of them, and seeing that rejection rate in
    the report is more useful than silently dropping them. `uncertain` is
    dropped, being the dataset's own "do not score this" marker.
    """
    if modality not in ("thermal", "rgb"):
        raise ValueError(f"modality must be thermal or rgb, got {modality!r}")
    root = Path(root)
    side = "thermal" if modality == "thermal" else "visible"
    annotations = _find_json(
        root, f"{split}_{side}.json",
        f"fetch it with `python tools/fetch_datasets.py rgbtdroneperson "
        f"--dest {root}`. Only sub_train is annotated on the visible half.")

    folder = "train" if split.startswith("sub_train") else split
    images_root = next(
        (p for p in sorted(root.rglob(f"{folder}/{side}")) if p.is_dir()), None)
    if images_root is None:
        raise FileNotFoundError(f"{root}: no {folder}/{side}/ folder underneath.")
    other = "visible" if side == "thermal" else "thermal"
    return _with_pair(coco_frames(annotations, images_root, drop=drop),
                      images_root.parent / other)


def vtuavdet_frames(root: str | Path, split: str = "train",
                    modality: str = "thermal",
                    drop: tuple[str, ...] = ("uncertain", "dontcare",
                                             "ignore")) -> list[BoxFrame]:
    """VTUAV-det: VTUAV re-annotated for detection, 1920x1080, person classes.

    The publisher's two names disagree by one word, so this maps them: the
    jsons are `train_ir.json` and `val_ir.json`, while the zip's folders are
    `train/` and `test/`. Either `val` or `test` selects the second one.

    **One box set serves both modalities.** The zip's xml says
    `<folder>rgb</folder>` and gives (769,217)-(793,258) for `train/00001.jpg`;
    `train_ir.json` gives `[768,216,24,41]` for the same frame -- the same
    rectangle, off only by the inclusive-max convention. So the annotation was
    drawn once and carried across, and VTUAV's known residual misregistration
    (`docs/encoder_mimari.md` section 3) is inherited rather than annotated.
    That makes the pool's `box_iou` gate the thing to watch here: its rejection
    rate on the thermal half *is* the misregistration measurement.

    Against that, the targets are large enough for a box prompt to mean
    something -- median sqrt(area) 69 px on train, 48 px on val -- which is why
    this, and not RGBTDronePerson, is the thermal branch's prompt source.
    """
    if modality not in ("thermal", "rgb"):
        raise ValueError(f"modality must be thermal or rgb, got {modality!r}")
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train, val or test, got {split!r}")
    root = Path(root)
    stem = "train" if split == "train" else "val"
    folder = "train" if split == "train" else "test"
    side = "ir" if modality == "thermal" else "rgb"
    annotations = _find_json(
        root, f"{stem}_ir.json",
        f"fetch it with `python tools/fetch_datasets.py vtuavdet --dest {root}`.")

    images_root = next(
        (p for p in sorted(root.rglob(f"{folder}/{side}")) if p.is_dir()), None)
    if images_root is None:
        raise FileNotFoundError(f"{root}: no {folder}/{side}/ folder underneath.")
    other = "rgb" if side == "ir" else "ir"
    return _with_pair(coco_frames(annotations, images_root, drop=drop),
                      images_root.parent / other)


def birdsai_frames(root: str | Path, split: str = "TrainReal",
                   drop_noisy: bool = True,
                   drop_occluded: bool = False) -> list[BoxFrame]:
    """BIRDSAI's night-time thermal sequences, one `BoxFrame` per frame.

    Layout: `<split>/annotations/<sequence>.csv` beside
    `<split>/images/<sequence>/`, one MOT row per box:
    `frame, id, x, y, w, h, class, species, occlusion, noise`.

    `drop_noisy` is on by default and it is the decision worth arguing about.
    The dataset page says the noise flag marks boxes whose position was
    **interpolated** from neighbouring frames rather than observed; a teacher
    prompted with an interpolated rectangle on a 10 px target is being asked
    to segment whatever happens to sit there. They are kept out of the pool
    and counted in the report instead. `drop_occluded` is off, because an
    occluded target is exactly the case the memory path is meant to survive.

    The csv's `id` column -- this is the only aerial thermal set on the list
    that ships track ids -- is parsed but not carried: `BoxFrame` has no field
    for it, and grouping frames into masklets belongs to
    `src/training/masklets.py`. Class names come from `BIRDSAI_SPECIES`, with
    the coarse animal/human column as the fallback when species is unknown.
    """
    root = Path(root)
    annotation_dirs = [p for p in sorted(root.rglob(f"{split}/annotations"))
                       if p.is_dir()]
    if not annotation_dirs:
        raise FileNotFoundError(
            f"{root}: no {split}/annotations/ underneath -- fetch it with "
            f"`python tools/fetch_datasets.py birdsai --dest {root}`.")

    frames: list[BoxFrame] = []
    for annotation_dir in annotation_dirs:
        images_dir = annotation_dir.parent / "images"
        for csv_path in sorted(annotation_dir.glob("*.csv")):
            sequence = csv_path.stem
            by_number, ordered = _birdsai_images(images_dir / sequence)
            if not ordered:
                continue
            rows: dict[int, list[tuple[list[float], str]]] = {}
            for line in csv_path.read_text().splitlines():
                cells = [c.strip() for c in line.split(",")]
                if len(cells) < 10 or not cells[0].lstrip("-").isdigit():
                    continue
                number = int(cells[0])
                x, y, w, h = (float(v) for v in cells[2:6])
                kind, species = int(float(cells[6])), int(float(cells[7]))
                occluded, noisy = int(float(cells[8])), int(float(cells[9]))
                if (drop_noisy and noisy) or (drop_occluded and occluded):
                    continue
                name = BIRDSAI_SPECIES.get(species, "unknown")
                if name == "unknown":
                    name = "human" if kind == 1 else "animal"
                rows.setdefault(number, []).append(([x, y, x + w, y + h], name))

            for number, boxes in sorted(rows.items()):
                image = by_number.get(number)
                if image is None and number < len(ordered):
                    # Some sequences number their frames from the video's own
                    # offset rather than from zero. Falling back to position in
                    # the sorted listing keeps those usable; a sequence where
                    # neither works simply contributes nothing.
                    image = ordered[number]
                if image is None:
                    continue
                frames.append(BoxFrame(
                    key=f"{sequence}/{number:06d}", image=image,
                    boxes=np.asarray([b[0] for b in boxes], dtype=np.float64),
                    classes=tuple(b[1] for b in boxes)))
    if not frames:
        raise ValueError(
            f"{root}: {split} had annotations but no row survived -- with "
            f"drop_noisy={drop_noisy}, check whether this split is entirely "
            f"interpolated.")
    return frames


def _birdsai_images(folder: Path) -> tuple[dict[int, Path], list[Path]]:
    """`({frame number: path}, [paths in order])` for one BIRDSAI sequence.

    Frame numbers live in the filename's trailing field
    (`0000000060_0000000000_0000000278.jpg`), which is what the csv's first
    column indexes -- when the two agree. The ordered list is the fallback.
    """
    if not folder.is_dir():
        return {}, []
    ordered = [p for p in sorted(folder.iterdir())
               if p.suffix.lower() in IMAGE_SUFFIXES]
    by_number: dict[int, Path] = {}
    for path in ordered:
        tail = path.stem.rsplit("_", 1)[-1]
        if tail.isdigit():
            by_number.setdefault(int(tail), path)
    return by_number, ordered


# --------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------


def class_histogram(frames: Iterable[BoxFrame]) -> dict[str, int]:
    """How many boxes each class name contributes, over an index.

    The box-dataset version of `aerial.probe_classes`: what the download
    actually says, to hold against what the reader assumed. A class that the
    names table missed shows up here as `class N` -- which is a prompt to fix
    the table, not a crash.
    """
    counts: dict[str, int] = {}
    for frame in frames:
        for name in frame.classes:
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def summarise_frames(frames: SequenceABC[BoxFrame], name: str = "") -> str:
    """One markdown block: volume, class balance, box-size distribution."""
    total = sum(len(f.classes) for f in frames)
    sides = []
    for frame in frames:
        boxes = frame.boxes
        if frame.normalized:
            continue                    # fractions -- no honest pixel number
        sides.extend(np.maximum(boxes[:, 2] - boxes[:, 0],
                                boxes[:, 3] - boxes[:, 1]).tolist())
    lines = [f"{name or 'dataset'}: {len(frames)} images, {total} boxes"
             + (f", median long side {np.median(sides):.0f} px" if sides else ""),
             "", "| class | boxes |", "|---|---:|"]
    for cls, count in class_histogram(frames).items():
        lines.append(f"| {cls} | {count} |")
    return "\n".join(lines)
