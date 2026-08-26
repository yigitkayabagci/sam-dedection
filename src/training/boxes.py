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
from collections.abc import (Callable, Iterable, Mapping,
                             Sequence as SequenceABC)
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


def yolo_label_frames(root: str | Path, names: SequenceABC[str] = VISDRONE_NAMES,
                      images: str = "images", labels: str = "labels",
                      suffix: str | None = None) -> list[BoxFrame]:
    """The same YOLO layout, indexed from the *label* side.

    `yolo_frames` lists the image folder and looks each label up beside it,
    which is the right way round when the pixels are on disk and the wrong way
    round while they are still in the archive. A detection export's labels are
    a few megabytes and unzip in seconds; its images are tens of gigabytes, and
    the census this reader feeds -- how many boxes, at what scale, from which
    source -- needs no pixels at all. So the frames are built from
    `labels/*.txt`, and `image` points at where each image *will* be whether or
    not it has been extracted yet; the caller pulls the handful it actually
    decodes out of the zip.

    That leaves the extension, which a label file does not carry. An image
    already on disk answers it per frame; otherwise `suffix` does; with
    neither, it is the extension the already-extracted images mostly use, and
    failing that `.jpg` -- what every YOLO export this repo has fetched ships.
    Pass `suffix` when the export is something else and nothing is unpacked.

    Two deliberate differences from `yolo_frames`, both because this reader
    surveys an export rather than feeding the teacher: a label file with no
    boxes is kept as a frame with none (a background image is part of what the
    export claims, and dropping it counts the download smaller than it is;
    `label_pool` skips it later, as it skips any frame with nothing to prompt),
    and a `classes.txt` sitting in the label folder is not read as one.

    `key` is the file stem, as in `yolo_frames`, so a per-file map -- the split
    or source a name came from -- indexes the frames directly. Splits are read
    one call at a time and an export that repeats a stem across two of them
    would collide on that key; the fetchers this repo uses do not.
    """
    root = Path(root)
    label_dir, image_dir = root / labels, root / images
    found = sorted(label_dir.glob("*.txt")) if label_dir.is_dir() else []
    files = [p for p in found if p.name != "classes.txt"]
    if not files:
        raise FileNotFoundError(
            f"{label_dir}: no label files. YOLO layout wants {images}/ and "
            f"{labels}/ side by side under {root}.")

    on_disk = _images_by_stem(image_dir)
    if suffix is None:
        seen: dict[str, int] = {}
        for path in on_disk.values():
            seen[path.suffix] = seen.get(path.suffix, 0) + 1
        suffix = max(seen, key=lambda s: seen[s]) if seen else ".jpg"

    frames = []
    for label in files:
        boxes, classes = [], []
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            index = int(float(parts[0]))
            boxes.append([float(v) for v in parts[1:5]])
            classes.append(names[index] if 0 <= index < len(names)
                           else f"class {index}")
        frames.append(BoxFrame(
            key=label.stem,
            image=on_disk.get(label.stem, image_dir / f"{label.stem}{suffix}"),
            boxes=np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
            classes=tuple(classes), normalized=True))
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


def _geometry_key(obj: ElementTree.Element) -> tuple | None:
    """The object's geometry exactly as the file wrote it, hashable.

    `_envelope` is lossy on purpose -- it is what a box prompt can express --
    which makes it the wrong thing to compare two annotation files with: two
    differently rotated polygons can share an envelope. This keeps all eight
    polygon coordinates (or the four `bndbox` ones) so "the same rectangle" in
    `dronevehicle_shared_frames` means the same rectangle.
    """
    polygon = obj.find("polygon")
    if polygon is not None:
        try:
            points = tuple(float(polygon.findtext(f"{axis}{i}"))
                           for i in range(1, 5) for axis in ("x", "y"))
        except (TypeError, ValueError):
            return None
        return ("polygon",) + points
    bndbox = obj.find("bndbox")
    if bndbox is not None:
        try:
            return ("bndbox",) + tuple(
                float(bndbox.findtext(k))
                for k in ("xmin", "ymin", "xmax", "ymax"))
        except (TypeError, ValueError):
            return None
    return None


def _dronevehicle_objects(path: Path) -> list[tuple[str, tuple, list[float]]]:
    """`[(class, geometry key, envelope)]` for one DroneVehicle XML."""
    try:
        root = ElementTree.parse(path).getroot()
    except (ElementTree.ParseError, OSError):
        return []
    found = []
    for obj in root.iter("object"):
        key, envelope = _geometry_key(obj), _envelope(obj)
        if key is None or envelope is None:
            continue
        name = (obj.findtext("name") or "object").strip()
        found.append((DRONEVEHICLE_ALIASES.get(name, name), key, envelope))
    return found


def _dronevehicle_halves(root: Path):
    """Yield `(image, twin, own objects, other objects)` per registered pair.

    The four folders DroneVehicle ships -- `*img` / `*imgr` for the two
    modalities and `*label` / `*labelr` for their two separate annotation sets
    -- under whichever split prefix the archive used. `train`, `val` and
    `test` all follow it, so a reader written against this walks all three
    without knowing their names.
    """
    for rgb_dir in sorted(root.rglob("*img")):
        if not rgb_dir.is_dir() or not rgb_dir.name.endswith("img"):
            continue
        prefix = rgb_dir.name[:-len("img")]
        folders = {
            "rgb": rgb_dir,
            "thermal": rgb_dir.parent / f"{prefix}imgr",
            "rgb_labels": rgb_dir.parent / f"{prefix}label",
            "thermal_labels": rgb_dir.parent / f"{prefix}labelr",
        }
        if not all(p.is_dir() for p in folders.values()):
            continue
        twins = _images_by_stem(folders["thermal"])
        for image in sorted(rgb_dir.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            twin = twins.get(image.stem)
            if twin is None:
                continue
            yield (image, twin,
                   _dronevehicle_objects(folders["rgb_labels"] / f"{image.stem}.xml"),
                   _dronevehicle_objects(folders["thermal_labels"] / f"{image.stem}.xml"))


def _centre(box: SequenceABC[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def dronevehicle_shared_frames(root: str | Path,
                               border: int = 100) -> list[BoxFrame]:
    """Only the boxes both modalities annotate identically, prompted on RGB.

    DroneVehicle labels its two halves separately, and a census of all 35 980
    train XMLs says how separately: 53.0 % of thermal boxes (167 644 of
    316 412, over 13 129 of 17 990 frames) appear in the RGB file as the *same
    polygon under the same class*, another 36 % sit within 40 px of one, and
    10.6 % have no RGB counterpart at all.

    That first slice is the one worth a single teacher pass. The two halves are
    registered -- for these boxes the annotators wrote the same coordinates
    twice, so the disagreement is zero by construction rather than by
    measurement -- which means one mask made from the RGB pixels is a correct
    mask for the thermal frame at the same coordinates. `label_pool(...,
    mirror=...)` writes it into both pools without a second forward pass.

    So `image` is the **RGB** frame (the pixels the teacher reads, and where it
    is strongest) and `pair` is its thermal twin (where the mask is mirrored).
    That is the opposite assignment from `dronevehicle_frames(modality=...)`,
    and deliberately so.

    What this deliberately gives up: the 10.6 % of targets only the thermal
    half sees -- night vehicles, the reason the dataset exists. Those need a
    thermal-prompted pass, which is `dronevehicle_frames(modality="thermal")`
    with `prompt="self"`. This reader is the cheap half, not the whole set.

    Shared boxes are not a biased sample by size: median sqrt(area) 43.8 px
    against 44.2 px over all thermal boxes, 85.9 % at or above 32 px.
    """
    root = Path(root)
    frames: list[BoxFrame] = []
    for image, twin, rgb_objects, thermal_objects in _dronevehicle_halves(root):
        available: dict[tuple[str, tuple], int] = {}
        for name, key, _ in rgb_objects:
            available[(name, key)] = available.get((name, key), 0) + 1

        boxes, classes = [], []
        for name, key, envelope in thermal_objects:
            if available.get((name, key), 0) <= 0:
                continue
            available[(name, key)] -= 1      # one thermal box per RGB box
            boxes.append(envelope)
            classes.append(name)
        if not boxes:
            continue
        frames.append(BoxFrame(
            key=f"{image.parent.parent.name}/{image.stem}", image=image,
            boxes=np.asarray(boxes, dtype=np.float64),
            classes=tuple(classes), pair=twin, inset=border))
    if not frames:
        raise FileNotFoundError(
            f"{root}: no frame had a box both halves annotate identically -- "
            f"is this the extracted DroneVehicle archive (`*img/`, `*imgr/`, "
            f"`*label/`, `*labelr/` side by side)?")
    return frames


def dronevehicle_only_frames(root: str | Path, modality: str = "thermal",
                             distance: float = 40.0,
                             border: int = 100) -> list[BoxFrame]:
    """The targets one modality annotates and the other does not, at all.

    The complement `dronevehicle_shared_frames` leaves behind has two very
    different halves, and only one of them is worth its own harvest. About
    36 % of thermal boxes have an RGB counterpart the annotator drew slightly
    differently -- same vehicle, different rectangle -- and labelling those
    again would put two masks on one object. The rest have **no** counterpart:
    a box is *only* in this modality when no same-class box in the other file
    has its centre within `distance` pixels of it.

    Counted over all 17 990 train pairs, that split is wildly asymmetric, and
    the asymmetry is the dataset's reason to exist:

    | | thermal-only | rgb-only |
    |---|---:|---:|
    | boxes | 33 383 (10.6 %) | 3 797 (1.3 %) |
    | frames carrying any | 4 420 (24.6 %) | 2 146 (11.9 %) |
    | median sqrt(area) | 45.7 px | 36.7 px |
    | at or above 32 px | 87.0 % | 65.6 % |

    Nine times as many vehicles are visible to the thermal camera and missing
    from the visible one as the other way round -- night, mostly.

    `image` is the frame these boxes were drawn on, and it is the only frame
    they may be prompted on: the whole point is that the other modality does
    not show the target. `pair` is carried for reference, never as a prompt
    source, so pass `prompt="self"` and no `mirror`.

    `distance` at 0 reduces this to "no same-class box at the identical
    centre", which is nearly the strict complement of the shared subset; the
    40 px default is the gate the published counts above were measured with.
    """
    if modality not in ("thermal", "rgb"):
        raise ValueError(f"modality must be thermal or rgb, got {modality!r}")
    root = Path(root)
    frames: list[BoxFrame] = []
    for image, twin, rgb_objects, thermal_objects in _dronevehicle_halves(root):
        own, other = ((thermal_objects, rgb_objects) if modality == "thermal"
                      else (rgb_objects, thermal_objects))
        available = list(other)
        boxes, classes = [], []
        for name, _key, envelope in own:
            here = _centre(envelope)
            best, best_distance = -1, float("inf")
            for index, (other_name, _, other_box) in enumerate(available):
                if other_name != name:
                    continue
                there = _centre(other_box)
                gap = float(np.hypot(here[0] - there[0], here[1] - there[1]))
                if gap < best_distance:
                    best, best_distance = index, gap
            if best >= 0 and best_distance <= distance:
                available.pop(best)          # one counterpart per box
                continue
            boxes.append(envelope)
            classes.append(name)
        if not boxes:
            continue
        source, mate = ((twin, image) if modality == "thermal"
                        else (image, twin))
        frames.append(BoxFrame(
            key=f"{image.parent.parent.name}/{image.stem}", image=source,
            boxes=np.asarray(boxes, dtype=np.float64),
            classes=tuple(classes), pair=mate, inset=border))
    if not frames:
        raise FileNotFoundError(
            f"{root}: no {modality}-only box found -- is this the extracted "
            f"DroneVehicle archive (`*img/`, `*imgr/`, `*label/`, `*labelr/` "
            f"side by side)?")
    return frames


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


VTUAV_STRIDE = 10        # verified on all 20 sequences of train_ST_001


def vtuav_frames(root: str | Path, modality: str = "rgb",
                 stride: int = VTUAV_STRIDE) -> list[BoxFrame]:
    """VTUAV's tracking archives: one target per sequence, one box per row.

    Layout, read off `train_ST_001.zip`'s central directory rather than the
    paper: `<sequence>/rgb/000000.jpg`, `<sequence>/ir/000000.jpg`,
    `<sequence>/rgb.txt`, `<sequence>/ir.txt`. Frame ids are contiguous from
    zero and each box file holds `ceil(frames / 10)` lines on all 20
    sequences, so **line k is frame `k * stride`** and nine frames in ten are
    unlabelled. `fetch_datasets.tracked_members` is the extractor that knows
    the same thing.

    Rows are `x y w h`; a row whose width or height is not positive marks the
    target as absent and is dropped. The class is the sequence name's prefix
    (`bus_017` -> `bus`), which is where VTUAV's object categories live.

    **Both modalities get their own boxes, and they disagree.** Over the 3 750
    annotated rows of `train_ST_001`, only 12.2 % of `rgb.txt` and `ir.txt`
    rows are identical; the median centre offset is 8.4 px (p90 29.4) against
    a median target of 77 px. So unlike DroneVehicle's shared subset there is
    no mask here that serves both halves -- each modality is prompted on its
    own frame with its own box, which is exactly why the RGB and thermal pools
    are two notebooks that can run at the same time.

    There is also nothing to harvest as *-only: the target is one physical
    object, so it is annotated in both files or in neither. Across the same
    3 750 rows, rgb-only and ir-only are both zero.
    """
    if modality not in ("rgb", "ir"):
        raise ValueError(f"modality must be rgb or ir, got {modality!r}")
    other = "ir" if modality == "rgb" else "rgb"
    root = Path(root)

    frames: list[BoxFrame] = []
    seen = empty = 0
    for boxes_file in sorted(root.rglob(f"*/{modality}.txt")):
        sequence = boxes_file.parent
        images = _images_by_stem(sequence / modality)
        twins = _images_by_stem(sequence / other)
        seen += 1
        if not images:
            empty += 1
            continue
        for index, line in enumerate(boxes_file.read_text().splitlines()):
            cells = line.replace(",", " ").split()
            if len(cells) < 4:
                continue
            try:
                x, y, w, h = (float(v) for v in cells[:4])
            except ValueError:
                continue
            if w <= 0 or h <= 0:
                continue                     # the target is out of view
            stem = f"{index * stride:06d}"
            image = images.get(stem)
            if image is None:
                continue
            frames.append(BoxFrame(
                key=f"{sequence.name}/{stem}", image=image,
                boxes=np.asarray([[x, y, x + w, y + h]], dtype=np.float64),
                classes=(sequence.name.rsplit("_", 1)[0],),
                pair=twins.get(stem)))
    if not frames:
        if seen and empty == seen:
            # `tracked_members` keeps both `.txt` files but only one modality's
            # frames, so this is what a shared extraction tree looks like: the
            # box files are all there and the image folders are all empty.
            raise FileNotFoundError(
                f"{root}: found {seen} {modality}.txt file(s) and not one "
                f"{modality}/ frame. This tree was extracted for the other "
                f"modality -- give each one its own root (e.g. "
                f"{root}_{modality}) and extract again.")
        raise FileNotFoundError(
            f"{root}: no VTUAV sequence underneath -- expected "
            f"`<sequence>/{modality}.txt` beside `<sequence>/{modality}/`.")
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
# Sets that ship drawn masks instead of a box file
# --------------------------------------------------------------------------


def semantic_frames(root: str | Path, spec, modality: str = "rgb",
                    group: str = "all", gates=None,
                    progress=None) -> list[BoxFrame]:
    """Boxes read out of a *semantic map*, for sets that annotate pixels.

    Every other reader in this file parses an annotation the dataset wrote as
    boxes. Kust4K writes pixels instead -- 4 024 registered RGB-T pairs with a
    real 9-class map each -- and its boxes are the tight envelopes of the thing
    classes' connected components. `aerial.decompose` draws them: the same
    function and the same `InstanceGates` stage B trains on, so a box here and
    an instance there are the same object rather than two readings of one file.

    `group` selects on the dataset's own broken-frame manifests
    (`aerial.excluded_keys`): `clean` is the frames no manifest names, `broken`
    is the ones they do -- 1 160 of Kust4K's 4 024, where **one of the two
    modalities is deliberately corrupted** to simulate a sensor failure -- and
    `all` ignores the manifests. The distinction matters more here than
    anywhere else in this module, because the pool's `mirror` stamps one
    modality's mask onto the other's pixels: sound on `clean`, and on `broken`
    a coin flip over which half was the corrupt one. The manifests do not say
    which half broke, so `pool.modality_agreement` measures it per frame.

    `modality` is which image the boxes are carried against; the other half
    becomes `pair`, so one index serves `prompt="self"` and `prompt="pair"`
    without being rebuilt.

    Indexing decodes every map, which no other reader here does -- a box file
    is cheap and a 4 024-PNG pass is about a minute. What it buys is the
    `summarise_frames` census before any GPU time is spent.
    """
    from .aerial import (InstanceGates, decompose, excluded_keys, list_frames,
                         read_mask)

    if group not in ("all", "clean", "broken"):
        raise ValueError(f"group must be all, clean or broken, got {group!r}")
    root = Path(root)
    # The manifests name what to *drop*; this reader needs to be able to ask
    # for them, so the listing runs with the exclusion off and filters after.
    listed = list_frames(root, replace(spec, exclude=""), modality)
    named = excluded_keys(root, spec)
    if group != "all":
        want = group == "broken"
        listed = [f for f in listed
                  if (f.name.rsplit("/", 1)[-1] in named) == want]
    if not listed:
        raise FileNotFoundError(
            f"{root}: no {group} {spec.name} frames for {modality}."
            + (f"\n{len(named)} frame(s) are named by a manifest."
               if named else
               f"\nNo {spec.exclude or 'manifest'} was found *directly* under "
               f"{root} -- the glob is not recursive, so a manifest one folder "
               f"down reads as no manifest at all and every frame counts as "
               f"clean."))

    stream = (progress(listed, total=len(listed), desc=f"{spec.name}/{modality}")
              if progress else listed)
    frames: list[BoxFrame] = []
    for frame in stream:
        _, instances, _ = decompose(read_mask(frame.mask), spec,
                                    gates or InstanceGates())
        if not instances:
            continue                     # a map with no thing class in it
        frames.append(BoxFrame(
            key=frame.name,
            image=frame.image,
            boxes=np.array([i.box for i in instances], dtype=np.float64),
            classes=tuple(spec.name_of(i.class_id) for i in instances),
            pair=frame.pair))
    return frames


def kust4k_frames(root: str | Path, modality: str = "rgb",
                  group: str = "all", gates=None,
                  progress=None) -> list[BoxFrame]:
    """`semantic_frames` bound to the Kust4K spec, which owns the manifests."""
    from .aerial import SPECS

    return semantic_frames(root, SPECS["kust4k"], modality=modality,
                           group=group, gates=gates, progress=progress)


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


def scale_table(frames: SequenceABC[BoxFrame],
                groups: Mapping[str, str] | Callable[[BoxFrame], str] | None = None,
                max_rel: float = 0.5) -> str:
    """How large the boxes are, relative to their own frame, per group.

    `summarise_frames` reports a median long side in pixels and gives up on
    normalised readers, because a YOLO fraction has no honest pixel number
    until an image is decoded. This is the probe for the other half: a
    fraction is already a scale, and it is the scale that decides whether a
    box is worth a teacher pass. A merged export mixes sources that annotate
    at wildly different distances -- one set's median box is 2 % of the frame
    and another's is half of it -- and the pool's zoom crop, its gates and its
    `max_boxes` cap all behave differently at the two ends.

    `groups` splits the rows: a mapping from `BoxFrame.key` to a group name
    (a key it does not carry lands in `?`), or a callable over the frame.
    Without it there is a single row over everything.

    `max_rel` is the "this is not an object, it is the picture" line: the
    share of boxes at least that fraction of the frame, which is what a scale
    gate would drop and what a mislabelled whole-image box looks like. `< 0.01`
    is the other end -- boxes under 1 % of the frame, where the teacher's mask
    is least trustworthy.

    Each box's scale is its longer side as a fraction, each side measured
    against its own axis. Frames from pixel-coordinate readers carry no such
    fraction; their boxes are counted in a footer rather than guessed at.
    """
    def group_of(frame: BoxFrame) -> str:
        if groups is None:
            return "all"
        if callable(groups):
            return str(groups(frame))
        return str(groups.get(frame.key, "?"))

    scales: dict[str, list[float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for frame in frames:
        name = group_of(frame)
        tally = counts.setdefault(name, {"images": 0, "boxes": 0, "unsized": 0})
        tally["images"] += 1
        tally["boxes"] += len(frame.classes)
        if frame.normalized:
            boxes = frame.boxes
            scales.setdefault(name, []).extend(
                np.maximum(np.abs(boxes[:, 2]), np.abs(boxes[:, 3])).tolist())
        else:
            tally["unsized"] += len(frame.classes)
            scales.setdefault(name, [])

    def row(name: str, rel: SequenceABC[float], tally: Mapping[str, int]) -> str:
        cells = [name, str(tally["images"]), str(tally["boxes"])]
        if len(rel):
            values = np.asarray(rel, dtype=np.float64)
            over = float((values >= max_rel).mean())
            tiny = float((values < 0.01).mean())
            cells += [f"{np.median(values):.3f}",
                      f"{np.percentile(values, 90):.3f}",
                      f"{values.max():.3f}",
                      f"{over * 100:.1f} %", f"{tiny * 100:.1f} %"]
        else:
            cells += ["--"] * 5
        return "| " + " | ".join(cells) + " |"

    order = sorted(counts, key=lambda name: -counts[name]["boxes"])
    lines = [f"| group | images | boxes | median | p90 | max | >= {max_rel:g} "
             "| < 0.01 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    lines += [row(name, scales[name], counts[name]) for name in order]
    if len(order) > 1:
        lines.append(row("all", [v for name in order for v in scales[name]],
                         {"images": sum(c["images"] for c in counts.values()),
                          "boxes": sum(c["boxes"] for c in counts.values())}))
    unsized = sum(c["unsized"] for c in counts.values())
    if unsized:
        lines += ["", f"{unsized} boxes came from a pixel-coordinate reader and "
                      "carry no fraction; they are counted but not measured."]
    return "\n".join(lines)


def _relative_scales(frame: BoxFrame) -> np.ndarray:
    """Each box's longer side as a fraction of the frame, or a loud refusal."""
    if not frame.normalized:
        raise ValueError(
            f"{frame.key}: scale filtering needs normalised boxes, and this "
            "frame carries pixel coordinates. A pixel box has no scale until "
            "its image is decoded; filter it after `resolved`, or read the "
            "set with a reader that keeps YOLO fractions.")
    return np.maximum(np.abs(frame.boxes[:, 2]), np.abs(frame.boxes[:, 3]))


def filter_by_scale(frames: SequenceABC[BoxFrame], min_rel: float = 0.01,
                    max_rel: float = 0.5, large_rel: float = 0.25,
                    large_quota: float | None = None,
                    seed: int = 0) -> tuple[list[BoxFrame], dict[str, int]]:
    """Drop the boxes at both ends of `scale_table`, and cap the close-ups.

    The two ends are the same lines that table draws, so its columns predict
    what this removes. Under `min_rel` the box is a handful of pixels and the
    teacher's mask for it is mostly noise -- the gates reject most of them
    anyway, after paying for the crop. At or over `max_rel` the box is not an
    object in a scene but nearly the scene, which on a merged export is
    usually a mislabelled whole-image annotation; a mask of it teaches the
    student that the answer is always "everything".

    The quota is the other problem a merged export has, and it is a *frame*
    decision rather than a box one. One source shooting close-ups can supply
    most of the images while a mask pool wants the distant, small-object
    frames the tracker will actually see. A frame counts as a close-up when
    its largest surviving box is at least `large_rel`; `large_quota` is the
    share of the *kept* set those frames may hold, and the surplus is dropped
    whole -- a close-up frame costs a decode and a teacher pass whether one of
    its boxes is dropped or not, so trimming its boxes would save nothing.

    Solving the quota against the final set, not the input, is deliberate:
    with `s` small frames kept, at most `floor(q * s / (1 - q))` close-ups can
    survive before they exceed `q` of everything, and that is the number this
    keeps. Which ones survive is a seeded draw over the frames in key order,
    so the same export and seed select the same images on any machine.

    Returns the kept frames in their input order and a report of what went:
    every count in it is a decision this made, and `boxes_kept` plus
    `below_min`, `above_max` and `quota_boxes` is `boxes`.
    """
    if not 0 <= min_rel <= max_rel:
        raise ValueError(f"min_rel {min_rel} is not below max_rel {max_rel}")

    report = {"images": 0, "boxes": 0, "below_min": 0, "above_max": 0,
              "emptied": 0, "large": 0, "over_quota": 0, "quota_boxes": 0}
    trimmed: list[tuple[BoxFrame, bool]] = []      # (frame, is a close-up)
    for frame in frames:
        report["images"] += 1
        report["boxes"] += len(frame.classes)
        scales = _relative_scales(frame)
        keep = [i for i, rel in enumerate(scales)
                if min_rel <= rel < max_rel]
        report["below_min"] += int((scales < min_rel).sum())
        report["above_max"] += int((scales >= max_rel).sum())
        if not keep:
            report["emptied"] += 1
            continue
        if len(keep) < len(frame.classes):
            frame = replace(frame, boxes=frame.boxes[keep],
                            classes=tuple(frame.classes[i] for i in keep))
        large = bool(scales[keep].max() >= large_rel)
        report["large"] += int(large)
        trimmed.append((frame, large))

    drop: set[int] = set()
    if large_quota is not None and 0 <= large_quota < 1:
        close_ups = [i for i, (_, large) in enumerate(trimmed) if large]
        small = len(trimmed) - len(close_ups)
        allowed = int(large_quota * small / (1 - large_quota))
        if len(close_ups) > allowed:
            order = sorted(close_ups, key=lambda i: trimmed[i][0].key)
            rng = np.random.default_rng(seed)
            cut = rng.permutation(len(order))[allowed:]
            drop = {order[int(i)] for i in cut}
            report["over_quota"] = len(drop)
            report["quota_boxes"] = sum(len(trimmed[i][0].classes)
                                        for i in drop)

    kept = [frame for i, (frame, _) in enumerate(trimmed) if i not in drop]
    report["kept"] = len(kept)
    report["boxes_kept"] = sum(len(f.classes) for f in kept)
    return kept, report


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
