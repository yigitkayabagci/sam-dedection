"""Box-annotated datasets, read into one shape the pool labeller consumes.

The mask-pool stage (`src/training/pool.py`) turns *detection* annotations --
one image, many boxes -- into teacher masks. The datasets that carry those
boxes each ship a different format: VisDrone is YOLO text next to the images,
HIT-UAV is one COCO json per split, DroneVehicle is VOC-ish XML whose boxes are
oriented polygons. This module reads each into a `BoxFrame` and nothing else;
everything downstream -- zoom crops, gates, storage -- is format-blind.

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
from dataclasses import dataclass, field
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
    carried on the frame as `inset`, applied when the image is decoded.
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
                    classes.append((obj.findtext("name") or "object").strip())
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
