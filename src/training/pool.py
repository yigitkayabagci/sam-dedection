"""Box-prompted mask pools: a strong image teacher over detection datasets.

`labels.py` turns one tracking box per frame into one pseudo-mask per frame.
This module is the same bargain struck over *detection* data -- one image,
many boxes -- and over two modalities at once, which is where the pools this
project actually needs come from:

- the **RGB pool** (notebook 13): aerial RGB detection sets (VisDrone,
  DroneVehicle's RGB half), teacher prompted on the image itself;
- the **thermal pool** (notebook 14): thermal detection sets (HIT-UAV,
  DroneVehicle's thermal half), teacher prompted either on the thermal frame
  (`prompt="self"`) or on its registered RGB twin (`prompt="pair"`), with the
  mask stored against the thermal frame either way.

Which of those two prompts to trust is not argued anywhere in this file -- it
is measured. `calibrate_spec` runs the teacher over instances whose masks a
dataset actually ships (Kust4K's 4 024 registered pairs are the working set:
real semantic masks, both modalities, day and night in the filename) and
reports IoU per class, per size, per route. The harvest cell reads that table
and then spends GPU-hours, in that order.

Everything proven by the Anti-UAV410 labeller is reused rather than re-decided:
`zoom_window` (a small target is an easy problem on a big crop), `Gates` with
per-gate reject counts (the acceptance rate is a measured number), and the
run-length store (`save_masks`), so whatever reads pseudo-masks today reads
pool masks tomorrow. One difference is deliberate: a pool frame's store is
keyed by **instance index within the frame**, not by frame index within a
sequence -- detection data has no time axis, and the record sitting beside
the store says which box each key answers.

The write order is the crash contract: `record.json` first, the `.npz` store
last. A frame is *done* when its store exists, so a run that died mid-frame
re-labels that frame and nothing else, and `--resume` (the default) never
double-counts.
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Mapping, Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .boxes import BoxFrame
from .labels import (MASK_STORE, Gates, measure, open_masks, reject_reason,
                     save_masks, zoom_window)

RECORD_FILE = "record.json"
INDEX_FILE = "pool_index.jsonl"


def _read_rgb(path: Path) -> np.ndarray:
    """HxWx3 uint8 RGB regardless of what the file held."""
    from .masklets import _read_frame

    return _read_frame(path)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


# --------------------------------------------------------------------------
# Labelling one frame
# --------------------------------------------------------------------------


def label_many(
    items: SequenceABC[tuple[np.ndarray, np.ndarray]],
    teacher,
    gates: Gates = Gates(),
    zoom: float = 4.0,
    min_size: int = 128,
    batch_size: int = 8,
) -> list[tuple[dict[int, np.ndarray], list[dict]]]:
    """Many frames' boxes through the teacher in *shared* batches.

    The reason this exists rather than a loop over `label_boxes`: batching only
    within a frame is barely batching at all on the data these pools are made
    from. DroneVehicle's shared-box subset carries a **median of 4 boxes per
    frame**, so a `batch_size` of 128 on an 80 GB card would still run 13 129
    forward passes of average size 9 -- the card idles between launches and the
    setting meant to make it fast does nothing. Pooling crops across frames
    turns the same work into a tenth as many full batches.

    Crops of different sizes batch fine (the processor pads), so frames of
    different densities pool without bucketing. Returns one `(masks, records)`
    pair per item, in the order given, each exactly what `label_boxes` would
    have returned for that frame alone.
    """
    plans: list[tuple[int, int, tuple[int, int, int, int]]] = []
    crops: list[np.ndarray] = []
    local_boxes: list[np.ndarray] = []
    for item_index, (pixels, boxes) in enumerate(items):
        height, width = pixels.shape[:2]
        for box_index, box in enumerate(boxes):
            x0, y0, w, h = zoom_window(box, (width, height), zoom, min_size)
            crops.append(pixels[y0:y0 + h, x0:x0 + w])
            local_boxes.append(np.array(
                [box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0]))
            plans.append((item_index, box_index, (x0, y0, w, h)))

    masks: list[dict[int, np.ndarray]] = [{} for _ in items]
    records: list[list[dict]] = [[] for _ in items]
    step = max(batch_size, 1)
    for start in range(0, len(crops), step):
        stop = start + step
        for (item_index, box_index, (x0, y0, w, h)), local_box, (
                crop_mask, teacher_iou) in zip(
            plans[start:stop], local_boxes[start:stop],
            teacher.masks_for(crops[start:stop], local_boxes[start:stop]),
        ):
            verdict = reject_reason(measure(crop_mask, local_box, teacher_iou),
                                    gates)
            records[item_index].append({
                "i": int(box_index),
                "teacher_iou": round(float(teacher_iou), 4),
                "verdict": verdict})
            if verdict is None:
                full = np.zeros(items[item_index][0].shape[:2], dtype=bool)
                full[y0:y0 + h, x0:x0 + w] = crop_mask[:h, :w]
                masks[item_index][int(box_index)] = full
    return list(zip(masks, records))


def label_boxes(
    pixels: np.ndarray,
    boxes: np.ndarray,
    teacher,
    gates: Gates = Gates(),
    zoom: float = 4.0,
    min_size: int = 128,
    batch_size: int = 8,
) -> tuple[dict[int, np.ndarray], list[dict]]:
    """Every box on one decoded frame, through the teacher and the gates.

    Returns `(masks, records)`: `masks` maps box index to a full-frame boolean
    mask for the boxes that survived, `records` carries one row per box
    attempted -- index, teacher confidence, and either `None` or the name of
    the gate that stopped it. The caller decides what the indices mean; this
    function only promises they are the row numbers of `boxes`.

    One frame's case of `label_many`, kept because most callers have one frame
    and this pair reads better than a list of length one at every call site.
    """
    return label_many([(pixels, boxes)], teacher, gates=gates, zoom=zoom,
                      min_size=min_size, batch_size=batch_size)[0]


# --------------------------------------------------------------------------
# Labelling a dataset
# --------------------------------------------------------------------------


def _frame_dir(out_root: Path, key: str) -> Path:
    parts = [p for p in Path(key).parts if p not in ("..", "/", "")]
    return out_root.joinpath(*parts) if parts else out_root / "frame"


def _image_size(path: Path) -> tuple[int, int] | None:
    """`(height, width)` of an image file, or None if nothing could read it.

    Only the mirror path needs this, and it needs it to be cheap -- it runs
    once per frame purely to refuse to stamp a mask onto a twin of a different
    shape. Pillow reads the header and stops, which is the cheap answer; the
    repo's own contract is numpy + OpenCV with Pillow optional, so a full
    decode is the fallback rather than the requirement.
    """
    try:
        from PIL import Image

        with Image.open(path) as handle:
            width, height = handle.size
        return int(height), int(width)
    except ImportError:
        pass
    except Exception:                       # noqa: BLE001 - unreadable file
        return None
    try:
        import cv2

        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return None if raw is None else (int(raw.shape[0]), int(raw.shape[1]))
    except Exception:                       # noqa: BLE001 - unreadable file
        return None


def _mirror(target: Path, record: dict, store: Path, dataset: str,
            image: Path) -> None:
    """The same masks, filed against the twin frame in the other pool.

    A copy, not a second forward pass: the boxes these frames carry are the
    ones both label files write identically, so the mask the teacher made from
    the RGB pixels *is* the thermal frame's mask at the same coordinates. The
    record says so -- `prompt` becomes `mirror` and `mirror_of` names the frame
    it was made on -- so nothing downstream has to infer it.

    Record first, store last, exactly as the primary write does: the store's
    existence is what `resume` reads as done.
    """
    target.mkdir(parents=True, exist_ok=True)
    twin = dict(record)
    twin["dataset"] = dataset
    twin["prompt"] = "mirror"
    twin["mirror_of"] = record.get("image")
    twin["image"] = str(image)
    (target / RECORD_FILE).write_text(json.dumps(twin, indent=1) + "\n")
    shutil.copyfile(store, target / MASK_STORE)


def label_pool(
    frames: SequenceABC[BoxFrame],
    teacher,
    out_dir: str | Path,
    dataset: str,
    prompt: str = "self",
    gates: Gates = Gates(),
    zoom: float = 4.0,
    min_size: int = 128,
    batch_size: int = 8,
    limit: int | None = None,
    max_boxes: int | None = None,
    resume: bool = True,
    mirror: str | None = None,
    frame_group: int = 1,
    progress=None,
) -> dict:
    """Run the teacher over a box dataset; write one store per image.

    `prompt` is which pixels the teacher looks at: `self` is the frame
    itself; `pair` is its registered twin (`BoxFrame.pair`), for the
    RGB-prompt route over aligned RGB-T data -- the mask is stored against
    the frame either way, which is the modality the pool supervises. Frames
    without a twin are counted and skipped under `pair` rather than silently
    labelled on the wrong pixels.

    `max_boxes` caps the boxes attempted per frame, densest images first cost
    the most and VisDrone frames carry hundreds; the cap takes the *largest*
    boxes first, because a 6-pixel box's mask is the least trustworthy thing
    the teacher makes and the gates reject most of them anyway.

    `mirror` is a second dataset name to file the *same* masks under, against
    `BoxFrame.pair`. It exists for `boxes.dronevehicle_shared_frames`, whose
    boxes are the ones both halves of an aligned RGB-T set annotate
    identically: one teacher pass over the RGB pixels then supervises both
    modalities, and the thermal pool costs a file copy instead of a second
    forward pass. A twin that is missing or a different size is counted
    (`mirror_missing`, `mirror_mismatch`) and not written -- stamping a mask
    onto pixels of another shape is the one way this could be silently wrong.

    `frame_group` is how many frames pool their crops into one set of teacher
    batches. It defaults to 1, which is the old behaviour exactly; raise it on
    a card big enough that `batch_size` outruns a single frame's box count.
    Frames are still written one at a time, after their group's inference, so
    the crash contract is unchanged.

    The report is the honest statement of what this produced: images and
    boxes attempted, accepted, skipped, and every gate's reject count.
    """
    if prompt not in ("self", "pair"):
        raise ValueError(f"prompt must be self or pair, got {prompt!r}")
    out_root = Path(out_dir) / dataset
    out_root.mkdir(parents=True, exist_ok=True)
    mirror_root = Path(out_dir) / mirror if mirror else None

    chosen = list(frames)[:limit] if limit is not None else list(frames)
    iterator = (progress(chosen, total=len(chosen), desc=dataset)
                if progress else chosen)

    counts = {"images": 0, "resumed": 0, "no_pair": 0, "unreadable": 0,
              "attempted": 0, "accepted": 0}
    if mirror_root is not None:
        counts.update(mirrored=0, mirror_missing=0, mirror_mismatch=0)
    rejected: dict[str, int] = {}
    by_class: dict[str, int] = {}

    def prepared():
        """The frames that still need the teacher, decoded and box-filtered."""
        for frame in iterator:
            target = _frame_dir(out_root, frame.key)
            store = target / MASK_STORE
            if resume and store.is_file():
                counts["resumed"] += 1
                counts["accepted"] += len(open_masks(store))
                if mirror_root is not None:
                    # A run that died between the two writes, or a mirror asked
                    # for on a pool built without one: copy, do not re-label.
                    twin_dir = _frame_dir(mirror_root, frame.key)
                    if not (twin_dir / MASK_STORE).is_file() and frame.pair:
                        _mirror(twin_dir,
                                json.loads((target / RECORD_FILE).read_text()),
                                store, mirror, frame.pair)
                        counts["mirrored"] += 1
                continue
            source = frame.image if prompt == "self" else frame.pair
            if source is None:
                counts["no_pair"] += 1
                continue
            try:
                pixels = _read_rgb(source)
            except FileNotFoundError:
                counts["unreadable"] += 1
                continue

            boxes, keep = frame.resolved(pixels.shape[:2])
            full_shape = pixels.shape[:2]
            if frame.inset:
                b = frame.inset
                pixels = pixels[b:-b, b:-b]
            indices = [i for i in range(len(boxes)) if keep[i]]
            if max_boxes is not None and len(indices) > max_boxes:
                areas = ((boxes[:, 2] - boxes[:, 0])
                         * (boxes[:, 3] - boxes[:, 1]))
                indices = sorted(
                    sorted(indices, key=lambda i: -areas[i])[:max_boxes])
            if not indices:
                continue
            yield frame, target, store, pixels, boxes, indices, full_shape

    def write(item, masks, rows) -> None:
        frame, target, store, pixels, boxes, indices, full_shape = item
        counts["images"] += 1
        counts["attempted"] += len(rows)
        record = {"key": frame.key, "dataset": dataset, "prompt": prompt,
                  "image": str(frame.image),
                  "shape": [int(pixels.shape[0]), int(pixels.shape[1])],
                  "teacher": getattr(teacher, "model_id", type(teacher).__name__),
                  "instances": []}
        for row in rows:
            index = indices[int(row["i"])]         # row in the frame's boxes
            cls = frame.classes[index]
            if row["verdict"] is None:
                counts["accepted"] += 1
                by_class[cls] = by_class.get(cls, 0) + 1
            else:
                rejected[row["verdict"]] = rejected.get(row["verdict"], 0) + 1
            record["instances"].append({
                "i": index, "class": cls,
                "box": [round(float(v), 1) for v in boxes[index]],
                "teacher_iou": row["teacher_iou"],
                "verdict": row["verdict"]})

        target.mkdir(parents=True, exist_ok=True)
        (target / RECORD_FILE).write_text(json.dumps(record, indent=1) + "\n")
        # The store is written last on purpose: its existence is what `resume`
        # reads as "this frame is done".
        save_masks(store, pixels.shape[:2],
                   {indices[i]: m for i, m in masks.items()})

        if mirror_root is not None:
            if frame.pair is None:
                counts["mirror_missing"] += 1
            elif _image_size(frame.pair) != full_shape:
                counts["mirror_mismatch"] += 1
            else:
                _mirror(_frame_dir(mirror_root, frame.key), record, store,
                        mirror, frame.pair)
                counts["mirrored"] += 1

    group: list = []

    def flush() -> None:
        if not group:
            return
        outputs = label_many(
            [(item[3], item[4][item[5]]) for item in group], teacher,
            gates=gates, zoom=zoom, min_size=min_size, batch_size=batch_size)
        for item, (masks, rows) in zip(group, outputs):
            write(item, masks, rows)
        group.clear()

    for item in prepared():
        group.append(item)
        if len(group) >= max(frame_group, 1):
            flush()
    flush()

    report = {"dataset": dataset, "prompt": prompt, "mirror": mirror, **counts,
              "acceptance_rate": (counts["accepted"] / counts["attempted"]
                                  if counts["attempted"] else 0.0),
              "rejected": rejected, "accepted_by_class": by_class,
              "teacher": getattr(teacher, "model_id", type(teacher).__name__)}
    return report


def write_index(out_dir: str | Path) -> Path:
    """Concatenate every frame's `record.json` into one `pool_index.jsonl`.

    Rebuilt from the per-frame records rather than appended during the run:
    append-only plus resume equals duplicate lines the first time a runtime
    dies mid-harvest, and a training reader has no way to know which line to
    believe. The records on disk are the truth; this file is their view.
    """
    out_dir = Path(out_dir)
    index = out_dir / INDEX_FILE
    with index.open("w") as handle:
        for record in sorted(out_dir.rglob(RECORD_FILE)):
            handle.write(json.dumps(json.loads(record.read_text())) + "\n")
    return index


def pool_report(out_dir: str | Path) -> dict:
    """The whole pool re-read off disk -- multiple runs, one statement."""
    out_dir = Path(out_dir)
    datasets: dict[str, dict] = {}
    for record_file in out_dir.rglob(RECORD_FILE):
        record = json.loads(record_file.read_text())
        entry = datasets.setdefault(record["dataset"], {
            "images": 0, "attempted": 0, "accepted": 0,
            "rejected": {}, "accepted_by_class": {}, "teachers": set()})
        entry["images"] += 1
        entry["teachers"].add(record.get("teacher", "?"))
        for instance in record["instances"]:
            entry["attempted"] += 1
            if instance["verdict"] is None:
                entry["accepted"] += 1
                cls = instance["class"]
                entry["accepted_by_class"][cls] = \
                    entry["accepted_by_class"].get(cls, 0) + 1
            else:
                entry["rejected"][instance["verdict"]] = \
                    entry["rejected"].get(instance["verdict"], 0) + 1
    for entry in datasets.values():
        entry["teachers"] = sorted(entry["teachers"])
    return datasets


def summarise_pool(datasets: Mapping[str, dict]) -> str:
    """One markdown block over `pool_report`'s answer."""
    lines = ["| dataset | images | boxes | accepted | rate | top rejects |",
             "|---|---:|---:|---:|---:|---|"]
    for name, entry in sorted(datasets.items()):
        rate = entry["accepted"] / entry["attempted"] if entry["attempted"] else 0
        top = ", ".join(f"{k} {v}" for k, v in sorted(
            entry["rejected"].items(), key=lambda kv: -kv[1])[:3]) or "—"
        lines.append(f"| {name} | {entry['images']} | {entry['attempted']} | "
                     f"{entry['accepted']} | {rate:.1%} | {top} |")
    classes: dict[str, int] = {}
    for entry in datasets.values():
        for cls, count in entry["accepted_by_class"].items():
            classes[cls] = classes.get(cls, 0) + count
    if classes:
        lines += ["", "| class | accepted masks |", "|---|---:|"]
        for cls, count in sorted(classes.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {cls} | {count} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Calibration -- the number that chooses the thermal route
# --------------------------------------------------------------------------


def calibrate_spec(
    root: str | Path,
    spec,
    teacher,
    modality: str = "thermal",
    prompt: str = "self",
    limit_frames: int | None = 200,
    per_frame: int = 8,
    zoom: float = 4.0,
    min_size: int = 128,
    batch_size: int = 8,
    seed: int = 0,
    progress=None,
) -> list[dict]:
    """Teacher-versus-drawn IoU over a dataset that ships real masks.

    `decompose` turns the semantic map into instances -- the same function,
    the same gates, that stage B trains on -- and each instance's box prompts
    the teacher on the `modality` image (`prompt="self"`) or on its registered
    twin (`prompt="pair"`). The score is IoU against the instance's own drawn
    mask, inside the teacher's crop, which is exactly the term the pool's
    `box_iou` gate cannot see: the gate knows where the mask *sits*, only a
    drawn mask knows whether its *shape* is right.

    Run it twice on Kust4K -- once per prompt -- and the two lists put a
    number on the question the thermal branch turns on: does the teacher read
    thermal well enough to prompt directly, or does the registration carry
    RGB masks across better? Per class and per size, because published
    experience says the answer differs along both.

    Frames are sampled with a seeded generator so two runs over the same
    download score the same instances; instances within a frame are taken
    largest-first, deliberately -- tiny instances are where *both* routes
    fail, and a calibration drowned in 6-pixel components measures the
    dataset's annotation noise more than the teacher.
    """
    from .aerial import decompose, list_frames, read_mask

    if prompt not in ("self", "pair"):
        raise ValueError(f"prompt must be self or pair, got {prompt!r}")
    frames = list_frames(root, spec, modality)
    rng = np.random.default_rng(seed)
    if limit_frames is not None and len(frames) > limit_frames:
        frames = [frames[i] for i in
                  sorted(rng.choice(len(frames), limit_frames, replace=False))]
    iterator = (progress(frames, total=len(frames), desc=f"calibrate/{prompt}")
                if progress else frames)

    records: list[dict] = []
    for frame in iterator:
        source = frame.image if prompt == "self" else frame.pair
        if source is None:
            continue
        semantic = read_mask(frame.mask)
        components, instances, _ = decompose(semantic, spec)
        if not instances:
            continue
        instances = sorted(instances, key=lambda i: -i.area)[:per_frame]
        pixels = _read_rgb(source)
        height, width = pixels.shape[:2]

        crops, local_boxes, windows = [], [], []
        for instance in instances:
            x0, y0, w, h = zoom_window(
                np.asarray(instance.box), (width, height), zoom, min_size)
            crops.append(pixels[y0:y0 + h, x0:x0 + w])
            box = instance.box
            local_boxes.append(np.array(
                [box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0]))
            windows.append((x0, y0, w, h))

        for start in range(0, len(crops), max(batch_size, 1)):
            stop = start + max(batch_size, 1)
            for instance, (mask, teacher_iou), (x0, y0, w, h) in zip(
                instances[start:stop],
                teacher.masks_for(crops[start:stop], local_boxes[start:stop]),
                windows[start:stop],
            ):
                drawn = components[y0:y0 + h, x0:x0 + w] == instance.label
                records.append({
                    "frame": frame.name,
                    "class": spec.name_of(instance.class_id),
                    "area": int(instance.area),
                    "teacher_iou": round(float(teacher_iou), 4),
                    "iou": round(iou(mask[:h, :w], drawn), 4)})
    return records


SIZE_BUCKETS = ((0, 16), (16, 32), (32, 64), (64, float("inf")))


def _bucket(area: int) -> str:
    side = float(np.sqrt(area))
    for low, high in SIZE_BUCKETS:
        if low <= side < high:
            return (f"≥{low} px" if high == float("inf")
                    else f"{low}–{int(high)} px")
    return "?"


def calibration_table(routes: Mapping[str, SequenceABC[dict]]) -> str:
    """Mean IoU per class and size bucket, one column per route, side by side.

    The table the harvest decision is read off: `routes` maps a route name
    (`thermal`, `rgb`) to `calibrate_spec`'s records. The overall row is last
    and the per-bucket rows exist because a mean over everything hides the
    regime this project lives in -- a 20-pixel vehicle can lose on a route
    that wins on average.
    """
    names = list(routes)
    keys: dict[tuple[str, str], dict[str, list[float]]] = {}
    for name in names:
        for record in routes[name]:
            cell = keys.setdefault((record["class"], _bucket(record["area"])),
                                   {n: [] for n in names})
            cell[name].append(record["iou"])

    def cells(values: dict[str, list[float]]) -> str:
        out = []
        for name in names:
            rows = values[name]
            out.append(f"{np.mean(rows):.3f} ({len(rows)})" if rows else "—")
        return " | ".join(out)

    header = " | ".join(f"{n} IoU (n)" for n in names)
    lines = [f"| class | size | {header} |",
             "|---|---|" + "---|" * len(names)]
    for (cls, bucket), values in sorted(keys.items()):
        lines.append(f"| {cls} | {bucket} | {cells(values)} |")
    overall = {n: [r["iou"] for r in routes[n]] for n in names}
    lines.append(f"| **all** | | {cells(overall)} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Which modality a drawn map still describes
# --------------------------------------------------------------------------


def boundary_agreement(image, semantic) -> float:
    """How much harder the image's edges push on the map's boundaries.

    One number, and it answers one question: is this picture a picture of the
    scene this map annotates? Class boundaries in a correct annotation sit on
    object outlines, so the image's gradient is stronger there than it is on
    average. The score is that ratio -- mean edge magnitude on the map's
    boundary over mean edge magnitude across the frame -- so `1.0` is "the map
    tells you nothing about where this image's edges are" and a registered
    pair scores well above it.

    It exists for Kust4K's 1 160 broken frames. The dataset corrupts **one of
    the two modalities** per frame and its manifests do not record which one,
    which leaves a choice that cannot be made from the filename: prompting a
    teacher on the corrupt half produces a confident mask of the wrong scene,
    and mirroring it onto the intact half produces two. Both halves share the
    one map, so scoring each against it separates them.

    Deliberately not a teacher pass. It is a Sobel over a 640x512 frame, it
    runs over the whole set in the time a GPU takes to warm up, and it does not
    have the failure mode a learned check would -- a strong segmenter finds
    *something* in a corrupt frame, which is exactly what has to be detected.
    """
    import cv2

    pixels = _read_rgb(Path(image)) if isinstance(image, (str, Path)) else image
    grey = (cv2.cvtColor(np.asarray(pixels), cv2.COLOR_RGB2GRAY)
            if np.ndim(pixels) == 3 else np.asarray(pixels))
    edges = np.abs(cv2.Sobel(grey.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
    edges += np.abs(cv2.Sobel(grey.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3))

    semantic = np.asarray(semantic)
    if semantic.ndim == 3:
        semantic = semantic[..., 0]
    if semantic.shape != edges.shape:
        # A pair whose halves are different sizes is a different problem, and
        # resizing the map here would invent boundaries. Say so with a score
        # that cannot be mistaken for a measurement.
        return float("nan")
    boundary = np.zeros(semantic.shape, dtype=bool)
    boundary[:-1, :] |= semantic[:-1, :] != semantic[1:, :]
    boundary[1:, :] |= semantic[:-1, :] != semantic[1:, :]
    boundary[:, :-1] |= semantic[:, :-1] != semantic[:, 1:]
    boundary[:, 1:] |= semantic[:, :-1] != semantic[:, 1:]
    if not boundary.any():
        return float("nan")               # a map of one class annotates nothing

    everywhere = float(edges.mean())
    return float(edges[boundary].mean() / everywhere) if everywhere else float("nan")


def modality_agreement(root: str | Path, spec, limit: int | None = None,
                       progress=None) -> list[dict]:
    """`boundary_agreement` for both halves of every registered pair.

    One record per frame -- `{frame, broken, rgb, thermal}` -- with `broken`
    read from the dataset's own manifests, so the clean frames in the same list
    are the reference distribution the broken ones are judged against. They
    have to be: a thermal frame's edges are not an RGB frame's edges, and an
    absolute threshold shared across the two modalities would call every
    thermal half suspicious.
    """
    from dataclasses import replace

    from .aerial import excluded_keys, list_frames, read_mask

    root = Path(root)
    frames = list_frames(root, replace(spec, exclude=""), "rgb")
    named = excluded_keys(root, spec)
    if limit is not None:
        frames = frames[:limit]
    stream = (progress(frames, total=len(frames), desc="agreement")
              if progress else frames)

    records = []
    for frame in stream:
        semantic = read_mask(frame.mask)
        records.append({
            "frame": frame.name,
            "broken": frame.name.rsplit("/", 1)[-1] in named,
            "rgb": round(boundary_agreement(frame.image, semantic), 4),
            "thermal": (round(boundary_agreement(frame.pair, semantic), 4)
                        if frame.pair else float("nan"))})
    return records


def intact_modalities(records: SequenceABC[dict], quantile: float = 0.02,
                      modalities: SequenceABC[str] = ("rgb", "thermal"),
                      ) -> dict[str, tuple[str, ...]]:
    """Per broken frame, which of its halves still describes the map.

    The threshold per modality is a low quantile of what the *clean* frames of
    the same dataset score -- 2 % by default, so roughly one clean frame in
    fifty would be called corrupt if it were in this group. Read it as the
    false-positive rate being bought, not as a tuning knob: the alternative is
    a constant that has to be re-guessed for every dataset and every sensor.

    A frame can come back with both halves intact (the corruption was mild
    enough to leave the outlines), one half (the case this is for), or none
    -- and none means *drop the frame*, not "pick the better one". Nothing
    forces exactly one half to fail, and pretending otherwise would file a
    mask of the wrong scene under a confident label, which is the failure the
    whole check exists to avoid.
    """
    clean = [r for r in records if not r["broken"]]
    if not clean:
        raise ValueError(
            "no clean frames to calibrate against -- `modality_agreement` was "
            "run over a root whose broken-frame manifests cover everything, or "
            "over none at all.")
    floors = {}
    for modality in modalities:
        scores = [r[modality] for r in clean if np.isfinite(r[modality])]
        floors[modality] = (float(np.quantile(scores, quantile))
                            if scores else float("-inf"))
    return {r["frame"]: tuple(m for m in modalities
                              if np.isfinite(r[m]) and r[m] >= floors[m])
            for r in records if r["broken"]}


def agreement_table(records: SequenceABC[dict],
                    verdicts: Mapping[str, SequenceABC[str]],
                    modalities: SequenceABC[str] = ("rgb", "thermal")) -> str:
    """The clean-versus-broken score gap, and what the verdicts did with it."""
    def median(rows, modality):
        scores = [r[modality] for r in rows if np.isfinite(r[modality])]
        return f"{np.median(scores):.2f}" if scores else "—"

    clean = [r for r in records if not r["broken"]]
    broken = [r for r in records if r["broken"]]
    lines = ["| group | frames | " + " | ".join(f"median {m}" for m in modalities) + " |",
             "|---|---:|" + "---:|" * len(modalities)]
    for name, rows in (("clean", clean), ("broken", broken)):
        lines.append(f"| {name} | {len(rows)} | "
                     + " | ".join(median(rows, m) for m in modalities) + " |")
    kept: dict[str, int] = {}
    for names in verdicts.values():
        kept["+".join(names) or "neither"] = kept.get("+".join(names) or "neither", 0) + 1
    lines += ["", "| broken frame keeps | frames |", "|---|---:|"]
    for name, count in sorted(kept.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {count} |")
    return "\n".join(lines)
