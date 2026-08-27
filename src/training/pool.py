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
import os
import shutil
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence as SequenceABC
from concurrent.futures import ThreadPoolExecutor
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


def luma(pixels: np.ndarray, step: int = 4) -> float:
    """Mean perceived brightness of `pixels`, 0-255, off every `step`-th row.

    The reason this exists is a question about the teacher rather than about
    the data: a promptable segmenter trained on web images is very good on a
    daylit street and much less good on a frame where the target is a smear of
    sensor noise around a headlight. Night RGB is where the pool quietly gets
    worse, and "quietly" is the part worth fixing -- the gates reject a mask
    that has drifted off its box, but a plausible mask drawn around glare
    passes them.

    So the number goes in the record and the acceptance rate can be read
    against it. It is a subsample because it is per frame and a full 1920x1080
    reduction is half a JPEG decode for a statistic that does not need the
    precision.
    """
    sample = pixels[::step, ::step] if step > 1 else pixels
    if sample.ndim == 3 and sample.shape[2] >= 3:
        sample = (0.299 * sample[..., 0].astype(np.float32)
                  + 0.587 * sample[..., 1].astype(np.float32)
                  + 0.114 * sample[..., 2].astype(np.float32))
    return round(float(sample.mean()), 1) if sample.size else float("nan")


def target_luma(pixels: np.ndarray, boxes: np.ndarray,
                indices: SequenceABC[int]) -> float:
    """`luma` inside the boxes rather than over the whole frame.

    The distinction is the whole point on aerial night footage: a scene can be
    almost black while the one thing being annotated sits under a street lamp,
    and the reverse -- a bright frame whose target is in shadow -- is just as
    common. The teacher only ever sees a zoom window around the box, so this is
    the closer statement of what it had to work with. Boxes are small, so it
    reads every pixel.
    """
    height, width = pixels.shape[:2]
    values = []
    for index in indices:
        x0, y0, x1, y1 = boxes[index]
        x0, y0 = max(int(x0), 0), max(int(y0), 0)
        x1 = min(int(np.ceil(x1)), width)
        y1 = min(int(np.ceil(y1)), height)
        if x1 > x0 and y1 > y0:
            values.append(luma(pixels[y0:y1, x0:x1], step=1))
    return round(float(np.mean(values)), 1) if values else float("nan")


def read_workers(readers: int = 0) -> int:
    """How many frames to decode at once, `0` meaning "ask the machine"."""
    return max(readers, 1) if readers else max(1, min(os.cpu_count() or 4, 16))


def ordered_map(function: Callable, items: Iterable, workers: int,
                depth: int) -> Iterable:
    """`map` on threads, results **in order**, at most `depth` in flight.

    The harvest's shape is one CPU stage feeding one GPU stage: decode a
    1920x1080 JPEG, hand its crops to a 1024-input teacher. Run serially, the
    card idles for the whole decode -- and on VTUAV that is one full-frame
    decode per *box*, because a tracking sequence carries one target per frame,
    so `frame_group` frames of decode sit in front of every batch.

    Order is kept because the caller's counters, its progress bar and its crash
    contract are all positional: a frame is done when its store exists, and a
    run that reordered its frames would still be correct but would stop being
    reproducible. `depth` rather than an unbounded queue because a decoded
    1920x1080 frame is 6.2 MB and a queue tied to the worker count would put
    gigabytes of pixels in front of a card happy with a few dozen.

    cv2's JPEG decode releases the GIL, which is the only reason threads are
    the right tool here rather than processes -- no pickling of frames, no
    second CUDA context.
    """
    if workers <= 1:
        for item in items:
            yield function(item)
        return
    source = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: deque = deque()

        def submit() -> bool:
            item = next(source, None)
            if item is None:
                return False
            pending.append(pool.submit(function, item))
            return True

        for _ in range(max(depth, workers)):
            if not submit():
                break
        while pending:
            result = pending.popleft().result()
            submit()
            yield result


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
            reading = measure(crop_mask, local_box, teacher_iou)
            verdict = reject_reason(reading, gates)
            # The whole measurement, not just the verdict it produced. Every
            # number here is already computed and thrown away, and keeping them
            # is the difference between "tighten the box-IoU cut to 0.7" being
            # a pass over the index and being a second harvest on the GPU.
            records[item_index].append({
                "i": int(box_index),
                "teacher_iou": round(float(teacher_iou), 4),
                "box_iou": round(float(reading.box_iou), 4),
                "area_ratio": round(float(reading.area_ratio), 4),
                "component": round(float(reading.component), 4),
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
    attempted -- index, all four gate readings, and either `None` or the name
    of the gate that stopped it. The caller decides what the indices mean; this
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
    min_luma: float = 0.0,
    readers: int = 0,
    read_ahead: int = 0,
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

    `min_luma` drops a frame whose **targets** are darker than this (0-255,
    `0` disables it, which is the default and the old behaviour). It is a knob
    for the RGB arm and only for it: a promptable teacher trained on web images
    reads a daylit street well and a near-black frame badly, and the gates do
    not save you there -- they catch a mask that drifted off its box, not a
    plausible one drawn around headlight glare. Do not raise it on a thermal
    harvest, where a low reading is a *cold* target and dropping those is
    exactly backwards. Every frame records `luma` and `target_luma` either way,
    so `summarise_luma` can say what the acceptance rate actually does with the
    light before anything is dropped for it.

    `readers` is how many frames are decoded at once, ahead of the teacher --
    `0` asks the machine for its core count. Decoding is the other half of this
    loop and it used to run on the same thread as inference, so the card sat
    idle through it: a 1920x1080 JPEG is ~20 ms and VTUAV carries **one box per
    frame**, which puts `frame_group` whole decodes in front of every batch.
    `read_ahead` bounds how many decoded frames may wait in host memory (`0` is
    twice the group, floored at the reader count); a decoded frame is 6.2 MB,
    so this is the knob that keeps the lookahead from becoming the memory
    problem. Order, counters and the crash contract are unchanged --
    `ordered_map` yields in submission order and every counter still moves on
    the calling thread.

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
              "too_dark": 0, "attempted": 0, "accepted": 0}
    if mirror_root is not None:
        counts.update(mirrored=0, mirror_missing=0, mirror_mismatch=0)
    rejected: dict[str, int] = {}
    by_class: dict[str, int] = {}

    def jobs():
        """The frames that still need the teacher -- one stat each, no pixels.

        Deliberately cheap and deliberately on this thread: everything it
        touches is a counter, so the reader threads below never write one.
        """
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
            yield frame, target, store, source

    def decode(job):
        """One frame's pixels and the boxes worth prompting, on any thread."""
        frame, target, store, source = job
        try:
            pixels = _read_rgb(source)
        except FileNotFoundError:
            return "unreadable", None

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
            return "no_boxes", None
        light = (luma(pixels), target_luma(pixels, boxes, indices))
        if min_luma > 0 and light[1] == light[1] and light[1] < min_luma:
            return "too_dark", None
        return "ok", (frame, target, store, pixels, boxes, indices, full_shape,
                      light)

    def write(item, masks, rows) -> None:
        frame, target, store, pixels, boxes, indices, full_shape, light = item
        counts["images"] += 1
        counts["attempted"] += len(rows)
        record = {"key": frame.key, "dataset": dataset, "prompt": prompt,
                  "image": str(frame.image),
                  "shape": [int(pixels.shape[0]), int(pixels.shape[1])],
                  "luma": light[0], "target_luma": light[1],
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
            # Every reading `label_many` measured, not the two this used to
            # keep: they are what lets a stricter cut be applied by
            # `pool_reader.index_pool(min_box_iou=...)` instead of by running
            # the teacher again, and `gate_report` ask each gate on its own
            # rather than repeating `reject_reason`'s first-failure order.
            instance = {"i": index, "class": cls,
                        "box": [round(float(v), 1) for v in boxes[index]]}
            for name in GATE_READINGS:
                if name in row:
                    instance[name] = row[name]
            instance["verdict"] = row["verdict"]
            record["instances"].append(instance)

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

    threads = read_workers(readers)
    lookahead = read_ahead if read_ahead > 0 else max(2 * frame_group, threads)
    for status, item in ordered_map(decode, jobs(), threads, lookahead):
        if status != "ok":
            if status in counts:
                counts[status] += 1
            continue
        group.append(item)
        if len(group) >= max(frame_group, 1):
            flush()
    flush()

    report = {"dataset": dataset, "prompt": prompt, "mirror": mirror,
              "readers": threads, "read_ahead": lookahead, **counts,
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


GATE_READINGS = ("teacher_iou", "box_iou", "area_ratio", "component")


def backfill_readings(out_dir: str | Path, progress=None) -> dict:
    """Recompute a pool's missing gate readings from the masks it already has.

    A pool harvested before the readings were stored carries `teacher_iou` and
    a verdict and nothing else, so `gate_report` can say nothing about the
    other three and `index_pool(min_box_iou=...)` has nothing to cut on. The
    obvious fix is to harvest again, and it is the wrong one: **the numbers do
    not need the teacher.** `box_iou`, `area_ratio` and `component` are
    functions of an accepted mask and its box, and the mask is in the store and
    the box is in the record. This walks both and writes them back -- CPU only,
    no GPU, no download, minutes rather than GPU-hours.

    All three are translation-invariant or scale-free, so recomputing them on
    the stored full-frame mask gives exactly what the harvest computed on the
    crop: `box_iou` compares two boxes, `area_ratio` is a ratio of areas, and
    `component` is a share of the mask's own pixels.

    Only **accepted** instances can be recovered. A rejected one has no mask in
    the store, and inventing a reading for it would be worse than leaving the
    gap -- so they are counted as `unrecoverable` and left alone. That is the
    right side to lose: every question worth asking here ("what would a
    stricter cut drop?") is a question about the set that was kept.
    """
    out_dir = Path(out_dir)
    records = sorted(out_dir.rglob(RECORD_FILE))
    counts = {"records": 0, "written": 0, "already": 0, "filled": 0,
              "unrecoverable": 0, "no_store": 0}
    stream = (progress(records, total=len(records), desc="backfill")
              if progress else records)
    for record_file in stream:
        counts["records"] += 1
        record = json.loads(record_file.read_text())
        instances = record.get("instances", [])
        wanted = [i for i in instances
                  if i.get("verdict") is None and i.get("box_iou") is None]
        counts["already"] += sum(
            1 for i in instances if i.get("box_iou") is not None)
        counts["unrecoverable"] += sum(
            1 for i in instances
            if i.get("verdict") is not None and i.get("box_iou") is None)
        if not wanted:
            continue
        store_file = record_file.parent / MASK_STORE
        if not store_file.is_file():
            counts["no_store"] += 1
            continue
        store = open_masks(store_file)
        changed = False
        for instance in wanted:
            mask = store.get(int(instance["i"]))
            if mask is None:
                counts["unrecoverable"] += 1
                continue
            reading = measure(mask, np.asarray(instance["box"], np.float64),
                              float(instance.get("teacher_iou", 1.0)))
            instance["box_iou"] = round(float(reading.box_iou), 4)
            instance["area_ratio"] = round(float(reading.area_ratio), 4)
            instance["component"] = round(float(reading.component), 4)
            counts["filled"] += 1
            changed = True
        if changed:
            record_file.write_text(json.dumps(record, indent=1) + "\n")
            counts["written"] += 1
    return counts


def gate_report(out_dir: str | Path, gates: Gates = Gates()) -> dict:
    """Each gate scored on its own, and what a stricter box-IoU would cost.

    `reject_reason` returns the **first** gate an instance fails, in a fixed
    order, which is right for the harvest -- one name per rejection, and the
    counts add up. It is misleading as a picture of the data. A run that
    reports `teacher_iou 2312, box_iou 9` is not saying nine masks sat badly on
    their box; it is saying nine of the ones *teacher_iou let through* did, and
    saying nothing at all about the 2 312 it stopped first.

    So this re-reads the stored readings and asks each gate independently, over
    every instance. It also reports what raising the box-IoU cut would remove
    from the accepted set, which is the number that decides whether tightening
    it is worth a thing -- `pool_reader.index_pool(min_box_iou=...)` applies it
    without touching the GPU, so the cost of the decision is this table.

    **Each reading is counted separately**, because pools are not all of one
    vintage: `teacher_iou` has been stored since the first harvest and the
    other three only since the readings were kept, so an older pool can answer
    for one gate and not the rest. Treating a record as all-or-nothing there
    reported 100 % "not recorded" on a pool that could in fact answer for the
    gate doing all the rejecting. `backfill_readings` recovers the other three
    from the stored masks without a teacher.
    """
    out_dir = Path(out_dir)
    cuts = (0.5, 0.6, 0.7, 0.8, 0.9)
    tables: dict[str, dict] = {}
    for record_file in out_dir.rglob(RECORD_FILE):
        record = json.loads(record_file.read_text())
        entry = tables.setdefault(record["dataset"], {
            "instances": 0, "accepted": 0,
            "fails": {name: 0 for name in GATE_READINGS},
            "missing": {name: 0 for name in GATE_READINGS},
            "accepted_scored": 0,
            "accepted_below": {cut: 0 for cut in cuts}})
        for instance in record["instances"]:
            entry["instances"] += 1
            accepted = instance.get("verdict") is None
            entry["accepted"] += int(accepted)
            for name in GATE_READINGS:
                value = instance.get(name)
                if value is None:
                    entry["missing"][name] += 1
                    continue
                value = float(value)
                if name == "teacher_iou" and value < gates.teacher_iou:
                    entry["fails"][name] += 1
                elif name == "box_iou" and value < gates.box_iou:
                    entry["fails"][name] += 1
                elif name == "area_ratio" and not (
                        gates.area[0] <= value <= gates.area[1]):
                    entry["fails"][name] += 1
                elif name == "component" and value < gates.component:
                    entry["fails"][name] += 1
            if accepted and instance.get("box_iou") is not None:
                entry["accepted_scored"] += 1
                for cut in cuts:
                    if float(instance["box_iou"]) < cut:
                        entry["accepted_below"][cut] += 1
    return {"cuts": list(cuts), "datasets": tables}


def summarise_gates(out_dir: str | Path, gates: Gates = Gates()) -> str:
    """`gate_report` as markdown: each gate alone, then the box-IoU cuts."""
    report = gate_report(out_dir, gates)
    lines: list[str] = []
    for dataset, entry in sorted(report["datasets"].items()):
        total = max(entry["instances"], 1)
        lines += [f"**{dataset}** -- each gate asked on its own, over all "
                  f"{entry['instances']} instances",
                  "", "| gate | would reject | share | not recorded |",
                  "|---|---:|---:|---:|"]
        for name in GATE_READINGS:
            count = entry["fails"][name]
            blind = entry["missing"][name]
            scored = max(total - blind, 1)
            lines.append(f"| `{name}` | {count} | {count / scored:.1%} | "
                         f"{blind} |")
        scored = entry["accepted_scored"]
        if not scored:
            lines += ["", f"No box-IoU reading on any of the "
                          f"{entry['accepted']} accepted instances -- this "
                          f"pool was harvested before they were stored. "
                          f"`backfill_readings` recomputes them from the "
                          f"masks already in the store; no teacher, no GPU.",
                      ""]
            continue
        lines += ["", f"| raise box_iou to | would drop of the {scored} "
                      f"scored accepted | left |", "|---|---:|---:|"]
        for cut in report["cuts"]:
            drop = entry["accepted_below"][cut]
            lines.append(f"| {cut:g} | {drop} ({drop / scored:.1%}) "
                         f"| {scored - drop} |")
        lines.append("")
    return "\n".join(lines).rstrip()


LUMA_EDGES = (24.0, 48.0, 80.0, 120.0, 160.0)


def luma_report(out_dir: str | Path,
                edges: SequenceABC[float] = LUMA_EDGES) -> dict:
    """Every record's acceptance, bucketed by how lit its targets were.

    The question this answers is the one nobody can answer by argument: a
    promptable teacher is good on a daylit street and worse on a near-black
    frame, but *how much* worse, on this data, at this altitude, is a number
    the harvest already has -- every record carries `target_luma` and every
    instance carries its verdict.

    Read it before deciding anything. A dark bucket whose acceptance holds is a
    bucket to keep; one that collapses is a `min_luma` on the next run, or an
    argument for harvesting that half of the flight on thermal instead. On a
    thermal pool the same table is a *temperature* reading and says nothing
    about night -- the modality is in the record beside it.
    """
    out_dir = Path(out_dir)
    limits = list(edges) + [float("inf")]
    empty = {"images": 0, "attempted": 0, "accepted": 0}
    tables: dict[str, list[dict]] = {}
    unknown: dict[str, dict] = {}
    for record_file in out_dir.rglob(RECORD_FILE):
        record = json.loads(record_file.read_text())
        name = record["dataset"]
        table = tables.setdefault(name, [dict(empty) for _ in limits])
        value = record.get("target_luma")
        accepted = sum(1 for i in record["instances"] if i["verdict"] is None)
        if value is None or value != value:
            row = unknown.setdefault(name, dict(empty))
        else:
            row = table[next(i for i, top in enumerate(limits) if value < top)]
        row["images"] += 1
        row["attempted"] += len(record["instances"])
        row["accepted"] += accepted
    return {"edges": list(edges), "datasets": tables, "unknown": unknown}


def summarise_luma(out_dir: str | Path,
                   edges: SequenceABC[float] = LUMA_EDGES) -> str:
    """`luma_report` as one markdown table per dataset."""
    report = luma_report(out_dir, edges)
    limits = report["edges"]
    names = [f"< {limits[0]:g}"]
    names += [f"{low:g}-{high:g}" for low, high in zip(limits, limits[1:])]
    names += [f">= {limits[-1]:g}"]

    lines: list[str] = []
    for dataset, rows in sorted(report["datasets"].items()):
        lines += [f"**{dataset}** -- acceptance by target brightness (0-255)",
                  "", "| target luma | images | boxes | accepted | rate |",
                  "|---|---:|---:|---:|---:|"]
        for name, row in zip(names, rows):
            if not row["images"]:
                continue
            rate = (row["accepted"] / row["attempted"]
                    if row["attempted"] else 0.0)
            lines.append(f"| {name} | {row['images']} | {row['attempted']} | "
                         f"{row['accepted']} | {rate:.1%} |")
        blind = report["unknown"].get(dataset)
        if blind and blind["images"]:
            lines.append(f"| not recorded | {blind['images']} | "
                         f"{blind['attempted']} | {blind['accepted']} | "
                         f"{blind['accepted'] / max(blind['attempted'], 1):.1%} |")
        lines.append("")
    return "\n".join(lines).rstrip()


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
