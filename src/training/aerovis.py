"""AeroVIS into the pool store, without a teacher pass.

Every other source in this repo arrives as boxes and needs a strong segmenter
to turn each box into a mask -- that is what `pool.label_pool` is for, and it
costs GPU-hours. AeroVIS arrives with the masks already made: 1 378 603
instances that carry a box **and** a mask on the same frame, SAM 3 output that
was then re-inspected by hand. So the work here is not labelling, it is
*translation* -- YTVIS annotations into the same `record.json` + run-length
store that `image_loop.py` already reads, so stage B cannot tell the difference
between this and a harvested pool.

Four things the schema does that a reader has to handle, all measured against
the real annotation file by `tools/inspect_aerovis.py` rather than assumed:

**`iscrowd=1` entries are ignore regions.** Eighty of them, and their `bboxes`
are `null` for every frame in the sequence. A reader that does not drop them
trips on the first one.

**`counts` is a `str`.** COCO's compressed RLE, but JSON has no bytes, so it
comes back as text and pycocotools wants bytes -- `.encode()` before decode, or
`decode_rle` below, which does it without the dependency.

**Boxes are XYWH and are not the masks' envelopes.** Only 3.8 % match the tight
envelope on all four sides; AeroVIS carries the source datasets' hand-drawn
boxes. That is the property that makes this worth having: the prompt is an
independent signal rather than the answer restated, which is exactly what a
box-prompted decoder has to be trained on.

**`areas` is the mask's pixel count**, not the box's -- 99.8 % of entries --
so it cannot be used as a box-area shortcut.

The one thing this module will not decide is the trade in
`docs/rgb_aerial_kaynaklar.md` §3.5: AeroVIS is the held-out benchmark until
somebody trains on it. `sequence_split` is what buys both halves, and it
stratifies by source prefix because **vehicle exists only in UAVDT and boat
only in SeaDronesSee** -- an unstratified split silently takes a whole class
out of one side.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Sequence as SequenceABC
from pathlib import Path

import numpy as np

# The prefix a sequence name carries, and which release it came from. Read off
# the archive's own directory listing; the triple counts per source are
# VisDrone 999 144, UAVDT 359 826, SeaDronesSee 19 633.
SOURCE_PREFIXES = {"vd_": "visdrone", "ud_": "uavdt", "sd_": "seadronessee"}


def source_of(name: str) -> str:
    """Which release a sequence came from, or `other` if the prefix is new."""
    for prefix, source in SOURCE_PREFIXES.items():
        if name.startswith(prefix):
            return source
    return "other"


def decode_rle(segmentation: dict) -> np.ndarray:
    """One COCO compressed-RLE mask as a boolean `HxW` array.

    pycocotools is used when it is installed, because it is the reference
    implementation and this is reference data. The fallback exists so a
    training box without it is not stuck, and it is checked against
    pycocotools in the tests rather than trusted.

    The fallback decodes COCO's LEB128-ish counts: six bits per byte offset by
    48, continuation in bit 5, and a sign bit on the *second* byte of a run
    that makes the value a delta from the run two back. Runs alternate starting
    with a zero run, and they are column-major -- the array is filled in
    Fortran order and returned in C order.
    """
    height, width = (int(v) for v in segmentation["size"])
    counts = segmentation["counts"]
    if isinstance(counts, str):
        counts = counts.encode()

    try:
        from pycocotools import mask as mask_utils

        return mask_utils.decode(
            {"size": [height, width], "counts": counts}).astype(bool)
    except ImportError:
        pass

    runs: list[int] = []
    index = 0
    while index < len(counts):
        value, shift, more = 0, 0, True
        while more:
            byte = counts[index] - 48
            value |= (byte & 0x1F) << shift
            more = bool(byte & 0x20)
            index += 1
            shift += 5
            if not more and (byte & 0x10):
                value |= -1 << shift
        if len(runs) > 2:
            value += runs[-2]
        runs.append(value)

    flat = np.zeros(height * width, dtype=bool)
    position, filled = 0, False
    for run in runs:
        if filled:
            flat[position:position + run] = True
        position += run
        filled = not filled
    return flat.reshape((height, width), order="F")


def load_index(path: str | Path) -> dict:
    """`aero_vis.json`, with the ignore regions already dropped.

    Returned as `{"videos", "annotations", "categories"}` so it stays the
    shape callers expect from a YTVIS file; only `annotations` is filtered.
    """
    data = json.loads(Path(path).read_text())
    for key in ("videos", "annotations", "categories"):
        if key not in data:
            raise ValueError(
                f"{path}: no {key!r} -- this is not a YTVIS annotation file. "
                f"Keys present: {sorted(data)[:8]}")
    return {"videos": data["videos"],
            "annotations": [a for a in data["annotations"]
                            if a.get("iscrowd") != 1],
            "categories": data["categories"]}


def sequence_split(videos: SequenceABC[dict], holdout: float = 0.0,
                   seed: int = 0) -> tuple[list[dict], list[dict]]:
    """`(train, held_out)` split whole sequences, stratified by source.

    Frames of one sequence are near-duplicates of each other -- the median
    track runs 117 frames -- so a frame-level split leaks the held-out half
    into training almost completely. Sequences are the unit.

    Stratified because the sources do not carry the same classes: **vehicle
    appears only in UAVDT and boat only in SeaDronesSee**, so a split that
    ignores the prefix can take an entire class out of one side and the loss
    will never mention it.

    `holdout=0.0` returns everything as training and an empty held-out list,
    which is the "AeroVIS is now training data" decision made explicit rather
    than by omission.
    """
    if not 0.0 <= holdout < 1.0:
        raise ValueError(f"holdout must be in [0, 1), got {holdout}")
    rng = np.random.default_rng(seed)
    train: list[dict] = []
    held: list[dict] = []
    by_source: dict[str, list[dict]] = {}
    for video in videos:
        by_source.setdefault(source_of(_video_name(video)), []).append(video)

    for source in sorted(by_source):
        group = sorted(by_source[source], key=_video_name)
        take = int(round(holdout * len(group)))
        if holdout > 0 and take == 0 and len(group) > 1:
            take = 1                       # a source with 3 sequences still splits
        order = rng.permutation(len(group))
        chosen = {int(i) for i in order[:take]}
        held += [group[i] for i in sorted(chosen)]
        train += [group[i] for i in range(len(group)) if i not in chosen]
    return train, held


def _video_name(video: dict) -> str:
    """A sequence's own name, from whichever field the release used."""
    if video.get("name"):
        return str(video["name"])
    files = video.get("file_names") or []
    if files:
        return str(Path(files[0]).parent.name or files[0])
    return str(video.get("id", ""))


def frame_instances(data: dict) -> dict[tuple[int, int], list[dict]]:
    """`{(video_id, frame_index): [instance, ...]}` for every labelled frame.

    An instance is kept only where the frame carries **both** a box and a
    segmentation -- 149 entries in the release have a box and no mask, and
    they are not a triple. Frames that end up with nothing are absent rather
    than empty, so a caller iterating this never writes an empty store.
    """
    frames: dict[tuple[int, int], list[dict]] = {}
    for annotation in data["annotations"]:
        video_id = annotation["video_id"]
        boxes = annotation.get("bboxes") or []
        segmentations = annotation.get("segmentations") or []
        for index, (box, segmentation) in enumerate(zip(boxes, segmentations)):
            if not box or not segmentation:
                continue
            frames.setdefault((video_id, index), []).append({
                "box": [float(v) for v in box],
                "segmentation": segmentation,
                "category_id": annotation["category_id"],
                "track_id": annotation.get("track_id")})
    return frames


def write_pool(data: dict, sequences: str | Path, out_dir: str | Path,
               dataset: str = "aerovis", videos: Iterable[dict] | None = None,
               limit: int | None = None, resume: bool = True,
               progress=None) -> dict:
    """Translate AeroVIS into per-frame pool stores, one per labelled frame.

    The output is byte-for-byte the shape `pool.label_pool` writes, so nothing
    downstream needs to know which pools were harvested and which were
    translated -- except that it can, because `record.json` records the
    provenance in `teacher` (`aerovis:ytvis`, not a model id) and keeps
    `track_id` per instance for the stage-C masklet store.

    Every instance is accepted. There are no gates here and that is the
    point: gates exist to decide whether a *teacher's* mask can be trusted,
    and these masks are the dataset's own.

    Write order is `pool.label_pool`'s crash contract exactly -- record first,
    store last -- so `resume` can read a store's existence as "this frame is
    done" and a run that died mid-frame redoes that frame and nothing else.
    """
    from .labels import MASK_STORE, save_masks
    from .pool import RECORD_FILE

    sequences = Path(sequences)
    out_root = Path(out_dir) / dataset
    out_root.mkdir(parents=True, exist_ok=True)

    names = {c["id"]: c["name"] for c in data["categories"]}
    chosen = list(videos) if videos is not None else list(data["videos"])
    wanted = {v["id"] for v in chosen}
    by_id = {v["id"]: v for v in chosen}

    frames = frame_instances(data)
    keys = sorted(k for k in frames if k[0] in wanted)
    if limit is not None:
        keys = keys[:limit]
    stream = progress(keys, total=len(keys), desc=dataset) if progress else keys

    counts = {"frames": 0, "resumed": 0, "instances": 0, "missing_image": 0,
              "no_file_name": 0}
    by_class: dict[str, int] = {}
    by_source: dict[str, int] = {}

    for video_id, index in stream:
        video = by_id[video_id]
        files = video.get("file_names") or []
        if index >= len(files):
            counts["no_file_name"] += 1
            continue
        relative = files[index]
        key = str(Path(relative).with_suffix(""))
        target = out_root.joinpath(*Path(key).parts)
        store = target / MASK_STORE
        if resume and store.is_file():
            counts["resumed"] += 1
            continue

        image = sequences / relative
        if not image.is_file():
            counts["missing_image"] += 1
            continue

        instances = frames[(video_id, index)]
        masks: dict[int, np.ndarray] = {}
        rows = []
        shape = None
        for position, instance in enumerate(instances):
            mask = decode_rle(instance["segmentation"])
            shape = mask.shape if shape is None else shape
            if mask.shape != shape:
                continue                   # two sizes in one frame: not ours
            masks[position] = mask
            x, y, w, h = instance["box"]
            rows.append({"i": position,
                         "class": names.get(instance["category_id"],
                                            f"class {instance['category_id']}"),
                         "box": [round(x, 1), round(y, 1),
                                 round(x + w, 1), round(y + h, 1)],
                         "track_id": instance["track_id"],
                         "area": int(mask.sum()),
                         "teacher_iou": None,
                         "verdict": None})
        if not masks:
            continue

        source = source_of(_video_name(video))
        # `image` is where the frame is *now*; `image_rel` is where it is
        # inside the archive. The store is a few hundred megabytes and the
        # frames are 12.6 GiB, so the pool travels to Drive without them and
        # a training run re-points at its own copy using the relative path.
        record = {"key": key, "dataset": dataset, "prompt": "dataset",
                  "image": str(image), "image_rel": str(relative),
                  "shape": [int(shape[0]), int(shape[1])],
                  "teacher": "aerovis:ytvis", "source": source,
                  "video_id": video_id, "frame_index": index,
                  "instances": rows}
        target.mkdir(parents=True, exist_ok=True)
        (target / RECORD_FILE).write_text(json.dumps(record, indent=1) + "\n")
        save_masks(store, shape, masks)

        counts["frames"] += 1
        counts["instances"] += len(masks)
        by_source[source] = by_source.get(source, 0) + len(masks)
        for row in rows:
            by_class[row["class"]] = by_class.get(row["class"], 0) + 1

    return {"dataset": dataset, "sequences": len(chosen), **counts,
            "instances_by_class": dict(sorted(by_class.items(),
                                              key=lambda kv: -kv[1])),
            "instances_by_source": by_source}


def summarise(data: dict) -> str:
    """Sequences, frames and triples per source -- the census before a run."""
    frames = frame_instances(data)
    by_video = {v["id"]: _video_name(v) for v in data["videos"]}
    rows: dict[str, list[int]] = {}
    for (video_id, _), instances in frames.items():
        source = source_of(by_video.get(video_id, ""))
        cell = rows.setdefault(source, [0, 0])
        cell[0] += 1
        cell[1] += len(instances)
    sequences: dict[str, int] = {}
    for name in by_video.values():
        source = source_of(name)
        sequences[source] = sequences.get(source, 0) + 1

    lines = ["| source | sequences | labelled frames | triples |",
             "|---|---:|---:|---:|"]
    for source in sorted(rows, key=lambda s: -rows[s][1]):
        frame_count, triples = rows[source]
        lines.append(f"| {source} | {sequences.get(source, 0)} | "
                     f"{frame_count} | {triples} |")
    total_frames = sum(v[0] for v in rows.values())
    total_triples = sum(v[1] for v in rows.values())
    lines.append(f"| **all** | {len(by_video)} | {total_frames} | "
                 f"{total_triples} |")
    return "\n".join(lines)
