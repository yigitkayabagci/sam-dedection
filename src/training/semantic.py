"""Rescuing semantic maps: let the teacher separate, let the drawing verify.

`aerial.decompose` turns a semantic map into instances by taking connected
components per thing class. That is the only thing a label map can do on its
own, and it has one failure that is not a detail:

    SegFly, measured  one-car-one-blob recovery 78.5 %; 20.9 % of vehicle
                      pixels sit in a component holding several vehicles
    iSAID,  measured  3 523 drawn vehicles -> 2 889 components (82.0 %),
                      26.9 % of instances sharing a blob with another

Two cars parked side by side touch, so they are one component, so stage B is
taught to answer a prompt on one car with a mask covering both. No threshold
fixes that: the pixels really are connected, and `InstanceGates.fill` only
catches the thin-bridge case, not the flush-parked one (measured on a real
six-vehicle blob: fill 0.79, through every gate).

**The map cannot separate them because it never knew there were two.** The
image does. So this module inverts the roles `pool.py` uses:

    pool.py       box from a human -> teacher draws  -> geometry verifies
    semantic.py   seed from the map -> teacher draws -> *the map* verifies

A click is what makes the separation possible: a box around a row of parked
cars contains the row and the teacher answers with the row, while a click on
one bonnet answers with one car. And because the map here is **drawn by a
human**, it can do something no box pool can -- say whether the mask that came
back is even the right class, pixel by pixel. That is `purity`, and it is the
load-bearing gate of this route the way `box_iou` is of the other.

Nothing here is asserted to be better than `decompose`. `measure_rescue` is
the function that decides: on a dataset that ships **drawn instances**, it
builds the semantic map those instances imply, runs both routes over it, and
scores each against the drawing it came from. Run it before spending GPU-hours,
exactly as `pool.calibrate_spec` is run before a harvest.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .labels import (MASK_STORE, largest_component_fraction, open_masks,
                     save_masks, zoom_window)
from .pool import RECORD_FILE, iou

INDEX_FILE = "semantic_index.jsonl"


# --------------------------------------------------------------------------
# Gates -- a different set, for a different prompt
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticGates:
    """Why a point-prompted mask is not a usable instance.

    Deliberately **not** `labels.Gates` with a field added. That set is built
    around `box_iou` -- "does the mask sit where the human drew the box" --
    and a click has no box, so the gate that carries the information there
    cannot carry it here. What replaces it is the map itself.

    `purity`      fraction of the mask's pixels whose semantic value is the
                  class the seed came from. **The load-bearing gate.** A mask
                  that swallowed the road under the car is impure even though
                  it is one clean blob in the right place; nothing geometric
                  sees that, and the drawing does.
    `containment` fraction of the mask inside the component that seeded it.
                  Purity alone would accept a mask that jumped to a *different*
                  car of the same class -- right class, wrong object -- and
                  after dedup that shows up as one seed wasted rather than as
                  an error, which is worse than it sounds when the seed count
                  is how many objects we think are there.
    `unit_area`   the mask's area over the estimated area of **one** object of
                  its class. This is the gate that catches the failure this
                  module exists to fix: a click that answered with the whole
                  parked row is several units and gets dropped, so a component
                  either separates or contributes nothing -- never a fused blob
                  wearing an instance label. The upper bound is **1.75 and not
                  2**, which is the one number here that is a judgement rather
                  than a measurement: two units is exactly the commonest
                  fusion, so a bound at 2 admits the case being fixed. It costs
                  the genuinely large single object -- a truck answering to a
                  car-median unit -- and that trade is deliberate, because the
                  two errors are not symmetric. A dropped instance is one less
                  training example; an accepted two-car mask *teaches the
                  model the exact mistake this route exists to remove*.
    `component`   one object is one blob, as everywhere else in this repo.
    `teacher_iou` the teacher's own score. Kept for symmetry and known to be
                  weak at these sizes (`labels.Gates` carries the numbers);
                  it costs nothing and it is never the reason anything passes.
    """

    purity: float = 0.65
    containment: float = 0.70
    unit_area: tuple[float, float] = (0.3, 1.75)
    component: float = 0.8
    teacher_iou: float = 0.0


@dataclass(frozen=True)
class SemanticMeasurement:
    purity: float
    containment: float
    unit_ratio: float
    component: float
    teacher_iou: float


def class_purity(mask: np.ndarray, semantic: np.ndarray,
                 class_ids: SequenceABC[int]) -> float:
    """Fraction of `mask` whose semantic value is one of `class_ids`.

    The one measurement in this repo that reads a *human's* opinion of a pixel
    rather than a geometry. An empty mask scores 0.0 rather than raising: it is
    a rejection either way, and the caller's gate table should say which.
    """
    mask = np.asarray(mask, dtype=bool)
    total = int(mask.sum())
    if not total:
        return 0.0
    values = np.asarray(semantic)[mask]
    return float(np.isin(values, np.asarray(class_ids)).sum() / total)


def measure_semantic(mask: np.ndarray, semantic: np.ndarray,
                     component: np.ndarray, class_ids: SequenceABC[int],
                     unit_area: float,
                     teacher_iou: float) -> SemanticMeasurement:
    """Everything `semantic_reject` needs, from one mask and its seed."""
    mask = np.asarray(mask, dtype=bool)
    area = float(mask.sum())
    inside = float(np.logical_and(mask, component).sum())
    return SemanticMeasurement(
        purity=class_purity(mask, semantic, class_ids),
        containment=inside / area if area else 0.0,
        unit_ratio=area / max(float(unit_area), 1e-6),
        component=largest_component_fraction(mask),
        teacher_iou=float(teacher_iou),
    )


def semantic_reject(m: SemanticMeasurement,
                    gates: SemanticGates = SemanticGates()) -> str | None:
    """The first gate this measurement fails, or None. Same contract as
    `labels.reject_reason`, so the two routes' reports read alike."""
    values = [m.purity, m.containment, m.unit_ratio, m.component, m.teacher_iou]
    if not np.isfinite(values).all():
        return "empty"
    if m.teacher_iou < gates.teacher_iou:
        return "teacher_iou"
    if m.purity < gates.purity:
        return "purity"
    if m.containment < gates.containment:
        return "containment"
    low, high = gates.unit_area
    if not low <= m.unit_ratio <= high:
        return "unit_area"
    if m.component < gates.component:
        return "component"
    return None


# --------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------


def unit_areas(instances, per_class: Mapping[int, float] | None = None
               ) -> dict[int, float]:
    """Estimated area of **one** object, per class, from the frame itself.

    The alternative is a ground-sample-distance prior -- metres per pixel times
    a car's real size -- which needs an altitude this pipeline does not always
    have and is wrong the moment a dataset mixes altitudes (SegFly flies 30,
    40 and 50 m). This uses the data instead: **most components are already one
    object** (91.9 % on SegFly, 91.5 % of iSAID's drawn vehicles land alone),
    so the *median* component area of a class estimates one object of it, and
    the fused minority pulls the median far less than it pulls a mean.

    `per_class` overrides any class where the caller does know better. A class
    with no components at all is simply absent from the result, and the caller
    decides what a seed with no unit means -- here, one seed and no `unit_area`
    gate, which is the honest thing to do when the frame said nothing.
    """
    by_class: dict[int, list[int]] = {}
    for instance in instances:
        by_class.setdefault(int(instance.class_id), []).append(int(instance.area))
    out = {cls: float(np.median(areas)) for cls, areas in by_class.items()}
    out.update({int(k): float(v) for k, v in (per_class or {}).items()})
    return out


def seed_count(area: float, unit: float | None, cap: int = 12) -> int:
    """How many objects a component of `area` plausibly holds."""
    if not unit or unit <= 0:
        return 1
    return int(np.clip(round(area / unit), 1, cap))


def seed_points(component: np.ndarray, count: int,
                interior: float = 0.5) -> list[tuple[int, int]]:
    """`count` well-spread interior points of one component, as `(x, y)`.

    Farthest-point sampling over the component's *ridge* -- the pixels whose
    distance to the boundary is at least `interior` of the deepest such
    distance. Two properties, and this route needs both:

    **Interior.** A click near the seam between two parked cars is the prompt
    most likely to come back with both. The ridge is the set of pixels that are
    unambiguously inside something, so seeding there asks the easy question.

    **Spread.** The first seed is the deepest pixel; every next one is the ridge
    pixel farthest from the seeds already placed. Suppressing a radius around
    each peak instead -- the obvious alternative -- fails on exactly the shape
    this module exists for: a row of parked cars is one long thin region, its
    distance transform is capped by the *short* axis, and a radius drawn from
    that is far smaller than one car is long, so every seed lands on the first
    vehicle. Farthest-point sampling takes its scale from the region rather
    than from the inradius, so a 2-car blob is seeded at its two ends whatever
    its aspect ratio.
    """
    import cv2

    mask = np.asarray(component, dtype=bool)
    count = max(int(count), 1)
    if not mask.any():
        return []
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    peak = float(distance.max())
    if peak <= 0:
        return []
    ridge = np.argwhere(distance >= max(peak * float(interior), 1e-6))
    if not len(ridge):
        return []

    first = int(np.argmax(distance[ridge[:, 0], ridge[:, 1]]))
    chosen = [first]
    if count > 1:
        far = np.hypot(ridge[:, 0] - ridge[first, 0],
                       ridge[:, 1] - ridge[first, 1])
        for _ in range(count - 1):
            nxt = int(np.argmax(far))
            if far[nxt] <= 0:
                break
            chosen.append(nxt)
            far = np.minimum(far, np.hypot(ridge[:, 0] - ridge[nxt, 0],
                                           ridge[:, 1] - ridge[nxt, 1]))
    return [(int(ridge[i][1]), int(ridge[i][0])) for i in chosen]


def dedupe_masks(masks: SequenceABC[np.ndarray],
                 threshold: float = 0.5) -> list[int]:
    """Indices of the masks to keep, dropping later near-duplicates.

    Two seeds inside one object return the same object twice, which is the
    expected cost of over-seeding rather than a failure -- over-seeding is the
    safe direction, since a seed too few loses an instance outright and a seed
    too many costs one forward pass. Larger masks are kept first so the
    survivor of a duplicate pair is the more complete answer.
    """
    order = sorted(range(len(masks)), key=lambda i: -int(np.sum(masks[i])))
    kept: list[int] = []
    for index in order:
        if all(iou(masks[index], masks[other]) < threshold for other in kept):
            kept.append(index)
    return sorted(kept)


# --------------------------------------------------------------------------
# One frame
# --------------------------------------------------------------------------


def rescue_frame(
    pixels: np.ndarray,
    semantic: np.ndarray,
    spec,
    teacher,
    gates: SemanticGates = SemanticGates(),
    instance_gates=None,
    mode: str = "components",
    zoom: float = 4.0,
    min_size: int = 128,
    batch_size: int = 8,
    seed_cap: int = 12,
    dedupe_iou: float = 0.5,
    units: Mapping[int, float] | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """`(masks, records)` -- the frame's instances, separated by the teacher.

    `decompose` still runs, and its components are still the only thing that
    says *where* the objects are. What changes is what a component means: a
    candidate region holding one or more objects, rather than one object by
    assertion. A component the size of one unit gets one seed and usually comes
    back as itself; a component six units large gets six seeds and comes back
    as up to six masks, each of which has to earn its place through the gates.

    The records carry the seed's component label, so a run can be read back as
    "component 14 produced three instances" -- which is the number this whole
    module exists to move.
    """
    from .aerial import InstanceGates, decompose

    components, instances, _ = decompose(
        semantic, spec, instance_gates or InstanceGates(), mode=mode)
    if not instances:
        return [], []

    if semantic.ndim == 3:
        semantic = semantic[..., 0]
    height, width = semantic.shape[:2]
    unit = unit_areas(instances, units)
    thing_ids = {int(spec.classes[name]) for name in spec.things
                 if name in spec.classes}

    jobs: list[dict] = []
    for instance in instances:
        component = components == instance.label
        count = seed_count(float(instance.area),
                           unit.get(int(instance.class_id)), seed_cap)
        x0, y0, w, h = zoom_window(np.asarray(instance.box), (width, height),
                                   zoom, min_size)
        for point in seed_points(component, count):
            jobs.append({"instance": instance, "component": component,
                         "window": (x0, y0, w, h),
                         "point": (point[0] - x0, point[1] - y0),
                         "seeds": count})
    if not jobs:
        return [], []

    # One forward pass per batch of seeds, then the gates, then the dedup --
    # in that order, because dedup across seeds of *different* components would
    # merge two genuinely different objects that happen to overlap in a crop.
    answers: list[tuple[np.ndarray, float]] = []
    for start in range(0, len(jobs), max(batch_size, 1)):
        chunk = jobs[start:start + max(batch_size, 1)]
        crops = [pixels[j["window"][1]:j["window"][1] + j["window"][3],
                        j["window"][0]:j["window"][0] + j["window"][2]]
                 for j in chunk]
        answers.extend(teacher.points_for(
            crops, [np.asarray(j["point"], dtype=np.float64) for j in chunk]))

    by_component: dict[int, list[dict]] = {}
    records: list[dict] = []
    for job, (crop_mask, score) in zip(jobs, answers):
        x0, y0, w, h = job["window"]
        full = np.zeros((height, width), dtype=bool)
        full[y0:y0 + h, x0:x0 + w] = crop_mask[:h, :w]
        instance = job["instance"]
        unit_for = unit.get(int(instance.class_id)) or float(instance.area)
        measurement = measure_semantic(
            full, semantic, job["component"], sorted(thing_ids) or [instance.class_id],
            unit_for, score)
        # A class the frame gave no unit for cannot fail a ratio against it.
        verdict = semantic_reject(measurement, gates)
        if verdict == "unit_area" and int(instance.class_id) not in unit:
            verdict = None
        records.append({"label": int(instance.label),
                        "class": int(instance.class_id),
                        "seeds": job["seeds"],
                        "purity": round(measurement.purity, 4),
                        "containment": round(measurement.containment, 4),
                        "unit_ratio": round(measurement.unit_ratio, 4),
                        "verdict": verdict})
        if verdict is None:
            by_component.setdefault(int(instance.label), []).append(
                {"mask": full, "record": records[-1]})

    masks: list[np.ndarray] = []
    for label in sorted(by_component):
        group = by_component[label]
        keep = dedupe_masks([g["mask"] for g in group], dedupe_iou)
        for index, item in enumerate(group):
            if index in keep:
                item["record"]["kept"] = True
                masks.append(item["mask"])
            else:
                item["record"]["verdict"] = "duplicate"
    return masks, records


# --------------------------------------------------------------------------
# The measurement that decides
# --------------------------------------------------------------------------


def semantic_from_instances(instance_map: np.ndarray,
                            class_of: Mapping[int, int]) -> np.ndarray:
    """The semantic map a set of drawn instances implies.

    The trick that makes this route measurable at all. A dataset with drawn
    instances knows the answer; flattening it to "class per pixel" throws that
    answer away and produces exactly the input `decompose` and `rescue_frame`
    are given in production. Run both over the flattened map, score against the
    instances it was flattened from, and the comparison is honest because
    neither route ever saw the separation.
    """
    instance_map = np.asarray(instance_map)
    out = np.zeros(instance_map.shape[:2], dtype=np.uint8)
    for value, class_id in class_of.items():
        out[instance_map == value] = int(class_id)
    return out


def match_instances(found: SequenceABC[np.ndarray],
                    drawn: SequenceABC[np.ndarray],
                    threshold: float = 0.5) -> dict:
    """Greedy one-to-one IoU matching, and what it says about a route.

    `recall` is the number this module is judged on: of the objects a human
    drew, how many came back as a mask of their own. `precision` guards the
    obvious way to cheat it -- emit a mask per seed and match something -- and
    `mean_iou` guards the other, which is emitting masks that overlap the truth
    without being it.
    """
    pairs = sorted(
        ((iou(f, d), i, j) for i, f in enumerate(found) for j, d in enumerate(drawn)),
        key=lambda t: -t[0])
    used_found: set[int] = set()
    used_drawn: set[int] = set()
    scores: list[float] = []
    for score, i, j in pairs:
        if score < threshold:
            break
        if i in used_found or j in used_drawn:
            continue
        used_found.add(i)
        used_drawn.add(j)
        scores.append(score)
    return {"drawn": len(drawn), "found": len(found), "matched": len(scores),
            "recall": len(scores) / len(drawn) if drawn else 0.0,
            "precision": len(scores) / len(found) if found else 0.0,
            "mean_iou": float(np.mean(scores)) if scores else 0.0}


def drawn_instances(instance_map: np.ndarray,
                    keep: SequenceABC[int] | None = None) -> list[np.ndarray]:
    """Every drawn instance as its own boolean mask. Background (0) is not one."""
    instance_map = np.asarray(instance_map)
    values = [int(v) for v in np.unique(instance_map) if int(v) != 0]
    if keep is not None:
        values = [v for v in values if v in set(keep)]
    return [instance_map == v for v in values]


def measure_rescue(
    samples,
    spec,
    teacher,
    gates: SemanticGates = SemanticGates(),
    instance_gates=None,
    mode: str = "components",
    zoom: float = 4.0,
    min_size: int = 128,
    batch_size: int = 8,
    seed_cap: int = 12,
    match_iou: float = 0.5,
    progress=None,
) -> dict[str, list[dict]]:
    """Both routes over data that knows the answer, scored against the answer.

    `samples` yields `(pixels, instance_map, class_of)`: the image, a map whose
    non-zero values are **drawn** instance ids, and which thing class each id
    belongs to. Every set that ships instances can produce that -- iSAID's
    `_instance_id_RGB.png` packs the id into a colour, VTUAV's split is one
    target per frame -- and nothing else here needs to know which set it was.

    Each sample is flattened to the semantic map those instances imply
    (`semantic_from_instances`), which is precisely the input production sees,
    and then:

        components  `decompose` alone -- today's stage B, the baseline
        rescue      `rescue_frame` -- seeds from the same components, the
                    teacher separating, the map verifying

    Both are matched one-to-one against the instances the map was flattened
    from. The comparison is fair because **neither route ever sees the
    separation** -- it exists only in the truth they are scored against.

    Returns `{route: [per-sample stats]}`, ready for `summarise_rescue`. A
    rescue that does not beat the baseline's recall here does not deserve the
    GPU-hours a harvest costs, and that is the whole reason this function runs
    first.
    """
    from .aerial import InstanceGates, decompose

    instance_gates = instance_gates or InstanceGates()
    routes: dict[str, list[dict]] = {"components": [], "rescue": []}

    # Pass one: flatten, decompose, and collect every component's area, because
    # the unit estimate has to be a *dataset* statistic and not a frame one.
    # Per frame it degenerates exactly where it is needed: a frame whose only
    # component is a fused pair has that pair as its median, so the seed count
    # is one and nothing separates. Across frames the singles dominate -- which
    # is the assumption `unit_areas` documents and the reason it holds.
    prepared = []
    pooled = []
    for pixels, instance_map, class_of in samples:
        truth = drawn_instances(instance_map)
        if not truth:
            continue
        semantic = semantic_from_instances(instance_map, class_of)
        components, instances, _ = decompose(semantic, spec, instance_gates,
                                             mode=mode)
        pooled.extend(instances)
        prepared.append((pixels, semantic, components, instances, truth))
    units = unit_areas(pooled)

    iterator = (progress(prepared, total=len(prepared), desc="rescue")
                if progress else prepared)
    for pixels, semantic, components, instances, truth in iterator:
        baseline = [components == i.label for i in instances]
        routes["components"].append(match_instances(baseline, truth, match_iou))
        masks, _ = rescue_frame(
            pixels, semantic, spec, teacher, gates=gates,
            instance_gates=instance_gates, mode=mode, zoom=zoom,
            min_size=min_size, batch_size=batch_size, seed_cap=seed_cap,
            units=units)
        routes["rescue"].append(match_instances(masks, truth, match_iou))
    return routes


def estimate_units(root: str | Path, spec, modality: str = "thermal",
                   mode: str = "components", instance_gates=None,
                   limit: int | None = 200, seed: int = 0,
                   progress=None) -> dict[int, float]:
    """Per-class unit area over a sample of a dataset's own frames.

    The harvest's version of `measure_rescue`'s first pass, and the same
    argument: one object's area is a property of the dataset and the altitude,
    not of the frame in hand. Frames are drawn with a seeded generator so two
    runs over one download estimate the same units and their stores stay
    comparable.
    """
    from .aerial import InstanceGates, decompose, list_frames, read_mask

    frames = list_frames(root, spec, modality)
    rng = np.random.default_rng(seed)
    if limit is not None and len(frames) > limit:
        frames = [frames[i] for i in
                  sorted(rng.choice(len(frames), limit, replace=False))]
    iterator = (progress(frames, total=len(frames), desc="units")
                if progress else frames)
    pooled = []
    for frame in iterator:
        _, instances, _ = decompose(read_mask(frame.mask), spec,
                                    instance_gates or InstanceGates(), mode=mode)
        pooled.extend(instances)
    return unit_areas(pooled)


def summarise_rescue(routes: Mapping[str, SequenceABC[dict]]) -> str:
    """One markdown table over `match_instances` rows, one column per route.

    The table the notebook's decision is read off. Recall first because that is
    the fusion question; precision beside it because a route that answers every
    seed with a mask can buy recall with noise and this is where that shows.
    """
    names = list(routes)
    lines = ["| route | drawn | found | matched | recall | precision | mean IoU |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        rows = list(routes[name])
        drawn = sum(r["drawn"] for r in rows)
        found = sum(r["found"] for r in rows)
        matched = sum(r["matched"] for r in rows)
        ious = [r["mean_iou"] for r in rows if r["matched"]]
        lines.append(
            f"| {name} | {drawn} | {found} | {matched} | "
            f"{matched / drawn if drawn else 0:.1%} | "
            f"{matched / found if found else 0:.1%} | "
            f"{np.mean(ious) if ious else 0:.3f} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# A dataset
# --------------------------------------------------------------------------


def label_semantic_pool(
    root: str | Path,
    spec,
    teacher,
    out_dir: str | Path,
    dataset: str,
    modality: str = "thermal",
    prompt: str = "self",
    gates: SemanticGates = SemanticGates(),
    instance_gates=None,
    mode: str = "components",
    zoom: float = 4.0,
    min_size: int = 128,
    batch_size: int = 8,
    seed_cap: int = 12,
    units: Mapping[int, float] | None = None,
    limit: int | None = None,
    resume: bool = True,
    progress=None,
) -> dict:
    """Every frame of a semantic dataset, rescued into per-instance masks.

    `units` is the per-class area of **one** object, from `estimate_units` over
    a sample of this same dataset. Passing it is not optional in spirit: left
    out, every frame falls back to its own median and a frame whose components
    are all fused has no way to know they are.

    The store is `labels.save_masks`', keyed by position in the frame's own
    accepted list, so stage B reads it through the reader it already has. The
    write order is `pool.py`'s crash contract, for the same reason: record
    first, store last, a frame is done when its store exists.

    `prompt="pair"` prompts the teacher on the registered twin, and carries the
    same requirement `pool.label_pool` documents and enforces -- the halves
    must be the same size, because nothing here resamples between grids.
    """
    from .aerial import list_frames, read_mask
    from .pool import _frame_dir, _image_shape, _read_rgb

    if prompt not in ("self", "pair"):
        raise ValueError(f"prompt must be self or pair, got {prompt!r}")
    out_root = Path(out_dir) / dataset
    out_root.mkdir(parents=True, exist_ok=True)

    frames = list_frames(root, spec, modality)
    chosen = frames[:limit] if limit is not None else frames
    iterator = (progress(chosen, total=len(chosen), desc=dataset)
                if progress else chosen)

    counts = {"frames": 0, "resumed": 0, "no_pair": 0, "size_mismatch": 0,
              "unreadable": 0, "components": 0, "seeds": 0, "accepted": 0}
    rejected: dict[str, int] = {}

    for frame in iterator:
        target = _frame_dir(out_root, frame.name)
        store = target / MASK_STORE
        if resume and store.is_file():
            counts["resumed"] += 1
            counts["accepted"] += len(open_masks(store))
            continue
        source = frame.image if prompt == "self" else frame.pair
        if source is None:
            counts["no_pair"] += 1
            continue
        try:
            pixels = _read_rgb(source)
            if prompt == "pair" and _image_shape(frame.image) != pixels.shape[:2]:
                counts["size_mismatch"] += 1
                continue
        except (FileNotFoundError, OSError):
            counts["unreadable"] += 1
            continue

        semantic = read_mask(frame.mask)
        if semantic.shape[:2] != pixels.shape[:2]:
            counts["size_mismatch"] += 1
            continue

        masks, records = rescue_frame(
            pixels, semantic, spec, teacher, gates=gates,
            instance_gates=instance_gates, mode=mode, zoom=zoom,
            min_size=min_size, batch_size=batch_size, seed_cap=seed_cap,
            units=units)

        counts["frames"] += 1
        counts["seeds"] += len(records)
        counts["components"] += len({r["label"] for r in records})
        counts["accepted"] += len(masks)
        for row in records:
            if row["verdict"] is not None:
                rejected[row["verdict"]] = rejected.get(row["verdict"], 0) + 1

        record = {"key": frame.name, "dataset": dataset, "modality": modality,
                  "prompt": prompt, "mode": mode, "image": str(frame.image),
                  "shape": [int(semantic.shape[0]), int(semantic.shape[1])],
                  "teacher": getattr(teacher, "model_id", type(teacher).__name__),
                  "seeds": records}
        target.mkdir(parents=True, exist_ok=True)
        (target / RECORD_FILE).write_text(json.dumps(record, indent=1) + "\n")
        save_masks(store, semantic.shape[:2],
                   {i: mask for i, mask in enumerate(masks)})

    seeds = counts["seeds"]
    return {"dataset": dataset, "modality": modality, "prompt": prompt,
            "mode": mode, **counts,
            "acceptance_rate": counts["accepted"] / seeds if seeds else 0.0,
            "instances_per_component": (
                counts["accepted"] / counts["components"]
                if counts["components"] else 0.0),
            "rejected": rejected,
            "teacher": getattr(teacher, "model_id", type(teacher).__name__)}


def summarise_semantic_pool(report: Mapping) -> str:
    """One markdown block over `label_semantic_pool`'s answer.

    `instances per component` is the row to read: at 1.00 the teacher returned
    exactly what `decompose` already had and this route bought nothing, and
    every point above it is a fused blob that came apart.
    """
    lines = ["| | |", "|---|---:|",
             f"| frames | {report['frames']} |",
             f"| components (decompose) | {report['components']} |",
             f"| seeds prompted | {report['seeds']} |",
             f"| instances accepted | {report['accepted']} |",
             f"| **instances per component** | "
             f"**{report['instances_per_component']:.2f}** |",
             f"| acceptance rate | {report['acceptance_rate']:.1%} |"]
    if report.get("rejected"):
        lines += ["", "| rejected by | seeds |", "|---|---:|"]
        for gate, count in sorted(report["rejected"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {gate} | {count} |")
    return "\n".join(lines)
