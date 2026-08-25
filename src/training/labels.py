"""Pseudo-mask labels for Anti-UAV410, and the gates that decide what to keep.

Anti-UAV410 annotates **boxes**; EdgeTAM predicts **masks**. Rather than weaken
the model with a box-only objective, a large SAM 2.1 teacher is run once,
offline, box-prompted per frame, and its masks become the training target. That
is the same relationship EdgeTAM already has to SAM 2 -- it was distilled from
it -- applied to one narrow domain instead of the whole of video.

Two decisions carry most of the quality:

**Zoom.** Many Anti-UAV410 targets are a handful of pixels. Prompting the
teacher on the full frame asks it to segment a 6-pixel object in a 640x512
field, and it will not. Prompting it on a crop a few times the size of the box,
resized up, is the same model on a much easier problem.

**Gates.** A teacher mask is accepted only if four independent checks agree
(`Gates`). Frames that fail keep their `exist` supervision and fall back to a
box-shaped objective. The **acceptance rate is a measured number** reported per
sequence -- it is the honest statement of how much mask supervision this
actually produced.

The teacher runs through `transformers`, not through Meta's `sam2` package, on
purpose: EdgeTAM installs itself *as* `sam2` (it is a fork), so the two cannot
coexist in one environment. Going through `transformers` keeps labelling and
training on the same machine if need be, and keeps the on-disk mask store as
the only interface between them either way.
"""
from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..accuracy import box_from_mask, box_iou

MASK_STORE = "pseudo_masks.npz"
REPORT_FILE = "pseudo_masks.json"


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def zoom_window(
    box: np.ndarray,
    frame_size: tuple[int, int],
    zoom: float = 4.0,
    min_size: int = 128,
) -> tuple[int, int, int, int]:
    """`(x0, y0, w, h)` of a square crop around `box`, clamped to the frame.

    `zoom` is relative to the box's longer side, `min_size` is the floor that
    stops a 4-pixel target producing a 16-pixel crop -- the teacher needs some
    context to tell an object from a hot pixel.
    """
    width, height = frame_size
    x0, y0, x1, y1 = (float(v) for v in box)
    side = max(max(x1 - x0, y1 - y0) * zoom, float(min_size))
    side = min(side, float(min(width, height)))

    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    left = int(round(np.clip(cx - side / 2.0, 0, width - side)))
    top = int(round(np.clip(cy - side / 2.0, 0, height - side)))
    return left, top, int(round(side)), int(round(side))


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Gates:
    """Four independent reasons to distrust a teacher mask.

    `teacher_iou`  the teacher's own confidence in the mask it drew.
                   **Measured to be a weak gate at small target sizes.** On
                   synthetic 12-36 px targets SAM 2's self-reported score
                   averaged 0.92-0.94 while true IoU was 0.56-0.67 -- an
                   over-confidence of +0.25 to +0.38, and *worse* for the
                   larger checkpoints. At those sizes a 0.7 threshold passes
                   essentially everything, so the acceptance rate this module
                   reports is a floor on how bad the masks are, not a measure
                   of how good they are. `box_iou` against the drawn
                   annotation is the gate that carries real information.
    `box_iou`      does the mask sit where the human said the drone is?
    `area`         a mask far smaller than its box is a fragment; far larger is
                   background bleed. Both look plausible to `box_iou` alone.
    `component`    a drone is one object. A mask split across the frame is the
                   teacher having latched onto clutter as well.
    """

    teacher_iou: float = 0.7
    box_iou: float = 0.6
    area: tuple[float, float] = (0.15, 1.3)
    component: float = 0.8


@dataclass(frozen=True)
class Measurement:
    teacher_iou: float
    box_iou: float
    area_ratio: float
    component: float


def reject_reason(m: Measurement, gates: Gates = Gates()) -> str | None:
    """The first gate this measurement fails, or None if it passes them all.

    Returning *which* gate failed rather than a bool is what makes the
    acceptance report diagnostic: "40 % rejected" says the labelling is weak,
    "40 % rejected on area" says the zoom is wrong.
    """
    if not np.isfinite([m.teacher_iou, m.box_iou, m.area_ratio, m.component]).all():
        return "empty"
    if m.teacher_iou < gates.teacher_iou:
        return "teacher_iou"
    if m.box_iou < gates.box_iou:
        return "box_iou"
    if not gates.area[0] <= m.area_ratio <= gates.area[1]:
        return "area"
    if m.component < gates.component:
        return "component"
    return None


def largest_component_fraction(mask: np.ndarray) -> float:
    """Share of the mask's area held by its largest connected component."""
    import cv2

    area = int(mask.sum())
    if area == 0:
        return 0.0
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return 0.0
    # Row 0 is the background component; the object is the largest of the rest.
    return float(stats[1:, cv2.CC_STAT_AREA].max() / area)


def measure(mask: np.ndarray, box: np.ndarray, teacher_iou: float) -> Measurement:
    """Everything `reject_reason` needs, computed from one mask and its box."""
    area = float(mask.sum())
    box_area = float(max((box[2] - box[0]) * (box[3] - box[1]), 1e-6))
    mask_box = box_from_mask(mask)
    return Measurement(
        teacher_iou=float(teacher_iou),
        box_iou=0.0 if np.isnan(mask_box).any()
        else float(box_iou(mask_box[None], np.asarray(box, np.float64)[None])[0]),
        area_ratio=area / box_area,
        component=largest_component_fraction(mask),
    )


# --------------------------------------------------------------------------
# Mask store
# --------------------------------------------------------------------------


def rle_encode(mask: np.ndarray) -> np.ndarray:
    """Run lengths over the row-major flattened mask, starting with a False run.

    A drone occupies a few hundred pixels of a 512x512 frame, so a packed
    bitmap would be 32 KB per frame and a few dozen runs is ~100 bytes. Across
    410 sequences that is the difference between a store that fits on a Colab
    disk and one that does not.
    """
    flat = np.asarray(mask, dtype=bool).ravel()
    if flat.size == 0:
        return np.zeros(0, dtype=np.int32)
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(flat)) + 1, [flat.size]))
    runs = np.diff(bounds).astype(np.int32)
    return np.concatenate(([0], runs)).astype(np.int32) if flat[0] else runs


def rle_decode(runs: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Inverse of `rle_encode`."""
    runs = np.asarray(runs, dtype=np.int64)
    flat = np.zeros(int(runs.sum()), dtype=bool)
    ends = np.cumsum(runs)
    starts = ends - runs
    for start, end in zip(starts[1::2], ends[1::2]):  # odd runs are the True ones
        flat[start:end] = True
    return flat.reshape(shape)


def save_masks(path: str | Path, shape: tuple[int, int], masks: dict[int, np.ndarray]) -> Path:
    """Write accepted masks to one `.npz`, ragged runs flattened with offsets."""
    frames = sorted(masks)
    encoded = [rle_encode(masks[f]) for f in frames]
    offsets = np.concatenate(([0], np.cumsum([len(e) for e in encoded]))).astype(np.int64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        runs=np.concatenate(encoded) if encoded else np.zeros(0, np.int32),
        offsets=offsets,
        frames=np.asarray(frames, dtype=np.int32),
        shape=np.asarray(shape, dtype=np.int32),
    )
    return path


class MaskStore(Mapping):
    """One sequence's accepted masks, decoded on demand.

    A `Mapping` rather than a dict because the decoded form is what costs: a
    512x640 boolean frame is 328 KB, so a training set of thirty thousand
    labelled frames is ~10 GB of RAM held for the whole run -- more than a Colab
    runtime has, and all of it to store a few hundred lit pixels per frame. The
    runs are ~100 bytes each and decoding one is a handful of slice
    assignments, so expanding inside the data loader (off the training thread,
    see `src/training/loader.py`) costs nothing measurable and turns the store
    into a few megabytes.

    Frames the teacher could not label are absent, exactly as in the dict
    `load_masks` returns -- `clip_masks` reads a missing key as "no mask
    supervision here", which is a different thing from an empty mask.
    """

    def __init__(self, shape: tuple[int, int], runs: np.ndarray,
                 offsets: np.ndarray, frames: np.ndarray) -> None:
        self.shape = shape
        self._runs = runs
        self._offsets = offsets
        self._index = {int(frame): i for i, frame in enumerate(frames)}

    def __getitem__(self, frame: int) -> np.ndarray:
        i = self._index[int(frame)]
        return rle_decode(self._runs[self._offsets[i]:self._offsets[i + 1]], self.shape)

    def __iter__(self):
        return iter(self._index)

    def __len__(self) -> int:
        return len(self._index)


def open_masks(path: str | Path) -> MaskStore:
    """A `MaskStore` over one `.npz`, holding the runs rather than the masks."""
    with np.load(path) as data:
        return MaskStore(
            shape=tuple(int(v) for v in data["shape"]),
            runs=data["runs"],
            offsets=data["offsets"],
            frames=data["frames"],
        )


def load_masks(path: str | Path) -> tuple[tuple[int, int], dict[int, np.ndarray]]:
    """`(shape, {frame_idx: mask})`, every mask decoded up front.

    Convenient for looking at one sequence; use `open_masks` for a training run,
    where holding thousands of decoded frames is what runs the machine out of
    memory.
    """
    store = open_masks(path)
    return store.shape, {frame: store[frame] for frame in store}


# --------------------------------------------------------------------------
# Teacher
# --------------------------------------------------------------------------


class _ImageTeacher:
    """One box on one crop in; one mask and its self-reported IoU out.

    Deliberately thin: everything version-sensitive about the `transformers`
    API is in `mask_for`, and everything this module actually reasons about --
    geometry, gates, storage -- is independent of it and unit-tested.
    Subclasses set `self.processor`, `self.model`, `self.model_id`,
    `self.device` and `self._batched`, and nothing else -- the same contract
    `masklets._VideoTeacher` holds one axis over.
    """

    model_id: str
    device: str

    def mask_for(self, crop_rgb: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, float]:
        """`(mask, teacher_iou)` for one box, in the crop's own coordinates."""
        import torch

        inputs = self.processor(
            images=crop_rgb,
            input_boxes=[[[float(v) for v in box]]],
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        masks = self.processor.post_process_masks(
            outputs.pred_masks.float().cpu(), inputs["original_sizes"]
        )[0]
        scores = outputs.iou_scores.float().cpu().reshape(-1)
        # A box prompt normally yields one mask, but the head can be configured
        # for three; take the one the model rates highest either way.
        best = int(scores.argmax())
        return masks.reshape(-1, *masks.shape[-2:])[best].numpy() > 0, float(scores[best])

    def masks_for(
        self, crops: list[np.ndarray], boxes: list[np.ndarray]
    ) -> list[tuple[np.ndarray, float]]:
        """`mask_for` over several crops in one forward pass.

        This is the whole cost of labelling: the teacher resizes every crop to
        its own 1024 input regardless of how small the crop was, so a 6-pixel
        drone costs exactly as much as a full frame and the only lever is how
        many go through at once. Crops of different sizes batch fine -- the
        processor resizes each and `post_process_masks` puts each mask back at
        its own `original_sizes` -- so nothing about the result changes, only
        how long it takes.
        """
        import torch

        if len(crops) == 1 or not self._batched:
            return [self.mask_for(crop, box) for crop, box in zip(crops, boxes)]

        try:
            inputs = self.processor(
                images=list(crops),
                input_boxes=[[[float(v) for v in box]] for box in boxes],
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            per_image = self.processor.post_process_masks(
                outputs.pred_masks.float().cpu(), inputs["original_sizes"]
            )
            scores = outputs.iou_scores.float().cpu()
        except Exception as exc:  # noqa: BLE001 -- any API mismatch, once
            self._batched = False
            warnings.warn(
                f"Batched teacher call failed ({exc!r}); falling back to one crop "
                "at a time for the rest of this run. Labelling will be slower "
                "but identical.",
                RuntimeWarning,
                stacklevel=2,
            )
            return [self.mask_for(crop, box) for crop, box in zip(crops, boxes)]

        out = []
        for i in range(len(crops)):
            masks = per_image[i]
            masks = masks.reshape(-1, *masks.shape[-2:])
            row = scores[i].reshape(-1)
            best = int(row.argmax())
            out.append((masks[best].numpy() > 0, float(row[best])))
        return out


def teacher_import_error(exc: ImportError, advice: str) -> SystemExit:
    """The right message for a failed teacher import, which is two failures.

    `from transformers import Sam3TrackerModel` raises ImportError with `name`
    set to `transformers` when the installed wheel is too old to carry the
    classes. That is the case `advice` is written for, and the only one where
    a version number is the answer.

    It raises the same ImportError with `name` set to something else entirely
    when importing transformers pulled in a dependency that is itself broken --
    and on Colab that is the common one, because a `pip install --upgrade` in
    cell 1 rewrites packages the kernel imported at startup. Pillow 12 lands on
    disk while `PIL._typing` from 11 is still live in `sys.modules`, the first
    module to want `PIL.ImageText` gets `cannot import name '_Ink'`, and the
    old message blamed a transformers version that was fine. Upgrading
    transformers again is exactly the wrong move there, so the two do not share
    a message: the fix is to restart the runtime, which is what this one says.
    """
    culprit = (getattr(exc, "name", "") or "").split(".")[0]
    if culprit in ("", "transformers"):
        return SystemExit(advice)
    return SystemExit(
        f"the teacher's import died inside {culprit}, not in transformers "
        f"({exc}).\n"
        f"This is not a transformers version and upgrading it will not help. "
        f"It is what a half-replaced package looks like: pip wrote a new "
        f"{culprit} to disk under a kernel that had already imported the old "
        f"one. Restart the runtime -- in Colab, Runtime > Restart session -- "
        f"and run the cells again.")


class Sam2Teacher(_ImageTeacher):
    """SAM 2.1's image predictor through `transformers`. The default.

    Ungated, Apache-2.0, and needs nothing newer than transformers 4.56 --
    the floor `requirements.txt` already documents.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-large",
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        import torch

        try:
            from transformers import Sam2Model, Sam2Processor
        except ImportError as exc:
            raise teacher_import_error(exc, (
                f"transformers has no Sam2 classes ({exc}).\n"
                f"They landed in transformers 4.56, which is the floor "
                f"requirements.txt documents -- `pip install -U "
                f"'transformers>=4.56'`.")) from exc

        self.model_id = model_id
        self.device = device
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(
            model_id, dtype=getattr(torch, dtype)
        ).to(device).eval()
        # Flipped to False the first time a batched call fails, so a
        # `transformers` version whose processor will not take a list degrades
        # to the per-crop path instead of taking the labelling run down with it.
        self._batched = True


class Sam3Teacher(_ImageTeacher):
    """SAM 3's promptable-visual-segmentation head, same call shape as SAM 2's.

    `Sam3TrackerModel` is SAM 3's answer to exactly this job -- point, box and
    mask prompts on a single image, one instance per prompt -- and its
    processor keeps SAM 2's API (`input_boxes`, `iou_scores`,
    `post_process_masks`), which is why this class is an `__init__` and nothing
    more. The published gaps over SAM 2.1 on image PVS favour it; whether they
    survive an aerial thermal crop is what the calibration cell measures,
    per dataset, before a harvest trusts it.

    The costs are the same three the video teacher documents: the repo is
    gated, the classes need `transformers>=5.0`, and the SAM licence carries
    use restrictions Apache-2.0 does not. `facebook/sam3.1` is **not** the
    upgrade it sounds like: it ships as a bare checkpoint with no transformers
    integration, and its headline feature (Object Multiplex) is throughput for
    many objects per frame -- this teacher prompts one box on one crop.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        import torch

        try:
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        except ImportError as exc:
            raise teacher_import_error(exc, (
                f"transformers has no Sam3Tracker classes ({exc}).\n"
                f"SAM 3 landed in transformers 5.0.0 -- `pip install -U "
                f"'transformers>=5'` -- or use the default SAM 2.1 teacher, "
                f"which needs nothing.")) from exc

        self.model_id = model_id
        self.device = device
        self.processor = Sam3TrackerProcessor.from_pretrained(model_id)
        self.model = Sam3TrackerModel.from_pretrained(
            model_id, dtype=getattr(torch, dtype)
        ).to(device).eval()
        self._batched = True


def teacher_class_for(model_id: str) -> type[_ImageTeacher]:
    """Dispatch on the id, exactly as `masklets.teacher_class_for` does.

    A separate flag could disagree with the checkpoint it describes; the id
    cannot.
    """
    return Sam3Teacher if "sam3" in model_id.lower() else Sam2Teacher


def build_image_teacher(model_id: str = "facebook/sam2.1-hiera-large",
                        device: str = "cuda",
                        dtype: str = "bfloat16") -> _ImageTeacher:
    try:
        return teacher_class_for(model_id)(model_id, device=device, dtype=dtype)
    except OSError as exc:
        raise SystemExit(
            f"could not load the image teacher {model_id!r}.\n\n"
            f"If it is a gated repository -- facebook/sam3 is -- open its "
            f"model page once, accept the terms, then set a token:\n"
            f"    from huggingface_hub import login; login()   # or export HF_TOKEN\n\n"
            f"An ungated teacher that needs no account: "
            f"--teacher facebook/sam2.1-hiera-large\n\n"
            f"original error: {exc}") from exc


def frames_to_label(
    sequence, stride: int = 1, max_frames: int | None = None
) -> np.ndarray:
    """Which of a sequence's annotated frames the teacher is asked about.

    `stride > 1` is the one honest way to trade labelling time for supervision:
    the skipped frames keep their `exist` flag and their box, so they still
    train the object-score head and still contribute a box-projection term --
    they just do not get a mask. Anti-UAV410 is 25 fps video of a drone that
    mostly drifts, so consecutive frames carry nearly the same mask anyway, and
    the teacher costs a full 1024x1024 encode per frame either way.
    """
    indices = sequence.labels.visible_indices()
    if stride > 1:
        indices = indices[::stride]
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def label_sequence(
    sequence,
    teacher: Sam2Teacher,
    out_dir: str | Path,
    gates: Gates = Gates(),
    zoom: float = 4.0,
    min_size: int = 128,
    frame_size: tuple[int, int] | None = None,
    stride: int = 1,
    max_frames: int | None = None,
    batch_size: int = 1,
) -> dict:
    """Run the teacher over one sequence and write its accepted masks.

    Returns the acceptance report: how many frames were attempted, how many
    survived the gates, and for the ones that did not, which gate stopped them.
    """
    from .antiuav import frame_shape, load_window

    out_dir = Path(out_dir)
    masks: dict[int, np.ndarray] = {}
    rejected: dict[str, int] = {}
    height, width = frame_size[::-1] if frame_size else frame_shape(sequence.frames[0])
    indices = frames_to_label(sequence, stride, max_frames)

    for start in range(0, len(indices), max(batch_size, 1)):
        chunk = indices[start:start + max(batch_size, 1)]
        crops, local_boxes, windows = [], [], []
        for index in chunk:
            box = sequence.labels.boxes[index]
            x0, y0, w, h = zoom_window(box, (width, height), zoom, min_size)
            crops.append(load_window(sequence.frames[int(index)], (x0, y0), (w, h), w))
            local_boxes.append(
                np.array([box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0])
            )
            windows.append((x0, y0, w, h))

        for index, (crop_mask, teacher_iou), local_box, (x0, y0, w, h) in zip(
            chunk, teacher.masks_for(crops, local_boxes), local_boxes, windows
        ):
            reason = reject_reason(measure(crop_mask, local_box, teacher_iou), gates)
            if reason is not None:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            full = np.zeros((height, width), dtype=bool)
            full[y0:y0 + h, x0:x0 + w] = crop_mask[:h, :w]
            masks[int(index)] = full

    attempted = len(indices)
    report = {
        "sequence": sequence.name,
        "frames": len(sequence),
        "visible": int(sequence.labels.exist.sum()),
        "attempted": attempted,
        "stride": stride,
        "accepted": len(masks),
        "acceptance_rate": (len(masks) / attempted) if attempted else 0.0,
        "rejected": rejected,
        "shape": [height, width],
    }
    (out_dir / sequence.name).mkdir(parents=True, exist_ok=True)
    save_masks(out_dir / sequence.name / MASK_STORE, (height, width), masks)
    (out_dir / sequence.name / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n")
    return report


def summarise(reports: list[dict]) -> str:
    """One markdown table over every sequence's acceptance report.

    The denominator is frames *attempted*, not frames annotated: with a
    labelling stride those differ, and dividing by the annotated count would
    report the stride as if it were the teacher failing.
    """
    attempted = sum(r.get("attempted", r["visible"]) for r in reports)
    accepted = sum(r["accepted"] for r in reports)
    reasons: dict[str, int] = {}
    for r in reports:
        for name, count in r["rejected"].items():
            reasons[name] = reasons.get(name, 0) + count

    lines = [
        f"{len(reports)} sequence(s), {attempted} frames attempted, "
        f"{accepted} accepted ({accepted / max(attempted, 1):.1%}).",
        "",
        "| gate that rejected | frames | share of rejects |",
        "|---|---:|---:|",
    ]
    total_rejected = max(attempted - accepted, 1)
    for name, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{name}` | {count} | {count / total_rejected:.1%} |")
    return "\n".join(lines)
