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


def load_masks(path: str | Path) -> tuple[tuple[int, int], dict[int, np.ndarray]]:
    """`(shape, {frame_idx: mask})`. Absent frames are simply absent from the
    dict -- the training loop reads that as "no mask supervision here"."""
    with np.load(path) as data:
        shape = tuple(int(v) for v in data["shape"])
        runs, offsets, frames = data["runs"], data["offsets"], data["frames"]
        masks = {
            int(frame): rle_decode(runs[offsets[i]:offsets[i + 1]], shape)
            for i, frame in enumerate(frames)
        }
    return shape, masks


# --------------------------------------------------------------------------
# Teacher
# --------------------------------------------------------------------------


class Sam2Teacher:
    """SAM 2.1 through `transformers`, prompted with one box on one crop.

    Deliberately thin: everything version-sensitive about the `transformers`
    API is in `mask_for`, and everything this module actually reasons about --
    geometry, gates, storage -- is independent of it and unit-tested.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-large",
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.device = device
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(
            model_id, dtype=getattr(torch, dtype)
        ).to(device).eval()

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


def label_sequence(
    sequence,
    teacher: Sam2Teacher,
    out_dir: str | Path,
    gates: Gates = Gates(),
    zoom: float = 4.0,
    min_size: int = 128,
    frame_size: tuple[int, int] | None = None,
) -> dict:
    """Run the teacher over one sequence and write its accepted masks.

    Returns the acceptance report: how many frames were labelled, and for the
    ones that were not, which gate stopped them.
    """
    from .antiuav import frame_shape, load_window

    out_dir = Path(out_dir)
    masks: dict[int, np.ndarray] = {}
    rejected: dict[str, int] = {}
    height, width = frame_size[::-1] if frame_size else frame_shape(sequence.frames[0])

    for index in sequence.labels.visible_indices():
        box = sequence.labels.boxes[index]
        x0, y0, w, h = zoom_window(box, (width, height), zoom, min_size)
        crop = load_window(sequence.frames[int(index)], (x0, y0), (w, h), w)
        local_box = np.array([box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0])

        crop_mask, teacher_iou = teacher.mask_for(crop, local_box)
        reason = reject_reason(measure(crop_mask, local_box, teacher_iou), gates)
        if reason is not None:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue

        full = np.zeros((height, width), dtype=bool)
        full[y0:y0 + h, x0:x0 + w] = crop_mask[:h, :w]
        masks[int(index)] = full

    visible = int(sequence.labels.exist.sum())
    report = {
        "sequence": sequence.name,
        "frames": len(sequence),
        "visible": visible,
        "accepted": len(masks),
        "acceptance_rate": (len(masks) / visible) if visible else 0.0,
        "rejected": rejected,
        "shape": [height, width],
    }
    (out_dir / sequence.name).mkdir(parents=True, exist_ok=True)
    save_masks(out_dir / sequence.name / MASK_STORE, (height, width), masks)
    (out_dir / sequence.name / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n")
    return report


def summarise(reports: list[dict]) -> str:
    """One markdown table over every sequence's acceptance report."""
    visible = sum(r["visible"] for r in reports)
    accepted = sum(r["accepted"] for r in reports)
    reasons: dict[str, int] = {}
    for r in reports:
        for name, count in r["rejected"].items():
            reasons[name] = reasons.get(name, 0) + count

    lines = [
        f"{len(reports)} sequence(s), {visible} annotated frames, "
        f"{accepted} accepted ({accepted / max(visible, 1):.1%}).",
        "",
        "| gate that rejected | frames | share of rejects |",
        "|---|---:|---:|",
    ]
    total_rejected = max(visible - accepted, 1)
    for name, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{name}` | {count} | {count / total_rejected:.1%} |")
    return "\n".join(lines)
