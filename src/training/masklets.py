"""Masklets for VTUAV: a video teacher turns tracking boxes into identity masks.

`labels.py`'s sibling, one axis over. That module prompts an image teacher with
one box on one frame; this one prompts a **video** teacher with one box on the
first frame of a chunk and lets it carry the mask -- and the identity -- through
the frames that follow. The output is stage C's missing ingredient: per-frame
masks that belong to *one* object across time, for sequences that ship only a
tracking box.

VTUAV is why this exists. Its VIS split has drawn masks on ~100 sequences and
those need nothing; the **other ~400 sequences** have a per-frame `x y w h` box
and ~1.7 M registered RGB-T frames. A video teacher run offline once converts
that into masklet supervision, which is two orders of magnitude more identity-
labelled video than the VIS split alone.

Three decisions carry the quality:

**Chunked propagation, re-anchored on the annotation.** Propagating one mask
through a 3 400-frame flight accumulates drift and holds every frame in memory.
Instead the visible frames are cut into chunks, and each chunk is prompted
fresh with *that frame's* ground-truth box. Drift is bounded by the chunk
length, memory is bounded by it too, and the annotation -- which exists on
every frame -- is used on every chunk boundary instead of once per video.

**Every frame is gated against its own box.** VTUAV annotates every frame, so
unlike the usual masklet mining there is a per-frame reference: a propagated
mask that no longer sits on the annotated box is drift, and it is dropped
rather than stored. The gates are `labels.Gates` minus the teacher-IoU term --
video propagation reports no per-frame confidence, so that gate never fires.

**The store is the one `labels.py` already writes.** Accepted masks go into the
same run-length `.npz` (`MASK_STORE`), so whatever reads pseudo-masks today
reads masklets tomorrow; a frame the gates dropped is *absent*, which readers
already treat as "no mask supervision here", not as an empty mask.

The teacher runs through `transformers`, never through Meta's `sam2` package --
EdgeTAM installs itself *as* `sam2`, so the two cannot share an environment.
`facebook/sam2.1-hiera-large` is the default: ungated, Apache-2.0, and video
propagation is its native job. `facebook/sam3` is one string away and is
dispatched to its tracker classes; it is gated, needs `transformers>=5`, and
its licence excludes some uses the Apache one does not -- worth reading before
a deliverable depends on it. Published gaps (SA-V +5.6, LVOSv2 +8.9 J&F over
SAM 2.1) say it earns that friction on long or hard videos, not on short easy
ones.

And the number that decides is measured, not assumed: the VIS split's drawn
masks are held out as a **calibration set**. Run `tools/make_masklets.py
--calibrate` on sequences that have them, read the masklet-versus-drawn IoU,
and only then spend teacher-hours on the 400 unlabelled sequences.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence as SequenceABC
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .labels import MASK_STORE, Gates, measure, reject_reason, save_masks

REPORT_FILE = "masklets.json"


# --------------------------------------------------------------------------
# What a VTUAV sequence looks like on disk
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoSequence:
    """One flight: frames per modality, one box per frame, GT masks if drawn.

    `boxes` is `[N, 4]` xyxy in frame coordinates and `exist[i]` says whether
    frame `i` has a target at all -- VTUAV's long-term sequences contain real
    disappearances, and propagating a mask through one would invent an object.
    """

    name: str
    root: Path
    frames: dict[str, tuple[Path, ...]]        # modality -> ordered frame files
    boxes: np.ndarray
    exist: np.ndarray
    gt_masks: dict[str, dict[int, Path]] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.exist.shape[0])


def read_boxes(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """`(boxes_xyxy, exist)` from a VTUAV `rgb.txt` of per-frame `x y w h`.

    Fields split on whitespace or commas -- both appear in the wild. A frame
    whose box has no positive extent, or carries a NaN, is an *absent* frame:
    the target left the view, and its row is kept (so indices still line up
    with the frame files) but flagged so nothing propagates through it.
    """
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.replace(",", " ").split()
        if not parts:
            continue
        rows.append([float(v) for v in parts[:4]])
    if not rows:
        raise ValueError(f"{path}: no boxes in the file.")
    xywh = np.asarray(rows, dtype=np.float64)
    exist = np.isfinite(xywh).all(axis=1) & (xywh[:, 2] > 0) & (xywh[:, 3] > 0)
    boxes = np.zeros_like(xywh)
    boxes[:, 0] = xywh[:, 0]
    boxes[:, 1] = xywh[:, 1]
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2]
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3]
    boxes[~exist] = np.nan
    return boxes, exist


def find_sequences(root: str | Path,
                   names: SequenceABC[str] | None = None) -> list[VideoSequence]:
    """Every VTUAV-shaped sequence under `root`, by its `rgb.txt`.

    The box file is the anchor because it is the one thing a sequence cannot
    be used without here. Frame files are matched to box rows by *sorted
    order*, which is how the VTUAV toolkit reads them; a count mismatch is
    tolerated to the shorter of the two, loudly, because half-extracted
    archives are a fact of Colab life.
    """
    root = Path(root)
    sequences = []
    for box_file in sorted(root.rglob("rgb.txt")):
        seq_dir = box_file.parent
        if names is not None and seq_dir.name not in names:
            continue
        frames = {}
        for modality in ("rgb", "ir"):
            folder = seq_dir / modality
            if folder.is_dir():
                frames[modality] = tuple(sorted(
                    p for p in folder.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")))
        if not frames.get("rgb"):
            continue
        boxes, exist = read_boxes(box_file)
        count = min(len(boxes), *(len(f) for f in frames.values()))
        if count < len(boxes):
            print(f"  {seq_dir.name}: {len(boxes)} boxes but "
                  f"{min(len(f) for f in frames.values())} frames -- "
                  f"using the first {count} of both")
        gt = {}
        for modality in frames:
            mask_dir = seq_dir / "mask" / modality
            if mask_dir.is_dir():
                gt[modality] = {int(p.stem): p for p in sorted(mask_dir.glob("*.png"))}
        sequences.append(VideoSequence(
            name=seq_dir.name, root=seq_dir,
            frames={m: f[:count] for m, f in frames.items()},
            boxes=boxes[:count], exist=exist[:count], gt_masks=gt))
    return sequences


def visible_runs(exist: np.ndarray, chunk: int,
                 max_frames: int | None = None) -> list[list[int]]:
    """Contiguous runs of visible frames, cut into chunks of at most `chunk`.

    A run breaks wherever the target is absent -- propagating across a
    disappearance would hand the teacher a first frame whose box describes
    nothing -- and again every `chunk` frames, which is both the memory bound
    and the drift bound (each chunk is re-prompted with its first frame's own
    ground-truth box).
    """
    runs: list[list[int]] = []
    current: list[int] = []
    budget = len(exist) if max_frames is None else max_frames
    taken = 0
    for index, visible in enumerate(exist):
        if taken >= budget:
            break
        if not visible:
            if current:
                runs.append(current)
                current = []
            continue
        current.append(index)
        taken += 1
        if len(current) >= max(chunk, 1):
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


# --------------------------------------------------------------------------
# Teachers
# --------------------------------------------------------------------------


class _VideoTeacher:
    """One box on the first frame in; one mask per frame out.

    Everything version-sensitive about `transformers` lives in the two methods
    below; everything this module reasons about -- chunking, gates, transfer,
    storage, calibration -- is independent of it and unit-tested with a fake.
    Subclasses set `self.processor` and `self.model` and nothing else.
    """

    model_id: str
    device: str

    def propagate(self, frames: SequenceABC[np.ndarray],
                  box: np.ndarray) -> list[np.ndarray]:
        """Boolean HxW masks, one per frame, prompted by `box` on frame 0."""
        import torch

        height, width = frames[0].shape[:2]
        session = self.processor.init_video_session(
            video=list(frames), inference_device=self.device)
        self.processor.add_inputs_to_inference_session(
            inference_session=session, frame_idx=0, obj_ids=1,
            input_boxes=[[[float(v) for v in box]]])

        out: list[np.ndarray | None] = [None] * len(frames)
        with torch.no_grad():
            for step in self.model.propagate_in_video_iterator(session):
                masks = self.processor.post_process_masks(
                    [step.pred_masks], original_sizes=[[height, width]])[0]
                if hasattr(masks, "cpu"):
                    masks = masks.cpu().numpy()
                out[int(step.frame_idx)] = (
                    np.asarray(masks).reshape(-1, height, width)[0] > 0)
        empty = np.zeros((height, width), dtype=bool)
        return [m if m is not None else empty for m in out]


class Sam2VideoTeacher(_VideoTeacher):
    """SAM 2.1's video predictor through `transformers`. The default.

    Ungated, Apache-2.0, 39.5 FPS on an A100 at 224 M parameters -- and video
    propagation from a box prompt is precisely the job it was trained for.
    """

    def __init__(self, model_id: str = "facebook/sam2.1-hiera-large",
                 device: str = "cuda", dtype: str = "bfloat16") -> None:
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        self.model_id = model_id
        self.device = device
        self.processor = Sam2VideoProcessor.from_pretrained(model_id)
        self.model = Sam2VideoModel.from_pretrained(
            model_id, dtype=getattr(torch, dtype)).to(device).eval()


class Sam3VideoTeacher(_VideoTeacher):
    """SAM 3's tracker through `transformers`, same call shape as SAM 2's.

    What it buys over SAM 2.1 is documented on the hard end of video: **SA-V
    +5.6 and LVOSv2 +8.9 J&F**, both of which SAM 3's own dataset-usage table
    marks Test-only, so the comparison is like for like.

    **The MOSEv2 +12.4 that used to be quoted here is not.** MOSEv2 sits in
    SAM 3's SA-Co/VIDEO-EXT training pool, while the SAM 2.1 number it is
    measured against is marked zero-shot in the same table. That is an
    in-domain model against a zero-shot one, and citing it overstates the case
    for switching. It also means MOSEv2 stops being a clean held-out set the
    moment SAM 3 becomes the teacher -- SAM 2.1 keeps that evaluation honest.

    The costs are three: the repo is gated, the classes need
    `transformers>=5.0`, and the SAM licence carries use restrictions
    Apache-2.0 does not. Calibrate both on the VIS split's drawn masks before
    deciding the friction is paid for.
    """

    def __init__(self, model_id: str = "facebook/sam3",
                 device: str = "cuda", dtype: str = "bfloat16") -> None:
        import torch

        try:
            from transformers import (Sam3TrackerVideoModel,
                                      Sam3TrackerVideoProcessor)
        except ImportError as exc:
            raise SystemExit(
                f"transformers has no Sam3TrackerVideo classes ({exc}).\n"
                f"SAM 3 landed in transformers 5.0.0 -- `pip install -U "
                f"'transformers>=5'` -- or use the default SAM 2.1 teacher, "
                f"which needs nothing.") from exc

        self.model_id = model_id
        self.device = device
        self.processor = Sam3TrackerVideoProcessor.from_pretrained(model_id)
        self.model = Sam3TrackerVideoModel.from_pretrained(
            model_id, dtype=getattr(torch, dtype)).to(device).eval()


def teacher_class_for(model_id: str) -> type[_VideoTeacher]:
    """Dispatch on the id, exactly as `distill.build_teacher` does.

    A separate flag could disagree with the checkpoint it describes; the id
    cannot.
    """
    return Sam3VideoTeacher if "sam3" in model_id.lower() else Sam2VideoTeacher


def build_video_teacher(model_id: str = "facebook/sam2.1-hiera-large",
                        device: str = "cuda",
                        dtype: str = "bfloat16") -> _VideoTeacher:
    try:
        return teacher_class_for(model_id)(model_id, device=device, dtype=dtype)
    except OSError as exc:
        raise SystemExit(
            f"could not load the video teacher {model_id!r}.\n\n"
            f"If it is a gated repository -- facebook/sam3 is -- open its "
            f"model page once, accept the terms, then set a token:\n"
            f"    from huggingface_hub import login; login()   # or export HF_TOKEN\n\n"
            f"An ungated teacher that needs no account: "
            f"--teacher facebook/sam2.1-hiera-large\n\n"
            f"original error: {exc}") from exc


# --------------------------------------------------------------------------
# The labelling pass
# --------------------------------------------------------------------------


def _read_frame(path: Path) -> np.ndarray:
    """HxWx3 uint8 RGB, whatever the file held -- the teachers want RGB."""
    import cv2

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Could not read frame: {path}")
    if raw.ndim == 2:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(raw[..., :3], cv2.COLOR_BGR2RGB)


def masklet_sequence(
    sequence: VideoSequence,
    teacher: _VideoTeacher,
    out_dir: str | Path,
    gates: Gates = Gates(),
    prompt_frames: str = "rgb",
    chunk: int = 200,
    max_frames: int | None = None,
    progress=None,
) -> dict:
    """Run the teacher over one sequence; write the store and its report.

    `prompt_frames` is which modality the teacher *looks at*. `rgb` is the
    default -- the teachers were trained on RGB, and VTUAV's halves are
    registered, so a mask drawn on the RGB frame lands on the thermal one (to
    within the registration the dataset itself offers; the same caveat stage A
    carries). `ir` runs the teacher directly on thermal instead -- a grayscale
    image it was never trained on -- and exists so the calibration split can
    *measure* which of the two survives the modality better, rather than
    anyone asserting it.

    Frames whose propagated mask fails a gate are counted and **absent** from
    the store -- readers treat that as "no mask supervision here". The
    `teacher_iou` gate never fires (propagation reports no per-frame
    confidence); `box_iou` is the load-bearing one, because VTUAV's per-frame
    boxes give every propagated frame its own reference, which ordinary
    masklet mining does not have.
    """
    frames = sequence.frames.get(prompt_frames)
    if not frames:
        raise ValueError(f"{sequence.name} has no {prompt_frames!r} frames.")

    masks: dict[int, np.ndarray] = {}
    rejected: dict[str, int] = {}
    runs = visible_runs(sequence.exist, chunk, max_frames)
    iterator = progress(runs, total=len(runs), desc=sequence.name) if progress else runs

    shape: tuple[int, int] | None = None
    for run in iterator:
        pixels = [_read_frame(frames[i]) for i in run]
        shape = pixels[0].shape[:2]
        propagated = teacher.propagate(pixels, sequence.boxes[run[0]])
        for index, mask in zip(run, propagated):
            # teacher_iou is pinned to 1.0: propagation reports no per-frame
            # score, so only the geometric gates decide.
            verdict = reject_reason(
                measure(mask, sequence.boxes[index], teacher_iou=1.0), gates)
            if verdict is not None:
                rejected[verdict] = rejected.get(verdict, 0) + 1
                continue
            masks[int(index)] = mask

    if shape is None:
        shape = _read_frame(frames[0]).shape[:2]
    attempted = sum(len(run) for run in runs)
    report = {
        "sequence": sequence.name,
        "frames": len(sequence),
        "visible": int(sequence.exist.sum()),
        "attempted": attempted,
        "accepted": len(masks),
        "acceptance_rate": (len(masks) / attempted) if attempted else 0.0,
        "rejected": rejected,
        "chunk": chunk,
        "prompt_frames": prompt_frames,
        "teacher": getattr(teacher, "model_id", type(teacher).__name__),
        "shape": [int(shape[0]), int(shape[1])],
    }

    out = Path(out_dir) / sequence.name
    out.mkdir(parents=True, exist_ok=True)
    save_masks(out / MASK_STORE, (int(shape[0]), int(shape[1])), masks)
    (out / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_store(sequence_dir: str | Path):
    """The masklet store `masklet_sequence` wrote for one sequence.

    A `labels.MaskStore` -- decoded per frame on demand -- because that is the
    same object every other pseudo-mask reader holds, and because calibration
    over a long sequence has no reason to decode frames nobody annotated.
    """
    from .labels import open_masks

    return open_masks(Path(sequence_dir) / MASK_STORE)


# --------------------------------------------------------------------------
# Calibration -- the number that decides whether to trust any of this
# --------------------------------------------------------------------------


def calibrate(sequence: VideoSequence, masks: Mapping[int, np.ndarray],
              modality: str = "ir") -> dict | None:
    """Masklet-versus-drawn IoU on the frames somebody actually annotated.

    Two means, deliberately: `iou_accepted` scores only the frames the gates
    kept and says how good a *kept* mask is; `iou_all` scores every annotated
    visible frame, counting a gated-out frame as an empty prediction, and says
    how much supervision survives end to end. A pipeline can have a flattering
    first number and a useless second one; stage C lives on the second.
    """
    import cv2

    drawn = sequence.gt_masks.get(modality)
    if not drawn:
        return None

    accepted, everything = [], []
    for frame, path in sorted(drawn.items()):
        if frame >= len(sequence) or not sequence.exist[frame]:
            continue
        gt = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if gt is None:
            continue
        gt = (gt if gt.ndim == 2 else gt[..., 0]) > 0
        predicted = masks.get(frame)

        def iou(a, b):
            union = np.logical_or(a, b).sum()
            return float(np.logical_and(a, b).sum() / union) if union else 1.0

        everything.append(iou(predicted if predicted is not None
                              else np.zeros_like(gt), gt))
        if predicted is not None:
            accepted.append(iou(predicted, gt))

    if not everything:
        return None
    return {
        "modality": modality,
        "drawn_frames": len(everything),
        "covered": len(accepted),
        "iou_accepted": float(np.mean(accepted)) if accepted else float("nan"),
        "iou_all": float(np.mean(everything)),
    }


def summarise_masklets(reports: SequenceABC[dict]) -> str:
    """One markdown block over every sequence's report, calibration included."""
    attempted = sum(r["attempted"] for r in reports)
    accepted = sum(r["accepted"] for r in reports)
    reasons: dict[str, int] = {}
    for report in reports:
        for name, count in report["rejected"].items():
            reasons[name] = reasons.get(name, 0) + count

    lines = [
        f"{len(reports)} sequence(s), {attempted} frames attempted, "
        f"{accepted} masklet frames accepted ({accepted / max(attempted, 1):.1%}).",
        "",
        "| sequence | attempted | accepted | calibration IoU (all / accepted) |",
        "|---|---:|---:|---|",
    ]
    for report in reports:
        cal = report.get("calibration")
        note = (f"{cal['iou_all']:.3f} / {cal['iou_accepted']:.3f}"
                if cal else "—")
        lines.append(f"| {report['sequence']} | {report['attempted']} | "
                     f"{report['accepted']} | {note} |")
    if reasons:
        lines += ["", "| gate that rejected | frames |", "|---|---:|"]
        for name, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{name}` | {count} |")
    return "\n".join(lines)
