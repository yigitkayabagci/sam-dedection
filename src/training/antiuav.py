"""Anti-UAV410 as short clips at the model's input size.

Anti-UAV410 is 410 thermal-infrared sequences of drones in the wild, 438K
hand-annotated boxes, split into train/val/test. Each sequence is a folder of
frames plus one `IR_label.json`:

    {"exist": [1, 1, 0, ...], "gt_rect": [[x, y, w, h], [], ...]}

Two things about it decide the whole design here:

* **Frames are 640x512 8-bit mono.** A 512x512 window is therefore *native
  pixels with no resize at all* -- the same input the deployment's `crop512`
  mode feeds the model (`tools/run_records.py`). Training on a resized 512 view
  and deploying on a cropped one would train the wrong thing.
* **`exist` is per frame.** That is direct supervision for
  `object_score_logits`, which is the signal behind the failure this project
  actually hit: one hard frame drives the score below zero, `no_obj_ptr` goes
  into the memory bank, and the next seven frames read it back.

Everything in this module is pure Python + numpy so the annotation parsing and
the crop geometry stay testable with no OpenCV, no EdgeTAM and no GPU. The two
functions that touch pixels (`load_window`, `clip_tensor`) import cv2 and torch
themselves.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence as SequenceABC

import numpy as np

# Anti-UAV410 ships JPEGs; the others are accepted so a re-encoded or
# locally-extracted copy of a sequence works without a flag.
FRAME_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
LABEL_FILE = "IR_label.json"

# ImageNet statistics, matching `load_video_frames`' normalisation. Training
# has to use the same ones the tracker uses at inference (src/pipeline.py).
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------
# Annotations
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SequenceLabels:
    """One sequence's per-frame ground truth.

    `boxes` is xyxy float32 and is only meaningful where `exist` is True; absent
    frames hold NaN rather than zeros, so a bug that ignores `exist` produces
    NaN instead of a plausible-looking box at the origin.
    """

    exist: np.ndarray  # (T,) bool
    boxes: np.ndarray  # (T, 4) float32, xyxy

    def __len__(self) -> int:
        return int(self.exist.shape[0])

    def visible_indices(self) -> np.ndarray:
        return np.flatnonzero(self.exist)


def load_labels(path: str | Path) -> SequenceLabels:
    """Parse an `IR_label.json`.

    `gt_rect` is xywh, and rows for absent frames are written inconsistently
    across the dataset -- `[]`, `[0, 0, 0, 0]`, or occasionally a stale box. All
    three are treated the same way: absent means absent, and the box is dropped.
    """
    raw = json.loads(Path(path).read_text())
    exist = np.asarray(raw["exist"], dtype=np.int64).astype(bool)
    rects = raw["gt_rect"]
    if len(rects) != exist.shape[0]:
        raise ValueError(
            f"{path}: 'exist' has {exist.shape[0]} entries but 'gt_rect' has "
            f"{len(rects)}."
        )

    boxes = np.full((exist.shape[0], 4), np.nan, dtype=np.float32)
    for i, rect in enumerate(rects):
        if not exist[i] or rect is None or len(rect) != 4:
            exist[i] = False
            continue
        x, y, w, h = (float(v) for v in rect)
        if w <= 0 or h <= 0:
            exist[i] = False
            continue
        boxes[i] = (x, y, x + w, y + h)
    return SequenceLabels(exist=exist, boxes=boxes)


def _frame_sort_key(path: Path):
    """Numeric order when the stems are numbers, lexical otherwise.

    Anti-UAV410 names frames `0.jpg, 1.jpg, ... 10.jpg`, which a lexical sort
    puts in the order 0, 1, 10, 2 -- silently shuffling the video.
    """
    stem = path.stem
    return (0, int(stem), "") if stem.isdigit() else (1, 0, stem)


@dataclass(frozen=True)
class Sequence:
    name: str
    split: str
    frames: tuple[Path, ...]
    labels: SequenceLabels

    def __len__(self) -> int:
        return len(self.frames)


def _load_sequence(folder: Path, split: str) -> Sequence:
    frames = sorted(
        (p for p in folder.iterdir()
         if p.is_file() and p.suffix.lower() in FRAME_SUFFIXES),
        key=_frame_sort_key,
    )
    if not frames:
        raise FileNotFoundError(f"{folder}: no frame images found.")
    labels = load_labels(folder / LABEL_FILE)
    # Annotations occasionally run one frame past the images (or the reverse);
    # truncating to the shorter is the only reading that keeps them aligned.
    n = min(len(frames), len(labels))
    return Sequence(
        name=folder.name,
        split=split,
        frames=tuple(frames[:n]),
        labels=SequenceLabels(exist=labels.exist[:n], boxes=labels.boxes[:n]),
    )


def list_sequences(root: str | Path, split: str = "train") -> list[Sequence]:
    """Load every sequence under `<root>/<split>/`, sorted by name.

    A `root` that is itself a split directory (or a single sequence folder) is
    accepted too, so one recording can be pointed at without restructuring it.
    """
    root = Path(root)
    if (root / LABEL_FILE).is_file():
        return [_load_sequence(root, split)]
    base = root / split if (root / split).is_dir() else root
    folders = sorted(d for d in base.iterdir() if d.is_dir() and (d / LABEL_FILE).is_file())
    if not folders:
        raise FileNotFoundError(f"No sequences with {LABEL_FILE} under {base}")
    return [_load_sequence(d, split) for d in folders]


# --------------------------------------------------------------------------
# Crop geometry
# --------------------------------------------------------------------------


def crop_window(
    boxes: np.ndarray,
    frame_size: tuple[int, int],
    size: int,
    jitter: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[int, int] | None:
    """Top-left of a `size`x`size` window holding every box, or None.

    None means the target's excursion over these frames is wider than the
    window: no fixed crop contains it, and the caller has to fall back to
    resizing the whole frame. That case is a real property of the clip, not an
    error -- fast, close-range passes do leave the window.

    The window is *fixed for the whole clip* on purpose. SAM 2's memory bank
    stores features in input coordinates, so a window that moved between frames
    would put every stored memory in a different coordinate frame from the one
    reading it. The same constraint governs the adaptive-ROI work.
    """
    width, height = frame_size
    boxes = np.asarray(boxes, dtype=np.float32)
    boxes = boxes[~np.isnan(boxes).any(axis=1)]
    if boxes.size == 0:
        return None

    x0, y0 = boxes[:, 0].min(), boxes[:, 1].min()
    x1, y1 = boxes[:, 2].max(), boxes[:, 3].max()
    if x1 - x0 > size or y1 - y0 > size:
        return None

    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    left = cx - size / 2.0
    top = cy - size / 2.0
    if jitter and rng is not None:
        left += rng.integers(-jitter, jitter + 1)
        top += rng.integers(-jitter, jitter + 1)

    # Clamp so the window stays inside the frame, then re-clamp against the
    # union box: shifting the window must never push the target out of it.
    left = float(np.clip(left, 0, max(width - size, 0)))
    top = float(np.clip(top, 0, max(height - size, 0)))
    left = float(np.clip(left, x1 - size, x0))
    top = float(np.clip(top, y1 - size, y0))
    left = float(np.clip(left, 0, max(width - size, 0)))
    top = float(np.clip(top, 0, max(height - size, 0)))
    return int(round(left)), int(round(top))


def map_boxes(
    boxes: np.ndarray,
    origin: tuple[int, int],
    window: tuple[int, int],
    size: int,
) -> np.ndarray:
    """Frame coordinates -> the model's input coordinates.

    The two axes scale independently: a 640x512 frame squeezed into a square
    input is compressed horizontally and left alone vertically, and a box that
    was scaled by one factor on both axes would not sit on its target.
    """
    out = np.asarray(boxes, dtype=np.float32).copy()
    out[:, 0::2] = (out[:, 0::2] - origin[0]) * (size / window[0])
    out[:, 1::2] = (out[:, 1::2] - origin[1]) * (size / window[1])
    return out


# --------------------------------------------------------------------------
# Clips
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Clip:
    """A window onto `length` frames of one sequence, ready to be loaded.

    `origin` + `window` is the source rectangle; `size` is what it is resized
    to. `window == (size, size)` is the native-pixel case a 512 crop of a
    640x512 thermal frame gives -- the deployment's `crop512` mode, no resize
    at all. `window == (640, 512)` is its `full512` mode.
    """

    sequence: Sequence
    indices: tuple[int, ...]
    origin: tuple[int, int]
    window: tuple[int, int]
    size: int

    @property
    def frames(self) -> tuple[Path, ...]:
        return tuple(self.sequence.frames[i] for i in self.indices)

    @property
    def exist(self) -> np.ndarray:
        return self.sequence.labels.exist[list(self.indices)]

    @property
    def boxes(self) -> np.ndarray:
        """Ground-truth boxes in model-input coordinates."""
        raw = self.sequence.labels.boxes[list(self.indices)]
        return map_boxes(raw, self.origin, self.window, self.size)

    @property
    def native(self) -> bool:
        """True when no resampling happens: source pixels reach the model."""
        return self.window == (self.size, self.size)


def _window_for(
    sequence: Sequence,
    indices: SequenceABC[int],
    size: int,
    frame_size: tuple[int, int],
    jitter: int,
    rng: np.random.Generator | None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """(origin, window) for one clip: a native crop if one fits, else the whole
    frame -- the same two input modes the deployment already offers."""
    boxes = sequence.labels.boxes[list(indices)]
    origin = crop_window(boxes, frame_size, size, jitter, rng)
    if origin is not None:
        return origin, (size, size)
    return (0, 0), frame_size


def sample_clips(
    sequences: SequenceABC[Sequence],
    length: int = 8,
    stride: int = 1,
    size: int = 512,
    frame_size: tuple[int, int] = (640, 512),
    min_visible: int = 2,
    jitter: int = 0,
    seed: int | None = None,
) -> list[Clip]:
    """Every clip of `length` frames at `stride`, one per starting position.

    A clip is kept only if its first frame is annotated (it carries the prompt)
    and at least `min_visible` frames are, so the loop is never asked to learn
    from a clip with nothing in it.
    """
    rng = np.random.default_rng(seed)
    span = (length - 1) * stride + 1
    clips: list[Clip] = []
    for sequence in sequences:
        exist = sequence.labels.exist
        for start in range(0, len(sequence) - span + 1):
            indices = tuple(range(start, start + span, stride))
            if not exist[indices[0]] or int(exist[list(indices)].sum()) < min_visible:
                continue
            origin, window = _window_for(sequence, indices, size, frame_size, jitter, rng)
            clips.append(Clip(sequence, indices, origin, window, size))
    return clips


def iter_clips(clips: SequenceABC[Clip], seed: int | None = None) -> Iterator[Clip]:
    """Shuffle clips reproducibly. A plain generator so a caller can wrap it."""
    order = np.random.default_rng(seed).permutation(len(clips))
    for i in order:
        yield clips[int(i)]


# --------------------------------------------------------------------------
# Pixels
# --------------------------------------------------------------------------


def frame_shape(path: str | Path) -> tuple[int, int]:
    """`(height, width)` of one frame, without decoding a whole sequence."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read frame: {path}")
    return int(img.shape[0]), int(img.shape[1])


def load_window(
    path: str | Path,
    origin: tuple[int, int],
    window: tuple[int, int],
    size: int,
) -> np.ndarray:
    """One frame's window as HxWx3 uint8 at the model's input size.

    Thermal frames are single channel; the checkpoint's stem takes three. The
    channel is replicated rather than the stem rewritten -- a 1-channel stem
    would change the ONNX graph and every engine to save a rounding error of a
    38.7 GFLOP encoder.
    """
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read frame: {path}")
    if img.ndim == 3:
        img = img[..., 0] if img.shape[2] == 1 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    x0, y0 = origin
    img = img[y0:y0 + window[1], x0:x0 + window[0]]
    if img.shape[:2] != (size, size):
        # INTER_AREA averages over discarded pixels on a downscale; it
        # degenerates to bilinear upwards, so pick by direction. Same choice as
        # the inference preprocessing in src/pipeline.py.
        shrinking = size < img.shape[0] or size < img.shape[1]
        img = cv2.resize(img, (size, size),
                         interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)
    return np.repeat(img[:, :, None], 3, axis=2)


def clip_tensor(clip: Clip, device: str = "cpu"):
    """A clip's frames as a normalised [T, 3, size, size] float32 tensor."""
    import torch

    frames = np.stack([
        load_window(p, clip.origin, clip.window, clip.size) for p in clip.frames
    ])
    out = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.as_tensor(MEAN, device=out.device)[:, None, None]
    std = torch.as_tensor(STD, device=out.device)[:, None, None]
    return out.sub_(mean).div_(std)


def clip_masks(clip: Clip, masks: dict[int, np.ndarray]) -> list[np.ndarray | None]:
    """A clip's pseudo-masks, moved into the same window as its frames.

    `None` for a frame the teacher could not label -- the training loop reads
    that as "no mask supervision here" and falls back to the box term, which is
    a different thing from an empty mask (which would mean "nothing is here").
    """
    import cv2

    x0, y0 = clip.origin
    width, height = clip.window
    out: list[np.ndarray | None] = []
    for index in clip.indices:
        mask = masks.get(int(index))
        if mask is None:
            out.append(None)
            continue
        window = mask[y0:y0 + height, x0:x0 + width].astype(np.uint8)
        if window.shape[:2] != (clip.size, clip.size):
            # Nearest neighbour: a drone can be a handful of pixels, and any
            # interpolation of a boolean mask at that size invents or erases
            # them.
            window = cv2.resize(window, (clip.size, clip.size),
                                interpolation=cv2.INTER_NEAREST)
        out.append(window.astype(bool))
    return out
