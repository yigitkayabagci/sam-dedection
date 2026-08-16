"""How good the tracking is, as opposed to `src/metrics.py`, which is how fast.

Three things are measured here, and the third is the one this project actually
needs:

`state_accuracy`   the Anti-UAV benchmark's own metric. Rewards overlap while
                   the target is visible *and* rewards correctly reporting that
                   it is gone -- so a tracker cannot buy a score by always
                   emitting a box.
`success_auc`      the area under the standard success plot, for comparison
                   against published numbers on Anti-UAV410.
`dropout_episodes` how often the tracker loses a visible target, and for how
                   long each time. Mean IoU hides exactly the failure this
                   pipeline hit: one hard frame drives `object_score_logits`
                   below zero, `no_obj_ptr` is written into the memory bank,
                   and the next frames read it back. That is not a small drop
                   in mean overlap; it is a small number of very long episodes.
                   Averages cannot tell those two apart. Episode lengths can.

Absent is spelled NaN throughout -- for the ground truth (the target is out of
view) and for a prediction (the tracker reports nothing). Zeros would be
indistinguishable from a box at the origin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _present(boxes: np.ndarray) -> np.ndarray:
    return ~np.isnan(np.asarray(boxes, dtype=np.float64)).any(axis=1)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise IoU of two (N, 4) xyxy arrays. NaN rows give 0."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    lo = np.maximum(a[:, :2], b[:, :2])
    hi = np.minimum(a[:, 2:], b[:, 2:])
    inter = np.prod(np.clip(hi - lo, 0, None), axis=1)
    area = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1) + \
        np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1) - inter
    return np.where(area > 0, inter / np.maximum(area, 1e-9), 0.0)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boolean masks. Two empty masks agree perfectly, not
    undefined -- both said 'nothing here' and both were right."""
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def box_from_mask(mask: np.ndarray) -> np.ndarray:
    """Tight xyxy box around a boolean mask, or NaN when it is empty.

    Also used on the tracking path to turn candidate masks into something a
    motion model can score (`src/trackers/samurai.py`).
    """
    rows = np.any(mask, axis=-1)
    cols = np.any(mask, axis=-2)
    if not rows.any():
        return np.full(4, np.nan, dtype=np.float32)
    y = np.flatnonzero(rows)
    x = np.flatnonzero(cols)
    # +1 on the far edge: a one-pixel mask spans one pixel, not zero.
    return np.array([x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=np.float32)


def per_frame_iou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """IoU per frame, 0 where either side is absent and the other is not."""
    pred, gt = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    both = _present(pred) & _present(gt)
    out = np.zeros(len(gt))
    if both.any():
        out[both] = box_iou(pred[both], gt[both])
    return out


def state_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    """The Anti-UAV metric.

        SA = mean_t [ IoU_t * v_t + p_t * (1 - v_t) ]

    `v_t` is whether the target is really there; `p_t` is 1 when the target is
    absent and the tracker said so too. Frames where the target is gone and the
    tracker keeps emitting a box score zero, which is what makes this metric
    worth using instead of mean IoU on visible frames.
    """
    pred, gt = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    if len(pred) != len(gt):
        raise ValueError(f"{len(pred)} predictions against {len(gt)} labels.")
    if len(gt) == 0:
        return 0.0
    visible = _present(gt)
    scores = per_frame_iou(pred, gt)
    scores[~visible] = (~_present(pred))[~visible].astype(np.float64)
    return float(scores.mean())


def success_curve(
    pred: np.ndarray, gt: np.ndarray, thresholds: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The success plot Anti-UAV410 and OTB report: P(IoU > t) against t.

    Over visible frames only, which is the convention -- an absent target has
    no overlap to threshold. Returned rather than plotted so the caller owns
    the figure.
    """
    thresholds = np.linspace(0.0, 1.0, 21) if thresholds is None else np.asarray(thresholds)
    visible = _present(np.asarray(gt, np.float64))
    if not visible.any():
        return thresholds, np.zeros_like(thresholds)
    scores = per_frame_iou(pred, gt)[visible]
    return thresholds, np.array([(scores > t).mean() for t in thresholds])


def success_auc(pred: np.ndarray, gt: np.ndarray) -> float:
    """Area under the success curve, computed exactly.

    For overlap in [0, 1], the integral of P(IoU > t) over t in [0, 1] *is*
    the mean overlap -- so this is computed directly rather than by summing a
    threshold grid, which only approximates it (a perfect tracker scores
    20/21 on a 21-point grid, not 1). Same quantity published trackers report,
    without the discretisation error.
    """
    visible = _present(np.asarray(gt, np.float64))
    if not visible.any():
        return 0.0
    return float(per_frame_iou(pred, gt)[visible].mean())


@dataclass(frozen=True)
class Dropouts:
    """Runs of frames where a visible target was not tracked."""

    lengths: tuple[int, ...]

    @property
    def episodes(self) -> int:
        return len(self.lengths)

    @property
    def frames(self) -> int:
        return sum(self.lengths)

    @property
    def longest(self) -> int:
        return max(self.lengths, default=0)

    @property
    def mean_length(self) -> float:
        return float(np.mean(self.lengths)) if self.lengths else 0.0

    def summary(self) -> str:
        if not self.lengths:
            return "no dropouts"
        return (f"{self.episodes} episode(s), {self.frames} frames lost, "
                f"mean {self.mean_length:.1f}, longest {self.longest}")


def dropout_episodes(pred: np.ndarray, gt: np.ndarray, iou_threshold: float = 0.1) -> Dropouts:
    """Maximal runs where the target is visible but the tracker has lost it.

    "Lost" is predicting nothing, or predicting a box that misses. The
    threshold is deliberately low: this counts losing the object, not tracking
    it imprecisely, and the two want different fixes.
    """
    visible = _present(np.asarray(gt, np.float64))
    lost = visible & (per_frame_iou(pred, gt) < iou_threshold)

    lengths, run = [], 0
    for is_lost in lost:
        if is_lost:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
    return Dropouts(tuple(lengths))


@dataclass(frozen=True)
class SequenceScore:
    name: str
    frames: int
    state_accuracy: float
    success_auc: float
    dropouts: Dropouts

    def row(self) -> str:
        return (f"| {self.name} | {self.frames} | {self.state_accuracy:.4f} "
                f"| {self.success_auc:.4f} | {self.dropouts.episodes} "
                f"| {self.dropouts.longest} |")


def score_sequence(name: str, pred: np.ndarray, gt: np.ndarray) -> SequenceScore:
    return SequenceScore(
        name=name,
        frames=len(gt),
        state_accuracy=state_accuracy(pred, gt),
        success_auc=success_auc(pred, gt),
        dropouts=dropout_episodes(pred, gt),
    )


def report(scores: list[SequenceScore]) -> str:
    """A markdown table plus the frame-weighted totals.

    Frame-weighted, not sequence-averaged: Anti-UAV410 sequences differ in
    length by more than an order of magnitude, and a per-sequence mean would
    let a 200-frame clip outvote a 2000-frame one.
    """
    lines = [
        "| sequence | frames | state acc | success AUC | dropouts | longest |",
        "|---|---:|---:|---:|---:|---:|",
        *(s.row() for s in scores),
    ]
    total = sum(s.frames for s in scores)
    if total:
        weighted = lambda f: sum(f(s) * s.frames for s in scores) / total  # noqa: E731
        lines += [
            f"| **all ({len(scores)} seq)** | {total} "
            f"| **{weighted(lambda s: s.state_accuracy):.4f}** "
            f"| **{weighted(lambda s: s.success_auc):.4f}** "
            f"| {sum(s.dropouts.episodes for s in scores)} "
            f"| {max((s.dropouts.longest for s in scores), default=0)} |",
        ]
    return "\n".join(lines)
