"""The training objective, one term per thing the model has to get right.

SAM 2's own recipe is the starting point -- focal + dice on the mask, L1 on the
IoU head -- with two additions this domain needs:

`object_score`   BCE against Anti-UAV410's per-frame `exist` flag. This is the
                 term that matters most here. EdgeTAM writes `no_obj_ptr` into
                 the memory bank whenever `object_score_logits` goes negative,
                 and the next frames read it back; a head that fires spuriously
                 on thermal clutter poisons its own memory. Nothing in the
                 stock checkpoint ever taught it what a drone-shaped absence
                 looks like in infrared.

`box_projection` the fallback for frames where the teacher's mask failed a
                 quality gate. Rather than drop those frames, the mask is
                 constrained through its own row and column projections
                 (BoxInst's construction): the mask must span the box's extent
                 on each axis and nothing outside it. Weaker than a real mask,
                 far better than no supervision, and it uses a label that is
                 always available.

Everything here takes logits, returns **per-sample** losses, and leaves the
reduction to the caller -- a clip has frames that should not contribute at all
(the target is absent, or the mask is unlabelled) and averaging inside would
hide them.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Sigmoid focal loss, mean over pixels, per sample.

    The class imbalance it corrects for is extreme here in a way it is not on
    natural video: a drone is a few hundred pixels of a 262144-pixel frame, so
    an unweighted BCE is minimised by predicting "background" everywhere.
    """
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * (1 - p_t).pow(gamma)
    if alpha >= 0:
        loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    return loss.flatten(1).mean(1)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """1 - Dice, per sample. Scale-free, which focal loss is not."""
    prob = logits.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (prob * targets).sum(1)
    denominator = prob.sum(1) + targets.sum(1)
    return 1 - (numerator + eps) / (denominator + eps)


def iou_head_loss(pred_iou: torch.Tensor, logits: torch.Tensor,
                  targets: torch.Tensor) -> torch.Tensor:
    """L1 between the predicted IoU and the IoU actually achieved.

    The target is detached: this teaches the head to *report* its quality, and
    must not become a second gradient path that changes the mask to make the
    report easier.
    """
    with torch.no_grad():
        pred = (logits > 0).flatten(1).float()
        gt = (targets > 0.5).flatten(1).float()
        intersection = (pred * gt).sum(1)
        union = pred.sum(1) + gt.sum(1) - intersection
        actual = intersection / union.clamp(min=1.0)
    return (pred_iou.flatten(1).squeeze(-1) - actual).abs()


def object_score_loss(logits: torch.Tensor, exist: torch.Tensor) -> torch.Tensor:
    """BCE on `object_score_logits` against the per-frame `exist` flag."""
    return F.binary_cross_entropy_with_logits(
        logits.flatten(1).squeeze(-1), exist.float(), reduction="none"
    )


def _projection_dice(projected: torch.Tensor, target: torch.Tensor,
                     eps: float = 1.0) -> torch.Tensor:
    numerator = 2 * (projected * target).sum(-1)
    denominator = projected.sum(-1) + target.sum(-1)
    return 1 - (numerator + eps) / (denominator + eps)


def box_projection_loss(logits: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """BoxInst's projection term: the mask's shadow on each axis is the box's.

    Taking the max down each column reduces a mask to "is anything lit in this
    column"; for a mask that exactly fills its box, that is the box's own
    horizontal extent. Matching both projections pins the mask's bounding box
    without ever claiming to know its shape -- which is all a box label
    actually says.
    """
    prob = logits.sigmoid()
    if prob.dim() == 4:
        prob = prob.squeeze(1)
    height, width = prob.shape[-2:]

    columns = torch.arange(width, device=prob.device, dtype=prob.dtype)
    rows = torch.arange(height, device=prob.device, dtype=prob.dtype)
    x0, y0, x1, y1 = (boxes[:, i:i + 1] for i in range(4))
    target_x = ((columns >= x0) & (columns < x1)).to(prob.dtype)
    target_y = ((rows >= y0) & (rows < y1)).to(prob.dtype)

    return (_projection_dice(prob.amax(dim=-2), target_x)
            + _projection_dice(prob.amax(dim=-1), target_y))


@dataclass(frozen=True)
class Weights:
    """SAM 2's published weighting, plus the two terms this domain adds.

    `focal` dominates by design: it is the only term whose scale reflects how
    few pixels a drone occupies.
    """

    focal: float = 20.0
    dice: float = 1.0
    iou: float = 1.0
    object_score: float = 1.0
    box_projection: float = 1.0


def frame_loss(
    outputs: dict,
    masks: torch.Tensor | None,
    boxes: torch.Tensor | None,
    exist: torch.Tensor,
    weights: Weights = Weights(),
) -> tuple[torch.Tensor, dict[str, float]]:
    """One frame's loss for a batch of clips, plus its terms for logging.

    `masks` is None for frames the teacher could not label, `boxes` is None
    only when the target is absent. Which supervision applies is decided per
    sample, not per batch: a batch mixes labelled and unlabelled frames, and
    collapsing that decision to the batch would either discard good masks or
    invent bad ones.
    """
    logits = outputs["pred_masks_high_res"]
    if logits.dim() == 3:
        logits = logits[:, None]
    visible = exist.bool()

    total = weights.object_score * object_score_loss(
        outputs["object_score_logits"], exist
    )
    terms = {"object_score": float(total.mean().detach())}

    if masks is not None and visible.any():
        labelled = visible & masks.flatten(1).any(1)
        if labelled.any():
            sub_logits = logits[labelled]
            sub_masks = masks[labelled].unsqueeze(1).to(sub_logits.dtype)
            mask_terms = (
                weights.focal * focal_loss(sub_logits, sub_masks)
                + weights.dice * dice_loss(sub_logits, sub_masks)
                + weights.iou * iou_head_loss(
                    outputs["ious"][labelled], sub_logits, sub_masks)
            )
            total = total.index_add(0, labelled.nonzero().flatten(), mask_terms)
            terms["mask"] = float(mask_terms.mean().detach())
            visible = visible & ~labelled  # these frames are already supervised

    if boxes is not None and visible.any():
        projection = weights.box_projection * box_projection_loss(
            logits[visible], boxes[visible]
        )
        total = total.index_add(0, visible.nonzero().flatten(), projection)
        terms["box_projection"] = float(projection.mean().detach())

    return total.mean(), terms
