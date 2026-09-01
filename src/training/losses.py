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


def neighbour_terms(
    logits: torch.Tensor,
    others: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The dense-cluster failure, as a number to read and a term to train on.

    Focal and dice already push the mask off a neighbour -- those pixels are
    zeros in the target -- but they push on every zero pixel equally, and a
    neighbour is a handful of them among a window of easy background. The mask
    that swallows the cluster is therefore barely worse by the published
    objective than the mask that does not, which is why the objective does not
    prevent it. Naming the pixels is what changes that.

    `others` is the union of every *other* indexed instance in the same
    window, with the target's own mask removed. Two quantities come back over
    those pixels, and they are deliberately not the same thing:

    `claimed`   the share of the neighbours' pixels this mask covers: 0.0
                claims nothing, 1.0 claims all of it. A fraction, so it is
                comparable across window sizes and target sizes and can be
                read directly -- "this run's masks claim 31% of the objects
                beside them". This is the diagnostic, and it is what gets
                reported whether or not anything is trained against it.

    `penalty`   the same pixels as binary cross-entropy against zero
                (`softplus(logit)` is exactly `-log(1 - sigmoid(logit))`).
                This is what enters the loss, and the reason is gradient:
                `claimed` is built on `sigmoid`, whose derivative vanishes
                once a logit saturates, so a mask that is *confidently*
                swallowing the cluster -- precisely the failure -- would be
                trained on almost nothing. The cross-entropy's gradient is
                `sigmoid(logit)`, which is largest exactly there.

    The third value says which instances actually have a neighbour. An
    instance alone in its window has nothing to leak onto and must not be
    averaged in as a zero, which would report the problem as smaller the more
    isolated targets the set contains.

    **What this cannot see.** `others` is built from the instances the gates
    kept. A component `max_area` or `fill` rejected is not in it, so a cluster
    that decomposed into one big blob is background here, not a neighbour. The
    term measures separation between *labelled* objects and says nothing about
    the ones the index dropped.
    """
    if logits.dim() == 4:
        logits = logits.squeeze(1)
    others = others.to(logits.dtype)
    area = others.sum(dim=(-2, -1))
    scale = area.clamp(min=eps)
    claimed = (logits.sigmoid() * others).sum(dim=(-2, -1)) / scale
    penalty = (F.softplus(logits) * others).sum(dim=(-2, -1)) / scale
    return claimed, penalty, area > 0


@dataclass(frozen=True)
class Weights:
    """SAM 2's published weighting, plus the two terms this domain adds.

    `focal` dominates by design: it is the only term whose scale reflects how
    few pixels a drone occupies.

    `object_score` and `box_projection` are both video-path terms and both go
    unused on static data -- see `instance_loss`.
    """

    focal: float = 20.0
    dice: float = 1.0
    iou: float = 1.0
    object_score: float = 1.0
    box_projection: float = 1.0
    # `neighbour` is off by default because every run before it existed was
    # taken on the published objective, and a term that changes the loss
    # changes what the validation number means. `instance_loss` reports the
    # leak whatever the weight, so a run measures the problem before anyone
    # decides to train against it.
    neighbour: float = 0.0


def instance_loss(
    outputs: dict,
    masks: torch.Tensor,
    weights: Weights = Weights(),
    others: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Promptable segmentation on a still image: the mask terms, and only those.

    SAM 2's image-pretraining objective exactly -- focal + dice on the mask, L1
    on the IoU head. Every instance is supervised; there is no per-sample
    decision to make, because a static set has no unlabelled frames and no
    frames where the target is absent.

    **The `object_score` term is deliberately gone, not merely zero-weighted.**
    It is BCE against Anti-UAV410's per-frame `exist` flag, and a static
    segmentation set has no such label: every prompted instance is present by
    construction. Keeping the term would train the head against a constant 1,
    which does not teach it nothing -- it teaches it to fire unconditionally,
    and `object_score_logits` is precisely the signal whose spurious firing
    poisons the memory bank on video. Leaving it untrained here and training it
    on video, where the label is real, is the only coherent split.

    `box_projection` is gone for the opposite reason: it is the fallback for
    frames whose teacher mask failed a quality gate, and here the mask *is* the
    ground truth. There is nothing to fall back to.

    **`others` is reported whether or not it is trained on.** Passing it always
    computes `neighbour_terms` into the returned terms; it enters the loss only
    where `weights.neighbour` is non-zero. That split is deliberate: the first
    thing a run should establish is how much of the dense-cluster failure it
    actually has, and a term that is in the loss from the start makes its own
    measurement unreadable.
    """
    logits = outputs["pred_masks_high_res"]
    if logits.dim() == 3:
        logits = logits[:, None]
    targets = masks.unsqueeze(1).to(logits.dtype) if masks.dim() == 3 else masks.to(logits.dtype)

    focal = focal_loss(logits, targets)
    dice = dice_loss(logits, targets)
    iou = iou_head_loss(outputs["ious"], logits, targets)
    total = weights.focal * focal + weights.dice * dice + weights.iou * iou
    loss = total.mean()
    terms = {
        "focal": float(focal.mean().detach()),
        "dice": float(dice.mean().detach()),
        "iou": float(iou.mean().detach()),
    }

    if others is not None:
        claimed, penalty, crowded = neighbour_terms(logits, others)
        # Only instances that have a neighbour carry the term. Averaging the
        # isolated ones in as zeros would make a set of lonely targets look
        # like a model that separates well.
        terms["crowded"] = float(crowded.to(claimed.dtype).mean().detach())
        if bool(crowded.any()):
            terms["neighbour"] = float(claimed[crowded].mean().detach())
            if weights.neighbour:
                loss = loss + weights.neighbour * penalty[crowded].mean()
        else:
            terms["neighbour"] = 0.0
    return loss, terms


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
