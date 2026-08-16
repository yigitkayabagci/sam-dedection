"""Quantisation-aware training, for whichever module post-training left behind.

PTQ comes first and usually finishes the job: three of EdgeTAM's four modules
clear 0.999 IoU under simulated INT8 (`docs/tensorrt_fp16.md`). The exception is
the image encoder, which collapses under min/max and only just clears 0.99 with
clipping -- roughly a hundred times fp16's error. QAT exists here for that one
module, and is not run on the others because there is nothing to recover.

Two rules keep it honest:

**QAT does not reopen the frozen memory path.** If a memory-path module fails
its accuracy gate, the answer is to leave that engine at fp16, not to retrain
the recurrent core -- for the same reasons it is frozen during fine-tuning
(`finetune.py`). `insert_quantizers` refuses unless told twice.

**The teacher is the model being quantised.** Not a bigger model: the target is
to reproduce the fp16 network exactly, so the loss that matters is disagreement
with *itself* before quantisation. That is also why a QAT schedule can be short
-- NVIDIA suggests around 10 % of the original training length -- since the
weights only have to move far enough to absorb rounding.

The only version-sensitive `nvidia-modelopt` calls live in `insert_quantizers`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# The four engines, expressed as the model attributes they are built from.
# Identical to `tools/analyze_precision.py:MODULE_ATTRS` on purpose: the module
# a measurement blames and the module QAT repairs have to be the same thing.
MODULE_ROOTS = {
    "image_encoder": ("image_encoder",),
    "memory_attention": ("memory_attention",),
    "memory_encoder": ("memory_encoder", "spatial_perceiver"),
    "sam_head": ("sam_prompt_encoder", "sam_mask_decoder", "obj_ptr_proj"),
}

# Retraining these changes what the memory bank stores; see the module header.
RECURRENT = ("memory_attention", "memory_encoder")


def insert_quantizers(model, modules, forward_loop, config=None,
                      allow_recurrent: bool = False):
    """Wrap `modules` in fake-quantisation and calibrate them, in place.

    `forward_loop(model)` must run a handful of real clips: the scales come
    from whatever activations it produces, so a loop over noise produces scales
    for a network that does not exist.
    """
    # Validate before importing: a typo in a module name should not depend on
    # having modelopt installed to be reported.
    unknown = set(modules) - set(MODULE_ROOTS)
    if unknown:
        raise KeyError(f"unknown module(s) {sorted(unknown)}; pick from {sorted(MODULE_ROOTS)}")
    blocked = set(modules) & set(RECURRENT)
    if blocked and not allow_recurrent:
        raise ValueError(
            f"{sorted(blocked)} is on the recurrent memory path, which this "
            "project keeps frozen -- a systematic change there accumulates in "
            "the memory bank instead of averaging out (docs/tensorrt_fp16.md). "
            "Leave that engine at fp16, or pass allow_recurrent=True and "
            "measure the drift across a whole clip, not one frame."
        )

    import modelopt.torch.quantization as mtq

    config = config or mtq.INT8_DEFAULT_CFG
    for name in modules:
        for attr in MODULE_ROOTS[name]:
            root = getattr(model, attr, None)
            if root is not None:
                mtq.quantize(root, config, forward_loop)
    return model


def disable_quantizers(model, patterns: list[str]) -> None:
    """Turn off quantisation for layers matching `patterns`.

    The escape hatch for a single layer that will not tolerate INT8 -- cheaper
    than dropping a whole engine back to fp16, and the first thing to try when
    a module fails its gate by a small margin.
    """
    import modelopt.torch.quantization as mtq

    for pattern in patterns:
        mtq.disable_quantizer(model, pattern)


def distillation_loss(student: dict, teacher: dict, mask_weight: float = 1.0,
                      score_weight: float = 1.0) -> torch.Tensor:
    """Disagreement between the quantised network and the fp16 one it came from.

    Soft targets, not thresholded masks: the teacher's probabilities carry how
    *confident* it was at each pixel, and reproducing that is what keeps the
    quantised mask boundary in the same place. A hard target would let the
    student drift anywhere inside the object and still score perfectly.

    The mask term is a KL divergence rather than a plain cross-entropy against
    soft targets, which differ only by the teacher's own entropy -- a constant
    with respect to the student, but one that puts the "perfect agreement"
    floor at some arbitrary positive number that moves with the teacher. As a
    KL, **zero means identical**, which is the only reading that makes a QAT
    curve interpretable.
    """
    logits = student["pred_masks_high_res"]
    with torch.no_grad():
        target = teacher["pred_masks_high_res"].detach().sigmoid().to(logits.dtype)
        target_score = teacher["object_score_logits"].detach()
        clamped = target.clamp(1e-6, 1 - 1e-6)
        entropy = -(clamped * clamped.log() + (1 - clamped) * (1 - clamped).log()).mean()

    mask = F.binary_cross_entropy_with_logits(logits, target) - entropy
    # The object score decides whether `no_obj_ptr` is written into the memory
    # bank, so a shift here is not a small error -- it changes the state of the
    # tracker for the next seven frames. Matched directly on the logit.
    score = F.mse_loss(student["object_score_logits"],
                       target_score.to(student["object_score_logits"].dtype))
    return mask_weight * mask + score_weight * score


def calibration_loop(model, batches):
    """A `forward_loop` for `insert_quantizers`, over real clips.

    Returns a callable rather than running anything, because `mtq.quantize`
    decides when the forward passes happen. It ignores the submodule it is
    handed and drives the *whole* predictor: the encoder's activations during a
    tracked clip are the ones the deployed engine sees, and running it on
    isolated frames would calibrate for a different distribution.
    """
    from .clip_loop import propagate

    def run(_submodule) -> None:
        for batch in batches:
            with torch.no_grad():
                propagate(model, batch.images, batch.boxes[:, 0])

    return run
