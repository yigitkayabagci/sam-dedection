"""Propagate a short clip through EdgeTAM *under autograd*.

EdgeTAM ships no training code, and neither does the SAM 2 predictor wrapped
around it: `propagate_in_video` is decorated `@torch.inference_mode()`, so it
cannot be used to compute a gradient. What it does per frame, though, is a
short and stable sequence -- encode the frame, call `track_step`, file the
result in an `output_dict` that the next frame's memory bank reads. That is
what this module reproduces, and nothing more.

Three decisions are worth knowing about:

**The model stays in `eval()` mode while training.** Not an oversight. EdgeTAM's
RepViT trunk is full of batch norms, and letting them drift during a fine-tune
on 410 sequences of one class would (a) diverge from the frozen statistics that
TensorRT folds into its weights at export, so the engine would stop matching
the checkpoint it came from, and (b) invalidate any INT8 calibration collected
before it. Independently, SAM 2's `track_step` only emits `object_score_logits`
when `self.training` is False -- both point the same way.

**No teacher forcing.** Each frame conditions on the memory the model itself
wrote on the frames before it, exactly as it will at inference. Feeding ground
truth into the memory bank would train a model that has never seen its own
mistakes -- which is the entire failure mode this project is trying to fix.

**The batch dimension is clips, not frames.** SAM 2 batches *objects*, and each
row of that batch carries its own memory, its own object pointer and its own
object score; the attention never mixes rows. So B independent clips ride the
same machinery B tracked objects would, with no change to the model.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from .antiuav import Clip, clip_masks, clip_tensor

REQUIRED = ("forward_image", "_prepare_backbone_features", "track_step",
            "_forward_sam_heads")


def check_model(model) -> None:
    """Fail at setup with a readable message rather than mid-epoch."""
    missing = [name for name in REQUIRED if not hasattr(model, name)]
    if missing:
        raise AttributeError(
            f"This does not look like a SAM 2 / EdgeTAM model: missing {missing}. "
            "Pass the object `build_sam2_video_predictor` returned."
        )
    if model.training:
        raise RuntimeError(
            "Call model.eval() before training. Batch-norm statistics must stay "
            "frozen so the checkpoint keeps matching its exported engines, and "
            "SAM 2 withholds object_score_logits in training mode."
        )


def box_prompt(boxes: torch.Tensor) -> dict:
    """SAM 2 encodes a box as two corner points labelled 2 and 3.

    Coordinates are in model-input pixels -- which is what `Clip.boxes` already
    produces, so no normalisation round-trip is needed here.
    """
    coords = boxes.reshape(-1, 2, 2).float()
    labels = torch.tensor([2, 3], dtype=torch.int32, device=boxes.device)
    return {"point_coords": coords, "point_labels": labels.expand(coords.shape[0], 2)}


@contextmanager
def _capture_head(model):
    """Collect the SAM head's full return, which `track_step` does not pass on.

    `track_step` keeps only the masks, the pointer and the object score; the
    IoU-head prediction never leaves `_forward_sam_heads`. Wrapping the method
    is the same idiom `src/trackers/edgetam_trt_tracker.py` uses to reach into
    this model, and it leaves the model untouched afterwards.
    """
    original = model._forward_sam_heads
    captured: list[tuple] = []

    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        captured.append(out)
        return out

    # Assigning the original back would leave a bound method sitting in the
    # instance dict for good, shadowing the class and holding a reference to
    # the model. `src/pipeline.py:_postprocess_timer` handles the same hazard
    # the same way: remember whether the attribute was ever the instance's.
    was_instance_attr = "_forward_sam_heads" in vars(model)
    model._forward_sam_heads = wrapped
    try:
        yield captured
    finally:
        if was_instance_attr:
            model._forward_sam_heads = original
        else:
            vars(model).pop("_forward_sam_heads", None)


def propagate(model, images: torch.Tensor, boxes: torch.Tensor) -> list[dict]:
    """Track `images` [B, T, 3, S, S] from a box on frame 0, keeping the graph.

    Returns one dict per frame with the keys `frame_loss` needs:
    `pred_masks_high_res`, `object_score_logits`, `ious`.
    """
    check_model(model)
    batch, frames = images.shape[:2]
    output_dict: dict[str, dict] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    per_frame: list[dict] = []

    with _capture_head(model) as captured:
        for t in range(frames):
            backbone = model.forward_image(images[:, t])
            _, feats, pos_embeds, feat_sizes = model._prepare_backbone_features(backbone)

            captured.clear()
            out = model.track_step(
                frame_idx=t,
                is_init_cond_frame=(t == 0),
                current_vision_feats=feats,
                current_vision_pos_embeds=pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=box_prompt(boxes) if t == 0 else None,
                mask_inputs=None,
                output_dict=output_dict,
                num_frames=frames,
                run_mem_encoder=True,
            )
            # The prompted frame is a conditioning frame; the rest are what the
            # memory bank is built from. Same split `SAM2VideoPredictor` makes.
            key = "cond_frame_outputs" if t == 0 else "non_cond_frame_outputs"
            output_dict[key][t] = out

            if not captured:
                raise RuntimeError(
                    f"frame {t}: the SAM head was never called, so there is no "
                    "IoU prediction to supervise. This model's track_step does "
                    "not follow the path this loop assumes."
                )
            # `track_step` can run the head more than once on a frame; the last
            # call is the one whose masks it kept.
            _, _, ious, _, high_res_masks, _, object_score_logits = captured[-1]
            per_frame.append({
                "pred_masks_high_res": high_res_masks,
                "object_score_logits": object_score_logits,
                "ious": ious.max(dim=-1, keepdim=True).values,
                "pred_masks": out["pred_masks"],
            })
    if len(per_frame) != frames:
        raise RuntimeError(f"produced {len(per_frame)} outputs for {frames} frames.")
    return per_frame


@dataclass
class Batch:
    """A collated batch of clips: the tensors the loop and the loss both take.

    `masks[t]` is None when *no* clip in the batch has a teacher mask on that
    frame; otherwise it is a [B, S, S] tensor whose unlabelled rows are empty.
    Those rows are then excluded per row by `frame_loss`, not per frame -- a
    batch routinely mixes labelled and unlabelled clips on the same frame.
    """

    images: torch.Tensor              # [B, T, 3, S, S]
    boxes: torch.Tensor               # [B, T, 4], NaN where absent
    exist: torch.Tensor               # [B, T]
    masks: list[torch.Tensor | None]  # T entries
    clips: list[Clip] | None = None

    def __len__(self) -> int:
        return int(self.images.shape[0])

    @property
    def frames(self) -> int:
        return int(self.images.shape[1])


def collate(clips: list[Clip], stores: list[dict], device: str = "cuda") -> Batch:
    """Build a `Batch` by reading the clips' pixels and their pseudo-masks."""
    rows = [clip_masks(c, s) for c, s in zip(clips, stores)]
    size = clips[0].size

    masks: list[torch.Tensor | None] = []
    for t in range(len(clips[0].indices)):
        frame = [row[t] for row in rows]
        if all(m is None for m in frame):
            masks.append(None)
            continue
        stacked = np.stack([
            m if m is not None else np.zeros((size, size), bool) for m in frame
        ])
        masks.append(torch.from_numpy(stacked).to(device))

    return Batch(
        images=torch.stack([clip_tensor(c, device) for c in clips]),
        boxes=torch.from_numpy(np.stack([c.boxes for c in clips])).to(device).float(),
        exist=torch.from_numpy(np.stack([c.exist for c in clips])).to(device).float(),
        masks=masks,
        clips=clips,
    )


def clip_losses(model, batch: Batch, weights=None, skip_first: bool = True):
    """Propagate a batch and score every frame after the prompted one.

    Frame 0 carries the user's box, so the model is being handed the answer;
    including it would reward memorising the prompt. `skip_first=False` is
    there for the overfit-one-clip sanity check, where seeing frame 0 collapse
    to zero is the point.
    """
    from .losses import Weights, frame_loss

    weights = weights or Weights()
    outputs = propagate(model, batch.images, batch.boxes[:, 0])

    total, log = 0.0, {}
    scored = range(1 if skip_first else 0, batch.frames)
    for t in scored:
        boxes = batch.boxes[:, t]
        loss, terms = frame_loss(
            outputs[t],
            batch.masks[t],
            None if torch.isnan(boxes).all() else torch.nan_to_num(boxes),
            batch.exist[:, t],
            weights,
        )
        total = total + loss
        for name, value in terms.items():
            log[name] = log.get(name, 0.0) + value
    count = max(len(scored), 1)
    return total / count, {k: v / count for k, v in log.items()}
