"""One image, several prompts, no memory bank -- under autograd.

`clip_loop.py`'s sibling, and deliberately the *same* code path for the frame
they have in common. That one propagates T frames and lets each condition on
the memory the model wrote before it. This one runs a single frame, which is
exactly what frame 0 of a clip already is: `track_step` with
`is_init_cond_frame=True` and an empty `output_dict`. Going through
`track_step` rather than reaching into the SAM head directly is what makes that
claim true instead of approximately true -- the initial frame's features are
conditioned the same way here as at inference, whatever the config says about
`directly_add_no_mem_embed`, and there is no second opinion about it to
maintain.

Three things follow from the shape of the problem:

**The memory path never runs.** `run_mem_encoder=False` skips the memory
encoder, and with no stored memories there is nothing for memory attention to
read. That is the point of training on static data: SAM 2's own recipe pretrains
the image encoder on still images and only then trains the memory path on
video, and it *freezes the image encoder* when it later extends to longer
sequences. The two are separate problems and this module is the first of them.

**The image is encoded once and prompted many times.** The encoder is ~38.7
GFLOP at 512 and does not depend on the prompt, so encoding per instance would
pay for it K times over for identical features. Instead the flattened backbone
features are `index_select`ed along their batch axis, once per instance, and
`track_step` sees a batch of N instances that happens to come from B images.
SAM 2 batches *objects* -- each row carries its own prompt, its own pointer and
its own score, and the attention never mixes rows -- so N instances ride the
same machinery N tracked objects would, with no change to the model.

**Padding is never computed, not merely ignored.** Instance counts differ per
window, so the batch is ragged. `valid` selects the real rows before the
features are expanded, which means a padded slot costs nothing rather than
costing a full decoder pass whose loss is then masked out.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .aerial import DatasetSpec, InstanceGates, Sample, load_image, normalise, sample_masks
from .clip_loop import _capture_head, box_prompt, check_model


# --------------------------------------------------------------------------
# Batches
# --------------------------------------------------------------------------


@dataclass
class ImageBatch:
    """B windows and up to K instances each; `valid` says which slots are real.

    `masks[b, k]` is only meaningful where `valid[b, k]` -- padded rows hold
    zeros, which would otherwise read as "the target here is empty", a
    different and much worse claim than "there is no target here".
    """

    images: torch.Tensor        # [B, 3, S, S]
    boxes: torch.Tensor         # [B, K, 4], model-input pixels
    masks: torch.Tensor         # [B, K, S, S] bool
    valid: torch.Tensor         # [B, K] bool
    samples: list[Sample] | None = None

    def __len__(self) -> int:
        return int(self.images.shape[0])

    @property
    def instances(self) -> int:
        return int(self.valid.sum())

    def to(self, device: str) -> "ImageBatch":
        """The same batch on another device.

        Collating on the CPU and moving here is what lets the loader assemble
        the next batch on a worker thread while the GPU is still busy with this
        one -- the same division of labour `clip_loop.Batch.to` exists for.
        """
        return ImageBatch(
            images=self.images.to(device, non_blocking=True),
            boxes=self.boxes.to(device, non_blocking=True),
            masks=self.masks.to(device, non_blocking=True),
            valid=self.valid.to(device, non_blocking=True),
            samples=self.samples,
        )


def collate(samples: Sequence[Sample], spec: DatasetSpec,
            gates: InstanceGates = InstanceGates(), device: str = "cpu",
            executor=None, gray: bool = True) -> ImageBatch:
    """Read `samples`' pixels and their instance masks into one padded batch.

    `executor` spreads the per-sample reads across a thread pool -- decoding a
    PNG and running connected components on its semantic map both release the
    GIL, so this is real parallelism. It is also where the parallelism belongs:
    a batch is a few dozen images, and assembling whole batches concurrently
    instead would hold a copy of each in host memory at once.
    """
    mapper = map if executor is None else executor.map
    size = samples[0].size
    pixels = list(mapper(lambda s: load_image(s.frame.image, s.origin, s.window,
                                              s.size, gray), samples))
    masks = list(mapper(lambda s: sample_masks(s, spec, gates), samples))

    width = max(len(s.instances) for s in samples)
    boxes = np.zeros((len(samples), width, 4), dtype=np.float32)
    stacked = np.zeros((len(samples), width, size, size), dtype=bool)
    valid = np.zeros((len(samples), width), dtype=bool)
    for b, (sample, rows) in enumerate(zip(samples, masks)):
        k = len(sample.instances)
        boxes[b, :k] = sample.boxes
        stacked[b, :k] = rows
        valid[b, :k] = True

    return ImageBatch(
        images=normalise(np.stack(pixels), device),
        boxes=torch.from_numpy(boxes).to(device),
        masks=torch.from_numpy(stacked).to(device),
        valid=torch.from_numpy(valid).to(device),
        samples=list(samples),
    )


# --------------------------------------------------------------------------
# Forward
# --------------------------------------------------------------------------


def propagate_image(model, images: torch.Tensor, boxes: torch.Tensor,
                    valid: torch.Tensor | None = None) -> dict:
    """Encode `images` [B, 3, S, S] once; prompt every valid box against them.

    Returns the flattened per-instance view the loss works in: `rows` is the
    flat `b * K + k` index of each surviving instance, so a caller can gather
    its own targets in the same order without re-deriving the selection.
    """
    check_model(model)
    batch, width = boxes.shape[:2]
    if valid is None:
        valid = torch.ones(batch, width, dtype=torch.bool, device=boxes.device)
    rows = valid.reshape(-1).nonzero().flatten()
    if rows.numel() == 0:
        raise ValueError("every instance slot in this batch is padding.")
    # Integer division, not floor of a float: the row index has to be exact for
    # a batch of a few thousand instances.
    image_of = torch.div(rows, width, rounding_mode="floor")

    backbone = model.forward_image(images)
    _, feats, pos_embeds, feat_sizes = model._prepare_backbone_features(backbone)
    # SAM 2 hands features on as HWxNxC, so the batch axis is 1. Selecting
    # rather than repeating is what keeps a ragged batch free of padded work.
    feats = [f.index_select(1, image_of) for f in feats]
    pos_embeds = [p.index_select(1, image_of) for p in pos_embeds]

    with _capture_head(model) as captured:
        out = model.track_step(
            frame_idx=0,
            is_init_cond_frame=True,
            current_vision_feats=feats,
            current_vision_pos_embeds=pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=box_prompt(boxes.reshape(-1, 4)[rows]),
            mask_inputs=None,
            output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            num_frames=1,
            run_mem_encoder=False,      # nothing will ever read this memory
        )
        if not captured:
            raise RuntimeError(
                "the SAM head was never called, so there is no mask and no IoU "
                "prediction to supervise. This model's track_step does not "
                "follow the path this loop assumes.")
        _, _, ious, _, high_res_masks, _, _ = captured[-1]

    return {
        "pred_masks_high_res": high_res_masks,
        # Under multimask output the head returns three IoU predictions and
        # keeps the mask it rates highest; that maximum is the prediction that
        # belongs to the mask being supervised. Same convention as `clip_loop`.
        "ious": ious.max(dim=-1, keepdim=True).values,
        "pred_masks": out["pred_masks"],
        "rows": rows,
        "image_of": image_of,
    }


def image_losses(model, batch: ImageBatch, weights=None):
    """Propagate a batch and score every prompted instance in it.

    Every frame is scored -- there is no prompted frame to exclude the way
    `clip_losses` excludes frame 0. That looks like the model being handed the
    answer, and it is not: a box says *where*, and the target is the mask,
    which is the whole task. SAM 2's image pretraining is this and nothing
    else.
    """
    from .losses import Weights, instance_loss

    weights = weights or Weights()
    outputs = propagate_image(model, batch.images, batch.boxes, batch.valid)
    targets = batch.masks.reshape(-1, *batch.masks.shape[-2:])[outputs["rows"]]
    return instance_loss(outputs, targets, weights)


@torch.no_grad()
def instance_iou(model, batch: ImageBatch) -> np.ndarray:
    """Per-instance mask IoU at the deployment threshold, as a flat array.

    The static stand-in for J&F. It is a proxy and should be read as one: it
    scores a single prompted frame, so it cannot see the failure this project
    exists to fix, which only appears once a memory bank is in the loop. What
    it *can* say is whether the encoder now separates one vehicle from the one
    beside it, which is the thing this stage is trying to buy.
    """
    outputs = propagate_image(model, batch.images, batch.boxes, batch.valid)
    logits = outputs["pred_masks_high_res"]
    if logits.dim() == 4:
        logits = logits[:, 0]
    predicted = (logits > 0.0).flatten(1)
    targets = batch.masks.reshape(-1, *batch.masks.shape[-2:])[outputs["rows"]].flatten(1)

    intersection = (predicted & targets).sum(1).float()
    union = (predicted | targets).sum(1).float()
    # Two empty masks agree perfectly rather than being undefined -- both said
    # "nothing here" and both were right. Same reading as `src/accuracy.py`.
    iou = torch.where(union > 0, intersection / union.clamp(min=1.0),
                      torch.ones_like(union))
    return iou.float().cpu().numpy()


# --------------------------------------------------------------------------
# Splits and the stream the schedule pulls on
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageSplit:
    """Windows for one split, plus what it takes to read them.

    The spec and the gates travel *with* the split rather than being read from
    a global, because the whole point of the comparison this feeds is that two
    runs see identical data. A gate value that differed between them would be a
    silent third variable.
    """

    samples: Sequence[Sample]
    spec: DatasetSpec
    gates: InstanceGates = InstanceGates()
    gray: bool = True


def auto_batch_size(model, split: ImageSplit, device: str = "cuda",
                    maximum: int | None = None, reserve: float = 0.15,
                    verbose: bool = True) -> int:
    """The largest window batch that trains on this GPU.

    Image mode fits far more than clip mode does -- there is no clip length
    multiplying the activations and no memory path holding a bank -- so the
    answer is a different number on the same card, and guessing it from the
    clip-mode figure would leave most of the GPU idle.

    The probe uses the *widest* windows in the pool, not the first ones: cost
    scales with the number of instances prompted, so measuring on a
    single-instance window would report a size that OOMs the moment a crowded
    scene comes round.
    """
    from .loader import measure_batch_size

    pool = sorted(split.samples, key=lambda s: -len(s.instances))

    def step(n: int):
        batch = collate(pool[:n], split.spec, split.gates, "cpu", gray=split.gray)
        return image_losses(model, batch.to(device))[0]

    return measure_batch_size(model, step, device,
                              maximum=min(maximum or len(pool), len(pool)),
                              reserve=reserve, verbose=verbose)


def stream(split: ImageSplit, batch: int, seed: int | None, limit: int | None,
           device: str = "cuda", workers: int = 8, depth: int = 2):
    """Shuffled, prefetched batches of windows -- the image-mode data pipeline.

    Reuses `loader.batch_clips` for the shuffle and `loader.prefetch_with` for
    the threading, so a run in image mode is reproducible from its seed in
    exactly the way a clip-mode run is, and for the same reason: futures are
    consumed in order rather than as they complete.
    """
    from .loader import batch_clips, prefetch_with

    chunks = batch_clips(split.samples, batch, seed=seed, limit=limit)
    return prefetch_with(
        chunks,
        lambda chunk, pool: collate(chunk, split.spec, split.gates, "cpu", pool,
                                    split.gray),
        device=device, workers=workers, depth=depth,
    )
