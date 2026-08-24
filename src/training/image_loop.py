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

from .aerial import Sample, load_image, normalise, sample_masks
from .clip_loop import (
    _capture_head,
    box_prompt,
    check_model,
    jitter_boxes,
    point_prompt,
)


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


def collate(samples: Sequence[Sample], device: str = "cpu",
            executor=None) -> ImageBatch:
    """Read `samples`' pixels and their instance masks into one padded batch.

    **Every sample decodes itself**, through the `Source` it carries. That is
    what lets one batch hold VTUAV's instance masks beside Kust4K's decomposed
    semantic ones -- and training on several datasets at once is the only way
    a trunk carrying general visual features sees past any single sensor,
    city and set of annotation habits.

    `executor` spreads the per-sample reads across a thread pool -- decoding a
    PNG and running connected components on its map both release the GIL, so
    this is real parallelism. It is also where the parallelism belongs: a batch
    is a few dozen images, and assembling whole batches concurrently instead
    would hold a copy of each in host memory at once.
    """
    mapper = map if executor is None else executor.map
    size = samples[0].size
    pixels = list(mapper(
        lambda s: load_image(s.frame.image, s.origin, s.window, s.size,
                             s.source.gray if s.source else True), samples))
    masks = list(mapper(sample_masks, samples))

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


# The measurement ladder, weakest last -- what `eval_instances.py` offers and
# what `12_encoder_probe.ipynb` scores a checkpoint through.
PROMPTS = ("box", "jitter", "point")
# The same three plus one that only makes sense while training; see `mix_boxes`.
# Kept apart from `PROMPTS` so a training-only mode can never be mistaken for a
# rung on the ladder a published number was taken from.
TRAIN_PROMPTS = PROMPTS + ("mix",)


def mix_boxes(boxes: torch.Tensor, fraction: float, share: float = 0.5,
              generator: torch.Generator | None = None) -> torch.Tensor:
    """Each box either exact or jittered, chosen per row.

    **Training on nothing but exact boxes is what `12_encoder_probe.ipynb`
    caught.** All four of 08's checkpoints score 0.057 to 0.100 IoU *below*
    stock EdgeTAM under a jittered prompt, at both window scales and under all
    three training methods, while beating it under an exact box. The cause is
    not subtle: every prompt the loop has ever built came from a ground-truth
    instance box, so the decoder learned box-to-mask as a tight geometric
    mapping and answers a displaced box by faithfully segmenting the displaced
    region. Stock EdgeTAM inherits SAM 2's prompt-noise augmentation and is
    more forgiving.

    That regression is the one that matters most for what comes next: in
    tracking only frame 0 is prompted by hand, and every frame after it is
    prompted by the memory path's own drifting estimate -- a jittered box by
    another name. A checkpoint that loses a tenth of an IoU to a loose box is
    the wrong thing to carry into stage C.

    `share` of the rows keep their exact box, the rest are jittered, so the
    model sees both distributions rather than trading one for the other. The
    two are mixed **per row and not per batch** because one window contributes
    up to `--max-instances` prompts against a single encode: per-batch choice
    would correlate every prompt in a window and cut the effective number of
    draws by that factor.

    `point` is deliberately not in the mix. It builds one token where a box
    builds two, so mixing it per row needs the prompt encoder's padding label,
    which is a change to what a batch *is* rather than to what is in it. It
    stays available as a whole-batch mode.
    """
    boxes = boxes.reshape(-1, 4).float()
    if not fraction:
        return boxes
    moved = jitter_boxes(boxes, fraction, generator)
    # Same idiom as `jitter_boxes`: drawn into a tensor that already sits on
    # the boxes' device, so a CPU generator and a CUDA batch fail loudly here
    # rather than silently sampling from somewhere else.
    keep = torch.empty(boxes.shape[0], 1, device=boxes.device,
                       dtype=boxes.dtype).uniform_(0.0, 1.0, generator=generator)
    return torch.where(keep < float(share), boxes, moved)


def build_prompt(boxes: torch.Tensor, prompt: str = "box",
                 jitter: float = 0.0,
                 generator: torch.Generator | None = None) -> dict:
    """The `point_inputs` dict for one flat [N, 4] stack of boxes.

    `box` is the ground-truth rectangle, exactly. `jitter` is the same box with
    each edge moved by up to `jitter` of its own side, and `point` is a single
    click at the centre. See `clip_loop.point_prompt` for why the weaker two
    are worth scoring: they are what separates a question about the encoder
    from a question about the prompt.

    `mix` is the fourth, and it is for **training** rather than for scoring:
    per row, either the exact box or a jittered one. `mix_boxes` says what it
    is for.
    """
    if prompt not in TRAIN_PROMPTS:
        raise ValueError(f"prompt must be one of {TRAIN_PROMPTS}, got {prompt!r}")
    if prompt == "point":
        return point_prompt(boxes)
    if prompt == "jitter":
        boxes = jitter_boxes(boxes, jitter, generator)
    elif prompt == "mix":
        boxes = mix_boxes(boxes, jitter, generator=generator)
    return box_prompt(boxes)


def propagate_image(model, images: torch.Tensor, boxes: torch.Tensor,
                    valid: torch.Tensor | None = None, prompt: str = "box",
                    jitter: float = 0.0,
                    generator: torch.Generator | None = None) -> dict:
    """Encode `images` [B, 3, S, S] once; prompt every valid box against them.

    Returns the flattened per-instance view the loss works in: `rows` is the
    flat `b * K + k` index of each surviving instance, so a caller can gather
    its own targets in the same order without re-deriving the selection.

    `prompt` selects how the box becomes a prompt -- see `build_prompt`. It
    defaults to `box`, which is what the training loop wants and what every
    number recorded before this argument existed was taken with.
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
    # The top level as a map, before the per-instance selection: this is the
    # tensor the mask decoder consumes and the one the ONNX encoder graph
    # exports, and handing it back means an anchoring term costs no second
    # forward pass through the encoder.
    top, (fh, fw) = feats[-1], feat_sizes[-1]
    encoded = top.permute(1, 2, 0).reshape(top.shape[1], top.shape[2], fh, fw)
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
            point_inputs=build_prompt(boxes.reshape(-1, 4)[rows], prompt,
                                      jitter, generator),
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
        "features": encoded,
        "rows": rows,
        "image_of": image_of,
    }


def image_losses(model, batch: ImageBatch, weights=None, anchor=None,
                 anchor_weight: float = 0.0, prompt: str = "box",
                 jitter: float = 0.0,
                 generator: torch.Generator | None = None):
    """Propagate a batch and score every prompted instance in it.

    Every frame is scored -- there is no prompted frame to exclude the way
    `clip_losses` excludes frame 0. That looks like the model being handed the
    answer, and it is not: a box says *where*, and the target is the mask,
    which is the whole task. SAM 2's image pretraining is this and nothing
    else.

    **`prompt` defaults to `box`, which is what every run before
    `12_encoder_probe.ipynb` used, and the probe is why it is now a knob.**
    A loop that only ever sees an exact ground-truth rectangle produces a
    checkpoint that is worse than stock EdgeTAM the moment the rectangle is
    loose -- measured, not supposed. `mix` is the setting that addresses it;
    `mix_boxes` carries the numbers and the argument.

    **`anchor` is the answer to what stage A costs if stage B undoes it.**
    Pretraining moves the encoder over a large unlabelled set; stage B then
    trains it on a labelled set two orders of magnitude smaller, and nothing
    stops it drifting back. Passing a frozen copy of the model *as stage B
    started* -- the stage-A output -- adds a term penalising how far the
    encoder's features have travelled from it, which is feature distillation
    used as a regulariser rather than as a teacher.

    The anchor is the model itself and not a foundation model, deliberately:
    what is worth preserving is what stage A actually learned, the frozen copy
    is 4.92 M parameters rather than 300 M, and the features come out already
    in the student's own space so no projection is needed. It also costs no
    extra encoder pass -- `propagate_image` hands back the map it computed.
    """
    from .losses import Weights, instance_loss

    weights = weights or Weights()
    outputs = propagate_image(model, batch.images, batch.boxes, batch.valid,
                              prompt, jitter, generator)
    targets = batch.masks.reshape(-1, *batch.masks.shape[-2:])[outputs["rows"]]
    loss, terms = instance_loss(outputs, targets, weights)

    if anchor is not None and anchor_weight:
        from .distill import distill_loss, encoder_features

        with torch.no_grad():
            reference = encoder_features(anchor, batch.images)
        drift, _ = distill_loss(outputs["features"], reference)
        loss = loss + anchor_weight * drift
        terms["anchor"] = float(drift.detach())
    return loss, terms


@torch.no_grad()
def instance_iou(model, batch: ImageBatch, prompt: str = "box",
                 jitter: float = 0.0,
                 generator: torch.Generator | None = None) -> np.ndarray:
    """Per-instance mask IoU at the deployment threshold, as a flat array.

    The static stand-in for J&F. It is a proxy and should be read as one: it
    scores a single prompted frame, so it cannot see the failure this project
    exists to fix, which only appears once a memory bank is in the loop. What
    it *can* say is whether the encoder now separates one vehicle from the one
    beside it, which is the thing this stage is trying to buy.

    **`prompt` decides how much of the answer the score is given.** Under the
    default `box` the prompt states all four edges of the target, and on a
    large, isolated object that is most of the mask -- which is why two
    encoders of visibly different quality can land within a hundredth of each
    other here. `point` and `jitter` take that away; the same checkpoints
    scored all three ways is the comparison that says whether an encoder change
    was small or merely unmeasured.
    """
    outputs = propagate_image(model, batch.images, batch.boxes, batch.valid,
                              prompt, jitter, generator)
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
    """Windows for one split. Nothing else -- each sample decodes itself.

    A split used to carry the spec, the gates and the decomposition mode for
    everything in it, which made it a single-dataset object by construction.
    Those now travel on each `Sample`'s `Source`, so concatenating the windows
    of several datasets **is** the multi-dataset split:

        ImageSplit(kust4k_windows + vtuav_windows + segfly_windows)

    and every batch drawn from it mixes them.
    """

    samples: Sequence[Sample]

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def sources(self) -> dict[str, int]:
        """`{source name: windows}` -- what a mixed split is actually made of."""
        counts: dict[str, int] = {}
        for sample in self.samples:
            key = sample.source.name if sample.source else "?"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))


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
        batch = collate(pool[:n], "cpu")
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
        lambda chunk, pool: collate(chunk, "cpu", pool),
        device=device, workers=workers, depth=depth,
    )
