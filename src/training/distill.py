"""Modality distillation: an RGB foundation model teaching a thermal encoder.

Stage A of the encoder plan, and the one route in it that needs **no labels at
all**. AnyThermal (arXiv 2602.06203) distils a DINOv2-class RGB visual
foundation model into a thermal encoder over registered RGB-T pairs: the
teacher looks at the RGB half, the student at the thermal half, and the student
is trained to reproduce what the teacher saw. No semantic map is involved, so
the instance-versus-semantic problem that governs stage B (`aerial.py`) does
not arise here at all.

**Why this fits EdgeTAM specifically.** The obvious pretraining route -- MAE,
which is how SAM 2's Hiera trunk was pretrained -- does not transfer: MAE masks
*tokens* and is built for a ViT, while EdgeTAM's trunk is a convolutional
RepViT with no token grid to mask. Feature distillation has no such
requirement. The teacher's architecture and the student's are unrelated by
construction; all that has to line up is a feature map, and a 1x1 projection
makes any two channel counts line up. So the open question the TODO raises --
"does AnyThermal's distillation target attach to a RepViT trunk?" -- is
answered by the shape of the method: yes, because the student side of a
distillation loss is architecture-free.

**What is distilled, and where it lands.** The target is
`image_encoder`'s top-level output -- the [B, 256, S/16, S/16] tensor the mask
decoder actually consumes, which is also what the exported ONNX encoder graph
produces. Improving that tensor is improving the thing downstream depends on;
distilling into an intermediate trunk stage would improve something the
decoder only sees through layers that were never trained to pass it on.

**The projection is scaffolding.** `Projector` maps the student's 256 channels
to the teacher's dimension so a cosine loss can be taken, and is thrown away
when the stage ends. The checkpoint that comes out is an ordinary EdgeTAM state
dict -- same keys, same loader, same exporter -- with a differently-trained
trunk and neck inside it. Nothing downstream needs to know this stage happened.

**And it has to beat doing nothing.** The baseline is stage B run straight from
the stock EdgeTAM checkpoint. A pretraining stage that does not beat that is a
pretraining stage that cost GPU hours and bought nothing, and only the
comparison can say which it is.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .aerial import load_image, normalise
from .finetune import EMA, Rates, apply_freeze, param_groups

# The two teachers worth running, named rather than spelled out at each use.
# `TEACHER_ID` is what the pipeline runs unless told otherwise; `DINO_ID` is the
# second run and the default for `FeatureTeacher`, so a ViT class never defaults
# to a non-ViT checkpoint.
TEACHER_ID = "facebook/sam2.1-hiera-base-plus"
DINO_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"

# DINOv3's convolutional students, which are candidate *teachers* here: Meta
# distilled them out of the 7B ViT, so they carry that model's representation
# in a body whose geometry is the student's own -- a stage-wise convolutional
# trunk rather than a token grid. Sizes are the card's own parameter counts.
# All four are gated on the Hub, like every other `facebook/dinov3-*` id.
CONVNEXT_IDS = {
    "tiny":  "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",    #  29 M
    "small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",   #  50 M
    "base":  "facebook/dinov3-convnext-base-pretrain-lvd1689m",    #  89 M
    "large": "facebook/dinov3-convnext-large-pretrain-lvd1689m",   # 198 M
}
STUDENT_SIZE = 512        # what the student is trained at, and the size to match
SAM2_SIZE = 1024          # what SAM 2.1's position embeddings were trained at


def patch_aligned(size: int, patch: int) -> int:
    """The smallest multiple of `patch` that is at least `size`.

    A ViT teacher can only be fed a multiple of its patch size, and the useful
    default is "run it at the student's own input resolution, snapped up to its
    grid". That rule lands exactly on both families' natural numbers without a
    per-model table: 512 with DINOv3's 16-pixel patches is **512**, giving a
    32x32 grid that is precisely the student's grid at 512 -- no resampling of
    the teacher's map at all -- and 512 with DINOv2's 14 is **518**, which is
    the resolution DINOv2's own weights were trained at.
    """
    return -(-int(size) // int(patch)) * int(patch)

# Both families normalise with ImageNet statistics, which is what `normalise`
# already applies -- so a batch built for the student feeds either teacher
# unchanged. Worth stating because it is the one silent way a distillation run
# can be wrong: a teacher fed differently-scaled pixels produces plausible
# features for the wrong image.


# --------------------------------------------------------------------------
# Teachers
# --------------------------------------------------------------------------


def tokens_to_map(tokens: torch.Tensor, side: int) -> torch.Tensor:
    """`[B, N, C]` transformer tokens -> `[B, C, side, side]`.

    The **trailing** `side * side` tokens are taken, which drops whatever leads
    the sequence: a class token, and in some checkpoints registers as well.
    Counting the leading tokens instead would need a per-checkpoint constant
    and would shift the whole grid by one wherever that constant was wrong --
    a failure that produces a plausible feature map for a misaligned image.

    They are dropped rather than reshaped because a class token summarises the
    whole image, and what a dense encoder has to learn is *where* things are,
    which only the patch tokens say.
    """
    wanted = side * side
    count = int(tokens.shape[1])
    if count < wanted:
        raise RuntimeError(
            f"teacher returned {count} tokens, too few for a {side}x{side} grid.")
    tokens = tokens[:, count - wanted:]
    return tokens.reshape(tokens.shape[0], side, side, -1).permute(0, 3, 1, 2).float()


@torch.no_grad()
def probe_geometry(model, device: str = "cuda") -> tuple[int, int, int]:
    """`(stride, prefix tokens, channels)` of a frozen backbone, **measured**.

    Read off the config this goes wrong quietly. DINOv3's ConvNeXt checkpoints
    have no `patch_size` field at all -- their config carries `hidden_sizes`
    and `depths` and nothing about stride -- so a `getattr(config,
    "patch_size", 14)` returns 14 for a trunk whose real stride is 32, and the
    grid computed from it is off by more than a factor of two. That is exactly
    the shape of bug this repo has been bitten by before: it does not raise
    where it is wrong, it raises somewhere else, or worse, it does not raise.

    So it is derived from two forward passes instead. A backbone that returns a
    token sequence returns `k + (S/stride)^2` tokens for input side `S`, with
    `k` prefix tokens (DINOv3's ViTs put one class token and four registers
    there; its ConvNeXt puts one pooled token). Run it at 128 and at 256 and
    the unknown `k` cancels:

        n(256) - n(128) = (2g)^2 - g^2 = 3g^2      so   g = sqrt((n2 - n1) / 3)

    from which the stride is `128 / g` and `k` is `n1 - g^2`. A backbone that
    returns a map instead of tokens needs none of this -- the shape says it.

    Two forwards at 128 and 256 cost milliseconds and are run once per stage.
    """
    counts, channels = [], 0
    for side in (128, 256):
        blank = torch.zeros(1, 3, side, side, device=device, dtype=model.dtype)
        hidden = model(pixel_values=blank).last_hidden_state
        if hidden.dim() == 4:
            return side // int(hidden.shape[-1]), 0, int(hidden.shape[1])
        counts.append(int(hidden.shape[1]))
        channels = int(hidden.shape[-1])

    small, large = counts
    cells = (large - small) / 3.0
    side = int(round(cells ** 0.5))
    if side < 1 or abs(side * side - cells) > 0.51:
        raise RuntimeError(
            f"could not work out this backbone's stride from its token counts "
            f"({small} at 128, {large} at 256). It does not scale as a square "
            f"grid plus a fixed prefix, so `size` has to be given explicitly.")
    return int(round(128 / side)), small - side * side, channels


def teacher_input(student_size: int, stride: int, student_stride: int = 16) -> int:
    """The teacher's input side, chosen so its map is never *coarser* than the
    student's.

    One rule, and it is the one that makes a convolutional teacher usable at
    all. The student's map is stride 16, so at 512 it is 32x32 and each cell
    covers 16 source pixels. A ViT/16 teacher at 512 produces the same 32x32
    and the two correspond exactly. A ConvNeXt teacher has **stride 32**: fed
    the same 512 it produces 16x16, and `distill_loss` would then upsample it
    to the student's grid -- supervising four student cells with one teacher
    cell, which throws away half the spatial resolution the student is being
    trained to produce. On aerial imagery, where a target is twenty pixels
    across, that is the resolution that matters.

    Feeding that teacher 1024 instead gives it a 32x32 map over the *same*
    field of view: exact correspondence, at the cost of a forward pass on an
    upsampled image. Under `no_grad` and bfloat16 that cost is small next to
    the student's backward, and it is paid once per batch.

    Teachers at or below the student's stride keep the existing rule --
    `patch_aligned` -- because there the teacher is the finer of the two and
    downsampling it to the student's grid loses nothing the student could have
    represented anyway.
    """
    if stride <= student_stride:
        return patch_aligned(student_size, stride)
    return (student_size // student_stride) * stride


class FeatureTeacher:
    """A frozen ViT foundation model (DINOv3, DINOv2 and friends), dense.

    Thin on purpose: everything version-sensitive about `transformers` lives in
    `features`, and everything this module reasons about -- the projection, the
    loss, the loop -- is independent of it.

    **DINOv3 is the default *here*, and the second run overall.** What this
    stage copies is a *dense* feature map, position by position, and dense
    feature quality is exactly what DINOv3's release is about -- its predecessor
    is documented to degrade in dense prediction as training scales, which is
    the failure DINOv3 set out to fix. Its 16-pixel patch is a second, smaller
    win: at the student's own 512 input it produces a 32x32 grid, the student's
    grid exactly, so the teacher's map is never resampled before the loss.

    What it is *not* is the pipeline default any more. The task is single-object
    tracking with no class anywhere in it -- the operator draws a box, the model
    follows that object -- and semantics is the one thing that task never asks
    for. `Sam2FeatureTeacher` says the rest.

    The DINOv3 weights are **gated** on the Hub -- accept the terms once on the
    model page and set `HF_TOKEN` -- and `build_teacher` says so rather than
    letting the download fail with a bare 401. `facebook/dinov2-base` is
    ungated and remains one string away.
    """

    def __init__(self, model_id: str = DINO_ID, device: str = "cuda",
                 dtype: str = "bfloat16", size: int | None = None) -> None:
        from transformers import AutoModel

        self.model_id = model_id
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_id, dtype=getattr(torch, dtype)).to(device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        # Measured rather than read: see `probe_geometry` for the checkpoint
        # whose config says nothing about its stride and the failure that
        # caused. `dim` still prefers the config where it has one, because a
        # named field is clearer in a log than a number that fell out of a
        # dummy forward -- but the probe is the fallback and the two are
        # checked against each other.
        self.patch, self.prefix, probed_dim = probe_geometry(self.model, device)
        config = self.model.config
        stated = int(getattr(config, "hidden_size", 0)
                     or (list(getattr(config, "hidden_sizes", None) or [0]))[-1])
        self.dim = stated or probed_dim
        if stated and probed_dim and stated != probed_dim:
            raise RuntimeError(
                f"{model_id}: config says {stated} channels, the model returns "
                f"{probed_dim}. One of the two is not describing the tensor "
                f"`features` will hand to the loss.")
        # Resolved after the geometry is known, so the rule is one rule rather
        # than a constant per checkpoint.
        self.size = teacher_input(size or STUDENT_SIZE, self.patch)

    @property
    def grid(self) -> int:
        """The side of the feature map this teacher produces at `self.size`."""
        return self.size // self.patch

    @torch.no_grad()
    def features(self, images: torch.Tensor) -> torch.Tensor:
        """`[B, D, h, w]` for ImageNet-normalised RGB `[B, 3, S, S]`."""
        resized = F.interpolate(images, size=(self.size, self.size),
                                mode="bilinear", align_corners=False)
        hidden = self.model(pixel_values=resized.to(self.model.dtype)).last_hidden_state
        if hidden.dim() == 4:
            # Some backbones hand back the map directly as [B, C, h, w]: no
            # prefix token to strip and no grid to reshape.
            return hidden.float()
        # Everything else -- including DINOv3's **ConvNeXt** checkpoints, which
        # flatten their final stage and prepend a pooled token before the final
        # LayerNorm, so they arrive shaped exactly like a ViT's output --
        # reshapes back to a grid. `tokens_to_map` drops the prefix by taking
        # the last h*w tokens, which is why one path covers a class token, four
        # registers and a single pooled token alike.
        return tokens_to_map(hidden, self.grid)


    @torch.no_grad()
    def maps(self, images: torch.Tensor, layers: int = 1) -> list[torch.Tensor]:
        """The last `layers` blocks as feature maps, deepest last.

        One student map aligned against the teacher's **last two** blocks
        rather than only its last is the one alignment ablation in EdgeCrafter
        (arXiv 2603.18739) that moved its number without moving anything else:
        +0.7 COCO AP for its compact student, and it reports that going to
        three *regresses* for larger students. So `layers=2` is worth a run and
        `layers=3` is worth nothing until someone measures it here.

        The loss over several maps is a sum against the *same* projected
        student, not a concatenation -- that is what makes it a constraint
        rather than extra capacity, and it is how the paper writes it.

        Only implemented where the intermediate states are the same shape as
        the final one. DINOv3's ConvNeXt exposes its four *stages*, which sit
        at strides 4, 8, 16 and 32 and are not adjacent blocks in the sense
        this is asking for; rather than quietly average a stride-4 map into the
        target, it says so.
        """
        if layers <= 1:
            return [self.features(images)]
        resized = F.interpolate(images, size=(self.size, self.size),
                                mode="bilinear", align_corners=False)
        out = self.model(pixel_values=resized.to(self.model.dtype),
                         output_hidden_states=True)
        states = out.hidden_states
        if not states:
            raise RuntimeError(
                f"{self.model_id} returned no hidden_states, so layers>1 is not "
                f"available for it. Use layers=1.")
        final = out.last_hidden_state
        usable = [h for h in states[-layers:]
                  if h.dim() == final.dim() and h.shape[1] == final.shape[1]]
        if len(usable) < layers:
            raise RuntimeError(
                f"{self.model_id}: only {len(usable)} of its last {layers} "
                f"hidden states have the final layer's shape "
                f"{tuple(final.shape)}. This backbone exposes stages at "
                f"different strides rather than adjacent blocks -- use "
                f"layers=1 for it.")
        return [tokens_to_map(h, self.grid) if h.dim() == 3 else h.float()
                for h in usable]


class Sam2FeatureTeacher:
    """SAM 2.1's own image encoder as the teacher. **The default.**

    The question this used to be an answer to -- "this is a SAM-family model,
    so why distil an *unrelated* RGB model into it?" -- turns out to have no
    good answer, so the default moved. Four things are in its favour:

    1. **It is the task.** What is being built is single-object tracking with
       no class in it: the operator draws a box and the model follows *that*
       object, and nothing anywhere asks what the object is. SAM 2's features
       are class-agnostic by construction -- "something is here, it ends
       there" -- which is that task's definition rather than a limitation of
       the teacher. DINOv3's semantics are real and are surplus here.
    2. **The tensors match exactly.** `fpn_hidden_states[-1]` is the FPN neck's
       coarsest output at 256 channels and stride 16 -- the same kind of tensor
       `encoder_features` pulls out of the student, so the projection is 256 to
       256 rather than a change of representation.
    3. **It continues the objective the checkpoint was born from.** EdgeTAM
       *is* a distillation of SAM 2. Asking it to produce SAM 2's RGB features
       from a thermal input is the same training objective it already had, with
       a modality gap added.
    4. **It pushes the encoder toward the space the memory path expects**,
       not away from it, so it shrinks the alignment risk stage C exists to
       repair rather than growing it. A DINOv3 run carries that risk instead.

    The one thing against is cost, and it is small: see below. DINOv3 stays one
    `--teacher` string away as the second run, with everything else -- data,
    seed, schedule, loss -- held fixed, so the difference between the two runs
    is the teacher.

    **Use base-plus, not large, and the reason is not cost.** EdgeTAM's own
    paper says its teacher was **SAM2-Hiera-B+**, and its distillation target
    was **F16, the image-encoder feature map at stride 16** -- the very tensor
    this class pulls out of `fpn_hidden_states[-1]`. EdgeTAM's trunk weights
    *are* the result of fitting Hiera-B+'s F16. Distilling from B+ resumes that
    objective with a modality gap added; distilling from Large asks the student
    to move to a *different* target than the one it was initialised against,
    which forfeits the single property that makes a SAM teacher interesting
    next to DINOv3. (Whether EdgeTAM used the 2.0 or 2.1 B+ checkpoint is not
    stated -- the paper names "SAM2-HieraB+" and points only at the repo root.)

    Cost is not what decides it at this scale: 600 steps at batch 32 is 0.037
    A100-hours for B+ against 0.104 for Large. Both trivial. The 2.8x per step
    is real but is not the argument.

    Note the input size. It **is** interpolated -- `Sam2HieraDetModel`
    bicubic-resizes `pos_embed` to the feature grid, so 512 runs -- but the
    features still move a long way: cosine similarity to the same model's 1024
    output is **0.842 for base-plus and 0.736 for Large** at 512, and 0.704 /
    0.512 at 256. So lowering `size` is a change to the teacher, not just to
    its cost, and Large degrades faster. The hard constraint is elsewhere: the
    window embedding is tiled by integer division, so the input must be a
    **multiple of 32** -- 500 raises, 512 does not.
    """

    def __init__(self, model_id: str = TEACHER_ID,
                 device: str = "cuda", dtype: str = "bfloat16",
                 size: int = SAM2_SIZE) -> None:
        from transformers import Sam2Model

        self.model_id = model_id
        self.device = device
        self.size = size
        self.patch = 16                      # the FPN's coarsest stride
        # Loaded through the full model and then narrowed: the checkpoint's
        # keys are prefixed `vision_encoder.`, and the prompt encoder and mask
        # decoder are dropped rather than carried on the card unused.
        full = Sam2Model.from_pretrained(model_id, dtype=getattr(torch, dtype))
        self.model = full.vision_encoder.to(device).eval()
        self.dim = int(full.config.vision_config.fpn_hidden_size)
        del full
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def features(self, images: torch.Tensor) -> torch.Tensor:
        """`[B, 256, S/16, S/16]` for ImageNet-normalised RGB `[B, 3, S, S]`.

        `fpn_hidden_states` is ordered high resolution to low, so the last
        entry is the stride-16 map -- the level the mask decoder reads and the
        one the student's `vision_feats[-1]` corresponds to.
        """
        resized = F.interpolate(images, size=(self.size, self.size),
                                mode="bilinear", align_corners=False)
        out = self.model(pixel_values=resized.to(self.model.dtype))
        maps = out.fpn_hidden_states
        if maps is None:
            raise RuntimeError(
                f"{self.model_id}: the vision encoder returned no fpn_hidden_states. "
                "This transformers version structures SAM 2's output differently; "
                "see src/training/distill.py:Sam2FeatureTeacher.")
        return maps[-1].float()


def build_teacher(model_id: str = TEACHER_ID, device: str = "cuda",
                  dtype: str = "bfloat16", size: int | None = None):
    """The right teacher class for `model_id`, at the right input size.

    Dispatching on the id rather than on a separate flag keeps the two from
    disagreeing -- a SAM 2 checkpoint run at a ViT's 518 would load, run, and
    quietly produce features from interpolated position embeddings it was never
    trained with. A ViT teacher's size is resolved from its own patch size
    (`patch_aligned`), so no default has to be remembered per checkpoint.
    """
    try:
        if "sam2" in model_id.lower():
            return Sam2FeatureTeacher(model_id, device, dtype, size or SAM2_SIZE)
        return FeatureTeacher(model_id, device, dtype, size)
    except OSError as exc:
        # The Hub answers a gated repo with a 401/403 whose message is about
        # authentication, which reads like a broken token rather than terms
        # nobody accepted yet. DINOv3 is gated; say so at the point of failure.
        raise SystemExit(
            f"could not load the teacher {model_id!r}.\n\n"
            f"If it is a gated repository -- every facebook/dinov3-* checkpoint "
            f"is -- open its model page once, accept the terms, then set a "
            f"token:\n"
            f"    from huggingface_hub import login; login()   # or export HF_TOKEN\n\n"
            f"An ungated alternative that needs no account: "
            f"--teacher facebook/dinov2-base\n\n"
            f"original error: {exc}") from exc


def teacher_maps(teacher, images: torch.Tensor,
                 layers: int = 1) -> list[torch.Tensor]:
    """`layers` maps from any teacher, falling back to one where it has to.

    `Sam2FeatureTeacher` reads an FPN level rather than a block, so "the last
    two layers" has no meaning for it and asking is a mistake worth naming
    rather than silently answering with one map twice.
    """
    if layers <= 1:
        return [teacher.features(images)]
    if not hasattr(teacher, "maps"):
        raise ValueError(
            f"{type(teacher).__name__} exposes one feature map, not blocks -- "
            f"layers={layers} has no meaning for it. Use layers=1.")
    return teacher.maps(images, layers)


class Projector(nn.Module):
    """1x1 convolution from the student's channels to the teacher's.

    Scaffolding, not architecture. It exists because a cosine loss needs two
    vectors of the same length and the two models were trained by different
    people, and it is discarded before the checkpoint is written. Keeping it
    would mean shipping a tensor no inference path ever calls.
    """

    def __init__(self, student_dim: int, teacher_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(student_dim, teacher_dim, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)


# --------------------------------------------------------------------------
# Features and loss
# --------------------------------------------------------------------------


def encoder_features(model, images: torch.Tensor) -> torch.Tensor:
    """The student's top-level feature map, `[B, C, H, W]`, under autograd.

    The same tensor `track_step` hands to the SAM head and the same one
    `tools/export_image_encoder_onnx.py` exports, reshaped back from SAM 2's
    flattened HWxNxC convention. Distilling into anything else would be
    distilling into a tensor the deployment does not use.
    """
    backbone = model.forward_image(images)
    _, feats, _, feat_sizes = model._prepare_backbone_features(backbone)
    flat = feats[-1]
    height, width = feat_sizes[-1]
    return flat.permute(1, 2, 0).reshape(flat.shape[1], flat.shape[2], height, width)


def moment_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Per-channel mean and spread **across positions**, matched.

    An anti-collapse term. Per-position cosine constrains each position's
    direction and says nothing about how the channels are *used across the
    map*: a student whose channel is nearly constant everywhere can still score
    well position by position, and that channel then carries no information for
    the decoder to read. Matching each channel's mean and standard deviation
    over the map penalises exactly that.

    **Deliberately computed on the normalised maps, unlike the published
    version.** arXiv 2604.27128's scale term matches *raw* feature magnitudes,
    which is right when the student's output is the deployed tensor. Here it is
    not: the student's raw scale is what EdgeTAM's own mask decoder was trained
    to read, the projector standing between student and teacher is discarded at
    the end of the stage, and so pulling the student's magnitude towards a
    foundation model's would be matching scaffolding at the decoder's expense.
    What survives normalisation -- which channels do work, and where -- is the
    part worth copying.
    """
    dims = (0, 2, 3)
    return ((student.mean(dims) - teacher.mean(dims)).pow(2).mean()
            + (student.std(dims) - teacher.std(dims)).pow(2).mean())


def distill_loss(student: torch.Tensor, teacher: torch.Tensor,
                 moment_weight: float = 0.0,
                 tolerance: int = 0) -> tuple[torch.Tensor, dict[str, float]]:
    """Cosine distance between two dense feature maps, position by position.

    Cosine and not L2 because the two models' features live on different
    scales, and matching the scale is not the point -- what transfers is
    *direction*: which positions the teacher considers similar to which. An L2
    term would spend most of the gradient making the norms agree.

    **Direction first, everything else second, and the order is load-bearing.**
    The published recipe closest to this one (arXiv 2604.27128, distilling SAM
    3's encoder into a small student) reports a *collapsed-variance failure
    mode* when its scale term is given weight comparable to the directional
    ones: the network converges on a solution reproducing the teacher's
    variance without aligning with its direction. So `moment_weight` is a small
    number beside an implicit 1.0 on cosine, and it defaults to **off** -- that
    paper chose its weights on pilot runs and reports no ablation, and our
    moment term is not even the same one (see `moment_loss`). Treat it as a
    hypothesis to test on the held-out split, not a setting to trust.

    The teacher is resized to the student's grid, not the other way round. The
    student's resolution is fixed by the deployment (S/16), and upsampling the
    student to compare would supervise interpolated positions that the model
    never actually produces.

    **`tolerance` is for pairs that are not perfectly registered.** At zero,
    student position *p* is matched against teacher position *p* and nothing
    else -- correct when the two cameras are aligned to the pixel, and quietly
    wrong when they are not, because then the teacher was looking somewhere the
    student was not and every position carries a systematic error. At
    `tolerance=1` each student position is scored against the *best* teacher
    position in the 3x3 neighbourhood around it, which absorbs a misalignment
    of up to one feature cell -- 16 source pixels at stride 16. VTUAV's authors
    note its modalities are not perfectly aligned; this is the remedy, and it
    is off by default because it costs real spatial precision: a model free to
    match a neighbour is a model that need not place a boundary exactly.
    """
    if student.shape[-2:] != teacher.shape[-2:]:
        teacher = F.interpolate(teacher, size=student.shape[-2:],
                                mode="bilinear", align_corners=False)
    student = F.normalize(student.float(), dim=1)
    teacher = F.normalize(teacher.float(), dim=1)

    if tolerance > 0:
        # Every offset in the neighbourhood at once: unfold lays the teacher's
        # (2r+1)^2 shifted copies along a new axis, the dot product is then one
        # einsum, and the best offset per position is a max. Cheaper and
        # clearer than a loop over shifts, and the max is what makes the term
        # "match something nearby" rather than "match exactly here".
        width = 2 * tolerance + 1
        batch, channels, height, columns = student.shape
        patches = F.unfold(teacher, kernel_size=width, padding=tolerance)
        patches = patches.reshape(batch, channels, width * width, height * columns)
        flat = student.reshape(batch, channels, 1, height * columns)
        similarity = (flat * patches).sum(dim=1).amax(dim=1)
        cosine = 1.0 - similarity.reshape(batch, height, columns)
    else:
        cosine = 1.0 - (student * teacher).sum(dim=1)

    total = cosine.mean()
    terms = {"cosine": float(total.detach())}
    if moment_weight:
        moments = moment_loss(student, teacher)
        total = total + moment_weight * moments
        terms["moments"] = float(moments.detach())
    return total, terms


# --------------------------------------------------------------------------
# Pairs
# --------------------------------------------------------------------------


@dataclass
class PairBatch:
    """Registered thermal and RGB halves of the same scene. No labels."""

    thermal: torch.Tensor          # [B, 3, S, S], the grey channel replicated
    rgb: torch.Tensor              # [B, 3, S, S]

    def __len__(self) -> int:
        return int(self.thermal.shape[0])

    def to(self, device: str) -> "PairBatch":
        return PairBatch(thermal=self.thermal.to(device, non_blocking=True),
                         rgb=self.rgb.to(device, non_blocking=True))


def shared_window(shape: tuple[int, int], place: tuple[float, float],
                  scale: float) -> tuple[tuple[int, int], tuple[int, int]]:
    """`(origin, window)` for a square crop placed in *normalised* coordinates.

    Normalised and not pixel coordinates because the two halves of a registered
    pair are not always the same size -- and the crop has to land on the same
    piece of the world in both, or the distillation is matching a teacher that
    looked somewhere else. `place` is one `(u, v)` in [0, 1) drawn once per
    pair and reused for both halves; `scale` is the side as a fraction of the
    frame's shorter axis.

    This is exact when the two halves share a field of view, which is what
    "registered" means in these datasets. It is *not* exact for a pair whose
    two cameras were cropped differently before publication -- there is no
    geometry here that could recover that, and `crop=None` is the safe choice
    if you are unsure.
    """
    height, width = shape
    side = max(int(round(min(height, width) * float(scale))), 1)
    side = min(side, height, width)
    x0 = int(round(float(place[0]) * (width - side)))
    y0 = int(round(float(place[1]) * (height - side)))
    return (x0, y0), (side, side)


class Pair(NamedTuple):
    """One registered pair, carrying how *its own* dataset must be read.

    A plain `(thermal, rgb)` tuple was enough while stage A drew from one
    dataset. It stops being enough the moment two are mixed, because the two
    need opposite treatment: VTUAV is 1920x1080 and has to be cropped to keep
    native pixels, DroneVehicle is already 640x512 and cropping it would throw
    resolution away; DroneVehicle carries a 100 px white border and VTUAV none.
    One `crop` and one `border` for the whole batch cannot express that, and
    the failure is silent -- the wrong one gets resized into a square, or the
    teacher reads a margin.

    Still a tuple, so anything that only wants `pair[0]` and `pair[1]` keeps
    working unchanged.
    """

    thermal: Path
    rgb: Path
    crop: float | None = None
    border: int = 0


def sources(specs, size: int = 512, crop: float | str | None = "auto") -> list[Pair]:
    """Every registered pair across several datasets, each tagged with its own
    reading parameters.

    `specs` is an iterable of `(DatasetSpec, root)`. `crop="auto"` derives the
    fraction per source from the picture that is actually there -- `size`
    divided by the shorter side once the border is off, or no crop at all when
    the source is already at or below `size`. That is the number that keeps
    native pixels, and deriving it removes the most error-prone hand-set knob
    in stage A: set it too high on a 1920x1080 source and every small target is
    resampled away before training sees it.
    """
    import cv2

    from .aerial import list_pairs

    out: list[Pair] = []
    for spec, root in specs:
        found = list_pairs(root, spec)
        if not found:
            continue
        fraction = crop
        if crop == "auto":
            probe = cv2.imread(str(found[0][0]), cv2.IMREAD_UNCHANGED)
            if probe is None:
                raise FileNotFoundError(f"Could not read {found[0][0]}")
            _, (width, height) = spec.inset(int(probe.shape[0]), int(probe.shape[1]))
            shorter = min(height, width)
            fraction = size / shorter if shorter > size else None
        out += [Pair(t, r, fraction, spec.border) for t, r in found]
    return out


def collate_pairs(pairs: Sequence[tuple[Path, Path]], size: int = 512,
                  device: str = "cpu", executor=None,
                  crop: float | None = None,
                  rng: np.random.Generator | None = None,
                  border: int = 0) -> PairBatch:
    """Read a batch of registered pairs at `size`, both halves normalised alike.

    `crop=None` resizes the whole frame, which is right for a source already
    near the model's input size -- Kust4K's 640x512 loses almost nothing.

    `crop=<fraction>` takes a square window instead, at the **same normalised
    position in both halves** so the registration survives. This is what a
    high-resolution source needs: squeezing a 1920x1080 VTUAV frame into
    512x512 both distorts the aspect ratio and shrinks a small vehicle below
    the size the encoder will ever see it at, so the encoder would be
    pretrained on image statistics the deployment never produces. Pass
    `size / min(height, width)` for native pixels -- 512/1080 = 0.474 on
    VTUAV, 1.0 or None on a 640x512 source.
    """
    import cv2

    mapper = map if executor is None else executor.map
    rng = rng or np.random.default_rng()
    # One placement per pair, drawn here and used by both halves.
    wants_place = crop or any(isinstance(p, Pair) and p.crop for p in pairs)
    places = (rng.random((len(pairs), 2)) if wants_place
              else np.zeros((len(pairs), 2)))

    def read(args):
        path, gray, place, this_crop, this_border = args
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        # Everything below works inside whatever padding the dataset ships, so
        # a crop can never land on the margin. DroneVehicle pads both halves
        # with 100 px of white; a border of 0 leaves the frame untouched.
        height = int(raw.shape[0]) - 2 * this_border
        width = int(raw.shape[1]) - 2 * this_border
        if height <= 0 or width <= 0:
            raise ValueError(
                f"border={this_border} leaves nothing of {path} "
                f"({raw.shape[1]}x{raw.shape[0]})")
        if this_crop:
            origin, window = shared_window((height, width), place, this_crop)
        else:
            origin, window = (0, 0), (width, height)
        return load_image(path, (origin[0] + this_border, origin[1] + this_border),
                          window, size, gray)

    # A `Pair` carries its own; a plain `(thermal, rgb)` tuple falls back to
    # the call's values and so behaves exactly as it always did. A `Pair` whose
    # crop is None means *no crop* and is honoured, not overridden.
    settings = [(p.crop, p.border) if isinstance(p, Pair) else (crop, border)
                for p in pairs]
    thermal = list(mapper(read, [(p[0], True, q, c, b)
                                 for p, q, (c, b) in zip(pairs, places, settings)]))
    rgb = list(mapper(read, [(p[1], False, q, c, b)
                             for p, q, (c, b) in zip(pairs, places, settings)]))
    return PairBatch(thermal=normalise(np.stack(thermal), device),
                     rgb=normalise(np.stack(rgb), device))


def subsample(pairs: Sequence, count: int | None,
              seed: int = 0) -> list:
    """`count` pairs spread across the whole set, not the first `count` of it.

    Truncating a sorted list is fine for a set of independent stills and wrong
    for one derived from video, which is most of them: VTUAV is 500 sequences
    of ~3 400 frames, so the first 5 000 pairs are **two flights**. A run
    capped that way trains on two scenes and reports it as five thousand
    samples. Spreading the sample also drops the near-duplicate neighbours that
    make consecutive video frames nearly the same image.
    """
    pairs = list(pairs)
    if count is None or count >= len(pairs):
        return pairs
    order = np.random.default_rng(seed).permutation(len(pairs))[:count]
    return [pairs[int(i)] for i in sorted(order)]


def stream(pairs: Sequence[tuple[Path, Path]], batch: int, size: int = 512,
           seed: int | None = None, limit: int | None = None,
           device: str = "cuda", workers: int = 8, depth: int = 2,
           crop: float | None = None, border: int = 0):
    """Shuffled, prefetched pair batches -- same threading as every other mode.

    The crop placement is drawn from a generator seeded per epoch, so a run is
    reproducible from its seed the way the clip and image modes are: the same
    pairs in the same order, cropped in the same places.
    """
    from .loader import batch_clips, prefetch_with

    rng = np.random.default_rng(seed)
    chunks = batch_clips(pairs, batch, seed=seed, limit=limit)
    return prefetch_with(
        chunks,
        lambda chunk, pool: collate_pairs(chunk, size, "cpu", pool, crop, rng, border),
        device=device, workers=workers, depth=depth,
    )


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def pretrain(
    model,
    pairs: Sequence[tuple[Path, Path]],
    teacher: FeatureTeacher,
    *,
    save,
    size: int = 512,
    epochs: int = 1,
    batch: int = 8,
    steps_per_epoch: int = 400,
    rates: Rates = Rates(neck=1e-4, trunk=5e-5),
    projector_lr: float = 1e-3,
    moment_weight: float = 0.0,
    freeze=apply_freeze,
    crop: float | None = None,
    border: int = 0,
    tolerance: int = 0,
    grad_clip: float = 1.0,
    ema_decay: float = 0.999,
    workers: int = 8,
    depth: int = 2,
    seed: int = 0,
    device: str = "cuda",
    progress=None,
    log=print,
) -> dict:
    """Distil `teacher`'s RGB features into `model`'s thermal encoder.

    A loop of its own rather than a `schedule.Loop`, because this stage differs
    in more than the batch: there is one stage and not two, the optimiser
    carries a parameter the model does not own (the projector), and there is no
    validation set to select a checkpoint on -- the objective is a proxy, and
    the number that decides whether this stage was worth running is stage B's,
    measured against the same stage B without it. Pretending otherwise by
    reusing `run_stages` would hide all three.

    `freeze` takes the same shape as everywhere else, so LoRA distillation is
    `lora.inject(model, "backbone"); freeze=lora.freeze` and nothing else
    changes.
    """
    from .finetune import summarise_freeze

    counts = freeze(model, "backbone")
    log(summarise_freeze(counts, model))

    student_dim = encoder_features(
        model, torch.zeros(1, 3, size, size, device=device)).shape[1]
    projector = Projector(student_dim, teacher.dim).to(device)
    log(f"projector {student_dim} -> {teacher.dim} "
        f"({sum(p.numel() for p in projector.parameters()) / 1e6:.2f} M, discarded after)")

    groups = param_groups(model, rates) + [
        {"params": list(projector.parameters()), "lr": projector_lr,
         "weight_decay": 0.0, "name": "projector"}]
    opt = torch.optim.AdamW(groups)
    trainable = [p for g in opt.param_groups for p in g["params"]]
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups],
        total_steps=max(epochs * steps_per_epoch, 1), pct_start=0.1)
    ema = EMA(model, decay=ema_decay)

    history = []
    for epoch in range(epochs):
        batches = stream(pairs, batch, size, seed=seed + 100 * epoch,
                         limit=steps_per_epoch, device=device, workers=workers,
                         depth=depth, crop=crop, border=border)
        if progress is not None:
            batches = progress(batches, total=steps_per_epoch, desc=f"distil e{epoch}")

        losses = []
        for pair in batches:
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.startswith("cuda")):
                target = teacher.features(pair.rgb)
                student = projector(encoder_features(model, pair.thermal))
            loss, terms = distill_loss(student, target, moment_weight, tolerance)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            opt.step()
            sched.step()
            ema.update(model)
            losses.append(float(loss.detach()))
            if hasattr(batches, "set_postfix"):
                batches.set_postfix(**{k: f"{v:.4f}" for k, v in terms.items()})

        mean = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch, "loss": mean})
        log(f"  epoch {epoch}: mean cosine distance {mean:.4f}")

    # The averaged weights, every time -- there is no held-out number here that
    # could have chosen between them, and on a stage this short the EMA is the
    # less noisy of the two by construction.
    with ema.applied(model):
        save(model, {"stage": "distill", "teacher": teacher.model_id,
                     "teacher_dim": teacher.dim, "teacher_size": teacher.size,
                     "pairs": len(pairs), "image_size": size, "crop": crop,
                     "tolerance": tolerance,
                     "epochs": epochs, "final_loss": history[-1]["loss"]})

    del projector, opt, sched, ema
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"history": history, "final_loss": history[-1]["loss"],
            "pairs": len(pairs), "steps": epochs * steps_per_epoch}
