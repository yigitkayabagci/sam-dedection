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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .aerial import load_image, normalise
from .finetune import EMA, Rates, apply_freeze, param_groups

TEACHER_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
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


class FeatureTeacher:
    """A frozen ViT foundation model (DINOv3, DINOv2 and friends), dense.

    Thin on purpose: everything version-sensitive about `transformers` lives in
    `features`, and everything this module reasons about -- the projection, the
    loss, the loop -- is independent of it.

    **DINOv3 is the default, and the reason is the thing being distilled.** What
    this stage copies is a *dense* feature map, position by position, and dense
    feature quality is exactly what DINOv3's release is about -- its predecessor
    is documented to degrade in dense prediction as training scales, which is
    the failure DINOv3 set out to fix. Its 16-pixel patch is a second, smaller
    win: at the student's own 512 input it produces a 32x32 grid, the student's
    grid exactly, so the teacher's map is never resampled before the loss.

    The DINOv3 weights are **gated** on the Hub -- accept the terms once on the
    model page and set `HF_TOKEN` -- and `build_teacher` says so rather than
    letting the download fail with a bare 401. `facebook/dinov2-base` is
    ungated and remains one string away.
    """

    def __init__(self, model_id: str = TEACHER_ID, device: str = "cuda",
                 dtype: str = "bfloat16", size: int | None = None) -> None:
        from transformers import AutoModel

        self.model_id = model_id
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_id, dtype=getattr(torch, dtype)).to(device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        config = self.model.config
        # ViT configs carry `hidden_size`; convolutional ones (DINOv3's
        # ConvNeXt students) carry `hidden_sizes` per stage, of which the last
        # is the map `features` returns.
        self.dim = int(getattr(config, "hidden_size", 0)
                       or list(getattr(config, "hidden_sizes", []))[-1])
        self.patch = int(getattr(config, "patch_size", 14))
        # Resolved after the config is known, so the rule is one rule rather
        # than a constant per checkpoint.
        self.size = patch_aligned(size or STUDENT_SIZE, self.patch)

    @torch.no_grad()
    def features(self, images: torch.Tensor) -> torch.Tensor:
        """`[B, D, h, w]` for ImageNet-normalised RGB `[B, 3, S, S]`."""
        resized = F.interpolate(images, size=(self.size, self.size),
                                mode="bilinear", align_corners=False)
        hidden = self.model(pixel_values=resized.to(self.model.dtype)).last_hidden_state
        if hidden.dim() == 4:
            # A convolutional teacher -- DINOv3 ships ConvNeXt students
            # distilled from its 7B ViT -- returns the map directly as
            # [B, C, h, w]: no class token to strip, no grid to reshape, and
            # conv-to-conv is the closer geometry for a RepViT student. Its
            # stride need not be 16; `distill_loss` resamples the teacher to
            # the student's grid either way.
            return hidden.float()
        return tokens_to_map(hidden, self.size // self.patch)


class Sam2FeatureTeacher:
    """SAM 2.1's own image encoder as the teacher, for the obvious question.

    The obvious question being: this is a SAM-family model, so why distil an
    unrelated RGB model into it rather than the thing it was distilled from?
    It is a fair question and it deserves a measurement, not an argument, so it
    is one `--teacher` string away. Three things are genuinely in its favour:

    1. **The tensors match exactly.** `fpn_hidden_states[-1]` is the FPN neck's
       coarsest output at 256 channels and stride 16 -- the same kind of tensor
       `encoder_features` pulls out of the student, so the projection is 256 to
       256 rather than a change of representation.
    2. **It continues the objective the checkpoint was born from.** EdgeTAM
       *is* a distillation of SAM 2. Asking it to produce SAM 2's RGB features
       from a thermal input is the same training objective it already had, with
       a modality gap added.
    3. Nothing about "SAM-ness" has to be argued about afterwards.

    And two against, which is why this is not simply the default:

    1. **The features are class-agnostic by design.** SAM 2 segments anything
       and deliberately carries no notion of *what* a region is; DINOv2's
       features are strongly semantic. What a thermal encoder is missing is
       arguably the semantics -- boundaries are often *easier* in thermal, a
       warm object against a cool background -- and if so the semantic teacher
       is the better one. Arguably. Nobody here has measured it.
    2. **It is much more expensive.** Hiera-Large at 1024 is the model EdgeTAM
       exists to avoid; it costs several times a DINOv2-base pass at 518, per
       step, for the whole stage. `facebook/sam2.1-hiera-base-plus` is the
       cheaper end of the same family.

    Note the input size: SAM 2.1's position embeddings were trained at 1024 and
    this does not interpolate them, so lowering `size` is a change to the
    teacher and not just to its cost.
    """

    def __init__(self, model_id: str = "facebook/sam2.1-hiera-large",
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


def collate_pairs(pairs: Sequence[tuple[Path, Path]], size: int = 512,
                  device: str = "cpu", executor=None,
                  crop: float | None = None,
                  rng: np.random.Generator | None = None) -> PairBatch:
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
    places = rng.random((len(pairs), 2)) if crop else np.zeros((len(pairs), 2))

    def read(args):
        path, gray, place = args
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        shape = (int(raw.shape[0]), int(raw.shape[1]))
        if crop:
            origin, window = shared_window(shape, place, crop)
        else:
            origin, window = (0, 0), (shape[1], shape[0])
        return load_image(path, origin, window, size, gray)

    thermal = list(mapper(read, [(p[0], True, q) for p, q in zip(pairs, places)]))
    rgb = list(mapper(read, [(p[1], False, q) for p, q in zip(pairs, places)]))
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
           crop: float | None = None):
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
        lambda chunk, pool: collate_pairs(chunk, size, "cpu", pool, crop, rng),
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
                         depth=depth, crop=crop)
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
