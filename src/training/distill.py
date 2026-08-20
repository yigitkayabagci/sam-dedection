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

# DINOv2 patches are 14x14, so the teacher's input has to be a multiple of 14.
# 518 = 37 x 14 is the resolution its own weights were trained at.
TEACHER_ID = "facebook/dinov2-base"
TEACHER_SIZE = 518


# --------------------------------------------------------------------------
# Teacher
# --------------------------------------------------------------------------


class FeatureTeacher:
    """A frozen RGB foundation model, as a dense feature map.

    Thin on purpose: everything version-sensitive about `transformers` lives in
    `features`, and everything this module reasons about -- the projection, the
    loss, the loop -- is independent of it. Swapping the teacher is a string.
    """

    def __init__(self, model_id: str = TEACHER_ID, device: str = "cuda",
                 dtype: str = "bfloat16", size: int = TEACHER_SIZE) -> None:
        from transformers import AutoModel

        self.device = device
        self.size = size
        self.model = AutoModel.from_pretrained(
            model_id, dtype=getattr(torch, dtype)).to(device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.dim = int(self.model.config.hidden_size)
        self.patch = int(getattr(self.model.config, "patch_size", 14))

    @torch.no_grad()
    def features(self, images: torch.Tensor) -> torch.Tensor:
        """`[B, D, h, w]` for normalised RGB `[B, 3, S, S]`.

        The class token is dropped: it summarises the image, and what a dense
        encoder has to learn is *where* things are, which only the patch tokens
        carry.
        """
        resized = F.interpolate(images, size=(self.size, self.size),
                                mode="bilinear", align_corners=False)
        out = self.model(pixel_values=resized.to(self.model.dtype))
        tokens = out.last_hidden_state[:, 1:]              # drop CLS
        side = self.size // self.patch
        batch, count, dim = tokens.shape
        if count != side * side:
            # Some checkpoints carry extra register tokens after the class
            # token; take the trailing patch grid rather than guessing at the
            # count, and fail loudly if that does not divide either.
            if count < side * side:
                raise RuntimeError(
                    f"teacher returned {count} tokens for a {side}x{side} grid.")
            tokens = tokens[:, count - side * side:]
        return tokens.reshape(batch, side, side, dim).permute(0, 3, 1, 2).float()


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


def distill_loss(student: torch.Tensor, teacher: torch.Tensor,
                 l1_weight: float = 0.0) -> tuple[torch.Tensor, dict[str, float]]:
    """Cosine distance between two dense feature maps, position by position.

    Cosine and not L2 because the two models' features live on different
    scales, and matching the scale is not the point -- what transfers is
    *direction*: which positions the teacher considers similar to which. An L2
    term would spend most of the gradient making the norms agree.

    The teacher is resized to the student's grid, not the other way round. The
    student's resolution is fixed by the deployment (S/16), and upsampling the
    student to compare would supervise interpolated positions that the model
    never actually produces.
    """
    if student.shape[-2:] != teacher.shape[-2:]:
        teacher = F.interpolate(teacher, size=student.shape[-2:],
                                mode="bilinear", align_corners=False)
    student = F.normalize(student.float(), dim=1)
    teacher = F.normalize(teacher.float(), dim=1)

    cosine = 1.0 - (student * teacher).sum(dim=1)
    total = cosine.mean()
    terms = {"cosine": float(total.detach())}
    if l1_weight:
        l1 = F.smooth_l1_loss(student, teacher)
        total = total + l1_weight * l1
        terms["l1"] = float(l1.detach())
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


def collate_pairs(pairs: Sequence[tuple[Path, Path]], size: int = 512,
                  device: str = "cpu", executor=None) -> PairBatch:
    """Read a batch of registered pairs at `size`, both halves normalised alike.

    The whole frame is resized rather than cropped. A crop would be fine for
    the student and wrong for the pair: the two halves must show the *same*
    scene for the distillation to mean anything, and cropping each
    independently would silently break the registration the datasets went to
    trouble to provide.
    """
    mapper = map if executor is None else executor.map
    thermal_paths = [p[0] for p in pairs]
    rgb_paths = [p[1] for p in pairs]

    def read(args):
        path, gray = args
        import cv2

        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        height, width = raw.shape[:2]
        return load_image(path, (0, 0), (width, height), size, gray)

    thermal = list(mapper(read, [(p, True) for p in thermal_paths]))
    rgb = list(mapper(read, [(p, False) for p in rgb_paths]))
    return PairBatch(thermal=normalise(np.stack(thermal), device),
                     rgb=normalise(np.stack(rgb), device))


def stream(pairs: Sequence[tuple[Path, Path]], batch: int, size: int = 512,
           seed: int | None = None, limit: int | None = None,
           device: str = "cuda", workers: int = 8, depth: int = 2):
    """Shuffled, prefetched pair batches -- same threading as every other mode."""
    from .loader import batch_clips, prefetch_with

    chunks = batch_clips(pairs, batch, seed=seed, limit=limit)
    return prefetch_with(
        chunks,
        lambda chunk, pool: collate_pairs(chunk, size, "cpu", pool),
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
    l1_weight: float = 0.0,
    freeze=apply_freeze,
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
                         depth=depth)
        if progress is not None:
            batches = progress(batches, total=steps_per_epoch, desc=f"distil e{epoch}")

        losses = []
        for pair in batches:
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device.startswith("cuda")):
                target = teacher.features(pair.rgb)
                student = projector(encoder_features(model, pair.thermal))
            loss, terms = distill_loss(student, target, l1_weight)

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
        save(model, {"stage": "distill", "teacher": teacher.model.name_or_path,
                     "pairs": len(pairs), "image_size": size,
                     "epochs": epochs, "final_loss": history[-1]["loss"]})

    del projector, opt, sched, ema
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"history": history, "final_loss": history[-1]["loss"],
            "pairs": len(pairs), "steps": epochs * steps_per_epoch}
