"""The training loop itself, so two methods can be compared rather than two loops.

A partial fine-tune and a LoRA run differ in exactly two places: which
parameters are unfrozen, and how the checkpoint is written (LoRA merges a copy).
Everything else -- the two stages, the batch order, the one-cycle schedule, the
gradient clipping, the EMA, which validation clips are scored and when the best
checkpoint is kept -- has to be *identical*, or the comparison measures the
difference between two notebooks instead of the difference between two methods.

So the loop lives here once and takes those two things as callbacks. The
notebook that explains it and the CLI that runs it twice execute the same code.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from .antiuav import Clip
from .clip_loop import clip_losses
from .finetune import EMA, Rates, param_groups, summarise_freeze
from .loader import batch_clips, prefetch


@dataclass(frozen=True)
class Split:
    """Clips and their mask stores, for one split."""

    clips: Sequence[Clip]
    stores: Mapping[str, Mapping[int, np.ndarray]]


@dataclass(frozen=True)
class Schedule:
    """Everything about the run that is not the model or the data.

    The defaults are the ones the notebook documents. `stages` is a list of
    `(stage name, epochs, Rates)`: the head alone first, so the mask decoder
    and the object-score head adapt to thermal statistics before the features
    under them move, then the encoder at a tenth of the rate.
    """

    stages: tuple[tuple[str, int, Rates], ...] = (
        ("head", 1, Rates(head=1e-4)),
        ("encoder", 2, Rates(head=5e-5, neck=5e-5, trunk=1e-5)),
    )
    batch: int = 4
    accum: int = 1
    steps_per_epoch: int = 400
    val_batches: int = 24
    workers: int = 8
    depth: int = 2
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    seed: int = 0
    meta: dict = field(default_factory=dict)


def validate(
    model,
    split: Split,
    schedule: Schedule,
    device: str = "cuda",
) -> float:
    """Mean clip loss over a fixed slice of the split.

    Fixed because the seed is fixed: the same clips every epoch, so a change in
    the number is the model and not the sample. The slice is bounded *while
    generating*, which is the whole point -- scoring the entire validation set
    every epoch is hours nobody asked for.
    """
    chunks = batch_clips(split.clips, schedule.batch, seed=1,
                         limit=schedule.val_batches)
    losses = []
    with torch.no_grad():
        for batch in prefetch(chunks, split.stores, device,
                              workers=schedule.workers, depth=schedule.depth):
            with _autocast(device):
                losses.append(float(clip_losses(model, batch)[0]))
    return float(np.mean(losses)) if losses else float("nan")


def _autocast(device: str):
    return torch.autocast("cuda", dtype=torch.bfloat16,
                          enabled=device.startswith("cuda"))


def run_stages(
    model,
    train: Split,
    val: Split,
    schedule: Schedule = Schedule(),
    *,
    freeze: Callable[[object, str], dict[str, int]],
    save: Callable[[object, dict], None],
    device: str = "cuda",
    progress: Callable | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Train `model` through `schedule.stages`; return the run's history.

    `freeze(model, stage)` decides what moves -- `finetune.apply_freeze` for a
    partial fine-tune, `lora.freeze` for LoRA -- and returns per-root trainable
    counts. `save(model, meta)` writes whichever checkpoint that method should
    produce, and is called only when the validation loss improves, inside the
    EMA context so the averaged weights are the ones stored.

    The EMA is rebuilt per stage on purpose: it averages the trainable
    parameters, and that set changes the moment the encoder unfreezes.
    """
    best, history = float("inf"), []

    for stage, epochs, rates in schedule.stages:
        counts = freeze(model, stage)
        log(f"\n===== stage {stage!r}: {epochs} epoch(s) x "
            f"{schedule.steps_per_epoch} steps =====")
        log(summarise_freeze(counts, model))

        opt = torch.optim.AdamW(param_groups(model, rates))
        trainable = [p for g in opt.param_groups for p in g["params"]]
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[g["lr"] for g in opt.param_groups],
            total_steps=max(epochs * schedule.steps_per_epoch // schedule.accum, 1),
            pct_start=0.1)
        ema = EMA(model, decay=schedule.ema_decay)

        for epoch in range(epochs):
            chunks = batch_clips(train.clips, schedule.batch,
                                 seed=schedule.seed + 100 * epoch,
                                 limit=schedule.steps_per_epoch)
            stream = prefetch(chunks, train.stores, device,
                              workers=schedule.workers, depth=schedule.depth)
            if progress is not None:
                stream = progress(stream, total=schedule.steps_per_epoch,
                                  desc=f"{stage} e{epoch}")

            for step, batch in enumerate(stream):
                with _autocast(device):
                    loss, terms = clip_losses(model, batch)
                (loss / schedule.accum).backward()
                if (step + 1) % schedule.accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, schedule.grad_clip)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    ema.update(model)
                if hasattr(stream, "set_postfix"):
                    stream.set_postfix(loss=f"{float(loss):.3f}",
                                       **{k: f"{v:.2f}" for k, v in terms.items()})

            with ema.applied(model):
                score = validate(model, val, schedule, device)
                improved = score < best
                if improved:
                    best = score
                    save(model, {"stage": stage, "epoch": epoch, "val_loss": score,
                                 **schedule.meta})
            history.append({"stage": stage, "epoch": epoch, "val_loss": score,
                            "saved": improved})
            log(f"  epoch {epoch}: val clip loss {score:.4f}"
                f"{'  <- saved' if improved else ''}")

        del opt, sched, ema
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return {"best_val_loss": best, "history": history,
            "batch": schedule.batch, "accum": schedule.accum,
            "steps_per_epoch": schedule.steps_per_epoch,
            "stages": [s[0] for s in schedule.stages], **schedule.meta}
