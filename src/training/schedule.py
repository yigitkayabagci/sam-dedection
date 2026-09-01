"""The training loop itself, so two methods can be compared rather than two loops.

A partial fine-tune and a LoRA run differ in exactly two places: which
parameters are unfrozen, and how the checkpoint is written (LoRA merges a copy).
Everything else -- the two stages, the batch order, the one-cycle schedule, the
gradient clipping, the EMA, which validation clips are scored and when the best
checkpoint is kept -- has to be *identical*, or the comparison measures the
difference between two notebooks instead of the difference between two methods.

So the loop lives here once and takes those two things as callbacks. The
notebook that explains it and the CLI that runs it twice execute the same code.

**A third thing is a callback for the same reason: what a batch is.** The
encoder work trains on single images with several prompts
(`src/training/image_loop.py`) rather than on clips, and that is a different
data pipeline and a different loss -- but it is not a different schedule, and
running it through a second copy of this file would make "static training
versus clip training" a comparison of two loops again. `Loop` is the pair of
functions that differ; `CLIPS` and `IMAGES` are the two of them that exist.
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
    # Epochs without an improvement before a stage gives up. 0 is off, and off
    # is the default because a stage that runs its full budget is the recorded
    # behaviour of every run before this existed.
    #
    # It is a safety net, not a tuning knob, and the reason is the one-cycle
    # schedule: `total_steps` is sized from the stage's epoch budget, so the
    # rate anneals over exactly that many steps and the best weights normally
    # appear near the end of the descent. Stopping early leaves the rate high
    # and the descent unfinished. So raise `epochs` to buy a longer anneal and
    # set `patience` generously, to cut a run that has plainly stalled rather
    # than to decide when it is done.
    patience: int = 0
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Loop:
    """What a batch is made of, and what it costs -- the two data-mode knobs.

    `stream(split, batch, seed, limit, device, workers, depth)` yields batches;
    `loss(model, batch)` returns `(loss, terms)`. Nothing else about a run is
    allowed to depend on which mode it is in.

    `val_loss` is what `validate` scores with, and it exists because the
    training loss is allowed to be **random** while the validation number is
    not. A mixed prompt (`image_loop.mix_boxes`) draws a fresh box-or-jitter
    choice per row per batch; running that inside `validate` would make the
    number move between epochs for a reason that is not the model, and the
    epoch-selection rule reads that number to decide which checkpoint to keep.
    Left `None` it is `loss`, so a mode with a deterministic loss says nothing
    and behaves exactly as before.
    """

    stream: Callable
    loss: Callable
    val_loss: Callable | None = None
    val_stream: Callable | None = None

    @property
    def scoring(self) -> Callable:
        """What `validate` calls: `val_loss` if a mode set one, else `loss`."""
        return self.val_loss or self.loss

    @property
    def batches(self) -> Callable:
        """What `validate` pulls on: `val_stream` if a mode set one, else `stream`.

        The same argument `val_loss` makes, one layer out. A training stream is
        allowed to be random -- photometric augmentation collapses a window's
        contrast on a coin flip -- and the validation number is not: it selects
        which epoch's weights are kept, so a run whose validation windows were
        augmented differently each epoch would be choosing on the draw as much
        as on the model.
        """
        return self.val_stream or self.stream


def _clip_stream(split: Split, batch: int, seed: int | None, limit: int | None,
                 device: str = "cuda", workers: int = 8, depth: int = 2):
    return prefetch(batch_clips(split.clips, batch, seed=seed, limit=limit),
                    split.stores, device, workers=workers, depth=depth)


def _clip_loss(model, batch):
    # Resolved through the module global rather than captured, so a caller (or
    # a test) that swaps `clip_losses` swaps what the loop actually runs.
    return clip_losses(model, batch)


CLIPS = Loop(stream=_clip_stream, loss=_clip_loss)


def images(anchor=None, anchor_weight: float = 0.0, prompt: str = "box",
           jitter: float = 0.0,
           generator: "torch.Generator | None" = None,
           augment=None, neighbour_weight: float = 0.0) -> Loop:
    """The image-mode `Loop`, imported lazily.

    `image_loop` pulls in `aerial`, which pulls in nothing heavy but does bind
    a dataset vocabulary that a clip-mode run has no use for. Keeping the
    import inside the call keeps `schedule` importable with nothing but numpy
    and torch, which is what `tests/test_schedule.py` relies on.

    `anchor` is a frozen copy of the model as this stage started; with a
    non-zero `anchor_weight` the loss gains a term for how far the encoder has
    drifted from it. See `image_loop.image_losses`.

    `prompt` is how a ground-truth box becomes the prompt the loop trains
    against -- `box` reproduces every run before `12_encoder_probe.ipynb`, and
    `mix` is what the probe's jitter regression argues for. **Validation always
    scores under `box`**, whatever training uses: the point of the validation
    number is that a change in it is the model, and a random prompt would put a
    second moving part into a number that selects checkpoints. It also keeps
    the number comparable to every run taken before this knob existed.

    `augment` hardens the *training* windows only -- `photometric.augmenter`
    builds one -- and `val_stream` is deliberately the plain stream, for the
    reason `Loop.batches` gives.

    `neighbour_weight` is a **training** term for the same reason `prompt` is
    a training setting: it changes the objective, and the validation number is
    what selects checkpoints and what every earlier run is compared against.
    So validation keeps the published weighting whatever training uses, and
    the leak itself is reported on both sides regardless -- see
    `losses.neighbour_terms`.
    """
    from .image_loop import image_losses, stream
    from .losses import Weights

    def training_stream(*args, **kwargs):
        return stream(*args, augment=augment, **kwargs)

    training_weights = (Weights(neighbour=neighbour_weight)
                        if neighbour_weight else None)
    return Loop(stream=training_stream if augment is not None else stream,
                val_stream=stream,
                loss=lambda model, batch: image_losses(
                    model, batch, weights=training_weights, anchor=anchor,
                    anchor_weight=anchor_weight,
                    prompt=prompt, jitter=jitter, generator=generator),
                val_loss=lambda model, batch: image_losses(
                    model, batch, anchor=anchor, anchor_weight=anchor_weight))


def validate(
    model,
    split,
    schedule: Schedule,
    device: str = "cuda",
    loop: Loop = CLIPS,
) -> float:
    """Mean loss over a fixed slice of the split.

    Fixed because the seed is fixed: the same batches every epoch, so a change
    in the number is the model and not the sample. The slice is bounded *while
    generating*, which is the whole point -- scoring the entire validation set
    every epoch is hours nobody asked for.
    """
    losses = []
    with torch.no_grad():
        for batch in loop.batches(split, schedule.batch, 1, schedule.val_batches,
                                  device, schedule.workers, schedule.depth):
            with _autocast(device):
                losses.append(float(loop.scoring(model, batch)[0]))
    return float(np.mean(losses)) if losses else float("nan")


def _autocast(device: str):
    return torch.autocast("cuda", dtype=torch.bfloat16,
                          enabled=device.startswith("cuda"))


def run_stages(
    model,
    train,
    val,
    schedule: Schedule = Schedule(),
    *,
    freeze: Callable[[object, str], dict[str, int]],
    save: Callable[[object, dict], None],
    device: str = "cuda",
    progress: Callable | None = None,
    log: Callable[[str], None] = print,
    loop: Loop = CLIPS,
) -> dict:
    """Train `model` through `schedule.stages`; return the run's history.

    `freeze(model, stage)` decides what moves -- `finetune.apply_freeze` for a
    partial fine-tune, `lora.freeze` for LoRA -- and returns per-root trainable
    counts. `save(model, meta)` writes whichever checkpoint that method should
    produce, and is called only when the validation loss improves, inside the
    EMA context so the averaged weights are the ones stored.

    `loop` decides what a batch is: `CLIPS` for the video path, `images()` for
    the static encoder path. Everything below this line is the same either way,
    which is the property that lets the two be compared.

    The EMA is rebuilt per stage on purpose: it averages the trainable
    parameters, and that set changes the moment the encoder unfreezes.
    """
    best, history, stopped = float("inf"), [], []

    for stage, epochs, rates in schedule.stages:
        stale = 0
        failed_nonfinite = False
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
            stream = loop.stream(train, schedule.batch,
                                 schedule.seed + 100 * epoch,
                                 schedule.steps_per_epoch, device,
                                 schedule.workers, schedule.depth)
            if progress is not None:
                stream = progress(stream, total=schedule.steps_per_epoch,
                                  desc=f"{stage} e{epoch}")

            for step, batch in enumerate(stream):
                with _autocast(device):
                    loss, terms = loop.loss(model, batch)
                if not bool(torch.isfinite(loss).all()):
                    # A non-finite update contaminates every later recurrent
                    # prediction.  More epochs cannot recover from NaN weights;
                    # the best checkpoint from an earlier finite validation is
                    # already on disk, so stop this stage immediately instead
                    # of burning the remainder of a Colab session.
                    opt.zero_grad(set_to_none=True)
                    failed_nonfinite = True
                    stopped.append({"stage": stage, "after_epoch": epoch,
                                    "of": epochs,
                                    "reason": "nonfinite_train_loss",
                                    "step": step})
                    log(f"  stopping {stage!r}: non-finite training loss at "
                        f"epoch {epoch}, step {step}. The last finite best "
                        f"checkpoint is unchanged.")
                    break
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

            if failed_nonfinite:
                break

            with ema.applied(model):
                score = validate(model, val, schedule, device, loop)
                if not np.isfinite(score):
                    history.append({"stage": stage, "epoch": epoch,
                                    "val_loss": score, "saved": False})
                    failed_nonfinite = True
                    stopped.append({"stage": stage, "after_epoch": epoch,
                                    "of": epochs,
                                    "reason": "nonfinite_validation_loss"})
                    log(f"  stopping {stage!r}: validation loss is not finite "
                        f"at epoch {epoch}. The last finite best checkpoint "
                        f"is unchanged.")
                    break
                improved = score < best
                if improved:
                    best = score
                    save(model, {"stage": stage, "epoch": epoch, "val_loss": score,
                                 **schedule.meta})
            history.append({"stage": stage, "epoch": epoch, "val_loss": score,
                            "saved": improved})
            stale = 0 if improved else stale + 1
            log(f"  epoch {epoch}: val clip loss {score:.4f}"
                f"{'  <- saved' if improved else f'  ({stale} without one)'}")
            if schedule.patience and stale >= schedule.patience:
                stopped.append({"stage": stage, "after_epoch": epoch,
                                "of": epochs, "patience": schedule.patience})
                log(f"  stopping {stage!r}: {stale} epoch(s) without an "
                    f"improvement, of {epochs} budgeted. The best checkpoint "
                    f"is already written.")
                break

        del opt, sched, ema
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    return {"best_val_loss": best, "history": history,
            "patience": schedule.patience, "stopped_early": stopped,
            "batch": schedule.batch, "accum": schedule.accum,
            "steps_per_epoch": schedule.steps_per_epoch,
            "stages": [s[0] for s in schedule.stages], **schedule.meta}
