"""What to train, what to leave alone, and how to save it back.

This module implements **partial fine-tuning**, and it used to argue that the
alternative was not worth having. That argument has since been run against a
number and it lost (`docs/lora_vs_finetune.md`): on the held-out split LoRA
matched this method's accuracy while moving 11% of the parameters, and got the
target back from a dropout in half as many frames. `configs/edgetam_512_lora.yaml`
is the default for thermal work now. Everything below still runs, is still what
`--method finetune` does, and is still what LoRA is measured against.

Of the three claims that argument rested on, two were true and turned out not
to be the deciding ones, and one was simply wrong:

* *"Nothing here is memory-constrained, so LoRA solves a problem we do not
  have."* True, and measured true -- identical peak memory, identical batch
  size, LoRA 14% slower. It is just not why LoRA won.
* *"A merged adapter is only weights, so the exported graph is unchanged."*
  True, and it is what makes the two interchangeable rather than what makes
  either of them better.
* *"LoRA targets `nn.Linear`, and the domain shift is in a convolutional
  trunk."* Wrong -- an objection to a Linear-only implementation, not to the
  method. `src/training/lora.py` adapts convolutions exactly.

What decided it was the one thing freezing cannot express: a rank-constrained
update. **The freeze policy below is unchanged and governs both methods** --
LoRA adapts exactly the modules this table would have trained:

**Always frozen: the memory path** -- `memory_attention`, `memory_encoder`,
`spatial_perceiver`, and the learned memory tokens. Three separate reasons, any
one of which would be enough:

1. It is the write port of a recurrent loop. This project already measured what
   a systematic change there does: quantising the memory encoder alone gave
   mean IoU 0.9397, and unlike every other module the damage was a *sustained
   decline across the clip* rather than isolated dropouts, because its output
   is what the bank stores and the next seven frames read back
   (`docs/tensorrt_fp16.md`).
2. Its ONNX rewrite is the intricate one -- fixed memory slots, tiled RoPE
   tables, an additive mask. Retraining the weights those graphs were derived
   from invites a silent mismatch between checkpoint and engine.
3. It operates on abstract features, not pixels. Thermal-versus-RGB is an
   *encoder* problem; there is no reason to expect the memory path to need it.

**Trained, in two stages.** The head first, on its own, so the mask decoder and
the object-score head adapt to thermal statistics before the features under
them start moving. Then the encoder joins at a lower rate.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

# The recurrent core. Never trained -- see the module docstring.
FROZEN_MODULES = ("memory_attention", "memory_encoder", "spatial_perceiver")
FROZEN_PARAMS = ("maskmem_tpos_enc", "no_mem_embed", "no_mem_pos_enc",
                 "no_obj_ptr", "no_obj_embed_spatial")

HEAD_MODULES = ("sam_mask_decoder", "sam_prompt_encoder", "obj_ptr_proj")
STAGES = {
    "head": HEAD_MODULES,
    "encoder": HEAD_MODULES + ("image_encoder",),
}


def _owned_by(name: str, modules: tuple[str, ...]) -> bool:
    return any(name == m or name.startswith(m + ".") for m in modules)


def apply_freeze(model, stage: str = "head") -> dict[str, int]:
    """Set `requires_grad` for `stage`, and report the trainable count per root.

    Returning the counts rather than printing them means the notebook can
    assert on them: a stage that silently trained the memory path, or trained
    nothing at all, shows up as a number rather than as a bad checkpoint three
    hours later.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {sorted(STAGES)}, got {stage!r}")
    trainable = STAGES[stage]

    counts: dict[str, int] = {}
    for name, param in model.named_parameters():
        root = name.split(".")[0]
        allowed = (
            _owned_by(name, trainable)
            and not _owned_by(name, FROZEN_MODULES)
            and root not in FROZEN_PARAMS
        )
        param.requires_grad_(allowed)
        if allowed:
            counts[root] = counts.get(root, 0) + param.numel()
    if not counts:
        raise RuntimeError(f"stage {stage!r} left nothing trainable.")
    return counts


@dataclass(frozen=True)
class Rates:
    """Learning rates per part, from the fastest-moving to the slowest.

    The trunk moves an order of magnitude slower than the head: it carries
    general visual features that took far more data to learn than this dataset
    contains, and the whole point of freezing the memory path is that it is not
    the part that needs to change.
    """

    head: float = 1e-4
    neck: float = 1e-4
    trunk: float = 1e-5
    weight_decay: float = 1e-4


def param_groups(model, rates: Rates = Rates()) -> list[dict]:
    """Optimiser groups, skipping anything `apply_freeze` turned off.

    Norm and bias parameters get no weight decay, which matters more than usual
    here: decaying a frozen-statistics batch norm's affine terms on a small
    dataset shifts the whole feature scale for no benefit.
    """
    buckets: dict[tuple[str, bool], list] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("image_encoder.trunk"):
            part = "trunk"
        elif name.startswith("image_encoder"):
            part = "neck"
        else:
            part = "head"
        decayed = param.ndim > 1
        buckets.setdefault((part, decayed), []).append(param)

    return [
        {"params": params,
         "lr": getattr(rates, part),
         "weight_decay": rates.weight_decay if decayed else 0.0,
         "name": f"{part}{'' if decayed else '/no-decay'}"}
        for (part, decayed), params in sorted(buckets.items())
    ]


class EMA:
    """Exponential moving average of the trainable weights.

    Kept for the trainable parameters only -- the frozen ones do not move, so
    averaging them would just be a second copy of the checkpoint. Evaluate
    inside `applied()`; the averaged weights are almost always the better ones
    on a dataset this small, and the raw ones are still there afterwards.
    """

    def __init__(self, model, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters() if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(param.detach(), 1.0 - self.decay)

    @contextmanager
    def applied(self, model):
        backup = {name: param.detach().clone() for name, param in model.named_parameters()
                  if name in self.shadow}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    param.copy_(self.shadow[name])
        try:
            yield model
        finally:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in backup:
                        param.copy_(backup[name])


def save_checkpoint(model, path: str | Path, meta: dict | None = None) -> Path:
    """Write a checkpoint EdgeTAM's own loader accepts unchanged.

    `build_sam2` reads `torch.load(path)["model"]` and calls `load_state_dict`
    strictly, so the fine-tuned weights drop straight into the existing
    configs, exporter and engine build with nothing else touched. That is the
    whole reason this fine-tune does not alter the architecture.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta or {}}, path)
    return path


def summarise_freeze(counts: dict[str, int], model) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(counts.values())
    lines = [f"trainable {trainable / 1e6:.2f} M of {total / 1e6:.2f} M "
             f"({trainable / total:.1%})", ""]
    lines += [f"  {root:<22} {count / 1e6:6.2f} M" for root, count in sorted(counts.items())]
    frozen = sorted(FROZEN_MODULES)
    lines += ["", f"  frozen throughout: {', '.join(frozen)}"]
    return "\n".join(lines)
