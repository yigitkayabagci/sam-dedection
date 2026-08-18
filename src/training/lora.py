"""LoRA for EdgeTAM, as the measurable alternative to partial fine-tuning.

`src/training/finetune.py` argues against LoRA. The argument is reasonable and
it is still only an argument -- nothing in this repo has measured it. This
module exists so the two can be run against each other under an identical loop,
identical data and identical evaluation, and the question settled with a number
on the held-out split instead of a table of expectations.

**What LoRA is here.** Every adapted layer keeps its frozen weight `W` and gains
`B @ A`, with `A` random and `B` zero, so the network starts exactly where the
checkpoint left it. Only `A` and `B` receive gradients. `alpha / r` scales the
update, which is the convention every published implementation uses and the
reason `alpha` is usually quoted as `2r`.

**Convolutions are adapted too, and that matters here.** The standard objection
to LoRA on this model -- that it only reaches `nn.Linear`, while the thermal
domain shift lives in the convolutional RepViT trunk -- is an objection to a
Linear-only implementation, not to the method. A `k x k` convolution composed
with a `1 x 1` convolution *is* a `k x k` convolution, so the same
low-rank factorisation applies exactly, and `merge` produces a weight tensor
indistinguishable from one a full fine-tune could have produced.

**Grouped convolutions are skipped**, depthwise ones included. Their weight is
block-diagonal and a dense `B @ A` cannot represent an update to it; adapting
them would need a per-group factorisation whose rank-1-per-group budget is
below what the layer already has. They are a small share of the parameters, and
`inject` reports how many it passed over rather than hiding it.

**Merging is the point of the deployment story.** `merge` folds every adapter
back into its base weight and restores the original module, so the state dict is
key-for-key the one `build_sam2` expects: the LoRA checkpoint loads into the
same config, exports to the same ONNX graph and builds the same engines as the
fine-tuned one. If it were not exactly interchangeable, the comparison would be
measuring two different deployments rather than two training methods.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .finetune import FROZEN_MODULES, STAGES, _owned_by, apply_freeze

LORA_PREFIXES = ("lora_A", "lora_B")


def is_lora_parameter(name: str) -> bool:
    return name.split(".")[-1] in LORA_PREFIXES


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """`W x + b` plus `(alpha / r) B A x`, with `W` frozen and `B` zero at init."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r
        kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, **kwargs))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, **kwargs))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return self.base(x) + update * self.scaling

    def delta(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling


class LoRAConv2d(nn.Module):
    """The same factorisation for convolutions, exact under merging.

    `A` carries the base layer's kernel, stride, padding and dilation; `B` is
    `1 x 1`. Composing them is a single convolution with weight `B A`, which is
    what `delta` returns and what `merge` folds into the base weight -- so the
    merged model is not an approximation of the adapted one, it is the same
    function.
    """

    def __init__(self, base: nn.Conv2d, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if base.groups != 1:
            raise ValueError("grouped convolutions are not adapted; see the module docstring")
        self.base = base
        self.r = r
        self.scaling = alpha / r
        kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_A = nn.Parameter(
            torch.empty(r, base.in_channels, *base.kernel_size, **kwargs))
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_channels, r, 1, 1, **kwargs))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base = self.base
        hidden = F.conv2d(self.dropout(x), self.lora_A, None, base.stride,
                          base.padding, base.dilation)
        return base(x) + F.conv2d(hidden, self.lora_B) * self.scaling

    def delta(self) -> torch.Tensor:
        out, r = self.lora_B.shape[:2]
        flat = self.lora_B.reshape(out, r) @ self.lora_A.reshape(r, -1)
        return flat.reshape(self.base.weight.shape) * self.scaling


ADAPTED = (LoRALinear, LoRAConv2d)


def _adapt(module: nn.Module, r: int, alpha: float, dropout: float = 0.0):
    """An adapter for `module`, or None if this kind of layer is left alone."""
    if isinstance(module, nn.Linear):
        if min(module.in_features, module.out_features) <= r:
            return None      # the factorisation would be larger than the layer
        return LoRALinear(module, r, alpha, dropout)
    if isinstance(module, nn.Conv2d):
        if module.groups != 1:
            return None
        if min(module.in_channels, module.out_channels) <= r:
            return None
        return LoRAConv2d(module, r, alpha, dropout)
    return None


# --------------------------------------------------------------------------
# Injection, freezing, merging
# --------------------------------------------------------------------------


def inject(
    model: nn.Module,
    stage: str = "encoder",
    r: int = 8,
    alpha: float | None = None,
    dropout: float = 0.0,
) -> dict:
    """Wrap every adaptable layer `stage` reaches. Returns what it did.

    The scope is `finetune.STAGES`, the same table the partial fine-tune uses,
    minus `FROZEN_MODULES` -- so "LoRA on the encoder stage" adapts exactly the
    modules "fine-tune the encoder stage" would have trained. Anything else
    would compare two different experiments.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {sorted(STAGES)}, got {stage!r}")
    alpha = float(2 * r if alpha is None else alpha)
    trainable = STAGES[stage]

    targets = [
        (name, module) for name, module in model.named_modules()
        if _owned_by(name, trainable) and not _owned_by(name, FROZEN_MODULES)
        and isinstance(module, (nn.Linear, nn.Conv2d))
    ]

    adapted, skipped = {}, {"grouped": 0, "too_small": 0}
    for name, module in targets:
        replacement = _adapt(module, r, alpha, dropout)
        if replacement is None:
            key = "grouped" if getattr(module, "groups", 1) != 1 else "too_small"
            skipped[key] += 1
            continue
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attribute, replacement)
        adapted[name] = sum(p.numel() for p in (replacement.lora_A, replacement.lora_B))

    if not adapted:
        raise RuntimeError(f"stage {stage!r} has no layer LoRA can adapt.")
    return {"stage": stage, "r": r, "alpha": alpha, "adapted": adapted,
            "skipped": skipped,
            "parameters": sum(adapted.values()),
            "total": sum(p.numel() for p in model.parameters())}


def freeze(model: nn.Module, stage: str = "encoder") -> dict[str, int]:
    """Only the adapters train, within `stage`'s scope.

    Mirrors `finetune.apply_freeze`'s contract -- per-root trainable counts,
    for `summarise_freeze` and for the notebook to assert on -- so the two
    methods are driven by the same schedule with one callback swapped.
    """
    apply_freeze(model, stage)
    counts: dict[str, int] = {}
    for name, param in model.named_parameters():
        allowed = param.requires_grad and is_lora_parameter(name)
        param.requires_grad_(allowed)
        if allowed:
            root = name.split(".")[0]
            counts[root] = counts.get(root, 0) + param.numel()
    if not counts:
        raise RuntimeError(
            f"stage {stage!r} left no adapter trainable -- was inject() called?")
    return counts


@torch.no_grad()
def merge(model: nn.Module) -> int:
    """Fold every adapter into its base weight and put the base layer back.

    Afterwards the model *is* an EdgeTAM again: same modules, same state-dict
    keys, loadable by `build_sam2` strictly. Returns how many layers merged.
    """
    merged = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, ADAPTED):
            continue
        base = module.base
        base.weight.data += module.delta().to(base.weight.dtype)
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attribute, base)
        merged += 1
    return merged


def save_merged_checkpoint(model: nn.Module, path, meta: dict | None = None):
    """Write the merged weights without disturbing the model still training.

    A merge in place would end the run: the adapters would be gone and the next
    optimiser step would have nothing to update. Copying 13.9 M parameters to
    merge the copy costs ~56 MB and a fraction of a second, which is nothing
    against an epoch.
    """
    import copy

    from .finetune import save_checkpoint

    clone = copy.deepcopy(model)
    merge(clone)
    written = save_checkpoint(clone, path, meta)
    del clone
    return written


def summarise(report: dict, counts: dict[str, int] | None = None) -> str:
    """One block: how many parameters move, against how many exist."""
    counts = counts or {}
    trainable = report["parameters"]
    lines = [
        f"LoRA r={report['r']} alpha={report['alpha']:.0f} on stage "
        f"{report['stage']!r}",
        f"  {len(report['adapted'])} layers adapted, "
        f"{trainable / 1e6:.3f} M trainable of {report['total'] / 1e6:.2f} M "
        f"({trainable / report['total']:.2%})",
    ]
    if report["skipped"]["grouped"]:
        lines.append(f"  {report['skipped']['grouped']} grouped/depthwise "
                     f"convolutions left alone (see src/training/lora.py)")
    if report["skipped"]["too_small"]:
        lines.append(f"  {report['skipped']['too_small']} layers narrower than r")
    for root, count in sorted(counts.items()):
        lines.append(f"  {root:<22} {count / 1e6:7.3f} M")
    return "\n".join(lines)
