"""LoRA for EdgeTAM, as the measurable alternative to partial fine-tuning.

`src/training/finetune.py` used to argue against LoRA. This module exists so
that argument could be run against a number rather than repeated -- under an
identical loop, identical data and identical evaluation -- and it has been.
LoRA matched the partial fine-tune's accuracy on the held-out split while
training 11% of the parameters, with dropout episodes half as long, so the
merged LoRA checkpoint is what `configs/edgetam_512_lora.yaml` deploys.
The whole table, and what it does and does not license, is in
`docs/lora_vs_finetune.md`.

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

**The delta is also kept on its own.** `save_adapter` writes `A` and `B`
without the base weights and `apply_adapter` puts them back onto a stock
checkpoint at load time (`configs/edgetam_512_lora_adapter.yaml`). Merging is
what makes deployment simple; keeping the adapter separately is what makes "the
original weights were never disturbed" a fact about the filesystem rather than
a figure of speech -- and it is what lets a second domain be a second few-MB
adapter beside the same base instead of a second 54 MB checkpoint.
"""
from __future__ import annotations

import math
from pathlib import Path

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


# --------------------------------------------------------------------------
# The adapter as an artefact of its own
# --------------------------------------------------------------------------

ADAPTER_FORMAT = 1

_ADAPTER_KINDS = {
    "LoRALinear": (LoRALinear, nn.Linear),
    "LoRAConv2d": (LoRAConv2d, nn.Conv2d),
}


def adapter_state(model: nn.Module, meta: dict | None = None) -> dict:
    """`A` and `B` for every adapted layer, plus enough to put them back.

    `save_merged_checkpoint` writes the deployment artefact -- base plus delta,
    folded, indistinguishable from what a fine-tune would have produced. This
    writes the delta on its own, a few MB against that file's 54, and it is
    what makes the usual reason for reaching for LoRA true here rather than
    merely stated: **the base checkpoint is never overwritten.** Adaptation
    becomes a file that can be attached, detached, scaled or swapped for
    another domain's, instead of a second full set of weights that has to be
    trusted as a whole.

    Per layer this records `r` and the *scaling*, not `alpha`. A reload then
    does not depend on the `alpha = 2r` convention still holding on the day it
    is read, and `load_adapter` needs no argument that could contradict how the
    run was actually configured.
    """
    tensors: dict[str, torch.Tensor] = {}
    layers: dict[str, dict] = {}
    for name, module in model.named_modules():
        if not isinstance(module, ADAPTED):
            continue
        for part in LORA_PREFIXES:
            tensors[f"{name}.{part}"] = (
                getattr(module, part).detach().to("cpu", torch.float32).clone())
        layers[name] = {"kind": type(module).__name__, "r": module.r,
                        "scaling": float(module.scaling)}
    if not layers:
        raise RuntimeError("this model carries no adapters -- was inject() called?")
    return {"format": ADAPTER_FORMAT, "layers": layers, "adapters": tensors,
            "meta": meta or {}}


def save_adapter(model: nn.Module, path, meta: dict | None = None) -> Path:
    """Write `adapter_state` beside the merged checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter_state(model, meta), path)
    return path


@torch.no_grad()
def load_adapter(model: nn.Module, source, scale: float = 1.0) -> dict:
    """Re-wrap exactly the layers the file names and restore their `A` and `B`.

    Deliberately *not* `inject()` followed by a `load_state_dict`: that would
    re-derive the target list from `STAGES` at load time, and would silently
    adapt a different set of layers if that table ever changed. The file names
    the layers it was trained on; anything it names that this model does not
    have, or that has a different shape, is an error here rather than a
    mysteriously unchanged weight much later.

    `scale` multiplies every adapter's contribution: 0.0 is the base checkpoint
    exactly, 1.0 is the run as it was trained. Values in between blend the two,
    which is a knob worth having when the adapter's domain and the footage's
    only partly overlap -- and which nothing in this repo has measured.
    """
    if not isinstance(source, dict):
        source = torch.load(source, map_location="cpu", weights_only=True)
    if source.get("format") != ADAPTER_FORMAT:
        raise ValueError(f"adapter format {source.get('format')!r}, "
                         f"expected {ADAPTER_FORMAT}")

    tensors, layers = source["adapters"], source["layers"]
    for name, spec in layers.items():
        if spec["kind"] not in _ADAPTER_KINDS:
            raise ValueError(f"{name!r}: unknown adapter kind {spec['kind']!r}")
        adapter_cls, base_cls = _ADAPTER_KINDS[spec["kind"]]
        try:
            base = model.get_submodule(name)
        except AttributeError as exc:
            raise KeyError(f"{name!r} is in the adapter file but not in this "
                           f"model") from exc
        if isinstance(base, ADAPTED):
            raise RuntimeError(f"{name!r} is adapted already; load into a fresh model")
        if not isinstance(base, base_cls):
            raise TypeError(f"{name!r} is a {type(base).__name__}, but the adapter "
                            f"was trained on a {base_cls.__name__}")

        adapter = adapter_cls(base, r=spec["r"], alpha=spec["r"])
        adapter.scaling = spec["scaling"] * scale
        for part in LORA_PREFIXES:
            saved, target = tensors[f"{name}.{part}"], getattr(adapter, part)
            if saved.shape != target.shape:
                raise ValueError(f"{name}.{part}: the file holds "
                                 f"{tuple(saved.shape)}, this model wants "
                                 f"{tuple(target.shape)}")
            target.copy_(saved.to(target.dtype))

        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attribute, adapter)

    return {"layers": len(layers), "scale": float(scale),
            "parameters": sum(t.numel() for t in tensors.values()),
            "meta": source.get("meta", {})}


@torch.no_grad()
def apply_adapter(model: nn.Module, source, scale: float = 1.0) -> dict:
    """Load an adapter onto a base model and fold it in, in one step.

    This is the inference path. `build_sam2` loads the stock checkpoint, this
    puts the thermal adaptation on top of it, and what comes back is an
    ordinary EdgeTAM again: same modules, same state-dict keys, same export,
    same engines. The difference from loading the merged checkpoint is on disk
    rather than in the model -- upstream's file stays exactly as it shipped,
    and the small one beside it is the whole of what this project changed.
    """
    report = load_adapter(model, source, scale=scale)
    report["merged"] = merge(model)
    return report


def summarise_adapter(report: dict) -> str:
    """One line for a run log: what was attached, from where, how strongly."""
    meta = report.get("meta") or {}
    bits = [f"{report['layers']} layers", f"{report['parameters'] / 1e6:.3f} M"]
    if meta.get("lora_r"):
        bits.append(f"r={meta['lora_r']:g}")
    if meta.get("dataset"):
        bits.append(str(meta["dataset"]))
    if report["scale"] != 1.0:
        bits.append(f"scaled to {report['scale']:g} of what was trained")
    return "lora adapter: " + ", ".join(bits)


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
