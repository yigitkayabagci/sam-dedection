#!/usr/bin/env python3
"""Build the two stage-A pretraining notebooks from plain text blocks.

    python tools/build_pretrain_notebooks.py          # both
    python tools/build_pretrain_notebooks.py mask     # just that one

Two notebooks come out of this one file:

    15_pretrain_automask.ipynb   the encoder supervises itself through a mask
    16_pretrain_convnext.ipynb   a frozen DINOv3 ConvNeXt supervises it

They are **one experiment with four arms**, not two notebooks. The corpora, the
schedule, the seed, the freeze, the checkpoint format and the handoff to stage
B are byte for byte the same in both; `ARM` is a setting and either notebook
can run any of the four. What differs is the default arm, the one diagnostic
cell that arm needs before it spends GPU hours, and the prose that argues it.

Generated rather than edited for the same reason 07-11 and 13-14 are: a
notebook is a JSON document in which prose, code and cell numbers drift apart
silently. `{{tag}}` resolves cross-references at build time and
`tools/check_notebook.py` walks the result for names used before they are
bound.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, fields
from pathlib import Path

CELLS: list[tuple[str, str]] = []
TAGS: dict[str, int] = {}
STAMP_VALUE = "unstamped"


@dataclass(frozen=True)
class Variant:
    """What differs between the two notebooks, and nothing else."""

    key: str
    path: str
    title: str
    blurb: str
    arm: str
    needs: str
    takes: str
    settings: str               # the arm's own settings block
    gate: str                   # the Hugging Face login cell, or a no-op
    check_md: str               # markdown above the pre-flight cell
    check: str                  # the cell that inspects before training
    invoke: str                 # the arguments handed to the CLI
    report: str                 # the arm's own reading of what came out
    appendix: str


VARIANTS: dict[str, Variant] = {}


def md(text: str, tag: str | None = None) -> None:
    _add("markdown", text, tag)


def code(text: str, tag: str | None = None) -> None:
    _add("code", text, tag)


def _add(kind: str, text: str, tag: str | None) -> None:
    if tag:
        TAGS[tag] = len(CELLS)
    CELLS.append((kind, text.strip("\n")))


def resolve(text: str) -> str:
    """Replace `{{tag}}` with the cell number that tag landed on."""
    substitutions = {"TITLE": V.title, "BLURB": V.blurb, "ARM": V.arm,
                     "NEEDS": V.needs, "TAKES": V.takes,
                     "SETTINGS": V.settings, "GATE": V.gate,
                     "CHECK_MD": V.check_md, "CHECK": V.check,
                     "INVOKE": V.invoke, "REPORT": V.report,
                     "APPENDIX": V.appendix,
                     "NOTEBOOK": Path(V.path).name, "STAMP": STAMP_VALUE,
                     **{t: str(i) for t, i in TAGS.items()}}
    for tag, value in substitutions.items():
        text = text.replace("{{" + tag + "}}", value)
    leftover = re.findall(r"\{\{(\w+)\}\}", text)
    if leftover:
        raise KeyError(f"unknown cell tags: {sorted(set(leftover))}")
    return text


# --------------------------------------------------------------------------
# The two variants
# --------------------------------------------------------------------------

VARIANTS["mask"] = Variant(
    key="mask",
    path="notebooks/15_pretrain_automask.ipynb",
    arm="mask",
    title="Stage A — the encoder supervises itself through its own mask",
    blurb=(
        "Runs the **mask arm**: hide part of the frame, ask the encoder to "
        "produce the features it would have produced had it seen the whole "
        "thing, and let the *model choose what to hide* from its own feature "
        "map. No teacher, no registered pair, no label — which is the point, "
        "because it means the thermal-only sets that the distillation arm "
        "cannot use are this arm's natural food. It is the long shot of the "
        "four and the notebook says so out loud; run it because the "
        "alternative is a report that says \"we did not try\"."),
    needs="a CUDA GPU and ~6 GB of free disk at the default corpora (no "
          "Hugging Face token — this arm downloads no teacher)",
    takes="~10 min to download and 40–90 min to train at the default 4 000 "
          "steps; the mask preview in cell {{check}} costs seconds and is "
          "worth reading first",
    settings='''ARM        = "mask"          # none | mask | distil | both
MASK_MODE  = "hint"          # random | attentive | hint | block | green
MASK_RATIO = 0.5             # fraction of the 32x32 feature grid hidden
MASK_HINT  = 0.1             # of the masked cells, the most salient tenth
                             # stays visible (AttMask's remedy for a task
                             # that becomes unsolvable rather than hard)
SALIENCY   = "distinct"      # distinct | norm -- how a cell's interest is
                             # scored, given a conv trunk has no attention
TARGET     = "ema"           # ema | frozen -- what the masked student aims
                             # at. `frozen` cannot collapse and cannot beat
                             # the stock encoder; `ema` can do both.
TARGET_EMA = 0.999''',
    gate='''# --- No gated download in this arm --------------------------------------
# The mask arm's supervision is the encoder's own feature map, so nothing is
# fetched from a gated repository and no Hugging Face account is needed. The
# teacher notebook logs in here instead.
print("mask arm: no teacher, no Hugging Face token required")''',
    check_md='''## Look at the mask before paying for it

The one thing that can go wrong here without raising anything: a
saliency-driven mask that hides the wrong half of the picture. On aerial
thermal the frame is mostly uniform ground with a few small hot targets, and
"mask the most distinctive cells" and "mask every target and leave the ground"
are the same instruction. That is either exactly the right pretext task or a
way to spend every gradient on 2 % of the frame.

The cell below draws it: the frame, the model's own saliency, and what
`MASK_MODE` actually hides — for the configured mode and for `random` and
`green` beside it, since those are the two baselines this arm has to beat.
Read it, then decide whether to run.''',
    check='''# --- What does the model choose to hide? --------------------------------
import matplotlib.pyplot as plt
import torch

from src.training import automask, pretrain
from src.training.distill import encoder_features
from tools.train_encoder import build_model

preview_model = build_model(SIZE, BASE_CKPT, DEVICE)
shown = pretrain.subsample(ITEMS, 4, seed=SEED)
batch = pretrain.collate(shown, size=SIZE, device=DEVICE, want_teacher=False)

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=DEVICE.startswith("cuda")):
    clean = encoder_features(preview_model, batch.student)
scores = automask.saliency(clean, SALIENCY)

modes = [MASK_MODE] + [m for m in ("random", "green") if m != MASK_MODE]
figure, axes = plt.subplots(len(shown), 2 + len(modes),
                            figsize=(3 * (2 + len(modes)), 3 * len(shown)))
axes = axes.reshape(len(shown), -1)
for row, item in enumerate(shown):
    picture = pretrain.to_pixels(batch.student[row:row + 1])[0].permute(1, 2, 0)
    axes[row, 0].imshow(picture.clamp(0, 1).float().cpu())
    axes[row, 0].set_title(item.corpus if row == 0 else "", fontsize=9)
    axes[row, 1].imshow(scores[row].float().cpu(), cmap="magma")
    axes[row, 1].set_title("saliency" if row == 0 else "", fontsize=9)
    for column, mode in enumerate(modes):
        hidden = automask.sample_mask(scores, ratio=MASK_RATIO, mode=mode,
                                      hint=MASK_HINT)
        blanked = automask.blank(batch.student, hidden)
        axes[row, 2 + column].imshow(
            pretrain.to_pixels(blanked[row:row + 1])[0]
            .permute(1, 2, 0).clamp(0, 1).float().cpu())
        axes[row, 2 + column].set_title(mode if row == 0 else "", fontsize=9)
for axis in axes.ravel():
    axis.axis("off")
plt.tight_layout(); plt.show()

del preview_model, clean, scores
import gc; gc.collect()
if DEVICE.startswith("cuda"):
    torch.cuda.empty_cache()
print("If the configured mode hides only the targets and leaves the ground, "
      "lower MASK_RATIO or switch to 'green' -- a mask that removes every "
      "object is not a hard task, it is an impossible one.")''',
    invoke='''ARGS = (f"--arm {ARM} --mask-mode {MASK_MODE} --mask-ratio {MASK_RATIO} "
        f"--mask-hint {MASK_HINT} --saliency {SALIENCY} "
        f"--target {TARGET} --target-decay {TARGET_EMA}")''',
    report='''# --- Did the representation survive? ------------------------------------
# The failure this checks for is the one a loss curve hides completely: an EMA
# self-distillation that collapses has a loss falling beautifully toward zero
# while every position in the map becomes the same vector.
from src.training.automask import summarise_collapse

print(summarise_collapse(RESULT["history"]))''',
    appendix='''## What this arm is, and what to expect from it

**The method, in one line.** A copy of the encoder sees the clean frame and
produces the target map; the encoder being trained sees the frame with half of
its 16-pixel cells blanked to the mean pixel and must reproduce that map at the
blanked positions. The mask is chosen from the target's own feature map — the
cells whose features are furthest from the frame's average direction — which is
the convolutional stand-in for AttMask's "mask what the class token attends
to", since a RepViT trunk has no attention map to read.

**Why there is no sparse convolution.** ConvNeXt V2's FCMAE needed it because a
dense convolution over a masked image leaks: the kernel straddles the boundary
and the hole's shape propagates, so the network can learn *where* the mask was.
That argument bites hardest when the target is pixels, because a model that
knows where the hole is can hedge. Here the target is a feature map and there
is no decoder, so the leak buys much less — and the cost of the alternative is
re-implementing a trunk this repo does not own.

**But the leak is not zero, and one part of it is named.** RepViT's later
stages carry squeeze-and-excitation blocks, and an SE gate is a global average
over the map. Averaged over a masked input that gate is biased by the visible
fraction, and at stage B there is no mask at all. The mitigations here are the
cheap ones — the fill is the mean pixel, which is the least informative value
available, and `MASK_RATIO` defaults to 0.5 rather than MAE's 0.75. If this arm
wins, patching the pooling to count only visible positions (what SparK does) is
the first thing to try next.

## The honest prior

Four measured numbers, none of them ours, all pointing the same way:

| what was measured | result |
|---|---|
| ConvNeXt V2's FCMAE on an **unmodified** architecture (its own Table 14) | −0.1 to +0.1 top-1. The headline +1.0 came from co-designing the GRN layer alongside it |
| MAE pretraining a **5.7 M** ViT (TinyMIM) | **−0.6** against training from scratch |
| a learned mask against a random one, everything else fixed (HPM's isolation ablation) | +0.46; and +0.26 of its +0.72 was simply adding the auxiliary predictor |
| aerial thermal segmentation at **4.8 M** parameters (Caltech RGB-T, ECCV 2024) | scratch 0.687, **ImageNet 0.725**, self-supervision on 40 k thermal images 0.714 |

The last row is the one to sit with: on this project's model scale and close to
its domain, ordinary RGB pretraining beat thermal self-supervision. That does
not make this arm pointless — it makes it a *measurement*, and the thing it
measures is whether a thermal-only corpus in the deployment's own modality can
add anything the other arm structurally cannot reach.

`MASK_MODE = "green"` is the setting worth respecting here. It is ColorMAE
(ECCV 2024): band-pass-filtered noise, no parameters, no extra forward pass,
and in its own controlled comparison it matched the learned maskers of the
three years before it. If `hint` does not beat `green` on stage B's number,
the self-generated mask earned nothing and the honest thing is to say so.

## Reading the collapse table

`similarity` is the mean cosine between two *different* positions in the same
frame. A dense map is useful precisely because its positions disagree; a number
approaching 1 means they have stopped, and the checkpoint is then worse than
doing nothing whatever the loss says. `channel_std` is the same failure seen
from the other side. `visible` is a sanity check rather than an objective — the
student can see those pixels, so it should sit near 1 and stay there.

If it collapses: lower `MASK_RATIO`, raise `TARGET_EMA` toward 0.9995, or set
`TARGET = "frozen"`, which cannot collapse because the target never moves.''',
)

VARIANTS["teacher"] = Variant(
    key="teacher",
    path="notebooks/16_pretrain_convnext.ipynb",
    arm="distil",
    title="Stage A — a frozen DINOv3 ConvNeXt teaching the thermal encoder",
    blurb=(
        "Runs the **teacher arm**: a frozen foundation model looks at one half "
        "of the scene, the encoder looks at the other, and the encoder is "
        "trained to reproduce the dense feature map the teacher saw. The "
        "default teacher is **DINOv3 ConvNeXt-Small (50 M)** — chosen for its "
        "*ratio* to a 6 M student rather than its size, which is the one "
        "teacher-selection finding that has been measured more than once. "
        "Before it trains anything it measures whether an RGB teacher shown a "
        "thermal frame is describing the scene at all, because that number "
        "decides which of your datasets this arm can even eat."),
    needs="a CUDA GPU, ~6 GB of free disk at the default corpora, and a "
          "**Hugging Face token** — every `facebook/dinov3-*` checkpoint is "
          "gated",
    takes="~10 min to download, ~3 min for the modality probe in cell "
          "{{check}}, and 45–120 min to train at the default 4 000 steps",
    settings='''ARM     = "distil"           # none | mask | distil | both
TEACHER = "facebook/dinov3-convnext-small-pretrain-lvd1689m"
TEACHER_SIZE = None          # None derives it from the teacher's *measured*
                             # stride so its map is never coarser than the
                             # student's: 1024 for a stride-32 ConvNeXt at
                             # SIZE 512, 512 for a /16 ViT.
LOSS    = "cosine"           # cosine | feature | frequency
LAYERS  = 1                  # align against the teacher's last N blocks
GRAM    = 0.0                # weight of the relational term; try 1.0
JITTER  = 0                  # pixels of teacher-window offset; try 2-4 on
                             # DroneVehicle, whose pairs are not pixel-exact
TOLERANCE = 0                # cosine-only: match against the best cell in a
                             # (2r+1)^2 neighbourhood instead of exactly p''',
    gate='''# --- Hugging Face: DINOv3 is gated --------------------------------------
# Every facebook/dinov3-* repository requires accepting its terms once, on the
# model page, with the same account as the token below. Done here rather than
# at first use: the download happens forty minutes in, and a 401 there costs
# the session.
from huggingface_hub import login, whoami

login()
print("logged in as", whoami()["name"])

from huggingface_hub import model_info
try:
    model_info(TEACHER)
    print(f"{TEACHER}: reachable")
except Exception as exc:
    raise SystemExit(
        f"cannot reach {TEACHER}: {exc}\\n\\n"
        f"Open https://huggingface.co/{TEACHER} , accept the terms, and re-run "
        f"this cell. An ungated fallback that needs no account at all: "
        f"TEACHER = 'facebook/dinov2-base'.")''',
    check_md='''## The measurement that decides which datasets this arm can eat

The teacher is an RGB model. The deployment is thermal. Registered pairs let
the teacher work on the modality it was trained on — but registered aerial
RGB-T pairs are scarce, and the large thermal sets have no RGB twin at all.
So: *is an RGB foundation model shown a raw thermal frame describing the scene,
or producing noise?*

That is a number, and it is cheap. On registered pairs, compare the teacher's
own feature map on the RGB half against its map on the thermal half, and read
that against two references — how far apart two **unrelated** frames land
(chance) and how far the same frame lands from itself under a mild brightness
change (this teacher's own **floor**). A third row, the same frame in
**grayscale**, says what losing colour alone costs, which is the part of the
thermal gap that is not really about thermal.

There is a published answer to expect, and it is split. On this project's own
domain — Caltech's aerial RGB-T set, ECCV 2024 — a foundation model's
**outputs** collapse on thermal (SAM's instance masks score AP 0.018 against
0.538 on grayscale), while its **features** mostly survive (a frozen DINOv2
read with a nonlinear head reaches 0.706 mIoU on thermal segmentation against
0.725 for the best network trained end to end on thermal, and beats the
thermal-specific FTNet's 0.613). This stage copies features and never touches a
prediction, which is why the question is worth asking rather than assuming.

The verdict the cell prints is a rule, not a judgement, so two runs cannot
disagree about what the same numbers meant.''',
    check='''# --- How much of the teacher survives a thermal input? ------------------
from src.training import pretrain
from src.training.distill import build_teacher

probe_teacher = build_teacher(TEACHER, device=DEVICE, size=TEACHER_SIZE)
print(f"{TEACHER}\\n  {probe_teacher.dim}-d at stride {probe_teacher.patch}, "
      f"input {probe_teacher.size}, grid {probe_teacher.grid} "
      f"(the student's is {SIZE // 16})\\n")

GAP = pretrain.modality_gap(probe_teacher, ITEMS, size=SIZE,
                            samples=PROBE_SAMPLES, batch=BATCH, seed=SEED,
                            device=DEVICE)

import json as _gapjson
(WORK / "modality_gap.json").write_text(_gapjson.dumps(GAP, indent=2))
import shutil as _gapshutil
_gapshutil.copy(WORK / "modality_gap.json", MIRROR / "modality_gap.json")

del probe_teacher
import gc; gc.collect()
import torch
if DEVICE.startswith("cuda"):
    torch.cuda.empty_cache()''',
    invoke='''ARGS = (f"--arm {ARM} --teacher {TEACHER} --loss {LOSS} "
        f"--layers {LAYERS} --gram {GRAM} --jitter {JITTER} "
        f"--tolerance {TOLERANCE}"
        + (f" --teacher-size {TEACHER_SIZE}" if TEACHER_SIZE else ""))''',
    report='''# --- What the teacher arm produced --------------------------------------
print(f"teacher      {RESULT['teacher']}")
print(f"loss         {RESULT['loss']}")
print(f"final        {RESULT['final_loss']:.4f}")
print()
print("Per epoch:")
for row in RESULT["history"]:
    print("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                           for k, v in row.items()))
print()
print("This number is a proxy and a small one. Two runs with different "
      "teachers do not even share a scale -- a cosine distance to "
      "ConvNeXt-Small's 768-d map and one to ViT-B's 768-d map are different "
      "quantities. Only stage B ranks them.")''',
    appendix='''## Which teacher, and why the default is what it is

`TEACHER` is the setting this notebook exists to vary, and the evidence behind
it is worth stating plainly because it is thinner than it looks.

**The one finding that has been measured more than once:** teacher capacity
should be *matched* to the student, not maximised. EdgeCrafter (arXiv
2603.18739, TMLR 2026) holds a compact student fixed and swaps the teacher —
DINOv3-S (21 M) **54.0** COCO AP, DINOv3-B (86 M) **54.3**, DINOv3-L (300 M)
**52.6**. An inverted U with the peak around 86 M, and a real cost for
overshooting. Proteus (ICLR 2025) reports the same shape more steeply for a
21 M student: 86 M teacher 85.8, 300 M teacher 82.2, 1.1 B teacher 80.6,
degrading consistently across 11 of 12 datasets.

**What that does *not* settle.** Both are single runs, and the classic
capacity-gap result they echo is a *logit*-distillation finding; the one
multi-seed study that tested a feature-based objective found the trend flatten
and the optimum move to a **much larger** teacher. TinyMIM, at exactly this
student scale (5.7 M), found an 86 M teacher beating a 21 M one. So the useful
reading is "somewhere between 5x and 15x the student, and 50x is the danger
zone" — which for a 6 M encoder is **30 M to 90 M**, and every candidate below
sits inside it except the last.

| teacher | params | ratio to 6 M | ADE20k mIoU (DINOv3's own table) | stride |
|---|---|---|---|---|
| `dinov3-convnext-tiny-…` | 29 M | 4.8x | 42.7 | 32 |
| **`dinov3-convnext-small-…`** *(default)* | **50 M** | **8x** | **44.8** | 32 |
| `dinov3-convnext-base-…` | 89 M | 15x | 46.3 | 32 |
| `dinov3-vits16-…` | 21 M | 3.5x | 47.0 | 16 |
| `dinov3-vitsplus16-…` | 29 M | 4.8x | **48.8** | 16 |
| `dinov3-vitb16-…` | 86 M | 14x | **51.8** | 16 |
| `dinov3-vitl16-…` | 300 M | 50x | 54.9 | 16 |

Read the last two columns together and the honest conclusion is uncomfortable
for this notebook's own title: **at matched parameters DINOv3's ViT students
carry visibly stronger dense features than its ConvNeXt ones** — 48.8 against
42.7 at 29 M — and ViT-B/16 is both the measured optimum in the two studies
above and, at stride 16, the *cheaper* option here, because a stride-32
ConvNeXt has to be run at 1024 to produce the student's 32x32 grid.

So why is a ConvNeXt the default? Because the one thing those numbers cannot
tell us is whether **conv-to-conv** transfers better into a convolutional
student than ViT-to-conv does, and nobody has run that experiment. It is one
string and one re-run to find out, and both runs write the same artifact, so
stage B ranks them directly. If you only have time for one, run `vitb16`.

**What is not in doubt:** do not use ViT-L/16 or anything above it. 50x is the
ratio at which both measured studies lost ground.

## Why the ConvNeXt teacher is fed 1024 and the ViT 512

A ConvNeXt's final stage is stride **32**; the student's map is stride **16**.
Fed the same 512 the teacher would produce a 16x16 map against the student's
32x32, and the loss would upsample it — supervising four student cells with one
teacher cell, which throws away exactly the resolution that matters when a
target is twenty pixels across. `TEACHER_SIZE = None` therefore resolves to
`(512 / 16) x 32 = 1024`, giving the teacher a 32x32 map over the same field of
view. It costs a forward pass on an upsampled image, under `no_grad` and
bfloat16, and buys an exact correspondence.

The stride is **measured**, not read off the config: DINOv3's ConvNeXt configs
carry no `patch_size` field at all, so anything that reads one gets a default
and is wrong by a factor of two. `distill.probe_geometry` runs the backbone at
128 and at 256 and solves for the stride from how the token count grows.

## The losses, and which of them was measured on this exact problem

| `LOSS` | what it does | evidence |
|---|---|---|
| `cosine` *(default)* | per-position cosine distance | this repo's own, and the safe choice: scale-free, and the only simple objective that has not been reported going backwards cross-modally |
| `feature` | whitened smooth-L1, beta 2.0 | the canonical published feature-distillation recipe (arXiv 2205.14141) — but validated **same-modality**, teacher and student on identical inputs |
| `frequency` | hard on low spatial frequencies, saturating on high ones | measured on RGB-to-thermal distillation from a DINO-family teacher (arXiv 2606.11572): **64.1 mAP50 against 62.1 for cosine, 61.1 for plain MSE and 61.7 for not distilling at all** |

That last row is the most useful number in this table and most of it is a
warning: **uniform full-band MSE scored below not distilling**, and
response-level distillation scored 2.9 below it. The measured reason is that
the two modalities' features diverge about 2.4x more in the high band than the
low — a thermal image's fine texture is genuinely different, its layout is not.

`GRAM` adds a relational term: match the *similarity matrix between positions*
rather than the positions themselves. DINOv3 uses one (Gram anchoring, weight
2) and relational distillation beat plain similarity distillation by +2.1 to
+2.7 zero-shot mIoU in an independent three-run study. It is also the one
spatial objective that tolerates misregistration, because it never asks what is
at position *p* — only whether *p* and *q* look alike.

`JITTER` is the other half of that. These archives are not registered to the
pixel: DroneVehicle's authors report over 20 % of boxes deviating by more than
3 px or 3 degrees. The measured damage curve (AR-CNN, ICCV 2019) is that 1-2 px
is free, 4 px costs a few points and 6 px costs about 65 % relative — and that
paper's remedy was to *train under* the shift, which cut the variance across
shifts from 11.55 to 1.24. `JITTER = 2` is that remedy.

## The corpora, and what each one is actually for

`paired` is the only route with real cross-modal grounding and it is the
scarcest. `rgb` is a **proxy** and is labelled as one: the student sees the
luminance of an image whose colour the teacher sees, which keeps the shape of
the task — one channel in, three-channel semantics out — while being honest
that the sensor is wrong. It is also the route to verify the plumbing on first,
because a pipeline that cannot learn luminance-to-colour is broken for reasons
that have nothing to do with thermal. `thermal` is the route cell {{check}}
decides.

One warning from the literature that applies directly to a run configured with
one corpus: AnyThermal reports that a variant distilled **only** on aerial data
came out *worse than the undistilled RGB model* on urban scenes. Narrow
distillation damages what it does not cover. Mix the corpora.''',
)


def swap(text: str, old: str, new: str) -> str:
    """`str.replace` that refuses to do nothing."""
    if old not in text:
        raise AssertionError(f"nothing matched:\n{old[:200]}")
    return text.replace(old, new)


# --------------------------------------------------------------------------
# The cells
# --------------------------------------------------------------------------

md(r"""
# {{TITLE}}

One runtime, one **Run all**: download unlabelled aerial imagery, pretrain
EdgeTAM's image encoder, and leave an ordinary EdgeTAM checkpoint on your Drive
that `tools/train_encoder.py --base` accepts unchanged.

{{BLURB}}

| | |
|---|---|
| **in** | nothing staged by hand — the download cell fetches everything |
| **out** | `edgetam_stage_a_{{ARM}}_512.pt` + a `stage_a.json` manifest, on your Drive |
| **needs** | {{NEEDS}} |
| **takes** | {{TAKES}} |

---

## Stage A is a choice between four arms, and this notebook runs one of them

| `ARM` | what supervises the encoder | which data it can eat |
|---|---|---|
| `none` | nothing — the stock EdgeTAM weights, copied | none |
| `mask` | the encoder's own features on the unmasked frame | **anything**; one modality is enough |
| `distil` | a frozen teacher's features on a paired input | needs a teacher-side image |
| `both` | `distil` first, then `mask` on top | whatever `distil` can eat |

`ARM` is a setting, so either notebook can run any of the four. **All four
write the same file**: an ordinary EdgeTAM state dict plus a manifest naming
what produced it. That is the whole architecture of this stage — one artifact,
four ways of making it, and a single line to feed it forward:

```
python tools/train_encoder.py --base <this notebook's checkpoint> ...
```

`none` writes one too, and that matters more than it sounds. **A baseline
reached by a different code path is a baseline that drifts** — stage B loading
stock weights from one place and pretrained weights from another will
eventually differ in something nobody meant to change. So the baseline is a
file, in the same folder, with the same manifest beside it.

## The number this notebook produces is not the number that decides anything

Every arm ends with a loss. That loss is a *proxy*, and two arms' proxies do
not even share a scale — a cosine distance to a foundation model's features and
a masked-position cosine to the model's own are different quantities. What
decides whether stage A was worth its GPU hours is **stage B run from this
checkpoint against stage B run from the `none` arm's**, and nothing here can
substitute for it.

That warning is not boilerplate. On this project's model scale (a 4.8 M
encoder) and close to its domain (aerial thermal segmentation), the Caltech
RGB-T paper measured: scratch 0.687, ordinary **ImageNet 0.725**, and
self-supervised pretraining on 40 k thermal images 0.714. Ordinary RGB
pretraining beat the domain-specific self-supervision. Run `none` first.
""")

# ---------------------------------------------------------------- 1
code(r"""
# --- Runtime, repo, GPU -------------------------------------------------
import os, sys
from pathlib import Path

REPO   = Path("/content/sam-dedection")
BRANCH = "claude/pretrain-notebook-sota-ndc37w"

if not REPO.exists():
    !git clone -q -b {BRANCH} https://github.com/yigitkayabagci/sam-dedection.git {REPO}
!git -C {REPO} fetch -q origin {BRANCH}
!git -C {REPO} checkout -q {BRANCH} && git -C {REPO} merge -q --ff-only origin/{BRANCH}
os.chdir(REPO)
sys.path.insert(0, str(REPO))

# Drop this repo's modules so the fast-forward above actually takes effect:
# `git pull` changes files on disk, not modules Python already imported.
for _stale in [n for n in list(sys.modules) if n.split(".")[0] in ("src", "tools")]:
    del sys.modules[_stale]

!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
!df -h /content | tail -1

# --- Is this notebook the one the repo expects? -------------------------
# The repo just fast-forwarded itself; the .ipynb is a file you uploaded, and
# the two drift apart silently. This says so at cell 1 instead of an hour in.
import json as _json
NOTEBOOK = "{{NOTEBOOK}}"
STAMP    = "{{STAMP}}"
_stamps  = REPO / "notebooks" / ".stamps.json"
_want    = _json.loads(_stamps.read_text()).get(NOTEBOOK) if _stamps.is_file() else None
if _want and _want != STAMP:
    print("\n" + "=" * 74)
    print(f"!!  STALE NOTEBOOK -- this file is build {STAMP}, the repo ships {_want}")
    print( "!!  Every cell below is the old version. Re-download and re-upload:")
    print(f"!!    https://github.com/yigitkayabagci/sam-dedection/raw/{BRANCH}/notebooks/{NOTEBOOK}")
    print("=" * 74 + "\n")
else:
    print(f"notebook build {STAMP} matches the repo")
""")

# ---------------------------------------------------------------- 2
code(r"""
# --- Does this torch actually have kernels for this GPU? ----------------
# Asked by launching one, not by reading an arch list: on a card newer than
# the torch build everything imports and the first real matmul dies mid-run.
import torch

print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"{torch.cuda.get_device_name(0)}  sm_{major}{minor}  "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
    probe = torch.randn(256, 256, device="cuda")
    (probe @ probe).sum().item()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        (probe @ probe).sum().item()
    print("a real matmul ran, in float32 and bfloat16 -- this GPU is usable")
    del probe
else:
    print("!! no CUDA device -- nothing below will train")
""")

# ---------------------------------------------------------------- 3
code(r"""
# --- Dependencies -------------------------------------------------------
# EdgeTAM *is* installed here, unlike the mask-pool notebooks: this stage
# trains EdgeTAM's own image encoder, so the package has to be importable.
# EdgeTAM installs itself as `sam2`, so Meta's SAM 2 must never share this
# runtime; the teacher (when there is one) runs through transformers, which
# carries an independent implementation under a different name.
before = torch.__version__
!bash scripts/setup_edgetam.sh 2>&1 | tail -5
!pip install -q -r requirements.txt
!pip install -q "transformers>=4.56" hf_transfer tqdm

# The one thing an install here must not do: replace the preinstalled torch.
# Read the version off *disk* -- reloading torch always raises, and pip cannot
# change the torch already live in this kernel. Only a restart can, which is
# exactly the window this check exists to catch.
import importlib.metadata as _md
installed = _md.version("torch")
assert installed == before, (
    f"pip replaced torch {before} with {installed} on disk. This kernel is "
    f"still running {before}, so nothing has broken yet -- restore it *before* "
    f"restarting:  !pip install -q --force-reinstall torch=={before}")

# An editable install is a .pth file that site.py reads at interpreter startup
# only, so a package installed into a running kernel never joins sys.path.
# Pointing at the checkout fixes it now rather than costing a restart.
EDGETAM = REPO / "third_party" / "EdgeTAM"
if str(EDGETAM) not in sys.path:
    sys.path.append(str(EDGETAM))
BASE_CKPT = EDGETAM / "checkpoints" / "edgetam.pt"
assert BASE_CKPT.is_file(), \
    "edgetam.pt did not download -- rerun scripts/setup_edgetam.sh and read its output"

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import transformers
print(f"transformers {transformers.__version__}, EdgeTAM at {EDGETAM}")

# The contracts everything below depends on, tested with no GPU and no data.
# If these fail, nothing after this point is worth running.
!python -m unittest -q tests.test_pretrain tests.test_automask 2>&1 | tail -3
""")

# ---------------------------------------------------------------- 4
code(r"""
# --- Drive: where the checkpoint and its manifest land ------------------
# Training reads a few hundred thousand small JPEGs in random order and the
# Drive FUSE mount serves those an order of magnitude slower than the GPU
# consumes them, so the *data* stays on local disk. Drive holds the output,
# which is megabytes.
from google.colab import drive

drive.mount("/content/drive")
MIRROR = Path("/content/drive/MyDrive/edgetam-pretrain/{{ARM}}")
MIRROR.mkdir(parents=True, exist_ok=True)
print("output ->", MIRROR)
""")

# ---------------------------------------------------------------- 5
md(r"""
## Settings

### The corpora, and why a corpus is one line

`CORPORA` is the list this stage trains on and it is meant to be **edited**.
A row is a root, a modality and either a spec name that `aerial.SPECS` already
knows or a pair of globs spelled out. Nothing else has to be touched when you
add a dataset: the crop is derived from the pictures actually on disk, the
pairing key is the one `list_pairs` already gets right, and the index cell
prints what it found per corpus — including a loud `!!` for a glob that matched
nothing, which is the way this goes wrong.

Three modalities, three routes, and the routes are the substance:

| modality | teacher sees | student sees | what it is |
|---|---|---|---|
| `paired` | the RGB half | the registered thermal half | the published route, and the only one with real cross-modal grounding |
| `rgb` | the colour image | **its own luminance** | a proxy: same one-channel-in shape, wrong sensor. This is what lets an aerial RGB set with no thermal twin contribute |
| `thermal` | the thermal frame | the same thermal frame | an RGB teacher off its home modality — worth what cell {{check}} says it is worth |

Feeding the student full colour on an `rgb` corpus would teach it to use a
channel the sensor does not have, and the deployed encoder has never seen
colour in its life. Graying the student's half instead keeps the structure of
the real task.

### The schedule

`STEPS` x `BATCH` is what actually sets the length; `EPOCHS` reshuffles. One
pass over the default corpora at batch 8 is a few thousand steps, so 4 000 is
roughly one pass — and a *starved* stage A is the failure this project has
already seen once: at 600 steps and 20 000 pairs, notebook 08's distilled arm
came out **below** the run that skipped stage A entirely.
""")

# ---------------------------------------------------------------- 6
code(r"""
# --- Shared settings ----------------------------------------------------
from src.training.pretrain import Corpus

DATA_ROOT = Path("/content/data")
WORK      = Path("/content/work"); WORK.mkdir(parents=True, exist_ok=True)

SIZE   = 512        # the student's input, and the resolution it deploys at
BATCH  = 8
STEPS  = 4000       # batches per epoch
EPOCHS = 1
SEED   = 0
DEVICE = "cuda"
LIMIT  = None       # cap per corpus, spread across it rather than truncated
PROBE_SAMPLES = 256 # windows the modality probe reads
TRUNK_LR = 5e-5
NECK_LR  = 1e-4

# --- The corpora --------------------------------------------------------
# Add a row as you find data. `spec=` borrows an existing DatasetSpec's globs
# and border; give `thermal=`/`rgb=` instead for a set nobody has written one
# for. `limit=` caps a corpus so a huge one cannot drown the others.
CORPORA = [
    # Registered RGB-T. 640x512 once the 100 px white band is cropped -- this
    # project's native resolution, so no resampling at all -- day and night.
    Corpus("dronevehicle", DATA_ROOT / "DroneVehicle", "paired",
           spec="dronevehicle", limit=20000),
    # Registered RGB-T, 4 024 pairs at 640x512, with drawn masks. Small, and
    # the probe's home: its halves are the cleanest registration on the list.
    Corpus("kust4k", DATA_ROOT / "Kust4K", "paired", spec="kust4k"),
    # Aerial RGB, no thermal twin. The proxy route, and the diversity that
    # keeps a narrow distillation from damaging what it does not cover.
    Corpus("visdrone", DATA_ROOT / "VisDrone", "rgb",
           rgb="**/images/*.jpg", limit=6000),
    # Thermal only, aerial, 80-130 m. The mask arm's natural food, and the
    # corpus whose usefulness to the teacher arm cell {{check}} decides.
    Corpus("hituav", DATA_ROOT / "HIT_UAV", "thermal",
           thermal="**/normal_json/**/*.jpg"),
]
OUT = WORK / f"edgetam_stage_a_{{ARM}}_{SIZE}.pt"
print(f"{len(CORPORA)} corpora configured -> {OUT.name}")
""")

# ---------------------------------------------------------------- 7
code(r"""
# --- This arm's settings ------------------------------------------------
{{SETTINGS}}
""", tag="settings")

# ---------------------------------------------------------------- 8
code(r"""
{{GATE}}
""", tag="gate")

# ---------------------------------------------------------------- 9
md(r"""
## The data

Every URL is baked into `tools/fetch_datasets.py` and was checked against the
live host, because each of these sets is served in a way that defeats the
obvious approach. Nothing here is staged by hand.

Stage A reads **no labels at all**, which is why these sets carry far more
usable frames than annotated ones — and why a set whose annotations are the
wrong shape for segmentation (DroneVehicle's oriented boxes) is perfectly good
here.
""")

# ---------------------------------------------------------------- 10
code(r"""
# --- Download -----------------------------------------------------------
# (name, destination, parts). A part list of [] takes the recipe's defaults.
FETCH = [
    ("dronevehicle", DATA_ROOT / "DroneVehicle", ["train"]),
    ("kust4k",       DATA_ROOT / "Kust4K",       ["tir", "rgb"]),
    ("visdrone",     DATA_ROOT / "VisDrone",     ["train"]),
    ("hituav",       DATA_ROOT / "HIT_UAV",      []),
]

for name, dest, parts in FETCH:
    have = dest.exists() and any(p.suffix.lower() in (".jpg", ".png", ".jpeg")
                                 for p in dest.rglob("*") if p.is_file())
    if have:
        print(f"{name}: already on disk at {dest}")
        continue
    extra = " --parts " + " ".join(parts) if parts else ""
    !python tools/fetch_datasets.py {name} --dest {dest}{extra} 2>&1 | tail -4

!df -h /content | tail -1
""")

# ---------------------------------------------------------------- 11
code(r"""
# --- What will actually train -------------------------------------------
# Read off disk, not off the settings: a glob that matched nothing is the way
# this goes wrong, and it is loud here rather than silent an hour later.
from src.training import pretrain

ITEMS = pretrain.index(CORPORA, size=SIZE, seed=SEED)
print()
print(pretrain.summarise(ITEMS))
print()
print(f"one pass at batch {BATCH} is {len(ITEMS) // BATCH:,} steps; "
      f"STEPS={STEPS} and EPOCHS={EPOCHS} asks for "
      f"{STEPS * EPOCHS:,} ({STEPS * EPOCHS * BATCH / max(len(ITEMS), 1):.2f} passes)")
""", tag="index")

# ---------------------------------------------------------------- 12
md(r"""
{{CHECK_MD}}
""")

# ---------------------------------------------------------------- 13
code(r"""
{{CHECK}}
""", tag="check")

# ---------------------------------------------------------------- 14
md(r"""
## Train

Everything below goes through `tools/pretrain_stage_a.py`, the same entry point
every arm uses, so the freeze, the parameter groups, the one-cycle schedule,
the EMA and the checkpoint format are identical across arms by construction —
which is the only reason two arms' stage-B numbers can be set beside each
other.

Only the trunk and the neck move. The mask decoder, the prompt encoder and the
whole memory path stay frozen: stage A is an encoder stage, and the memory path
is the write port of a recurrent loop where error accumulates.
""")

# ---------------------------------------------------------------- 15
code(r"""
# --- Run the arm --------------------------------------------------------
{{INVOKE}}

CORPORA_JSON = WORK / "corpora.json"
import json as _cjson
CORPORA_JSON.write_text(_cjson.dumps(
    [{"name": c.name, "root": str(c.root), "modality": c.modality,
      "spec": c.spec, "thermal": c.thermal, "rgb": c.rgb, "route": c.route,
      "limit": c.limit} for c in CORPORA], indent=2))

REPORT_JSON = WORK / "stage_a_report.json"
!python tools/pretrain_stage_a.py {ARGS} \
    --corpora {CORPORA_JSON} --base {BASE_CKPT} --out {OUT} \
    --size {SIZE} --batch {BATCH} --steps {STEPS} --epochs {EPOCHS} \
    --trunk-lr {TRUNK_LR} --neck-lr {NECK_LR} --seed {SEED} \
    --device {DEVICE} --json {REPORT_JSON}

RESULT = _cjson.loads(REPORT_JSON.read_text())
print(f"\n{RESULT['arm']}: {RESULT['steps']} steps over {RESULT['samples']:,} "
      f"samples in {RESULT['seconds'] / 60:.0f} min")
""", tag="train")

# ---------------------------------------------------------------- 16
code(r"""
{{REPORT}}
""")

# ---------------------------------------------------------------- 17
code(r"""
# --- Stage it, with the manifest beside it ------------------------------
# The manifest is the point: stage B takes a path and asks no questions, so
# without it the record of which arm, which corpora and which seed produced a
# checkpoint lives in a Colab cell that scrolls away.
import shutil

from src.training.pretrain import read_manifest, stage_b_command

STAGED = MIRROR / OUT.name
shutil.copy(OUT, STAGED)
MANIFEST = OUT.with_suffix(".stage_a.json")
if MANIFEST.is_file():
    shutil.copy(MANIFEST, MIRROR / MANIFEST.name)
shutil.copy(REPORT_JSON, MIRROR / REPORT_JSON.name)

print(f"{STAGED}  ({STAGED.stat().st_size / 2**20:.0f} MiB)")
for key, value in sorted(read_manifest(OUT).items()):
    if key != "history":
        print(f"  {key:<14} {value}")

print()
print("Feed it to stage B with:")
print(stage_b_command(STAGED))
""", tag="stage")

# ---------------------------------------------------------------- 18
md(r"""
## Handing this to stage B

The checkpoint is an **ordinary EdgeTAM state dict** — same keys, same loader,
same exporter — with a differently-trained trunk and neck inside it. Nothing
downstream needs to know this stage happened, which is exactly what makes the
comparison possible:

```
# the arm you just ran
python tools/train_encoder.py --base <MIRROR>/edgetam_stage_a_{{ARM}}_512.pt \
    --out checkpoints/stage_b_{{ARM}}.pt --dataset ...

# the baseline it has to beat, same everything else
python tools/pretrain_stage_a.py --arm none --out work/stage_a_none.pt
python tools/train_encoder.py --base work/stage_a_none.pt \
    --out checkpoints/stage_b_none.pt --dataset ...
```

Notebook `07_encoder_aerial_rgbt.ipynb` is stage B; it runs its own inline
stage A today, so point its `--base` at this file to use this one instead. Hold
**everything else** fixed — the datasets, the schedule, the seed, the prompt
mix — or the two `test/instance_iou` numbers are not comparable and the
comparison was the only reason to run this.

### What a result looks like

Three numbers, from three stage-B runs that differ in one flag:

| stage A | stage B `test/instance_iou` |
|---|---|
| `none` | — |
| this arm | — |
| the other notebook's arm | — |

If this arm does not beat `none`, it cost GPU hours and bought nothing, and
that is a finding worth writing down rather than a run worth hiding. The
literature is full of exactly that outcome at this model scale.

---

{{APPENDIX}}
""")


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def stamp_for(variant: Variant) -> str:
    """The build id: this variant's notebook, hashed -- cells and fields both,
    so a swapped pair or a changed setting cannot stamp identically."""
    body = "\n".join(f"{kind}\n{text}" for kind, text in CELLS)
    body += "\n".join(str(getattr(variant, f.name)) for f in fields(variant))
    return hashlib.sha256(body.encode()).hexdigest()[:10]


def build() -> dict:
    global STAMP_VALUE
    STAMP_VALUE = stamp_for(V)
    return {
        "cells": [
            {"cell_type": kind,
             "metadata": {},
             "source": (resolve(text) + "\n").splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if kind == "code" else {})}
            for kind, text in CELLS
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    if len(sys.argv) > 1:
        key = sys.argv[1]
        if key not in VARIANTS:
            raise SystemExit(f"unknown variant {key!r} -- have {sorted(VARIANTS)}")
        V = VARIANTS[key]
        out = repo / V.path
        out.parent.mkdir(parents=True, exist_ok=True)
        document = build()
        out.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n")

        stamps = repo / "notebooks" / ".stamps.json"
        known = json.loads(stamps.read_text()) if stamps.is_file() else {}
        known[out.name] = STAMP_VALUE
        stamps.write_text(json.dumps(known, indent=1, sort_keys=True) + "\n")
        print(f"wrote {out.relative_to(repo)} with {len(CELLS)} cells "
              f"[{STAMP_VALUE}]")
    else:
        # Re-run per variant: the cells are emitted at import time against a
        # single `V`, and a second pass in-process would append to the first.
        for key in VARIANTS:
            subprocess.run([sys.executable, __file__, key], check=True, cwd=repo)
