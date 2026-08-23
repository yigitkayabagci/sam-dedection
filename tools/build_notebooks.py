#!/usr/bin/env python3
"""Build the encoder notebooks from plain text blocks.

    python tools/build_notebooks.py          # all of them
    python tools/build_notebooks.py vtuav    # just that one

Four notebooks come out of this one file. All four run the same code on the
same schedule and the same seed; each changes exactly one thing, so the gap
between any pair of final numbers is attributable to that one thing:

    07_encoder_aerial_rgbt.ipynb           the reference run
    08_encoder_vtuav_only.ipynb            stage B on VTUAV alone  (data)
    09_encoder_stage_a_dronevehicle.ipynb  stage A on DroneVehicle (pairs)
    10_encoder_teacher_dinov3.ipynb        stage A from DINOv3     (teacher)

Every variant writes to its own Drive folder and its own index directory, and
none of them touch the others' outputs -- running any number of them at once,
on separate runtimes, is the intended way to use this file.

The notebooks are generated rather than edited because a notebook is a JSON
document where prose, code and cell numbers drift apart silently: three
`.replace()` calls in an earlier version of this file matched nothing, and the
notebook shipped referring to six variables no cell defined. `swap()` now
refuses to match nothing, `{{tag}}` resolves cell cross-references at build
time, and `tools/check_notebook.py` walks the result for names used before they
are bound.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field, fields
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
    blurb: str                     # one paragraph under the title
    datasets: str                  # the body of the DATASETS list
    fetch: str                     # the body of the download loop
    mirror: str                    # Drive folder for weights, per variant
    distill: str                   # the stage-A block: source, crop, tolerance
    teacher: str = "facebook/sam2.1-hiera-base-plus"
    gpu: str = "A100"


VARIANTS: dict[str, Variant] = {}

SAM_NOTE = """**DINOv3 is the other arm, not a dead option.** `TEACHER` is one string, and
everything else — data, seed, schedule, loss, held-out sequences — is held
fixed, so the difference between the two runs is the teacher and nothing else.
`10_encoder_teacher_dinov3.ipynb` is this notebook with that one string
changed; it can run on a second runtime at the same time as this one. DINOv3 is
gated: open its model page once, accept the terms, then
`from huggingface_hub import login; login()` in that runtime. AnyThermal's
published result used DINOv2, which is why `facebook/dinov2-base` is still one
string away and needs no account."""

DINO_NOTE = """**DINOv3 is this notebook's teacher, and 07 is the arm it is measured
against.** `TEACHER` is one string, and everything else — data, seed, schedule,
loss, held-out sequences — is held fixed, so the gap between this run's
`test/instance_iou` and `07_encoder_aerial_rgbt.ipynb`'s is the teacher and
nothing else. The two are independent: different Drive folders, different index
directories, so running them on two runtimes at the same time is the intended
way to get the number.

DINOv3 is **gated**, and accepting its terms is the one extra step this
notebook needs. Open
[its model page](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m),
accept, then run `from huggingface_hub import login; login()` in this runtime
*before* stage A. Skip it and the teacher download 401s partway into the run.
AnyThermal's published result used DINOv2, which is why `facebook/dinov2-base`
is still one string away and needs no account."""

SAM_TROUBLE = ('`TEACHER = "facebook/dinov3-vitb16-pretrain-lvd1689m"`, '
               "the other arm")
DINO_TROUBLE = ('`TEACHER = "facebook/sam2.1-hiera-base-plus"`, '
                "the other arm")


def md(text: str, tag: str | None = None) -> None:
    _add("markdown", text, tag)


def code(text: str, tag: str | None = None) -> None:
    _add("code", text, tag)


def _add(kind: str, text: str, tag: str | None) -> None:
    if tag:
        TAGS[tag] = len(CELLS)
    CELLS.append((kind, text.strip("\n")))


def resolve(text: str) -> str:
    """Replace `{{tag}}` with the cell number that tag landed on.

    Cell numbers in the prose have drifted twice already, and a notebook that
    tells you to look at the wrong cell is worse than one that does not point
    anywhere. Tagging the cell and substituting at build time means they
    cannot go stale again.
    """
    dino = "dinov3" in V.teacher
    substitutions = {**{t: str(i) for t, i in TAGS.items()},
                     "TEACHER": V.teacher,
                     "TEACHER_HEADING": (
                         "### Which teacher, and why this run is the DINO arm"
                         if dino else
                         "### Which teacher, and why it is SAM 2 and not DINO"),
                     "SAM_TAG": "" if dino else " (this notebook)",
                     "DINO_TAG": " (this notebook)" if dino else "",
                     "TEACHER_NOTE": DINO_NOTE if dino else SAM_NOTE,
                     "OTHER_TEACHER": DINO_TROUBLE if dino else SAM_TROUBLE,
                     "TITLE": V.title, "BLURB": V.blurb,
                     "DATASETS": V.datasets, "FETCH": V.fetch,
                     "MIRROR": V.mirror, "SIBLING": SIBLING[V.key],
                     "DISTILL": V.distill,
                     "NOTEBOOK": Path(V.path).name, "STAMP": STAMP_VALUE}
    for tag, value in substitutions.items():
        text = text.replace("{{" + tag + "}}", value)
    leftover = re.findall(r"\{\{(\w+)\}\}", text)
    if leftover:
        raise KeyError(f"unknown cell tags: {sorted(set(leftover))}")
    return text




def swap(text: str, old: str, new: str) -> str:
    """`str.replace` that refuses to do nothing.

    Every edit to this file below was a plain `.replace()`, and three of them
    silently matched nothing -- the generator ran, the notebook rebuilt, and
    the settings it was supposed to gain were simply absent until
    `tools/check_notebook.py` found the names missing. A no-op replace is
    always a bug in the edit, never an intention.
    """
    if old not in text:
        raise AssertionError(f"nothing matched:\n{old[:200]}")
    return text.replace(old, new)


# --------------------------------------------------------------------------
# The two variants
# --------------------------------------------------------------------------

VARIANTS["all"] = Variant(
    key="all",
    path="notebooks/07_encoder_aerial_rgbt.ipynb",
    title="Encoder for aerial RGB-T — all three datasets",
    blurb=(
        "Trains on **VTUAV VIS, SegFly and Kust4K together**: three sensors, "
        "three cities, three sets of annotation habits, mixed into the same "
        "batches. This is the run to trust — an encoder is a general feature "
        "extractor, and the fastest way to make one that is not is to train it "
        "on a single flight.\n\n"
        "Its sibling `08_encoder_vtuav_only.ipynb` runs the identical schedule "
        "on VTUAV alone. Run both and compare `test/instance_iou`: the gap is "
        "what the extra two datasets bought, measured on the same held-out "
        "VTUAV sequences in both."),
    datasets="""DATASETS = [
    # Real drawn instance masks at 1920x1080. Trains, *and* is what the score
    # is measured on -- the one set here whose annotation nobody reconstructed.
    f"vtuav_vis:{DATA_ROOT}/VTUAV_VIS:thermal:labels:all",
    # Semantic maps at 640x512 over 30/40/50 m. The bulk of the frames, and the
    # only altitude variation on the list. Instances are reconstructed from the
    # map, so it trains and is never scored on.
    f"segfly:{DATA_ROOT}/SegFly:thermal:components:train",
    # Semantic maps at 640x512, 4 024 registered day/night pairs, 4 thing
    # classes -- the finest label set of the three.
    f"kust4k:{DATA_ROOT}/Kust4K:thermal:components:train",
]""",
    fetch="""FETCH = [
    ("vtuav_vis", f"{DATA_ROOT}/VTUAV_VIS",
     ["train_001", "train_002", "train_003"]),
    ("segfly",    f"{DATA_ROOT}/SegFly",    []),
    ("kust4k",    f"{DATA_ROOT}/Kust4K",    []),
]""",
    mirror="edgetam-encoder/all",
    distill="""DISTILL_SPEC    = "vtuav"            # where the *unlabelled* pairs come from
DISTILL_ROOT    = f"{DATA_ROOT}/VTUAV_VIS"
DISTILL_CROP    = 512 / 1080         # native pixels on 1920x1080; None on 640x512
DISTILL_TOL     = 1                  # VTUAV's modalities are not pixel-aligned""",
)

VARIANTS["vtuav"] = Variant(
    key="vtuav",
    path="notebooks/08_encoder_vtuav_only.ipynb",
    title="Encoder for aerial RGB-T — VTUAV VIS alone",
    blurb=(
        "The same schedule as `07_encoder_aerial_rgbt.ipynb`, on **VTUAV VIS "
        "and nothing else**: 1920x1080 frames with masks somebody actually "
        "drew, rather than instances reconstructed from a semantic map.\n\n"
        "The case for it is resolution and label quality. Every mask here is a "
        "real target at full sensor resolution, and none of `decompose`'s "
        "guesswork — connected components, gates, bridge splitting — sits "
        "between the annotation and the loss. The case against it is that 14 "
        "sequences of one campus is a narrow view of the world for a general "
        "feature extractor.\n\n"
        "Which case wins is the experiment. Run both notebooks and compare "
        "`test/instance_iou`; the held-out sequences are the same in both, so "
        "the two numbers are directly comparable."),
    datasets="""DATASETS = [
    # The whole training set: real masks, 1920x1080, `labels` mode because the
    # annotation already *is* instances and nothing needs reconstructing.
    f"vtuav_vis:{DATA_ROOT}/VTUAV_VIS:thermal:labels:all",
]""",
    fetch="""FETCH = [
    ("vtuav_vis", f"{DATA_ROOT}/VTUAV_VIS",
     ["train_001", "train_002", "train_003"]),
]""",
    mirror="edgetam-encoder/vtuav",
    distill="""DISTILL_SPEC    = "vtuav"            # where the *unlabelled* pairs come from
DISTILL_ROOT    = f"{DATA_ROOT}/VTUAV_VIS"
DISTILL_CROP    = 512 / 1080         # native pixels on 1920x1080; None on 640x512
DISTILL_TOL     = 1                  # VTUAV's modalities are not pixel-aligned""",
)

VARIANTS["drone"] = Variant(
    key="drone",
    path="notebooks/09_encoder_stage_a_dronevehicle.ipynb",
    title="Encoder for aerial RGB-T — stage A on DroneVehicle",
    blurb='Stage B is byte for byte `07_encoder_aerial_rgbt.ipynb` -- the same three datasets, the same schedule, the same seed. **Only the stage-A source differs**, so the gap between the two final numbers is attributable to what the encoder was pretrained on and to nothing else.\n\n07 distils from VTUAV: 26 059 pairs at 1920x1080 cropped down to 512, one campus, all daylight. This one distils from **DroneVehicle**: 28 442 pairs already at 640x512 -- the sensor size this project deploys at, so no resampling at all -- across day *and* night, from a set whose oriented-box labels make it useless for segmentation and therefore unused by anyone doing what we are doing.\n\nThe question it answers: does stage A want **resolution and one scene**, or **native pixels and variety**? Nothing in the literature settles that for aerial thermal, which is why it is a run rather than an argument.',
    datasets=VARIANTS["all"].datasets,
    fetch="""FETCH = [
    ("vtuav_vis",    f"{DATA_ROOT}/VTUAV_VIS",
     ["train_001", "train_002", "train_003"]),
    ("segfly",       f"{DATA_ROOT}/SegFly",       []),
    ("kust4k",       f"{DATA_ROOT}/Kust4K",       []),
    # Stage A's source here. 14 GB, anonymous HTTPS, no quota and no form.
    ("dronevehicle", f"{DATA_ROOT}/DroneVehicle", []),
]""",
    mirror="edgetam-encoder/drone",
    distill='DISTILL_SPEC    = "dronevehicle"     # where the *unlabelled* pairs come from\nDISTILL_ROOT    = f"{DATA_ROOT}/DroneVehicle"\nDISTILL_CROP    = None               # already 640x512 once the white band is\n                                     # cropped -- resampling it would be a loss\nDISTILL_TOL     = 1                  # registered, but not claimed pixel-exact',
)

VARIANTS["dino"] = Variant(
    key="dino",
    path="notebooks/10_encoder_teacher_dinov3.ipynb",
    title="Encoder for aerial RGB-T — DINOv3 teacher",
    blurb=(
        "**`07_encoder_aerial_rgbt.ipynb` with one string changed.** Same three "
        "datasets, same schedule, same seed, same held-out sequences — only the "
        "stage-A teacher differs, so the gap between the two final numbers is "
        "attributable to the teacher and to nothing else. The two runs share no "
        "output directory, so the intended way to get that gap is to start both "
        "on separate runtimes at once.\n\n"
        "07 distils from **SAM 2.1 Hiera-B+**. That is the safe arm: EdgeTAM is "
        "itself a distillation of SAM 2, and its published teacher was Hiera-B+ "
        "with the stride-16 feature map as the target — the same tensor read "
        "here. Asking it to produce those features from a thermal input is the "
        "objective the checkpoint was born from with a modality gap added, and "
        "it pushes the encoder *toward* the space the memory path expects.\n\n"
        "This one distils from **DINOv3 ViT-B/16**. The argument for it is that "
        "SAM's features are class-agnostic by design, and in thermal the "
        "boundary is the easy part — a warm object against a cool background is "
        "already obvious. What a thermal encoder lacks is *semantics*: whether "
        "the bright blob is a vehicle or a rock that sat in the sun all day. "
        "DINOv3 carries that; SAM deliberately does not. It also lands on a "
        "32x32 token grid at this notebook's 512 input, which is the student's "
        "grid exactly.\n\n"
        "Higher ceiling, higher risk, and nobody has measured which wins on "
        "aerial thermal — which is why this is a run rather than an argument. "
        "**DINOv3 is gated:** accept its terms and `login()` before starting, or "
        "stage A fails partway in."),
    datasets=VARIANTS["all"].datasets,
    fetch=VARIANTS["all"].fetch,
    mirror="edgetam-encoder/dino",
    distill=VARIANTS["all"].distill,
    teacher="facebook/dinov3-vitb16-pretrain-lvd1689m",
)

SIBLING = {"all": "08_encoder_vtuav_only.ipynb, 09_encoder_stage_a_dronevehicle.ipynb and 10_encoder_teacher_dinov3.ipynb",
           "vtuav": "07_encoder_aerial_rgbt.ipynb",
           "drone": "07_encoder_aerial_rgbt.ipynb",
           "dino": "07_encoder_aerial_rgbt.ipynb (the SAM 2.1 teacher)"}

V = VARIANTS[sys.argv[1]] if len(sys.argv) > 1 else VARIANTS["all"]


# ---------------------------------------------------------------- 0
md(r"""
# {{TITLE}}

One runtime, one **Run all**: download the data, turn whatever annotation it
ships into per-object training targets, train EdgeTAM's image encoder two
different ways, and score both on sequences neither has seen.

{{BLURB}}

| | |
|---|---|
| **in** | nothing staged by hand — the download cell fetches everything |
| **out** | `edgetam_aerial_512.pt` and `edgetam_aerial_lora_512.pt` on your Drive, plus a held-out instance score for each |
| **needs** | a CUDA GPU and ~25 GB of free disk |
| **takes** | ~20 min to download, ~5 min to index, ~20 min to distil, ~40 min per training run |

This is **stage B** of `docs/encoder_training_todo.md`, with stage A beside it.
No video and no memory bank appear anywhere in it — that is stage C, and it is
deliberately not here.

---

## The architecture, in one page

EdgeTAM is four parts. Which of them a training stage touches is the entire
design, so here they are with what each one does:

```
    image ──▶ image_encoder ──▶ features ──┬──▶ mask_decoder ──▶ mask
                (RepViT trunk               │         ▲
                 + FPN neck)                │         │  prompt (box / point)
                  4.92 M                    │    sam_prompt_encoder
                                            │
                                            ▼
                          memory_attention ◀── memory bank ◀── memory_encoder
                              2.96 M                              1.62 M
                          ── the recurrent part: only exists across frames ──
```

The image encoder and the mask decoder are a **still-image segmenter**. The
memory path is what makes it a **tracker**. They fail differently, they need
different data, and SAM 2 trains them in that order — image pretraining on
still images first, video afterwards. When SAM 2 later extends to longer
sequences it *freezes the image encoder* outright. So the split is not our
invention; it is the recipe.

### The three stages, and which is which

| | stage A — pretrain | **stage B — this notebook** | stage C — video |
|---|---|---|---|
| **trains** | `image_encoder` | `image_encoder` + `mask_decoder` | `memory_attention` + `memory_encoder` |
| **frozen** | everything else | the whole memory path | **the image encoder** |
| **data** | registered RGB-T pairs, *no labels* | still images + dense masks | video with instance identity |
| **target** | a frozen RGB teacher's features | one object's mask, per prompt | the same, across frames |
| **loss** | cosine distance | 20·focal + dice + IoU-L1 | + object-score BCE |
| **freeze call** | `apply_freeze(model, "backbone")` | `apply_freeze(model, "encoder")` | a new stage, C's own |

Everything the two stages below need already exists in `src/training/`:
`apply_freeze` is the mechanism, `losses.py` is the objective, `schedule.py` is
the loop, and the memory path is frozen by default for three reasons argued in
`finetune.py`.

### The one hard problem, and what solves it

**Most of these datasets label the wrong thing.** Kust4K, SegFly and Caltech
Aerial are *semantic* segmentation: every pixel gets a class, so "all cars are
car". A tracker needs the opposite — *this* car, not the one parked beside it.
Training a promptable segmenter on a semantic map pulls two adjacent cars'
features **together**, which is precisely the confusion the tracker then
inherits.

**One does not**, and it is in `DATASETS` by default: VTUAV's
video-instance-segmentation split ships per-frame **instance** masks at
1920×1080. On that dataset none of what follows runs — `mode=labels` reads each
value as a target directly. Everything below is what to do with the sets that
only have a semantic map, which is still most of them. SAM 2 never sees a semantic map either — it samples
individual masks and prompts each one separately. So:

```
   semantic map                   thing classes only          per class:
   ┌───────────────┐              ┌───────────────┐           connected
   │ road road road│              │ ·  ·  ·  ·  · │           components
   │ road CAR  CAR │   filter     │ ·  CAR CAR  · │  ───────▶  ┌───┐┌───┐
   │ bldg CAR  CAR │  ─────────▶  │ ·  CAR CAR  · │            │ 1 ││ 2 │
   │ bldg bldg tree│   (drop      │ ·  ·  ·  ·  · │            └───┘└───┘
   └───────────────┘    stuff)    └───────────────┘         two instances,
                                                            two prompts,
                                                            two targets
```

1. keep only the **thing** classes — 4 of Kust4K's 8: motorcycle, car, truck,
   person. Road, building and vegetation are *stuff*: no instances, not
   tracking targets;
2. **connected components, one class at a time** — not over the union, or a car
   touching a pedestrian would fuse into one component spanning two classes;
3. gate what survives (too small, too large, too sparse to be an object);
4. one box prompt per component, that component's mask as the target.

**The risk this creates is real and is measured below, not assumed away.** Two
cars parked bumper to bumper share a pixel edge and become *one* component, and
no gate can separate them. Cell {{fusion}} puts a number on how often that
happens before a single GPU-hour is spent on it. That number is the go/no-go for the
whole stage.

### How a batch is put together

The encoder is ~38.7 GFLOP at 512 and **does not depend on the prompt**, so an
image is encoded once and every instance inside the window is prompted against
those same features:

```
   window (512×512 native pixels)        one forward through image_encoder
   ┌─────────────────────┐                          │
   │   [car A]           │                          ▼
   │           [car B]   │   ──────▶   features ──┬──▶ decoder ← box A  → mask A
   │   [person]          │                        ├──▶ decoder ← box B  → mask B
   └─────────────────────┘                        └──▶ decoder ← box P  → mask P
```

Cheaper *and* a better signal: the batch now contains several objects sharing a
scene, which is exactly the discrimination the loss has to make.

### Two more choices carried over unchanged

- **The model trains in `eval()` mode.** RepViT is full of batch norms and
  their statistics must stay frozen, or the checkpoint stops matching the
  engines TensorRT folds them into and any INT8 calibration taken before it is
  invalid.
- **512 is a crop, not a resize.** A 512 window on a 640×512 thermal frame is
  native sensor pixels — the same input the Orin's `crop512` mode feeds the
  model. `Sample.native` reports it per window and the notebook checks it.

### What comes out, and what it is not

An ordinary EdgeTAM checkpoint: same keys, same loader, same ONNX export, same
engine build. Its **memory path is still the stock one**, so the number this
notebook produces is an encoder result read through an untrained memory path.
That is the honest first measurement and the reason stage C exists — the
verdict that matters is J&F on video, and nothing here can produce it.
""")

# ---------------------------------------------------------------- 1
code(r"""
# --- Runtime, repo, GPU -------------------------------------------------
import os, sys
from pathlib import Path

REPO   = Path("/content/sam-dedection")
BRANCH = "claude/encoder-architecture-colab-myo61y"

if not REPO.exists():
    !git clone -q -b {BRANCH} https://github.com/yigitkayabagci/sam-dedection.git {REPO}
!git -C {REPO} fetch -q origin {BRANCH}
!git -C {REPO} checkout -q {BRANCH} && git -C {REPO} merge -q --ff-only origin/{BRANCH}
os.chdir(REPO)
sys.path.insert(0, str(REPO))

# Drop this repo's modules so the fast-forward above actually takes effect.
# `git pull` changes files on disk; it cannot change a module Python already
# imported, so re-running a later cell after an update quietly keeps running
# the old code -- which looks exactly like the fix not working. (Safe here:
# these are pure Python. The same move on torch is not -- see the cell below.)
for _stale in [n for n in list(sys.modules) if n.split(".")[0] in ("src", "tools")]:
    del sys.modules[_stale]

# Image mode holds fewer, larger activations than clip mode; the allocator
# still fragments across the batch-size search, and this costs nothing.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
!df -h /content | tail -1

# --- Is this notebook the one the repo expects? -------------------------
# The repo just fast-forwarded itself; the .ipynb is a file you uploaded, and
# the two drift apart silently. When they do you hit bugs that were fixed
# upstream days ago, and the traceback points at a line the repo no longer
# contains -- which is a genuinely confusing hour. This says so at cell 1.
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
    print( "!!  (In Colab: File > Upload notebook, or Runtime > Disconnect first.)")
    print("=" * 74 + "\n")
else:
    print(f"notebook build {STAMP} matches the repo")
""")

# ---------------------------------------------------------------- 1b
code(r"""
# --- Does this torch actually have kernels for this GPU? ----------------
# Asked by launching one, not by reading an arch list. On a card newer than
# the torch build -- Blackwell (sm_120) is the current case -- everything
# imports, the model builds, and the first real matmul dies with "no kernel
# image is available for execution on the device" forty minutes in.
import torch

print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"{torch.cuda.get_device_name(0)}  sm_{major}{minor}  "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
    print(f"this build ships: {', '.join(torch.cuda.get_arch_list())}")
    try:
        probe = torch.randn(256, 256, device="cuda")
        (probe @ probe).sum().item()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            (probe @ probe).sum().item()
        print("a real matmul ran, in float32 and bfloat16 -- this GPU is usable")
    except RuntimeError as exc:
        raise SystemExit(
            f"torch {torch.__version__} cannot run on sm_{major}{minor}: {exc}\n\n"
            f"Blackwell needs a CUDA 12.8+ build. In Colab, do NOT let anything "
            f"downgrade the preinstalled torch -- if a pip install replaced it, "
            f"restore it with:\n"
            f"    pip install --upgrade torch torchvision "
            f"--index-url https://download.pytorch.org/whl/cu128\n"
            f"then Runtime > Restart session and run this notebook from the top.")
    # Cheap on any card, and the trunk's convolutions are the part that gains.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("!! no CUDA device -- nothing below will run")
""")

# ---------------------------------------------------------------- 2
code(r"""
# --- Dependencies -------------------------------------------------------
# EdgeTAM installs itself *as* the package `sam2` -- it is a fork -- so Meta's
# sam2 must never share this environment. The teacher is loaded through
# transformers' own Sam2Model, which is independent of both -- that is the
# whole reason a SAM 2 teacher does not collide with the SAM 2 fork below.
before = torch.__version__
!bash scripts/setup_edgetam.sh 2>&1 | tail -5
!pip install -q -r requirements.txt
# 4.56 is the floor that knows the teachers: Sam2Model and DINOv3 both landed
# there, and an older wheel satisfies a lower pin and then fails at
# from_pretrained, forty minutes into the session.
!pip install -q "transformers>=4.56" datasets

# The one thing an install here must not do. EdgeTAM's setup.py declares a
# torch floor, and on a card newer than any wheel on PyPI a "helpful"
# reinstall is how a working runtime stops working.
#
# Read the version off *disk*. Reloading the module cannot work: re-running
# torch/__init__.py re-registers its `triton` TORCH_LIBRARY namespace, which
# C++ refuses outright -- so `importlib.reload(torch)` raises whether or not
# anything is wrong, which is the worst possible behaviour for a safety check.
# Disk is also the right thing to read, because pip cannot change the torch
# already live in this kernel; only a restart can. That is exactly the window
# this check exists to catch.
import importlib.metadata as _md
installed = _md.version("torch")
assert installed == before, (
    f"pip replaced torch {before} with {installed} on disk.\n"
    f"This kernel is still running {before}, so nothing has broken yet -- but "
    f"restarting would load a build that may have no kernels for this GPU.\n"
    f"Restore the Colab build *before* restarting:\n"
    f"    !pip install -q --force-reinstall torch=={before}")

# setup_edgetam.sh installs EdgeTAM with `pip install -e`, and an editable
# install is a .pth file in site-packages -- which site.py reads at interpreter
# *startup* only. A package installed into a kernel that is already running
# therefore never joins sys.path: `import sam2` raises ModuleNotFoundError with
# the install sitting on disk two directories away, and `invalidate_caches()`
# does not help because the path entry was never added. Restarting the runtime
# would fix it and cost the GPU session and every download in it. Pointing
# sys.path at the checkout fixes it now.
#
# Placed *after* the repo, not at the front: EdgeTAM's own top level carries
# `tools/`, `notebooks/` and `examples/`, and so does this repo. Ours would win
# anyway -- a regular package beats a namespace portion -- but ordering means
# that does not have to be true.
import importlib
EDGETAM = REPO / "third_party" / "EdgeTAM"
if str(EDGETAM) not in sys.path:
    sys.path.insert(sys.path.index(str(REPO)) + 1 if str(REPO) in sys.path else 0,
                    str(EDGETAM))
importlib.invalidate_caches()

import sam2, transformers
print(f"sam2 (EdgeTAM) {Path(sam2.__file__).parent}\ntransformers {transformers.__version__}")

# The premise of the cell above, checked rather than trusted: the `sam2` that
# imported has to be the EdgeTAM checkout this repo patched, not Meta's package
# under the same name. If both are present the wrong one loads silently and
# fails much later, inside the model.
_from = Path(sam2.__file__).resolve().parent.parent
assert _from == EDGETAM.resolve(), (
    f"`sam2` imported from {_from}, not the EdgeTAM checkout at {EDGETAM}.\n"
    f"Meta's sam2 is probably installed alongside it: `pip uninstall -y sam2 "
    f"SAM-2` and re-run this cell.")
assert (EDGETAM / "checkpoints" / "edgetam.pt").is_file(), \
    "edgetam.pt did not download -- rerun scripts/setup_edgetam.sh and read its output"
""")

# ---------------------------------------------------------------- 3
code(r"""
# The contracts everything below depends on, tested with no GPU and no
# dataset. If these fail, nothing after this point is worth running.
!python -m unittest tests.test_aerial tests.test_image_loop tests.test_distill \
    tests.test_training_losses tests.test_clip_loop tests.test_schedule \
    tests.test_loader tests.test_lora 2>&1 | tail -3
""")

# ---------------------------------------------------------------- 4
md(r"""
## Settings

The only cell you should need to edit, and the one place the paths live.

**The datasets live on local disk, never on Drive.** Training reads thousands
of images in random order and decodes an annotation map for each; the Drive
FUSE mount serves small files at a few hundred a second, an order of magnitude
under what the GPU consumes, and the run ends up loader-bound with the
accelerator idle. `/content` is local NVMe, so that is where `FETCH` puts them.

Drive is the right home for exactly two things here, and it is where both go:
the **checkpoints** and the **instance index**. Both survive a runtime dying;
the datasets can always be downloaded again.

`MIRROR` is per-notebook (`{{MIRROR}}`), so this and `{{SIBLING}}` can run at
the same time on two runtimes without overwriting each other.

**`DATASETS` is a list, and that is the point.** One dataset is one sensor, one
city and one set of annotation habits — fine for a head, thin for a trunk that
carries general visual features. Every entry listed is mixed into the *same*
batches:

```
spec        : path                    : modality : mode
vtuav_vis   : /content/data/VTUAV_VIS : thermal  : labels      1920×1080, instance masks
kust4k      : /content/data/Kust4K    : thermal  : components  640×512, semantic
segfly      : /content/data/SegFly    : thermal  : watershed   640×512, semantic
```

The `mode` field is what each dataset's annotation *is*:

| mode | the map is | what happens |
|---|---|---|
| `components` | semantic | connected components per thing class — the reconstruction described above |
| `watershed` | semantic | the same, plus splitting objects joined by a thin bridge |
| `labels` | **already instances** | nothing is decomposed; each non-zero value is one target |

The `role` field is which splits it feeds, and it is doing real work here:

| role | feeds | use it for |
|---|---|---|
| `all` | train + val + test | a set with **real** instance annotation |
| `train` | train only | a set whose instances were **reconstructed** from a semantic map |
| `eval` | val + test only | a held-out set you never want trained on |

**Why the reconstructed sets are `train`.** A held-out number is supposed to say
how good the model is. Where `decompose` fused two parked cars, its "ground
truth" is one blob — so a model that correctly separates them is scored
**wrong**, and the number measures the decomposition as much as the model.
Training on a slightly noisy target is normal and fine; being *graded* by one
is not. So Kust4K trains, VTUAV grades.

**`labels` is why VTUAV changes this notebook.** Its video-instance-segmentation
split ships per-frame instance masks at 1920×1080 — the annotation the whole
`components` path exists to reconstruct — so on that dataset the fusion problem
does not arise at all, and the resolution is three times Kust4K's.

Specs name layouts and palettes (`src/training/aerial.py:SPECS`). All three
were read off the actual archives — Kust4K's from the `visual.py` shipped
beside it, VTUAV's by listing a training zip's central directory, SegFly's from
the authors' table — and Kust4K's is checked end to end against a real
download. Cells {{layout}} and {{specs}} re-check them on **your** copy anyway,
because a spec is still only an assumption about an archive, and the way that
fails is silent: a run with zero instances and no error to explain it.

Two of these palettes were wrong before they were checked, in the same way and
with the same consequence. Kust4K's assumed a `vegetation` class that does not
exist, which shifted every id above it by one, so `things` picked up id 6 —
**tree** — and missed id 3, **motorcycle**. SegFly's assumed contiguous ids and
made grass, vegetation, tree and ground obstacle into tracking targets. Either
one trains the model to answer a prompt with a hedge.
""")

# ---------------------------------------------------------------- 5
code(r"""
# --- Where things live --------------------------------------------------
from google.colab import drive
drive.mount("/content/drive")

# Nothing to stage by hand. Every URL is baked into tools/fetch_datasets.py,
# checked against the live host, and md5-verified where the publisher gives one.
DATA_ROOT = "/content/data"          # datasets -- local NVMe, never Drive
{{FETCH}}

# Per-notebook, so this and its sibling
#   {{SIBLING}}
# can run at once without overwriting each other's checkpoints or indexes.
MIRROR   = Path("/content/drive/MyDrive/{{MIRROR}}")   # <-- weights land here

DATA_DIR = Path(DATA_ROOT)
WORK     = Path("/content/work")     # logs and scratch, this runtime only
CKPT     = REPO / "checkpoints"

# --- What we are training on -------------------------------------------
# Repeatable: "spec:path[:modality[:mode[:role]]]". Every dataset listed is
# mixed into the same batches -- one dataset is one sensor, one city and one
# set of annotation habits, and an encoder carries general features.
#
#   mode = components  the map is semantic -> connected components per class
#          watershed   the same, plus splitting bridge-joined objects
#          labels      the map already IS instances (VTUAV's VIS split)
#   role = all    normal 80/10/10
#          train  training only -- never scored on
#          eval   val and test only -- never trained on
{{DATASETS}}
SIZE      = 512                      # model input == the native crop on 640x512

# --- Instance decomposition (see the diagram above) --------------------
MIN_AREA, MIN_SIDE = 48, 4           # below this a component is annotation noise
MAX_AREA, FILL     = 0.9, 0.25       # SAM 2's >90%-of-frame rule; fill catches fusion
PER_IMAGE          = 2               # windows sampled per frame
MAX_INSTANCES      = 8               # prompts per window -- one encode covers all

# --- Stage A: the unlabelled pretraining pass ---------------------------
PRETRAIN        = True               # distil an RGB teacher into the encoder
TEACHER         = "{{TEACHER}}"
{{DISTILL}}
DISTILL_STEPS   = 600
DISTILL_PAIRS   = 20000              # sampled across the set, never truncated
DISTILL_MOMENTS = 0.0                # anti-collapse term; small or off (docs 4)
ANCHOR_WEIGHT   = 0.5                # stage B: how hard to hold on to stage A

# --- Budget -------------------------------------------------------------
STEPS_PER_EPOCH = 400
EPOCHS          = (1, 3)             # head-only, then head + encoder
VAL_BATCHES     = 24
BATCH_CEILING   = 64
LORA_R          = 16
JITTER          = 32
SEED            = 0
LOADER_WORKERS  = min(2 * (os.cpu_count() or 4), 24)
PREFETCH_DEPTH  = 2

# One index file per dataset, named after it, so changing one dataset's mode
# does not invalidate the others -- and so an index built one way can never be
# loaded by a run decomposing the other way (it is stamped and checked).
INDEX = MIRROR / "index"

for d in (DATA_DIR, WORK, CKPT, MIRROR, INDEX):
    d.mkdir(parents=True, exist_ok=True)

# Copy a finished artefact to Drive the moment it exists. Everything below
# writes to local disk and reaches Drive only in the last cell, which is fine
# right up until a Colab runtime dies three runs in -- and then the whole
# session's GPU time goes with it. Each run calls this as it finishes, so what
# has already been paid for survives.
def mirror_now(*paths):
    import shutil
    for path in paths:
        path = Path(path)
        if path.is_file():
            shutil.copy(path, MIRROR / path.name)
            print(f"   saved -> {MIRROR / path.name}")
        else:
            print(f"   !! {path} was not written -- the run above did not finish")

import torch
VRAM = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
print(f"{VRAM:.0f} GiB of VRAM, {os.cpu_count()} cores -> {LOADER_WORKERS} loader threads")
print(f"weights -> {MIRROR}, instance indexes -> {INDEX}")
print("datasets:"); [print(f"  {d}") for d in DATASETS]
""")

# ---------------------------------------------------------------- 6
md(r"""
## Downloading the data

Nothing to stage by hand and no Drive folder to fill first. `FETCH` above names
the datasets; the cell below downloads them, verifies them and extracts them
into the layout the readers glob.

The URLs live in `tools/fetch_datasets.py`, and each was checked against the
live host rather than copied off a paper — which mattered, because all three
are served in a way that defeats the obvious approach:

* **Kust4K.** The whole-article figshare link answers **HTTP 202 with an empty
  body**: it asks figshare to *build* the zip and returns before it exists, so
  a downloader writes a zero-byte file and reports success. The per-file
  endpoints are already built, answer range requests, and come with the
  publisher's md5. Those are what this uses.
* **VTUAV.** The mask split is a Drive *folder* of 8–17 GB zips, and past a few
  hundred megabytes Drive serves a "can't scan this file for viruses" form
  instead of the file — with a 200 on it. `gdown` replays the form; `curl`
  saves the HTML.
* **SegFly** is parquet on the Hub, not files, so it is exported into PNGs.

Downloads resume, and each archive is deleted once extracted, which halves the
peak disk this needs.

**If Drive refuses VTUAV** — *"Too many users have viewed or downloaded this
file recently"* — that is a quota on **who is asking**, not on the file. Colab
shares its egress addresses with a great many people, so the same archive can
serve fine elsewhere while a Colab session is refused it. The fetcher retries
and tries a second route before giving up, and prints three ways out. The
reliable one: open the
[mask-split folder](https://drive.google.com/drive/folders/11E-WPkCPVL49hOKRdCzfgQULmGU8pyz8),
right-click `train_001.zip` → **Make a copy**, move the copy into
`MyDrive/datasets/`, and re-run this cell. Copying is server-side and is not a
download, so no quota applies — and neither does one to reading your own file
through the mount. The cell looks in `MyDrive/datasets/` **before** it touches
the network, so a staged copy is used as-is and never re-downloaded.
""")

# ---------------------------------------------------------------- 7
code(r"""
# --- Download everything, once ------------------------------------------
import shutil

from src.training.datasets import describe, parse
from tools.fetch_datasets import fetch, human, report

requests = [parse(d) for d in DATASETS]
print(describe(requests), "\n")
print(f"{human(shutil.disk_usage('/content').free)} of disk free\n")

for name, dest, parts in FETCH:
    if Path(dest).exists() and any(Path(dest).rglob("*.png")):
        print(f"== {name}: already downloaded to {dest}")
    else:
        fetch(name, dest, tuple(parts) or None)
    print(report(name, dest, "thermal"))
""")

# ---------------------------------------------------------------- 7b
md(r"""
### If the cell above could not pair anything

`SPECS` in `src/training/aerial.py` holds globs written against what the papers
say their archives contain. Archives get repacked, renamed and nested inside an
extra folder, so **this is the most likely place a first run stops**. The cell
below prints what the download *actually* contains, in the shape a glob is
written in — one look, then one `replace(...)` line in the cell after it, and
nothing in the repo needs editing.
""")

code(r"""
# --- What did we actually get? ------------------------------------------
from src.training.aerial import describe_layout

for request in requests:
    print(f"===== {request.label} =====")
    print(describe_layout(request.root))
    print()
""", tag="layout")

# ---------------------------------------------------------------- 8
code(r"""
# --- Do the specs match the downloads? ----------------------------------
from dataclasses import replace
from src.training.aerial import InstanceGates, Source, list_frames, probe_classes

gates = InstanceGates(min_area=MIN_AREA, min_side=MIN_SIDE,
                      max_area=MAX_AREA, fill=FILL)

# An escape hatch, empty on purpose: every spec here was read off its archive
# and Kust4K's was checked end to end against a real download, so a first run
# should need nothing in it. Fill one in only if a layout above disagrees with
# its spec -- a repacked archive, a newer release -- and prefer fixing SPECS in
# the repo once you know what is right.
#
# Globs are relative to that dataset's root; `strip` removes a suffix from a
# *mask* stem before pairing ("0001_label.png" pairs with "0001.png" when
# strip=("_label",)); `{modality}` in a mask glob is filled with the directory
# that modality's images live in (VTUAV annotates `mask/ir` and `mask/rgb`
# separately).
#
# Do not paste a palette in here from memory. The two that were wrong were both
# plausible-looking guesses.
OVERRIDES: dict[str, dict] = {
    # "vtuav_vis": dict(masks="**/mask/{modality}/*.png"),
}

sources, frames_of = {}, {}
for request in requests:
    spec = request.source.spec
    if spec.name in OVERRIDES:
        spec = replace(spec, **OVERRIDES[spec.name])
    # `role` travels too: dropping it here would put the reconstructed sets
    # back into val and test in every table this notebook prints, while the
    # training subprocesses -- which re-parse DATASETS themselves -- keep them
    # out. The two views of the split have to be the same view.
    source = Source(spec=spec, gates=gates, mode=request.source.mode,
                    gray=request.source.gray, role=request.source.role)
    frames = list_frames(request.root, spec, request.modality)
    sources[request.label], frames_of[request.label] = source, frames

    print(f"===== {request.label} =====")
    print(f"{len(frames)} frames, {sum(1 for f in frames if f.pair)} with a "
          f"registered RGB twin")
    present = probe_classes(frames, limit=48)
    expected = {v: k for k, v in spec.classes.items()}
    if source.mode == "labels":
        # Nothing to check: every non-zero value is an instance id, so the
        # palette is not a claim about anything.
        print(f"  mode=labels -> {len(present) - (0 in present)} distinct "
              f"instance ids across the sample; no class filter applies")
    else:
        for value, count in present.items():
            mark = ("thing" if value in spec.thing_ids
                    else "stuff" if value in expected else "??")
            print(f"  value {value:>3}  {expected.get(value, 'NOT IN SPEC'):<14} "
                  f"{mark:<6} in {count} of {min(48, len(frames))} sampled maps")
        missing = sorted(set(spec.thing_ids) - set(present))
        assert not missing, (
            f"{spec.name}: no pixels of thing class(es) {missing}. The palette "
            f"does not match this download -- fill in OVERRIDES above and re-run.")
    print()
print("every spec matches its download")
""", tag="specs")

# ---------------------------------------------------------------- 9
md(r"""
## The cheapest and riskiest question, asked first

Everything below depends on one thing being true: **connected-component
decomposition produces clean single objects.** If two parked cars fuse into one
component, the target is wrong, and no amount of training fixes a wrong target
— it trains the model to answer "which car?" with "both".

This is also the cheapest thing to check. It reads the semantic maps and
nothing else — no images decoded, no GPU touched — so it costs minutes, and it
runs *before* anything expensive. The index it produces is cached to Drive, so
the second run of this notebook skips it.

Read the output for three things:

- **instances per class** — enough of the classes you care about to train on?
- **median size** — the tracking targets are the small ones; if the median
  instance is 4 000 px this dataset's vehicles are not the small targets the
  deployment sees
- **`fill` rejections** — the fusion warning. A component filling a quarter of
  its own bounding box is usually two objects joined by a thin bridge of
  pixels. A few percent is normal; tens of percent means this dataset's objects
  routinely touch and its instances are not usable as targets.
""")

# ---------------------------------------------------------------- 10
code(r"""
# --- Decompose every annotation map, once -------------------------------
from tqdm.auto import tqdm
from src.training.aerial import index_frames, load_index, save_index, summarise

bar = lambda st, total, desc: tqdm(st, total=total, desc=desc)

index = []
for request in requests:
    label = request.label
    source = sources[label]
    cache = INDEX / f"{label.replace(':', '_')}.json"
    if cache.is_file():
        part = load_index(cache, source)          # refuses a different mode
        print(f"{label}: index reused from {cache.name}")
    else:
        part = index_frames(frames_of[label], source, workers=LOADER_WORKERS,
                            progress=bar)
        save_index(cache, part)
        print(f"{label}: index written to {cache.name}")
    index.extend(part)

print()
print(summarise(index))
""")

# ---------------------------------------------------------------- 11
code(r"""
# --- Are those instances clean? -----------------------------------------
import numpy as np
import matplotlib.pyplot as plt

items = [i for e in index for i in e.instances]
fill  = np.array([i.fill for i in items])
side  = np.array([max(i.width, i.height) for i in items])
per_frame = np.array([len(e.instances) for e in index])

rejects = {}
for e in index:
    for reason, n in e.rejects.items():
        rejects[reason] = rejects.get(reason, 0) + n
fused = rejects.get("fill", 0) / max(sum(rejects.values()) + len(items), 1)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
axes[0].hist(side, bins=40); axes[0].set_xlabel("longest side, source px")
axes[0].axvline(32, color="crimson", lw=1); axes[0].set_title("instance size")
axes[1].hist(fill, bins=40, range=(0, 1)); axes[1].set_xlabel("area / bbox area")
axes[1].axvline(FILL, color="crimson", lw=1); axes[1].set_title("fill — low is fusion")
axes[2].hist(per_frame, bins=range(0, per_frame.max() + 2))
axes[2].set_xlabel("instances per frame"); axes[2].set_title("density")
for ax in axes: ax.set_ylabel("count")
plt.tight_layout(); plt.show()

print(f"{len(items)} instances   median side {np.median(side):.0f} px   "
      f"{(side < 32).mean():.0%} under 32 px (the deployment's regime)")
print(f"{fused:.1%} of all components were rejected on `fill` -- the fusion signal")
if fused > 0.15:
    print("!! high. Objects in this set routinely touch. Look at the panels below "
          "before trusting the targets, and consider dropping the densest class.")
""", tag="fusion")

# ---------------------------------------------------------------- 12
code(r"""
# --- Look at them. A histogram cannot tell you a target is wrong --------
import cv2
from src.training.aerial import decompose, read_mask

busiest = sorted(index, key=lambda e: -len(e.instances))[:6]
fig, axes = plt.subplots(2, 6, figsize=(15, 5.2))
for col, entry in enumerate(busiest):
    raw = cv2.imread(str(entry.frame.image), cv2.IMREAD_UNCHANGED)
    image = raw if raw.ndim == 2 else cv2.cvtColor(raw[..., :3], cv2.COLOR_BGR2RGB)
    source = entry.source
    components, instances, _ = decompose(read_mask(entry.frame.mask), source.spec,
                                        source.gates, source.mode)

    axes[0, col].imshow(image, cmap="gray" if image.ndim == 2 else None)
    axes[0, col].set_title(f"{source.spec.name} {entry.frame.name}\n"
                            f"{len(instances)} instances", fontsize=8)
    # Random colours per component: two adjacent objects sharing one colour is
    # the fusion failure, visible instantly and invisible in any average.
    tinted = np.random.default_rng(0).random((int(components.max()) + 1, 3))
    tinted[0] = 0
    axes[1, col].imshow(tinted[components])
    for inst in instances:
        x0, y0, x1, y1 = inst.box
        axes[1, col].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                             fill=False, ec="w", lw=0.6))
    for ax in (axes[0, col], axes[1, col]): ax.axis("off")
plt.tight_layout(); plt.show()
print("Top row: the image. Bottom: one colour per training target.")
print("Two vehicles sharing a colour is the failure; one white box drawn "
      "around two of them is the same failure.")
""", tag="panels")

# ---------------------------------------------------------------- 13
md(r"""
### If the fusion number is bad: four ways out, ranked

The decomposition above is one strategy, not the only one. If cell {{fusion}}
says a large share of components fail on `fill`, or cell {{panels}} shows two
vehicles sharing a colour, these are the options in order of cost:

**1. `mode=watershed` on that dataset — free, partial.** Objects joined by a
*thin bridge* of pixels are one component but two peaks in the distance
transform, with a valley between them; seeding at the peaks and flooding
outwards separates them. Change that dataset's fourth field in `DATASETS`
(`kust4k:...:thermal:watershed`), re-run from the index cell, and compare the
`fill` reject count. Each mode caches its own index, so switching back costs
nothing. **Its limit is worth knowing before you try it:** two
rectangles abutting along a full edge tile into a larger rectangle whose
distance transform has *no* valley, and nothing about mask geometry can split
those. It rescues the bridged case and nothing else.

**2. A promptable teacher as the splitter — the only one that can split
abutting objects.** SAM 2.1 is already in this repo (`src/training/labels.py`
uses it to turn Anti-UAV410's boxes into masks). Prompt it with a *point*
inside each component and keep its mask where it agrees with the semantic
class: it decides from **appearance**, not from mask geometry, so two cars that
share a pixel edge are still two cars to it. Costs one teacher pass per
component, offline and once. Not built yet — say the word.

**3. Filter harder and accept less data.** `fill`, `min_area` and a per-class
`max_area` already drop the suspicious components; raising `FILL` toward 0.4
drops more. Fewer instances, all trustworthy. Cheapest of all, and often
enough — 20 000 clean instances beat 40 000 half-fused ones.

**4. Use a set that ships instances and skip the problem — and one already
does.** **VTUAV's VIS split** is aerial, RGB-T, 1920×1080 and instance-masked;
`mode=labels` reads it directly and no decomposition runs. It is in `DATASETS`
by default for exactly this reason. **AeroVIS** gives instance masks *plus*
identity in YTVIS format but is RGB-only, and **iSAID** is the large aerial
instance-segmentation set (655 k instances) if RGB is acceptable for this stage.
The decomposition exists because dense *thermal* aerial sets are mostly
semantic — not because it is preferable to a real instance label.
""", tag="rescue")

md(r"""
## Windows, and the split

**The split is on whole sequences, never on windows and never on frames.**
There are three ways to leak here and the split closes all three.

Two windows of one image share most of their pixels, so splitting *windows*
would put near-duplicates on both sides of the line. Consecutive frames of one
flight sit next to each other on disk, so a *file-order* split would do the
same; a seeded permutation fixes that.

Neither is enough on VTUAV. It annotates every 30th frame, so two masks from
one flight are **one second apart** — the same target, the same background,
barely a different pixel. Splitting frames puts one in train and one in test,
and the score then measures memory rather than generalisation. So `split_frames`
holds out whole sequences: `bike_009` is entirely in train or entirely in test,
never both. Sets with no sequence structure (Kust4K's `00001D`) are unaffected —
each frame is its own group and the result is the ordinary permutation.

This is also what makes the two notebooks comparable. Both hold out the same
VTUAV sequences, because both use the same seed on the same sequence names.

Then each frame gets `PER_IMAGE` windows, anchored on a randomly chosen
instance, with every other instance that falls **entirely** inside coming along
as a free extra prompt. Instances straddling the window edge are dropped rather
than truncated — a box that claims an extent its pixels do not have is a worse
target than no target.

Watch the **native** share. On 640×512 sources it should be ~100 %: the window
is a crop and the model sees sensor pixels. On a 4000×3000 source it will also
be high, because a 512 window is a small piece of a large frame. A low number
means most windows are whole-frame resizes, and small objects are being
destroyed by the resampling before training ever sees them.
""")

# ---------------------------------------------------------------- 14
code(r"""
# --- Split on frames, sample windows ------------------------------------
from src.training.aerial import sample_windows, split_index
from src.training.image_loop import ImageSplit

# Stratified per dataset, and on index entries rather than frame names --
# `000123.png` exists in most of these sets, so a name-keyed split would train
# on one dataset's frame while scoring another's.
splits = split_index(index, seed=SEED)

def windows(name, jitter):
    return ImageSplit(sample_windows(splits[name], size=SIZE, per_image=PER_IMAGE,
                                     max_instances=MAX_INSTANCES, jitter=jitter,
                                     seed=SEED))

train, val, test = windows("train", JITTER), windows("val", 0), windows("test", 0)
for name, split in (("train", train), ("val", val), ("test", test)):
    inst   = sum(len(s.instances) for s in split.samples)
    native = np.mean([s.native for s in split.samples]) if split.samples else 0
    print(f"{name:<6} {len(splits[name]):>5} frames  {len(split.samples):>6} windows  "
          f"{inst:>7} instances  {native:.0%} native pixels")
print("\ntrain windows by dataset:")
for source, count in train.sources.items():
    print(f"  {source:<26} {count:>6}")

overlap = {id(e) for e in splits["train"]} & {id(e) for e in splits["test"]}
assert not overlap, "train and test share a frame"
""")

# ---------------------------------------------------------------- 15
md(r"""
## The model, and what is allowed to move

Two stages, and the order is the argument: the **head first, alone**, so the
mask decoder adapts to these statistics before the features under it start
moving; then the **encoder joins at a tenth of the rate**, because the trunk
carries general visual features that took far more data to learn than this
dataset contains.

**The memory path never moves, in any stage.** Three separate reasons, any one
of which would be enough (`src/training/finetune.py` argues all three): it is
the write port of a recurrent loop and this project already measured what a
systematic change there does; its ONNX rewrite is the intricate one; and it
operates on abstract features, not pixels — a modality shift is an *encoder*
problem.

The assertion below is not decoration. A run that quietly trained the memory
path would show up as a bad checkpoint three hours later and nowhere else.
""")

# ---------------------------------------------------------------- 16
code(r"""
# --- Build it, and pin the freeze policy --------------------------------
from sam2.build_sam import build_sam2_video_predictor
from src.trackers._hydra_overrides import image_size_overrides
from src.training.finetune import Rates, apply_freeze, summarise_freeze

# image_size_overrides also fixes the cross-attention rotary table, which does
# not self-adjust to a new resolution.
model = build_sam2_video_predictor(
    "configs/edgetam.yaml", "third_party/EdgeTAM/checkpoints/edgetam.pt",
    device="cuda", hydra_overrides_extra=image_size_overrides(SIZE))
model.eval()   # deliberate: frozen batch-norm statistics keep the engines matching

print(summarise_freeze(apply_freeze(model, "encoder"), model))
for name, p in model.named_parameters():
    if name.startswith(("memory_attention", "memory_encoder", "spatial_perceiver")):
        assert not p.requires_grad, name
print("\nmemory path is frozen in the encoder stage")
""")

# ---------------------------------------------------------------- 17
md(r"""
## First: one window, before spending GPU hours

The question this answers is **plumbing**: do the prompt boxes, the mask and
the window geometry refer to the same pixels? If they do not, nothing further
down can work, and this is the cheapest possible place to find out.

The measure is **IoU on one window**, taken before any training and again
after. Loss is the wrong yardstick for it. Mask targets vary enormously in how
much of the frame they occupy, so the same loss means different things on
different windows — and an earlier version of this cell asked for the loss to
fall 3× whatever it started at, which failed a window the model *already*
segmented at 0.88 for the crime of leaving too little room to improve.

Two outcomes and they are not the same fault:

- **IoU stays low, before and after.** The plumbing is wrong. Check the
  overlay above, then the prompt coordinates and the mask alignment.
- **IoU starts high and the loss then runs away.** The plumbing is fine and
  the *optimiser* is the problem. Lower `Rates()` or lengthen the warm-up.

The cell distinguishes them, and restores the model either way.
""")

# ---------------------------------------------------------------- 18
code(r"""
# --- Can it learn one window? -------------------------------------------
import copy, gc
from src.training.finetune import param_groups
from src.training.image_loop import collate, image_losses, instance_iou

probe_sample = max(train.samples, key=lambda s: len(s.instances))
probe = collate([probe_sample], "cpu").to("cuda")
print(f"{probe_sample.source.spec.name} {probe_sample.frame.name}: "
      f"{len(probe_sample.instances)} instances, native={probe_sample.native}")

# Where it *starts* is most of the answer, and this cell used not to look. A
# window whose boxes and mask do not correspond scores near zero here no
# matter what the training then does; one that starts high has already proved
# the plumbing, and no amount of overfitting can prove it harder.
with torch.autocast("cuda", dtype=torch.bfloat16):
    start_iou = float(instance_iou(model, probe).mean())
print(f"IoU before a single step: {start_iou:.3f}")

snapshot = copy.deepcopy(model.state_dict())
# The production rates, warmed up -- not three times them from a cold
# optimiser. Adam's first step moves every parameter by the full lr whatever
# the gradient is, so an unwarmed 3e-4 on a window the model already segments
# is a large unprovoked perturbation rather than a gradient step. That is what
# this cell used to do, and on VTUAV it drove the decoder to "foreground
# everywhere" -- focal 14.4, dice 1.00, a flat line for the remaining 100
# steps -- on a window it had scored 0.22 at step 0.
STEPS, WARMUP = 150, 25
opt = torch.optim.AdamW(param_groups(model, Rates()))
schedule = [g["lr"] for g in opt.param_groups]
trainable = [p for g in opt.param_groups for p in g["params"]]

history = []
for step in range(STEPS):
    for group, lr in zip(opt.param_groups, schedule):
        group["lr"] = lr * min(1.0, (step + 1) / WARMUP)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, terms = image_losses(model, probe)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
    history.append(float(loss.detach()))
    if step % 25 == 0 or step == STEPS - 1:
        print(f"step {step:>4}  loss {history[-1]:8.4f}   " +
              "  ".join(f"{k} {v:.3f}" for k, v in terms.items()))
    # Stop the moment it runs away. Grinding out the remaining steps of a
    # saturated model costs minutes and adds nothing to the diagnosis.
    if history[-1] > max(20 * history[0], 5.0):
        print(f"step {step:>4}  runaway -- stopping here")
        break

with torch.autocast("cuda", dtype=torch.bfloat16):
    end_iou = float(instance_iou(model, probe).mean())
print(f"\nloss {history[0]:.3f} -> {min(history[-10:]):.3f}    "
      f"IoU {start_iou:.3f} -> {end_iou:.3f}")

# Put the model back *before* judging, so a failure here leaves something
# runnable behind. The old order raised with the overfit still loaded and the
# snapshot still held, which meant rebuilding the model to try anything.
model.load_state_dict(snapshot)
del snapshot, opt, probe
gc.collect(); torch.cuda.empty_cache()

# Two questions, and the assertion this replaces confused them into one. It
# demanded the loss fall 3x whatever it started at -- so a window the model
# already segments well, where 0.22 would have to reach 0.07, failed for being
# too easy -- and then blamed the data for it.
best = max(start_iou, end_iou)
if min(history[-10:]) > history[0]:
    raise AssertionError(
        f"the probe diverged: loss ended above where it began "
        f"({history[0]:.3f} -> {min(history[-10:]):.3f}). This is an "
        f"optimisation failure and not a plumbing one -- IoU {start_iou:.2f} "
        f"before any step says the prompt coordinates, the mask alignment and "
        f"the window geometry are already right. Lower `Rates()` or raise "
        f"WARMUP above, and re-run this cell.")
assert best > 0.60, (
    f"{len(history)} steps on one window and IoU never passed {best:.2f}. "
    f"*Now* the plumbing is the suspect: check prompt coordinates, mask "
    f"alignment and window geometry against the overlay above before "
    f"training on anything larger.")
print(f"plumbing sound (IoU {best:.2f} on one window); the loop learns")
""")

# ---------------------------------------------------------------- 19
md(r"""
## How many windows at once

Measured, not chosen. Image mode fits far more than clip mode does — no clip
length multiplying the activations, no memory bank — so the clip-mode number
would leave most of the card idle.

The probe uses the **widest** windows in the pool, not the first ones: cost
scales with the number of instances prompted, and measuring on a
single-instance window would report a size that OOMs the moment a crowded scene
comes round. Every candidate gets a real forward *and backward*, because the
backward is where the activation graph is actually held.

**What actually fills the card here is the loss, not the encoder.** EdgeTAM's
trunk is 4.92 M parameters; the mask terms run on `[N, 1, 512, 512]` logits
where `N = batch × instances`, and focal loss alone holds several tensors that
size. So on a large card the lever worth pulling first is **`MAX_INSTANCES`**,
not `BATCH_CEILING`: another instance costs one more decoder pass and **no
extra encode**, and it buys more supervision per image. On an 80 GB card,
`MAX_INSTANCES = 16` with the measured batch is a reasonable next step — but
only if cell {{fusion}} said the frames are dense enough to have that many.
""")

# ---------------------------------------------------------------- 20
code(r"""
from src.training.image_loop import auto_batch_size

apply_freeze(model, "encoder")            # measure under the heavier stage
BATCH = auto_batch_size(model, train, device="cuda", maximum=BATCH_CEILING)
ACCUM = max(1, 16 // BATCH)               # keep the effective batch comparable
print(f"batch {BATCH} x accum {ACCUM}")
del model; gc.collect(); torch.cuda.empty_cache()
""")

# ---------------------------------------------------------------- 21
md(r"""
## Stage A — distil an RGB foundation model into the thermal encoder

Optional (`PRETRAIN`), and the most interesting thing in this notebook.

AnyThermal's result is that a DINOv2-class RGB model can teach a thermal
encoder through **registered RGB-T pairs and no labels at all**: the teacher
looks at the RGB half, the student at the thermal half, and the student learns
to reproduce what the teacher saw. The instance-versus-semantic problem does
not arise, because no semantic map is read.

**Why this and not MAE.** MAE is how SAM 2's Hiera trunk was pretrained, so it
is the obvious first thought — and it does not transfer: MAE masks *tokens* and
is built for a ViT, while EdgeTAM's trunk is a convolutional RepViT with no
token grid to mask. Feature distillation has no such requirement. The teacher's
architecture and the student's are unrelated by construction; all that has to
line up is a feature map, and a 1×1 projection makes any two channel counts
line up. That is the answer to the open question in the TODO: yes, the target
attaches to a RepViT trunk, because the student side of a distillation loss is
architecture-free.

The pairs cost nothing to come by — this dataset already ships them registered,
and *every* pair counts, labelled or not.

### Where to get enough pairs: this is where VTUAV belongs

Stage A reads **no labels**, so a dataset's annotation is irrelevant to it and
only its *pair count* matters. That completely reorders the dataset list:

| source | registered RGB-T pairs | usable for stage A? |
|---|---:|---|
| **VTUAV** | **~1 700 000** (500 sequences, 1920×1080) | **yes — by far the largest** |
| MVUAV | ~53 800 | yes |
| SegFly | 15 007 | yes |
| Kust4K | 4 024 | yes |
| *AnyThermal's own training set* | *16 943* | *for scale* |

So the answer to "is VTUAV too little because it is unmasked?" is the reverse:
**being unmasked costs it nothing here.** Stage A reads registered pairs, not
labels, and VTUAV has more of them than everything else on the list put
together. Its box annotations and its mask subset are simply not what this
stage reads — `train_001` alone hands stage A 26 059 pairs, more than
AnyThermal's entire training set, out of the same download stage B uses for its
875 masks.

Two settings matter more for VTUAV than anywhere else, both because it is video
at 1920×1080:

- **`DISTILL_CROP = 512/1080` (≈ 0.474).** Squeezing a 16:9 frame into 512×512
  distorts the aspect ratio *and* shrinks a small vehicle below the size the
  encoder will ever see it at — so the encoder would be pretrained on image
  statistics the deployment never produces. A crop takes the same square from
  **both halves at the same normalised position**, so the registration
  survives, and at this fraction it is native pixels.
- **`DISTILL_PAIRS`.** The cap samples *across* the set rather than truncating
  it. On a video-derived set that is not a nicety: the first 5 000 pairs of
  VTUAV are two flights, so a truncating cap would train on two scenes and
  report five thousand samples.

The catch is disk, not labels: 1.7 M frames at 1920×1080 is far more than a
Colab runtime holds, and the mask split alone is ~120 GB across its eight
archives. `FETCH` asks for the three **training** archives and stops there —
43.1 GB, which a Colab disk holds and the full 120 GB does not. `train_001`
alone would do for stage A (26 059 pairs, already above AnyThermal's whole
training set, and enough to saturate it at `DISTILL_PAIRS = 20 000`); the other
two are there for stage B, for the reason below.

**For stage B, one archive is not enough, and the reason is not its size.**
Reading the three training archives' listings, the target kinds are badly
unevenly spread:

| archive | | sequences | masks | targets |
|---|---:|---:|---:|---|
| `train_001` | 9.1 GB | 14 | 875 | bike 1, bus 4, c-vehicle 1, car 8 |
| `train_002` | 16.1 GB | 18 | 1 408 | car 4, elebike 3, excavator 2, pedestrian 9 |
| `train_003` | 17.9 GB | 18 | 1 778 | pedestrian 15, train 1, tricycle 1, truck 1 |

`train_001` **has no pedestrians in it at all.** On its own it teaches that a
target is a vehicle, which for a drone tracker is a hole rather than a bias.

It also decides what a held-out number is worth. 14 sequences split 11/2/1, so
`test` is a **single flight** and its quirks dominate any comparison between two
runs — which is the entire point of running this notebook against its sibling.
All three archives give 50 sequences and a 40/5/5.

That is why `FETCH` names all three. If the disk will not take 43 GB, cut
`train_002` first — `train_003` carries the most masks and the most
pedestrians, and dropping to one archive costs the held-out split its meaning.
With all three, raising `DISTILL_PAIRS` past 20 000 starts to buy something
too; on `train_001` alone it does not.

{{TEACHER_HEADING}}

The task is **single-object tracking with no class in it**: the operator draws
a box, the model follows *that* object, and nothing anywhere asks what the
object is. That sentence decides the teacher, so it is worth putting the
candidates side by side.

| | **SAM 2.1 Hiera-B+**{{SAM_TAG}} | DINOv3-ViT-B/16{{DINO_TAG}} | DINOv2-base |
|---|---|---|---|
| what its features encode | **class-agnostic boundaries** — something is here, it ends there | **semantics**, and specifically **dense** ones — fixing dense degradation is what the release is about | semantics |
| fit to *this* task | the task's definition: no class is ever asked for | semantics is real, and is surplus here | same, one generation back |
| match to our target tensor | `fpn_hidden_states[-1]` **is** 256-d at stride 16 — the projection is 256→256, not a change of representation | 768-d → projected to 256 | 768-d → projected to 256 |
| relation to the student | EdgeTAM **is** a distillation of SAM 2; this resumes that objective across a modality gap | unrelated model | unrelated model |
| alignment risk | pushes the encoder **toward** the space `memory_attention` expects | pushes it away — this run carries the risk stage C exists to repair | same |
| grid at our 512 input | 1024/16 = 64×64, an exact 2× down to the student's 32 | 512/16 = **32×32 — the student's grid exactly**, no resampling | 518/14 = 37×37, a fractional ratio |
| cost per step | ~3× a base ViT. 600 steps at batch 32 is 0.037 A100-hours — not a constraint at this size | 512 input, base-sized | 518 input, base-sized |
| access | open | **gated** — accept the terms once, set `HF_TOKEN` | open |

**Base-plus and not Large, and cost is not the reason.** EdgeTAM's own paper
distilled from **SAM2-Hiera-B+** targeting **F16, the encoder feature map at
stride 16** — the exact tensor this stage reads. The student's trunk weights
already *are* a fit to that target, so B+ resumes the objective the checkpoint
was born from, while Large would pull it toward a different one and give up the
single property that makes a SAM teacher interesting.

{{TEACHER_NOTE}}

*Whichever it is, it cannot break anything structurally.* Distillation changes
**weight values inside `image_encoder`** and nothing else: same modules, same
state-dict keys, same ONNX graph, same engines, and the projection head is
discarded. Stage A is only an **initialisation** — stage B then trains against
SAM 2's own objective (promptable segmentation, focal + dice + IoU) on real
masks, so an unhelpful teacher costs GPU time and not correctness. The
`distilled+ft` run below exists to put a number on that.

*The real risk is not structural:* stage A moves the encoder while the mask
decoder is frozen, so the decoder can end up reading features it was never
trained for. That is why the trunk moves at 5e-5, why there is an EMA, and why
stage B starts with a **head-only** stage — the decoder re-adapts before the
encoder moves again. A SAM teacher shrinks this risk; a DINO teacher grows it.

The input size is picked from the teacher's id, and for SAM 2 it is 1024 for a
reason: the position embeddings *are* interpolated to smaller grids, but the
features move a long way when they are — cosine similarity to the model's own
1024 output is 0.842 for B+ at 512, 0.736 for Large. Lowering it changes the
teacher, not only its cost.

### Holding on to it: `ANCHOR_WEIGHT`

Stage A moves the encoder over ~1.7 M unlabelled pairs. Stage B then trains it
on a labelled set two orders of magnitude smaller — and nothing stops it
drifting straight back. The third run below adds a term for how far the encoder
has travelled from where stage A left it:

```
stage B loss = 20·focal + dice + IoU  +  λ · (1 − cos(features now, features after stage A))
```

That is feature distillation used as a **regulariser, not a teacher**. The
reference is the model's own frozen copy — not the stage-A teacher — for three
reasons: what is worth keeping is what stage A *learned*, the copy is 4.92 M
parameters rather than a foundation model, and its features are already in the
student's own space so no projection is needed. It costs **no extra encoder pass**: `propagate_image`
hands back the map it already computed.

`ANCHOR_WEIGHT = 0` turns it off and makes the third run a pure
"did stage A survive?" test. Both are worth having; the notebook uses 0.5.

### Two knobs from the literature, both off or small by default

`DISTILL_MOMENTS` adds a per-channel mean/std term beside the cosine one. Pure
direction says nothing about how a channel is used *across the map*, so a
student can score well position by position while half its channels go
constant. The closest published recipe (arXiv 2604.27128) warns of a
**collapsed-variance failure mode** when that kind of term gets weight
comparable to the directional one, and reports no ablation — so it is a
hypothesis, default 0.

`TEACHER` also accepts `facebook/dinov3-vitl16-pretrain-sat493m`, DINOv3
pretrained on 493 M **satellite** tiles. The intuition says a satellite model
must beat a web one on aerial imagery. **DINOv3's own results say otherwise**:
satellite pretraining helps *metric* tasks like depth, while the web-pretrained
model wins on segmentation and detection — and sets the state of the art on
high-resolution remote-sensing segmentation. Ours is segmentation. There is a
second reason to doubt it: SAT-493M is 0.6 m/px ortho and UAV footage is a few
cm/px — overhead in the same sense, ~50× apart in scale. It is one string away
if you want the measurement anyway.

**And whichever it is, it has to beat doing nothing.** The two runs below start
from the stock checkpoint precisely so this one can be compared against them. A
pretraining stage that does not win is a pretraining stage that cost GPU hours
and bought nothing, and only the comparison can say which it is.

The full assessment of this design — what the literature supports, what it
contradicts, and what was changed because of it — is `docs/encoder_arastirma.md`.
""")

# ---------------------------------------------------------------- 22
code(r"""
%%time
# The teacher is a second model on the card, so it gets half the batch. The
# student's activations are the same size either way; the teacher -- a ViT-B/16
# at 512, so a 32x32 patch grid -- is what has to fit beside them.
DISTILLED     = CKPT / "edgetam_distilled_512.pt"
DISTILL_BATCH = max(BATCH // 2, 1)
# Stage B may have corrected this dataset's globs in OVERRIDES; stage A has to
# read the same halves of the same archive, so they travel with it.
_pair_spec    = next((src.spec for label, src in sources.items()
                      if label.startswith(DISTILL_SPEC.replace("_vis", ""))), None)
DISTILL_ARGS  = ((f"--pairs {DISTILL_PAIRS} " if DISTILL_PAIRS else "")
                 + (f"--crop {DISTILL_CROP} " if DISTILL_CROP else "")
                 + (f"--thermal-glob '{_pair_spec.thermal}' "
                    f"--rgb-glob '{_pair_spec.rgb}'" if _pair_spec else ""))
print("stage A args:", DISTILL_ARGS or "(defaults)")

if PRETRAIN:
    !python tools/pretrain_encoder.py --data {DISTILL_ROOT} --spec {DISTILL_SPEC} \
        --teacher {TEACHER} --out {DISTILLED} --size {SIZE} \
        --batch {DISTILL_BATCH} --steps {DISTILL_STEPS} {DISTILL_ARGS} \
        --tolerance {DISTILL_TOL} --moments {DISTILL_MOMENTS} \
        --workers {LOADER_WORKERS} --seed {SEED} \
        --json {WORK}/run_distill.json 2>&1 | tail -12
    mirror_now(DISTILLED, WORK / "run_distill.json")
else:
    print("skipped -- set PRETRAIN = True to run stage A")
""")

# ---------------------------------------------------------------- 23
md(r"""
## The runs

Same command, same budget, one flag apart. Everything except the swapped
callback goes through `src/training/schedule.py`: the same stages, the same
windows in the same order from the same seed, the same losses, the same
one-cycle schedule, the same gradient clipping, the same EMA, the same
validation slice, the same rule for keeping a checkpoint. `--method` changes
which parameters receive a gradient and how the checkpoint is written. Nothing
else.

| | partial fine-tune | LoRA (r=16) |
|---|---|---|
| what moves | the head, then the head + image encoder | `B @ A` beside each of those layers' weights |
| trainable | ~9.3 M | ~0.3 M, measured below |
| regularisation | freezing the memory path | the rank itself, plus that same freeze |
| what ships | a checkpoint | **the same checkpoint** — adapters merged before writing |

That last row is what makes it a fair test. A `k × k` convolution composed with
a `1 × 1` **is** a `k × k` convolution, so the factorisation is exact for the
RepViT trunk too — the usual "LoRA only reaches `nn.Linear`" objection is an
objection to a Linear-only implementation, not to the method.

**One thing is deliberately not held identical: the learning rate.** LoRA starts
from `B = 0`, so its update has further to travel; every published recipe gives
it roughly an order of magnitude more, and matching the fine-tune's rate would
be testing a badly-tuned LoRA rather than LoRA. Both rates are in
`tools/train_encoder.py:RATES`, in one place, visible.

The third run answers the *other* question — whether stage A paid for itself —
by changing only `--base`.
""")

# ---------------------------------------------------------------- 24
code(r"""
%%time
DATASET_FLAGS = " ".join(f"--dataset {d}" for d in DATASETS)
COMMON = (f"{DATASET_FLAGS} --index {INDEX} "
          f"--size {SIZE} --per-image {PER_IMAGE} --max-instances {MAX_INSTANCES} "
          f"--min-area {MIN_AREA} --min-side {MIN_SIDE} --max-area {MAX_AREA} "
          f"--fill {FILL} --jitter {JITTER} --batch {BATCH} --accum {ACCUM} "
          f"--steps {STEPS_PER_EPOCH} --epochs {EPOCHS[0]} {EPOCHS[1]} "
          f"--val-batches {VAL_BATCHES} --workers {LOADER_WORKERS} "
          f"--depth {PREFETCH_DEPTH} --seed {SEED}")

!python tools/train_encoder.py --method finetune {COMMON} \
    --out {CKPT}/edgetam_aerial_512.pt --json {WORK}/run_finetune.json 2>&1 | tail -22
mirror_now(CKPT / "edgetam_aerial_512.pt", WORK / "run_finetune.json")
""")

# ---------------------------------------------------------------- 25
code(r"""
%%time
!python tools/train_encoder.py --method lora --lora-r {LORA_R} {COMMON} \
    --out {CKPT}/edgetam_aerial_lora_512.pt --json {WORK}/run_lora.json 2>&1 | tail -22
mirror_now(CKPT / "edgetam_aerial_lora_512.pt", WORK / "run_lora.json")
""")

# ---------------------------------------------------------------- 26
code(r"""
%%time
# Same recipe as the fine-tune, different starting weights -- so the only thing
# this run can be measuring is what stage A did.
if PRETRAIN:
    !python tools/train_encoder.py --method finetune {COMMON} \
        --base {DISTILLED} --anchor-weight {ANCHOR_WEIGHT} \
        --out {CKPT}/edgetam_aerial_distilled_512.pt \
        --json {WORK}/run_distilled.json 2>&1 | tail -22
    mirror_now(CKPT / "edgetam_aerial_distilled_512.pt",
               WORK / "run_distilled.json")
else:
    print("skipped -- no stage A checkpoint to start from")
""")

# ---------------------------------------------------------------- 27
code(r"""
# --- What each run cost -------------------------------------------------
import json

names = ["finetune", "lora"] + (["distilled"] if PRETRAIN else [])
runs  = {n: json.loads((WORK / f"run_{n}.json").read_text()) for n in names}

print(f"{'':<12}{'trainable':>12}{'batch':>8}{'peak GiB':>10}{'minutes':>9}{'best val':>11}")
for name, run in runs.items():
    moving = run.get("lora_parameters") or run.get("trainable_parameters", 0)
    print(f"{name:<12}{moving / 1e6:>11.3f}M{run['batch']:>8}{run['peak_gib']:>10.1f}"
          f"{run['seconds'] / 60:>9.0f}{run['best_val_loss']:>11.4f}")

share = runs["lora"].get("lora_parameters", 1) / max(
    runs["finetune"].get("trainable_parameters", 1), 1)
print(f"\nLoRA trains {share:.2%} of what the fine-tune does "
      f"({runs['lora'].get('adapted_layers', 0)} layers adapted, r={LORA_R})")
if PRETRAIN:
    d = json.loads((WORK / "run_distill.json").read_text())
    print(f"stage A: teacher {d['teacher']}, {d['pairs']} unlabelled pairs, "
          f"{d['steps']} steps, cosine distance {d['final_loss']:.4f}, "
          f"{d['seconds'] / 60:.0f} min")
    print("  (a cosine distance is not comparable across teachers -- only the "
          "test-split column below is)")
""")

# ---------------------------------------------------------------- 28
md(r"""
## The measurement that decides

Validation loss chose each checkpoint, so validation cannot also be the
verdict. Everything below runs on **`test`** — frames no run has seen in any
form, from the same seeded frame-level split, scored the same way: box in, mask
out, IoU.

Four checkpoints, one command each, identical except for the weights. The stock
one is the floor; without it, "0.71 IoU" is a number with no scale.

**Read the small-object column, not the mean.** An aerial dataset's mean IoU is
dominated by its trucks, and the targets this pipeline exists for are the ones
twenty pixels across. A method that moves the mean by improving large objects
has not helped the deployment at all.

And read all of it as a **proxy**. This scores one prompted frame, so it cannot
see the failure this project is actually trying to fix — that failure needs a
memory bank, and the memory bank is stage C.
""")

# ---------------------------------------------------------------- 29
code(r"""
CHECKPOINTS = {
    "stock":     "third_party/EdgeTAM/checkpoints/edgetam.pt",
    "finetune":  f"{CKPT}/edgetam_aerial_512.pt",
    "lora":      f"{CKPT}/edgetam_aerial_lora_512.pt",
}
if PRETRAIN:
    CHECKPOINTS["distilled+ft"] = f"{CKPT}/edgetam_aerial_distilled_512.pt"

for label, path in CHECKPOINTS.items():
    print(f"\n===== {label} =====")
    tag = label.replace("+", "_")          # a shell-safe filename
    !python tools/eval_instances.py {DATASET_FLAGS} --index {INDEX} \
        --checkpoint {path} --split test --size {SIZE} \
        --per-image {PER_IMAGE} --max-instances {MAX_INSTANCES} \
        --min-area {MIN_AREA} --min-side {MIN_SIDE} --max-area {MAX_AREA} \
        --fill {FILL} --batch {BATCH} --seed {SEED} \
        --json {WORK}/test_{tag}.json 2>&1 | tail -14
""")

# ---------------------------------------------------------------- 30
code(r"""
# --- Side by side -------------------------------------------------------
scores = {label: json.loads((WORK / f"test_{label.replace('+', '_')}.json").read_text())
          for label in CHECKPOINTS}

print(f"test split, {list(scores.values())[0]['instances']} instances, "
      f"seen by no run\n")
print(f"{'':<14}{'mean IoU':>10}{'IoU>=.5':>10}{'IoU>=.75':>10}"
      f"{'small IoU':>11}{'larger':>9}")
for label, row in scores.items():
    print(f"{label:<14}{row['mean_iou']:>10.4f}{row['iou_50']:>10.3f}"
          f"{row['iou_75']:>10.3f}{row['small_mean_iou']:>11.4f}"
          f"{row['large_mean_iou']:>9.4f}")

base = scores["stock"]
classes = sorted({c for row in scores.values() for c in row["per_class"]})
y = np.arange(len(classes))
fig, ax = plt.subplots(figsize=(8.0, 0.5 * len(classes) + 2.0))
palette = ["#4c72b0", "#dd8452", "#55a868"]
runs_shown = [k for k in scores if k != "stock"]
for i, label in enumerate(runs_shown):
    offset = (i - (len(runs_shown) - 1) / 2) * 0.8 / len(runs_shown)
    delta = [scores[label]["per_class"].get(c, {}).get("mean_iou", np.nan)
             - base["per_class"].get(c, {}).get("mean_iou", np.nan)
             for c in classes]
    ax.barh(y + offset, delta, height=0.8 / len(runs_shown), label=label,
            color=palette[i % len(palette)])
ax.set_yticks(y); ax.set_yticklabels(classes, fontsize=8)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("mean IoU − stock  (test split, per class)")
ax.legend(loc="lower right")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 31
code(r"""
# --- Keep it ------------------------------------------------------------
import shutil

verdict = {"datasets": DATASETS, "image_size": SIZE,
           "frames": {k: len(v) for k, v in splits.items()},
           "train_windows_by_dataset": train.sources,
           "instances": len(items), "gates": gates.__dict__,
           "teacher": TEACHER if PRETRAIN else None,
           "pretrained": PRETRAIN, "runs": runs, "test": scores}
(WORK / "encoder_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")

for path in CKPT.glob("edgetam_aerial*.pt"):
    shutil.copy(path, MIRROR / path.name)
if PRETRAIN and DISTILLED.is_file():
    shutil.copy(DISTILLED, MIRROR / DISTILLED.name)
for name in ("encoder_verdict.json", *(f"run_{n}.json" for n in names)):
    shutil.copy(WORK / name, MIRROR / name)

print(f"checkpoints, logs and the instance indexes -> {MIRROR}")
!ls -la {MIRROR}
""")

# ---------------------------------------------------------------- 32
code(r"""
# --- In plain words: what this run produced -----------------------------
def gain(name, key="mean_iou"):
    return scores[name][key] - scores["stock"][key]

line = "-" * 72
print(line)
print(f"FILES ON YOUR DRIVE  ({MIRROR})")
print("  edgetam_aerial_512.pt            the fine-tuned encoder")
print("  edgetam_aerial_lora_512.pt       the LoRA encoder, adapters merged")
if PRETRAIN:
    print("  edgetam_distilled_512.pt         stage A only -- saw no masks at all")
    print("  edgetam_aerial_distilled_512.pt  stage A, then stage B on top")
print("  index/*.json                     the instance index, reusable next session")
print("  encoder_verdict.json             every number below, machine-readable")
print()
print("  Each .pt is an ordinary EdgeTAM checkpoint: same keys, same loader,")
print("  same ONNX export. configs/edgetam_512_aerial{,_lora}.yaml point at them.")

print(line)
print("WHAT IT TRAINED ON")
total_frames = sum(len(v) for v in splits.values())
print(f"  {total_frames} frames -> {len(items)} instances -> "
      f"{len(train.samples)} train windows")
for source, count in train.sources.items():
    print(f"    {source:<28} {count:>7} windows")

print(line)
print("THE FOUR QUESTIONS IT ANSWERED")
verdict_fill = "OK" if fused <= 0.15 else "HIGH -- read cell {{rescue}} before trusting anything below"
print(f"  1. are the targets clean?   {fused:.1%} of components rejected on `fill`  [{verdict_fill}]")
winner = "fine-tune" if runs["finetune"]["best_val_loss"] < runs["lora"]["best_val_loss"] else "LoRA"
print(f"  2. fine-tune or LoRA?       {winner} won on validation loss")
print(f"                              on test, mean IoU vs stock: "
      f"finetune {gain('finetune'):+.4f}   lora {gain('lora'):+.4f}")
if PRETRAIN:
    print(f"  3. did pretraining pay?     distilled+ft {gain('distilled+ft'):+.4f} "
          f"vs finetune {gain('finetune'):+.4f}")
else:
    print("  3. did pretraining pay?     not run (PRETRAIN = False)")
print(f"  4. did SMALL targets move?  finetune {gain('finetune', 'small_mean_iou'):+.4f}   "
      f"lora {gain('lora', 'small_mean_iou'):+.4f}")
print("                              <- the column the deployment actually lives in")

print(line)
print("WHAT THIS RUN DID *NOT* MEASURE")
print("  Tracking. Every number above scores one prompted frame, so none of them")
print("  can see the failure this project exists to fix -- that needs a memory")
print("  bank, and the memory bank is stage C. A better encoder is a")
print("  precondition for a better tracker, not evidence of one.")
print(line)
""")

md(r"""
## What you have, and how to read it

`edgetam_aerial_512.pt` and `edgetam_aerial_lora_512.pt` are on your Drive, in
EdgeTAM's own `{"model": state_dict}` layout, so `configs/edgetam_512_aerial.yaml`
and `configs/edgetam_512_aerial_lora.yaml` already point at them and the
export/build chain needs no change:

```bash
python tools/export_edgetam_onnx.py --outdir models512_aerial/ --image-size 512 \
    --checkpoint checkpoints/edgetam_aerial_512.pt --verify
python tools/build_trt_engines.py --outdir models512_aerial/ --max-batch 4
python tools/check_trt_parity.py --outdir models512_aerial/
```

Only the **encoder** engine actually changed — but the encoder ONNX graph folds
the mask decoder's `conv_s0`/`conv_s1` projections into itself, and this stage
trained the decoder too, so re-export both and run parity on all four.

### The four outcomes, and what each one means

| what the test split says | what it means, and what to do |
|---|---|
| fine-tune ahead of stock on **small** instances | the stage worked. Freeze this encoder and start stage C — the video data is in `docs/datasets.md` |
| ahead on the mean, flat on small instances | it learned the trucks. The deployment does not track trucks. Raise `MAX_INSTANCES`, lower `MIN_AREA`, and check the size histogram before rerunning |
| LoRA level with the fine-tune | the regularisation argument was real, at 3 % of the parameters and a smaller optimiser state. Prefer LoRA and record it — this is the case where `finetune.py`'s §1 needs rewriting |
| everything level with stock | the bottleneck is upstream of the method. Look again at the fusion number in cell {{fusion}}: if the instances are fused, the targets were wrong and nothing downstream could have helped |
| `distilled+ft` no better than `finetune` | stage A did not pay for itself **with this teacher, on this dataset**. Two things to try before dropping it: {{OTHER_TEACHER}}, and more pairs — it is a data-hungry stage, and VTUAV's ~1.7 M pairs are the honest test of it, not Kust4K's 2 864 |
| `distilled+ft` **worse** than `finetune` | the encoder drifted somewhere the frozen decoder cannot read. Lower `--trunk-lr`, or raise the head-only epoch count (`EPOCHS = (2, 3)`) so the decoder gets longer to re-adapt before the trunk moves |

**One run is one sample.** Every command takes `--seed`; re-run each with two or
three before believing a gap under ~0.01 mean IoU. The per-class chart is the
honest view — a method that wins the mean by rescuing one class and breaking
another is not the same result as one that helps everywhere.

### What this notebook cannot tell you

Nothing here measures **tracking**. The metric this project is judged on is
J&F over a video sequence, and it depends on the memory path, which stage B
never touches and never sees. A better encoder is a *precondition* for a better
tracker, not evidence of one.

**Next:** stage C. Freeze the encoder this notebook produced, unfreeze
`memory_attention` and `memory_encoder`, and train on video with instance
identity — AeroVIS or UAVScenes, alternating with the static data the way SAM 2
does. `FROZEN_MODULES` in `src/training/finetune.py` does the opposite today,
so that stage needs its own entry in `STAGES` before it can start.
""")


def stamp_for(variant: Variant) -> str:
    """The build id: this variant's notebook, hashed.

    Hashed *before* substitution, so `{{STAMP}}` is still a literal in the
    text and the stamp does not depend on itself. Any change to any cell --
    prose included -- gives a new one.

    The variant's own fields go in too: they are still `{{placeholders}}` in
    the cell text, so without them every notebook would stamp identically and
    a swapped pair would look correct. They are read off the dataclass rather
    than listed by hand. The hand-written list this replaces named six of the
    ten fields, and two it missed were `teacher` and `distill` -- so changing
    10's teacher, the one setting that notebook exists to change, rebuilt it
    to a byte-identical stamp. A stamp that cannot move is a staleness check
    that always answers "current", which is the failure the mechanism was
    added to catch.
    """
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
        out = repo / V.path
        out.parent.mkdir(parents=True, exist_ok=True)
        document = build()
        out.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n")

        # Read-modify-write: each variant is built in its own process, so
        # overwriting here would leave the file holding only the last one.
        stamps = repo / "notebooks" / ".stamps.json"
        known = json.loads(stamps.read_text()) if stamps.is_file() else {}
        known[out.name] = STAMP_VALUE
        stamps.write_text(json.dumps(known, indent=1, sort_keys=True) + "\n")
        print(f"wrote {out.relative_to(repo)} with {len(CELLS)} cells "
              f"[{STAMP_VALUE}]")
    else:
        # Re-run per variant rather than rebuilding in-process: the cells above
        # are emitted at import time against a single `V`, and a second pass in
        # the same interpreter would append to the first one's CELLS.
        for key in VARIANTS:
            subprocess.run([sys.executable, __file__, key], check=True,
                           cwd=repo)
