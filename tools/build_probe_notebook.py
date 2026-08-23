#!/usr/bin/env python3
"""Build `notebooks/12_encoder_probe.ipynb` from plain text blocks.

    python tools/build_probe_notebook.py

**Why this is its own file rather than a sixth `Variant`.** The four notebooks
`tools/build_notebooks.py` emits are one program with one thing changed --
same schedule, same seed, same cells, a different dataset or teacher -- and the
`Variant` dataclass exists to hold exactly that difference. This notebook is
not that. It trains almost nothing, re-scores checkpoints that already exist,
and its whole subject is the *measurement* rather than the model. Forcing it
into that mould would mean wrapping seven hundred lines of emission in a
conditional to skip them.

What it is for: `08_encoder_vtuav_only.ipynb` finished with `finetune` at
+0.0168 mean IoU over stock and `lora` at +0.0257, on a test split of 420
instances drawn from five held-out flights, with the `small` column reading
`nan` throughout. Three separate things could produce that table --

    the encoder work bought little,
    the encoder work bought something the prompt hides,
    the encoder work bought something this data cannot show --

and the run as written cannot tell them apart, because it varies the
checkpoint and holds the *question* fixed. This notebook varies the question:
the same four checkpoints, scored under three prompts and two window scales.
A gap that opens when the prompt weakens was always there and was being given
away by the box; a gap that stays shut under every condition is a gap that is
not there.

It also settles the one thing 08 left genuinely ambiguous. `distilled+ft` came
in behind plain `finetune` on both val and test, and two mechanisms explain
that equally well: stage A moved the encoder somewhere worse, or
`ANCHOR_WEIGHT` held it there. One run at `--anchor-weight 0` separates them,
and it is the only training this notebook does.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

CELLS: list[tuple[str, str]] = []
TAGS: dict[str, int] = {}
STAMP_VALUE = "unstamped"

NOTEBOOK = "12_encoder_probe.ipynb"
BRANCH = "claude/encoder-architecture-colab-myo61y"


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

    Same mechanism, and the same reason, as `tools/build_notebooks.py`: prose
    that points at a cell number goes stale the moment a cell is inserted
    above it, and a notebook that sends you to the wrong cell is worse than
    one that points nowhere.
    """
    substitutions = {**{t: str(i) for t, i in TAGS.items()},
                     "NOTEBOOK": NOTEBOOK, "BRANCH": BRANCH,
                     "STAMP": STAMP_VALUE}
    for tag, value in substitutions.items():
        text = text.replace("{{" + tag + "}}", value)
    leftover = re.findall(r"\{\{(\w+)\}\}", text)
    if leftover:
        raise KeyError(f"unknown cell tags: {sorted(set(leftover))}")
    return text


# --------------------------------------------------------------------------
# The notebook
# --------------------------------------------------------------------------

md(r"""
# Was the encoder work small, or was it unmeasured?

`08_encoder_vtuav_only.ipynb` produced a table that reads like a null result:

| | mean IoU | IoU≥0.5 | IoU≥0.75 | small IoU |
|---|---:|---:|---:|---:|
| stock | 0.8444 | 0.998 | 0.900 | `nan` |
| finetune | 0.8609 | 1.000 | 0.888 | `nan` |
| lora | 0.8666 | 1.000 | 0.914 | `nan` |
| distilled+ft | 0.8360 | 1.000 | 0.871 | `nan` |

Two hundred GPU-minutes for +0.017 and +0.022 mean IoU, a `small` column that
is empty, and the two fine-tuning methods tied on validation loss to four
decimal places.

**One error bar is already known, and it is uncomfortably large.** 08 was run
twice — once with a 600-step stage A, once with 12 000 — and `finetune` and
`lora` never touch stage A, so those two rows should have been identical
across the pair. They were not: `finetune` moved 0.8611 → 0.8609 and `lora`
0.8700 → 0.8666, and `lora`'s IoU≥0.75 swung 0.940 → 0.914. So **re-running
the same configuration moves mean IoU by about 0.003**, and the tail metric by
ten times that. The `finetune`-to-`lora` gap of 0.0057 is under two of those.
Nothing in the matrix below separates two rows closer together than that.

**A null result and an unasked question look identical from here.** That table
varies the checkpoint and holds the question fixed, and the question it holds
fixed happens to be the easiest one available:

* **The prompt gives the answer away.** `eval_instances.py` prompts with the
  ground-truth box. On one large, high-contrast target against uniform ground,
  four exact edges are most of the mask — the decoder can do well from the
  prompt nearly alone, and the encoder underneath barely enters into it. Stock
  EdgeTAM already scores 419 of 420 instances above IoU 0.5 *before any
  training at all*. There is no room left for an encoder to win in.
* **Every target is large.** VTUAV's mask split is one tracked object per
  frame, a median 186 source pixels across. The bucket the deployment lives
  in — targets a few dozen pixels wide — has no members, so `small` is `nan`
  and stays `nan` however good the encoder gets.

So this notebook re-asks the question three ways and two scales, on **the same
four checkpoints, already trained**. Nothing is retrained to fill the table,
which is why the whole matrix costs about ten minutes.

## The two axes

**Prompt.** `box` is what 08 used: the ground-truth rectangle, exactly.
`jitter` moves each edge by up to 30% of its own side — a loose rectangle, the
kind an operator draws or a detector emits. `point` is one click at the centre
and nothing else, so extent has to come from the features. Reading left to
right across a row is reading *how much of the score was the prompt*.

**Window.** The crop is taken in source pixels and resized to the model's 512,
and 08 held those equal — a 512 crop at 512, no resampling. Doubling the crop
to 1024 halves every target's apparent size without touching a single
annotation. VTUAV is 1080 pixels tall, so 2× is very nearly the ceiling here:
a square crop cannot exceed the short edge, and past it the fallback squashes
the whole 1920×1080 frame anisotropically instead. Half of 186 is 93, so this
does **not** manufacture the 20-pixel regime — it populates the lower tail and
no more. Read it as a direction, not as the deployment.

## What each outcome would mean

| the matrix shows | reading |
|---|---|
| gaps **open** as the prompt weakens | the encoder work was real and the box was hiding it. Stage B is worth continuing, and every number in 08 is an underestimate |
| gaps stay shut **everywhere** | the encoder work genuinely bought little on this data. Look at the data, not the recipe — 07's mixed sets, or an altitude range with small targets |
| gaps open **only** at window 1024 | what moved was scale robustness, not features. Interesting, and an argument for training at mixed scales |
| `point` collapses for **every** checkpoint alike | the mask decoder cannot work from a click at this target size, and the prompt axis says nothing about encoders. Fall back to `jitter` as the discriminating column |

## And one training run

`distilled+ft` finished behind plain `finetune`, and **giving stage A twenty
times more data made it worse, not better**:

| stage A budget | cosine distance | head-epoch val | best val | test mean IoU |
|---|---:|---:|---:|---:|
| 600 steps, 20 000 pairs | 0.2869 | 0.2388 | 0.1527 | 0.8536 |
| 12 000 steps, 121 401 pairs | **0.1968** | **1.5461** | **0.3138** | **0.8360** |

Read the first two numeric columns together, because that is the whole story.
The distillation objective got substantially *better* — cosine distance fell
from 0.287 to 0.197 — and the very first thing stage B measures got six times
*worse*. That is not a starved stage. That is a stage doing exactly what it was
asked to do, to an encoder whose decoder cannot follow it.

The mechanism is not mysterious. EdgeTAM's mask decoder was trained against
EdgeTAM's *own* encoder features. Distilling the encoder toward SAM 2.1
Hiera-B+ moves it out of the space the decoder reads, and the harder the
distillation pulls, the further out it goes. At 600 steps the encoder barely
moved and the damage was small; at 12 000 it moved a long way and the head
epoch opened at 1.5461 against 0.2065 for a run starting from stock. Three
encoder epochs claw it back to 0.31 and there it plateaus — nowhere near the
0.142 the other two runs reach.

So the anchor question is now the narrow one: `ANCHOR_WEIGHT = 0.5` held the
encoder at that point (the anchor term sat at 0.01 throughout, so it never
drifted). Cell {{anchor_run}} releases it. If the encoder crawls back toward
something the decoder can read, the recipe is salvageable — a longer head-only
warmup, `EPOCHS = (3, 3)`, would then be the fix. If it stays at 0.31, feature
distillation into this encoder is not salvageable by tuning and the stage
should be dropped rather than re-tuned.

---

**Prerequisite: `08_encoder_vtuav_only.ipynb` must have finished**, because
this notebook scores the checkpoints it left on Drive and reuses its instance
index. Cell {{inventory}} checks for both and stops with a clear message
rather than failing later. VTUAV itself is re-extracted here (the frames are
needed to score), which is the slow part of the setup.

**Safe to run beside a fresh 08.** Cell {{inventory}} copies the checkpoints
and the index off Drive to local disk before anything is scored, so a second
runtime re-running 08 cannot replace them halfway through the matrix and turn
it into a comparison across two experiments.

What the snapshot does **not** do is choose which run it captures: 08 writes
to one folder and overwrites it every time, so this notebook measures whatever
08 wrote **last**. Check the timestamps cell {{inventory}} prints against the
run you mean to be scoring. If you want to keep an older set, copy it aside
before re-running 08 — nothing here does that for you.

*This notebook is generated — edit `tools/build_probe_notebook.py`, not the
`.ipynb`.*
""")

code(r"""
# --- Runtime, repo, GPU -------------------------------------------------
import os, sys
from pathlib import Path

REPO   = Path("/content/sam-dedection")
BRANCH = "{{BRANCH}}"

if not REPO.exists():
    !git clone -q -b {BRANCH} https://github.com/yigitkayabagci/sam-dedection.git {REPO}
!git -C {REPO} fetch -q origin {BRANCH}
!git -C {REPO} checkout -q {BRANCH} && git -C {REPO} merge -q --ff-only origin/{BRANCH}
os.chdir(REPO)
sys.path.insert(0, str(REPO))

# Drop this repo's modules so the fast-forward above actually takes effect.
# `git pull` changes files on disk; it cannot change a module Python already
# imported, so re-running a later cell after an update quietly keeps running
# the old code -- which looks exactly like the fix not working.
for _stale in [n for n in list(sys.modules) if n.split(".")[0] in ("src", "tools")]:
    del sys.modules[_stale]

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
!df -h /content | tail -1

# --- Is this notebook the one the repo expects? -------------------------
import json as _json
NOTEBOOK = "{{NOTEBOOK}}"
STAMP    = "{{STAMP}}"
_stamps  = REPO / "notebooks" / ".stamps.json"
_want    = _json.loads(_stamps.read_text()).get(NOTEBOOK) if _stamps.is_file() else None
if _want is None:
    # The dangerous case, and it used to print "matches the repo". A notebook
    # the branch has never heard of is almost always one uploaded by hand while
    # the code it drives is still unpushed -- so `src/` and `tools/` here are
    # older than this file expects, and the failure surfaces much later as a
    # tool rejecting an argument.
    print("\n" + "=" * 74)
    print(f"!!  {NOTEBOOK} is NOT in this branch's notebooks/.stamps.json.")
    print( "!!  This runtime's checkout predates the notebook, so the code it")
    print( "!!  calls may be older than the code it was written against.")
    print(f"!!  Push the branch, then re-run this cell.")
    print("=" * 74 + "\n")
elif _want != STAMP:
    print("\n" + "=" * 74)
    print(f"!!  STALE NOTEBOOK -- this file is build {STAMP}, the repo ships {_want}")
    print( "!!  Every cell below is the old version. Re-download and re-upload:")
    print(f"!!    https://github.com/yigitkayabagci/sam-dedection/raw/{BRANCH}/notebooks/{NOTEBOOK}")
    print( "!!  (In Colab: File > Upload notebook, or Runtime > Disconnect first.)")
    print("=" * 74 + "\n")
else:
    print(f"notebook build {STAMP} matches the repo")
""")

code(r"""
# --- Does this torch actually have kernels for this GPU? ----------------
# Asked by launching one, not by reading an arch list. On a card newer than
# the torch build everything imports, the model builds, and the first real
# matmul dies with "no kernel image is available" forty minutes in.
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
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("!! no CUDA device -- nothing below will run")
""")

code(r"""
# --- Dependencies -------------------------------------------------------
# EdgeTAM installs itself *as* the package `sam2` -- it is a fork -- so Meta's
# sam2 must never share this environment.
before = torch.__version__
!bash scripts/setup_edgetam.sh 2>&1 | tail -5
!pip install -q -r requirements.txt
!pip install -q "transformers>=4.56" datasets

# The one thing an install here must not do. Read the version off *disk*: pip
# cannot change the torch already live in this kernel, only a restart can, and
# that restart is exactly the window this check exists to catch.
import importlib.metadata as _md
installed = _md.version("torch")
assert installed == before, (
    f"pip replaced torch {before} with {installed} on disk.\n"
    f"This kernel is still running {before}, so nothing has broken yet -- but "
    f"restarting would load a build that may have no kernels for this GPU.\n"
    f"Restore the Colab build *before* restarting:\n"
    f"    !pip install -q --force-reinstall torch=={before}")

# `pip install -e` writes a .pth that site.py reads at interpreter *startup*
# only, so a package installed into a running kernel never joins sys.path.
# Pointing sys.path at the checkout fixes it without costing the GPU session.
import importlib
EDGETAM = REPO / "third_party" / "EdgeTAM"
if str(EDGETAM) not in sys.path:
    sys.path.insert(sys.path.index(str(REPO)) + 1 if str(REPO) in sys.path else 0,
                    str(EDGETAM))
importlib.invalidate_caches()

import sam2, transformers
print(f"sam2 (EdgeTAM) {Path(sam2.__file__).parent}\ntransformers {transformers.__version__}")

_from = Path(sam2.__file__).resolve().parent.parent
assert _from == EDGETAM.resolve(), (
    f"`sam2` imported from {_from}, not the EdgeTAM checkout at {EDGETAM}.\n"
    f"Meta's sam2 is probably installed alongside it: `pip uninstall -y sam2 "
    f"SAM-2` and re-run this cell.")
assert (EDGETAM / "checkpoints" / "edgetam.pt").is_file(), \
    "edgetam.pt did not download -- rerun scripts/setup_edgetam.sh and read its output"
""")

code(r"""
# The contracts everything below depends on, tested with no GPU and no
# dataset. `test_image_loop` covers the prompt axis and `test_aerial` the
# window one, so a failure here means the matrix would be measuring the
# harness rather than the checkpoints.
!python -m unittest tests.test_aerial tests.test_image_loop tests.test_distill \
    tests.test_training_losses tests.test_clip_loop tests.test_schedule \
    tests.test_loader tests.test_lora 2>&1 | tail -3
""")

md(r"""
## Settings

`MIRROR` points at **08's Drive folder**, and that is the whole dependency:
the four checkpoints and the cached instance index both live there. Nothing is
written into it that would disturb a re-run of 08 — the probe's own outputs go
to `PROBE`, a sibling folder.

`SEED` must stay 0. `split_index` derives the held-out flights from it, so a
different seed scores a different five sequences and the numbers stop being
comparable to 08's.
""")

code(r"""
# --- Where things live --------------------------------------------------
from google.colab import drive
drive.mount("/content/drive")

DATA_ROOT = "/content/data"          # datasets -- local NVMe, never Drive
FETCH = [
    ("vtuav_vis", f"{DATA_ROOT}/VTUAV_VIS",
     ["train_001", "train_002", "train_003"]),
]
DATASETS = [
    f"vtuav_vis:{DATA_ROOT}/VTUAV_VIS:thermal:labels:all",
]

# 08's folder: read the checkpoints and the index from here, write nothing.
MIRROR = Path("/content/drive/MyDrive/edgetam-encoder/vtuav")
INDEX  = MIRROR / "index"
# This notebook's own outputs, kept apart so re-running 08 is still safe.
PROBE  = Path("/content/drive/MyDrive/edgetam-encoder/probe")

DATA_DIR = Path(DATA_ROOT)
WORK     = Path("/content/work")     # logs and scratch, this runtime only
CKPT     = REPO / "checkpoints"

# --- The two axes -------------------------------------------------------
SIZE           = 512                 # model input, unchanged from 08
PROMPT_MODES   = ["box", "jitter", "point"]
PROMPT_JITTER  = 0.3                 # each edge, as a fraction of its own side
# 512 is 08's setting: a 512 crop at 512, no resampling. 1024 halves every
# target. VTUAV is 1080 tall, so a square crop cannot go much past this --
# beyond the short edge the fallback squashes the whole frame anisotropically,
# which is a different experiment.
WINDOWS        = [SIZE, 2 * SIZE]

# --- Held fixed, and every one of these matches 08 ----------------------
MIN_AREA, MIN_SIDE = 48, 4
MAX_AREA, FILL     = 0.9, 0.25
PER_IMAGE          = 2
MAX_INSTANCES      = 8
SEED               = 0               # decides the held-out flights -- do not move
BATCH              = 32              # eval only; no gradients, so this is generous

# --- The anchor run (cell {{anchor_run}}) -------------------------------
STEPS_PER_EPOCH = 400
EPOCHS          = (1, 3)
VAL_BATCHES     = 24
JITTER          = 32                 # window placement, unrelated to PROMPT_JITTER
LORA_R          = 16
LOADER_WORKERS  = min(2 * (os.cpu_count() or 4), 24)
PREFETCH_DEPTH  = 2

for d in (DATA_DIR, WORK, CKPT, PROBE):
    d.mkdir(parents=True, exist_ok=True)

def mirror_now(*paths):
    import shutil
    for path in paths:
        path = Path(path)
        if path.is_file():
            shutil.copy(path, PROBE / path.name)
            print(f"   saved -> {PROBE / path.name}")
        else:
            print(f"   !! {path} was not written -- the run above did not finish")

VRAM = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
print(f"{VRAM:.0f} GiB of VRAM, {os.cpu_count()} cores -> {LOADER_WORKERS} loader threads")
print(f"reading 08's checkpoints from {MIRROR}")
print(f"writing this notebook's results to {PROBE}")
print(f"matrix: {len(PROMPT_MODES)} prompts x {len(WINDOWS)} windows "
      f"= {len(PROMPT_MODES) * len(WINDOWS)} conditions per checkpoint")
""")

md(r"""
## What has to be on Drive already

Two things, both left by `08_encoder_vtuav_only.ipynb`:

* **the four checkpoints** — `edgetam_aerial_512.pt`, `edgetam_aerial_lora_512.pt`,
  `edgetam_aerial_distilled_512.pt`, and `edgetam_distilled_512.pt` (stage A
  alone, the starting point cell {{anchor_run}} needs);
* **`index/`** — the instance index. Reusing it is not only a time saving: an
  index rebuilt here could differ, and then "the same held-out flights" would
  be a claim rather than a fact.

`stock` needs nothing from Drive; it is EdgeTAM's own released checkpoint,
downloaded by `setup_edgetam.sh` above.
""")

code(r"""
# --- Download the frames, once ------------------------------------------
# The checkpoints are on Drive but the pixels are not: scoring reads the actual
# images, so VTUAV is extracted here exactly as 08 did it. This is the slow
# cell in the notebook -- everything after it is minutes, not tens of minutes.
import shutil

from src.training.datasets import describe, parse
from tools.fetch_datasets import fetch, human, report

requests = [parse(d) for d in DATASETS]
print(describe(requests), "\n")
print(f"{human(shutil.disk_usage('/content').free)} of disk free\n")

for name, dest, parts in FETCH:
    if Path(dest).exists() and any(Path(dest).rglob("*.jpg")):
        print(f"== {name}: already extracted to {dest}")
    else:
        fetch(name, dest, tuple(parts) or None)
    print(report(name, dest, "thermal"))
""")

code(r"""
# --- Is 08's output actually there? -------------------------------------
# Checked here, by name, rather than discovered as a 40-minute-old traceback
# inside the matrix loop.
STOCK = "third_party/EdgeTAM/checkpoints/edgetam.pt"
CHECKPOINTS = {
    "stock":        STOCK,
    "finetune":     f"{MIRROR}/edgetam_aerial_512.pt",
    "lora":         f"{MIRROR}/edgetam_aerial_lora_512.pt",
    "distilled+ft": f"{MIRROR}/edgetam_aerial_distilled_512.pt",
}
DISTILLED = MIRROR / "edgetam_distilled_512.pt"   # stage A alone, for cell {{anchor_run}}

missing = [f"{label}: {path}" for label, path in CHECKPOINTS.items()
           if not Path(path).is_file()]
if missing:
    raise SystemExit(
        "these checkpoints are not on Drive:\n  " + "\n  ".join(missing) +
        f"\n\n{MIRROR} is written by 08_encoder_vtuav_only.ipynb. Run that "
        f"notebook to completion first -- this one only re-scores what it left.")

index_files = sorted(INDEX.glob("*.json")) if INDEX.is_dir() else []
if not index_files:
    raise SystemExit(
        f"no instance index at {INDEX}. It is written by 08 and reusing it is "
        f"what makes the held-out flights here the same five it held out. "
        f"Run 08 first.")

for label, path in CHECKPOINTS.items():
    _st = Path(path).stat()
    _when = __import__("time").strftime("%Y-%m-%d %H:%M", __import__("time").localtime(_st.st_mtime))
    print(f"  {label:<14} {_st.st_size / 2**20:>6.1f} MiB  {_when}  {path}")
if DISTILLED.is_file():
    print(f"  {'stage A only':<14} "
          f"{DISTILLED.stat().st_size / 2**20:>6.1f} MiB  {DISTILLED}")
else:
    print(f"  !! {DISTILLED} is absent -- cell {{anchor_run}} will skip")
print(f"\nindex: {', '.join(p.name for p in index_files)}")

# --- Copy them off Drive before scoring anything ------------------------
# Two reasons, and the first one is correctness. 08 writes these files, and it
# writes each one *the moment that run finishes* rather than at the end -- so
# re-running 08 on a second runtime while this notebook is working replaces
# them underneath it, and the matrix silently becomes a comparison between
# checkpoints from two different experiments. Reading a snapshot means the two
# notebooks can run at the same time and this one still knows what it scored.
#
# The second is speed: every one of the 24 evals loads its checkpoint fresh,
# and 56 MB over the Drive FUSE mount is not the same as 56 MB off local NVMe.
import shutil

SNAPSHOT = Path("/content/snapshot")
SNAPSHOT.mkdir(parents=True, exist_ok=True)
for label, path in list(CHECKPOINTS.items()):
    if label == "stock":
        continue                       # already local, in the EdgeTAM checkout
    local = SNAPSHOT / Path(path).name
    if not local.is_file():
        shutil.copy(path, local)
    CHECKPOINTS[label] = str(local)
if DISTILLED.is_file():
    local = SNAPSHOT / DISTILLED.name
    if not local.is_file():
        shutil.copy(DISTILLED, local)
    DISTILLED = local

# The index is small and 08 writes it once, before any training, so it is not
# exposed to the same race -- but copying it costs nothing and removes the
# last read of a live folder.
LOCAL_INDEX = SNAPSHOT / "index"
LOCAL_INDEX.mkdir(exist_ok=True)
for path in index_files:
    if not (LOCAL_INDEX / path.name).is_file():
        shutil.copy(path, LOCAL_INDEX / path.name)
INDEX = LOCAL_INDEX

print(f"\nsnapshot taken -> {SNAPSHOT}")
print("08 can now re-run on another runtime without disturbing this one.")
""", tag="inventory")

md(r"""
## The matrix

Four checkpoints × three prompts × two windows = 24 scored runs, each over the
same 420 held-out instances, about 25 seconds apiece.

**Every axis except the one being varied is pinned to 08's value** — the same
gates, the same `--seed`, the same cached index, so the same five flights. The
jitter generator is seeded from `--seed` too, which matters more than it
looks: it means every checkpoint in a `jitter` column is handed the *same*
perturbed boxes, and the difference between two rows is still the checkpoint
rather than the draw.

The loop skips any condition whose JSON already exists, so a disconnected
runtime resumes where it stopped instead of starting over.
""")

code(r"""
%%time
# --- Score everything ---------------------------------------------------
# Preflight, and it earns its two seconds. This notebook drives
# `eval_instances.py` through `--window` and `--prompt`, which are newer than
# the notebook is old: a runtime whose checkout predates them gets an argparse
# "unrecognized arguments" and a exit code nobody reads, because the `| tail`
# below keeps the output short. Twenty-four of those in a row look like a slow
# cell rather than a broken one, and the first thing that actually complains is
# the table cell, with a message about empty results that points nowhere near
# the cause.
import subprocess

_help = subprocess.run([sys.executable, "tools/eval_instances.py", "--help"],
                       capture_output=True, text=True).stdout
_absent = [f for f in ("--window", "--prompt") if f not in _help]
if _absent:
    raise SystemExit(
        f"tools/eval_instances.py in this runtime does not accept "
        f"{', '.join(_absent)}.\n\n"
        f"The checkout is behind the notebook. Re-run cell 1 to fast-forward "
        f"the repo; if the flags are still missing afterwards, the branch on "
        f"GitHub does not carry them yet and nothing below can run.")

DATASET_FLAGS = " ".join(f"--dataset {d}" for d in DATASETS)
GATES = (f"--min-area {MIN_AREA} --min-side {MIN_SIDE} --max-area {MAX_AREA} "
         f"--fill {FILL} --per-image {PER_IMAGE} --max-instances {MAX_INSTANCES}")

# A filename for one cell of the matrix, safe to hand to a shell.
# `!python ... --json {out}` goes through one, so a label carrying a space --
# `distilled+ft (anchor 0)`, added further down -- would split into two
# arguments and the eval would write somewhere nobody looks. Everything that
# is not alphanumeric becomes an underscore.
def tag_for(label, prompt, window):
    safe = "".join(c if c.isalnum() else "_" for c in label).strip("_")
    return f"{safe}__{prompt}__{window}"

todo = [(label, prompt, window)
        for label in CHECKPOINTS
        for window in WINDOWS
        for prompt in PROMPT_MODES]
done = 0
for label, prompt, window in todo:
    tag = tag_for(label, prompt, window)
    out = WORK / f"probe_{tag}.json"
    if out.is_file():
        print(f"-- {tag}: already scored")
        done += 1
        continue
    print(f"\n===== {label}  prompt={prompt}  window={window} "
          f"({done + 1}/{len(todo)}) =====")
    weights = CHECKPOINTS[label]
    !python tools/eval_instances.py {DATASET_FLAGS} --index {INDEX} \
        --checkpoint {weights} --split test --size {SIZE} \
        --window {window} --prompt {prompt} --prompt-jitter {PROMPT_JITTER} \
        {GATES} --batch {BATCH} --seed {SEED} \
        --json {out} 2>&1 | tail -6
    # Stop on the first failure rather than grinding through the other 23. The
    # eval's own error is in the six lines printed just above.
    if not out.is_file():
        raise SystemExit(
            f"{tag}: eval_instances.py exited without writing {out}.\n"
            f"Its error is in the lines above this message.")
    done += 1

print(f"\n{done}/{len(todo)} conditions scored -> {WORK}")
""")

code(r"""
# --- Read it back into one table ----------------------------------------
import json

import numpy as np

scores = {}
for label in CHECKPOINTS:
    for window in WINDOWS:
        for prompt in PROMPT_MODES:
            path = WORK / f"probe_{tag_for(label, prompt, window)}.json"
            if path.is_file():
                scores[(label, prompt, window)] = json.loads(path.read_text())

def cell(label, prompt, window, key="mean_iou"):
    row = scores.get((label, prompt, window))
    return float("nan") if row is None else row[key]

def gain(label, prompt, window, key="mean_iou"):
    return cell(label, prompt, window, key) - cell("stock", prompt, window, key)

if not scores:
    raise SystemExit("no results on disk -- the cell above scored nothing")
first = next(iter(scores.values()))
print(f"test split, {first['instances']} instances, seen by no run\n")

for window in WINDOWS:
    scale = window / SIZE
    print(f"===== window {window} source px -> {SIZE} model px"
          f"{'  (targets at native size)' if scale == 1 else f'  (targets {scale:.0f}x smaller)'} =====")
    small = int(cell("stock", "box", window, "small_instances"))
    print(f"      {small} of {first['instances']} instances fall under 32 px here\n")
    header = "".join(f"{p:>22}" for p in PROMPT_MODES)
    print(f"{'mean IoU':<14}{header}")
    for label in CHECKPOINTS:
        cells = "".join(
            f"{cell(label, p, window):>13.4f}"
            + (f"{'':>9}" if label == "stock" else f"{gain(label, p, window):>+9.4f}")
            for p in PROMPT_MODES)
        print(f"{label:<14}{cells}")
    print()

print("Each pair is  absolute  then  gain over stock in the same column.")
print("Reading across a row is reading how much of the score was the prompt.")
""", tag="table")

code(r"""
# --- The same thing, drawn -----------------------------------------------
# The number that matters is not any single cell but the *shape*: does the
# distance between stock and the trained checkpoints grow as the prompt gets
# weaker? A table makes that a subtraction; a plot makes it obvious.
import matplotlib.pyplot as plt

runs_shown = [k for k in CHECKPOINTS if k != "stock"]
palette = {"finetune": "#4c72b0", "lora": "#dd8452", "distilled+ft": "#55a868"}
fig, axes = plt.subplots(1, len(WINDOWS), figsize=(6.2 * len(WINDOWS), 4.2),
                         sharey=True, squeeze=False)

x = np.arange(len(PROMPT_MODES))
for column, window in enumerate(WINDOWS):
    ax = axes[0][column]
    for label in runs_shown:
        ax.plot(x, [gain(label, p, window) for p in PROMPT_MODES],
                marker="o", label=label, color=palette.get(label))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n(±{PROMPT_JITTER:g})" if p == "jitter" else p
                        for p in PROMPT_MODES])
    scale = window / SIZE
    ax.set_title(f"window {window} px"
                 + ("  — native" if scale == 1 else f"  — targets {scale:.0f}x smaller"))
    ax.set_xlabel("prompt, weakening to the right")
    ax.grid(alpha=0.3)
axes[0][0].set_ylabel("mean IoU − stock")
axes[0][0].legend(loc="best")
plt.tight_layout(); plt.show()

# Lines sloping *up* to the right is the finding: a gap that the box was
# hiding. Flat lines near zero say the encoder work really did buy little.
print("Up and to the right = the box was hiding a real gain.")
print("Flat and near zero  = the gain is genuinely small on this data.")
""")

code(r"""
# --- Did the small targets move? ----------------------------------------
# The column 08 could not fill. At window 1024 the lower tail of VTUAV's size
# distribution crosses under 32 px, so this is finally a number rather than a
# `nan` -- but it is a *tail*, a few percent of instances, not the deployment's
# regime. Read it as the first evidence, not as the answer.
wide = WINDOWS[-1]
base = scores.get(("stock", "box", wide))
print(f"window {wide} px, targets under 32 px in the window\n")
print(f"{'':<14}{'small n':>9}{'small IoU':>12}{'vs stock':>11}{'larger':>10}")
for label in CHECKPOINTS:
    row = scores.get((label, "box", wide))
    if row is None or base is None:
        continue
    delta = "" if label == "stock" else \
        f"{row['small_mean_iou'] - base['small_mean_iou']:>+11.4f}"
    print(f"{label:<14}{row['small_instances']:>9}"
          f"{row['small_mean_iou']:>12.4f}{delta:>11}"
          f"{row['large_mean_iou']:>10.4f}")

if not np.isfinite(cell("stock", "box", wide, "small_mean_iou")):
    print("\nStill empty: even halved, nothing in this split lands under 32 px.")
    print("That is a fact about VTUAV's mask annotation, not about the models --")
    print("it tracks one large object per flight. The small-target question")
    print("needs a different dataset; 07's SegFly and Kust4K are where it lives.")
""")

md(r"""
## Stage A, or the anchor?

08 ran stage A and then stage B on top of it, with `ANCHOR_WEIGHT = 0.5`
holding the encoder near where stage A left it. The result came in behind plain
`finetune`, and the anchor term sat at 0.01 for all three encoder epochs — the
encoder never drifted. What that cannot say is whether it *would* have drifted
somewhere better if allowed.

The run below is byte for byte 08's distilled run with `--anchor-weight 0`. It
starts from the same stage-A checkpoint and is free to leave it.

| it lands | reading |
|---|---|
| level with or above `finetune` | the anchor was the problem, not stage A. Stage A is worth keeping; the regulariser was too strong at 0.5 — try 0.1 |
| still behind `finetune` | stage A itself moved the encoder somewhere worse for this task. With this teacher on this data it does not pay, and the SAM→SAM redundancy is the first suspect — EdgeTAM is already a distillation of SAM 2 |

Either way the run costs about 20 minutes, and it is the only training this
notebook does.
""")

code(r"""
%%time
# --- Stage B from stage A, with nothing holding it back -----------------
ANCHOR_FREE = CKPT / "edgetam_aerial_anchor0_512.pt"
COMMON = (f"{DATASET_FLAGS} --index {INDEX} "
          f"--size {SIZE} {GATES} "
          f"--jitter {JITTER} --batch {BATCH} --accum 1 "
          f"--steps {STEPS_PER_EPOCH} --epochs {EPOCHS[0]} {EPOCHS[1]} "
          f"--val-batches {VAL_BATCHES} --workers {LOADER_WORKERS} "
          f"--depth {PREFETCH_DEPTH} --seed {SEED}")

if DISTILLED.is_file():
    !python tools/train_encoder.py --method finetune {COMMON} \
        --base {DISTILLED} --anchor-weight 0 \
        --out {ANCHOR_FREE} --json {WORK}/run_anchor0.json 2>&1 | tail -22
    mirror_now(ANCHOR_FREE, WORK / "run_anchor0.json")
else:
    print(f"skipped -- {DISTILLED} is not on Drive, so there is no stage-A")
    print("checkpoint to start from. Re-run 08 with PRETRAIN = True.")
""", tag="anchor_run")

code(r"""
# --- Score it through the same matrix ------------------------------------
if ANCHOR_FREE.is_file():
    CHECKPOINTS["distilled+ft (anchor 0)"] = str(ANCHOR_FREE)
    for window in WINDOWS:
        for prompt in PROMPT_MODES:
            tag = tag_for("distilled+ft (anchor 0)", prompt, window)
            out = WORK / f"probe_{tag}.json"
            if out.is_file():
                continue
            print(f"\n===== anchor 0  prompt={prompt}  window={window} =====")
            !python tools/eval_instances.py {DATASET_FLAGS} --index {INDEX} \
                --checkpoint {ANCHOR_FREE} --split test --size {SIZE} \
                --window {window} --prompt {prompt} \
                --prompt-jitter {PROMPT_JITTER} \
                {GATES} --batch {BATCH} --seed {SEED} \
                --json {out} 2>&1 | tail -4
            scores[("distilled+ft (anchor 0)", prompt, window)] = \
                json.loads(out.read_text())

    print(f"\n{'':<26}{'box':>10}{'jitter':>10}{'point':>10}   (window {SIZE})")
    for label in ("finetune", "distilled+ft", "distilled+ft (anchor 0)"):
        row = "".join(f"{cell(label, p, SIZE):>10.4f}" for p in PROMPT_MODES)
        print(f"{label:<26}{row}")
else:
    print("no anchor-0 checkpoint -- the cell above did not run")
""")

code(r"""
# --- In plain words: what this notebook decided --------------------------
line = "-" * 74
print(line)
print("DID THE PROMPT HIDE A GAIN?")
for label in [k for k in CHECKPOINTS if k != "stock"]:
    box_gain   = gain(label, "box", SIZE)
    point_gain = gain(label, "point", SIZE)
    verdict = ("the box was hiding it" if point_gain > box_gain + 0.01 else
               "no -- the gap is the same either way"
               if abs(point_gain - box_gain) <= 0.01 else
               "the opposite: it shrinks without the box")
    print(f"  {label:<24} box {box_gain:+.4f} -> point {point_gain:+.4f}   {verdict}")

print(line)
print("DID SHRINKING THE TARGETS CHANGE THE RANKING?")
wide = WINDOWS[-1]
for label in [k for k in CHECKPOINTS if k != "stock"]:
    print(f"  {label:<24} window {SIZE}: {gain(label, 'box', SIZE):+.4f}"
          f"   window {wide}: {gain(label, 'box', wide):+.4f}")

print(line)
print("STAGE A, OR THE ANCHOR?")
if "distilled+ft (anchor 0)" in CHECKPOINTS:
    free = cell("distilled+ft (anchor 0)", "box", SIZE)
    held = cell("distilled+ft", "box", SIZE)
    plain = cell("finetune", "box", SIZE)
    print(f"  anchor 0.5 {held:.4f}   anchor 0 {free:.4f}   plain finetune {plain:.4f}")
    print("  -> the anchor was holding it back" if free > held + 0.005 else
          "  -> stage A itself is what costs; the anchor was not the problem")
else:
    print("  not run")

print(line)
print("WHAT THIS STILL DOES NOT MEASURE")
print("  Tracking. Every number above scores one prompted frame, so none of")
print("  them can see the failure this project exists to fix -- that needs a")
print("  memory bank, and the memory bank is stage C.")
print(line)

verdict = {"axes": {"prompts": PROMPT_MODES, "windows": WINDOWS,
                    "prompt_jitter": PROMPT_JITTER},
           "checkpoints": CHECKPOINTS, "seed": SEED,
           "scores": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in scores.items()}}
(WORK / "probe_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
mirror_now(WORK / "probe_verdict.json")
""")

md(r"""
## How to read what came out

**The prompt axis is the one that decides whether stage B continues.** If
`point` opens a gap that `box` did not show, the encoder training was working
the whole time and 08's table was an artefact of scoring under the easiest
possible prompt — in which case the right next move is to keep training the
encoder and to switch the reported metric to a weak prompt permanently. If the
gap is flat across all three, then the encoder genuinely learned little from
4 057 frames of one campus, and no amount of schedule tuning will change that;
the fix is data, and `07_encoder_aerial_rgbt.ipynb` is where the mixed sets are.

**The window axis is weaker evidence than it looks.** Halving target size is
the most this sensor allows before the crop stops being square, and half of
186 pixels is 93 — nowhere near the twenty-pixel regime the deployment cares
about. Treat a change here as a direction to follow, not as a measurement of
the small-target case. That measurement needs a dataset whose targets are
small to begin with.

**The anchor result is the only clean single-number answer in the notebook,**
because it is a controlled comparison: same start, same data, same schedule,
one flag.

### If everything came out flat

That is a real result and worth taking at face value rather than tuning
against. Ranked by what it would buy:

1. **More stage-A data.** 08 used 20 000 of the 121 401 registered pairs
   VTUAV ships, for 600 steps at batch 32 — about 19 200 samples, less than a
   single epoch over even the subsample. `DISTILL_PAIRS` and `DISTILL_STEPS`
   in 08 are the two numbers to raise, and there is 6× more data on disk
   already.
2. **A dataset with small targets**, which is 07's three-set mix. The
   deployment's regime cannot be reached from VTUAV's mask split at any
   window size.
3. **Stage C.** Nothing here measures tracking, and tracking is the metric
   this project is judged on. A better encoder is a precondition for a better
   tracker, never evidence of one — and if the encoder is not the bottleneck,
   the memory path is where the remaining error lives.

### What it would take to trust a gap under 0.01

Five held-out flights, and VTUAV annotates every thirtieth frame — 420
instances is not 420 independent samples, it is closer to five. Re-run the
matrix at `SEED = 1` and `SEED = 2`; each draws a different five flights, and
the spread across the three is the honest error bar. A difference smaller than
that spread is not a difference.
""")


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def stamp() -> str:
    """The build id: this notebook's cells, hashed before substitution."""
    body = "\n".join(f"{kind}\n{text}" for kind, text in CELLS)
    return hashlib.sha256(body.encode()).hexdigest()[:10]


def build() -> dict:
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
    STAMP_VALUE = stamp()
    out = repo / "notebooks" / NOTEBOOK
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")

    stamps = repo / "notebooks" / ".stamps.json"
    known = json.loads(stamps.read_text()) if stamps.is_file() else {}
    known[NOTEBOOK] = STAMP_VALUE
    stamps.write_text(json.dumps(known, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out.relative_to(repo)} with {len(CELLS)} cells [{STAMP_VALUE}]")
    sys.exit(0)
