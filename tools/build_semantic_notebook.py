#!/usr/bin/env python3
"""Build `notebooks/15_semantic_mask_pool.ipynb` from plain text blocks.

    python tools/build_semantic_notebook.py

One notebook, and unlike 13/14 it has no variants: there is one question here
and both modalities ask it the same way. The question is whether a **click**
can separate what a semantic map fused, and the notebook's first real output
is the answer measured against drawn instances rather than asserted.

Generated rather than edited for the reason 07-14 are: a notebook is a JSON
document where prose, code and cell numbers drift apart silently. `{{tag}}`
resolves cross-references at build time, `tools/check_notebook.py` walks the
result for names used before they are bound, and `tests/test_notebooks.py`
holds the shipped file to the stamp this build writes into
`notebooks/.stamps.json`.
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
NOTEBOOK = "15_semantic_mask_pool.ipynb"


def md(text: str, tag: str | None = None) -> None:
    _add("markdown", text, tag)


def code(text: str, tag: str | None = None) -> None:
    _add("code", text, tag)


def _add(kind: str, text: str, tag: str | None) -> None:
    if tag:
        TAGS[tag] = len(CELLS)
    CELLS.append((kind, text.strip("\n")))


def resolve(text: str) -> str:
    substitutions = {"NOTEBOOK": NOTEBOOK, "STAMP": STAMP_VALUE,
                     **{t: str(i) for t, i in TAGS.items()}}
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
# Semantic rescue — turning label maps into *instances*, and proving it worked

One runtime, one **Run all**. This notebook takes the semantic aerial
datasets this project already downloads — Kust4K, SegFly, Caltech — and
produces **per-instance** masks from them: `(image, prompt, mask)` in the same
store stage B already reads.

| | |
|---|---|
| **in** | nothing staged by hand — the download cell fetches everything |
| **out** | per-frame run-length instance stores + a before/after table, zipped to `edgetam-pool/semantic` on your Drive |
| **needs** | a CUDA GPU and ~8 GB of free disk (Kust4K 2.7 GB, iSAID annotations 0.2 GB, the rest optional) |
| **takes** | ~10 min to download, ~20 min for the measurement, ~1–2 h per dataset harvest (resumable) |

**This notebook trains nothing**, and it is not a third labeller competing
with 13 and 14 — it is the one that needs no boxes at all:

```
13   detection boxes  ->  teacher masks     (aerial RGB)
14   detection boxes  ->  teacher masks     (thermal, route measured first)
15   semantic map     ->  instances         (no boxes anywhere)
```

---

## The problem, stated as a number

`aerial.decompose` turns a semantic map into instances by taking connected
components per thing class. That is the only thing a label map can do alone,
and it has one failure that is not a detail:

| set | measured |
|---|---|
| **SegFly** | one-car-one-blob recovery **78.5 %**; **20.9 %** of vehicle pixels sit in a component holding several vehicles |
| **iSAID** (drawn instances, same test) | 3 523 drawn vehicles → 2 889 components (**82.0 %**); **26.9 %** of instances share a blob |

Two cars parked flush *touch*. Their boundary is not a class boundary, so it
is not in the map, so they are one component, so stage B learns to answer a
prompt on one car with a mask covering both. No threshold fixes it:
`InstanceGates.fill` catches two objects joined by a thin bridge, and a real
six-vehicle blob measured **fill 0.79** — straight through every gate.

**The map cannot separate them because it never knew there were two. The image
can.**

## The idea, in one line

A **box** around a row of parked cars contains the row, and the teacher
answers with the row. A **click** on one bonnet answers with one car.

```
  semantic map              per component:                 teacher          gates
  ┌───────────┐        ┌────────────────────┐        ┌──────────────┐  ┌───────────────┐
  │ class per │        │ decompose -> region│        │ SAM, prompted│  │ purity  (map!)│
  │  pixel    │ ─────▶ │ estimate how many  │ ─────▶ │ with ONE     │─▶│ containment   │─▶ RLE store
  │ (drawn by │        │  objects it holds  │        │ interior     │  │ unit_area     │   per frame
  │  a human) │        │ seed that many pts │        │ click each   │  │ component     │  + record.json
  └───────────┘        └────────────────────┘        └──────────────┘  └───────────────┘
```

The roles are the inverse of 13/14's:

| | prompt comes from | verified by |
|---|---|---|
| 13 / 14 | a human's **box** | geometry (`box_iou`) |
| **15** | the map's **component** | the human's **map**, pixel by pixel (`purity`) |

That second column is what this route has and a box pool cannot: the label map
is *drawn*, so it can say whether the mask that came back is even the right
class. A mask that swallowed the road under the car is one clean blob in the
right place — nothing geometric sees the problem, and the drawing does.

## What is measured before anything is spent

`src/training/semantic.py` asserts nothing about being better. Cell
{{measure}} decides it, on data that knows the answer:

1. take a dataset that ships **drawn instances** (iSAID; VTUAV's VIS split),
2. **flatten** them into the semantic map they imply — throwing the
   separation away, which is exactly the input production sees,
3. run **both** routes over that map: plain `decompose`, and the rescue,
4. match each one-to-one against the instances it was flattened from.

Neither route ever sees the separation; it exists only in the truth they are
scored against. If `rescue` does not beat `components` on recall there, it
will not beat it on Kust4K where nobody can check — and the honest move is to
close this notebook and spend the GPU-hours on 14.
""")

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

# Drop this repo's modules so the fast-forward above actually takes effect:
# `git pull` changes files on disk, not modules Python already imported.
for _stale in [n for n in list(sys.modules) if n.split(".")[0] in ("src", "tools")]:
    del sys.modules[_stale]

!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
!df -h /content | tail -1

# --- Is this notebook the one the repo expects? -------------------------
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

code(r"""
# --- Does this torch actually have kernels for this GPU? ----------------
import torch

print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    print(f"{torch.cuda.get_device_name(0)}  sm_{major}{minor}  "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
    try:
        probe = torch.randn(256, 256, device="cuda")
        (probe @ probe).sum().item()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            (probe @ probe).sum().item()
        print("a real matmul ran, in float32 and bfloat16 -- this GPU is usable")
    except RuntimeError as exc:
        raise SystemExit(
            f"torch {torch.__version__} cannot run on sm_{major}{minor}: {exc}\n"
            f"In Colab, restore the preinstalled torch rather than letting "
            f"anything downgrade it, then Runtime > Restart session.")
else:
    print("!! no CUDA device -- the teacher will not run")
""")

code(r"""
# --- Dependencies -------------------------------------------------------
# No EdgeTAM: this notebook trains nothing. transformers>=5 has the SAM 3
# classes; SAM 2.1 needs only 4.56, so one install covers both choices.
before = torch.__version__
!pip install -q "transformers>=5" hf_transfer tqdm

import importlib.metadata as _md
installed = _md.version("torch")
assert installed == before, (
    f"pip replaced torch {before} with {installed} on disk. Restore it "
    f"before restarting:  !pip install -q --force-reinstall torch=={before}")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
""")

code(r"""
# --- Where things live, and every knob ----------------------------------
DATA_ROOT = Path("/content/data")      # datasets; local disk, never Drive
WORK      = Path("/content/work")      # scratch
POOL      = WORK / "semantic"          # the product: stores + records

# The teacher. facebook/sam3 is gated -- accept its terms once, then log in
# below. The ungated fallback needs no account:
#     TEACHER = "facebook/sam2.1-hiera-large"
TEACHER = "facebook/sam3"
DEVICE  = "cuda"
DTYPE   = "bfloat16"

from src.training.semantic import SemanticGates
GATES = SemanticGates()                # purity .65, containment .70,
                                       # unit_area (0.30, 1.75), component .80

ZOOM      = 4.0                        # crop side, in component long-sides
MIN_SIZE  = 128                        # crop floor, px
BATCH     = 8                          # clicks per teacher forward
SEED_CAP  = 12                         # most seeds one component may get
MODE      = "components"               # decompose mode: the seeds' source
FRAME_LIMIT = None                     # frames per dataset; 200 for a smoke run
UNIT_FRAMES = 200                      # frames sampled to size one object
RESUME    = True
SEED      = 0

# The measurement (cell {{measure}}): which drawn-instance set, how many frames,
# and how much overlap counts as "the same object".
TRUTH        = "isaid"                 # or "vtuav_vis"
TRUTH_FRAMES = 30
MATCH_IOU    = 0.5

# The harvest (cell {{harvest}}): which semantic sets to rescue, and on which
# modality. Kust4K is the one to start with -- 640x512 native, urban, and the
# only aerial RGB-T set here whose things are dense enough to fuse.
HARVEST = [("kust4k", "thermal"), ("segfly", "thermal")]

for d in (DATA_ROOT, WORK, POOL):
    d.mkdir(parents=True, exist_ok=True)

def tqdm_over(stream, total, desc):
    from tqdm.auto import tqdm
    return tqdm(stream, total=total, desc=desc)

print(f"teacher {TEACHER}")
print(f"gates {GATES}")
print(f"pool -> {POOL}")
""")

code(r"""
# --- Drive, and a mirror that runs the moment something exists ----------
from google.colab import drive as _drive
_drive.mount("/content/drive")
MIRROR = Path("/content/drive/MyDrive/edgetam-pool/semantic")
MIRROR.mkdir(parents=True, exist_ok=True)

import shutil

def stage_pool(name: str) -> None:
    # Zip one dataset's stores to Drive as soon as that dataset finishes --
    # per dataset and not at the end, because a session that dies three
    # datasets in should keep three datasets. The lesson 074f0f2 recorded.
    source = POOL / name
    if not source.exists():
        print(f"  {name}: nothing to stage yet")
        return
    archive = shutil.make_archive(str(WORK / f"semantic_{name}"), "zip", source)
    shutil.copy(archive, MIRROR / Path(archive).name)
    print(f"  staged {Path(archive).name} -> {MIRROR}")

print(f"mirror -> {MIRROR}")
""")

code(r"""
# --- Hugging Face, only if the teacher is gated -------------------------
if "sam3" in TEACHER.lower():
    from huggingface_hub import login
    login()          # facebook/sam3 is gated: accept its terms once, then this
else:
    print(f"{TEACHER} is ungated -- no token needed")
""")

md(r"""
## Downloading

Two things: the **truth** set that decides whether this route is worth running
(iSAID's drawn instances — annotations only, 200 MB, ungated), and the
**semantic** sets to rescue. Everything below is fetched by
`tools/fetch_datasets.py`, which carries each link and the reason the obvious
method fails for it.
""")

code(r"""
# --- The data -----------------------------------------------------------
from tools.fetch_datasets import fetch

DESTS = {"kust4k": DATA_ROOT / "Kust4K", "segfly": DATA_ROOT / "SegFly",
         "caltech": DATA_ROOT / "Caltech"}
for name in ("kust4k",):                       # add "segfly" for the big one
    print(f"== {name}")
    fetch(name, DESTS[name])

# iSAID's *annotations* are the truth for cell {{measure}}: instance-id PNGs,
# ungated, no account, and the images are not needed -- the rescue is scored
# on separation, and separation is in the id map.
ISAID = DATA_ROOT / "iSAID"
ISAID.mkdir(parents=True, exist_ok=True)
if not any(ISAID.rglob("*_instance_id_RGB.png")):
    !curl -sSL -o /content/isaid_val.tar.gz \
      https://huggingface.co/datasets/isaaccorley/isaid/resolve/main/isaid_annotations_val.tar.gz
    !tar xzf /content/isaid_val.tar.gz -C {ISAID} && rm /content/isaid_val.tar.gz
print(f"iSAID instance maps: {len(list(ISAID.rglob('*_instance_id_RGB.png')))}")
""")

code(r"""
# --- What arrived, and does the reader agree? ---------------------------
from src.training.aerial import SPECS, decompose, describe_layout, list_frames, read_mask
from src.training.semantic import unit_areas

KUST = DATA_ROOT / "Kust4K"
frames = list_frames(KUST, SPECS["kust4k"], "thermal")
print(f"Kust4K: {len(frames)} thermal frames")
if not frames:
    print(describe_layout(KUST))
    raise SystemExit("no Kust4K frames -- read the layout above")

# The population this notebook exists to fix, counted before it is fixed:
# how many components are already too big to be one object?
pooled = []
for frame in tqdm_over(frames[:200], total=min(200, len(frames)), desc="probe"):
    _, instances, _ = decompose(read_mask(frame.mask), SPECS["kust4k"])
    pooled.extend(instances)
UNITS = unit_areas(pooled)
spec = SPECS["kust4k"]
print("\n| class | components | unit area px | over 1.75 units |")
print("|---|---:|---:|---:|")
for class_id, unit in sorted(UNITS.items()):
    rows = [i for i in pooled if int(i.class_id) == class_id]
    over = sum(1 for i in rows if i.area > 1.75 * unit)
    print(f"| {spec.name_of(class_id)} | {len(rows)} | {unit:.0f} | "
          f"{over} ({over / max(len(rows), 1):.1%}) |")
print("\nThe last column is this notebook's headroom: components already too "
      "\nlarge to be one object. Near zero means nothing here is fused and "
      "\nplain decompose is enough.")
""", tag="probe")

code(r"""
# --- The teacher --------------------------------------------------------
from src.training.labels import build_image_teacher

teacher = build_image_teacher(TEACHER, device=DEVICE, dtype=DTYPE)
print(f"loaded {teacher.model_id}")
""")

md(r"""
## Look before spending

Two panels per frame: what `decompose` calls one instance, and what the
teacher returns when clicked inside it. Colours are per instance — the whole
question is whether the right-hand panel has **more colours** than the left.
""")

code(r"""
# --- Eyes on it ---------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from src.training.pool import _read_rgb
from src.training.semantic import rescue_frame

def panels(frame, spec, units, count=2):
    pixels = _read_rgb(frame.image)
    semantic = read_mask(frame.mask)
    components, instances, _ = decompose(semantic, spec, mode=MODE)
    masks, records = rescue_frame(pixels, semantic, spec, teacher, gates=GATES,
                                  mode=MODE, zoom=ZOOM, min_size=MIN_SIZE,
                                  batch_size=BATCH, seed_cap=SEED_CAP,
                                  units=units)
    after = np.zeros(components.shape, np.int32)
    for i, mask in enumerate(masks, 1):
        after[mask] = i
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(pixels); axes[0].set_title(frame.name)
    axes[1].imshow(components, cmap="tab20", interpolation="nearest")
    axes[1].set_title(f"decompose: {len(instances)} instances")
    axes[2].imshow(after, cmap="tab20", interpolation="nearest")
    axes[2].set_title(f"rescue: {len(masks)} instances")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.show()
    return records

# The frames worth looking at are the ones with the most thing pixels.
busy = sorted(frames[:200],
              key=lambda f: -int(np.isin(read_mask(f.mask),
                                         [SPECS["kust4k"].classes[n]
                                          for n in SPECS["kust4k"].things]).sum()))
for frame in busy[:2]:
    panels(frame, SPECS["kust4k"], UNITS)
""")

md(r"""
## The measurement the whole notebook turns on

Both routes, over instances a human drew, scored against that drawing.

Read **recall** first: of the objects drawn, how many came back as a mask of
their own. That is the fusion question. Read **precision** beside it, because
a route that answers every seed with a mask can buy recall with noise, and
this is where that shows.

The bar to clear is not "better than nothing" — it is **better than
`components`**, which costs no GPU at all. If the two rows are within a point
of each other, the honest read is that this dataset had nothing fused and the
rescue is buying nothing; keep `decompose`.
""")

code(r"""
# --- Both routes against drawn truth ------------------------------------
from src.training.semantic import measure_rescue, summarise_rescue
from tools.make_semantic_pool import SAMPLERS
from src.training.aerial import DatasetSpec

# iSAID's fifteen categories are not this project's palette, and separation
# has no class in it: one synthetic thing class is all the measurement needs.
TRUTH_SPEC = (SPECS["vtuav_vis"] if TRUTH == "vtuav_vis" else DatasetSpec(
    name="isaid_truth", masks="**/*.png",
    classes={"background": 0, "object": 1}, things=("object",),
    thermal="**/*.png", rgb=None, ignore=(0,),
    palette_source="synthetic: one thing class, for this measurement only"))
TRUTH_ROOT = ISAID if TRUTH == "isaid" else DATA_ROOT / "VTUAV_VIS"

ROUTES = measure_rescue(
    SAMPLERS[TRUTH](TRUTH_ROOT, TRUTH_SPEC, TRUTH_FRAMES),
    TRUTH_SPEC, teacher, gates=GATES, mode=MODE, zoom=ZOOM,
    min_size=MIN_SIZE, batch_size=BATCH, seed_cap=SEED_CAP,
    match_iou=MATCH_IOU, progress=tqdm_over)

print()
print(summarise_rescue(ROUTES))
print("\nIf `rescue` recall does not beat `components` recall here, stop: the "
      "\nharvest below cannot do better on data where nobody can check.")
""", tag="measure")

md(r"""
## The harvest

Resumable by construction — a frame is done when its store exists — and staged
to Drive per dataset, as each finishes.

The unit area is estimated **per dataset, not per frame**, and that is not a
detail: per frame it degenerates exactly where it is needed, because a frame
whose only component is a fused pair has that pair as its own median and so
seeds it once. Across frames the singles dominate, which is the assumption the
whole seeding rests on.
""")

code(r"""
# --- Rescue each semantic dataset ---------------------------------------
from src.training.semantic import (estimate_units, label_semantic_pool,
                                   summarise_semantic_pool)

ROOTS = {"kust4k": DATA_ROOT / "Kust4K", "segfly": DATA_ROOT / "SegFly",
         "caltech_rgbt": DATA_ROOT / "Caltech" / "pairs"}
REPORTS = {}
for name, modality in HARVEST:
    root = ROOTS[name]
    if not root.exists():
        print(f"{name}: not downloaded, skipped")
        continue
    units = estimate_units(root, SPECS[name], modality=modality, mode=MODE,
                           limit=UNIT_FRAMES, seed=SEED, progress=tqdm_over)
    REPORTS[name] = label_semantic_pool(
        root, SPECS[name], teacher, POOL, dataset=name, modality=modality,
        gates=GATES, mode=MODE, zoom=ZOOM, min_size=MIN_SIZE,
        batch_size=BATCH, seed_cap=SEED_CAP, units=units,
        limit=FRAME_LIMIT, resume=RESUME, progress=tqdm_over)
    print()
    print(summarise_semantic_pool(REPORTS[name]))
    stage_pool(name)
""", tag="harvest")

md(r"""
## What this run produced

`instances per component` is the row to read. At **1.00** the teacher returned
exactly what `decompose` already had and this notebook bought nothing; every
point above it is a fused blob that came apart.

That number and the table in cell {{measure}} answer different halves of the
same question, and you need both: the measurement says the separation is
*right*, this says it *happened*.
""")

code(r"""
# --- The statement ------------------------------------------------------
for name, report in REPORTS.items():
    print(f"== {name} ({report['modality']}, teacher {report['teacher']})")
    print(summarise_semantic_pool(report))
    print()

print("Stores are per frame, run-length, in the same format stage B already "
      "\nreads -- wire one in with:")
print("    --dataset <spec>:<pool root>/<name>:<modality>:labels:train")
print("\n`labels` and not `components`: these stores already carry one value "
      "\nper instance, so decomposing them again would undo the work.")
""")

md(r"""
## When this route is the wrong one

Three cases, all measured rather than guessed, so nobody has to rediscover
them:

* **Nothing is fused.** Cell {{probe}}'s last column is the test. On sparse
  sets the components are already single objects and the rescue can only cost
  precision — Caltech's paired archive measured **138 instances from 43 of 289
  frames**, most of them alone in open terrain.
* **The things are tiny.** A click needs an interior to land in. Under roughly
  20 px a target has no ridge to seed and `seed_points` returns one point,
  which is `decompose` with extra steps.
* **The map is already instances.** VTUAV's VIS split and iSAID *ship* the
  separation; `--mode labels` reads it directly and nothing here should run.

And one that is not about this notebook at all: if what you need is **volume**
in thermal, the box pools of 14 remain the larger source — HIT-UAV alone
carries 24 899 boxes in 0.4 GB, which is more supervision than every semantic
thermal set here put together.
""")


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def stamp_for() -> str:
    body = "\n".join(f"{kind}\n{text}" for kind, text in CELLS)
    return hashlib.sha256(body.encode()).hexdigest()[:10]


def build() -> dict:
    global STAMP_VALUE
    STAMP_VALUE = stamp_for()
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
    out = repo / "notebooks" / NOTEBOOK
    document = build()
    out.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n")

    stamps = repo / "notebooks" / ".stamps.json"
    known = json.loads(stamps.read_text()) if stamps.is_file() else {}
    known[NOTEBOOK] = STAMP_VALUE
    stamps.write_text(json.dumps(known, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out.relative_to(repo)} with {len(CELLS)} cells [{STAMP_VALUE}]")
