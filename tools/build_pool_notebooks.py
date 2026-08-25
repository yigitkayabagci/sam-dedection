#!/usr/bin/env python3
"""Build the mask-pool notebooks from plain text blocks.

    python tools/build_pool_notebooks.py            # both of them
    python tools/build_pool_notebooks.py thermal    # just that one

Two notebooks come out of this one file:

    13_rgb_mask_pool.ipynb        RGB aerial boxes -> teacher masks
    14_thermal_mask_pool.ipynb    thermal boxes    -> teacher masks (the point)

They are one pipeline, not two: the same teacher, the same zoom-crop prompt,
the same four gates, the same run-length store. What differs is which datasets
feed it, which modality the store supervises, and -- only in 14 -- the
calibration that *chooses* between prompting the teacher on thermal pixels and
riding an RGB-T registration, plus a video pass that turns RGBT234's tracking
boxes into masklets.

Generated rather than edited for the same reason 07-11 are: a notebook is a
JSON document where prose, code and cell numbers drift apart silently.
`swap()` refuses to match nothing, `{{tag}}` resolves cross-references at
build time, and `tools/check_notebook.py` walks the result for names used
before they are bound. `tests/test_notebooks.py` holds the shipped file to the
stamp this build writes into `notebooks/.stamps.json`.
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
    blurb: str                  # one paragraph under the title
    disk: str
    takes: str
    fetch: str                  # the FETCH list in the settings cell
    fetch_note: str             # the markdown above the download cell
    probe: str                  # the cell that builds FRAMES and prints counts
    calibration_md: str
    calibration: str            # the cell that fills CALIBRATION
    harvest: str                # the harvest loop body
    mirror: str                 # Drive folder, per branch
    appendix: str               # closing markdown


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
    """Replace `{{tag}}` with the cell number that tag landed on.

    The variant fields go **first**: a field's text may itself point at a
    tagged cell (`fetch_note` says "cell {{probe}}"), and a single pass only
    resolves what is present when its turn comes.
    """
    substitutions = {"TITLE": V.title, "BLURB": V.blurb, "DISK": V.disk,
                     "TAKES": V.takes, "FETCH": V.fetch,
                     "FETCH_NOTE": V.fetch_note, "PROBE": V.probe,
                     "CALIBRATION_MD": V.calibration_md,
                     "CALIBRATION": V.calibration, "HARVEST": V.harvest,
                     "MIRROR": V.mirror, "APPENDIX": V.appendix,
                     "NOTEBOOK": Path(V.path).name, "STAMP": STAMP_VALUE,
                     **{t: str(i) for t, i in TAGS.items()}}
    for tag, value in substitutions.items():
        text = text.replace("{{" + tag + "}}", value)
    leftover = re.findall(r"\{\{(\w+)\}\}", text)
    if leftover:
        raise KeyError(f"unknown cell tags: {sorted(set(leftover))}")
    return text


def swap(text: str, old: str, new: str) -> str:
    """`str.replace` that refuses to do nothing."""
    if old not in text:
        raise AssertionError(f"nothing matched:\n{old[:200]}")
    return text.replace(old, new)


# --------------------------------------------------------------------------
# The two variants
# --------------------------------------------------------------------------

VARIANTS["rgb"] = Variant(
    key="rgb",
    path="notebooks/13_rgb_mask_pool.ipynb",
    title="Mask pool — aerial RGB boxes through a promptable teacher",
    blurb=(
        "Builds the **RGB pool**: aerial detection boxes (VisDrone, and "
        "optionally DroneVehicle's RGB half) become per-instance teacher "
        "masks, gated and stored as `(image, box prompt, mask)` triples for "
        "stage B. This is the easy branch — the teachers were trained on RGB "
        "— and it exists beside `14_thermal_mask_pool.ipynb`, which is the "
        "one this project is actually about. Run this one to grow the RGB "
        "side of the encoder's diet and to rehearse the pipeline where the "
        "teacher is strongest."),
    disk="~6 GB of free disk (VisDrone ~1.6 GB, Kust4K's RGB half ~1.7 GB "
         "for calibration; +14 GB if `INCLUDE_DRONEVEHICLE`)",
    takes="~10 min to download, ~15 min to calibrate, and roughly 1.5–3 h to "
          "harvest VisDrone-train at the default caps (teacher-bound; scale "
          "`FRAME_LIMIT` down for a first pass)",
    fetch="""FETCH = [
    # VisDrone2019-DET, YOLO-converted mirror: plain files, snapshot-fetched.
    ("visdrone", DATA_ROOT / "VisDrone", []),
    # Kust4K's RGB half + labels -- not harvested, *calibrated on*: real
    # drawn masks under aerial RGB, the closest thing this branch has to
    # ground truth for the teacher itself.
    ("kust4k",   DATA_ROOT / "Kust4K",   ["rgb", "labels"]),
]
if INCLUDE_DRONEVEHICLE:
    FETCH.append(("dronevehicle", DATA_ROOT / "DroneVehicle", ["train"]))""",
    fetch_note="""* **VisDrone** rides `snapshot_download` against a plain-file Hub mirror
  (`banu4prasad/VisDrone-Dataset`) — the official release lives behind
  per-file Google Drive links that quota out under Colab's shared egress
  addresses. Third-party mirror, so cell {{probe}} counts what arrived and
  checks the class table against the download before anything is labelled.
* **Kust4K** comes from figshare's per-file endpoints with the publisher's
  md5 — the whole-article link answers HTTP 202 with an empty body (it
  *starts building* the zip and returns before it exists). Only the RGB half
  and the labels are fetched here; the thermal half belongs to notebook 14.
* **DroneVehicle** (optional, `INCLUDE_DRONEVEHICLE`) is the same 28 442-pair
  archive notebooks 09/14 use; its RGB half carries the same oriented boxes.""",
    probe="""# --- What arrived, and does the reader agree? ---------------------------
from src.training import boxes as B
from src.training.aerial import describe_layout

FRAMES = {}

visdrone_split = sorted((DATA_ROOT / "VisDrone").rglob("*DET-train"))
if not visdrone_split:
    print(describe_layout(DATA_ROOT / "VisDrone"))
    raise SystemExit("no *DET-train under the VisDrone root -- the snapshot "
                     "did not land; re-run the download cell and read its output")
FRAMES["visdrone"] = B.yolo_frames(visdrone_split[0])

if INCLUDE_DRONEVEHICLE:
    FRAMES["dronevehicle_rgb"] = B.dronevehicle_frames(
        DATA_ROOT / "DroneVehicle", modality="rgb")

for name, frames in FRAMES.items():
    print(B.summarise_frames(frames, name)); print()

# The names table is an assumption about a third-party conversion; the
# histogram is the download's own answer. Ten classes, vehicles and people
# dominant -- anything else here means the yaml in the mirror changed and
# `B.VISDRONE_NAMES` must be fixed before a single mask is made.
histogram = B.class_histogram(FRAMES["visdrone"])
unknown = [name for name in histogram if name.startswith("class ")]
assert not unknown, (
    f"label indexes outside VISDRONE_NAMES: {unknown} -- the mirror's class "
    f"table moved; fix src/training/boxes.py before labelling")
print(f"{len(histogram)} classes, all inside VISDRONE_NAMES")""",
    calibration_md="""## Calibrate before you spend

VisDrone has boxes and no masks, so nothing in the harvest can say how *good*
a teacher mask is — the gates only say where it sits. Kust4K can: its RGB half
is aerial urban imagery with real semantic masks, and `decompose` turns those
into instances whose boxes prompt the teacher exactly the way the harvest
will. IoU against the drawn mask, per class and per size bucket, is the
number to read before spending teacher-hours — and the small-bucket rows are
the ones that matter, because that is where this project lives.""",
    calibration="""# --- Teacher vs drawn masks, on Kust4K's RGB half -----------------------
from src.training.aerial import SPECS
from src.training.pool import calibrate_spec, calibration_table

CALIBRATION = {"rgb": calibrate_spec(
    DATA_ROOT / "Kust4K", SPECS["kust4k"], teacher,
    modality="rgb", prompt="self", limit_frames=CAL_FRAMES,
    per_frame=CAL_PER_FRAME, zoom=ZOOM, min_size=MIN_SIZE,
    batch_size=BATCH, seed=SEED, progress=tqdm_over)}

print(f"{len(CALIBRATION['rgb'])} instances scored\\n")
print(calibration_table(CALIBRATION))""",
    harvest="""# --- The harvest: every source, resumable, staged as it finishes --------
REPORTS = {}
for name, frames in FRAMES.items():
    REPORTS[name] = label_pool(
        frames, teacher, POOL, dataset=name, prompt="self", gates=GATES,
        zoom=ZOOM, min_size=MIN_SIZE, batch_size=BATCH, limit=FRAME_LIMIT,
        max_boxes=MAX_BOXES, resume=RESUME, progress=tqdm_over)
    write_index(POOL)
    report = REPORTS[name]
    print(f"{name}: {report['accepted']}/{report['attempted']} accepted "
          f"({report['acceptance_rate']:.1%}), "
          f"{report['resumed']} image(s) resumed")
    stage_pool(name)          # to Drive the moment it exists, not at the end""",
    mirror="edgetam-pool/rgb",
    appendix="""## What to do with the pool

The stores are `labels.py`'s own format — `pseudo_masks.npz` keyed by box
index, `record.json` beside each saying which box, which class, which verdict
— so stage B reads them the way it already reads Anti-UAV410's pseudo-labels.
Wiring the pool into `07`'s `DATASETS` list is a follow-up change to
`src/training/datasets.py`, deliberately not part of this notebook: this one
produces supervision and *measures* it, it does not train.

## The contamination rule this pool creates

**AeroVIS is VisDrone re-labelled** (plus UAVDT and SeaDronesSee, boxes
through SAM3). A model trained on this pool has therefore *seen* AeroVIS's
frames, and AeroVIS stops being a held-out evaluation for it — frame for
frame, not in spirit. Clean video evaluations that remain: UAVScenes, MVUAV.
`docs/datasets.md` carries the full overlap table.

## Licences

VisDrone's mirror carries the original's academic-use terms; Kust4K is
CC BY 4.0; DroneVehicle is academic-use. The *teacher's* licence matters too:
masks made by SAM 3 inherit its use restrictions — read them once before a
deliverable depends on this pool; `facebook/sam2.1-hiera-large` is Apache-2.0
and one string away.""",
)

VARIANTS["thermal"] = Variant(
    key="thermal",
    path="notebooks/14_thermal_mask_pool.ipynb",
    title="Mask pool — thermal aerial boxes, with the route measured first",
    blurb=(
        "Builds the **thermal pool**, which is the branch this project is "
        "for: thermal detection boxes (HIT-UAV, DroneVehicle's thermal half) "
        "become per-instance teacher masks, and RGBT234's tracking boxes "
        "become per-frame masklets. Thermal is the modality the teachers "
        "never trained on, so the notebook's first real result is a "
        "**measurement, not a harvest**: on Kust4K's drawn instances it "
        "scores the teacher prompted on thermal pixels against the teacher "
        "prompted on the registered RGB twin — per class, per size — and the "
        "harvest then runs whichever route the table supports. Its sibling "
        "`13_rgb_mask_pool.ipynb` is the same pipeline where the teacher is "
        "at home."),
    disk="~24 GB of free disk (DroneVehicle train 8.9 GB, RGBT234 7.7 GB, "
         "Kust4K 2.7 GB, HIT-UAV 0.4 GB)",
    takes="~35 min to download, ~30 min to calibrate both routes, ~1–2 h for "
          "the box harvest, and ~2–5 h for the RGBT234 masklet pass at the "
          "default caps (every pass resumes; scale the caps for a first run)",
    fetch="""FETCH = [
    # 2 898 thermal frames with COCO boxes; the git repo *is* the dataset.
    ("hituav",       DATA_ROOT / "HIT_UAV",      []),
    # 28 442 registered RGB-T pairs, oriented boxes per modality. The RGB
    # half rides along in the same archive -- it is the `pair` the teacher
    # looks at if calibration votes for the RGB-prompt route.
    ("dronevehicle", DATA_ROOT / "DroneVehicle", ["train"]),
    # Kust4K, all three parts: thermal + labels to calibrate the direct
    # route, RGB to calibrate the pair route on the *same* instances.
    ("kust4k",       DATA_ROOT / "Kust4K",       ["tir", "labels", "rgb"]),
    # 234 aligned RGB-T videos with per-modality boxes -- the masklet pass.
    ("rgbt234",      DATA_ROOT / "RGBT234",      []),
]""",
    fetch_note="""* **HIT-UAV**'s git repository is the dataset — annotations *and* frames,
  ~0.4 GB through the archive endpoint (no resume there; it is small enough
  not to matter). The Kaggle mirror needs an API token and the Roboflow
  re-export keeps one class of four, so the repo it is.
* **DroneVehicle** downloads from an anonymous Hub mirror with range
  support; the authors' own distribution is Baidu. Every frame ships a
  100 px white band on all four sides — `boxes.dronevehicle_frames` carries
  it as an inset and the labeller cuts it before the teacher ever looks.
* **Kust4K** comes from figshare's per-file endpoints with the publisher's
  md5 (the whole-article link answers HTTP 202 with an empty body). All
  three parts land here because the calibration needs both halves of the
  same registered pair.
* **RGBT234** is one 7.7 GB tar.gz on an anonymous Hub mirror
  (`xche32/rgbt234`) — the official release is Baidu-only. Third-party
  mirror, so cell {{probe}} counts sequences, frames and box files before
  anything trusts it. **LasHeR** (1 224 sequences, same layout) exists on
  the same terms but is **~224 GB** and is deliberately not in `FETCH`; the
  appendix says how to stream sequences of it selectively.""",
    probe="""# --- What arrived, and does the reader agree? ---------------------------
from src.training import boxes as B
from src.training.aerial import describe_layout
from src.training.masklets import find_rgbt_sequences

FRAMES = {}
FRAMES["hituav"] = B.hituav_frames(DATA_ROOT / "HIT_UAV", split="train")
FRAMES["dronevehicle_thermal"] = B.dronevehicle_frames(
    DATA_ROOT / "DroneVehicle", modality="thermal")

for name, frames in FRAMES.items():
    print(B.summarise_frames(frames, name)); print()

# HIT-UAV's names ride inside its own COCO json -- nothing to assume -- but
# the *set* is still worth pinning: person/car/bicycle is the population this
# pool exists to cover, and a json that suddenly says otherwise is the mirror
# having moved under us.
histogram = B.class_histogram(FRAMES["hituav"])
assert any("person" in name.lower() for name in histogram), (
    f"HIT-UAV without a person class: {sorted(histogram)} -- wrong json?")

SEQUENCES = find_rgbt_sequences(DATA_ROOT / "RGBT234")
paired = sum(1 for s in SEQUENCES if s.gate_boxes is not None)
total_frames = sum(len(s) for s in SEQUENCES)
print(f"RGBT234: {len(SEQUENCES)} sequences, {total_frames} frame pairs, "
      f"{paired} with separate infrared boxes")
if not SEQUENCES:
    print(describe_layout(DATA_ROOT / "RGBT234"))
    raise SystemExit("no visible.txt-shaped sequences under the RGBT234 root "
                     "-- read the layout above and check the extraction")
assert len(SEQUENCES) >= 200, (
    f"{len(SEQUENCES)} sequences where ~234 were promised -- partial "
    f"extraction? re-run the download cell")""",
    calibration_md="""## The measurement the whole branch turns on

Two ways to get a thermal mask out of an RGB-trained teacher:

1. **direct** — prompt it on the thermal frame and hope grayscale heat reads
   enough like imagery;
2. **pair** — prompt it on the registered RGB twin and let the registration
   carry the mask across, paying whatever residual misalignment the pair has.

Every RGB-T paper picks one and argues. Kust4K lets us *measure* instead: its
4 024 registered pairs have real masks, so the same drawn instance can prompt
the teacher both ways and be scored against the same truth. Per class and per
size bucket, because the answer is allowed to differ — and day/night rides in
the frame stems (`...D`/`...N`) if you want to slice further.

The harvest cell below reads `HARVEST_PROMPT` from the settings; **come back
and set it** after reading this table. DroneVehicle can use either route (its
pairs are registered); HIT-UAV is thermal-only, so it always runs direct —
which is also why the direct column of this table matters even if pair wins.""",
    calibration="""# --- Teacher vs drawn masks, both routes, same instances ----------------
from src.training.aerial import SPECS
from src.training.pool import calibrate_spec, calibration_table

CALIBRATION = {}
for route, prompt in (("thermal-direct", "self"), ("rgb-pair", "pair")):
    CALIBRATION[route] = calibrate_spec(
        DATA_ROOT / "Kust4K", SPECS["kust4k"], teacher,
        modality="thermal", prompt=prompt, limit_frames=CAL_FRAMES,
        per_frame=CAL_PER_FRAME, zoom=ZOOM, min_size=MIN_SIZE,
        batch_size=BATCH, seed=SEED, progress=tqdm_over)
    print(f"{route}: {len(CALIBRATION[route])} instances scored")

print()
print(calibration_table(CALIBRATION))
print()
print("Read the small buckets first. Then set HARVEST_PROMPT in the settings"
      "\\ncell -- 'self' for thermal-direct, 'pair' to ride the registration --"
      "\\nand run the harvest below. HIT-UAV runs direct either way.")""",
    harvest="""# --- The box harvest: every source, resumable, staged as it finishes ----
REPORTS = {}
for name, frames in FRAMES.items():
    prompt = "self" if name == "hituav" else HARVEST_PROMPT
    REPORTS[name] = label_pool(
        frames, teacher, POOL, dataset=name, prompt=prompt, gates=GATES,
        zoom=ZOOM, min_size=MIN_SIZE, batch_size=BATCH, limit=FRAME_LIMIT,
        max_boxes=MAX_BOXES, resume=RESUME, progress=tqdm_over)
    write_index(POOL)
    report = REPORTS[name]
    print(f"{name} (prompt={prompt}): "
          f"{report['accepted']}/{report['attempted']} accepted "
          f"({report['acceptance_rate']:.1%}), "
          f"{report['resumed']} image(s) resumed, "
          f"{report['no_pair']} without a pair, "
          f"{report['size_mismatch']} on a mismatched grid")
    stage_pool(name)          # to Drive the moment it exists, not at the end""",
    mirror="edgetam-pool/thermal",
    appendix="""## LasHeR, when a bigger pool is worth a bigger machine

LasHeR is RGBT234's layout at 5x the sequences and ~30x the bytes: 1 224
aligned sequences, ~224 GB as five tar.gz slices on `xche32/lasher`. That does
not fit a Colab disk, so it is not in `FETCH`. The fetcher can stream it —
the slices are read as one gzip stream and only the sequences named are ever
written to disk:

```
python tools/fetch_datasets.py lasher --dest /content/data/LasHeR \\
    --parts part_aa part_ab part_ac part_ad part_ae \\
    --sequences 1boygo 2boysup [...]
```

The stream still *reads* up to the last wanted member (a tar.gz has no
index), so the network cost stays high even when little is kept. A full
LasHeR masklet pass belongs on a machine with a real disk; the masklet cell
above works unchanged on it, because `find_rgbt_sequences` reads LasHeR and
RGBT234 alike.

## VTUAV, the third video source

VTUAV's ~400 box-only sequences go through the *same* store via
`tools/make_masklets.py` — that path predates this notebook and stays where
it is. Calibrate it on the VIS split's drawn masks (`--calibrate`), which is
a stronger check than RGBT234 can offer, then harvest.

## The two teacher rules worth repeating

* **SAM 3 saw MOSEv2 in training.** Fine for making training pools; fatal
  for a MOSEv2 evaluation. Every store's `record.json`/`masklets.json`
  carries the teacher id, so which masks came from which teacher is a fact
  on disk, not a memory.
* **`facebook/sam3.1` is not the upgrade it sounds like** — checkpoint-only,
  no transformers integration, and its Object Multiplex throughput win is
  for many-objects-per-frame, which a box-per-crop labeller never sees.

## Licences

HIT-UAV is CC BY 4.0; DroneVehicle, RGBT234 and LasHeR are academic-use.
Masks inherit the *teacher's* terms too: SAM 3's licence carries use
restrictions Apache-2.0 does not — `facebook/sam2.1-hiera-large` is the
unencumbered fallback, one string away in the settings.""",
)

V = VARIANTS[sys.argv[1]] if len(sys.argv) > 1 else VARIANTS["rgb"]


# ---------------------------------------------------------------- 0
md(r"""
# {{TITLE}}

One runtime, one **Run all**: download box-annotated aerial data, prompt a
promptable-segmentation teacher with every box, gate what comes back, and
leave a **mask pool** on your Drive — `(image, box prompt, mask)` supervision
in the exact store the rest of this repo already reads.

{{BLURB}}

| | |
|---|---|
| **in** | nothing staged by hand — the download cell fetches everything |
| **out** | per-image run-length mask stores + acceptance/calibration reports, zipped to `{{MIRROR}}` on your Drive |
| **needs** | a CUDA GPU and {{DISK}} |
| **takes** | {{TAKES}} |

**This notebook trains nothing.** It manufactures and *measures* supervision:
stage B (`07`–`11`) is where training numbers come from, stage C is where the
masklets go. Keeping production separate from consumption is what lets a pool
be inspected, calibrated and staged once, then reused by every run after it.

---

## The pipeline, in one diagram

```
  box dataset          per box:                       teacher            gates
  ┌───────────┐        ┌────────────────┐        ┌──────────────┐   ┌──────────────┐
  │ image     │        │ zoom crop      │        │ SAM 3 (or    │   │ teacher_iou  │
  │  + boxes  │ ─────▶ │ around the box │ ─────▶ │ SAM 2.1),    │──▶│ box_iou      │──▶ RLE store
  │  + class  │        │ (x4 long side, │        │ box-prompted │   │ area         │    per image
  └───────────┘        │  ≥128 px)      │        └──────────────┘   │ component    │  + record.json
                       └────────────────┘                           └──────────────┘
```

Three decisions carried over from the Anti-UAV410 labeller, because they were
measured there:

* **Zoom.** A 15-pixel vehicle in a full frame is a problem teachers fail;
  the same vehicle on a 128-pixel crop resized up is one they solve. The crop
  is the whole trick.
* **Gates.** A teacher mask is kept only if four independent checks agree —
  its own confidence, agreement with the prompting box, area sanity, and
  single-component-ness. What was rejected and *why* is reported per gate;
  the acceptance rate is a measured number, not a hope.
* **The store.** Accepted masks land in the same run-length `.npz` the
  Anti-UAV410 pseudo-labels use, keyed by box index, with a `record.json`
  naming each box, class and verdict. Whatever reads pseudo-masks today reads
  this pool tomorrow; a rejected box is *absent*, which readers already treat
  as "no mask supervision here", not as an empty mask.

## Which teacher, and why it is a setting

`TEACHER` defaults to **`facebook/sam3`** — the strongest promptable
segmenter with a real `transformers` integration (≥5.0): box prompts, batched
crops, SAM 2's API. It is **gated**: open its model page once, accept the
terms, and log in below. Two facts to hold while choosing:

* **`facebook/sam2.1-hiera-large`** is one string away: ungated, Apache-2.0,
  needs only transformers 4.56. If SAM 3's gate or licence is in the way,
  switch and re-run — every report records which teacher made which mask.
* **`facebook/sam3.1` is not an option**, checked rather than assumed: it
  ships as a bare checkpoint (no transformers classes), pins `numpy<2`
  through its GitHub package, and its headline Object Multiplex is a
  many-objects-per-frame throughput win — this labeller prompts one box per
  crop and would gain nothing.

The choice is *measured* anyway: cell {{calibration}} scores the configured
teacher against real drawn masks before the harvest spends an hour on it.
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

# Drop this repo's modules so the fast-forward above actually takes effect:
# `git pull` changes files on disk, not modules Python already imported.
for _stale in [n for n in list(sys.modules) if n.split(".")[0] in ("src", "tools")]:
    del sys.modules[_stale]

!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
!df -h /content | tail -1

# --- Is this notebook the one the repo expects? -------------------------
# The repo just fast-forwarded itself; the .ipynb is a file you uploaded, and
# the two drift apart silently. This says so at cell 1 instead of forty
# minutes in.
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

# ---------------------------------------------------------------- 1b
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

# ---------------------------------------------------------------- 2
code(r"""
# --- Dependencies -------------------------------------------------------
# No EdgeTAM here -- this notebook trains nothing, so the sam2-fork/teacher
# collision the training notebooks manage does not exist. The teacher runs
# through transformers alone. 5.0 is the floor that has the Sam3Tracker
# classes; SAM 2.1 needs only 4.56, so one install covers both choices.
before = torch.__version__
!pip install -q "transformers>=5" hf_transfer tqdm

# The one thing an install here must not do: replace the preinstalled torch.
# Read the version off *disk* (reloading torch always raises); pip cannot
# change the torch already live in this kernel, only a restart can -- which
# is exactly the window this check exists to catch.
import importlib.metadata as _md
installed = _md.version("torch")
assert installed == before, (
    f"pip replaced torch {before} with {installed} on disk. Restore it "
    f"before restarting:  !pip install -q --force-reinstall torch=={before}")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import transformers
print(f"transformers {transformers.__version__}")

# The contracts everything below depends on, tested with no GPU and no
# dataset. If these fail, nothing after this point is worth running.
!python -m unittest -q tests.test_mask_pool tests.test_masklets 2>&1 | tail -3
""")

# ---------------------------------------------------------------- settings
code(r"""
# --- Where things live, and every knob ----------------------------------
DATA_ROOT = Path("/content/data")      # datasets; local disk, never Drive
WORK      = Path("/content/work")      # scratch
POOL      = WORK / "pool"              # the product: stores + records

# The teacher. facebook/sam3 is gated -- accept its terms once, then log in
# below. The ungated fallback needs no account and only transformers>=4.56:
#     TEACHER = "facebook/sam2.1-hiera-large"
TEACHER = "facebook/sam3"
DEVICE  = "cuda"
DTYPE   = "bfloat16"

# The gates, exactly labels.Gates -- see that docstring for why teacher_iou
# is a weak gate at small sizes and box_iou is the load-bearing one.
from src.training.labels import Gates
GATES = Gates()                        # Gates(box_iou=0.6, area=(0.15, 1.3), ...)

ZOOM      = 4.0                        # crop side, in box long-sides
MIN_SIZE  = 128                        # crop floor, px -- context for tiny targets
BATCH     = 8                          # crops per teacher forward
MAX_BOXES = 64                         # per image, largest first (VisDrone
                                       # frames can carry hundreds)
FRAME_LIMIT = None                     # images per dataset; 200 for a smoke run
RESUME     = True                      # skip images whose store exists
SEED       = 0

# Calibration budget: frames sampled (seeded), instances per frame.
CAL_FRAMES    = 150
CAL_PER_FRAME = 6

INCLUDE_DRONEVEHICLE = False           # 13 only: +14 GB, same pipeline
HARVEST_PROMPT = "self"                # 14 only: set from the calibration
                                       # table -- "self" = thermal-direct,
                                       # "pair" = ride the registration
# RGBT234 masklet pass (14 only): sequences and frames are capped for a
# first pass; raise them once the acceptance and the overlays look right.
RGBT234_SEQUENCES  = 24
RGBT234_MAX_FRAMES = 600               # per sequence
RGBT234_CHUNK      = 200               # frames per re-prompt; drift bound

for d in (DATA_ROOT, WORK, POOL):
    d.mkdir(parents=True, exist_ok=True)

def tqdm_over(stream, total, desc):
    from tqdm.auto import tqdm
    return tqdm(stream, total=total, desc=desc)

print(f"teacher {TEACHER}  gates {GATES}")
print(f"pool -> {POOL}")
""", tag="settings")

# ---------------------------------------------------------------- drive
code(r"""
# --- Drive: where finished work survives a dead runtime -----------------
from google.colab import drive as _drive
_drive.mount("/content/drive")

MIRROR = Path("/content/drive/MyDrive/{{MIRROR}}")
MIRROR.mkdir(parents=True, exist_ok=True)

# Zip one dataset's stores to Drive the moment its harvest finishes. The
# pool is thousands of small files; as a folder copy it would trickle for
# minutes and litter Drive's change log, as one stored (not deflated -- the
# payload is already-compressed npz) zip it lands in seconds and later
# runtimes can pull it back with one copy.
from tools.fetch_datasets import archive_to

def stage_pool(dataset):
    source = POOL / dataset
    if not any(source.rglob("*.npz")):
        print(f"   !! nothing under {source} -- not staging")
        return
    archive_to(source, f"pool_{dataset}", MIRROR)

print(f"staging to {MIRROR}")
""")

# ---------------------------------------------------------------- hf login
code(r"""
# --- Hugging Face: the teacher's gate -----------------------------------
# facebook/sam3 is a gated repository: open
#     https://huggingface.co/facebook/sam3
# once, accept the terms, then run this cell and paste a token (read scope).
# Skippable entirely when TEACHER is the ungated SAM 2.1.
if "sam3" in TEACHER.lower():
    from huggingface_hub import login, whoami
    try:
        print(f"already logged in as {whoami()['name']}")
    except Exception:
        login()
else:
    print(f"{TEACHER} is ungated -- no login needed")
""")

# ---------------------------------------------------------------- 6
md(r"""
## Downloading the data

`FETCH` in the settings cell names the datasets; the cell below downloads,
verifies and extracts them into the layout the readers glob. Every URL lives
in `tools/fetch_datasets.py` and was checked against the live host, because
several are served in a way that defeats the obvious approach:

{{FETCH_NOTE}}

Downloads resume where the host allows it, archives are deleted once
extracted, and a copy staged in `MyDrive/datasets/` is used **before** the
network is touched — the standing escape hatch for anything quota-shaped.
""")

# ---------------------------------------------------------------- 7
code(r"""
# --- Download everything, once ------------------------------------------
from tools.fetch_datasets import fetch

{{FETCH}}

for name, dest, parts in FETCH:
    marker = Path(dest) / ".fetched"
    if marker.is_file():
        print(f"{name}: already fetched -> {dest}")
        continue
    fetch(name, Path(dest), parts=tuple(parts) or None)
    marker.touch()

!df -h /content | tail -1
""")

# ---------------------------------------------------------------- probe
code(r"""
{{PROBE}}
""", tag="probe")

# ---------------------------------------------------------------- teacher
code(r"""
# --- The teacher, loaded once -------------------------------------------
# build_image_teacher dispatches on the id ("sam3" -> Sam3TrackerModel,
# otherwise Sam2Model) and turns a gated-repo refusal into instructions
# rather than a stack trace.
from src.training.labels import build_image_teacher

teacher = build_image_teacher(TEACHER, device=DEVICE, dtype=DTYPE)
print(f"{teacher.model_id} on {DEVICE} ({DTYPE})")
""")

# ---------------------------------------------------------------- look
md(r"""
## Look before spending

A histogram cannot tell you a mask is wrong. Eight frames, every box drawn:
**green** masks passed the gates, **red** boxes are what the teacher failed
and *why*. If the green here does not look like objects, stop — the settings
cell is where the zoom and the gates live, and GPU-hours spent past a bad
panel are not recovered.
""")

code(r"""
# --- Eight frames, every verdict visible --------------------------------
import numpy as np
import matplotlib.pyplot as plt
from src.training.pool import label_boxes, _read_rgb

_name, _frames = next(iter(FRAMES.items()))
figure, axes = plt.subplots(2, 4, figsize=(22, 10))
for axis, frame in zip(axes.ravel(), _frames[:8]):
    pixels = _read_rgb(frame.image)
    resolved, keep = frame.resolved(pixels.shape[:2])
    if frame.inset:
        pixels = pixels[frame.inset:-frame.inset, frame.inset:-frame.inset]
    chosen = [i for i in range(len(resolved)) if keep[i]][:MAX_BOXES]
    masks, rows = label_boxes(pixels, resolved[chosen], teacher, gates=GATES,
                              zoom=ZOOM, min_size=MIN_SIZE, batch_size=BATCH)
    canvas = pixels.copy()
    for local, mask in masks.items():
        canvas[mask] = 0.45 * canvas[mask] + np.array([0, 140, 0])
    axis.imshow(canvas)
    for row in rows:
        x0, y0, x1, y1 = resolved[chosen[row["i"]]]
        ok = row["verdict"] is None
        axis.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                     edgecolor="lime" if ok else "red",
                                     linewidth=1.2))
        if not ok:
            axis.text(x0, max(y0 - 3, 0), row["verdict"], color="red", fontsize=7)
    accepted = sum(1 for r in rows if r["verdict"] is None)
    axis.set_title(f"{frame.key}: {accepted}/{len(rows)} accepted", fontsize=9)
    axis.axis("off")
plt.suptitle(f"{_name} through {teacher.model_id}", y=1.0)
plt.tight_layout(); plt.show()
""", tag="look")

# ---------------------------------------------------------------- calibration
md(r"""
{{CALIBRATION_MD}}
""")

code(r"""
{{CALIBRATION}}
""", tag="calibration")

# ---------------------------------------------------------------- harvest
md(r"""
## The harvest

Resumable by construction — a frame is *done* when its store exists, so a
dead runtime costs the frame it was on and nothing else — and staged to
Drive **per dataset, as each finishes**, because a session that dies three
datasets in should keep three datasets. Scale `FRAME_LIMIT` down for a first
pass; the caps, the gates and this cell's reports all land in the record
files, so two pools are always comparable.
""")

code(r"""
from src.training.pool import label_pool, write_index

{{HARVEST}}
""", tag="harvest")

# ---------------------------------------------------------------- masklets (14)
if V.key == "thermal":
    md(r"""
## RGBT234: boxes become masklets

The video half of the pool. RGBT234's 234 aligned sequences annotate **both
modalities separately** — `visible.txt` prompts the teacher (RGB is where it
was trained), `infrared.txt` gates the result, and that second box is the
insurance on the registration: a mask that no longer sits on the *thermal*
annotation is dropped, whatever the RGB half thought of it.

Chunked propagation (`RGBT234_CHUNK` frames per re-prompt) bounds drift and
memory at once, exactly as the VTUAV masklet pass does; the store is
`labels.py`'s own, so stage C reads masklets and pseudo-labels through one
interface. The caps make the first pass affordable — raise them once the
acceptance table looks right.
""")

    code(r"""
# --- The masklet pass ---------------------------------------------------
# A second model: SAM 3's *video* tracker (or SAM 2.1's video predictor --
# same dispatch-by-id as the image teacher). The image teacher is dropped
# first; two checkpoints of this size do not share a T4.
import gc
del teacher
gc.collect(); torch.cuda.empty_cache()

from src.training.masklets import (build_video_teacher, masklet_sequence,
                                   summarise_masklets)

video_teacher = build_video_teacher(TEACHER, device=DEVICE, dtype=DTYPE)

MASKLETS = POOL / "rgbt234"
MASKLET_REPORTS = []
for sequence in SEQUENCES[:RGBT234_SEQUENCES]:
    done = MASKLETS / sequence.name / "masklets.json"
    if RESUME and done.is_file():
        MASKLET_REPORTS.append(_json.loads(done.read_text()))
        print(f"  {sequence.name}: already done, "
              f"{MASKLET_REPORTS[-1]['accepted']} masklet frames")
        continue
    report = masklet_sequence(
        sequence, video_teacher, MASKLETS, gates=GATES,
        prompt_frames="rgb", chunk=RGBT234_CHUNK,
        max_frames=RGBT234_MAX_FRAMES, progress=tqdm_over)
    MASKLET_REPORTS.append(report)
    print(f"  {sequence.name}: {report['accepted']}/{report['attempted']} "
          f"accepted ({report['acceptance_rate']:.1%})")

print()
print(summarise_masklets(MASKLET_REPORTS))
stage_pool("rgbt234")
""", tag="masklets")

# ---------------------------------------------------------------- report
md(r"""
## What this run produced
""")

code(r"""
# --- The pool, re-read off disk -----------------------------------------
# Off disk and not from this session's variables, so the statement is true
# across resumed runs and partial sessions alike.
from src.training.pool import pool_report, summarise_pool

REPORT = pool_report(POOL)
print(summarise_pool(REPORT))

total = sum(entry["accepted"] for entry in REPORT.values())
teachers = sorted({t for entry in REPORT.values() for t in entry["teachers"]})
print(f"\n{total} accepted masks across {len(REPORT)} dataset(s), "
      f"teacher(s): {', '.join(teachers)}")
print(f"stores under {POOL}, staged zips under {MIRROR}")

import json as _json2
(WORK / "pool_report.json").write_text(_json2.dumps(REPORT, indent=2))
import shutil as _shutil
_shutil.copy(WORK / "pool_report.json", MIRROR / "pool_report.json")
print("pool_report.json -> Drive")
""", tag="report")

# ---------------------------------------------------------------- appendix
md(r"""
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
            subprocess.run([sys.executable, __file__, key], check=True,
                           cwd=repo)
