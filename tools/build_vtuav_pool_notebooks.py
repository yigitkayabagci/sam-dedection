#!/usr/bin/env python3
"""Generate notebooks 16, 17, 24 and 25: VTUAV's halves and splits, one pool each.

Same shape as 15 -- code cells only, no prose, no comments -- and the same
generator pattern as `build_pool_notebooks.py`: one file, four variants, so no
two of them can drift apart by an edit that only landed on one.

**Four, because the tracking download is two splits.** 16 and 17 harvest the
short-term parts; 24 and 25 harvest the long-term ones, into pools of their
own (`vtuav_lt_rgb`, `vtuav_lt_thermal`) rather than into 16 and 17's. Three
reasons, none of them tidiness:

- a short-term pool already mirrored to Drive stays valid, and the long-term
  harvest is *additive* -- `pool_reader.discover_pools`, which is how 19 and 20
  find their food, picks a new folder up with no edit anywhere;
- `aerial.split_index` stratifies per dataset, so a separate pool gets its own
  held-out sequences instead of being diluted into the short-term split;
- the long-term parts are the ones with real disappearances, and a pool that
  keeps them separate can be weighed separately.

**What "long-term" changes, and what it does not.** LT and ST are the same
sensor, the same 1920x1080 registered RGB-T pairs and the same
`<sequence>/{rgb,ir}/` + `<sequence>/{rgb,ir}.txt` layout; the split is about
the *tracking task*, not the data. What differs is that LT sequences are
longer, the target leaves the frame and comes back, and those frames are
annotated absent. `boxes.vtuav_frames` drops an absent row rather than
prompting the teacher with a zero-area or NaN box, and cell 2 prints how many
rows that was per archive before anything is staged.

**Night is a question about the teacher, and it is asked with a number.** A
promptable segmenter trained on web images reads a daylit street well and a
frame where the target is a smear around a headlight badly, and the gates do
not save you there -- they catch a mask that has drifted off its box, not a
plausible one drawn around glare. The structural answer is already in place and
is why 16/17 and 24/25 are pairs: each modality is prompted on **its own**
pixels, so a night frame's thermal half is 25's business and its RGB half is
24's, and there is no route where a night RGB mask is mirrored onto thermal
(VTUAV closed that door anyway -- 12.2 % box agreement).

What was missing was the evidence. Every record now carries `luma` (the frame)
and `target_luma` (inside the boxes -- an aerial night frame is mostly black
with the annotated thing under a lamp, so the frame's own mean files a well-lit
target under "night" and reads the failure off the wrong rows), and cell 5
prints acceptance bucketed by the second. `MIN_LUMA` then drops frames whose
targets are darker than a threshold -- **for the RGB arm only**. It defaults to
off rather than to a sensible-looking number because on a thermal harvest a low
reading is a *cold* target, and dropping those is exactly backwards.

**Cell 1 installs nothing it does not have to, and stops when it does.** The
line used to be an unconditional `pip install --upgrade transformers>=5
accelerate huggingface_hub pillow tqdm` on every run, and the `pillow` in it
was the trap. Colab has already imported PIL by the time a cell runs, so pip
writes Pillow 12 to disk while `PIL._typing` stays in `sys.modules` from
Pillow 11 -- and the next new import, `PIL.ImageText`, reads the new file and
asks the old module for a name it does not have. What surfaces is
`ImportError: cannot import name '_Ink' from 'PIL._typing'` thirty frames
under `build_image_teacher`, blamed on SAM 3, which is not involved.

So: `pillow` is gone (transformers depends on it and Colab ships it), the
install runs only when `transformers` is missing or older than 5, and when it
does run the cell **stops** and asks for a session restart rather than
continuing into a half-old, half-new interpreter. The same three lines were in
15 and 18 and got the same fix.

**Cell 1 updates the clone rather than skipping it.** It used to clone only
when `/content/sam-dedection` was absent, which is right exactly once: a
runtime that has already run any of these notebooks keeps whatever it cloned
first, so a fix pushed since then is invisible and surfaces as an `ImportError`
on a name the repo does have. It now fetches and hard-resets to the branch
every run, and drops every already-imported `src.*` and `tools.*` module from
`sys.modules` so re-running the cell picks the new code up without a kernel
restart. The commit it landed on is printed.

**Neither stage is allowed to leave the other idle.** Two knobs, because the
harvest is two costs that used to take turns:

- **Unzipping** is not bandwidth, it is seeks. `tracked_ir` keeps a twentieth
  of a 15.7 GiB part, so the read is a few thousand random seeks over a Drive
  mount rather than one stream, and a FUSE seek is latency. `UNZIP_WORKERS`
  reads on that many threads, each with its own zip handle -- the only way to
  hide latency is to have more of it outstanding.
- **Decoding** used to run on the thread that then waited for the teacher. A
  1920x1080 JPEG is ~20 ms and a VTUAV frame carries **one box**, so a group of
  32 put ~0.6 s of decode in front of every batch with the card doing nothing.
  `READERS` decodes ahead on its own threads (cv2 releases the GIL, so this is
  real parallelism), `READ_AHEAD` bounds how many decoded frames may wait --
  6.2 MB each, and cell 1 prints what that comes to.

`cv2.setNumThreads(1)` goes with them: OpenCV parallelises `cvtColor`
internally, and eight readers each spawning a pool of its own is
oversubscription on a memory-bound op, not speed.

**No fallback teacher.** The cell that loads `facebook/sam3` used to catch a
failure and quietly continue on `facebook/sam2.1-hiera-large`. That is the one
error worth stopping for: the pools these four write are meant to be mixed into
one training set, and a pool whose masks came from a different teacher than its
neighbour's is a variable nobody chose and the run cannot see. `build_image_teacher`
already fails with the gated-repo instructions, so an unset `HF_TOKEN` says so
instead of costing a harvest that has to be thrown away. 18 made the same call
for the same reason.

**Neither split ships a mask.** VTUAV's drawn instance masks are the separate
*VIS* release -- 100 sequences, a different download -- and the other 400 carry
one `x y w h` per annotated frame and nothing else. That is the whole reason
these notebooks exist: the teacher is what turns the box into a mask.

**Why two notebooks rather than one with a mirror.** DroneVehicle earned its
single pass: 53 % of its thermal boxes are the same rectangle as an RGB one, so
a mask made from RGB pixels is the thermal frame's mask too. VTUAV is not like
that, and the numbers are what say so. Over the 3 750 annotated rows of
`train_ST_001`, `rgb.txt` and `ir.txt` agree exactly on **12.2 %** of rows, the
median centre offset is **8.4 px** (p90 29.4) against a median target of 77 px,
and neither file has a row the other lacks -- the target is one physical
object, annotated in both or in neither, so `*-only` is empty by construction.

So there is no mask to share and nothing to mirror: each modality is prompted
on its own frame with its own box. That makes the two harvests genuinely
independent, which is the point -- they run at the same time on two runtimes
and write to different pools and different Drive folders.

**What to put in Drive.** Only the **RGB-T** archives. Sampling
`train_ST_001.zip` from both of the project's Drive folders, every RGB member
matched on CRC32 and compressed size: the "RGB version" is the RGB-T archive
with `ir/` and `ir.txt` deleted. Both notebooks read the same RGB-T zips and
each unzips only its own half.

**Each arm extracts into its own `DATA_ROOT`.** `tracked_members` keeps both
`.txt` files but only one modality's frames, so two arms sharing an extraction
tree is a trap: the second one finds the first one's staging markers, skips
extracting, then finds `<sequence>/rgb.txt` present and `<sequence>/rgb/` empty
and quietly indexes nothing. The modality is in the path so that cannot happen,
whether the arms run on two runtimes or one after another on the same disk.

**Only a tenth of each archive is used.** Frame ids run 0..n-1 with no gaps and
each box file holds `ceil(n / 10)` lines, so line k is frame 10k and nine
frames in ten carry no label. `fetch_datasets.tracked_members` (reached through
`extract(frames="tracked_rgb" | "tracked_ir")`) keeps only the labelled frames
of one modality, which turns a 15.4 GiB archive into roughly 0.8 GiB on disk.

**The parts are alphabetical by sequence name**, so each carries two to four
object kinds: ST_001 is animal/bike/bus, ST_005 car/elebike, ST_008 pedestrian
(24 of 28 sequences), ST_011 car/pedestrian/truck. `ARCHIVES` ships with a
spread rather than a run, because consecutive parts make a two-category pool.
The long-term split is only four parts, so 24 and 25 take all of it and the
question does not arise.

**Cell 2 is a probe, and it is there because of LT.** The stride -- line k of
the box file is frame 10k -- was measured on `train_ST_001` and on nothing
else, and a stride assumed one too small does not fail loudly: it labels frame
9 with frame 10's box and mirrors a pool of quietly wrong masks to Drive. So
before a single byte is extracted, the probe opens each archive's central
directory (seconds, it is a seek to the end of the file), reads the kilobyte
of `.txt` per sequence, and prints frames, rows, the stride those two counts
actually imply, and the share of rows marking the target absent.
`boxes.annotated_stride` is the same function the extractor and the indexer
use, so what the probe prints is what the run will do -- and a sequence whose
counts imply no single stride is dropped by all three rather than guessed at.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, fields
from pathlib import Path

STAMP_VALUE = "unstamped"

CELLS: list[str] = []


@dataclass(frozen=True)
class Variant:
    key: str
    path: str
    modality: str            # the folder inside a sequence: rgb or ir
    pool: str
    data: str                # this arm's own extraction tree
    mirror: str              # where on Drive this arm's pool is zipped
    extract_mode: str        # `extract(frames=...)`, one modality's frames
    archives: str            # the parts of the tracking download to read


ST_ARCHIVES = '["train_ST_001.zip", "train_ST_005.zip",\n' \
              '               "train_ST_008.zip", "train_ST_011.zip"]'
LT_ARCHIVES = '["train_LT_001.zip", "train_LT_002.zip",\n' \
              '               "train_LT_003.zip", "train_LT_004.zip"]'

VARIANTS = {
    "rgb": Variant(key="rgb", path="notebooks/16_vtuav_rgb_pool.ipynb",
                   modality="rgb", pool="vtuav_rgb",
                   data="/content/data/VTUAV_rgb",
                   mirror="/content/drive/MyDrive/edgetam-pool/vtuav_rgb",
                   extract_mode="tracked_rgb", archives=ST_ARCHIVES),
    "ir": Variant(key="ir", path="notebooks/17_vtuav_thermal_pool.ipynb",
                  modality="ir", pool="vtuav_thermal",
                  data="/content/data/VTUAV_ir",
                  mirror="/content/drive/MyDrive/edgetam-pool/vtuav_thermal",
                  extract_mode="tracked_ir", archives=ST_ARCHIVES),
    "lt_rgb": Variant(key="lt_rgb", path="notebooks/24_vtuav_lt_rgb_pool.ipynb",
                      modality="rgb", pool="vtuav_lt_rgb",
                      data="/content/data/VTUAV_lt_rgb",
                      mirror="/content/drive/MyDrive/edgetam-pool/vtuav_lt_rgb",
                      extract_mode="tracked_rgb", archives=LT_ARCHIVES),
    "lt_ir": Variant(key="lt_ir", path="notebooks/25_vtuav_lt_thermal_pool.ipynb",
                     modality="ir", pool="vtuav_lt_thermal",
                     data="/content/data/VTUAV_lt_ir",
                     mirror="/content/drive/MyDrive/edgetam-pool/vtuav_lt_thermal",
                     extract_mode="tracked_ir", archives=LT_ARCHIVES),
}

V = VARIANTS[sys.argv[1]] if len(sys.argv) > 1 else VARIANTS["rgb"]


def code(text: str) -> None:
    CELLS.append(text.strip("\n"))


def resolve(text: str) -> str:
    for field in fields(Variant):
        text = text.replace("{{%s}}" % field.name, str(getattr(V, field.name)))
    return text.replace("{{STAMP}}", STAMP_VALUE)


# --------------------------------------------------------------------------
# 1. Settings, environment, and the batch this card can take
# --------------------------------------------------------------------------

code('''
DRIVE_DIR   = "/content/drive/MyDrive/VTUAV"
DATA_ROOT   = "{{data}}"
POOL_ROOT   = "/content/pool"
POOL        = "{{pool}}"
MODALITY    = "{{modality}}"
EXTRACT_MODE = "{{extract_mode}}"
MIRROR_DIR  = "{{mirror}}"
ARCHIVES    = {{archives}}
TEACHER     = "facebook/sam3"
DTYPE       = "bfloat16"
ZOOM        = 4.0
MIN_SIZE    = 128
MIN_LUMA    = 0.0
BATCH       = 0
FRAME_GROUP = 0
READERS     = 0
READ_AHEAD  = 0
UNZIP_WORKERS = 16
MAX_BOXES   = None
LIMIT       = None
BOX_IOU     = 0.5
REPO_URL    = "https://github.com/yigitkayabagci/sam-dedection.git"
BRANCH      = "claude/thermal-stage-b-training-43ktcl"
REPO_DIR    = "/content/sam-dedection"
NOTEBOOK = "{{path}}".split("/")[-1]
STAMP    = "{{STAMP}}"

import json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

if Path(REPO_DIR).exists():
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "--depth", "1",
                    "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "reset", "--hard", "FETCH_HEAD"],
                   check=True)
else:
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                    REPO_URL, REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
import importlib
for _stale in [_m for _m in list(sys.modules)
               if _m.split(".")[0] in ("src", "tools")]:
    del sys.modules[_stale]
importlib.invalidate_caches()
print("repo at", subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "--short",
                                 "HEAD"], capture_output=True,
                                text=True).stdout.strip())

_missing = []
try:
    import transformers as _transformers
    if int(_transformers.__version__.split(".")[0]) < 5:
        _missing.append("transformers>=5.0.0")
except Exception:
    _missing.append("transformers>=5.0.0")
for _name in ("accelerate", "huggingface_hub", "tqdm"):
    try:
        __import__(_name)
    except ImportError:
        _missing.append(_name)
if _missing:
    print("installing", _missing)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                    *_missing], check=False)
    raise SystemExit(
        "installed " + ", ".join(_missing) + " into a running kernel. "
        "Runtime > Restart session, then run this cell again. pip replaced "
        "files this kernel has already imported, and the half-old half-new "
        "import that follows fails somewhere unrelated -- PIL, torchvision -- "
        "rather than here.")

try:
    from google.colab import drive as _drive
    _drive.mount("/content/drive")
except Exception as _mount_error:
    print("no Colab Drive mount:", _mount_error)

try:
    from google.colab import userdata as _userdata
    _token = _userdata.get("HF_TOKEN")
    if _token:
        os.environ["HF_TOKEN"] = _token
except Exception as _token_error:
    print("no HF_TOKEN secret:", _token_error)

if os.environ.get("HF_TOKEN"):
    try:
        from huggingface_hub import login as _login
        _login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
    except Exception as _login_error:
        print("hf login skipped:", _login_error)

import torch, cv2
from src.training.pool import read_workers

cv2.setNumThreads(1)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

_device = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
VRAM = round(_device.total_memory / 2 ** 30, 1) if _device else 0.0
CORES = os.cpu_count() or 1
if BATCH <= 0:
    BATCH = max(4, int(VRAM * 0.8)) if VRAM else 4
if FRAME_GROUP <= 0:
    FRAME_GROUP = BATCH
READERS = read_workers(READERS)
if READ_AHEAD <= 0:
    READ_AHEAD = max(2 * FRAME_GROUP, READERS)

def progress(stream, total, desc):
    from tqdm.auto import tqdm
    return tqdm(stream, total=total, desc=desc)

_stamps = Path(REPO_DIR) / "notebooks" / ".stamps.json"
_want = json.loads(_stamps.read_text()).get(NOTEBOOK) if _stamps.is_file() else None
print(NOTEBOOK, STAMP, "| repo:", _want,
      "| OK" if _want == STAMP else "| STALE, re-open from the repo")
print(_device.name if _device else "no GPU", VRAM, "GiB",
      "| BATCH", BATCH, "| FRAME_GROUP", FRAME_GROUP, "| modality", MODALITY)
print(CORES, "cores | READERS", READERS, "| READ_AHEAD", READ_AHEAD,
      "|", round(READ_AHEAD * 1920 * 1080 * 3 / 2 ** 30, 2), "GiB of frames "
      "in flight | UNZIP_WORKERS", UNZIP_WORKERS)
''')


# --------------------------------------------------------------------------
# 2. What is in each archive, before 15 GiB of it is read
# --------------------------------------------------------------------------

code('''
import math
from src.training.boxes import annotated_stride

def absent_row(text):
    cells = text.replace(",", " ").split()
    if len(cells) < 4:
        return True
    try:
        x, y, w, h = (float(v) for v in cells[:4])
    except ValueError:
        return True
    return not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0

MISSING, PROBE = [], []
for _archive in ARCHIVES:
    _source = Path(DRIVE_DIR) / _archive
    if not _source.is_file():
        MISSING.append(_archive)
        continue
    _rows, _present, _gone = {}, {}, {}
    with zipfile.ZipFile(_source) as _zip:
        for _name in _zip.namelist():
            _parts = _name.split("/")
            if len(_parts) == 2 and _parts[1] == f"{MODALITY}.txt":
                _lines = [_l for _l in _zip.read(_name).decode(
                    "utf-8", "replace").splitlines() if _l.strip()]
                _rows[_parts[0]] = len(_lines)
                _gone[_parts[0]] = sum(1 for _l in _lines if absent_row(_l))
            elif len(_parts) == 3 and _parts[1] == MODALITY:
                if Path(_parts[-1]).stem.isdigit():
                    _present[_parts[0]] = _present.get(_parts[0], 0) + 1
    _strides, _unreadable = {}, []
    for _sequence, _lines in sorted(_rows.items()):
        try:
            _step = annotated_stride(_present.get(_sequence, 0), _lines)
        except ValueError as _mismatch:
            _unreadable.append(f"{_sequence}: {_mismatch}")
            continue
        _strides[_step] = _strides.get(_step, 0) + 1
    PROBE.append({"archive": _archive, "sequences": len(_rows),
                  "frames": sum(_present.values()), "rows": sum(_rows.values()),
                  "absent": sum(_gone.values()), "strides": _strides,
                  "unreadable": _unreadable,
                  "GiB": round(_source.stat().st_size / 2 ** 30, 1)})

print(f"{'archive':<22}{'seq':>5}{'frames':>10}{'rows':>8}{'absent':>9}"
      f"{'GiB':>7}  strides")
for _row in PROBE:
    _share = f"{_row['absent'] / max(_row['rows'], 1):.1%}"
    print(f"{_row['archive']:<22}{_row['sequences']:>5}{_row['frames']:>10}"
          f"{_row['rows']:>8}{_share:>9}{_row['GiB']:>7}  {_row['strides']}")
    for _line in _row["unreadable"]:
        print("   !! dropped --", _line)
if MISSING:
    print("\\nnot in", DRIVE_DIR, "--", MISSING)
KEEPS = sum(_r["rows"] for _r in PROBE)
PROMPTS = sum(_r["rows"] - _r["absent"] for _r in PROBE)
print(f"\\n{KEEPS} annotated {MODALITY} frame(s) to extract, {PROMPTS} of them "
      f"with a target to prompt; "
      f"{round(sum(_r['GiB'] for _r in PROBE), 1)} GiB read from Drive, "
      f"about {round(sum(_r['GiB'] for _r in PROBE) / 20, 1)} GiB on disk.")
assert PROBE, f"no archive of {ARCHIVES} is under {DRIVE_DIR} -- set DRIVE_DIR"
if any(set(_r["strides"]) - {10} or _r["unreadable"] for _r in PROBE):
    print("\\n!! a stride other than 10, or a sequence whose counts imply "
          "none. The extractor and the indexer derive it the same way this "
          "cell does, so the harvest follows what is printed above -- but "
          "read it before spending the GPU-hours.")
''')


# --------------------------------------------------------------------------
# 3. Unzip one modality's labelled frames, then index them
# --------------------------------------------------------------------------

code('''
from tools.fetch_datasets import extract

Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
_done = Path(DATA_ROOT) / "_staged"
_done.mkdir(parents=True, exist_ok=True)
for _archive in ARCHIVES:
    _source = Path(DRIVE_DIR) / _archive
    _marker = _done / _archive
    if _marker.exists():
        print("already staged:", _archive)
        continue
    if not _source.is_file():
        raise SystemExit(f"{_source} is not there -- set DRIVE_DIR in cell 1")
    print("staging", _archive,
          round(_source.stat().st_size / 2 ** 30, 2), "GiB from Drive")
    extract(_source, Path(DATA_ROOT), frames=EXTRACT_MODE,
            workers=UNZIP_WORKERS)
    _marker.write_text("ok")

from src.training import boxes as B

FRAMES = B.vtuav_frames(DATA_ROOT, modality=MODALITY)
SEQUENCES = sorted({f.key.split("/")[0] for f in FRAMES})
print(len(FRAMES), "labelled frames /", len(SEQUENCES), "sequences")
print(B.summarise_frames(FRAMES, POOL))
print(round(sum(p.stat().st_size for p in Path(DATA_ROOT).rglob("*")
                if p.is_file()) / 2 ** 30, 2), "GiB on disk")
''')


# --------------------------------------------------------------------------
# 4. Teacher, and four frames drawn before any of it runs
# --------------------------------------------------------------------------

code('''
import numpy as np, cv2
import matplotlib.pyplot as plt
from src.training.labels import Gates, build_image_teacher
from src.training.pool import label_boxes

TEACHER_MODEL = build_image_teacher(TEACHER, dtype=DTYPE)
GATES = Gates(box_iou=BOX_IOU)
print("teacher:", TEACHER)

_shown = [FRAMES[i] for i in
          np.linspace(0, len(FRAMES) - 1, 4).astype(int).tolist()]
_figure, _axes = plt.subplots(2, 2, figsize=(14, 9))
for _panel, _frame in zip(_axes.ravel(), _shown):
    _image = cv2.cvtColor(cv2.imread(str(_frame.image)), cv2.COLOR_BGR2RGB)
    _boxes, _keep = _frame.resolved(_image.shape[:2])
    _masks, _rows = label_boxes(_image, _boxes[_keep], TEACHER_MODEL,
                                gates=GATES, zoom=ZOOM, min_size=MIN_SIZE,
                                batch_size=BATCH)
    _x0, _y0, _x1, _y1 = (int(v) for v in _boxes[0])
    for _mask in _masks.values():
        _image[_mask] = (0.4 * _image[_mask]
                         + 0.6 * np.array([255, 40, 40])).astype(np.uint8)
    _pad = 200
    _panel.imshow(_image[max(_y0 - _pad, 0):_y1 + _pad,
                         max(_x0 - _pad, 0):_x1 + _pad])
    _panel.set_title(f"{_frame.key} {_frame.classes[0]} "
                     f"{len(_masks)}/{len(_rows)}")
    _panel.axis("off")
plt.tight_layout()
plt.show()
''')


# --------------------------------------------------------------------------
# 5. Harvest this modality, on its own boxes
# --------------------------------------------------------------------------

code('''
from src.training.pool import (label_pool, pool_report, summarise_luma,
                               summarise_pool, write_index)

def harvest(frames, dataset):
    global BATCH, FRAME_GROUP
    while True:
        try:
            return label_pool(frames, TEACHER_MODEL, POOL_ROOT, dataset=dataset,
                              prompt="self", gates=GATES, zoom=ZOOM,
                              min_size=MIN_SIZE, batch_size=BATCH,
                              frame_group=FRAME_GROUP, limit=LIMIT,
                              max_boxes=MAX_BOXES, min_luma=MIN_LUMA,
                              readers=READERS, read_ahead=READ_AHEAD,
                              progress=progress)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            BATCH = max(1, BATCH // 2)
            FRAME_GROUP = max(1, FRAME_GROUP // 2)
            print("out of memory, retrying at BATCH", BATCH,
                  "FRAME_GROUP", FRAME_GROUP)

REPORT = harvest(FRAMES, POOL)
print(json.dumps(REPORT, indent=1))
write_index(POOL_ROOT)
print(summarise_pool(pool_report(POOL_ROOT)))
print()
print(summarise_luma(POOL_ROOT))
''')


# --------------------------------------------------------------------------
# 6. The pool back to Drive
# --------------------------------------------------------------------------

code('''
Path(MIRROR_DIR).mkdir(parents=True, exist_ok=True)
_target = Path(MIRROR_DIR) / f"{POOL}.zip"
_files = [p for p in sorted((Path(POOL_ROOT) / POOL).rglob("*")) if p.is_file()]
with zipfile.ZipFile(_target, "w", zipfile.ZIP_STORED, allowZip64=True) as _zip:
    for _file in _files:
        _zip.write(_file, _file.relative_to(Path(POOL_ROOT)))
print(_target, len(_files), "files",
      round(_target.stat().st_size / 2 ** 20, 1), "MiB")

_index = Path(POOL_ROOT) / "pool_index.jsonl"
if _index.is_file():
    shutil.copy(_index, Path(MIRROR_DIR) / _index.name)
print("teacher:", TEACHER, "| modality:", MODALITY,
      "| accepted:", REPORT["accepted"], "of", REPORT["attempted"])
''')


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def stamp_for(variant: Variant) -> str:
    body = "\n".join(CELLS)
    body += "\n".join(str(getattr(variant, f.name)) for f in fields(variant))
    return hashlib.sha256(body.encode()).hexdigest()[:10]


def build() -> dict:
    global STAMP_VALUE
    STAMP_VALUE = stamp_for(V)
    return {
        "cells": [
            {"cell_type": "code", "metadata": {}, "outputs": [],
             "execution_count": None,
             "source": (resolve(text) + "\n").splitlines(keepends=True)}
            for text in CELLS
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100",
                      "machine_shape": "hm"},
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
        out.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
        stamps = repo / "notebooks" / ".stamps.json"
        known = json.loads(stamps.read_text()) if stamps.is_file() else {}
        known[out.name] = STAMP_VALUE
        stamps.write_text(json.dumps(known, indent=1, sort_keys=True) + "\n")
        print(f"wrote {out.relative_to(repo)} with {len(CELLS)} cells "
              f"[{STAMP_VALUE}]", file=sys.stderr)
    else:
        # Re-run per variant: the cells are emitted at import time against one
        # `V`, so a second pass in-process would append to the first.
        for key in VARIANTS:
            subprocess.run([sys.executable, __file__, key], check=True, cwd=repo)
