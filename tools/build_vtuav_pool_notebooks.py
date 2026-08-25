#!/usr/bin/env python3
"""Generate notebooks 16 and 17: VTUAV's two halves, two pools, two runtimes.

Same shape as 15 -- code cells only, no prose, no comments -- and the same
generator pattern as `build_pool_notebooks.py`: one file, two variants, so the
pair cannot drift apart by an edit that only landed on one of them.

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
    mirror: str              # where on Drive this arm's pool is zipped
    extract_mode: str        # `extract(frames=...)`, one modality's frames


VARIANTS = {
    "rgb": Variant(key="rgb", path="notebooks/16_vtuav_rgb_pool.ipynb",
                   modality="rgb", pool="vtuav_rgb",
                   mirror="/content/drive/MyDrive/edgetam-pool/vtuav_rgb",
                   extract_mode="tracked_rgb"),
    "ir": Variant(key="ir", path="notebooks/17_vtuav_thermal_pool.ipynb",
                  modality="ir", pool="vtuav_thermal",
                  mirror="/content/drive/MyDrive/edgetam-pool/vtuav_thermal",
                  extract_mode="tracked_ir"),
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
DATA_ROOT   = "/content/data/VTUAV_{{modality}}"
POOL_ROOT   = "/content/pool"
POOL        = "{{pool}}"
MODALITY    = "{{modality}}"
EXTRACT_MODE = "{{extract_mode}}"
MIRROR_DIR  = "{{mirror}}"
ARCHIVES    = ["train_ST_001.zip", "train_ST_005.zip",
               "train_ST_008.zip", "train_ST_011.zip"]
TEACHER     = "facebook/sam3"
FALLBACK    = "facebook/sam2.1-hiera-large"
DTYPE       = "bfloat16"
ZOOM        = 4.0
MIN_SIZE    = 128
BATCH       = 0
FRAME_GROUP = 0
MAX_BOXES   = None
LIMIT       = None
BOX_IOU     = 0.5
REPO_URL    = "https://github.com/yigitkayabagci/sam-dedection.git"
BRANCH      = "claude/aerial-rgb-thermal-data-qhjpxd"
REPO_DIR    = "/content/sam-dedection"
NOTEBOOK = "{{path}}".split("/")[-1]
STAMP    = "{{STAMP}}"

import json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

if not Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                    REPO_URL, REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                "transformers>=5.0.0", "accelerate", "huggingface_hub",
                "pillow", "tqdm"], check=False)

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

import torch

_device = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
VRAM = round(_device.total_memory / 2 ** 30, 1) if _device else 0.0
if BATCH <= 0:
    BATCH = max(4, int(VRAM * 0.8)) if VRAM else 4
if FRAME_GROUP <= 0:
    FRAME_GROUP = BATCH

def progress(stream, total, desc):
    from tqdm.auto import tqdm
    return tqdm(stream, total=total, desc=desc)

_stamps = Path(REPO_DIR) / "notebooks" / ".stamps.json"
_want = json.loads(_stamps.read_text()).get(NOTEBOOK) if _stamps.is_file() else None
print(NOTEBOOK, STAMP, "| repo:", _want,
      "| OK" if _want == STAMP else "| STALE, re-open from the repo")
print(_device.name if _device else "no GPU", VRAM, "GiB",
      "| BATCH", BATCH, "| FRAME_GROUP", FRAME_GROUP, "| modality", MODALITY)
''')


# --------------------------------------------------------------------------
# 2. Unzip one modality's labelled frames, then index them
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
    extract(_source, Path(DATA_ROOT), frames=EXTRACT_MODE)
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
# 3. Teacher, and four frames drawn before any of it runs
# --------------------------------------------------------------------------

code('''
import numpy as np, cv2
import matplotlib.pyplot as plt
from src.training.labels import Gates, build_image_teacher
from src.training.pool import label_boxes

try:
    TEACHER_USED = TEACHER
    TEACHER_MODEL = build_image_teacher(TEACHER, dtype=DTYPE)
except (Exception, SystemExit) as _teacher_error:
    print("falling back:", str(_teacher_error).splitlines()[0])
    TEACHER_USED = FALLBACK
    TEACHER_MODEL = build_image_teacher(FALLBACK, dtype=DTYPE)
GATES = Gates(box_iou=BOX_IOU)
print("teacher:", TEACHER_USED)

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
# 4. Harvest this modality, on its own boxes
# --------------------------------------------------------------------------

code('''
from src.training.pool import label_pool, pool_report, summarise_pool, write_index

def harvest(frames, dataset):
    global BATCH, FRAME_GROUP
    while True:
        try:
            return label_pool(frames, TEACHER_MODEL, POOL_ROOT, dataset=dataset,
                              prompt="self", gates=GATES, zoom=ZOOM,
                              min_size=MIN_SIZE, batch_size=BATCH,
                              frame_group=FRAME_GROUP, limit=LIMIT,
                              max_boxes=MAX_BOXES, progress=progress)
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
''')


# --------------------------------------------------------------------------
# 5. The pool back to Drive
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
print("teacher:", TEACHER_USED, "| modality:", MODALITY,
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
