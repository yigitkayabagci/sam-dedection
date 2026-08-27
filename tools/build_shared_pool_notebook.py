#!/usr/bin/env python3
"""Generate notebook 15: DroneVehicle's shared boxes, one teacher pass, two pools.

Its own generator rather than a third variant of `build_pool_notebooks.py`,
because the notebook it writes is a different kind of document. 13 and 14
explain themselves -- they carry the prose that argues the thermal route, the
class probes, the calibration table. 15 was asked for as the opposite: **no
prose and no comments, as few cells as possible**, because it is not a decision
to be read, it is a job to be run. Folding that shape into the templated
builder would mean a variant flag on every cell.

What the notebook does, in six cells:

  1. clone, install, mount Drive, size the batch to the card
  2. unzip DroneVehicle out of the user's own Drive, build the shared index
  3. load the teacher, draw two frames so the mirror is visible before it runs
  4. harvest the shared boxes: prompt on RGB, file the mask under both pools
  5. harvest the two *-only sets, each prompted on the half that can see them
  6. zip all four pools back to Drive

Four pools, because provenance has to stay readable: `dronevehicle_rgb` and
`dronevehicle_thermal` are the shared boxes (one mask, two files),
`dronevehicle_thermal_only` is the 33 383 vehicles the visible half never
annotates -- night, mostly -- and `dronevehicle_rgb_only` is the 3 797 the
thermal half misses. A thermal training run reads the second and third; merging
them into one directory would make a mirrored mask indistinguishable from a
thermal-prompted one.

Three decisions inside it are not arbitrary and are argued where they live
rather than in the notebook:

**Drive, not the Hub.** `fetch_datasets.py` pulls DroneVehicle from
`McCheng/DroneVehicle` at 8.88 GB over HTTPS. A copy already in the user's own
Drive is a mounted read with no download at all, which is why `DRIVE_DIR` is
the only data setting and it ships as a placeholder.

**SAM 3 by default.** `docs/mask_pool_plan.md` section 1 measured this: SAM 3
is ahead of SAM 2.1 on image PVS and is the pools' default teacher. It is a
gated repo, so the notebook falls back to `facebook/sam2.1-hiera-large` and
says which one it used -- and the report records the model id either way, so a
pool built on the fallback is not silently mistaken for one built on SAM 3.

**The *-only sets are prompted on their own modality, never on the twin.**
They are the boxes the other half does not annotate, usually because the target
is not visible there; `--prompt pair` on them would ask the teacher to segment
whatever happens to sit at those coordinates in a frame that shows nothing.

**`frame_group`, not just `batch_size`.** The shared subset carries a median of
4 boxes per frame. Batching within a frame on an 80 GB card is batching in
name only, so the notebook sizes `FRAME_GROUP` to fill `BATCH` from several
frames at once (`pool.label_many`). Both are auto-sized from the card's VRAM
and halved on an OOM, which resume makes free to retry.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

NOTEBOOK = "15_dronevehicle_shared_pool.ipynb"
STAMP_VALUE = "unstamped"

CELLS: list[str] = []


def code(text: str) -> None:
    CELLS.append(text.strip("\n"))


# --------------------------------------------------------------------------
# 1. Environment, settings, and the batch the card can take
# --------------------------------------------------------------------------

code('''
DRIVE_DIR   = "/content/drive/MyDrive/DroneVehicle"
DATA_ROOT   = "/content/data/DroneVehicle"
POOL_ROOT   = "/content/pool"
RGB_POOL    = "dronevehicle_rgb"
TIR_POOL    = "dronevehicle_thermal"
TIR_ONLY    = "dronevehicle_thermal_only"
RGB_ONLY    = "dronevehicle_rgb_only"
MIRROR_DIR  = "/content/drive/MyDrive/edgetam-pool/dronevehicle"
ARCHIVES    = ["train.zip"]
ONLY_DISTANCE = 40.0
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
NOTEBOOK = "15_dronevehicle_shared_pool.ipynb"
STAMP    = "{{STAMP}}"

import json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

if not Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                    REPO_URL, REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

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
    try:
        next(Path("/content/drive/MyDrive").iterdir(), None)
    except OSError as _stale_mount:
        print("stale Drive mount:", _stale_mount, "-- remounting")
        _drive.mount("/content/drive", force_remount=True)
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
    FRAME_GROUP = max(1, BATCH // 4)

def progress(stream, total, desc):
    from tqdm.auto import tqdm
    return tqdm(stream, total=total, desc=desc)

_stamps = Path(REPO_DIR) / "notebooks" / ".stamps.json"
_want = json.loads(_stamps.read_text()).get(NOTEBOOK) if _stamps.is_file() else None
print(NOTEBOOK, STAMP, "| repo:", _want,
      "| OK" if _want == STAMP else "| STALE, re-open from the repo")
print(_device.name if _device else "no GPU", VRAM, "GiB",
      "| BATCH", BATCH, "| FRAME_GROUP", FRAME_GROUP)
''')


# --------------------------------------------------------------------------
# 2. Stage the archives off Drive and build the shared-box index
# --------------------------------------------------------------------------

code('''
Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
for _archive in ARCHIVES:
    _source = Path(DRIVE_DIR) / _archive
    _marker = Path(DATA_ROOT) / _archive[:-4]
    if _marker.exists():
        print("already staged:", _marker)
        continue
    if not _source.is_file():
        raise SystemExit(f"{_source} is not there -- set DRIVE_DIR in cell 1")
    print("unzipping", _source, round(_source.stat().st_size / 2 ** 30, 2), "GiB")
    with zipfile.ZipFile(_source) as _zip:
        _zip.extractall(DATA_ROOT)

from src.training import boxes as B

FRAMES = B.dronevehicle_shared_frames(DATA_ROOT)
BOXES = sum(len(f.classes) for f in FRAMES)
print(len(FRAMES), "frames /", BOXES, "shared boxes /",
      round(BOXES / max(len(FRAMES), 1), 1), "per frame")
print(B.summarise_frames(FRAMES, "dronevehicle_shared"))
''')


# --------------------------------------------------------------------------
# 3. Teacher, and the same mask drawn on both halves before anything runs
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

def read(path, inset):
    _raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if _raw.ndim == 2:
        _raw = cv2.cvtColor(_raw, cv2.COLOR_GRAY2BGR)
    _raw = _raw[:, :, :3]
    if inset:
        _raw = _raw[inset:-inset, inset:-inset]
    return cv2.cvtColor(_raw, cv2.COLOR_BGR2RGB)

_shown = [f for f in FRAMES if 2 <= len(f.classes) <= 12][:2]
_figure, _axes = plt.subplots(len(_shown), 2, figsize=(13, 5 * len(_shown)))
_axes = np.atleast_2d(_axes)
for _row, _frame in enumerate(_shown):
    _rgb = read(_frame.image, _frame.inset)
    _tir = read(_frame.pair, _frame.inset)
    _boxes, _keep = _frame.resolved((_rgb.shape[0] + 2 * _frame.inset,
                                     _rgb.shape[1] + 2 * _frame.inset))
    _masks, _rows = label_boxes(_rgb, _boxes[_keep], TEACHER_MODEL, gates=GATES,
                                zoom=ZOOM, min_size=MIN_SIZE, batch_size=BATCH)
    _overlay = np.zeros(_rgb.shape[:2], bool)
    for _mask in _masks.values():
        _overlay |= _mask
    for _column, (_image, _title) in enumerate(
            ((_rgb, "rgb (prompted)"), (_tir, "thermal (mirrored)"))):
        _panel = _image.copy()
        _panel[_overlay] = (0.4 * _panel[_overlay] +
                            0.6 * np.array([255, 40, 40])).astype(np.uint8)
        _axes[_row, _column].imshow(_panel)
        _axes[_row, _column].set_title(f"{_frame.key} {_title}")
        _axes[_row, _column].axis("off")
    print(_frame.key, len(_masks), "/", len(_rows), "accepted")
plt.tight_layout()
plt.show()
''')


# --------------------------------------------------------------------------
# 4. Harvest -- one pass over the RGB half, filed under both pools
# --------------------------------------------------------------------------

code('''
from src.training.pool import label_pool, pool_report, summarise_pool, write_index

def harvest(frames, dataset, mirror=None):
    global BATCH, FRAME_GROUP
    while True:
        try:
            return label_pool(frames, TEACHER_MODEL, POOL_ROOT, dataset=dataset,
                              prompt="self", mirror=mirror, gates=GATES,
                              zoom=ZOOM, min_size=MIN_SIZE, batch_size=BATCH,
                              frame_group=FRAME_GROUP, limit=LIMIT,
                              max_boxes=MAX_BOXES, progress=progress)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            BATCH = max(1, BATCH // 2)
            FRAME_GROUP = max(1, FRAME_GROUP // 2)
            print("out of memory, retrying at BATCH", BATCH,
                  "FRAME_GROUP", FRAME_GROUP)

REPORT = harvest(FRAMES, RGB_POOL, mirror=TIR_POOL)
print(json.dumps(REPORT, indent=1))
write_index(POOL_ROOT)
print(summarise_pool(pool_report(POOL_ROOT)))
''')


# --------------------------------------------------------------------------
# 5. The other half: targets one modality annotates and the other never does
# --------------------------------------------------------------------------

code('''
ONLY_REPORTS = {}
for _modality, _pool in (("thermal", TIR_ONLY), ("rgb", RGB_ONLY)):
    _only = B.dronevehicle_only_frames(DATA_ROOT, modality=_modality,
                                       distance=ONLY_DISTANCE)
    print(_modality, "only:", len(_only), "frames /",
          sum(len(f.classes) for f in _only), "boxes")
    ONLY_REPORTS[_modality] = harvest(_only, _pool)
    print(json.dumps(ONLY_REPORTS[_modality], indent=1))

write_index(POOL_ROOT)
print(summarise_pool(pool_report(POOL_ROOT)))
''')


# --------------------------------------------------------------------------
# 6. All four pools back to Drive
# --------------------------------------------------------------------------

code('''
Path(MIRROR_DIR).mkdir(parents=True, exist_ok=True)
for _pool in (RGB_POOL, TIR_POOL, TIR_ONLY, RGB_ONLY):
    _target = Path(MIRROR_DIR) / f"{_pool}.zip"
    _files = [p for p in sorted((Path(POOL_ROOT) / _pool).rglob("*")) if p.is_file()]
    with zipfile.ZipFile(_target, "w", zipfile.ZIP_STORED, allowZip64=True) as _zip:
        for _file in _files:
            _zip.write(_file, _file.relative_to(Path(POOL_ROOT)))
    print(_target, len(_files), "files",
          round(_target.stat().st_size / 2 ** 20, 1), "MiB")

_index = Path(POOL_ROOT) / "pool_index.jsonl"
if _index.is_file():
    shutil.copy(_index, Path(MIRROR_DIR) / _index.name)
print("teacher:", TEACHER_USED,
      "| shared accepted:", REPORT["accepted"],
      "| mirrored frames:", REPORT["mirrored"],
      "| thermal-only:", ONLY_REPORTS["thermal"]["accepted"],
      "| rgb-only:", ONLY_REPORTS["rgb"]["accepted"])
''')


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def stamp() -> str:
    return hashlib.sha256("\n".join(CELLS).encode()).hexdigest()[:10]


def build() -> dict:
    global STAMP_VALUE
    STAMP_VALUE = stamp()
    return {
        "cells": [
            {"cell_type": "code", "metadata": {}, "outputs": [],
             "execution_count": None,
             "source": (text.replace("{{STAMP}}", STAMP_VALUE) + "\n")
                       .splitlines(keepends=True)}
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
    out = repo / "notebooks" / NOTEBOOK
    document = build()
    out.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n")

    stamps = repo / "notebooks" / ".stamps.json"
    known = json.loads(stamps.read_text()) if stamps.is_file() else {}
    known[NOTEBOOK] = STAMP_VALUE
    stamps.write_text(json.dumps(known, indent=1, sort_keys=True) + "\n")
    print(f"wrote notebooks/{NOTEBOOK} with {len(CELLS)} cells [{STAMP_VALUE}]",
          file=sys.stderr)
