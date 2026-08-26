#!/usr/bin/env python3
"""Generate notebook 18: Kust4K's drawn maps into two teacher-mask pools.

The dataset arrives differently from every other pool source. Kust4K ships no
box file at all -- it ships **real 9-class semantic maps** for 4 024 registered
640x512 RGB-T pairs -- so the prompts are the envelopes of its thing classes'
connected components (`boxes.kust4k_frames`), and the teacher's job is to turn
each of those envelopes into an instance mask.

Which makes the whole notebook one decision, taken twice:

**On the 2 864 frames both halves survive (71 %),** the teacher is prompted on
the RGB half once and the mask is filed under both pools. Kust4K's pairs are
geometrically registered, so the RGB mask *is* the thermal mask; a second
forward pass on the thermal frame would cost the same GPU-hour to answer a
question the registration already answered. This is `label_pool`'s `mirror`,
the same mechanism notebook 15 uses on DroneVehicle's shared boxes.

**On the 1 160 frames the dataset marks broken (29 %),** it is not. Those
frames have **one of the two modalities deliberately corrupted** to simulate a
sensor failure -- the file decodes fine, it is simply not a picture of the
scene -- and Kust4K's five `broken_in_*.txt` manifests do not record *which*
half broke. Mirroring there is a coin flip: half the time the mask is stamped
onto pixels of another scene entirely. So the notebook measures it
(`pool.boundary_agreement`: does this image's gradient rise on this map's
boundaries?), keeps the half that still describes the map, and harvests each
kept half on its own pixels with no mirror at all.

That measurement is what makes the 29 % worth having rather than worth
dropping. `src/training/aerial.py` drops them today, correctly, because
training on a frame whose modality is noise teaches noise. But an intact half
with a drawn map is a perfectly good prompt source, and there are roughly a
thousand of them in there.

Six settings and eight cells. What it writes to Drive:

    kust4k_rgb/           clean RGB masks, teacher prompted on RGB
    kust4k_thermal/       the same masks, mirrored onto the thermal half
    kust4k_broken_rgb/    broken frames whose RGB half survived
    kust4k_broken_thermal/ broken frames whose thermal half survived
    kust4k_prompts.jsonl  every box, its class, and the verdict it earned
    kust4k_agreement.json per-frame scores and the intact-half verdict

**SAM 3 with no fallback**, by request. `facebook/sam3` is a gated repo and
needs `transformers>=5`, so the teacher cell fails loudly with the two things
to fix rather than quietly harvesting 4 024 frames on a different model -- a
pool built on the wrong teacher is not visibly wrong anywhere downstream.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

NOTEBOOK = "18_kust4k_mask_pool.ipynb"
STAMP_VALUE = "unstamped"

CELLS: list[str] = []


def code(text: str) -> None:
    CELLS.append(text.strip("\n"))


# --------------------------------------------------------------------------
# 1. Settings, environment, and the batch this card can take
# --------------------------------------------------------------------------

code('''
DRIVE_DIR   = "/content/drive/MyDrive/datasets/kust4k"
DATA_ROOT   = "/content/data/Kust4K"
POOL_ROOT   = "/content/pool"
MIRROR_DIR  = "/content/drive/MyDrive/edgetam-pool/kust4k"
CLEAN_RGB   = "kust4k_rgb"
CLEAN_TIR   = "kust4k_thermal"
BROKEN_RGB  = "kust4k_broken_rgb"
BROKEN_TIR  = "kust4k_broken_thermal"
TEACHER     = "facebook/sam3"        # no fallback, by request
DTYPE       = "bfloat16"
ZOOM        = 4.0
MIN_SIZE    = 128
BATCH       = 0                      # 0 = size it from the card
FRAME_GROUP = 0
MAX_BOXES   = 64
LIMIT       = None
BOX_IOU     = 0.5
CAL_FRAMES  = 120                    # frames in the route measurement
QUANTILE    = 0.02                   # clean-score floor the broken half must clear
DROP_CORRUPT = True                  # False harvests both halves regardless
EXPORT_PNG  = True                   # instance-id PNGs beside the run-length stores
REPO_URL    = "https://github.com/yigitkayabagci/sam-dedection.git"
BRANCH      = "claude/aerial-rgb-thermal-data-qhjpxd"
REPO_DIR    = "/content/sam-dedection"
NOTEBOOK = "18_kust4k_mask_pool.ipynb"
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
# 2. Stage Kust4K off Drive, however it was uploaded
# --------------------------------------------------------------------------

code('''
# Whatever shape the download left it in. Three of them break the obvious
# loop over `*.zip`: figshare's *Download all* returns **one archive named
# after the article id** holding the three real ones, so routing on the outer
# name skips everything silently; all three inner archives are **flat** and
# carry the same 4 024 filenames, so extracting them side by side would leave
# two thirds of the data overwritten; and a file downloaded by hand is often
# renamed to something that names no half at all, which is why
# `route_by_pixels` decodes a few members and decides from the values.
from tools.fetch_datasets import RECIPES, fetch_extra, stage_rgbt_archives

_drive_root = Path(DRIVE_DIR)
if not _drive_root.is_dir():
    raise SystemExit(f"{DRIVE_DIR} is not there -- set DRIVE_DIR in cell 1")
print("what is in", DRIVE_DIR)
for _entry in sorted(_drive_root.rglob("*"))[:40]:
    if _entry.is_file():
        print(f"  {_entry.relative_to(_drive_root)} "
              f"{round(_entry.stat().st_size / 2 ** 20, 1)} MiB")
    else:
        print(f"  {_entry.relative_to(_drive_root)}/")
print()

STAGED = stage_rgbt_archives(_drive_root, DATA_ROOT)
ON_DISK = {_route: len(list((Path(DATA_ROOT) / _route).glob("*.png")))
           for _route in ("tir", "rgb", "label")}
print(ON_DISK)
if not all(ON_DISK.values()):
    # Failing here rather than three cells later, where the same problem
    # arrives as "no image/mask pairs" and looks like a bad glob.
    raise SystemExit(
        f"{[k for k, v in ON_DISK.items() if not v]} came out empty. Kust4K "
        f"needs all three -- TIR, RGB and Seg_annos, 4 024 PNGs each. The "
        f"listing above is what {DRIVE_DIR} actually holds.")

# The five broken-frame manifests. `aerial.excluded_keys` globs DATA_ROOT and
# does **not** recurse, so they have to sit at the top; a manifest one folder
# down reads as no manifest at all and all 4 024 frames look clean.
for _name, _source in RECIPES["kust4k"].extras:
    _target = Path(DATA_ROOT) / _name
    if not _target.is_file():
        print("fetching manifest", _name)
        fetch_extra(_source, _target)

from src.training.aerial import SPECS, describe_layout, excluded_keys

SPEC = SPECS["kust4k"]
print(describe_layout(DATA_ROOT))
print()
print(len(excluded_keys(Path(DATA_ROOT), SPEC)), "frames named broken by the "
      "manifests (the published number is 1160)")
''')


# --------------------------------------------------------------------------
# 3. Probe the palette and the pairing before spending anything
# --------------------------------------------------------------------------

code('''
# Two things that have to hold before a GPU is worth booking, and both have
# burned this repo before. The palette: Kust4K's ids were once written from
# the paper and were wrong from id 3 on, which made `things` select **tree**
# and miss **motorcycle**. And the pair geometry: `mirror` stamps an RGB mask
# onto the thermal frame, which is only meaningful if the two are the same
# size -- `label_pool` counts a mismatch rather than writing it, so a silent
# zero here would look like a successful run that produced no thermal pool.
import numpy as np
from src.training.aerial import list_pairs
from tools.fetch_aerial import describe

print(describe(Path(DATA_ROOT), SPEC, "thermal", sample=60))

PAIRS = list_pairs(DATA_ROOT, SPEC)
print("\\n", len(PAIRS), "registered pairs the reader can see (clean only)")

import cv2

_sizes = {}
for _thermal, _rgb in PAIRS[:40]:
    _key = (cv2.imread(str(_thermal), cv2.IMREAD_UNCHANGED).shape[:2],
            cv2.imread(str(_rgb), cv2.IMREAD_UNCHANGED).shape[:2])
    _sizes[_key] = _sizes.get(_key, 0) + 1
for (_tir_shape, _rgb_shape), _count in sorted(_sizes.items(), key=lambda kv: -kv[1]):
    print(f"  thermal {_tir_shape}  rgb {_rgb_shape}  x{_count}",
          "<- mirror works" if _tir_shape == _rgb_shape else "<- MIRROR WILL SKIP")
''')


# --------------------------------------------------------------------------
# 4. Index: boxes drawn out of the maps, clean and broken kept apart
# --------------------------------------------------------------------------

code('''
# No box file exists. `kust4k_frames` decodes every semantic map and takes the
# tight envelope of each thing-class component -- the same `decompose` and the
# same `InstanceGates` stage B trains on, so a prompt here and a target there
# are the same object. One pass over 4 024 PNGs, about a minute.
from src.training import boxes as B

CLEAN = B.kust4k_frames(DATA_ROOT, modality="rgb", group="clean",
                        progress=progress)
BROKEN_RGB_FRAMES = B.kust4k_frames(DATA_ROOT, modality="rgb", group="broken",
                                    progress=progress)
BROKEN_TIR_FRAMES = B.kust4k_frames(DATA_ROOT, modality="thermal",
                                    group="broken", progress=progress)

print(B.summarise_frames(CLEAN, "kust4k clean (rgb half, thermal paired)"))
print()
print(len(BROKEN_RGB_FRAMES), "broken frames /",
      sum(len(f.classes) for f in BROKEN_RGB_FRAMES), "boxes -- one of the two "
      "halves in each is corrupt, and the manifests do not say which")
''')


# --------------------------------------------------------------------------
# 5. The teacher: SAM 3, and nothing else
# --------------------------------------------------------------------------

code('''
# Deliberately no fallback. `facebook/sam3` is gated and wants
# transformers>=5, and both failures are fixable in a minute -- whereas a pool
# silently built on a different teacher is not visibly wrong anywhere
# downstream. The record beside every store writes the model id either way.
import matplotlib.pyplot as plt
from src.training.labels import Gates, build_image_teacher
from src.training.pool import label_boxes

try:
    TEACHER_MODEL = build_image_teacher(TEACHER, dtype=DTYPE)
except (Exception, SystemExit) as _teacher_error:
    raise SystemExit(
        f"{TEACHER} did not load: {str(_teacher_error).splitlines()[0]}\\n"
        f"  1. accept the licence at https://huggingface.co/{TEACHER}\\n"
        f"  2. put the token in a Colab secret named HF_TOKEN, then re-run "
        f"cell 1\\n"
        f"  3. SAM 3 needs transformers>=5.0.0 -- restart the runtime if the "
        f"install in cell 1 upgraded it under a running kernel")
GATES = Gates(box_iou=BOX_IOU)
print("teacher:", TEACHER_MODEL.model_id)

def read_rgb(path):
    _raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if _raw.ndim == 2:
        _raw = cv2.cvtColor(_raw, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(_raw[:, :, :3], cv2.COLOR_BGR2RGB)

# The mirror, drawn before it runs: one mask, prompted on RGB, shown on both.
_shown = [f for f in CLEAN if 3 <= len(f.classes) <= 15][:2]
_figure, _axes = plt.subplots(len(_shown), 2, figsize=(13, 5 * len(_shown)))
_axes = np.atleast_2d(_axes)
for _row, _frame in enumerate(_shown):
    _rgb = read_rgb(_frame.image)
    _tir = read_rgb(_frame.pair)
    _boxes, _keep = _frame.resolved(_rgb.shape[:2])
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
# 6. Is the mirror earned? Measured against the drawn masks, on this download
# --------------------------------------------------------------------------

code('''
# The one thing Kust4K can do that no box dataset can: score the teacher
# against masks a human actually drew. `prompt="self"` reads the thermal
# frame; `prompt="pair"` reads its registered RGB twin. Both are scored on the
# **thermal** drawn mask, so the columns answer exactly the question the
# harvest turns on -- and per size bucket, because a route that wins on
# average can lose on the 20-pixel vehicles this project lives on.
from src.training.pool import calibrate_spec, calibration_table

ROUTES = {}
for _route in ("self", "pair"):
    ROUTES[_route] = calibrate_spec(
        DATA_ROOT, SPEC, TEACHER_MODEL, modality="thermal", prompt=_route,
        limit_frames=CAL_FRAMES, zoom=ZOOM, min_size=MIN_SIZE,
        batch_size=BATCH, progress=progress)
print(calibration_table({"thermal prompt": ROUTES["self"],
                         "rgb prompt (mirrored)": ROUTES["pair"]}))
''')


# --------------------------------------------------------------------------
# 7. Harvest the clean 71 %: one RGB pass, filed under both pools
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

CLEAN_REPORT = harvest(CLEAN, CLEAN_RGB, mirror=CLEAN_TIR)
print(json.dumps(CLEAN_REPORT, indent=1))
if CLEAN_REPORT.get("mirror_mismatch"):
    print("\\n!!", CLEAN_REPORT["mirror_mismatch"], "frames had halves of "
          "different sizes and got no thermal mask -- see the size table in "
          "cell 3")
write_index(POOL_ROOT)
print(summarise_pool(pool_report(POOL_ROOT)))
''')


# --------------------------------------------------------------------------
# 8. The broken 29 %: measure which half is real, harvest that one alone
# --------------------------------------------------------------------------

code('''
# `boundary_agreement` asks one question per image: does this picture's
# gradient rise where this map says an object ends? A corrupt half is a
# picture of another scene, so it answers about 1.0 -- no relationship -- and
# the intact half answers well above it. The floor is a low quantile of what
# the *clean* frames of this same download score, per modality, because
# thermal edges are not RGB edges and one constant for both would condemn
# every thermal half.
from src.training.pool import (agreement_table, intact_modalities,
                               modality_agreement)

AGREEMENT = modality_agreement(DATA_ROOT, SPEC, progress=progress)
VERDICTS = intact_modalities(AGREEMENT, quantile=QUANTILE)
print(agreement_table(AGREEMENT, VERDICTS))

def surviving(frames, modality):
    if not DROP_CORRUPT:
        return frames
    return [f for f in frames if modality in VERDICTS.get(f.key, ())]

BROKEN_REPORTS = {}
for _modality, _frames, _pool in (("rgb", BROKEN_RGB_FRAMES, BROKEN_RGB),
                                  ("thermal", BROKEN_TIR_FRAMES, BROKEN_TIR)):
    _kept = surviving(_frames, _modality)
    print(f"\\n{_modality}: {len(_kept)}/{len(_frames)} broken frames keep "
          f"this half")
    if not _kept:
        continue
    # No mirror on this group, ever: the other half is the one under suspicion.
    BROKEN_REPORTS[_modality] = harvest(_kept, _pool)
    print(json.dumps(BROKEN_REPORTS[_modality], indent=1))

write_index(POOL_ROOT)
print(summarise_pool(pool_report(POOL_ROOT)))
''')


# --------------------------------------------------------------------------
# 9. Prompts, masks and the measurements, back to Drive
# --------------------------------------------------------------------------

code('''
# Three kinds of output, and they are not interchangeable. The `.npz` stores
# are what training reads (run-length, ~100 bytes an instance). The PNGs are
# for looking at -- one uint16 map per frame, pixel value = the box's row in
# `record.json`, 0 = nothing. And the prompts file is every box that was
# *offered*, including the ones a gate rejected, because "the teacher was
# asked and refused" is a different fact from "nobody asked".
from src.training.labels import MASK_STORE, open_masks

Path(MIRROR_DIR).mkdir(parents=True, exist_ok=True)
POOLS = [CLEAN_RGB, CLEAN_TIR, BROKEN_RGB, BROKEN_TIR]

if EXPORT_PNG:
    for _pool in POOLS:
        for _store in sorted((Path(POOL_ROOT) / _pool).rglob(MASK_STORE)):
            _masks = open_masks(_store)
            _canvas = np.zeros(_masks.shape, dtype=np.uint16)
            for _index in sorted(_masks):
                _canvas[_masks[_index]] = _index + 1
            cv2.imwrite(str(_store.parent / "instances.png"), _canvas)

PROMPTS = Path(POOL_ROOT) / "kust4k_prompts.jsonl"
with PROMPTS.open("w") as _handle:
    for _record in sorted(Path(POOL_ROOT).rglob("record.json")):
        _row = json.loads(_record.read_text())
        if not _row["dataset"].startswith("kust4k"):
            continue
        for _instance in _row["instances"]:
            _handle.write(json.dumps({
                "pool": _row["dataset"], "frame": _row["key"],
                "image": _row["image"], "box": _instance["box"],
                "class": _instance["class"], "i": _instance["i"],
                "teacher": _row["teacher"], "prompt": _row["prompt"],
                "verdict": _instance["verdict"]}) + "\\n")

(Path(POOL_ROOT) / "kust4k_agreement.json").write_text(json.dumps(
    {"quantile": QUANTILE, "teacher": TEACHER_MODEL.model_id,
     "scores": AGREEMENT,
     "intact": {k: list(v) for k, v in VERDICTS.items()}}, indent=1) + "\\n")

for _pool in POOLS:
    _folder = Path(POOL_ROOT) / _pool
    if not _folder.is_dir():
        continue
    _target = Path(MIRROR_DIR) / f"{_pool}.zip"
    _files = [p for p in sorted(_folder.rglob("*")) if p.is_file()]
    with zipfile.ZipFile(_target, "w", zipfile.ZIP_STORED, allowZip64=True) as _zip:
        for _file in _files:
            _zip.write(_file, _file.relative_to(Path(POOL_ROOT)))
    print(_target, len(_files), "files",
          round(_target.stat().st_size / 2 ** 20, 1), "MiB")

for _name in ("pool_index.jsonl", "kust4k_prompts.jsonl", "kust4k_agreement.json"):
    _side = Path(POOL_ROOT) / _name
    if _side.is_file():
        shutil.copy(_side, Path(MIRROR_DIR) / _name)

print("\\nteacher:", TEACHER_MODEL.model_id,
      "| clean accepted:", CLEAN_REPORT["accepted"],
      "| mirrored:", CLEAN_REPORT.get("mirrored"),
      "| broken rgb:", BROKEN_REPORTS.get("rgb", {}).get("accepted", 0),
      "| broken thermal:", BROKEN_REPORTS.get("thermal", {}).get("accepted", 0))
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
