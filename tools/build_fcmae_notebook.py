#!/usr/bin/env python3
"""Generate notebook 36: FCMAE pretraining, the ConvNeXt V2 way, on RepViT.

Stage A with no labels at all. `src/training/fcmae.py` holds the method and the
paper's numbers; `tools/pretrain_fcmae.py` is the run; this is the notebook that
puts a Colab session around it and, before spending a GPU hour, *shows* the two
things the method rests on:

* that a masked patch cannot reach a visible one through the real trunk -- cell
  4 measures it on RepViT rather than trusting the unit test's toy stack;
* that GRN was actually inserted, and where -- cell 5 prints the sites and
  refuses to continue at zero, because a run with half the co-design missing is
  the one the paper says buys nothing.

Two arms, `ARM = "grn"` and `ARM = "plain"`, landing in separate folders. Both
are read against stage B from stock: a pretraining stage that does not beat
doing nothing cost GPU hours and bought nothing.

    36_pretrain_fcmae_thermal.ipynb

Cells only, no markdown, no comments -- the shape 15-21 were asked for.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

BRANCH = "claude/thermal-stage-b-training-43ktcl"
NOTEBOOK = "36_pretrain_fcmae_thermal.ipynb"

CELLS: list[str] = []


def code(text: str) -> None:
    CELLS.append(text.strip("\n"))


# --------------------------------------------------------------------------
# 1. Settings
# --------------------------------------------------------------------------
#
# `ARM` is the experiment, not a convenience. The paper's ablation reads
# V1 + supervised 83.8, V1 + FCMAE 83.7, V2 + FCMAE 84.6 -- masking alone does
# not beat supervised training and the co-design does. Running both arms here
# is that claim, re-tested on a different convnet in a different domain.
#
# FRAME_ROOTS are walked recursively for images and nothing else is read: no
# record, no mask, no gate. That is the whole reason this stage has more data
# than stage B -- a frame whose teacher mask failed MIN_BOX_IOU is still a
# frame, and stage A wants it.

code('''
ARM         = "grn"
RUN         = f"fcmae_{ARM}"
MIRROR_DIR  = f"/content/drive/MyDrive/edgetam-stage-a/{RUN}"

FRAME_ROOTS = ["/content/data", "/content/pool"]
FRAME_LIMIT = 0
MODALITY    = "thermal"

DRIVE_POOLS = "/content/drive/MyDrive/edgetam-pool"
POOL_ROOT   = "/content/pool"
DATA_ROOT   = "/content/data"
WORK        = f"/content/work_{RUN}"
REPO_DIR    = "/content/sam-dedection"

SIZE        = 512
PATCH       = 32
MASK_RATIO  = 0.6
DECODER_DIM = 512
NORM_FLOOR  = 1e-3

EPOCHS      = 20
STEPS       = 1000
BATCH       = 32
BASE_LR     = 1.5e-4
WARMUP      = 500
WEIGHT_DECAY = 0.05
EMA_DECAY   = 0.999
WORKERS     = 8
DEPTH       = 2
SEED        = 0

BASE_CHECKPOINT = ""
CHECKPOINT  = f"{REPO_DIR}/checkpoints/edgetam_{RUN}_{SIZE}.pt"
NOTEBOOK = "{{NOTEBOOK}}"
STAMP    = "{{STAMP}}"
BRANCH   = "{{BRANCH}}"
''')


# --------------------------------------------------------------------------
# 2. The repo, the mount, the disk
# --------------------------------------------------------------------------

code('''
import json, os, shutil, subprocess, sys, time
from pathlib import Path

assert ARM in ("grn", "plain"), f"ARM is grn or plain, not {ARM!r}"

try:
    from google.colab import drive as _drive
    _drive.mount("/content/drive")
    try:
        next(Path("/content/drive/MyDrive").iterdir(), None)
    except OSError as _stale:
        print("stale Drive mount:", _stale, "-- remounting")
        _drive.mount("/content/drive", force_remount=True)
except Exception as _mount_error:
    print("no Colab Drive mount:", _mount_error)

if not Path(REPO_DIR).is_dir():
    subprocess.run(["git", "clone", "--branch", BRANCH, "--depth", "1",
                    "https://github.com/yigitkayabagci/sam-dedection.git",
                    REPO_DIR], check=True)
os.chdir(REPO_DIR)
subprocess.run(["git", "fetch", "origin", BRANCH], check=False)
subprocess.run(["git", "checkout", BRANCH], check=False)
subprocess.run(["git", "pull", "origin", BRANCH], check=False)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

_stamps = json.loads(Path("notebooks/.stamps.json").read_text())
if _stamps.get(NOTEBOOK) != STAMP:
    print(f"!! this notebook is {STAMP} and the branch says "
          f"{_stamps.get(NOTEBOOK)}. Re-generate it before trusting a run.")

for _dir in (POOL_ROOT, DATA_ROOT, WORK, MIRROR_DIR,
             str(Path(REPO_DIR) / "checkpoints")):
    Path(_dir).mkdir(parents=True, exist_ok=True)

if not Path("third_party/EdgeTAM/checkpoints/edgetam.pt").is_file():
    subprocess.run(["bash", "scripts/setup_edgetam.sh"], check=True)

import torch
_props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
print(f"\\n{NOTEBOOK} {STAMP} | arm {ARM} | run {RUN}")
print(_props.name if _props else "no GPU",
      f"{_props.total_memory / 2**30:.0f} GiB" if _props else "")
print(f"free disk {shutil.disk_usage('/content').free / 2**30:.1f} GiB")
''')


# --------------------------------------------------------------------------
# 3. The frames
# --------------------------------------------------------------------------
#
# The census is the point of this cell, not the extraction. Stage A's argument
# is that it has an order of magnitude more data than stage B, and a number is
# the only thing that settles whether that is true on this Drive today.

code('''
from tools.pretrain_fcmae import describe, find_frames

for _zip in sorted(Path(DRIVE_POOLS).glob("*.zip")) if Path(DRIVE_POOLS).is_dir() else []:
    if _zip.stat().st_size / 2**20 > 2048:
        print(f"skipping {_zip.name} ({_zip.stat().st_size / 2**30:.1f} GiB) "
              f"-- name it in FRAME_ROOTS already extracted instead")
        continue
    _target = Path(POOL_ROOT) / _zip.stem
    if not _target.is_dir():
        shutil.unpack_archive(str(_zip), str(_target))
        print(f"extracted {_zip.name} -> {_target}")

FRAMES, CENSUS = find_frames(FRAME_ROOTS, FRAME_LIMIT, MODALITY)
print()
print(describe(CENSUS, MODALITY))
print(f"\\n{len(FRAMES)} {MODALITY} frames under {FRAME_ROOTS}")
assert len(FRAMES) >= BATCH * 10, (
    f"only {len(FRAMES)} frames. Stage A reads no labels, so its whole "
    f"argument is volume: point FRAME_ROOTS at the extracted image trees.")

print("\\nstage B trains on the gated mask pools; this reads every frame of "
      "the wanted sensor, gate or no gate. The rgb column above is what a "
      "walk with no modality filter would have pretrained on as grey.")
''')


# --------------------------------------------------------------------------
# 4. The leak, measured on the real trunk
# --------------------------------------------------------------------------
#
# `tests/test_fcmae.py` proves this on a toy convolutional stack, which is the
# right place for a unit test and the wrong place to stop: the property has to
# hold for RepViT, whose depthwise kernels and strides are not the toy's. So it
# is measured here, on the model this run is about to train, and printed.

code('''
from src.training.fcmae import (FCMAEConfig, expand_mask, masked_convolutions,
                                patch_mask)
from src.training.aerial import load_image, normalise
from src.training.distill import encoder_features
from tools.train_encoder import build_model

CONFIG = FCMAEConfig(image_size=SIZE, patch=PATCH, mask_ratio=MASK_RATIO,
                     decoder_dim=DECODER_DIM, norm_floor=NORM_FLOOR)
BASE_CKPT = BASE_CHECKPOINT or "third_party/EdgeTAM/checkpoints/edgetam.pt"
MODEL = build_model(SIZE, BASE_CKPT, "cuda")

_image = normalise(load_image(FRAMES[0], (0, 0), (0, 0), SIZE)[None], "cuda")
_mask = patch_mask(1, SIZE, CONFIG, torch.Generator().manual_seed(0)).cuda()
print(f"mask: {int(_mask.sum())}/{_mask.numel()} patches removed "
      f"({float(_mask.float().mean()):.0%}), {PATCH}px each")

def _features(images):
    with torch.no_grad(), masked_convolutions(MODEL.image_encoder, _mask) as _m:
        out = encoder_features(MODEL, images)
    return out, _m.touched

_first, _touched = _features(_image)
_scrambled = _image.clone()
_grid = expand_mask(_mask, SIZE, SIZE).bool()
_scrambled[_grid.expand_as(_scrambled)] = torch.randn(
    int(_grid.sum()) * 3, device=_image.device) * 5
_second, _ = _features(_scrambled)
_drift = float((_first - _second).abs().max())

with torch.no_grad():
    _plain = encoder_features(MODEL, _image)
    _plain_other = encoder_features(MODEL, _scrambled)
print(f"convolutions masked: {_touched}")
print(f"rewriting every masked pixel moves the features by {_drift:.3e} "
      f"(masked) and {float((_plain - _plain_other).abs().max()):.3e} (unmasked)")
assert _drift < 1e-4, (
    f"a masked patch reached a visible position: {_drift:.3e}. The encoder is "
    f"seeing what it is supposed to reconstruct, which is the paper's 79.3% "
    f"row rather than its 83.7% one.")
print("no leak: the trunk cannot see what it is being asked to invent")
''')


# --------------------------------------------------------------------------
# 5. GRN, and where it landed
# --------------------------------------------------------------------------
#
# The discovery pass is by shape, because RepViT's FFN is not called `mlp`. A
# silent zero would be a run with half the method missing, so this refuses.

code('''
from src.training.fcmae import GRN, insert_grn

REPORT = {"inserted": 0, "channels": {}, "parameters": 0}
if ARM == "grn":
    REPORT = insert_grn(MODEL.image_encoder)
    MODEL.cuda()
    assert REPORT["inserted"], (
        "GRN found no dimension-expanding layer in this trunk. That is half "
        "the co-design missing and the paper's own table says the other half "
        "alone buys nothing -- fix the discovery before spending the hours.")
    print(f"GRN at {REPORT['inserted']} site(s), "
          f"{REPORT['parameters']} new parameters:")
    for _name, _width in list(REPORT["channels"].items())[:12]:
        print(f"  {_name:<52} {_width:>5} channels")
    _grn = [m for m in MODEL.image_encoder.modules() if isinstance(m, GRN)]
    _zero = all(float(m.gamma.abs().max()) == 0.0 for m in _grn)
    print(f"\\nall {len(_grn)} start as the identity: {_zero} -- so inserting "
          f"them has not changed what this checkpoint does yet")
else:
    print("arm 'plain': no GRN. This is the ablation the paper predicts buys "
          "nothing over supervised training (V1 + FCMAE 83.7 vs 83.8).")
''')


# --------------------------------------------------------------------------
# 6. The run
# --------------------------------------------------------------------------

code('''
from tools.pretrain_fcmae import main as fcmae_main

_started = time.time()
_argv = []
for _root in FRAME_ROOTS:
    _argv += ["--frames", _root]
_argv += ["--base", BASE_CKPT, "--out", CHECKPOINT, "--mirror", MIRROR_DIR,
          "--json", str(Path(WORK) / "fcmae.json"),
          "--size", str(SIZE), "--patch", str(PATCH),
          "--mask-ratio", str(MASK_RATIO), "--decoder-dim", str(DECODER_DIM),
          "--norm-floor", str(NORM_FLOOR),
          "--epochs", str(EPOCHS), "--steps", str(STEPS), "--batch", str(BATCH),
          "--lr", str(BASE_LR), "--warmup", str(WARMUP),
          "--weight-decay", str(WEIGHT_DECAY), "--ema-decay", str(EMA_DECAY),
          "--workers", str(WORKERS), "--depth", str(DEPTH),
          "--seed", str(SEED), "--modality", MODALITY, "--device", "cuda",
          "--grn" if ARM == "grn" else "--no-grn"]
if FRAME_LIMIT:
    _argv += ["--limit", str(FRAME_LIMIT)]

del MODEL
torch.cuda.empty_cache()
assert fcmae_main(_argv) == 0
RUN_LOG = json.loads((Path(WORK) / "fcmae.json").read_text())
print(f"wall clock {(time.time() - _started) / 60:.0f} min")
''')


# --------------------------------------------------------------------------
# 7. Did the reconstruction actually learn anything
# --------------------------------------------------------------------------
#
# A falling reconstruction loss is not the result -- the result is stage B --
# but a *flat* one says the run failed before stage B is asked to prove it, and
# that is worth knowing now rather than after another notebook.

code('''
_history = RUN_LOG["history"]
_first, _last = _history[0]["mse"], _history[-1]["mse"]
print(f"reconstruction mse {_first:.5f} -> {_last:.5f} over "
      f"{len(_history)} epochs ({100 * (1 - _last / max(_first, 1e-9)):.0f}% down)")
for _row in _history:
    print(f"  epoch {_row['epoch']:>3}  mse {_row['mse']:.5f}  lr {_row['lr']:.2e}")
if _last >= _first * 0.95:
    print("\\n!! the loss barely moved. Before running stage B on this "
          "checkpoint, check the mask ratio and the learning rate: a flat "
          "reconstruction loss is a pretraining stage that did nothing.")
print(f"\\nfrom {RUN_LOG['frames']} frames, {RUN_LOG['steps']} steps, "
      f"peak lr {RUN_LOG['peak_lr']:.2e}, batch {RUN_LOG['batch']}")
''')


# --------------------------------------------------------------------------
# 8. The handoff, and the export caveat
# --------------------------------------------------------------------------

code('''
assert Path(CHECKPOINT).is_file(), "the run wrote no checkpoint"
shutil.copy(CHECKPOINT, Path(MIRROR_DIR) / Path(CHECKPOINT).name)
shutil.copy(Path(WORK) / "fcmae.json", Path(MIRROR_DIR) / "fcmae.json")
_MIRRORED = str(Path(MIRROR_DIR) / Path(CHECKPOINT).name)
print(f"checkpoint -> {_MIRRORED}")

print("\\nto measure it, put this in cell 1 of 32 and run the arm:")
print(f'    BASE_CHECKPOINT = "{_MIRRORED}"')
print("\\n32 names the run after it, so the arm lands beside the one that "
      "starts from stock rather than on top of it. That comparison IS the "
      "result: a pretraining stage that does not beat stage B from stock cost "
      "GPU hours and bought nothing.")
if ARM == "grn":
    print("\\nGRN added parameters to the trunk, so this checkpoint does not "
          "match the engines already built. Before it reaches TensorRT:")
    print("    python tools/export_edgetam_onnx.py --size 512 "
          f"--checkpoint {_MIRRORED} --all-pointers --verify")
    print("    python tools/build_trt_engines.py --outdir models512_fcmae/")
    print("PyTorch evaluation needs none of that.")
''')


# --------------------------------------------------------------------------
# 9. The verdict
# --------------------------------------------------------------------------

code('''
VERDICT = {
    "notebook": NOTEBOOK, "stamp": STAMP, "run": RUN, "arm": ARM,
    "frames": RUN_LOG["frames"], "steps": RUN_LOG["steps"],
    "modality": RUN_LOG["modality"], "census": RUN_LOG["census"],
    "mse_first": _history[0]["mse"], "mse_last": _history[-1]["mse"],
    "grn": REPORT, "config": RUN_LOG["config"], "peak_lr": RUN_LOG["peak_lr"],
    "batch": RUN_LOG["batch"], "seed": SEED, "minutes": RUN_LOG["seconds"] / 60,
    "checkpoint": _MIRRORED, "base": BASE_CKPT,
    "paper": "ConvNeXt V2, arXiv 2301.00808",
    "ablation_to_reproduce": {"v1_supervised": 83.8, "v1_fcmae": 83.7,
                              "v2_fcmae": 84.6},
    "next": "32 with BASE_CHECKPOINT set to the checkpoint above, against 32 "
            "from stock",
}
Path(MIRROR_DIR, "verdict.json").write_text(json.dumps(VERDICT, indent=2) + "\\n")
print(json.dumps(VERDICT, indent=2))
print(f"\\nrun the other arm by setting ARM = "
      f'"{"plain" if ARM == "grn" else "grn"}" in cell 1 and running again; it '
      f"lands in its own folder.")
''')


def render() -> tuple[list[str], str]:
    cells = [text.replace("{{BRANCH}}", BRANCH).replace("{{NOTEBOOK}}", NOTEBOOK)
             for text in CELLS]
    return cells, hashlib.sha256("\n".join(cells).encode()).hexdigest()[:10]


def build() -> dict:
    cells, stamp = render()
    return {
        "cells": [
            {"cell_type": "code", "metadata": {}, "outputs": [],
             "execution_count": None,
             "source": (text.replace("{{STAMP}}", stamp) + "\n")
                       .splitlines(keepends=True)}
            for text in cells
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
    stamps_file = repo / "notebooks" / ".stamps.json"
    known = json.loads(stamps_file.read_text()) if stamps_file.is_file() else {}
    document = build()
    (repo / "notebooks" / NOTEBOOK).write_text(
        json.dumps(document, indent=1, ensure_ascii=False) + "\n")
    known[NOTEBOOK] = render()[1]
    stamps_file.write_text(json.dumps(known, indent=1, sort_keys=True) + "\n")
    print(f"wrote notebooks/{NOTEBOOK} with {len(document['cells'])} cells "
          f"[{known[NOTEBOOK]}]", file=sys.stderr)
