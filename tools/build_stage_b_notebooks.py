#!/usr/bin/env python3
"""Generate notebooks 19 and 20: stage B on the mask pools, and nothing else.

Two arms of one experiment, generated from one file the way 07-11 and 16/17
are, and for the same reason: they differ in **one setting** and every other
byte is identical, which is the only thing that lets their two numbers be set
beside each other.

    19_thermal_stage_b_pool.ipynb       MODALITIES = ["thermal"]
    20_thermal_stage_b_pool_rgb.ipynb   MODALITIES = ["thermal", "rgb"]

Both are the shape 15-18 were asked for -- **no markdown, no comments, as few
cells as possible** -- because this is a job to run, not a decision to read.
Everything the run needs to say, it prints.

## What these do that 07-11 do not

**No stage A.** `07` distils an unlabelled RGB teacher into the encoder before
it trains, and `12` then measured the anchor meant to protect that stage
*eating* it (0.8670 at weight 0 against 0.8536 at 0.5, with a plain fine-tune
at 0.8611). These notebooks skip the whole question: one stage B from stock
EdgeTAM, so what the pools bought is the only thing in the number.

**The data is the pools.** 07's stage B is Kust4K and SegFly's *reconstructed*
instances plus VTUAV's 875 drawn masks. These read what notebooks 13-18
harvested -- real detection boxes turned into teacher masks, tens of thousands
of them -- through `src/training/pool_reader.py`, which was the follow-up
`docs/mask_pool_plan.md` left open.

**Before and after, on the same instances.** The whole point of the run is
whether stage B on pools is worth anything, so both notebooks score **stock
EdgeTAM** and the trained checkpoint on the same held-out split, under two
prompts, and then draw the instances whose IoU moved most -- in both
directions. A table of means can hide a model that got better at trucks and
worse at people; the panel cannot.

## The two grades, and why there are two

`pool test` is a held-out slice of the pools themselves: the same distribution
the run trained on, so it is the sensitive number, and its "truth" is a
teacher's guess gated four ways.

`drawn test` is a dataset whose masks a human drew (Kust4K's semantic maps,
decomposed the way `07` decomposes them), held at `role=eval` so no window of
it is ever trained on. It is the honest number and the conservative one.

The two cannot overlap, so cell 2 **drops any pool built from the drawn set's
own frames** and says it did. Training on `kust4k_thermal` while grading on
Kust4K's drawn maps would be scoring on frames the run had seen, and the
stratified split cannot prevent it: the pool and the semantic set are separate
sources with separate permutations.

## Filling an 80 GB card

`--batch` is measured, not chosen, and the ladder now reaches 512 (image mode
holds no clip length and no memory bank, so it fits far more than the video
path). `--steps` is fixed, so a bigger batch means more samples behind the same
number of updates -- and the linear scaling rule says the step should grow with
it. The notebook sets `--lr-scale` from the measured batch against a 16-window
reference, capped, and prints both.

## What none of it measures

Tracking. Every number here scores one prompted frame with the memory path
frozen and never executed. A better encoder is a precondition for a better
tracker, not evidence of one -- that is stage C.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# Only these move between the two arms. Everything else is shared source, so a
# difference anywhere else is a bug rather than a variant.
ARMS = {
    "19_thermal_stage_b_pool.ipynb": {
        "MODALITIES": '["thermal"]',
        "RUN": '"thermal"',
        "MIRROR_DIR": '"/content/drive/MyDrive/edgetam-stage-b/thermal"',
    },
    "20_thermal_stage_b_pool_rgb.ipynb": {
        "MODALITIES": '["thermal", "rgb"]',
        "RUN": '"thermal_rgb"',
        "MIRROR_DIR": '"/content/drive/MyDrive/edgetam-stage-b/thermal_rgb"',
    },
}

BRANCH = "claude/thermal-stage-b-training-43ktcl"

CELLS: list[str] = []


def code(text: str) -> None:
    CELLS.append(text.strip("\n"))


# --------------------------------------------------------------------------
# 1. Every knob, the repo, the card, and what the card can take
# --------------------------------------------------------------------------

code('''
MODALITIES  = {{MODALITIES}}
RUN         = {{RUN}}
MIRROR_DIR  = {{MIRROR_DIR}}
DRIVE_POOLS = "/content/drive/MyDrive/edgetam-pool"
POOL_ROOT   = "/content/pool"
DATA_ROOT   = "/content/data"
WORK        = "/content/work"
STAGE_DIR   = "/content/drive/MyDrive/datasets"

EVAL_DRAWN  = "kust4k"
EVAL_SPEC   = "kust4k:{root}:thermal:components:eval"
SKIP_POOLS  = ["broken", "rgbt234"]
POOL_ROLE   = "all"

VTUAV_PARTS = []

IMAGES = [
    ["hituav",       "hituav",       "HIT_UAV",      []],
    ["dronevehicle", "dronevehicle", "DroneVehicle", ["train"]],
    ["kust4k",       "kust4k",       "Kust4K",       ["tir", "labels", "rgb"]],
    ["segfly_rgb",   "segfly_rgb",   "SegFly",       []],
    ["segfly",       "segfly",       "SegFly",       []],
    ["visdrone",     "visdrone",     "VisDrone",     []],
    ["vtuav",        "vtuav_track",  "VTUAV",        VTUAV_PARTS],
]

SIZE           = 512
PER_IMAGE      = 2
MAX_INSTANCES  = 8
MIN_AREA       = 48
MIN_SIDE       = 4
MAX_AREA       = 0.9
FILL           = 0.25
JITTER         = 32
PROMPT         = "mix"
PROMPT_JITTER  = 0.3
EPOCHS         = [1, 3]
STEPS          = 400
VAL_BATCHES    = 24
BATCH          = 0
BATCH_CEILING  = 512
BATCH_RESERVE  = 0.12
LR_REFERENCE   = 16
LR_SCALE_MAX   = 4.0
SEED           = 0
SCORE_PROMPTS  = ["box", "point"]
PANEL_CASES    = 6
PANEL_WINDOWS  = 400
FETCH          = True

REPO_URL = "https://github.com/yigitkayabagci/sam-dedection.git"
BRANCH   = "{{BRANCH}}"
REPO_DIR = "/content/sam-dedection"
NOTEBOOK = "{{NOTEBOOK}}"
STAMP    = "{{STAMP}}"

import json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if not Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                    REPO_URL, REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
for _stale in [n for n in list(sys.modules) if n.split(".")[0] in ("src", "tools")]:
    del sys.modules[_stale]

try:
    from google.colab import drive as _drive
    _drive.mount("/content/drive")
except Exception as _mount_error:
    print("no Colab Drive mount:", _mount_error)

import torch
_TORCH_WAS = torch.__version__

subprocess.run(["bash", "scripts/setup_edgetam.sh"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                "requirements.txt"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hf_transfer",
                "tqdm", "matplotlib"], check=False)

import importlib.metadata as _md
assert _md.version("torch") == _TORCH_WAS, (
    f"pip replaced torch {_TORCH_WAS} with {_md.version('torch')} on disk. "
    f"Restore it before restarting: pip install --force-reinstall "
    f"torch=={_TORCH_WAS}")

import importlib
EDGETAM = str(Path(REPO_DIR) / "third_party" / "EdgeTAM")
if EDGETAM not in sys.path:
    sys.path.insert(sys.path.index(REPO_DIR) + 1, EDGETAM)
importlib.invalidate_caches()
import sam2
assert Path(sam2.__file__).resolve().parent.parent == Path(EDGETAM).resolve(), (
    f"sam2 imported from {Path(sam2.__file__).parent} and not the EdgeTAM "
    f"checkout. pip uninstall -y sam2 SAM-2, then re-run this cell.")
BASE_CKPT = str(Path(EDGETAM) / "checkpoints" / "edgetam.pt")
assert Path(BASE_CKPT).is_file(), "edgetam.pt did not download"

for _dir in (POOL_ROOT, DATA_ROOT, WORK, MIRROR_DIR,
             str(Path(WORK) / "index"), str(Path(REPO_DIR) / "checkpoints")):
    Path(_dir).mkdir(parents=True, exist_ok=True)
INDEX_DIR = str(Path(WORK) / "index")
CHECKPOINT = str(Path(REPO_DIR) / "checkpoints" / f"edgetam_pool_{RUN}_{SIZE}.pt")

_props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
VRAM = round(_props.total_memory / 2 ** 30, 1) if _props else 0.0
if _props:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    _probe = torch.randn(256, 256, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        (_probe @ _probe).sum().item()
    del _probe
WORKERS = min(2 * (os.cpu_count() or 4), 24)

def progress(stream, total, desc):
    from tqdm.auto import tqdm
    return tqdm(stream, total=total, desc=desc)

_stamps = Path(REPO_DIR) / "notebooks" / ".stamps.json"
_want = json.loads(_stamps.read_text()).get(NOTEBOOK) if _stamps.is_file() else None
print(NOTEBOOK, STAMP, "| repo:", _want,
      "| OK" if _want == STAMP else "| STALE, re-open from the repo")
print(_props.name if _props else "no GPU", VRAM, "GiB |", WORKERS, "loader threads")
print("run:", RUN, "| modalities:", MODALITIES, "| checkpoint ->", CHECKPOINT)
print("stage B only: no stage A, no distillation, base =", BASE_CKPT)
''')


# --------------------------------------------------------------------------
# 2. Unpack the pools, work out what each one is, and fetch its frames
# --------------------------------------------------------------------------
#
# Discovery rather than a hard-coded list, because the Drive folder's shape is
# whatever the harvest notebooks left: 15 writes four pools as four zips, 16/17
# write one per sequence, 18 writes four, and a sharded staging writes numbered
# parts whose members still carry the pool name. Unzipping everything into one
# root and then asking `pool_datasets` what is there is the only reading of
# that folder that cannot go stale.
#
# The drop rule is the measurement rule: a pool built from the drawn set's own
# frames cannot be trained on while that set is the grade.

code('''
from src.training.pool_reader import discover_pools

_done = Path(POOL_ROOT) / ".unpacked"
_done.mkdir(parents=True, exist_ok=True)
assert Path(DRIVE_POOLS).is_dir(), f"{DRIVE_POOLS} is not there -- set DRIVE_POOLS"

for _zip in sorted(Path(DRIVE_POOLS).rglob("*.zip")):
    _marker = _done / (_zip.stem + ".zip.done")
    if _marker.is_file():
        continue
    try:
        with zipfile.ZipFile(_zip) as _handle:
            _handle.extractall(POOL_ROOT)
    except Exception as _unzip_error:
        print("!! could not read", _zip.name, "--", _unzip_error)
        continue
    _marker.touch()
    print("unpacked", _zip.name, round(_zip.stat().st_size / 2 ** 20, 1), "MiB")

for _name, _folder in discover_pools(DRIVE_POOLS).items():
    _marker = _done / (_name + ".folder.done")
    if _marker.is_file() or (Path(POOL_ROOT) / _name).is_dir():
        continue
    shutil.copytree(_folder, Path(POOL_ROOT) / _name, dirs_exist_ok=True)
    _marker.touch()
    print("copied", _name, "from Drive (it was staged unzipped)")

POOLS = discover_pools(POOL_ROOT)
FOUND = sorted(POOLS)
print("\\npools on disk:", FOUND or "none")

def modality_of(pool):
    lowered = pool.lower()
    if lowered.endswith("_rgb") or "_rgb_" in lowered or "rgb" in lowered.split("_"):
        return "rgb"
    return "thermal"

def images_for(pool):
    lowered = pool.lower()
    for key, recipe, folder, parts in IMAGES:
        if key in lowered:
            return key, recipe, str(Path(DATA_ROOT) / folder), list(parts)
    return None, "", "", []

PLAN, DROPPED = [], []
for _pool in FOUND:
    _key, _recipe, _root, _parts = images_for(_pool)
    _modality = modality_of(_pool)
    if any(skip in _pool.lower() for skip in SKIP_POOLS):
        DROPPED.append((_pool, "SKIP_POOLS"))
    elif _modality not in MODALITIES:
        DROPPED.append((_pool, f"modality {_modality} not in MODALITIES"))
    elif _key is None:
        DROPPED.append((_pool, "no entry in IMAGES -- add one"))
    elif EVAL_DRAWN and _key == EVAL_DRAWN:
        DROPPED.append((_pool, f"its frames are {EVAL_DRAWN}'s, which is the grade"))
    else:
        PLAN.append({"pool": _pool, "dir": str(POOLS[_pool]), "key": _key,
                     "recipe": _recipe, "images": _root, "parts": _parts,
                     "modality": _modality})

print(f"\\n{'pool':<28}{'modality':<10}{'frames from':<28}")
for _row in PLAN:
    print(f"{_row['pool']:<28}{_row['modality']:<10}{_row['images']:<28}")
for _pool, _why in DROPPED:
    print(f"{_pool:<28}dropped   {_why}")
assert PLAN, "nothing left to train on -- read the dropped list above"

_wanted = {}
for _row in PLAN:
    _wanted.setdefault((_row["recipe"], _row["images"]), set()).update(_row["parts"])
if EVAL_DRAWN:
    _drawn = next((r for r in IMAGES if r[0] == EVAL_DRAWN), None)
    assert _drawn, f"EVAL_DRAWN={EVAL_DRAWN} has no IMAGES entry"
    EVAL_ROOT = str(Path(DATA_ROOT) / _drawn[2])
    _wanted.setdefault((_drawn[1], EVAL_ROOT), set()).update(_drawn[3])
else:
    EVAL_ROOT = ""

if FETCH:
    from tools.fetch_datasets import fetch
    Path(STAGE_DIR).mkdir(parents=True, exist_ok=True)
    for (_recipe, _root), _parts in sorted(_wanted.items()):
        if not _recipe or (_recipe == "vtuav_track" and not _parts):
            print("not fetched:", _root, "-- stage it yourself, or name its "
                  "parts (VTUAV_PARTS in cell 1; every part is ~16 GiB)")
            continue
        if Path(_root).is_dir() and any(
                any(Path(_root).rglob(_glob)) for _glob in ("*.jpg", "*.png")):
            print("already on disk:", _root)
            continue
        try:
            fetch(_recipe, Path(_root), tuple(sorted(_parts)) or None,
                  stage=STAGE_DIR)
        except Exception as _fetch_error:
            print("!!", _recipe, "->", _root, "failed:", _fetch_error)

subprocess.run(["df", "-h", "/content"], check=False)
''')


# --------------------------------------------------------------------------
# 3. The index, the split, and the flags every later cell reuses
# --------------------------------------------------------------------------
#
# Built in-process and cached, so the training and scoring subprocesses below
# reuse the same file and therefore provably the same instances. A pool whose
# frames were never downloaded is dropped here with its reason rather than
# taking the run down: five pools that resolve are worth more than a traceback.

code('''
from src.training.aerial import InstanceGates, sample_windows, split_index
from src.training.datasets import parse
from src.training.image_loop import ImageSplit
from src.training.pool_reader import index_pool, load_pool_index, parse_pool, save_pool_index
from tools.train_encoder import build_indexes

GATES = InstanceGates(min_area=MIN_AREA, min_side=MIN_SIDE, max_area=MAX_AREA,
                      fill=FILL)

POOL_FLAGS, INDEX, FAILED = [], [], []
for _row in PLAN:
    _flag = f"{_row['dir']}:{_row['images']}:{_row['modality']}:{POOL_ROLE}"
    _request = parse_pool(_flag, GATES)
    _cache = Path(INDEX_DIR) / f"{_request.cache_name}.json"
    try:
        if _cache.is_file():
            _part = load_pool_index(_cache, GATES, POOL_ROLE)
        else:
            _part = index_pool(_request.pool, _request.images, _request.modality,
                               POOL_ROLE, GATES, _request.name,
                               workers=WORKERS, progress=progress)
            save_pool_index(_cache, _part)
    except (ValueError, FileNotFoundError) as _index_error:
        FAILED.append((_row["pool"], str(_index_error).splitlines()[0]))
        continue
    POOL_FLAGS.append(_flag)
    INDEX.extend(_part)
    print(f"{_row['pool']:<28}{len(_part):>7} frames "
          f"{sum(len(e.instances) for e in _part):>9} instances")

for _pool, _why in FAILED:
    print(f"{_pool:<28}unusable  {_why}")
assert POOL_FLAGS, "no pool resolved its frames -- check the IMAGES roots above"

DATASET_FLAGS = []
if EVAL_DRAWN:
    try:
        _drawn_flag = EVAL_SPEC.format(root=EVAL_ROOT)
        _drawn_index = build_indexes([parse(_drawn_flag, GATES)],
                                     Path(INDEX_DIR), WORKERS)
        DATASET_FLAGS.append(_drawn_flag)
        INDEX.extend(_drawn_index)
        print(f"\\ndrawn grade: {EVAL_DRAWN} at role=eval, "
              f"{len(_drawn_index)} frames, "
              f"{sum(len(e.instances) for e in _drawn_index)} instances")
    except (ValueError, FileNotFoundError, AssertionError) as _drawn_error:
        print(f"\\n!! no drawn grade: {EVAL_DRAWN} -> "
              f"{str(_drawn_error).splitlines()[0]}")
        print("   the test split is the pools' own held-out slice, so its "
              "truth is the teacher's. Download it, or read the number "
              "knowing that.")

COMMON = []
for _flag in POOL_FLAGS:
    COMMON += ["--pool", _flag]
for _flag in DATASET_FLAGS:
    COMMON += ["--dataset", _flag]
COMMON += ["--index", INDEX_DIR, "--size", str(SIZE),
           "--per-image", str(PER_IMAGE), "--max-instances", str(MAX_INSTANCES),
           "--min-area", str(MIN_AREA), "--min-side", str(MIN_SIDE),
           "--max-area", str(MAX_AREA), "--fill", str(FILL), "--seed", str(SEED)]

SPLITS = split_index(INDEX, seed=SEED)
def windows(name, jitter):
    return ImageSplit(sample_windows(SPLITS[name], size=SIZE, per_image=PER_IMAGE,
                                     max_instances=MAX_INSTANCES, jitter=jitter,
                                     seed=SEED))
TRAIN, VAL, TEST = windows("train", JITTER), windows("val", 0), windows("test", 0)

print()
for _name, _split in (("train", TRAIN), ("val", VAL), ("test", TEST)):
    print(f"{_name:<6}{len(SPLITS[_name]):>7} frames{len(_split.samples):>8} windows"
          f"{sum(len(s.instances) for s in _split.samples):>9} instances")
print("\\ntrain windows by source:")
for _source, _count in TRAIN.sources.items():
    print(f"  {_source:<34}{_count:>8}")
print("\\ntest windows by source:")
for _source, _count in TEST.sources.items():
    print(f"  {_source:<34}{_count:>8}")
assert not ({id(e) for e in SPLITS["train"]} & {id(e) for e in SPLITS["test"]})
assert TEST.samples, "the test split is empty -- POOL_ROLE=train leaves nothing to grade"
''')


# --------------------------------------------------------------------------
# 4. The batch this card takes, and the before picture
# --------------------------------------------------------------------------
#
# Stock is scored first and deliberately: it is the floor everything after this
# is measured against, and taking it before the training run means a runtime
# that dies during training still leaves the baseline on Drive.
#
# Two prompts, because 12 measured that one of them cannot see an encoder
# change: an exact ground-truth box states most of the mask on an isolated
# target, and the same checkpoints separated by 0.086 under a centre point
# landed within 0.02 under a box.

code('''
from src.training.finetune import apply_freeze
from src.training.image_loop import auto_batch_size
from tools.train_encoder import build_model

def score_to(checkpoint, tag, prompt):
    out = Path(WORK) / f"score_{tag}_{prompt}.json"
    if out.is_file():
        return json.loads(out.read_text())
    subprocess.run(
        [sys.executable, "tools/eval_instances.py", *COMMON,
         "--checkpoint", checkpoint, "--split", "test", "--prompt", prompt,
         "--batch", str(max(BATCH // 2, 1)), "--device", "cuda",
         "--json", str(out)], check=True)
    return json.loads(out.read_text())

if BATCH <= 0:
    _model = build_model(SIZE, BASE_CKPT, "cuda")
    apply_freeze(_model, "encoder")
    BATCH = auto_batch_size(_model, TRAIN, device="cuda",
                            maximum=BATCH_CEILING, reserve=BATCH_RESERVE)
    del _model
    torch.cuda.empty_cache()
ACCUM = 1
LR_SCALE = round(min(max(BATCH / LR_REFERENCE, 1.0), LR_SCALE_MAX), 3)
print(f"batch {BATCH} x accum {ACCUM} on {VRAM} GiB | lr-scale {LR_SCALE} "
      f"(linear rule against {LR_REFERENCE} windows, capped at {LR_SCALE_MAX})")

BEFORE = {p: score_to(BASE_CKPT, "stock", p) for p in SCORE_PROMPTS}
for _prompt, _row in BEFORE.items():
    print(f"stock  {_prompt:<6} mean {_row['mean_iou']:.4f}  "
          f">=.5 {_row['iou_50']:.3f}  small {_row['small_mean_iou']:.4f}  "
          f"n={_row['instances']}")
''')


# --------------------------------------------------------------------------
# 5. Stage B
# --------------------------------------------------------------------------

code('''
import time
_started = time.time()
subprocess.run(
    [sys.executable, "tools/train_encoder.py", *COMMON,
     "--method", "finetune", "--base", BASE_CKPT, "--out", CHECKPOINT,
     "--prompt", PROMPT, "--prompt-jitter", str(PROMPT_JITTER),
     "--jitter", str(JITTER), "--batch", str(BATCH), "--accum", str(ACCUM),
     "--lr-scale", str(LR_SCALE), "--steps", str(STEPS),
     "--epochs", str(EPOCHS[0]), str(EPOCHS[1]),
     "--val-batches", str(VAL_BATCHES), "--workers", str(WORKERS),
     "--anchor-weight", "0.0", "--device", "cuda",
     "--json", str(Path(WORK) / "run.json")], check=True)

RUN_LOG = json.loads((Path(WORK) / "run.json").read_text())
assert Path(CHECKPOINT).is_file(), "training wrote no checkpoint"
shutil.copy(CHECKPOINT, Path(MIRROR_DIR) / Path(CHECKPOINT).name)
shutil.copy(Path(WORK) / "run.json", Path(MIRROR_DIR) / "run.json")
print(f"{RUN_LOG['best_val_loss']:.4f} best val loss, "
      f"{RUN_LOG['seconds'] / 60:.0f} min, peak {RUN_LOG['peak_gib']:.1f} GiB, "
      f"batch {RUN_LOG['batch']} -> {MIRROR_DIR}")
print("wall clock", round((time.time() - _started) / 60, 1), "min")
''')


# --------------------------------------------------------------------------
# 6. The after picture, on the same instances
# --------------------------------------------------------------------------
#
# Split by modality and by source as well as in aggregate: a mixed run's mean
# blends two problems, and the thermal rows are the ones the two arms of this
# experiment can be compared on.

code('''
AFTER = {p: score_to(CHECKPOINT, RUN, p) for p in SCORE_PROMPTS}

print(f"{'prompt':<8}{'':<10}{'mean IoU':>10}{'>=.50':>8}{'>=.75':>8}"
      f"{'small':>10}{'larger':>9}")
for _prompt in SCORE_PROMPTS:
    for _label, _row in (("stock", BEFORE[_prompt]), ("stage B", AFTER[_prompt])):
        print(f"{_prompt:<8}{_label:<10}{_row['mean_iou']:>10.4f}"
              f"{_row['iou_50']:>8.3f}{_row['iou_75']:>8.3f}"
              f"{_row['small_mean_iou']:>10.4f}{_row['large_mean_iou']:>9.4f}")
    _d = AFTER[_prompt]["mean_iou"] - BEFORE[_prompt]["mean_iou"]
    _s = AFTER[_prompt]["small_mean_iou"] - BEFORE[_prompt]["small_mean_iou"]
    print(f"{_prompt:<8}{'delta':<10}{_d:>+10.4f}{'':>16}{_s:>+10.4f}\\n")

for _prompt in SCORE_PROMPTS:
    for _modality in sorted(AFTER[_prompt].get("per_modality", {})):
        _b = BEFORE[_prompt]["per_modality"][_modality]
        _a = AFTER[_prompt]["per_modality"][_modality]
        print(f"{_prompt:<8}{_modality:<10}{_a['instances']:>8} inst  "
              f"mean {_b['mean_iou']:.4f} -> {_a['mean_iou']:.4f} "
              f"({_a['mean_iou'] - _b['mean_iou']:+.4f})  small "
              f"{_b['small_mean_iou']:.4f} -> {_a['small_mean_iou']:.4f} "
              f"({_a['small_mean_iou'] - _b['small_mean_iou']:+.4f})")

import numpy as np
import matplotlib.pyplot as plt

_classes = sorted(set(AFTER[SCORE_PROMPTS[0]]["per_class"])
                  & set(BEFORE[SCORE_PROMPTS[0]]["per_class"]))
_y = np.arange(len(_classes))
_fig, _ax = plt.subplots(figsize=(9.0, 0.42 * len(_classes) + 2.0))
for _i, _prompt in enumerate(SCORE_PROMPTS):
    _offset = (_i - (len(SCORE_PROMPTS) - 1) / 2) * 0.8 / len(SCORE_PROMPTS)
    _delta = [AFTER[_prompt]["per_class"][c]["mean_iou"]
              - BEFORE[_prompt]["per_class"][c]["mean_iou"] for c in _classes]
    _ax.barh(_y + _offset, _delta, height=0.8 / len(SCORE_PROMPTS),
             label=_prompt, color=["#4c72b0", "#dd8452", "#55a868"][_i % 3])
_ax.set_yticks(_y); _ax.set_yticklabels(_classes, fontsize=8)
_ax.axvline(0, color="k", lw=0.8)
_ax.set_xlabel("mean IoU after stage B, minus stock (test split)")
_ax.legend(loc="lower right")
plt.tight_layout(); plt.show()
''')


# --------------------------------------------------------------------------
# 7. The cases, before and after
# --------------------------------------------------------------------------
#
# A mean can move for reasons a table cannot show: a model that got better at
# trucks and worse at people reports the average of the two. This scores every
# held-out instance twice, ranks by the change, and draws both ends of the
# ranking -- the gains and the regressions, in one figure, so the second is not
# something a reader has to go looking for.
#
# Scored under the weakest prompt on the list, which is the one 12 found can
# see an encoder change at all.

code('''
from src.training.aerial import image_origin, load_image
from src.training.image_loop import collate, propagate_image
from src.training.loader import batch_clips

PANEL_PROMPT = SCORE_PROMPTS[-1]

def predict(model, samples, batch_size, want_masks=False):
    scored, drawn = [], []
    for chunk in batch_clips(samples, max(batch_size, 1), seed=0, drop_last=False):
        batch = collate(chunk, "cpu").to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            result = propagate_image(model, batch.images, batch.boxes,
                                     batch.valid, PANEL_PROMPT)
        logits = result["pred_masks_high_res"]
        logits = logits[:, 0] if logits.dim() == 4 else logits
        predicted = (logits.float() > 0.0).cpu().numpy()
        targets = batch.masks.reshape(-1, *batch.masks.shape[-2:])
        targets = targets[result["rows"]].cpu().numpy()
        width = batch.boxes.shape[1]
        for _n, _flat in enumerate(result["rows"].cpu().numpy().tolist()):
            _b, _k = divmod(int(_flat), width)
            _union = np.logical_or(predicted[_n], targets[_n]).sum()
            _hit = np.logical_and(predicted[_n], targets[_n]).sum()
            scored.append((chunk[_b], _k, float(_hit / _union) if _union else 1.0))
            if want_masks:
                drawn.append((predicted[_n].copy(), targets[_n].copy()))
        del batch, result, predicted, targets
    torch.cuda.empty_cache()
    return scored, drawn

PANEL_POOL = TEST.samples[:PANEL_WINDOWS]
_model = build_model(SIZE, BASE_CKPT, "cuda")
_before, _ = predict(_model, PANEL_POOL, max(BATCH // 4, 1))
del _model; torch.cuda.empty_cache()
_model = build_model(SIZE, CHECKPOINT, "cuda")
_after, _ = predict(_model, PANEL_POOL, max(BATCH // 4, 1))
del _model; torch.cuda.empty_cache()

CASES = [{"sample": a[0], "k": a[1], "before": b[2], "after": a[2],
          "delta": a[2] - b[2]}
         for a, b in zip(_after, _before) if (a[0], a[1]) == (b[0], b[1])]
CASES.sort(key=lambda c: c["delta"])
_half = min(PANEL_CASES, len(CASES) // 2)
SHOWN = CASES[-_half:][::-1] + CASES[:_half]
print(f"{len(CASES)} held-out instances scored twice under `{PANEL_PROMPT}`: "
      f"{sum(1 for c in CASES if c['delta'] > 0.01)} better, "
      f"{sum(1 for c in CASES if c['delta'] < -0.01)} worse, "
      f"{sum(1 for c in CASES if abs(c['delta']) <= 0.01)} unchanged")
assert SHOWN, "no instance was scored by both checkpoints"

PICKED = sorted({id(c["sample"]): c["sample"] for c in SHOWN}.values(),
                key=lambda s: s.frame.name)
MASKS = {}
for _tag, _path in (("before", BASE_CKPT), ("after", CHECKPOINT)):
    _model = build_model(SIZE, _path, "cuda")
    _rows, _drawn = predict(_model, PICKED, 1, want_masks=True)
    for (_sample, _k, _), (_mask, _target) in zip(_rows, _drawn):
        MASKS[(_tag, id(_sample), _k)] = _mask
        MASKS[("target", id(_sample), _k)] = _target
    del _model; torch.cuda.empty_cache()

_fig, _axes = plt.subplots(2, _half, figsize=(3.1 * _half, 7.0), squeeze=False)
for _ax, _case in zip(_axes.ravel(), SHOWN):
    _sample, _k = _case["sample"], _case["k"]
    _stock = MASKS[("before", id(_sample), _k)]
    _trained = MASKS[("after", id(_sample), _k)]
    _target = MASKS[("target", id(_sample), _k)]
    _canvas = load_image(_sample.frame.image, image_origin(_sample),
                         _sample.window, _sample.size,
                         _sample.source.gray if _sample.source else True
                         ).astype(np.float32)
    _both = _stock & _trained
    _canvas[_both] = 0.5 * _canvas[_both] + np.array([70, 110, 210])
    _canvas[_stock & ~_trained] = [225, 70, 70]
    _canvas[_trained & ~_stock] = [60, 215, 95]
    _canvas[_target ^ np.roll(_target, 1, axis=0)] = [255, 235, 0]
    _ax.imshow(_canvas.clip(0, 255).astype(np.uint8))
    _box = _sample.boxes[_k]
    _ax.add_patch(plt.Rectangle((_box[0], _box[1]), _box[2] - _box[0],
                                _box[3] - _box[1], fill=False, ec="w", lw=0.8))
    _ax.set_title(f"{_sample.source.spec.name.split('/')[-1]} "
                  f"{_sample.frame.name}\\n{_case['before']:.2f} -> "
                  f"{_case['after']:.2f}  ({_case['delta']:+.2f})", fontsize=8)
    _ax.axis("off")
plt.suptitle(f"top row: stage B gained   |   bottom row: stage B lost   "
             f"(prompt: {PANEL_PROMPT})", y=1.0)
plt.tight_layout(); plt.show()
print("yellow = the target's outline | green = only stage B found it | "
      "red = only stock found it | blue = both agreed")
''')


# 8. Everything worth keeping, and the verdict in one block
# --------------------------------------------------------------------------

code('''
VERDICT = {
    "run": RUN, "modalities": MODALITIES, "image_size": SIZE,
    "pools": POOL_FLAGS, "datasets": DATASET_FLAGS,
    "dropped": DROPPED, "unusable": FAILED,
    "frames": {k: len(v) for k, v in SPLITS.items()},
    "train_windows_by_source": TRAIN.sources,
    "test_windows_by_source": TEST.sources,
    "gates": GATES.__dict__, "batch": BATCH, "lr_scale": LR_SCALE,
    "prompt": PROMPT, "prompt_jitter": PROMPT_JITTER,
    "epochs": EPOCHS, "steps": STEPS, "seed": SEED,
    "run_log": RUN_LOG, "before": BEFORE, "after": AFTER,
    "panel_prompt": PANEL_PROMPT,
    "cases": [{"frame": c["sample"].frame.name,
               "source": c["sample"].source.spec.name, "k": c["k"],
               "before": c["before"], "after": c["after"], "delta": c["delta"]}
              for c in SHOWN],
}
(Path(WORK) / "verdict.json").write_text(json.dumps(VERDICT, indent=2) + "\\n")
for _name in ("verdict.json", "run.json"):
    shutil.copy(Path(WORK) / _name, Path(MIRROR_DIR) / _name)
for _file in sorted(Path(WORK).glob("score_*.json")):
    shutil.copy(_file, Path(MIRROR_DIR) / _file.name)

_line = "-" * 74
print(_line)
print("WHAT THIS RUN TRAINED ON")
print(f"  {sum(len(v) for v in SPLITS.values())} frames, "
      f"{len(TRAIN.samples)} train windows, "
      f"{sum(len(s.instances) for s in TRAIN.samples)} instances")
for _source, _count in TRAIN.sources.items():
    print(f"    {_source:<36}{_count:>8} windows")
print(f"  base {Path(BASE_CKPT).name}, stage A: none")
print(_line)
print("DID IT HELP")
for _prompt in SCORE_PROMPTS:
    _d = AFTER[_prompt]["mean_iou"] - BEFORE[_prompt]["mean_iou"]
    _s = AFTER[_prompt]["small_mean_iou"] - BEFORE[_prompt]["small_mean_iou"]
    print(f"  prompt {_prompt:<6} mean IoU {_d:+.4f}   small (<32 px) {_s:+.4f}")
print(f"  {sum(1 for c in CASES if c['delta'] > 0.01)} of {len(CASES)} held-out "
      f"instances improved, {sum(1 for c in CASES if c['delta'] < -0.01)} got worse")
print("  read the `point` row, not the `box` one: an exact box states most of")
print("  the mask on an isolated target, which is why 12 could not tell two")
print("  visibly different encoders apart under it.")
print(_line)
print("WHAT IT DID NOT MEASURE")
print("  Tracking. Every number above is one prompted frame with the memory")
print("  path frozen and never run. A better encoder is a precondition for a")
print("  better tracker, not evidence of one -- that is stage C.")
print(_line)
print("ON YOUR DRIVE:", MIRROR_DIR)
for _file in sorted(Path(MIRROR_DIR).iterdir()):
    print(f"  {_file.name:<40}{round(_file.stat().st_size / 2 ** 20, 1):>9} MiB")
''')


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def render(notebook: str) -> tuple[list[str], str]:
    """One arm's cell texts and the stamp over them.

    The stamp folds in the variant, not just the shared source: two arms built
    from one file share most of their bytes, and a stamp that ignored the
    difference would call a swapped pair correct.
    """
    fields = {**ARMS[notebook], "BRANCH": BRANCH, "NOTEBOOK": notebook}
    cells = []
    for text in CELLS:
        for key, value in fields.items():
            text = text.replace("{{" + key + "}}", value)
        cells.append(text)
    return cells, hashlib.sha256("\n".join(cells).encode()).hexdigest()[:10]


def build(notebook: str) -> dict:
    cells, stamp = render(notebook)
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
    for name in ARMS:
        document = build(name)
        (repo / "notebooks" / name).write_text(
            json.dumps(document, indent=1, ensure_ascii=False) + "\n")
        known[name] = render(name)[1]
        print(f"wrote notebooks/{name} with {len(document['cells'])} cells "
              f"[{known[name]}]", file=sys.stderr)
    stamps_file.write_text(json.dumps(known, indent=1, sort_keys=True) + "\n")
