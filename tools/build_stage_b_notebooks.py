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
DRIVE_MY    = "/content/drive/MyDrive"

EVAL_DRAWN  = "kust4k"
EVAL_SPEC   = "kust4k:{root}:thermal:components:all"
SKIP_POOLS  = []
POOL_ROLE   = "all"
POOL_ROLES  = {"kaggle_uav_thermal": "train", "aerovis_train": "train",
               "aerovis_heldout": "eval"}
POOL_MODALITIES = {"aerovis_train": "rgb", "aerovis_heldout": "rgb"}
POOL_LIMITS = {"aerovis_train": 10000, "aerovis_heldout": 1500}
POOL_ZIP_MAX_MB = 2048

VTUAV_PARTS = []

IMAGE_ROOTS = {"kaggle_uav_thermal": "/content/data/kaggle_uav_thermal",
               "aerovis_train": "/content/data/AeroVIS",
               "aerovis_heldout": "/content/data/AeroVIS"}
KAGGLE_DATASETS = {
    "kaggle_uav_thermal": "umuttuygurr/aerial-uav-thermal-inferred-unified-dataset",
}

SOURCE_ZIPS = [
    ["/content/drive/MyDrive/edgetam-pool/segfly/segfly.zip", "SegFly"],
    ["/content/drive/MyDrive/edgetam-pool/kust4k/29476610.zip", "Kust4K"],
]

IMAGES = [
    ["hituav",       "hituav",       "HIT_UAV",      []],
    ["dronevehicle", "dronevehicle", "DroneVehicle", ["train"]],
    ["kust4k",       "kust4k",       "Kust4K",       ["tir", "labels", "rgb"]],
    ["visdrone",     "visdrone",     "VisDrone",     []],
    ["segfly_rgb",   "",             "SegFly",       []],
    ["segfly",       "",             "SegFly",       []],
    ["vtuav",        "vtuav_track",  "VTUAV",        VTUAV_PARTS],
    ["kaggle",       "",             "kaggle_uav_thermal", []],
    ["aerovis",      "",             "AeroVIS",      []],
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

REQUIREMENTS = [_line.split("#")[0].strip() for _line in
                (Path(REPO_DIR) / "requirements.txt").read_text().splitlines()]
REQUIREMENTS = [_line for _line in REQUIREMENTS
                if _line and not _line.lower().startswith("opencv")]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *REQUIREMENTS],
               check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hf_transfer",
                "tqdm", "matplotlib"], check=False)

import importlib.metadata as _md
assert _md.version("torch") == _TORCH_WAS, (
    f"pip replaced torch {_TORCH_WAS} with {_md.version('torch')} on disk. "
    f"Restore it before restarting: pip install --force-reinstall "
    f"torch=={_TORCH_WAS}")

import importlib

OPENCV = ["opencv-python", "opencv-python-headless", "opencv-contrib-python",
          "opencv-contrib-python-headless"]

def cv2_works():
    try:
        import cv2
        return hasattr(cv2, "imread")
    except Exception:
        return False

CV2_REPAIRED = False
if not cv2_works():
    print("cv2 is broken -- reinstalling one distribution over the several "
          "that share its directory")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                    *OPENCV], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "opencv-python-headless>=4.8"], check=False)
    for _stale in [n for n in list(sys.modules) if n.split(".")[0] == "cv2"]:
        del sys.modules[_stale]
    importlib.invalidate_caches()
    CV2_REPAIRED = True
assert cv2_works(), (
    "cv2 imports but has no imread, and reinstalling did not fix it in this "
    "kernel. Runtime > Restart session (your /content files survive it) and "
    "run this cell again -- the repair is already on disk.")
if CV2_REPAIRED:
    print("cv2 repaired; restart the runtime if anything below still fails")

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
from src.training.pool_reader import (acceptance, discover_pools,
                                      group_records, link_pool)

_done = Path(POOL_ROOT) / ".unpacked"
_done.mkdir(parents=True, exist_ok=True)
assert Path(DRIVE_POOLS).is_dir(), f"{DRIVE_POOLS} is not there -- set DRIVE_POOLS"

for _zip in sorted(Path(DRIVE_POOLS).rglob("*.zip")):
    _size = round(_zip.stat().st_size / 2 ** 20, 1)
    _marker = _done / (str(_zip.relative_to(DRIVE_POOLS)).replace("/", "__")
                       + ".done")
    if _marker.is_file():
        continue
    if _size > POOL_ZIP_MAX_MB:
        print("skipped", _zip.name, _size, "MiB -- a pool holds masks only, "
              "so this is source data, not a pool")
        continue
    try:
        with zipfile.ZipFile(_zip) as _handle:
            _members = _handle.namelist()
            if not any(_m.endswith("record.json") for _m in _members):
                print("skipped", _zip.name, "-- no record.json in it")
                continue
            _handle.extractall(POOL_ROOT)
    except Exception as _unzip_error:
        print("!! could not read", _zip.name, "--", _unzip_error)
        continue
    _marker.parent.mkdir(parents=True, exist_ok=True)
    _marker.touch()
    print("unpacked", _zip.name, _size, "MiB,", len(_members), "files")

for _name, _folder in discover_pools(DRIVE_POOLS).items():
    _marker = _done / (_name + ".folder.done")
    if _marker.is_file() or (Path(POOL_ROOT) / _name).is_dir():
        continue
    shutil.copytree(_folder, Path(POOL_ROOT) / _name, dirs_exist_ok=True)
    _marker.touch()
    print("copied", _name, "from Drive (it was staged unzipped)")

print()
RAW = group_records(POOL_ROOT)
POOLS, TEACHERS, HARVEST = {}, {}, {}
print(f"{'pool':<28}{'frames':>8}{'boxes':>9}{'masks':>9}{'kept':>7}   "
      f"{'first gate that stopped the rest':<34}teacher")
for _name, _records in RAW.items():
    POOLS[_name] = str(link_pool(_records, Path(POOL_ROOT) / "_by_name" / _name))
    HARVEST[_name] = acceptance(_records)
    TEACHERS[_name] = "+".join(HARVEST[_name]["teachers"])
    _top = ", ".join(f"{k} {v}" for k, v in
                     list(HARVEST[_name]["rejected"].items())[:2]) or "nothing"
    print(f"{_name:<28}{HARVEST[_name]['frames']:>8}"
          f"{HARVEST[_name]['attempted']:>9}{HARVEST[_name]['accepted']:>9}"
          f"{HARVEST[_name]['rate']:>7.1%}   {_top:<34}{TEACHERS[_name]}")
print("\\na box the first gate stops is never measured against its own "
      "annotation: `reject_reason` returns at the first failure, and "
      "`teacher_iou` is the teacher's own confidence, which this repo "
      "measured as the weak gate. A pool whose rejects pile up there has "
      "not been shown to hold bad masks -- only unsure ones.")
FOUND = sorted(POOLS)
assert FOUND, f"no record.json under {POOL_ROOT} -- nothing was unpacked"

GUESSED = set()

def modality_of(pool):
    if pool in POOL_MODALITIES:
        return POOL_MODALITIES[pool]
    GUESSED.add(pool)
    lowered = pool.lower()
    if lowered.endswith("_rgb") or "_rgb_" in lowered or "rgb" in lowered.split("_"):
        return "rgb"
    return "thermal"

def images_for(pool):
    lowered = pool.lower()
    for key, recipe, folder, parts in IMAGES:
        if key in lowered:
            root = IMAGE_ROOTS.get(pool) or str(Path(DATA_ROOT) / folder)
            return key, recipe, root, list(parts)
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
    else:
        PLAN.append({"pool": _pool, "dir": str(POOLS[_pool]), "key": _key,
                     "recipe": _recipe, "images": _root, "parts": _parts,
                     "modality": _modality,
                     "role": POOL_ROLES.get(_pool, POOL_ROLE)})

print(f"\\n{'pool':<28}{'modality':<10}{'role':<7}{'frames from':<40}")
for _row in PLAN:
    print(f"{_row['pool']:<28}{_row['modality']:<10}{_row['role']:<7}"
          f"{_row['images']:<40}")
for _pool, _role in POOL_ROLES.items():
    if _role == "train" and any(r["pool"] == _pool for r in PLAN):
        print(f"   {_pool} feeds training only -- it never reaches val or test, "
              f"so it cannot inflate the score with a domain the deployment "
              f"does not have.")
_guessed = sorted(_pool for _pool in GUESSED & {_r["pool"] for _r in PLAN}
                  if "rgb" not in _pool.lower()
                  and "thermal" not in _pool.lower()
                  and "tir" not in _pool.lower())
if _guessed:
    print(f"   !! modality guessed for {_guessed} -- their names say neither. "
          f"A pool not naming `rgb` is read as thermal, which converts colour "
          f"frames to grey and trains them as thermal. Put them in "
          f"POOL_MODALITIES if that is wrong.")
if {"visdrone", "aerovis"} <= {_row["key"] for _row in PLAN}:
    print("   !! a VisDrone pool and an AeroVIS pool are both in this run. "
          "AeroVIS *is* VisDrone re-labelled (plus UAVDT and SeaDronesSee), "
          "so they are the same frames and no AeroVIS grade is held out any "
          "more. Drop one.")
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

def unpack(archive, target, label=""):
    """Every member the archive can still give, one at a time.

    `extractall` stops at the first bad member and leaves the rest on the
    floor. A Drive copy read by two runtimes at once returns a bad CRC often
    enough that one corrupt JPEG out of seventy thousand must not cost the
    other sixty-nine thousand. Already-extracted members are skipped, so this
    is also the resume path.
    """
    bad, taken = [], 0
    with zipfile.ZipFile(archive) as handle:
        members = handle.namelist()
        for member in members:
            landing = Path(target) / member
            if landing.exists() and (landing.is_dir() or landing.stat().st_size):
                continue
            try:
                handle.extract(member, target)
                taken += 1
            except Exception:
                bad.append(member)
    print(f"   {label or Path(archive).name}: {taken} extracted, "
          f"{len(members) - taken - len(bad)} already there, "
          f"{len(bad)} unreadable")
    if bad:
        print("   unreadable:", bad[:5], "..." if len(bad) > 5 else "")
    return bad

for _archive, _folder in SOURCE_ZIPS:
    _target = Path(DATA_ROOT) / _folder
    if not Path(_archive).is_file():
        print("not there, skipping:", _archive)
        continue
    if _target.is_dir() and any(
            any(_target.rglob(_glob)) for _glob in ("*.jpg", "*.png")):
        print("already on disk:", _target)
        continue
    print("unzipping", _archive,
          round(Path(_archive).stat().st_size / 2 ** 30, 2), "GiB ->", _target)
    unpack(_archive, _target, _folder)

if FETCH:
    from tools.fetch_datasets import fetch, staged
    Path(STAGE_DIR).mkdir(parents=True, exist_ok=True)
    for (_recipe, _root), _parts in sorted(_wanted.items()):
        if not _recipe or (_recipe == "vtuav_track" and not _parts):
            print("not fetched:", _root, "-- no recipe, or no parts named. "
                  "SegFly is a 761-shard parquet plan, VTUAV is ~16 GiB per "
                  "part and Kaggle sets come through kagglehub below, so none "
                  "of them is downloaded on a whim. Stage it, name its parts "
                  "in cell 1, or let the pool drop.")
            continue
        if Path(_root).is_dir() and any(
                any(Path(_root).rglob(_glob)) for _glob in ("*.jpg", "*.png")):
            print("already on disk:", _root)
            continue
        try:
            fetch(_recipe, Path(_root), tuple(sorted(_parts)) or None,
                  stage=STAGE_DIR,
                  staging=(STAGE_DIR, str(Path(DRIVE_MY) / Path(_root).name),
                           DRIVE_MY, "/content/staging"))
        except Exception as _fetch_error:
            print("!!", _recipe, "->", _root, "failed:", _fetch_error)
            _search = (STAGE_DIR, str(Path(DRIVE_MY) / Path(_root).name),
                       DRIVE_MY, "/content/staging")
            for _part in sorted(_parts) or [_recipe]:
                _copy = staged(_part, _search)
                if _copy is None:
                    continue
                print("   retrying", _copy.name, "member by member")
                unpack(_copy, Path(_root), _part)

for _pool, _slug in KAGGLE_DATASETS.items():
    if not any(_row["pool"] == _pool for _row in PLAN):
        continue
    _root = Path(IMAGE_ROOTS.get(_pool, ""))
    if _root and _root.is_dir() and any(
            any(_root.rglob(_glob)) for _glob in ("*.jpg", "*.png")):
        print("already on disk:", _root)
        continue
    try:
        import kagglehub
        _got = kagglehub.dataset_download(_slug)
        _root.parent.mkdir(parents=True, exist_ok=True)
        if not _root.exists():
            os.symlink(_got, _root)
        print("kaggle:", _slug, "->", _got)
    except Exception as _kaggle_error:
        print("!! could not fetch", _slug, "--", _kaggle_error)
        print("   put its frames under", _root, "or set IMAGE_ROOTS in cell 1")

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
import numpy as np

from src.training.aerial import InstanceGates, sample_windows, split_index
from src.training.datasets import parse
from src.training.image_loop import ImageSplit
from src.training.pool_reader import (SKIP_REASONS, exclude_frames, frame_keys,
                                      index_pool, load_pool_index, parse_pool,
                                      save_pool_index, spread)
from tools.train_encoder import build_indexes

GATES = InstanceGates(min_area=MIN_AREA, min_side=MIN_SIDE, max_area=MAX_AREA,
                      fill=FILL)

INDEX, DATASET_FLAGS, DRAWN_HELD = [], [], set()
if EVAL_DRAWN:
    try:
        _drawn_flag = EVAL_SPEC.format(root=EVAL_ROOT)
        _drawn = build_indexes([parse(_drawn_flag, GATES)], Path(INDEX_DIR),
                               WORKERS)
        _parts = split_index(_drawn, seed=SEED)
        DRAWN_HELD = frame_keys(_parts["val"]) | frame_keys(_parts["test"])
        DATASET_FLAGS.append(_drawn_flag)
        INDEX.extend(_drawn)
        print(f"drawn grade: {EVAL_DRAWN}, {len(_drawn)} frames, "
              f"{sum(len(e.instances) for e in _drawn)} instances, "
              f"{len(DRAWN_HELD)} frames held out of it")
        print("   these are drawn *semantic* maps decomposed into instances, "
              "so where the decomposition fused two vehicles a model that "
              "separates them is scored wrong. It is the best drawn "
              "annotation here without VTUAV's archives; read it as a floor.")
    except (ValueError, FileNotFoundError, AssertionError) as _drawn_error:
        print(f"!! no drawn grade: {EVAL_DRAWN} -> "
              f"{str(_drawn_error).splitlines()[0]}")
        print("   the test split becomes the pools' own held-out slice, so "
              "its truth is the teacher's. Read the number knowing that.")

print()
POOL_FLAGS, FAILED = [], []
for _row in PLAN:
    _flag = f"{_row['dir']}:{_row['images']}:{_row['modality']}:{_row['role']}"
    _request = parse_pool(_flag, GATES)
    _cache = Path(INDEX_DIR) / f"{_request.cache_name}.json"
    try:
        if _cache.is_file():
            _part = load_pool_index(_cache, GATES, _row["role"])
        else:
            _part = index_pool(_request.pool, _request.images, _request.modality,
                               _row["role"], GATES, _request.name,
                               workers=WORKERS, progress=progress)
            save_pool_index(_cache, _part)
    except (ValueError, FileNotFoundError) as _index_error:
        FAILED.append((_row["pool"], str(_index_error).splitlines()[0]))
        continue
    _skips = {k: v for k, v in _part[0].rejects.items() if k in SKIP_REASONS}
    _capped = ""
    _cap = POOL_LIMITS.get(_row["pool"])
    if _cap and len(_part) > _cap:
        _was, _seqs = len(_part), len({e.frame.name.rsplit("/", 1)[0]
                                       for e in _part})
        _part = spread(_part, _cap, SEED)
        _kept = len({e.frame.name.rsplit("/", 1)[0] for e in _part})
        _capped = (f"  (spread to {_cap} of {_was} over {_kept}/{_seqs} "
                   f"sequences)")
    _leak = ""
    if DRAWN_HELD and _row["key"] == EVAL_DRAWN:
        _before = len(_part)
        _part = exclude_frames(_part, DRAWN_HELD)
        _cut = _before - len(_part)
        assert _cut, (
            f"{_row['pool']} is built from {EVAL_DRAWN}'s frames but shares "
            f"none of the {len(DRAWN_HELD)} held-out keys, so the two readers "
            f"name frames differently and the overlap cannot be removed. Set "
            f"EVAL_DRAWN = None, or drop this pool.")
        _leak = f"  (-{_cut} frames the drawn grade holds out)"
    if not _part:
        FAILED.append((_row["pool"], "nothing left after the overlap filter"))
        continue
    POOL_FLAGS.append(_flag)
    INDEX.extend(_part)
    print(f"{_row['pool']:<28}{len(_part):>7} frames "
          f"{sum(len(e.instances) for e in _part):>9} instances  "
          f"{_row['modality']:<8}{TEACHERS[_row['pool']]}"
          f"{('  skipped ' + str(_skips)) if _skips else ''}{_capped}{_leak}")

for _pool, _why in FAILED:
    print(f"{_pool:<28}unusable  {_why}")
assert POOL_FLAGS, "no pool resolved its frames -- check the IMAGES roots above"

_used = {TEACHERS[_row["pool"]] for _row in PLAN
         if any(_row["dir"] in _f for _f in POOL_FLAGS)}
if len(_used) > 1:
    print(f"\\n!! two teachers in one training set: {sorted(_used)}. The run "
          f"still works, but a per-pool difference now has two causes and "
          f"neither can be read off the result.")

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

DATASET_OF = {f"pool/{_row['pool']}": _row["key"] for _row in PLAN}
if EVAL_DRAWN:
    DATASET_OF[EVAL_DRAWN] = EVAL_DRAWN

def dataset_of(entry):
    _spec = entry.source.spec.name if entry.source else ""
    return DATASET_OF.get(_spec, _spec)

def stem_of(entry):
    return Path(entry.frame.name).stem.lower()

GRADED = {}
for _name in ("val", "test"):
    for _entry in SPLITS[_name]:
        GRADED.setdefault(dataset_of(_entry), set()).add(stem_of(_entry))

_kept, _dropped = [], {}
for _entry in SPLITS["train"]:
    _key = dataset_of(_entry)
    if stem_of(_entry) in GRADED.get(_key, ()):
        _dropped[_key] = _dropped.get(_key, 0) + 1
    else:
        _kept.append(_entry)
SPLITS["train"] = _kept
if _dropped:
    print("dropped from training, graded elsewhere in the same dataset:")
    for _key, _count in sorted(_dropped.items(), key=lambda kv: -kv[1]):
        print(f"  {_key:<24}{_count:>7} frames")
    print("   registered pairs are the same scene twice -- DroneVehicle's RGB "
          "and thermal halves, Kust4K's -- and each half is its own source "
          "with its own permutation, so a frame can train in one and be "
          "scored in the other. Same geometry, same objects, same mask.")

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
_by_modality = {}
for _sample in TRAIN.samples:
    _key = "thermal" if (_sample.source is None or _sample.source.gray) else "rgb"
    _by_modality[_key] = _by_modality.get(_key, 0) + 1
_total = max(sum(_by_modality.values()), 1)
print("  " + "  ".join(f"{_k}: {_v} ({_v / _total:.0%})"
                       for _k, _v in sorted(_by_modality.items())))
print("\\ntest windows by source:")
for _source, _count in TEST.sources.items():
    print(f"  {_source:<34}{_count:>8}")
assert not ({id(e) for e in SPLITS["train"]} & {id(e) for e in SPLITS["test"]})
assert TEST.samples, ("the test split is empty -- every pool is role=train "
                      "and there is no drawn grade, so nothing can be scored")

_leaked = DRAWN_HELD & frame_keys(
    [e for e in SPLITS["train"] if e.source and e.source.mode == "pool"])
assert not _leaked, (
    f"{len(_leaked)} frames the drawn grade holds out are still in a pool's "
    f"training half, e.g. {sorted(_leaked)[:3]}")
print(f"\\nno pool trains on any of the {len(DRAWN_HELD)} frames the drawn "
      f"grade holds out" if DRAWN_HELD else "\\nno drawn grade to protect")
''')


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

_shuffled = np.random.default_rng(SEED).permutation(len(TEST.samples))
PANEL_POOL = [TEST.samples[int(i)] for i in _shuffled[:PANEL_WINDOWS]]
_panel_sources = {}
for _sample in PANEL_POOL:
    _name = _sample.source.spec.name if _sample.source else "?"
    _panel_sources[_name] = _panel_sources.get(_name, 0) + 1
print("panel pool:", len(PANEL_POOL), "windows from", len(_panel_sources),
      "sources", dict(sorted(_panel_sources.items(), key=lambda kv: -kv[1])))
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
_tally = {}
for _case in CASES:
    _name = (_case["sample"].source.spec.name if _case["sample"].source else "?")
    _cell = _tally.setdefault(_name, [0, 0, 0.0])
    _cell[0 if _case["delta"] > 0.01 else 1] += 1 if abs(_case["delta"]) > 0.01 else 0
    _cell[2] += _case["delta"]
print(f"{'source':<34}{'better':>8}{'worse':>8}{'mean delta':>12}")
for _name, (_up, _down, _sum) in sorted(_tally.items(), key=lambda kv: -kv[1][2]):
    _n = sum(1 for c in CASES
             if (c["sample"].source.spec.name if c["sample"].source else "?") == _name)
    print(f"{_name:<34}{_up:>8}{_down:>8}{_sum / max(_n, 1):>+12.4f}")
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
    "pools": POOL_FLAGS, "datasets": DATASET_FLAGS, "teachers": TEACHERS,
    "harvest": HARVEST,
    "dropped": DROPPED, "unusable": FAILED,
    "frames": {k: len(v) for k, v in SPLITS.items()},
    "train_windows_by_source": TRAIN.sources,
    "test_windows_by_source": TEST.sources,
    "gates": GATES.__dict__, "batch": BATCH, "lr_scale": LR_SCALE,
    "roles": {_row["pool"]: _row["role"] for _row in PLAN},
    "modalities": {_row["pool"]: _row["modality"] for _row in PLAN},
    "limits": POOL_LIMITS,
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
