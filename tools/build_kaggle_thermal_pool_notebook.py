#!/usr/bin/env python3
"""Generate notebook 19: the Kaggle merged UAV-thermal set, cut to scale, masked.

`umuttuygurr/aerial-uav-thermal-inferred-unified-dataset` is 27 925 images and
90 448 YOLO boxes over seven classes, merged from six sources, CC0, 13.21 GB.
It is a genuinely useful thermal pool source and it is **not** what its title
promises, which is the whole reason this notebook has a filter cell before it
has a teacher cell. Its own manifest says so:

    llvip       15 488   street-level surveillance pairs, pole-mounted
    uav2uav      3 856   drone filming drone, target against sky
    mixdataset   2 975   mixed perspective
    hituav       2 898   genuinely aerial thermal, 640x512, 80-130 m
    gundataset   1 900   hand-held weapons, close range
    munitions      808   mines, close range

More than half is ground-level and roughly a tenth is photographs of objects
held at arm's length. Trained on undifferentiated, the close-ups set the scale:
they carry the most pixels per box, so they dominate the loss, and a model
meant for 100 m learns three metres.

Scale is the axis that separates them, and `boxes.filter_by_scale` cuts on it
at two levels -- per box (outside `[MIN_REL, MAX_REL]` goes) and per image (a
frame whose *median* box is a close-up goes entirely, so one lorry among twelve
cars keeps its cars). The large band is not deleted, it is **capped**: near
targets stay in at `LARGE_QUOTA` of the selection, because a model that never
sees one forgets they exist. Every threshold is printed against the data by
`scale_table` in the cell before it is applied, per source and per class.

The order the cells run in is the point:

  1. environment
  2. download the archive; extract **only** manifest.csv and the label files
  3. index from those labels; print the scale table and probe each source's
     modality -- twelve images on disk, no more
  4. filter, then extract only the images that survived
  5. SAM 3 (no fallback), and the accepted masks drawn before the run
  6. harvest
  7. pool, prompts and the filter report back to Drive

Step 2 and 3 are what make this cheap. The labels are a few tens of megabytes
against 13 GB of PNGs, so the whole selection is decided before the disk is
spent -- `boxes.yolo_label_frames` exists for exactly that, indexing from the
label files with `image` naming where each frame *will* be.

`SOURCES` therefore takes **one** of the six, and the five it leaves out are
left out for three different kinds of reason, which is worth keeping straight.
**gundataset and munitions** are photographs of objects at arm's length -- the
scale filter would cut most of them anyway, and cutting them by name is both
cheaper and honest about why. **hituav** is excluded despite being the only
unambiguously aerial thermal part of the collection, because
`tools/fetch_datasets.py hituav` already pulls those same 2 898 frames at
0.4 GB with their **original** COCO annotations rather than a re-encode
through a seven-class remap. Harvesting both routes would put the same frames
in the pool twice under two names, which is the one way a held-out split stops
being held out. **uav2uav (3 856) and mixdataset (2 975)** are out because the
user decided to take only `llvip` -- not because anything here measured them
and found them wanting, and this file will not invent a justification it does
not have. The modality probe in cell 3 stays exactly where it is: it costs
twelve images per source and it is the thing that would answer whether either
is worth revisiting, which is an argument for keeping it, not for quietly
dropping it along with the sources.

So what is left is what the download is actually for. **LLVIP** is 15 488
infrared frames, and it is the only reason 13.21 GB is worth pulling at all;
without it this notebook would be a slow way to fetch a few thousand images.
Its camera is on a pole rather than a drone, which is a real objection and not
one to wave away -- it is in anyway on two grounds. First, this pool trains
box-to-mask, and where a mask boundary lies inside a given box is a local
skill that survives a change of viewpoint far better than detection does;
a detector trained on street level and flown at 100 m is lost, a segmenter
handed a box is mostly not. Second, LLVIP's targets sit at a relative scale of
roughly 0.05-0.2, which overlaps the aerial band the filter is aiming at, so
the scale objection that sinks the close-up sources does not apply here. One
caveat travels with LLVIP: its **car boxes are pseudo-labels** from a
pretrained detector (5 484 boxes over 3 876 images), so a mask made from one
is a pseudo-label of a pseudo-label. The prompts file records the source, so
they stay separable.

One last thing not to "simplify": `MIRROR_DIR` points at a *per-pool subfolder*
inside `edgetam-pool`, not at `edgetam-pool` itself. Cell 7 copies sidecars
into it under fixed names, and `pool_index.jsonl` is one of them -- notebook 18
writes a file of exactly that name into its own subfolder. Flatten either and
the two notebooks overwrite each other's index.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

NOTEBOOK = "19_kaggle_uav_thermal_pool.ipynb"
STAMP_VALUE = "unstamped"

CELLS: list[str] = []


def code(text: str) -> None:
    CELLS.append(text.strip("\n"))


# --------------------------------------------------------------------------
# 1. Settings, environment, and the batch this card can take
# --------------------------------------------------------------------------

code('''
KAGGLE_DATASET = "umuttuygurr/aerial-uav-thermal-inferred-unified-dataset"
DOWNLOAD_DIR = "/content/dl"
EXTRACT_DIR  = "/content/data"
POOL_ROOT    = "/content/pool"
POOL_NAME    = "kaggle_uav_thermal"
MIRROR_DIR   = "/content/drive/MyDrive/edgetam-pool/kaggle_uav_thermal"
SPLITS       = ("train", "val")
CLASS_NAMES  = ("person", "car", "bicycle", "other_vehicle",
                "drone", "mine", "gun")
SOURCES      = ("llvip",)           # llvip only, by decision -- see cell 3
DROP_CLASSES = ("gun", "mine")                      # close-range by nature
MIN_REL      = 0.01          # below this a box is a few pixels
MAX_REL      = 0.35          # above this the frame is a close-up
LARGE_REL    = 0.15          # the "near target" band starts here
LARGE_QUOTA  = 0.15          # ...and is capped at this share of the selection
SEED         = 0
TEACHER      = "facebook/sam3"       # no fallback, by request
DTYPE        = "bfloat16"
ZOOM         = 4.0
MIN_SIZE     = 128
BATCH        = 0
FRAME_GROUP  = 0
MAX_BOXES    = 64
LIMIT        = None
BOX_IOU      = 0.5
EXPORT_PNG   = True
KEEP_ARCHIVE = False         # True keeps the 13 GB zip after extraction
REQUIRE_DRIVE = True         # False runs without Drive, no mirror
PATCH_FILE   = "/content/drive/MyDrive/edgetam-pool/repo_patch.py"
REPO_URL     = "https://github.com/yigitkayabagci/sam-dedection.git"
BRANCH       = "claude/aerial-rgb-thermal-data-qhjpxd"
REPO_DIR     = "/content/sam-dedection"
NOTEBOOK = "19_kaggle_uav_thermal_pool.ipynb"
STAMP    = "{{STAMP}}"

import json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

# An existing clone is *updated*, not reused as-is. The old version cloned
# only when the folder was absent, so a push landed while a runtime was alive
# was invisible and the notebook went on running last week's code.
if not Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                    REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "--depth", "1", "origin",
                    BRANCH], check=False)
    subprocess.run(["git", "-C", REPO_DIR, "reset", "--hard",
                    f"origin/{BRANCH}"], check=False)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Pillow is deliberately **not** in this list. Colab ships one, the kernel
# has already imported it, and upgrading it underneath leaves a mixed
# site-packages -- a new `ImageDraw.py` beside an old `_typing.py` -- whose
# symptom is `cannot import name '_Ink' from 'PIL._typing'` raised from deep
# inside transformers, where it reads as a SAM 3 problem. Nothing here needs
# a newer Pillow: `pool._image_size` prefers it and falls back to OpenCV.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                "transformers>=5.0.0", "accelerate", "huggingface_hub",
                "kaggle", "pandas", "tqdm"], check=False)

# transformers 5 is a large upgrade over the preinstalled one, so the kernel
# that just installed it is usually still holding the old modules. Probing
# the one class this notebook needs turns that into a restart instruction
# here, instead of a 30-frame traceback three cells and one dataset later.
try:
    from transformers import Sam3TrackerModel as _probe
    del _probe
    TRANSFORMERS_OK = True
except Exception as _transformers_error:
    TRANSFORMERS_OK = False
    print("transformers is not usable yet:", _transformers_error)
    print("--> Runtime > Restart session, then run this cell again.")
    print("If the message mentions PIL._typing or _Ink, Pillow is "
          "half-upgraded; repair it first with:")
    print("    !pip install -q --force-reinstall pillow")
if not TRANSFORMERS_OK:
    raise SystemExit("restart the runtime, then re-run this cell")

# The mount is retried with `force_remount` before it is believed, because a
# half-finished earlier mount fails the plain call and succeeds the forced
# one. And a failure is fatal by default rather than printed: MIRROR_DIR is on
# Drive, so a run that cannot mount is a run that harvests for hours and then
# has nowhere to put the result.
DRIVE_OK = False
try:
    from google.colab import drive as _drive
except ImportError:
    _drive = None
    print("not a Colab runtime; Drive skipped")
if _drive is not None:
    for _forced in (False, True):
        try:
            _drive.mount("/content/drive", force_remount=_forced)
            DRIVE_OK = True
            break
        except Exception as _mount_error:
            print("drive mount failed:", _mount_error,
                  "-- retrying with force_remount" if not _forced else "")
if not DRIVE_OK and REQUIRE_DRIVE:
    raise SystemExit(
        "Drive is not mounted and MIRROR_DIR is on it.\\n"
        "  - re-run this cell and accept the authorisation popup; a popup "
        "blocker or blocked third-party cookies is the usual cause\\n"
        "  - Runtime > Disconnect and delete runtime, then try again\\n"
        "  - or set REQUIRE_DRIVE = False to run anyway and copy the pool out "
        "by hand")

# Two secrets, both set by you in the Colab key panel and never printed here:
# HF_TOKEN for the gated SAM 3 repo, KAGGLE_USERNAME + KAGGLE_KEY for the
# dataset. `~/.kaggle/kaggle.json` works instead if you already have one.
try:
    from google.colab import userdata as _userdata
    for _name in ("HF_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        try:
            _value = _userdata.get(_name)
        except Exception:
            _value = None
        if _value:
            os.environ[_name] = _value
except Exception as _secret_error:
    print("no Colab secrets:", _secret_error)

# What the clone actually contains, checked here rather than discovered as an
# ImportError four cells in: a branch that has not been pushed looks exactly
# like one that has, right up to the traceback. `PATCH_FILE` is the way out
# when pushing is not available -- a Drive-side copy of the functions the
# clone is missing, applied only if it is missing them, so it becomes a no-op
# the day the branch lands and can be deleted then.
REQUIRED = {
    "src.training.boxes": ("yolo_label_frames", "filter_by_scale",
                           "scale_table", "box_scale"),
}

def _missing():
    import importlib
    out = []
    for _module_name, _names in REQUIRED.items():
        try:
            _module = importlib.import_module(_module_name)
        except Exception as _import_error:
            out.append(f"{_module_name} ({_import_error})")
            continue
        out += [f"{_module_name}.{_n}" for _n in _names
                if not hasattr(_module, _n)]
    return out

_stale = _missing()
if _stale and Path(PATCH_FILE).is_file():
    print("clone is behind; applying", PATCH_FILE)
    exec(compile(Path(PATCH_FILE).read_text(), PATCH_FILE, "exec"),
         {"__name__": "repo_patch"})
    _stale = _missing()
if _stale:
    raise SystemExit(
        "this clone of " + BRANCH + " is missing:\\n  " + "\\n  ".join(_stale)
        + "\\n\\nEither push the branch:\\n"
        "    git push origin " + BRANCH + "\\n"
        "or put an up-to-date repo_patch.py at " + PATCH_FILE)


# No `login()` call. `huggingface_hub` reads HF_TOKEN from the environment on
# its own, and calling login as well makes it print a warning that the env var
# is the active token "independently from the token you've just configured" --
# which reads like an access problem and is not one.
KAGGLE_OK = bool(os.environ.get("KAGGLE_USERNAME")
                 and os.environ.get("KAGGLE_KEY")) \
    or Path("~/.kaggle/kaggle.json").expanduser().is_file()

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
      "| BATCH", BATCH, "| FRAME_GROUP", FRAME_GROUP,
      "| HF_TOKEN", "set" if os.environ.get("HF_TOKEN") else "MISSING")
if not KAGGLE_OK:
    # Two names, both needed, and the toggle is the part people miss: a Colab
    # secret exists but is unreadable until "Notebook access" is on for this
    # notebook, and `userdata.get` then raises rather than returning None.
    print("KAGGLE creds MISSING. kaggle.com > Settings > API > Create New "
          "Token downloads kaggle.json; then either")
    print("  a) Colab key icon (left bar): add KAGGLE_USERNAME and KAGGLE_KEY "
          "from that file, turn ON 'Notebook access' for BOTH, re-run")
    print("  b) upload kaggle.json and run in a scratch cell:")
    print("     !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && "
          "chmod 600 ~/.kaggle/kaggle.json")
else:
    print("KAGGLE creds: ok")
''')


# --------------------------------------------------------------------------
# 2. Download, and extract the labels only
# --------------------------------------------------------------------------

code('''
# The archive is 13.21 GB of PNGs and a few tens of megabytes of label text.
# Extracting the text first means the whole selection is decided before any
# image is written -- see cell 4, which extracts only what survives the filter.
Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
ARCHIVE = next(iter(sorted(Path(DOWNLOAD_DIR).glob("*.zip"))), None)
if ARCHIVE is None:
    if not KAGGLE_OK:
        raise SystemExit("no Kaggle credentials -- see what cell 1 printed")
    subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
                    "-p", DOWNLOAD_DIR], check=True)
    ARCHIVE = next(iter(sorted(Path(DOWNLOAD_DIR).glob("*.zip"))))
print(ARCHIVE, round(ARCHIVE.stat().st_size / 2 ** 30, 2), "GiB")

ZIP = zipfile.ZipFile(ARCHIVE)
MEMBERS = ZIP.namelist()
LABEL_MEMBERS = [n for n in MEMBERS if "/labels/" in n and n.endswith(".txt")]
META_MEMBERS = [n for n in MEMBERS
                if n.endswith(("manifest.csv", "data.yaml", ".json"))]
if not LABEL_MEMBERS:
    raise SystemExit(f"{ARCHIVE}: no labels/*.txt members -- has the dataset "
                     f"been repacked? First few names: {MEMBERS[:5]}")

_manifest_member = next((n for n in META_MEMBERS if n.endswith("manifest.csv")),
                        None)
ROOT_REL = (Path(_manifest_member).parent if _manifest_member
            else Path(LABEL_MEMBERS[0]).parents[2])
DATA_ROOT = str(Path(EXTRACT_DIR) / ROOT_REL)

for _member in progress(LABEL_MEMBERS + META_MEMBERS,
                        total=len(LABEL_MEMBERS) + len(META_MEMBERS),
                        desc="labels"):
    ZIP.extract(_member, EXTRACT_DIR)
print(len(LABEL_MEMBERS), "label files ->", DATA_ROOT)

import pandas as pd

MANIFEST_PATH = Path(DATA_ROOT) / "manifest.csv"
MANIFEST = pd.read_csv(MANIFEST_PATH) if MANIFEST_PATH.is_file() else None
if MANIFEST is None:
    print("!! no manifest.csv -- the per-source table in cell 3 will be blank")
    SOURCE_OF = {}
else:
    _column = ("output_filename" if "output_filename" in MANIFEST.columns
               else "output_image")
    SOURCE_OF = {Path(str(row[_column])).stem: str(row["dataset"])
                 for _, row in MANIFEST.iterrows()}
    print(MANIFEST.groupby(["dataset", "split"]).size().unstack(fill_value=0))
''')


# --------------------------------------------------------------------------
# 3. The scale table: where the thresholds come from
# --------------------------------------------------------------------------

code('''
# Nothing is filtered here. This cell exists so the numbers in cell 1 are read
# off the data rather than guessed -- p50 is the typical target, p99 and the
# last column say how much of a source is close-up. An aerial source sits
# around 0.03-0.08; a source that sits at 0.4 is photographs, whatever the
# collection is called.
from src.training import boxes as B

ALL_FRAMES = {}
for _split in SPLITS:
    ALL_FRAMES[_split] = B.yolo_label_frames(
        DATA_ROOT, names=CLASS_NAMES,
        images=f"images/{_split}", labels=f"labels/{_split}")
    print(_split, len(ALL_FRAMES[_split]), "annotated images /",
          sum(len(f.classes) for f in ALL_FRAMES[_split]), "boxes")

FLAT = [f for split in SPLITS for f in ALL_FRAMES[split]]
print("\\nby source dataset")
print(B.scale_table(FLAT, groups=SOURCE_OF or None, max_rel=MAX_REL))
print("\\nby class")
print(B.scale_table(FLAT, max_rel=MAX_REL))

# The collection is sold as thermal and its manifest never says which sources
# are. A thermal frame is single-channel written out as three, so the mean
# spread between a pixel's channels is ~0 for thermal and tens of counts for
# colour. Twelve images per source answers it for the price of twelve images.
#
# It probes every source, not only the ones in SOURCES. SOURCES is llvip alone
# because that was decided, not because uav2uav or mixdataset were measured
# and rejected -- and this table plus the one above it are what you would read
# before deciding whether either is worth adding back.
import numpy as np, cv2

_by_source = {}
for _frame in FLAT:
    _by_source.setdefault(SOURCE_OF.get(_frame.key, "?"), []).append(_frame)

MODALITY = {}
print("\\n| source | sampled | channel spread | mean | verdict |")
print("|---|---:|---:|---:|---|")
for _name, _frames in sorted(_by_source.items(), key=lambda kv: -len(kv[1])):
    _sample = _frames[:: max(len(_frames) // 12, 1)][:12]
    _spreads, _means = [], []
    for _frame in _sample:
        _member = str(Path(_frame.image).relative_to(EXTRACT_DIR))
        if not (Path(EXTRACT_DIR) / _member).is_file():
            if _member not in set(MEMBERS):
                continue
            ZIP.extract(_member, EXTRACT_DIR)
        _raw = cv2.imread(str(_frame.image), cv2.IMREAD_UNCHANGED)
        if _raw is None:
            continue
        if _raw.ndim == 2:
            _spreads.append(0.0)
        else:
            _rgb = _raw[:, :, :3].astype(np.int16)
            _spreads.append(float((_rgb.max(2) - _rgb.min(2)).mean()))
        _means.append(float(_raw.mean()))
    if not _spreads:
        continue
    _spread = float(np.mean(_spreads))
    MODALITY[_name] = "thermal" if _spread < 3.0 else "colour"
    print(f"| {_name} | {len(_spreads)} | {_spread:.1f} | "
          f"{np.mean(_means):.0f} | {MODALITY[_name]} |")
''')


# --------------------------------------------------------------------------
# 4. Filter, then extract only the images that survived
# --------------------------------------------------------------------------

code('''
from dataclasses import replace as _replace

def prefilter(frames):
    """Source and class cuts, before scale gets a say."""
    out = []
    for frame in frames:
        if SOURCES and SOURCE_OF.get(frame.key) not in SOURCES:
            continue
        keep = [i for i, c in enumerate(frame.classes) if c not in DROP_CLASSES]
        if not keep:
            continue
        out.append(frame if len(keep) == len(frame.classes) else _replace(
            frame, boxes=frame.boxes[keep],
            classes=tuple(frame.classes[i] for i in keep)))
    return out

KEPT, REPORTS = {}, {}
for _split in SPLITS:
    _frames = prefilter(ALL_FRAMES[_split])
    KEPT[_split], REPORTS[_split] = B.filter_by_scale(
        _frames, min_rel=MIN_REL, max_rel=MAX_REL, large_rel=LARGE_REL,
        large_quota=LARGE_QUOTA, seed=SEED)
    print(_split, json.dumps(REPORTS[_split], indent=1))

SELECTED = [f for split in SPLITS for f in KEPT[split]]
print("\\n", len(SELECTED), "images selected /",
      sum(len(f.classes) for f in SELECTED), "boxes")
if SOURCE_OF:
    _by_source = {}
    for _frame in SELECTED:
        _name = SOURCE_OF.get(_frame.key, "?")
        _by_source[_name] = _by_source.get(_name, 0) + 1
    print("kept per source:", dict(sorted(_by_source.items(),
                                          key=lambda kv: -kv[1])))
print()
print(B.scale_table(SELECTED, groups=SOURCE_OF or None, max_rel=MAX_REL))

# Only now is disk spent, and only on these.
_wanted = [str(Path(f.image).relative_to(EXTRACT_DIR)) for f in SELECTED]
_present = set(MEMBERS)
_missing = [w for w in _wanted if w not in _present]
if _missing:
    print(f"\\n!! {len(_missing)} selected images are not in the archive under "
          f"the name the labels imply, e.g. {_missing[0]} -- check the image "
          f"suffix (yolo_label_frames defaults to .png)")
for _member in progress([w for w in _wanted if w in _present],
                        total=len(_wanted) - len(_missing), desc="images"):
    if not (Path(EXTRACT_DIR) / _member).is_file():
        ZIP.extract(_member, EXTRACT_DIR)
ZIP.close()
if not KEEP_ARCHIVE:
    ARCHIVE.unlink()
    print("archive removed;", round(sum(
        p.stat().st_size for p in Path(DATA_ROOT).rglob("*") if p.is_file())
        / 2 ** 30, 2), "GiB on disk")
''')


# --------------------------------------------------------------------------
# 5. The teacher: SAM 3, and nothing else
# --------------------------------------------------------------------------

code('''
# Deliberately no fallback. `facebook/sam3` is gated and wants
# transformers>=5; both are a minute to fix, whereas a pool silently built on
# a different teacher is not visibly wrong anywhere downstream.
import matplotlib.pyplot as plt
from src.training.labels import Gates, build_image_teacher
from src.training.pool import label_boxes

try:
    TEACHER_MODEL = build_image_teacher(TEACHER, dtype=DTYPE)
except (Exception, SystemExit) as _teacher_error:
    raise SystemExit(
        f"{TEACHER} did not load: {str(_teacher_error).splitlines()[0]}\\n"
        f"  1. accept the licence at https://huggingface.co/{TEACHER}\\n"
        f"  2. put the token in a Colab secret named HF_TOKEN, re-run cell 1\\n"
        f"  3. SAM 3 needs transformers>=5.0.0 -- restart the runtime if the "
        f"install in cell 1 upgraded it under a running kernel")
GATES = Gates(box_iou=BOX_IOU)
print("teacher:", TEACHER_MODEL.model_id)

def read_rgb(path):
    _raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if _raw.ndim == 2:
        _raw = cv2.cvtColor(_raw, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(_raw[:, :, :3], cv2.COLOR_BGR2RGB)

# Two frames from the small band and one from the large, so the panel shows
# the scale range the filter actually let through rather than the median.
_small = [f for f in SELECTED if B.box_scale(f).max() < LARGE_REL][:2]
_large = [f for f in SELECTED if B.box_scale(f).max() >= LARGE_REL][:1]
_shown = _small + _large
_figure, _axes = plt.subplots(1, len(_shown), figsize=(6 * len(_shown), 6))
for _ax, _frame in zip(np.atleast_1d(_axes), _shown):
    _pixels = read_rgb(_frame.image)
    _boxes, _keep = _frame.resolved(_pixels.shape[:2])
    _masks, _rows = label_boxes(_pixels, _boxes[_keep], TEACHER_MODEL,
                                gates=GATES, zoom=ZOOM, min_size=MIN_SIZE,
                                batch_size=BATCH)
    _panel = _pixels.copy()
    for _mask in _masks.values():
        _panel[_mask] = (0.4 * _panel[_mask] +
                         0.6 * np.array([255, 40, 40])).astype(np.uint8)
    _ax.imshow(_panel)
    _ax.set_title(f"{SOURCE_OF.get(_frame.key, _frame.key)} "
                  f"| rel {B.box_scale(_frame).max():.3f} "
                  f"| {len(_masks)}/{len(_rows)}")
    _ax.axis("off")
plt.tight_layout()
plt.show()
''')


# --------------------------------------------------------------------------
# 6. Harvest
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

# The split goes in the key, not the pool name: one pool, and a training reader
# that wants to honour the original val split can still read it off the path.
POOL_REPORTS = {}
for _split in SPLITS:
    _keyed = [_replace(f, key=f"{_split}/{f.key}") for f in KEPT[_split]]
    POOL_REPORTS[_split] = harvest(_keyed, POOL_NAME)
    print(_split, json.dumps(POOL_REPORTS[_split], indent=1))

write_index(POOL_ROOT)
print(summarise_pool(pool_report(POOL_ROOT)))
''')


# --------------------------------------------------------------------------
# 7. Pool, prompts and the filter report, back to Drive
# --------------------------------------------------------------------------

code('''
from src.training.labels import MASK_STORE, open_masks

Path(MIRROR_DIR).mkdir(parents=True, exist_ok=True)
_pool_dir = Path(POOL_ROOT) / POOL_NAME

if EXPORT_PNG:
    for _store in sorted(_pool_dir.rglob(MASK_STORE)):
        _masks = open_masks(_store)
        _canvas = np.zeros(_masks.shape, dtype=np.uint16)
        for _index in sorted(_masks):
            _canvas[_masks[_index]] = _index + 1
        cv2.imwrite(str(_store.parent / "instances.png"), _canvas)

# Every box that was *offered*, rejects included: "the teacher was asked and
# refused" is a different fact from "nobody asked", and only one of them says
# something about the teacher.
PROMPTS = Path(POOL_ROOT) / f"{POOL_NAME}_prompts.jsonl"
with PROMPTS.open("w") as _handle:
    for _record in sorted(_pool_dir.rglob("record.json")):
        _row = json.loads(_record.read_text())
        _stem = Path(_row["key"]).name
        for _instance in _row["instances"]:
            _handle.write(json.dumps({
                "frame": _row["key"], "source": SOURCE_OF.get(_stem, "?"),
                "image": _row["image"], "box": _instance["box"],
                "class": _instance["class"], "i": _instance["i"],
                "teacher": _row["teacher"], "verdict": _instance["verdict"]})
                + "\\n")

(Path(POOL_ROOT) / f"{POOL_NAME}_selection.json").write_text(json.dumps({
    "dataset": KAGGLE_DATASET, "teacher": TEACHER_MODEL.model_id,
    "class_names": list(CLASS_NAMES), "sources": SOURCES,
    "drop_classes": list(DROP_CLASSES), "splits": list(SPLITS),
    "filter": REPORTS, "harvest": POOL_REPORTS}, indent=1) + "\\n")

_target = Path(MIRROR_DIR) / f"{POOL_NAME}.zip"
_files = [p for p in sorted(_pool_dir.rglob("*")) if p.is_file()]
with zipfile.ZipFile(_target, "w", zipfile.ZIP_STORED, allowZip64=True) as _zip:
    for _file in _files:
        _zip.write(_file, _file.relative_to(Path(POOL_ROOT)))
print(_target, len(_files), "files",
      round(_target.stat().st_size / 2 ** 20, 1), "MiB")

for _name in ("pool_index.jsonl", f"{POOL_NAME}_prompts.jsonl",
              f"{POOL_NAME}_selection.json"):
    _side = Path(POOL_ROOT) / _name
    if _side.is_file():
        shutil.copy(_side, Path(MIRROR_DIR) / _name)

print("\\nteacher:", TEACHER_MODEL.model_id,
      "| accepted:", sum(r["accepted"] for r in POOL_REPORTS.values()),
      "of", sum(r["attempted"] for r in POOL_REPORTS.values()), "boxes")
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
