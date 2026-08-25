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
  2. copy DroneVehicle off Drive, unpack it entry by entry, build the index
  3. load the teacher, draw two frames so the mirror is visible before it runs
  4. harvest the shared boxes in chunks, shipping each chunk to Drive
  5. harvest the two *-only sets the same way
  6. close the pool out: last shard, one zip per pool, index, manifest

Four pools, because provenance has to stay readable: `dronevehicle_rgb` and
`dronevehicle_thermal` are the shared boxes (one mask, two files),
`dronevehicle_thermal_only` is the 33 383 vehicles the visible half never
annotates -- night, mostly -- and `dronevehicle_rgb_only` is the 3 797 the
thermal half misses. A thermal training run reads the second and third; merging
them into one directory would make a mirrored mask indistinguishable from a
thermal-prompted one.

Seven decisions inside it are not arbitrary and are argued where they live
rather than in the notebook:

**Drive, not the Hub.** `fetch_datasets.py` pulls DroneVehicle from
`McCheng/DroneVehicle` at 8.88 GB over HTTPS. A copy already in the user's own
Drive is a mounted read with no download at all, which is why `DRIVE_DIR` is
the only data setting and it ships as a placeholder.

**A clone that is already there is updated, not trusted.** The cell used to
clone only when `REPO_DIR` was missing, which means a session that had ever
run an older revision kept it: the code the notebook imports and the code the
notebook was generated from silently drift apart, and the stamp line reports
STALE against a repo nobody asked to move. It now fetches the branch and
resets to it when the directory exists, with `check=False` on both -- an
offline runtime should run against what is on disk rather than refuse to
start, and the stamp line is what says which of those happened.

**Nothing is upgraded over a package the kernel already imported.** Cell 1's
pip line used to carry `pillow`, and `--upgrade pillow` inside a live Colab
session is a trap: Colab imports matplotlib, and through it `PIL`, before the
first cell runs, so pip writes Pillow 12 to disk under a kernel still holding
Pillow 11's `PIL._typing`. Nothing fails at that point. It fails two cells
later, when transformers is the first thing to want `PIL.ImageText` -- a
12-only module, so it loads from disk -- and that module asks the live 11 for
`_Ink`. `pillow` is off the install list, since `requirements.txt` needs only
>= 10 and Colab ships 11. What stays is the general check, because the next
dependency to move will not be Pillow: any watched package whose live
`__version__` differs from the one now on disk means pip moved house under
this kernel, and the cell restarts the runtime once -- marked in `/content`,
so it cannot loop -- instead of letting the mismatch surface three cells later
as somebody else's import error.

**The archive is copied to local disk before it is opened.** `extractall`
straight off the mount is what the notebook used to do, and on an 8.3 GiB zip
it reliably ends in `BadZipFile: Bad CRC-32`. Extraction is a seek-heavy read
-- central directory, then a jump per entry -- and the Drive FUSE mount answers
some of those jumps with bytes that are not the file's, without ever raising:
the CRC in the zip is the first thing to notice. One sequential copy is the
read pattern the mount is good at, and it is retried per chunk when it does
fail outright. Extraction then runs against local disk, where a CRC mismatch
means what it says -- the archive itself is damaged -- so the notebook can tell
the two apart and say which one happened. The copy is deleted once it is
unpacked, and skipped entirely if the disk cannot hold it.

**A bad entry drops one frame, not the whole run.** Entries are extracted one
at a time and a failed CRC removes that half-written file and moves on, because
a damaged image left on disk is worse than a missing one: `dronevehicle_shared_frames`
walks pairs and simply does not yield a frame whose twin or XML is absent,
while a truncated PNG surfaces as a `None` from `cv2.imread` in cell 3. Past
1 % of entries the archive is not worth working around and the cell says so.
The done-marker moved with it: the old one was the extracted `train/` directory,
which a run that died halfway through creates, so the next run reported
"already staged" over a half-unpacked dataset.

**A runtime that dies is the expected case, not the failure case.** This is
hours of A100 time against 33 000 frames, in a session whose owner has to
close the laptop. The old cell 6 wrote everything to Drive once, at the end,
which makes every hour before it worthless if the runtime goes away. The
harvest now runs in `CHUNK`-frame slices and ships each slice to Drive as a
numbered zip under `shards/` -- append-only, written to `.part` and renamed,
so a shard on Drive is either whole or absent. `restore()` at the top of cell
4 extracts every shard back into `POOL_ROOT` before anything runs, and
`label_pool(resume=True)` then skips the frames whose store is already there.
Re-running cells 1-5 in a fresh runtime is the recovery procedure, and it
costs one Drive read.

What ships is the *new* files only: the set of paths already on Drive is
rebuilt at restore from what the shards put on disk, not carried in a ledger
that could disagree with them. A shard write that fails after four tries
leaves `SHIPPED` untouched and prints why, so the next checkpoint carries
those files instead of losing them. The four consolidated `<pool>.zip` files
are still written at the end -- one zip per pool is what a training run wants
to unpack -- but they are now the convenience and the shards are the
guarantee.

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
STAGE_DIR   = "/content/stage"
ARCHIVES    = ["train.zip"]
CHUNK       = 400
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
BRANCH      = "claude/colab-notebook-15-zipfile-p5czwt"
REPO_DIR    = "/content/sam-dedection"
NOTEBOOK = "15_dronevehicle_shared_pool.ipynb"
STAMP    = "{{STAMP}}"

import json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

if not Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                    REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "-q", "--depth", "1",
                    "origin", BRANCH], check=False)
    subprocess.run(["git", "-C", REPO_DIR, "checkout", "-q", "-B", BRANCH,
                    "FETCH_HEAD"], check=False)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                "transformers>=5.0.0", "accelerate", "huggingface_hub",
                "tqdm"], check=False)

import importlib.metadata as _metadata

LIVE = {"PIL": "pillow", "numpy": "numpy", "torch": "torch",
        "matplotlib": "matplotlib", "transformers": "transformers",
        "huggingface_hub": "huggingface_hub"}
_moved = []
for _module, _dist in LIVE.items():
    if _module not in sys.modules:
        continue
    try:
        _disk = _metadata.version(_dist).split("+")[0]
    except Exception:
        continue
    _live = getattr(sys.modules[_module], "__version__", _disk).split("+")[0]
    if _live != _disk:
        _moved.append(f"{_dist} {_live} in this kernel, {_disk} on disk")

_restarted = Path("/content/.pip_restarted")
if _moved and not _restarted.is_file():
    _restarted.write_text("; ".join(_moved))
    print("pip replaced a package this kernel had already imported:")
    for _line in _moved:
        print("   ", _line)
    print("restarting the runtime -- run this cell again when it comes back")
    try:
        get_ipython().kernel.do_shutdown(True)
    except Exception:
        os.kill(os.getpid(), 9)
    raise SystemExit("restarting the runtime")
if _moved:
    print("!! still mixed after one restart:", "; ".join(_moved))
elif _restarted.is_file():
    _restarted.unlink()

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
# 2. Stage the archives off Drive and build the shared-box index
# --------------------------------------------------------------------------

code('''
Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
STAGED = Path(DATA_ROOT) / ".staged"
STAGED.mkdir(parents=True, exist_ok=True)

def copy_off_drive(source, target):
    from tqdm.auto import tqdm
    _size = source.stat().st_size
    if target.is_file() and target.stat().st_size == _size:
        print("local copy already there:", target)
        return target
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < _size + (2 << 30):
        print("not enough local disk to copy, reading Drive directly")
        return None
    _partial = target.with_name(target.name + ".part")
    _at = 0
    with open(source, "rb") as _in, open(_partial, "wb") as _out, tqdm(
            total=_size, unit="B", unit_scale=True,
            desc="copy " + source.name) as _bar:
        while _at < _size:
            _chunk = b""
            for _attempt in range(5):
                try:
                    _in.seek(_at)
                    _chunk = _in.read(1 << 24)
                    break
                except OSError as _read_error:
                    print("drive read failed at", _at, "--", _read_error)
                    time.sleep(2 ** _attempt)
            if not _chunk:
                raise SystemExit(
                    f"{source}: Drive stopped at {_at} of {_size} bytes -- "
                    f"remount Drive and run this cell again")
            _out.write(_chunk)
            _at += len(_chunk)
            _bar.update(len(_chunk))
    _partial.replace(target)
    return target

def unzip(source, dest):
    _bad = []
    with zipfile.ZipFile(source) as _zip:
        _members = _zip.infolist()
        for _member in progress(_members, len(_members), "unzip " + source.name):
            try:
                _zip.extract(_member, dest)
            except (zipfile.BadZipFile, EOFError):
                _bad.append(_member.filename)
                _dropped = Path(dest) / _member.filename
                if _dropped.is_file():
                    _dropped.unlink()
    return len(_members), _bad

for _archive in ARCHIVES:
    _source = Path(DRIVE_DIR) / _archive
    _marker = STAGED / (_archive + ".done")
    if _marker.is_file():
        print("already staged:", _marker.read_text().strip())
        continue
    if not _source.is_file():
        raise SystemExit(f"{_source} is not there -- set DRIVE_DIR in cell 1")
    print("staging", _source, round(_source.stat().st_size / 2 ** 30, 2), "GiB")
    _local = copy_off_drive(_source, Path(STAGE_DIR) / _archive)
    try:
        _entries, _bad = unzip(_local or _source, DATA_ROOT)
    except zipfile.BadZipFile as _archive_error:
        raise SystemExit(f"{_source}: {_archive_error} -- the copy in Drive is "
                         f"not a readable zip, upload it again")
    if _local is not None and _local.is_file():
        _local.unlink()
    if len(_bad) > max(16, _entries // 100):
        raise SystemExit(
            f"{_source}: {len(_bad)} of {_entries} entries failed their CRC "
            f"even from a local copy -- the archive in Drive is damaged, "
            f"upload it again and re-run this cell")
    if _bad:
        print(len(_bad), "damaged entries dropped:", ", ".join(_bad[:4]))
    _marker.write_text(f"{_archive}: {_entries - len(_bad)} of {_entries} entries")

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

POOLS = (RGB_POOL, TIR_POOL, TIR_ONLY, RGB_ONLY)
SHARD_DIR = Path(MIRROR_DIR) / "shards"
SHARD_DIR.mkdir(parents=True, exist_ok=True)
SHIPPED = set()

def pool_files():
    return [p for p in sorted(Path(POOL_ROOT).rglob("*"))
            if p.is_file() and p.suffix != ".jsonl"]

def restore():
    _shards = sorted(SHARD_DIR.glob("*.zip"))
    for _shard in progress(_shards, len(_shards), "restore"):
        with zipfile.ZipFile(_shard) as _zip:
            _zip.extractall(POOL_ROOT)
    SHIPPED.update(str(p.relative_to(POOL_ROOT)) for p in pool_files())
    print("restored", len(_shards), "shards /", len(SHIPPED), "files from Drive")

def manifest(reports=None):
    (Path(MIRROR_DIR) / "manifest.json").write_text(json.dumps({
        "notebook": NOTEBOOK, "stamp": STAMP, "teacher": TEACHER_USED,
        "written": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "data_root": DATA_ROOT, "pool_root": POOL_ROOT, "datasets": list(POOLS),
        "shards": [p.name for p in sorted(SHARD_DIR.glob("*.zip"))],
        "files": len(SHIPPED), "gates": vars(GATES), "zoom": ZOOM,
        "min_size": MIN_SIZE, "reports": reports}, indent=1))

def checkpoint(note):
    _new = [p for p in pool_files()
            if str(p.relative_to(POOL_ROOT)) not in SHIPPED]
    if not _new:
        print("checkpoint", note, "| nothing new")
        return
    _shard = SHARD_DIR / f"{len(list(SHARD_DIR.glob('*.zip'))):04d}.zip"
    _partial = _shard.with_name(_shard.name + ".part")
    for _attempt in range(4):
        try:
            with zipfile.ZipFile(_partial, "w", zipfile.ZIP_DEFLATED,
                                 allowZip64=True) as _zip:
                for _file in _new:
                    _zip.write(_file, _file.relative_to(POOL_ROOT))
            _partial.replace(_shard)
            SHIPPED.update(str(p.relative_to(POOL_ROOT)) for p in _new)
            manifest()
            print("checkpoint", note, "->", _shard.name, len(_new), "files",
                  round(_shard.stat().st_size / 2 ** 20, 1), "MiB")
            return
        except OSError as _shard_error:
            print("shard write failed, retrying:", _shard_error)
            time.sleep(2 ** _attempt)
    print("!!", note, "not written to Drive -- it stays local and the next "
          "checkpoint ships it")

def merge(reports):
    _out = {}
    for _report in reports:
        for _key, _value in _report.items():
            if _value is None or isinstance(_value, (str, bool)):
                _out[_key] = _value
            elif isinstance(_value, dict):
                _into = _out.setdefault(_key, {})
                for _name, _count in _value.items():
                    _into[_name] = _into.get(_name, 0) + _count
            elif _key != "acceptance_rate":
                _out[_key] = _out.get(_key, 0) + _value
    _out["acceptance_rate"] = (_out.get("accepted", 0) / _out["attempted"]
                               if _out.get("attempted") else 0.0)
    return _out

def harvest(frames, dataset, mirror=None):
    global BATCH, FRAME_GROUP
    _frames = list(frames)[:LIMIT] if LIMIT is not None else list(frames)
    _reports = []
    for _start in range(0, len(_frames), CHUNK):
        _part = _frames[_start:_start + CHUNK]
        while True:
            try:
                _reports.append(label_pool(
                    _part, TEACHER_MODEL, POOL_ROOT, dataset=dataset,
                    prompt="self", mirror=mirror, gates=GATES, zoom=ZOOM,
                    min_size=MIN_SIZE, batch_size=BATCH,
                    frame_group=FRAME_GROUP, max_boxes=MAX_BOXES,
                    progress=progress))
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                BATCH = max(1, BATCH // 2)
                FRAME_GROUP = max(1, FRAME_GROUP // 2)
                print("out of memory, retrying at BATCH", BATCH,
                      "FRAME_GROUP", FRAME_GROUP)
        checkpoint(f"{dataset} {min(_start + CHUNK, len(_frames))}"
                   f"/{len(_frames)}")
    return merge(_reports)

restore()
REPORT = harvest(FRAMES, RGB_POOL, mirror=TIR_POOL)
print(json.dumps(REPORT, indent=1))
INDEX = write_index(POOL_ROOT)
shutil.copy(INDEX, Path(MIRROR_DIR) / INDEX.name)
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

INDEX = write_index(POOL_ROOT)
shutil.copy(INDEX, Path(MIRROR_DIR) / INDEX.name)
print(summarise_pool(pool_report(POOL_ROOT)))
''')


# --------------------------------------------------------------------------
# 6. All four pools back to Drive
# --------------------------------------------------------------------------

code('''
checkpoint("final")
REPORTS = {"shared": REPORT, "thermal_only": ONLY_REPORTS["thermal"],
           "rgb_only": ONLY_REPORTS["rgb"]}
manifest(REPORTS)

_packed = 0
for _pool in POOLS:
    _target = Path(MIRROR_DIR) / f"{_pool}.zip"
    _partial = _target.with_name(_target.name + ".part")
    _files = [p for p in sorted((Path(POOL_ROOT) / _pool).rglob("*"))
              if p.is_file()]
    with zipfile.ZipFile(_partial, "w", zipfile.ZIP_DEFLATED,
                         allowZip64=True) as _zip:
        for _file in progress(_files, len(_files), _pool):
            _zip.write(_file, _file.relative_to(Path(POOL_ROOT)))
    _partial.replace(_target)
    _packed += len(_files)
    print(_target, len(_files), "files",
          round(_target.stat().st_size / 2 ** 20, 1), "MiB")

INDEX = write_index(POOL_ROOT)
shutil.copy(INDEX, Path(MIRROR_DIR) / INDEX.name)
print(summarise_pool(pool_report(POOL_ROOT)))
print(len(pool_files()), "files in the pool /", _packed, "in the four zips /",
      len(list(SHARD_DIR.glob("*.zip"))), "shards |",
      "COMPLETE" if _packed == len(pool_files()) else "MISMATCH")
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
