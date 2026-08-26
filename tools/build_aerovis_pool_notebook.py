#!/usr/bin/env python3
"""Generate notebook 20: AeroVIS into a stage-B pool, with no teacher at all.

Every other pool notebook in this repo spends GPU-hours turning boxes into
masks. This one spends none. AeroVIS ships **1 378 603 instances that carry a
box and a mask on the same frame** -- SAM 3 output that was then re-inspected
by hand -- so the job is translation, not labelling: YTVIS annotations into the
same `record.json` + run-length store `image_loop.py` already reads. A CPU
runtime finishes it.

The schema quirks it handles are the ones `tools/inspect_aerovis.py` measured
against the real release rather than assumed, and each of them breaks a naive
reader: the 80 `iscrowd=1` ignore regions whose `bboxes` are null throughout,
`counts` arriving as `str` where pycocotools wants bytes, boxes in XYWH, and
149 boxes with no mask. `src/training/aerovis.py` argues each one where it
lives.

**The decision this notebook makes explicit.** `docs/rgb_aerial_kaynaklar.md`
§1 says leave AeroVIS alone and keep it as the held-out benchmark; §3.5 states
the trade rather than deciding it. Training on it saves running a teacher over
a million instances and costs the benchmark outright. `HOLDOUT` is where that
lands: it defaults to 0.15, which keeps a measurement set at the price of a
sixth of the data, and `HOLDOUT = 0.0` is the other choice made deliberately
rather than by omission.

The split is over **whole sequences** and stratified by source prefix. Frames
of one sequence are near-duplicates -- the median track runs 117 frames -- so a
frame-level split leaks almost completely; and **vehicle exists only in UAVDT,
boat only in SeaDronesSee**, so an unstratified split can take a whole class
out of one side without the loss ever mentioning it.

Six cells:

  1. environment, and the settings the two decisions live in
  2. fetch the 12.6 GiB archive, extract the annotation file alone
  3. census and split -- printed before any frame is written to disk
  4. extract only the sequences that survived the split
  5. translate to the pool store, and mirror it to Drive
  6. spot-check: masks and boxes drawn on real frames

Only the store goes to Drive. It is a few hundred megabytes against 12.6 GiB of
JPEGs, and every record carries `image_rel`, so a training run re-points at its
own copy of the frames instead of a second copy travelling with the masks.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

NOTEBOOK = "20_aerovis_stage_b_pool.ipynb"
STAMP_VALUE = "unstamped"

CELLS: list[str] = []


def code(text: str) -> None:
    CELLS.append(text.strip("\n"))


code('''
DRIVE_ID     = "1DMLagGZMPntrvxk5W0PsaIoybsE7WX56"
DOWNLOAD_DIR = "/content/dl"
EXTRACT_DIR  = "/content/data"
POOL_ROOT    = "/content/pool"
MIRROR_DIR   = "/content/drive/MyDrive/edgetam-pool/aerotrack"
TRAIN_POOL   = "aerovis_train"
HELD_POOL    = "aerovis_heldout"
HOLDOUT      = 0.15          # 0.0 = train on all of it and lose the benchmark
SEED         = 0
LIMIT        = None
EXPORT_PNG   = False         # 49 048 extra PNGs; the store is what training reads
KEEP_ARCHIVE = False
REQUIRE_DRIVE = True
PATCH_FILE   = "/content/drive/MyDrive/edgetam-pool/repo_patch_aerovis.py"
REPO_URL     = "https://github.com/yigitkayabagci/sam-dedection.git"
BRANCH       = "claude/aerial-rgb-thermal-data-qhjpxd"
REPO_DIR     = "/content/sam-dedection"
NOTEBOOK = "20_aerovis_stage_b_pool.ipynb"
STAMP    = "{{STAMP}}"

import json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

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

# No transformers, no accelerate, no GPU: there is no teacher in this notebook.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                "gdown", "pycocotools", "tqdm"], check=False)

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
        "  - re-run this cell and accept the authorisation popup\\n"
        "  - or set REQUIRE_DRIVE = False and copy the pool out by hand")

REQUIRED = {
    "src.training.aerovis": ("write_pool", "sequence_split", "decode_rle"),
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

def progress(stream, total, desc):
    from tqdm.auto import tqdm
    return tqdm(stream, total=total, desc=desc)

_stamps = Path(REPO_DIR) / "notebooks" / ".stamps.json"
_want = json.loads(_stamps.read_text()).get(NOTEBOOK) if _stamps.is_file() else None
print(NOTEBOOK, STAMP, "| repo:", _want,
      "| OK" if _want == STAMP else "| STALE, re-open from the repo")
print("ready | HOLDOUT", HOLDOUT, "| mirror", MIRROR_DIR)
''')


code('''
# 12.63 GiB behind a Drive interstitial, so gdown rather than curl. Only the
# annotation file comes out here -- 332 MB against 12.6 GiB -- because the
# split in cell 3 decides which sequences are worth the disk.
Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
ARCHIVE = next(iter(sorted(Path(DOWNLOAD_DIR).glob("*.zip"))), None)
if ARCHIVE is None:
    import gdown
    gdown.download(id=DRIVE_ID, output=f"{DOWNLOAD_DIR}/AeroVIS.zip",
                   quiet=False)
    ARCHIVE = next(iter(sorted(Path(DOWNLOAD_DIR).glob("*.zip"))))
print(ARCHIVE, round(ARCHIVE.stat().st_size / 2 ** 30, 2), "GiB")

ZIP = zipfile.ZipFile(ARCHIVE)
MEMBERS = ZIP.namelist()
_json_member = next((n for n in MEMBERS if n.endswith("aero_vis.json")), None)
if _json_member is None:
    raise SystemExit(f"no aero_vis.json in {ARCHIVE}; first names: {MEMBERS[:5]}")
ROOT_REL = Path(_json_member).parent
SEQUENCES = Path(EXTRACT_DIR) / ROOT_REL / "sequences"
ANNOTATIONS = Path(EXTRACT_DIR) / _json_member
if not ANNOTATIONS.is_file():
    print("extracting", _json_member,
          round(ZIP.getinfo(_json_member).file_size / 2 ** 20, 1), "MiB")
    ZIP.extract(_json_member, EXTRACT_DIR)

from src.training import aerovis as A

DATA = A.load_index(ANNOTATIONS)
print(len(DATA["videos"]), "sequences |", len(DATA["annotations"]), "tracks |",
      len(DATA["categories"]), "classes")
print([c["name"] for c in DATA["categories"]])
''')


code('''
# The census and the split, both before a single frame is written. The split
# is over whole sequences (median track = 117 frames, so a frame-level split
# leaks) and stratified by source prefix (vehicle is UAVDT-only, boat is
# SeaDronesSee-only -- an unstratified split can empty a class from one side).
print(A.summarise(DATA))

TRAIN, HELD = A.sequence_split(DATA["videos"], holdout=HOLDOUT, seed=SEED)
print(f"\\nsplit: {len(TRAIN)} train / {len(HELD)} held out")
for _name, _group in (("train", TRAIN), ("heldout", HELD)):
    _by = {}
    for _video in _group:
        _s = A.source_of(A._video_name(_video))
        _by[_s] = _by.get(_s, 0) + 1
    print(" ", _name, dict(sorted(_by.items())))
if HOLDOUT == 0.0:
    print("\\n!! HOLDOUT = 0: AeroVIS becomes training data and stops being a "
          "held-out benchmark. That is a decision, not a default.")
''')


code('''
# Only the sequences that survived the split, and only their frames. The
# archive is one flat tree, so membership is a prefix test on the name.
WANTED = {A._video_name(v) for v in TRAIN} | {A._video_name(v) for v in HELD}
_prefixes = tuple(f"{ROOT_REL.as_posix()}/sequences/{n}/" for n in sorted(WANTED))
_members = [n for n in MEMBERS if n.startswith(_prefixes)
            and not n.endswith("/")]
print(len(_members), "frames to extract,",
      round(sum(ZIP.getinfo(n).file_size for n in _members) / 2 ** 30, 2), "GiB")

for _member in progress(_members, total=len(_members), desc="frames"):
    if not (Path(EXTRACT_DIR) / _member).is_file():
        ZIP.extract(_member, EXTRACT_DIR)
ZIP.close()
if not KEEP_ARCHIVE:
    ARCHIVE.unlink()
    print("archive removed")
print("sequences at", SEQUENCES, "|", len(list(SEQUENCES.glob("*"))), "folders")
''')


code('''
# The translation. No teacher, no gates: gates exist to decide whether a
# *teacher's* mask can be trusted, and these are the dataset's own.
REPORTS = {}
for _pool, _videos in ((TRAIN_POOL, TRAIN), (HELD_POOL, HELD)):
    if not _videos:
        continue
    REPORTS[_pool] = A.write_pool(DATA, SEQUENCES, POOL_ROOT, dataset=_pool,
                                  videos=_videos, limit=LIMIT,
                                  progress=progress)
    print(_pool, json.dumps(REPORTS[_pool], indent=1))

from src.training.labels import MASK_STORE, open_masks

if EXPORT_PNG:
    import numpy as np, cv2
    for _store in sorted(Path(POOL_ROOT).rglob(MASK_STORE)):
        _masks = open_masks(_store)
        _canvas = np.zeros(_masks.shape, np.uint16)
        for _i in sorted(_masks):
            _canvas[_masks[_i]] = _i + 1
        cv2.imwrite(str(_store.parent / "instances.png"), _canvas)

Path(MIRROR_DIR).mkdir(parents=True, exist_ok=True)
with (Path(POOL_ROOT) / "aerovis_prompts.jsonl").open("w") as _handle:
    for _record in sorted(Path(POOL_ROOT).rglob("record.json")):
        _row = json.loads(_record.read_text())
        for _instance in _row["instances"]:
            _handle.write(json.dumps({
                "pool": _row["dataset"], "frame": _row["key"],
                "image_rel": _row["image_rel"], "source": _row["source"],
                "box": _instance["box"], "class": _instance["class"],
                "track_id": _instance["track_id"], "i": _instance["i"]}) + "\\n")

(Path(POOL_ROOT) / "aerovis_selection.json").write_text(json.dumps({
    "drive_id": DRIVE_ID, "holdout": HOLDOUT, "seed": SEED,
    "train_sequences": sorted(A._video_name(v) for v in TRAIN),
    "heldout_sequences": sorted(A._video_name(v) for v in HELD),
    "reports": REPORTS}, indent=1) + "\\n")

for _pool in REPORTS:
    _folder = Path(POOL_ROOT) / _pool
    _target = Path(MIRROR_DIR) / f"{_pool}.zip"
    _files = [p for p in sorted(_folder.rglob("*")) if p.is_file()]
    with zipfile.ZipFile(_target, "w", zipfile.ZIP_STORED, allowZip64=True) as _z:
        for _f in _files:
            _z.write(_f, _f.relative_to(Path(POOL_ROOT)))
    print(_target, len(_files), "files",
          round(_target.stat().st_size / 2 ** 20, 1), "MiB")
for _name in ("aerovis_prompts.jsonl", "aerovis_selection.json"):
    shutil.copy(Path(POOL_ROOT) / _name, Path(MIRROR_DIR) / _name)
print("\\ntotal instances:", sum(r["instances"] for r in REPORTS.values()))
''')


code('''
SHOW = 3
SEED_SHOW = 0

import random
import numpy as np, cv2
import matplotlib.pyplot as plt

def pool_keys(pool):
    root = Path(POOL_ROOT) / pool
    return sorted(str(p.parent.relative_to(root)) for p in root.rglob(MASK_STORE))

random.seed(SEED_SHOW)
_keys = pool_keys(TRAIN_POOL)
print("Boxes are the source datasets' own, not the mask envelope -- only 3.8 %")
print("match on all four sides. So a box sitting a few px off its mask is the")
print("dataset being honest, not the translation being wrong.")
for _key in random.sample(_keys, min(SHOW, len(_keys))):
    _folder = Path(POOL_ROOT) / TRAIN_POOL / _key
    _rec = json.loads((_folder / "record.json").read_text())
    _store = open_masks(_folder / MASK_STORE)
    _raw = cv2.imread(_rec["image"], cv2.IMREAD_UNCHANGED)
    if _raw is None:
        print("unreadable:", _rec["image"])
        continue
    _img = cv2.cvtColor(_raw[:, :, :3], cv2.COLOR_BGR2RGB)
    _shown = _img.copy()
    for _i in sorted(_store):
        _m = _store[_i]
        _shown[_m] = (0.5 * _shown[_m] + 0.5 * np.array([255, 60, 60])).astype(np.uint8)
    _fig, _ax = plt.subplots(figsize=(13, 8))
    _ax.imshow(_shown)
    for _ins in _rec["instances"]:
        _x0, _y0, _x1, _y1 = _ins["box"]
        _ax.add_patch(plt.Rectangle((_x0, _y0), _x1 - _x0, _y1 - _y0,
                                    fill=False, linewidth=1.0, edgecolor="lime"))
    _ax.set_title(f"{_key} | {_rec['source']} | {len(_store)} instances", fontsize=10)
    _ax.axis("off")
    plt.tight_layout(); plt.show()
    print(_key, _rec["source"],
          {c: sum(1 for i in _rec["instances"] if i["class"] == c)
           for c in sorted({i["class"] for i in _rec["instances"]})})
''')


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
            "colab": {"provenance": [], "machine_shape": "hm"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    out = repo / "notebooks" / NOTEBOOK
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    stamps = repo / "notebooks" / ".stamps.json"
    known = json.loads(stamps.read_text()) if stamps.is_file() else {}
    known[NOTEBOOK] = STAMP_VALUE
    stamps.write_text(json.dumps(known, indent=1, sort_keys=True) + "\n")
    print(f"wrote notebooks/{NOTEBOOK} with {len(CELLS)} cells [{STAMP_VALUE}]",
          file=sys.stderr)
