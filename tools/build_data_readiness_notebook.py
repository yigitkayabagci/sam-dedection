#!/usr/bin/env python3
"""Generate notebook 21: get every pool's frames onto disk, and say what is not.

The first real run of 19/20 lost six pools -- VTUAV, AeroVIS, HIT-UAV, SegFly's
RGB half -- to the same message, `no_image` on every record, and the four fixes
behind that one message were four different things. Finding that out cost a
re-index each time, inside a notebook whose real job was training.

So this is the job on its own. It fetches nothing it does not name, it trains
nothing, and every cell prints enough to be pasted back into a conversation:
what each pool recorded, what is on disk, which of the three causes applies,
and what it would take to fix. The training notebooks stay about training.

    21_pool_data_readiness.ipynb

Cells only, no markdown, no comments -- the same shape 15-20 were asked for.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

BRANCH = "claude/thermal-stage-b-training-43ktcl"
NOTEBOOK = "21_pool_data_readiness.ipynb"

CELLS: list[str] = []


def code(text: str) -> None:
    CELLS.append(text.strip("\n"))


# --------------------------------------------------------------------------
# 1. The knobs, the repo, the mount, the disk
# --------------------------------------------------------------------------

code('''
DRIVE_POOLS = "/content/drive/MyDrive/edgetam-pool"
POOL_ROOT   = "/content/pool"
DATA_ROOT   = "/content/data"
WORK        = "/content/work"
STAGE_DIR   = "/content/drive/MyDrive/datasets"
DRIVE_MY    = "/content/drive/MyDrive"
ARCHIVE_DIRS = ["/content/drive/MyDrive/datasets",
                "/content/drive/MyDrive/VTUAV",
                "/content/drive/MyDrive/DroneVehicle",
                "/content/drive/MyDrive",
                "/content/staging"]

FETCH       = []

VTUAV_PARTS     = []
VTUAV_VIS_PARTS = []

POOL_ZIP_MAX_MB = 2048

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
    ["segfly_rgb",   "segfly_rgb",   "SegFly_RGB",   []],
    ["segfly",       "",             "SegFly",       []],
    ["vtuav_vis",    "vtuav_vis",    "VTUAV_VIS",    VTUAV_VIS_PARTS],
    ["vtuav",        "vtuav_track",  "VTUAV",        VTUAV_PARTS],
    ["kaggle",       "",             "kaggle_uav_thermal", []],
    ["aerovis",      "",             "AeroVIS",      []],
]

REPO_URL = "https://github.com/yigitkayabagci/sam-dedection.git"
BRANCH   = "{{BRANCH}}"
REPO_DIR = "/content/sam-dedection"
NOTEBOOK = "{{NOTEBOOK}}"
STAMP    = "{{STAMP}}"

import json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

if not Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                    REPO_URL, REPO_DIR], check=True)
else:
    for _git in (["fetch", "--depth", "1", "origin", BRANCH],
                 ["reset", "--hard", f"origin/{BRANCH}"]):
        subprocess.run(["git", "-C", REPO_DIR, *_git], check=False)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
for _stale in [n for n in list(sys.modules) if n.split(".")[0] in ("src", "tools")]:
    del sys.modules[_stale]
print("repo:", subprocess.run(["git", "-C", REPO_DIR, "log", "-1",
                               "--format=%h %s"], capture_output=True,
                              text=True).stdout.strip())

try:
    from google.colab import drive as _drive
    _drive.mount("/content/drive")
except Exception as _mount_error:
    print("no Colab Drive mount:", _mount_error)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown",
                "huggingface_hub", "hf_transfer", "kagglehub", "tqdm",
                "pillow"], check=False)
import cv2
if not hasattr(cv2, "imread"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "--force-reinstall", "opencv-python-headless"], check=False)
    for _stale in [n for n in list(sys.modules) if n.split(".")[0] == "cv2"]:
        del sys.modules[_stale]
    import cv2
print("cv2", cv2.__version__, "| imread", hasattr(cv2, "imread"))

for _dir in (POOL_ROOT, DATA_ROOT, WORK):
    Path(_dir).mkdir(parents=True, exist_ok=True)

def _free(path="/content"):
    _s = os.statvfs(path)
    return _s.f_bavail * _s.f_frsize / 2 ** 30

def _used(path):
    _total = 0
    for _f in Path(path).rglob("*"):
        try:
            if _f.is_file():
                _total += _f.stat().st_size
        except OSError:
            pass
    return _total / 2 ** 30

print(f"\\n{NOTEBOOK} {STAMP}")
print(f"free disk {_free():.1f} GiB")
print(f"{DRIVE_POOLS} exists: {Path(DRIVE_POOLS).is_dir()}")
for _name in sorted(Path(DATA_ROOT).iterdir()) if Path(DATA_ROOT).is_dir() else []:
    if _name.is_dir():
        print(f"  {DATA_ROOT}/{_name.name:<28}{_used(_name):>8.2f} GiB")

print("\\narchives already staged where `fetch` looks for them")
STAGED = {}
for _dir in ARCHIVE_DIRS:
    if not Path(_dir).is_dir():
        print(f"  {_dir}  (not there)")
        continue
    _found = [f for f in sorted(Path(_dir).iterdir())
              if f.is_file() and f.name.endswith((".zip", ".tar.gz", ".tgz"))]
    print(f"  {_dir}  {len(_found)} archives")
    for _archive in _found:
        _gib = _archive.stat().st_size / 2 ** 30
        STAGED.setdefault(_archive.stem.replace(".tar", ""), str(_archive))
        print(f"      {_archive.name:<40}{_gib:>8.2f} GiB")
print("\\nan archive here is never re-downloaded -- `staged()` finds it by "
      "name, so a part called train_ST_008 needs a train_ST_008.zip in one "
      "of these folders.")
''')


# --------------------------------------------------------------------------
# 2. Every pool Drive holds, and what each one recorded
# --------------------------------------------------------------------------

code('''
from src.training.pool_reader import (acceptance, discover_pools,
                                      group_records, link_pool)
from src.training.pool import RECORD_FILE

_done = Path(POOL_ROOT) / ".unpacked"
_done.mkdir(parents=True, exist_ok=True)
assert Path(DRIVE_POOLS).is_dir(), f"{DRIVE_POOLS} is not there -- set DRIVE_POOLS"

print("what is in", DRIVE_POOLS)
for _entry in sorted(Path(DRIVE_POOLS).iterdir()):
    _kind = "dir " if _entry.is_dir() else "file"
    _size = "" if _entry.is_dir() else f"{_entry.stat().st_size / 2 ** 20:>10.1f} MiB"
    print(f"  {_kind} {_entry.name:<32}{_size}")

print()
for _zip in sorted(Path(DRIVE_POOLS).rglob("*.zip")):
    _size = round(_zip.stat().st_size / 2 ** 20, 1)
    _marker = _done / (str(_zip.relative_to(DRIVE_POOLS)).replace("/", "__")
                       + ".done")
    if _marker.is_file():
        print("already unpacked", _zip.name)
        continue
    if _size > POOL_ZIP_MAX_MB:
        print("skipped", _zip.name, _size, "MiB -- a pool holds masks only, "
              "so this is source data, not a pool")
        continue
    try:
        with zipfile.ZipFile(_zip) as _handle:
            _members = _handle.namelist()
            if not any(_m.endswith(RECORD_FILE) for _m in _members):
                print("skipped", _zip.name, "-- no", RECORD_FILE, "in it")
                continue
            _handle.extractall(POOL_ROOT)
    except Exception as _unzip_error:
        print("!! could not read", _zip.name, "--", _unzip_error)
        continue
    _marker.touch()
    print("unpacked", _zip.name, _size, "MiB,", len(_members), "files")

for _name, _folder in discover_pools(DRIVE_POOLS).items():
    _marker = _done / (_name + ".folder.done")
    if _marker.is_file() or (Path(POOL_ROOT) / _name).is_dir():
        continue
    shutil.copytree(_folder, Path(POOL_ROOT) / _name, dirs_exist_ok=True)
    _marker.touch()
    print("copied", _name, "from Drive (it was staged unzipped)")

RAW = group_records(POOL_ROOT)
POOLS, TEACHERS, HARVEST = {}, {}, {}
print(f"\\n{'pool':<28}{'frames':>8}{'boxes':>9}{'masks':>9}{'kept':>7}  teacher")
for _name, _records in sorted(RAW.items()):
    POOLS[_name] = str(link_pool(_records, Path(POOL_ROOT) / "_by_name" / _name))
    HARVEST[_name] = acceptance(_records)
    TEACHERS[_name] = "+".join(HARVEST[_name]["teachers"]) or "?"
    print(f"{_name:<28}{HARVEST[_name]['frames']:>8}"
          f"{HARVEST[_name]['attempted']:>9}{HARVEST[_name]['accepted']:>9}"
          f"{HARVEST[_name]['rate']:>7.1%}  {TEACHERS[_name]}")
assert POOLS, f"no {RECORD_FILE} under {POOL_ROOT} -- nothing was unpacked"

print("\\nclasses each pool actually carries")
for _name in sorted(POOLS):
    _classes = HARVEST[_name]["accepted_by_class"]
    _top = ", ".join(f"{k} {v}" for k, v in
                     sorted(_classes.items(), key=lambda kv: -kv[1])[:6])
    print(f"  {_name:<28}{len(_classes):>3} classes  {_top}")

print("\\nwhich sequences each pool was harvested from")
from src.training.aerial import group_of

GROUPS = {}
for _name in sorted(POOLS):
    _seen = {}
    for _record_path in sorted(Path(POOLS[_name]).rglob(RECORD_FILE)):
        _key = json.loads(_record_path.read_text())["key"]
        _group = _key.rsplit("/", 1)[0] if "/" in _key else "(no sequence)"
        _seen[_group] = _seen.get(_group, 0) + 1
    GROUPS[_name] = _seen
    _shown = ", ".join(f"{k} {v}" for k, v in
                       sorted(_seen.items(), key=lambda kv: -kv[1])[:8])
    print(f"  {_name:<28}{len(_seen):>4} sequences  {_shown}"
          f"{' ...' if len(_seen) > 8 else ''}")
print("   a pool's sequences say which archive parts it needs: VTUAV files "
      "one sequence per directory and each tracking part holds a different "
      "set of them, so this is the list to match VTUAV_PARTS against.")

print("\\nthe path each pool recorded for its first frame")
RECORDED = {}
for _name in sorted(POOLS):
    _first = sorted(Path(POOLS[_name]).rglob(RECORD_FILE))[0]
    _record = json.loads(_first.read_text())
    RECORDED[_name] = {"image": _record.get("image"),
                       "image_rel": _record.get("image_rel"),
                       "shape": _record.get("shape")}
    print(f"  {_name:<28}{_record.get('image')}")
    if _record.get("image_rel"):
        print(f"  {'':<28}image_rel = {_record['image_rel']}")
''')


# --------------------------------------------------------------------------
# 3. Which of the three causes, per pool
# --------------------------------------------------------------------------

code('''
from src.training.pool_reader import Relocator, why_no_image

def images_for(pool):
    lowered = pool.lower()
    for key, recipe, folder, parts in IMAGES:
        if key in lowered:
            root = IMAGE_ROOTS.get(pool) or str(Path(DATA_ROOT) / folder)
            return key, recipe, root, list(parts)
    return None, "", "", []

READY, BROKEN, UNMAPPED = [], [], []
print(f"{'pool':<28}{'resolved':>10}{'of':>8}  root")
for _name in sorted(POOLS):
    _key, _recipe, _root, _parts = images_for(_name)
    if _key is None:
        UNMAPPED.append(_name)
        print(f"{_name:<28}{'--':>10}{'--':>8}  no entry in IMAGES")
        continue
    _records = sorted(Path(POOLS[_name]).rglob(RECORD_FILE))
    _probe = _records[::max(len(_records) // 50, 1)][:50]
    _relocate = Relocator(_root)
    _hits = 0
    for _record_path in _probe:
        _record = json.loads(_record_path.read_text())
        _found = (_relocate.direct(_record.get("image_rel"))
                  or _relocate(_record["image"]))
        _hits += _found is not None
    print(f"{_name:<28}{_hits:>10}{len(_probe):>8}  {_root}")
    (READY if _hits == len(_probe) else BROKEN).append(
        {"pool": _name, "key": _key, "recipe": _recipe, "root": _root,
         "parts": _parts, "hits": _hits, "probed": len(_probe)})

print("\\n" + "=" * 74)
print("POOLS WHOSE FRAMES ARE NOT ON DISK")
print("=" * 74)
for _row in BROKEN:
    _report = why_no_image(POOLS[_row["pool"]], _row["root"])
    print(f"\\n{_row['pool']}  ({_row['hits']}/{_row['probed']} probed frames found)")
    print(f"  recorded : {_report['recorded'][0]['image']}")
    print(f"  image_rel: {_report['recorded'][0]['image_rel']}")
    print(f"  root     : {_report['root']}  exists={_report['root_exists']}")
    if _report["root_exists"]:
        print(f"  under it : {_report['files_under_root']} files "
              f"{_report['extensions']}")
    if _report.get("same_name_here"):
        print(f"  same name: {_report['same_name_here']}")
    print(f"  VERDICT  : {_report['verdict']}")
    print(f"  fetch    : recipe={_row['recipe'] or '(none)'} "
          f"parts={_row['parts'] or '(none named)'}")

if UNMAPPED:
    print("\\nno IMAGES row at all:", UNMAPPED)
print(f"\\nready now: {[r['pool'] for r in READY]}")
print(f"not ready: {[r['pool'] for r in BROKEN]}")
''')


# --------------------------------------------------------------------------
# 4. Fetch exactly what FETCH names, and nothing else
# --------------------------------------------------------------------------

code('''
from tools.fetch_datasets import RECIPES, fetch, human, staged

print("what each recipe would cost")
for _key, _recipe, _folder, _parts in IMAGES:
    if not _recipe or _recipe not in RECIPES:
        print(f"  {_key:<14}no recipe -- stage it yourself into "
              f"{Path(DATA_ROOT) / _folder}")
        continue
    _r = RECIPES[_recipe]
    _default = [p for p in _r.parts if getattr(p, "default", True)]
    _all = sum(getattr(p, "size", 0) or 0 for p in _r.parts)
    print(f"  {_key:<14}{_recipe:<14}{len(_r.parts):>3} parts, "
          f"{len(_default)} on by default, {human(_all)} for all of it")
    for _part in _r.parts:
        _mark = " " if getattr(_part, "default", True) else "*"
        print(f"      {_mark}{_part.name:<18}"
              f"{human(getattr(_part, 'size', 0) or 0):>12}")
print("  * = off unless named. FETCH is a list of recipe names; parts come "
      "from VTUAV_PARTS / VTUAV_VIS_PARTS or the IMAGES row.")

def unpack(archive, target, label=""):
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

for _pool, _dataset in KAGGLE_DATASETS.items():
    _root = Path(IMAGE_ROOTS.get(_pool, Path(DATA_ROOT) / _pool))
    if _root.is_dir() and any(_root.rglob("*.jpg")) or any(_root.rglob("*.png")):
        print("already on disk:", _root)
        continue
    try:
        import kagglehub
        _got = Path(kagglehub.dataset_download(_dataset))
        _root.parent.mkdir(parents=True, exist_ok=True)
        if not _root.exists():
            _root.symlink_to(_got)
        print("kaggle", _dataset, "->", _got)
    except Exception as _kaggle_error:
        print("!! kagglehub", _dataset, "failed:", _kaggle_error)

print()
for _name in FETCH:
    _row = next((r for r in IMAGES if r[1] == _name), None)
    if _row is None:
        print(f"!! {_name} names no IMAGES row -- nothing to fetch it into")
        continue
    _key, _recipe, _folder, _parts = _row
    _root = Path(DATA_ROOT) / _folder
    if _recipe not in RECIPES:
        print(f"!! {_recipe} is not a recipe. Known: {sorted(RECIPES)}")
        continue
    print(f"\\n--- {_recipe} -> {_root}  (free {_free():.1f} GiB)")
    try:
        fetch(_recipe, _root, tuple(_parts) or None, stage=STAGE_DIR,
              staging=tuple(ARCHIVE_DIRS + [str(Path(DRIVE_MY) / _root.name)]))
    except Exception as _fetch_error:
        print("!!", _recipe, "failed:", type(_fetch_error).__name__, _fetch_error)
        for _part in sorted(_parts) or [_recipe]:
            _copy = staged(_part, tuple(ARCHIVE_DIRS
                                        + [str(Path(DRIVE_MY) / _root.name)]))
            if _copy is None:
                continue
            print("   retrying", _copy.name, "member by member")
            unpack(_copy, _root, _part)
    print(f"    {_root}: {_used(_root):.2f} GiB, free now {_free():.1f} GiB")
if not FETCH:
    print("FETCH is empty -- nothing downloaded. Put recipe names in it, e.g. "
          "FETCH = ['hituav', 'segfly_rgb', 'vtuav_vis']")
''')


# --------------------------------------------------------------------------
# 5. Re-check, and the exact flags a training run would take
# --------------------------------------------------------------------------

code('''
from src.training.aerial import InstanceGates
from src.training.pool_reader import SKIP_REASONS, index_pool

GATES = InstanceGates(min_area=48, min_side=4, max_area=0.9, fill=0.25)

print("=" * 74)
print("READINESS")
print("=" * 74)
print(f"{'pool':<28}{'frames':>9}{'instances':>11}{'modality':<10}  state")
FLAGS, STILL_BROKEN = [], []
for _name in sorted(POOLS):
    _key, _recipe, _root, _parts = images_for(_name)
    if _key is None:
        STILL_BROKEN.append((_name, "no IMAGES row"))
        continue
    _lowered = _name.lower()
    _modality = "rgb" if ("rgb" in _lowered.split("_")
                          or _lowered.endswith("_rgb")) else "thermal"
    try:
        _part = index_pool(POOLS[_name], _root, _modality, "train", GATES,
                           _name, workers=8)
    except Exception as _error:
        STILL_BROKEN.append((_name, f"{type(_error).__name__}: "
                                    f"{str(_error).splitlines()[0]}"))
        print(f"{_name:<28}{'--':>9}{'--':>11}{_modality:<10}  UNUSABLE")
        continue
    _skips = {k: v for k, v in _part[0].rejects.items() if k in SKIP_REASONS}
    print(f"{_name:<28}{len(_part):>9}"
          f"{sum(len(e.instances) for e in _part):>11}{_modality:<10}  ready"
          f"{('  skipped ' + str(_skips)) if _skips else ''}")
    FLAGS.append(f"{POOLS[_name]}:{_root}:{_modality}:train")

print(f"\\n{len(FLAGS)} pools ready, {len(STILL_BROKEN)} not")
for _name, _why in STILL_BROKEN:
    print(f"  {_name:<28}{_why}")

print("\\n--pool flags a run would take:")
for _flag in FLAGS:
    print(f"  {_flag}")

(Path(WORK) / "readiness.json").write_text(json.dumps(
    {"ready": FLAGS, "broken": STILL_BROKEN, "recorded": RECORDED,
     "harvest": {k: {kk: vv for kk, vv in v.items() if kk != "accepted_by_class"}
                 for k, v in HARVEST.items()},
     "free_gib": round(_free(), 1)}, indent=2) + "\\n")
print("\\nwrote", Path(WORK) / "readiness.json")
''')


# --------------------------------------------------------------------------


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
