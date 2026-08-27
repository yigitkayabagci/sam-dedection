#!/usr/bin/env python3
"""Build notebook 26: reproducible SegFly semantic-to-instance audit."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

BRANCH = "claude/thermal-stage-b-training-43ktcl"
NOTEBOOK = "26_segfly_instance_audit.ipynb"


def markdown(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": (text.strip() + "\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": (text.strip() + "\n").splitlines(keepends=True),
    }


CELLS = [
    markdown("""
# SegFly semantic map → instance target audit

Bu notebook training yapmaz. Gerçek SegFly thermal sample'larını indirir,
`semantic map → thing filter → connected components / watershed → gates →
instance mask + box` dönüşümünü repo koduyla çalıştırır ve rapor çıktısını
üretir. Son cell'deki SAM check opsiyoneldir: disagreement bulur, ground truth
üretmez.
"""),
    code('''
DATA_ROOT = "/content/data/SegFly_audit"
OUT_ROOT  = "/content/segfly-audit"
ROWS      = 64
RUN_SAM   = False
SAM_MODEL = "facebook/sam2.1-hiera-large"
SAM_FRAMES = 32

REPO_URL = "https://github.com/yigitkayabagci/sam-dedection.git"
BRANCH   = "{{BRANCH}}"
REPO_DIR = "/content/sam-dedection"
NOTEBOOK = "{{NOTEBOOK}}"
STAMP    = "{{STAMP}}"

import json, os, shutil, subprocess, sys
from pathlib import Path

if not Path(REPO_DIR).exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                    REPO_URL, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "--depth", "1",
                    "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "reset", "--hard",
                    f"origin/{BRANCH}"], check=True)
os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "pillow", "matplotlib", "pyarrow", "huggingface_hub",
                "hf_transfer", "opencv-python-headless"], check=True)
Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)
print(NOTEBOOK, STAMP)
print(subprocess.run(["git", "-C", REPO_DIR, "log", "-1", "--format=%h %s"],
                     capture_output=True, text=True, check=True).stdout.strip())
'''),
    markdown("""
## 1. Export

SegFly Hub'da tek logical split içinde 761 parquet shard olarak dağıtılır.
Exporter önce yalnız `modality` kolonunu tarar, sonra matching shard'ları
indirir. `--spread`, küçük slice'ın tek uçuşa sıkışmasını önler. Label her
zaman lossless PNG kalır; class IDs image preview'den okunmaz.
"""),
    code('''
from tools.export_hf_dataset import export, verify

_existing = list((Path(DATA_ROOT) / "labels").glob("*.png"))
if len(_existing) < ROWS:
    _result = export("markus-42/SegFly", Path(DATA_ROOT), "thermal", "train",
                     ROWS, streaming=False, passthrough=False, spread=True)
    print(_result["written"], "rows written")
    print(verify(_result["values"], "segfly"))
else:
    print(len(_existing), "exported rows reused")
'''),
    markdown("""
## 2. Schema, palette ve raw data

`label` bir `semantic map`tir; pixel value doğrudan class ID'dir. Bu projede
yalnız `vehicle=13` ve `truck=36` thing class kabul edilir. Diğer class'lar
training target değildir.
"""),
    code('''
from src.training.aerial import (SPECS, Source, InstanceGates, list_frames,
                                 probe_classes, describe_layout)

_spec = SPECS["segfly"]
_frames = list_frames(DATA_ROOT, _spec, "thermal")
print(describe_layout(DATA_ROOT))
print()
print("palette source:", _spec.palette_source)
print("thing IDs:", dict(zip(_spec.things, _spec.thing_ids)))
print("values in sample:", probe_classes(_frames, limit=ROWS))
print("frames:", len(_frames), "| paired RGB:", sum(f.pair is not None for f in _frames))
'''),
    markdown("""
## 3. Exact conversion ve report figure

Tool diagnostic bir frame seçer. `components` 8-connectivity kullanır;
`watershed` thin bridge için ek split dener. İki sonuç birlikte çizilir.
"""),
    code('''
_figure = Path(OUT_ROOT) / "segfly-instance-conversion.png"
_record = Path(OUT_ROOT) / "segfly-instance-conversion.json"
subprocess.run([sys.executable, "tools/analyze_segfly_instances.py",
                "--root", DATA_ROOT, "--scan", str(ROWS),
                "--figure", str(_figure), "--json", str(_record)], check=True)

from IPython.display import Image as DisplayImage, display
display(DisplayImage(filename=str(_figure)))
print(_record.read_text())
'''),
    markdown("""
## 4. Sensitivity table

Bu tablo sample slice üzerinde `mode` ve `fill gate` değişince kaç target
kaldığını gösterir. Karar full dataset üstünde tekrar alınmalıdır. Yüksek
`kept` tek başına kalite değildir; `fill reject`, possible fusion alarmıdır.
"""),
    code('''
from collections import Counter
from src.training.aerial import decompose, read_mask

_rows = []
for _fill in (0.20, 0.25, 0.35, 0.45):
    _gates = InstanceGates(fill=_fill)
    for _mode in ("components", "watershed"):
        _kept, _rejects, _classes = 0, Counter(), Counter()
        for _frame in _frames:
            _, _instances, _rejected = decompose(read_mask(_frame.mask), _spec,
                                                  _gates, _mode)
            _kept += len(_instances)
            _rejects.update(_rejected)
            _classes.update(_spec.name_of(i.class_id) for i in _instances)
        _rows.append({"mode": _mode, "fill": _fill, "kept": _kept,
                      "vehicle": _classes["vehicle"], "truck": _classes["truck"],
                      "min_area": _rejects["min_area"],
                      "min_side": _rejects["min_side"],
                      "max_area": _rejects["max_area"],
                      "fill_reject": _rejects["fill"]})

import pandas as pd
display(pd.DataFrame(_rows))
'''),
    markdown("""
## 5. Optional SAM disagreement audit

`RUN_SAM=True` olduğunda aynı reconstructed instance box, önce thermal image'a,
sonra `RGB_aligned` image'a prompt edilir. Çıktı, reconstructed mask ile SAM
mask arasındaki IoU'dur. Düşük agreement review priority'dir; SAM ve SegFly iki
farklı pseudo-label olduğu için bu sayı accuracy ya da ground truth değildir.
"""),
    code('''
if RUN_SAM:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers>=4.56,<5"], check=True)
    _sam_json = Path(OUT_ROOT) / "segfly-sam-audit.json"
    _sam_figure = Path(OUT_ROOT) / "segfly-sam-audit.png"
    subprocess.run([sys.executable, "tools/analyze_segfly_instances.py",
                    "--root", DATA_ROOT, "--scan", str(ROWS),
                    "--figure", str(_sam_figure), "--json", str(_sam_json),
                    "--sam-teacher", SAM_MODEL, "--sam-route", "both",
                    "--sam-frames", str(SAM_FRAMES), "--device", "cuda"],
                   check=True)
    _sam = json.loads(_sam_json.read_text())["sam_disagreement_audit"]
    display(pd.DataFrame([
        {k: v for k, v in _summary.items() if k != "records"}
        | {"route": _route}
        for _route, _summary in _sam.items()
    ]))
else:
    print("SAM audit skipped. Set RUN_SAM=True and run this cell on a GPU runtime.")
'''),
    markdown("""
## 6. Deliverables

PNG ve JSON browser'dan indirilebilir. Drive mounted ise aynı dosyalar kalıcı
klasöre de kopyalanır.
"""),
    code('''
try:
    from google.colab import files
    for _path in sorted(Path(OUT_ROOT).glob("*")):
        print(_path, f"{_path.stat().st_size / 2**20:.2f} MiB")
    files.download(str(_figure))
    files.download(str(_record))
except Exception as _download_error:
    print("Colab download unavailable:", _download_error)
'''),
]


def render() -> tuple[list[dict], str]:
    serialized = json.dumps(CELLS, ensure_ascii=False, sort_keys=True)
    stamp = hashlib.sha256(serialized.encode()).hexdigest()[:10]
    cells = []
    for cell in CELLS:
        copy = json.loads(json.dumps(cell, ensure_ascii=False))
        copy["source"] = [
            line.replace("{{BRANCH}}", BRANCH)
                .replace("{{NOTEBOOK}}", NOTEBOOK)
                .replace("{{STAMP}}", stamp)
            for line in copy["source"]
        ]
        cells.append(copy)
    return cells, stamp


def build() -> dict:
    cells, _ = render()
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100"},
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
        json.dumps(document, indent=1, ensure_ascii=False) + "\n"
    )
    known[NOTEBOOK] = render()[1]
    stamps_file.write_text(json.dumps(known, indent=1, sort_keys=True) + "\n")
    print(
        f"wrote notebooks/{NOTEBOOK} with {len(document['cells'])} cells "
        f"[{known[NOTEBOOK]}]",
        file=sys.stderr,
    )
