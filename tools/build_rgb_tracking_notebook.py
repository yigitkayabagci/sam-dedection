#!/usr/bin/env python3
"""Build notebook 37: the RGB Stage C, on VTUAV-VIS alone with a real held-out set.

31 stays what it is -- the thermal Stage C, six VTUAV tracking archives plus
BIRDSAI. This is its RGB twin and it is deliberately narrower:

* **One source.** VTUAV-VIS is the only set here whose masks are the authors'
  own per-frame instance masks rather than a teacher's, so a run on it alone
  is a run whose supervision nobody has to argue about. `vtuav` tracking and
  `birdsai` are thermal and do not belong in an RGB run at all.
* **The authors' own held-out split.** `test_00x` are the sequences VTUAV-VIS
  says nobody trains on. Passing them to `split_flights(hold_out=...)` grades
  on that choice instead of on a hash of the sequence names.
* **35's checkpoint is the base.** Despite its name, 35 is a Stage B in shape
  -- `METHOD="finetune"`, head then encoder, real mask labels on the RGB pools
  -- so nothing is being skipped by going straight to Stage C from it.

**Stock is in-domain here and that changes what the comparison means.** EdgeTAM
was trained on RGB video (SA-V). On thermal, beating stock is beating a model
outside its domain; on RGB it is not. The two-arm precheck this inherits from
31 prints both, and a loss to stock on RGB is a weaker result than it looks on
thermal -- worth knowing before the number arrives rather than after.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks" / "31_aerial_thermal_tracking.ipynb"
OUTPUT = ROOT / "notebooks" / "37_aerial_rgb_tracking.ipynb"


def source(text: str) -> list[str]:
    return dedent(text).strip("\n").splitlines(keepends=True)


def cell(kind: str, text: str) -> dict:
    item = {"cell_type": kind, "metadata": {}, "source": source(text)}
    if kind == "code":
        item.update({"execution_count": None, "outputs": []})
    return item


def replace(notebook: dict, old: str, new: str, *, count: int = 1) -> None:
    """Replace exact notebook source and fail if 31 drifted underneath us."""
    found = 0
    for item in notebook["cells"]:
        text = "".join(item.get("source", []))
        hits = text.count(old)
        if hits:
            item["source"] = text.replace(old, new).splitlines(keepends=True)
            found += hits
    if found != count:
        raise RuntimeError(f"expected {count} occurrence(s), found {found}: {old!r}")


TITLE = r"""
# 37 — Havadan **RGB** görüntü için Stage C (VTUAV-VIS)

31'in RGB ikizi. 31 termal kalır; bu notebook ona dokunmaz.

**Tek kaynak: VTUAV-VIS.** Burada kullanılan maskeler yazarların kendi kare
başına instance maskeleri — bir teacher modelin ürettikleri değil. `vtuav`
tracking ve `birdsai` termal oldukları için bu koşuda hiç yer almıyor,
Anti-UAV410 ise zaten ters perspektif.

**Held-out gerçek.** `test_00x` arşivleri yazarların "kimse bunlarla eğitmesin"
dediği diziler. `split_flights(hold_out=...)` ile test seti o seçim oluyor,
dizi adlarının hash'i değil.

**Taban 35.** Adı "pretrain" ama yapısı Stage B: `METHOD="finetune"`, önce
head sonra encoder, RGB havuzlarında gerçek maske etiketleriyle. Yani buradan
Stage C'ye geçerken atlanan bir aşama yok.

**Stock burada alan içinde.** EdgeTAM RGB videoda (SA-V) eğitildi. Termalde
stock'u yenmek alan dışı bir modeli yenmekti; RGB'de değil. İki kollu precheck
ikisini de basar — RGB'de stock'a yenilmek termaldekinden daha az şaşırtıcıdır,
ve bunu sayı gelmeden önce bilmek gerekir.
"""

SETTINGS_MD = r"""
## Ayarlar

VTUAV-VIS parçaları büyük: `train_001` 9.1 GB, `train_002` 16.1, `train_003`
17.9; test arşivleri 15-17.7 GB. Maskeler ~her 30. karede ve `--frames masked`
yalnız onları çıkarıyor, yani **bir eğitim + bir test arşivi** ilk koşu için
yeterli. Sekizini birden indirmek 126 GB ve Drive quota'sı demek.

`fetch_datasets` her parçayı **kendi klasörüne** açar ve dizi adları parçayı
taşır. Sebep isimlendirme: VTUAV-VIS dizileri hedef türü + parça başına
sıfırlanan bir sayaçla adlandırılıyor, yani `test_001.zip` içinde `train_003`
adlı bir dizi var ve `train_003.zip` bambaşka bir arşiv. Düz açılsalardı
kareleri aynı klasörde karışırdı.

Drive'da "Bir kopyasını oluştur" ile aldığınız arşivler `MyDrive/datasets/`
altında `test_001.zip adlı dosyanın kopyası` gibi adlanır; `staged()` artık
bu adları da tanıyor, yeniden adlandırmanız gerekmiyor.
"""

SETTINGS = r"""
import errno, gc, json, math, shutil, subprocess, zipfile
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

SIZE, CLIP_LEN, CLIP_STRIDE = 512, 8, 1
SEED = 0
MODALITY = "rgb"

VTUAV_VIS_TRAIN_PARTS = ["train_001"]
VTUAV_VIS_HOLD_OUT_PARTS = ["test_001"]

SOURCE_WEIGHTS = {"vtuav_vis": 1.0}
TRAIN_CLIP_POOL = 6000
VAL_CLIP_POOL   = 800
TEST_CLIP_POOL  = 800
LOW_CONTRAST_QUANTILE = 0.40
LOW_CONTRAST_REPEAT   = 2
CONTRAST_AUDIT_STRIDE = 20

AUGMENT_PROB     = 0.75
GLOBAL_CONTRAST  = (0.35, 0.90)
TARGET_CONTRAST  = (0.15, 0.65)
BRIGHTNESS_SHIFT = (-0.05, 0.05)
SENSOR_NOISE     = (0.0, 0.018)
BLUR_PROB        = 0.20

STEPS_PER_EPOCH = 400
VAL_BATCHES     = 24
BATCH_CEILING   = 128
LOADER_WORKERS  = min(2 * (os.cpu_count() or 4), 24)
PREFETCH_DEPTH  = 2
EVAL_PER_SOURCE = 6
PRECHECK_PER_SOURCE = 4
PRECHECK_FRAMES = 240
PRECHECK_MIN_STATE_ACCURACY = 0.65
PRECHECK_MAX_LOST_RATE = 0.10
PRECHECK_MAX_LONGEST_DROPOUT = 24
PRECHECK_AGAINST_STOCK = True
STOP_AFTER_PRECHECK = False
INSPECT_DATA = True
INSPECT_PER_SOURCE = 6
INSPECT_SPAN = 12
INSPECT_WINDOWS = 2
RENDER_BEFORE_AFTER = True
DEMO_FRAMES, DEMO_FPS, DEMO_PRE_ROLL = 400, 20, 80

DATA = Path("/content/data")
VTUAV_VIS_DATA = DATA / "VTUAV_VIS_rgb_stage_c"
WORK = Path("/content/work/aerial_rgb_tracking")
for directory in (DATA, VTUAV_VIS_DATA, WORK):
    directory.mkdir(parents=True, exist_ok=True)

from google.colab import drive
drive.mount("/content/drive")
DATASETS_DRIVE = Path("/content/drive/MyDrive/datasets")
STAGE_B_DRIVE = Path("/content/drive/MyDrive/edgetam-stage-b")
BASE_STAGE_B_OVERRIDE = ""
BASE_STAGE_B_CANDIDATES = [
    STAGE_B_DRIVE / "pretrain_rgb_aerial"
                  / "edgetam_pool_pretrain_rgb_aerial_512.pt",
]
BASE_STAGE_B = (Path(BASE_STAGE_B_OVERRIDE) if BASE_STAGE_B_OVERRIDE
                else next((path for path in BASE_STAGE_B_CANDIDATES
                           if path.is_file()),
                          BASE_STAGE_B_CANDIDATES[0]))
assert BASE_STAGE_B.is_file(), (
    f"RGB Stage-B checkpoint yok: {BASE_STAGE_B}. Önce notebook 35'i "
    f"bitirin, ya da BASE_STAGE_B_OVERRIDE'a tam yolu yazın.")
BASE_TAG = (Path(BASE_STAGE_B).stem
            .replace("edgetam_pool_", "").replace(f"_{SIZE}", ""))[:48]
MIRROR = Path("/content/drive/MyDrive/edgetam-stage-c") / (
    f"aerial_rgb_tracking_from_{BASE_TAG}")
MIRROR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = REPO / (
    f"checkpoints/edgetam_aerial_rgb_tracking_from_{BASE_TAG}_{SIZE}.pt")
CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
print("Stage-B base:", BASE_STAGE_B)
print("Stage-C output:", MIRROR)
"""

FETCH_MD = r"""
## Veriyi getir

Her parça tek tek çekilir ve `.done` işareti bırakır: bir parça Drive
quota'sına takılırsa yeniden koşmak yalnız eksik olana mal olur. Drive'daki
kendi kopyalarınız `MyDrive/datasets/` altındaysa `staged()` onları ağa
gitmeden bulur -- kopyalama sırasında aldıkları "... kopyası" adlarıyla
birlikte.
"""

FETCH = r"""
VIS_STAGED = VTUAV_VIS_DATA / "_staged"
VIS_STAGED.mkdir(parents=True, exist_ok=True)

def fetch_vis(part):
    marker = VIS_STAGED / f"{part}.done"
    if marker.is_file():
        print("already staged", part)
        return
    subprocess.run([
        sys.executable, "tools/fetch_datasets.py", "vtuav_vis",
        "--dest", str(VTUAV_VIS_DATA), "--parts", part,
        "--frames", "masked"], check=True)
    marker.write_text("ok\n")

for _part in VTUAV_VIS_TRAIN_PARTS + VTUAV_VIS_HOLD_OUT_PARTS:
    fetch_vis(_part)

for _part in VTUAV_VIS_TRAIN_PARTS + VTUAV_VIS_HOLD_OUT_PARTS:
    _where = VTUAV_VIS_DATA / _part
    assert _where.is_dir(), (
        f"{_where} yok. `fetch_datasets` her parçayı kendi klasörüne açar; "
        f"bu klasör yoksa parça inmemiştir.")
    print(f"{_part:<12}{len(list(_where.rglob('mask/' + MODALITY))):>4} "
          f"maskeli dizi")
"""

SPLIT_MD = r"""
## Split: yazarların held-out'u, hash değil

Eğitim parçaları kendi içlerinde train/val'e bölünür; test seti **tamamen**
`test_00x` arşivlerinden gelir. `split_flights(hold_out=...)` verilen test
setini örneklenenin yerine koyar — ikisini karıştırmak, çıkan sayının hangi
yarıdan geldiğini geri alınamaz hale getirirdi.

Kesişme kontrolü burada yapılır ve sessiz geçilmez: iki taraf aynı dizi adını
taşıyorsa held-out held-out değildir.
"""

SPLIT = r"""
from src.training import (
    Sequence, SequenceLabels, empty_stores, flight_name, source_name,
    split_flights, video_clips, vtuav_vis_sequences, weighted_clip_sample,
)

def vis_from(parts):
    rows, stores = [], {}
    for part in parts:
        part_rows, part_stores = vtuav_vis_sequences(
            VTUAV_VIS_DATA / part, modality=MODALITY)
        rows += part_rows
        stores.update(part_stores)
    return rows, stores

train_sequences, TRAIN_STORES = vis_from(VTUAV_VIS_TRAIN_PARTS)
held_sequences, HELD_STORES = vis_from(VTUAV_VIS_HOLD_OUT_PARTS)
sequences = train_sequences + held_sequences
VIS_STORES = {**TRAIN_STORES, **HELD_STORES}

assert train_sequences, "eğitim parçalarından hiç dizi çıkmadı"
assert held_sequences, "held-out parçalarından hiç dizi çıkmadı"
_overlap = ({row.name for row in train_sequences}
            & {row.name for row in held_sequences})
assert not _overlap, (
    f"held-out ve eğitim aynı dizi adını paylaşıyor: {sorted(_overlap)}. "
    f"Parçalar kendi klasörlerine açılmadıysa adlar parçayı taşımaz ve "
    f"held-out held-out değildir.")

SPLITS = split_flights(train_sequences, seed=SEED, val_fraction=0.20,
                       test_fraction=0.0, hold_out=held_sequences)
print(f"{'part':<8}{'diziler':>9}{'kare':>10}{'maskeli':>10}")
for _part, _rows in SPLITS.items():
    print(f"{_part:<8}{len(_rows):>9}{sum(len(r) for r in _rows):>10}"
          f"{sum(len(VIS_STORES.get(r.name, {})) for r in _rows):>10}")
assert all(SPLITS.values()), "split'lerden biri boş."
"""

CLIPS = r"""
CONTRAST = {row.name: sequence_contrast(row) for row in sequences}
LOW_NAMES = set()
_rows = [row for row in SPLITS["train"] if np.isfinite(CONTRAST[row.name])]
_rows.sort(key=lambda row: CONTRAST[row.name])
LOW_NAMES.update(row.name for row in
                 _rows[:max(1, math.ceil(len(_rows) * LOW_CONTRAST_QUANTILE))])
print("low-contrast cut",
      CONTRAST[_rows[max(0, math.ceil(len(_rows) * LOW_CONTRAST_QUANTILE) - 1)].name]
      if _rows else "n/a")

def make_clips(rows, jitter):
    return video_clips(rows, length=CLIP_LEN, stride=1, size=SIZE,
                       min_visible=2, jitter=jitter, seed=SEED)

raw_train = make_clips(SPLITS["train"], jitter=32)
raw_train += [clip for clip in raw_train
              if clip.sequence.name in LOW_NAMES] * (LOW_CONTRAST_REPEAT - 1)
raw_val = make_clips(SPLITS["val"], jitter=0)
raw_test = make_clips(SPLITS["test"], jitter=0)
TRAIN_CLIPS = weighted_clip_sample(
    raw_train, SOURCE_WEIGHTS, TRAIN_CLIP_POOL, SEED)
VAL_CLIPS = weighted_clip_sample(raw_val, SOURCE_WEIGHTS, VAL_CLIP_POOL, SEED + 1)
TEST_CLIPS = weighted_clip_sample(raw_test, SOURCE_WEIGHTS, TEST_CLIP_POOL, SEED + 2)

STORES = empty_stores(sequences)
STORES.update(VIS_STORES)

if INSPECT_DATA:
    from tools.inspect_stage_c import render
    print("\ncontact sheets -- what Stage C is about to train on:")
    render(sequences, STORES, MIRROR / "inspect",
           per_source=INSPECT_PER_SOURCE, span=INSPECT_SPAN,
           windows=INSPECT_WINDOWS)

MASK_FRAMES = sum(len(STORES[row.name]) for row in sequences)
VISIBLE_FRAMES = sum(int(row.labels.exist.sum()) for row in sequences)
print(f"\nmask supervision  {MASK_FRAMES}/{VISIBLE_FRAMES} visible frames "
      f"({MASK_FRAMES / max(VISIBLE_FRAMES, 1):.1%})")
print("!! VTUAV-VIS'te `exist` her karede True: bu koşuda kaybolma denetimi "
      "YOK,\n   yani object-score kafası hiçbir şey öğrenmiyor. 31'in vtuav "
      "kolu gerçek\n   `exist` taşıyan tek kaynaktır.")

for _label, _clips in (("train", TRAIN_CLIPS), ("val", VAL_CLIPS),
                       ("test", TEST_CLIPS)):
    print(_label, len(_clips))
"""


CLIPS_MD = r"""
## Gerçek yerel kontrast ve clip havuzu

Tek kaynak olduğu için kaynaklar arası ağırlıklandırma burada bir şey yapmıyor;
kalan iş düşük kontrastı ayakta tutmak. Diziler kontrast sırasına sokulur ve
alt %40 aday havuzuna iki kez girer, böylece parlak gündüz uçuşları havuzu
tek başına doldurmaz.
"""

DEMO_MD = r"""
## Before / after videoları

Validation'da ölçülmüş dizilerden iki vaka: düşük-kontrast alt grubunda en iyi
kazanç ve kaydedilmiş **en kötü gerileme**. Yeşil ground truth, kırmızı taban
(35), camgöbeği seçilmiş final koldur. İki model aynı ilk kutuyu ve aynı sabit
crop'u kullanır.

En kötü gerilemenin de gösterilmesi bilinçli: ortalama iyileşen bir koşu tek
tek dizilerde çökebilir ve karar o dizilerde verilir.
"""

VERDICT_MD = r"""
## Karar

**Önce stock satırına bakın.** EdgeTAM RGB videoda eğitildi, yani burada alan
içinde. Termalde stock'u yenmek alan dışı bir modeli yenmekti; RGB'de öyle
değil. Stock'a yakın durmak bile 35 + Stage C zincirinin RGB'yi bozmadığı
anlamına gelir; yenmek gerçek bir sonuçtur.

Sonra held-out tablosuna bakın — `test_00x` yazarların seçtiği dizilerdir,
hash'in değil. Test gerilerse checkpoint'i deploy etmeyin: validation'da iyi
görünen tek bir video karar değildir.

**Bu koşuda kaybolma denetimi yok.** VTUAV-VIS `exist`'i her karede True, yani
object-score kafası hiçbir şey öğrenmiyor ve kaybolup geri gelme davranışı
burada ne eğitiliyor ne ölçülüyor. O soru 31'in `vtuav` kolunda yaşar.

Bu model bir sınıf dedektörü değildir: ilk prompt kutusundan sonra herhangi bir
instance maskesini takip eden EdgeTAM'dır. Stage C kimliği sınıf adı yerine
zaman içindeki aynı nesne olarak öğrenir.
"""


def main() -> None:
    notebook = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    cells[0] = cell("markdown", TITLE)
    cells[3] = cell("markdown", SETTINGS_MD)
    cells[4] = cell("code", SETTINGS)
    cells[5] = cell("markdown", FETCH_MD)
    cells[6] = cell("code", FETCH)
    cells[7] = cell("markdown", SPLIT_MD)
    cells[8] = cell("code", SPLIT)
    cells[9] = cell("markdown", CLIPS_MD)
    cells[23] = cell("markdown", DEMO_MD)
    cells[25] = cell("markdown", VERDICT_MD)

    # Cell 10 keeps `frame_local_contrast` and `sequence_contrast` -- they read
    # a frame and a box and do not care which modality produced them -- and
    # loses everything downstream of them: the per-source loop over a single
    # source, the thermal teacher pool, and its assert.
    text = "".join(cells[10]["source"])
    keep = text[:text.index("CONTRAST = {row.name:")]
    cells[10]["source"] = source(keep + CLIPS.strip("\n"))

    # The run records name what the run was made of, and 31's fields are for
    # sources this notebook does not have. Replaced rather than dropped: a
    # report that does not say which archives it read is the one nobody can
    # reproduce.
    replace(notebook,
            '"sources": SOURCE_WEIGHTS, "use_antiuav410": USE_ANTIUAV410,\n'
            '        "vtuav_archives": VTUAV_ARCHIVES,',
            '"sources": SOURCE_WEIGHTS, "modality": MODALITY,\n'
            '        "vtuav_vis_train_parts": VTUAV_VIS_TRAIN_PARTS,\n'
            '        "vtuav_vis_hold_out_parts": VTUAV_VIS_HOLD_OUT_PARTS,')
    replace(notebook,
            '"use_antiuav410": USE_ANTIUAV410, "vtuav_archives": VTUAV_ARCHIVES,',
            '"modality": MODALITY,\n'
            '    "vtuav_vis_train_parts": VTUAV_VIS_TRAIN_PARTS,\n'
            '    "vtuav_vis_hold_out_parts": VTUAV_VIS_HOLD_OUT_PARTS,')
    replace(notebook,
            '"vtuav_vis_parts": VTUAV_VIS_PARTS if USE_VTUAV_VIS else [],',
            '"base_stage_b_tag": BASE_TAG,', count=2)
    # The teacher pool is 31's, and with it go the two numbers the records
    # quoted from it. Recorded here instead: how much of the run was actually
    # supervised by a mask -- the same question, answered directly, because
    # this source's masks are the authors' own.
    replace(notebook, '"teacher_mask_frames": teacher_mask_frames,',
            '"mask_frames": MASK_FRAMES, "visible_frames": VISIBLE_FRAMES,',
            count=2)
    replace(notebook, '"teacher_min_box_iou": TEACHER_MIN_BOX_IOU,',
            '"hold_out": VTUAV_VIS_HOLD_OUT_PARTS,', count=2)

    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(notebook['cells'])} cells)")


if __name__ == "__main__":
    main()
