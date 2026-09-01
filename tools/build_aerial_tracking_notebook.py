#!/usr/bin/env python3
"""Build notebook 31: aerial-first thermal video continuation of Stage B."""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks" / "29_thermal_contrast_tracking.ipynb"
OUTPUT = ROOT / "notebooks" / "31_aerial_thermal_tracking.ipynb"


def source(text: str) -> list[str]:
    return dedent(text).strip("\n").splitlines(keepends=True)


def cell(kind: str, text: str) -> dict:
    item = {"cell_type": kind, "metadata": {}, "source": source(text)}
    if kind == "code":
        item.update({"execution_count": None, "outputs": []})
    return item


def main() -> None:
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    cells = [
        cell("markdown", r"""
        # 31 — Havadan termal görüntü için çok-veri-setli Stage C

        Bu notebook tercihen `32_aerial_thermal_stage_b_stable.ipynb`
        checkpoint'inden (yoksa 22 fallback'inden) devam eder, fakat
        `29`daki yer-kamerası ağırlıklı Anti-UAV410 eğitimini ana kaynak yapmaz.

        - **VTUAV-TIR:** UAV üstündeki termal kameradan tek-hedef tracking;
          short/long-term diziler, gerçek kaybolma satırları. Notebook 17/25'in
          SAM-teacher maskeleri eşleşen karelerde doğrudan kullanılır.
        - **VTUAV-VIS:** Resmî instance-mask video sürümü. Maskeli kareler
          yaklaşık 30 gerçek kare aralıklı olduğundan uzun-zaman maskeli
          denetim verir; kutular maskenin kendisinden çıkarılır.
        - **BIRDSAI:** gece, UAV üstündeki TIR kamera; insan/hayvan ve gerçek track
          ID. İnterpole edilmiş `noise=1` kutular yanlış negatif yapılmaz, klibi
          böler.
        - **Anti-UAV410:** ters perspektif olduğu için varsayılan kapalıdır.
          `USE_ANTIUAV410=True` ile yaklaşık %9 düzenleyici payla eklenebilir.

        HIT-UAV, DroneVehicle-TIR, SegFly, Kust4K ve VTUAV statik havuzlarının
        görevi Stage B'de **şekil/termal görünüş** öğretmektir. Burada yalnız video
        kimliği olan kaynaklar bellek yolunu eğitir. Gerçek veya kabul edilmiş
        teacher maskesi olan karelerde mask loss, diğerlerinde box-projection +
        object-score kullanılır. Teacher'ın reddettiği maske boş hedef sayılmaz.

        Kaynaklar: [VTUAV (CVPR 2022)](https://github.com/zhang-pengyu/DUT-VTUAV),
        [BIRDSAI (WACV 2020)](https://openaccess.thecvf.com/content_WACV_2020/html/Bondi_BIRDSAI_A_Dataset_for_Detection_and_Tracking_in_Aerial_Thermal_Infrared_Videos_WACV_2020_paper.html),
        [HIT-UAV](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset),
        [DroneVehicle](https://github.com/VisDrone/DroneVehicle).
        """),
        cell("code", r"""
        # --- Runtime -----------------------------------------------------------
        import os, sys
        from pathlib import Path

        REPO   = Path("/content/sam-dedection")
        BRANCH = "claude/thermal-stage-b-training-43ktcl"
        if not REPO.exists():
            !git clone -q -b {BRANCH} https://github.com/yigitkayabagci/sam-dedection.git {REPO}
        !git -C {REPO} fetch -q origin {BRANCH}
        !git -C {REPO} checkout -q {BRANCH} && git -C {REPO} merge -q --ff-only origin/{BRANCH}
        os.chdir(REPO)
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        !bash scripts/setup_edgetam.sh 2>&1 | tail -5
        !pip install -q -r requirements.txt
        !pip install -q gdown tqdm matplotlib
        !nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        """),
        cell("code", r"""
        !python -m unittest -q tests.test_aerial_video tests.test_clip_loop \
            tests.test_training_losses tests.test_samurai tests.test_fetch_datasets \
            2>&1 | tail -3
        """),
        cell("markdown", r"""
        ## Ayarlar

        VTUAV'ın bütün tracking training arşivleri 200 GiB üstüdür; notebook
        yalnız seçilen arşivlerden termal ve kutulu kareleri çıkarır. Dört LT
        parçası gerçek kaybolma/yeniden belirme davranışı için varsayılan açıktır.
        Daha önce 17/25'i çalıştırdıysanız kaynak zip'ler ve teacher maskeleri
        zaten Drive'dadır.

        VTUAV-VIS'in üç eğitim zip'i toplam yaklaşık 43 GiB'dır. Varsayılan yalnız
        `train_001` (yaklaşık 9.1 GiB indirme, 875 maskeli kare) ve `--frames
        masked` kullanır. Daha çok insan sınıfı isterseniz `train_003`ü listeye
        ekleyin; ilk doğrulama koşusunda üçünü birden indirmeyin.

        `USE_ANTIUAV410=False` kasıtlıdır: o veri düşük-kontrast tracking için
        değerli olsa da kamerası yerde, hedefi havadadır. Deployment alanını ana
        eğitim dağılımında outvote etmesine izin vermiyoruz.
        """),
        cell("code", r"""
        import errno, gc, json, math, shutil, subprocess, zipfile
        import cv2
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        import yaml
        from tqdm.auto import tqdm

        SIZE, CLIP_LEN, CLIP_STRIDE = 512, 8, 1
        SEED = 0
        VTUAV_ARCHIVES = [
            "train_ST_001.zip",  # animal / bike / bus
            "train_ST_008.zip",  # çoğunlukla pedestrian
            "train_LT_001.zip",  # disappear/reappear davranışı
            "train_LT_002.zip",
            "train_LT_003.zip",
            "train_LT_004.zip",
        ]
        # Hangi kaynak açık. Hepsi kapatılabilir olmasının sebebi ölçüm:
        # üç kaynaklı karışım bir kolun sonucunu diğerine karıştırıyor, ve
        # VTUAV tracking arşivleri başkasının Drive'ındaki kısayollar olduğu
        # için koşuyu en sık düşüren parça onlar. `weighted_clip_sample`
        # kaynağı olmayan ağırlığı zaten düşürüp yeniden normalize ediyor,
        # yani SOURCE_WEIGHTS'e dokunmadan kol seçilebiliyor.
        USE_VTUAV      = False
        USE_VTUAV_VIS  = True
        USE_BIRDSAI    = False
        USE_ANTIUAV410 = False
        # Sekiz resmî VTUAV-VIS arşivi. Boyutlar toplam ~126 GB, ama
        # `--frames masked` yalnız maskeli kareleri ve eşleniklerini açıyor,
        # yani diske inen bunun %4'ü kadarı. Pahalı olan okuma, disk değil.
        # Recipe'nin ölçülmüş sayıları: train_001 14 dizi / 875 maske ve
        # **hiç pedestrian yok**, train_002 18 / 1408, train_003 18 / 1778.
        # test_001..005 yazarların held-out'u; maske sayıları ölçülmedi.
        VTUAV_VIS_PARTS = ["train_001", "train_002", "train_003",
                           "test_001", "test_002", "test_003",
                           "test_004", "test_005"]
        ANTIUAV_LIMIT = 24
        SOURCE_WEIGHTS = {"vtuav": 0.40, "vtuav_vis": 0.30,
                          "birdsai": 0.30,
                          "antiuav410": 0.10}

        TRAIN_CLIP_POOL = 12000
        VAL_CLIP_POOL   = 1200
        TEST_CLIP_POOL  = 1200
        LOW_CONTRAST_QUANTILE = 0.40
        LOW_CONTRAST_REPEAT   = 2
        CONTRAST_AUDIT_STRIDE = 20
        TEACHER_MIN_BOX_IOU   = 0.80
        MIN_TEACHER_MASK_FRAMES = 1000

        AUGMENT_PROB     = 0.75
        GLOBAL_CONTRAST  = (0.35, 0.90)
        TARGET_CONTRAST  = (0.15, 0.65)
        BRIGHTNESS_SHIFT = (-0.05, 0.05)
        SENSOR_NOISE     = (0.0, 0.018)
        BLUR_PROB        = 0.20

        STEPS_PER_EPOCH = 500
        VAL_BATCHES     = 32
        BATCH_CEILING   = 128
        LOADER_WORKERS  = min(2 * (os.cpu_count() or 4), 24)
        PREFETCH_DEPTH  = 2
        EVAL_PER_SOURCE = 4
        PRECHECK_PER_SOURCE = 2
        PRECHECK_FRAMES = 240
        PRECHECK_MIN_STATE_ACCURACY = 0.65
        PRECHECK_MAX_LOST_RATE = 0.10
        PRECHECK_MAX_LONGEST_DROPOUT = 24
        # Precheck'i stock EdgeTAM ile de koş. Aynı kliplerin üstünde iki kol:
        # "Stage C'ye ihtiyaç var mı" ile "Stage B takibi bozmuş mu" farklı
        # sorular, ve ikincisi ancak stock satırı yanında dururken okunur.
        # Maliyeti bir GPU geçişi daha, eğitim değil.
        PRECHECK_AGAINST_STOCK = True
        STOP_AFTER_PRECHECK = False  # True: yalnız audit yap, eğitim hücresine geçme
        # Gözle kontrol. Kaynak başına birkaç dizi, her birinde `exist`
        # değişimlerinin etrafına yerleşmiş kareler; sayfalar Drive'a
        # (MIRROR/inspect) yazılır. Kayboluş/geri geliş anları uniform
        # örneklemede neredeyse hiç görünmediği için pencereler oraya konur.
        INSPECT_DATA = True
        INSPECT_PER_SOURCE = 4
        INSPECT_SPAN = 12
        INSPECT_WINDOWS = 2
        RENDER_BEFORE_AFTER = True
        DEMO_FRAMES, DEMO_FPS, DEMO_PRE_ROLL = 400, 20, 80

        DATA = Path("/content/data")
        VTUAV_DATA = DATA / "VTUAV_aerial_stage_c"
        VTUAV_VIS_DATA = DATA / "VTUAV_VIS_aerial_stage_c"
        BIRDSAI_DATA = DATA / "BIRDSAI"
        ANTI_DATA = DATA / "AntiUAV410"
        TEMPORAL_POOL_ROOT = Path("/content/pool/aerial_stage_c")
        WORK = Path("/content/work/aerial_thermal_tracking")
        # CHECKPOINT ve MIRROR aşağıda, taban seçildikten sonra kurulur: ikisi
        # de tabanın adını taşır.
        for directory in (DATA, VTUAV_DATA, VTUAV_VIS_DATA, BIRDSAI_DATA,
                          TEMPORAL_POOL_ROOT, WORK):
            directory.mkdir(parents=True, exist_ok=True)

        from google.colab import drive
        drive.mount("/content/drive")
        VTUAV_DRIVE = Path("/content/drive/MyDrive/VTUAV")
        POOL_DRIVE = Path("/content/drive/MyDrive/edgetam-pool")
        STAGE_B_DRIVE = Path("/content/drive/MyDrive/edgetam-stage-b")
        # 32'nin bir koşusunun tam yolu. Boş bırakılırsa aşağıdaki aday
        # kullanılır. 32 pretrain tabanıyla koşulduğunda çıktısının adı
        # `_from_<pretrain>` eki alır, yani bu satır o koşuya geçmenin yeridir:
        #   aerial_thermal_stable_from_pretrain_thermal_aerial/
        #     edgetam_pool_aerial_thermal_stable_from_pretrain_thermal_aerial_512.pt
        BASE_STAGE_B_OVERRIDE = ""
        BASE_STAGE_B_CANDIDATES = [
            STAGE_B_DRIVE / "aerial_thermal_stable"
                          / "edgetam_pool_aerial_thermal_stable_512.pt",
        ]
        BASE_STAGE_B = (Path(BASE_STAGE_B_OVERRIDE) if BASE_STAGE_B_OVERRIDE
                        else next((path for path in BASE_STAGE_B_CANDIDATES
                                   if path.is_file()),
                                  BASE_STAGE_B_CANDIDATES[0]))
        assert BASE_STAGE_B.is_file(), (
            f"Stage-B checkpoint yok: {BASE_STAGE_B}. Önce notebook 32'yi "
            f"bitirin, ya da BASE_STAGE_B_OVERRIDE'a koşmak istediğiniz "
            f"checkpoint'in tam yolunu yazın.")
        # Her Stage C koşusu tabanının adını taşır. Taşımasaydı, iki farklı
        # Stage B çıktısından koşulan iki Stage C aynı klasöre ve aynı
        # checkpoint adına yazar ve ikincisi birincisini sessizce ezerdi --
        # 32'nin `_from_` eki bunun için var, burada da aynısı gerekiyor.
        BASE_TAG = (Path(BASE_STAGE_B).stem
                    .replace("edgetam_pool_", "").replace(f"_{SIZE}", ""))[:48]
        MIRROR = Path("/content/drive/MyDrive/edgetam-stage-c") / (
            f"aerial_thermal_tracking_from_{BASE_TAG}")
        MIRROR.mkdir(parents=True, exist_ok=True)
        CHECKPOINT = REPO / (
            f"checkpoints/edgetam_aerial_thermal_tracking_from_{BASE_TAG}"
            f"_{SIZE}.pt")
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        print("Stage-B base:", BASE_STAGE_B)
        print("Stage-C output:", MIRROR)
        """),
        cell("markdown", r"""
        ## Veriyi getir

        BIRDSAI'nin yalnız gerçek gece videoları indirilir; 42 GiB AirSim sentetik
        kol varsayılan tarifte kapalıdır. Tracking zip'leri ağdan sessizce çekilmez:
        seçilen büyük arşivlerin Drive'da bulunması zorunludur. VTUAV-VIS ise
        resmî Drive kimlikleri üzerinden yalnız seçilen parçayı indirir.
        """),
        cell("code", r"""
        from tools.fetch_datasets import extract

        missing_archives = [name for name in VTUAV_ARCHIVES
                            if USE_VTUAV and not (VTUAV_DRIVE / name).is_file()]
        if missing_archives:
            print("Drive mount bu shortcut/dosyaları çözemedi; resmî kimliklerden "
                  "sırayla indirilecek:", missing_archives)
            subprocess.run([
                sys.executable, "tools/fetch_datasets.py", "vtuav_track",
                "--dest", str(VTUAV_DATA), "--parts",
                *[Path(name).stem for name in missing_archives],
                "--frames", "tracked_ir"], check=True)

        staged = VTUAV_DATA / "_staged"
        staged.mkdir(exist_ok=True)
        for archive_name in (VTUAV_ARCHIVES if USE_VTUAV else []):
            marker = staged / f"{archive_name}.done"
            if marker.is_file():
                print("already staged", archive_name)
                continue
            archive = VTUAV_DRIVE / archive_name
            if archive.is_file():
                print("extracting labelled thermal frames from", archive_name)
                staged_ok = False
                for threads in (16, 8, 4, 1):
                    try:
                        extract(archive, VTUAV_DATA, frames="tracked_ir",
                                workers=threads)
                        staged_ok = True
                        break
                    except OSError as dropped:
                        if dropped.errno != errno.ENOTCONN:
                            raise
                        print(f"   !! the Drive mount dropped mid-read "
                              f"(errno 107) on {threads} threads. "
                              f"`tracked_ir` keeps a twentieth of the part, so "
                              f"the read is thousands of random seeks and each "
                              f"thread holds its own handle -- remounting and "
                              f"retrying with fewer.")
                        drive.flush_and_unmount()
                        drive.mount("/content/drive", force_remount=True)
                if not staged_ok:
                    raise RuntimeError(
                        f"{archive_name}: the Drive mount dropped on every "
                        f"attempt down to a single thread. The archives are "
                        f"shortcuts to files this account does not own, which "
                        f"is the fragile case; copy the part to local disk "
                        f"first (`!cp {archive} /content/`) and point "
                        f"VTUAV_DRIVE at /content, or run the part on its own "
                        f"in a fresh runtime.")
            else:
                print("downloaded directly and extracted:", archive_name)
            marker.write_text("ok\n")

        # 17/25'in küçük mask-only arşivlerini aç. Kaynak görüntüler kopyalanmaz.
        # Bu havuz yalnız VTUAV tracking karelerini maskeleyebilir -- kutu
        # eşleşmesi o dizilere karşı yapılıyor -- yani vtuav kapalıyken
        # indirmenin bir alıcısı yok.
        pool_markers = TEMPORAL_POOL_ROOT / ".unpacked"
        pool_markers.mkdir(exist_ok=True)
        for pool_name in (("vtuav_thermal", "vtuav_lt_thermal") if USE_VTUAV else ()):
            folder = POOL_DRIVE / pool_name
            archives = sorted(folder.glob("*.zip")) if folder.is_dir() else []
            if not archives:
                print("teacher pool not found (training öncesi doğrulama durduracak):",
                      folder)
            for archive in archives:
                marker = pool_markers / f"{pool_name}__{archive.name}.done"
                if marker.is_file():
                    continue
                with zipfile.ZipFile(archive) as handle:
                    members = handle.namelist()
                    if not any(name.endswith("record.json") for name in members):
                        print("not a mask pool, skipped:", archive)
                        continue
                    handle.extractall(TEMPORAL_POOL_ROOT)
                marker.write_text("ok\n")

        if USE_VTUAV_VIS:
            # One part per call, with the same `.done` markers the tracking
            # archives above use. These parts are 8.5, 15 and 17 GB, and a
            # Drive refusal on one of them is the failure that actually
            # happens; asking for all three again would re-download the ones
            # that already landed and spend the quota before reaching the one
            # that did not. `fetch_datasets` keeps what it got either way --
            # this is what stops the *retry* from costing anything.
            vis_staged = VTUAV_VIS_DATA / "_staged"
            vis_staged.mkdir(parents=True, exist_ok=True)
            for vis_part in VTUAV_VIS_PARTS:
                vis_marker = vis_staged / f"{vis_part}.done"
                if vis_marker.is_file():
                    print("already staged", vis_part)
                    continue
                # Her parça kendi klasörüne. VTUAV dizilerini hedefin türüne
                # göre adlandırıyor -- `train_003` üçüncü *tren* videosudur,
                # train split'i değil -- ve o yüzden `test_001.zip` içinden
                # `train_003/` çıkıyor. Sekizi tek klasöre açmak, iki arşivde
                # aynı ada rastlanırsa kareleri diskte sessizce birleştirir ve
                # ortaya hiçbir yerde olmayan bir dizi çıkarırdı.
                subprocess.run([
                    sys.executable, "tools/fetch_datasets.py", "vtuav_vis",
                    "--dest", str(VTUAV_VIS_DATA / vis_part), "--parts", vis_part,
                    "--frames", "masked"], check=True)
                vis_marker.write_text("ok\n")

        if USE_BIRDSAI:
            subprocess.run([
                sys.executable, "tools/fetch_datasets.py", "birdsai",
                "--dest", str(BIRDSAI_DATA), "--parts", "train_real"], check=True)

        if USE_ANTIUAV410:
            subprocess.run([
                sys.executable, "tools/fetch_antiuav410.py", "--dest", str(ANTI_DATA),
                "--splits", "train"], check=True)
        """),
        cell("markdown", r"""
        ## Gerçek uçuş bazında split ve veri ağırlığı

        BIRDSAI'da bir uçuş içindeki bütün track ID'ler aynı split'te kalır.
        Kaynaklar kendi içinde bölünür; büyük VTUAV küçük BIRDSAI validation'ını
        yok edemez. Ağırlıklar `SOURCE_WEIGHTS`'ten gelir ama **kapalı kaynağın
        payı düşürülüp kalanlar yeniden normalize edilir**
        (`weighted_clip_sample`), yani kolu `USE_*` anahtarları seçer. Üçü de
        açıkken pay VTUAV %40 / VTUAV-VIS %30 / BIRDSAI %30, yalnız VTUAV-VIS
        açıkken %100.

        Kaynak başına **absent** (kaybolma) sayısı burada basılır. Bu bir
        raporlama süsü değil: aşağıdaki `object_score` ağırlığı doğrudan ona
        bakıyor, çünkü üç kaynaktan yalnız VTUAV tracking gerçek `exist`
        taşıyor -- diğer ikisi `np.ones`. Yalnız VTUAV-VIS ile koşulan bir
        Stage C'de hiç kayboluş karesi yoktur ve terim kapanır.
        """),
        cell("code", r"""
        from src.training import (
            Sequence, SequenceLabels, birdsai_sequences, empty_stores,
            flight_name, list_sequences, pool_sequence_stores, source_name,
            split_flights, video_clips, vtuav_sequences,
            vtuav_vis_sequences, weighted_clip_sample,
        )

        sequences = []
        vtuav = vtuav_sequences(VTUAV_DATA, modality="ir") if USE_VTUAV else []
        sequences += vtuav
        VIS_STORES = {}
        if USE_VTUAV_VIS:
            vis_sequences, VIS_STORES = vtuav_vis_sequences(
                VTUAV_VIS_DATA, modality="ir")
            _vis_names = [row.name for row in vis_sequences]
            _clashing = sorted({name for name in _vis_names
                                if _vis_names.count(name) > 1})
            assert not _clashing, (
                f"iki VTUAV-VIS parçası aynı dizi adını taşıyor: {_clashing}. "
                f"Parçalar ayrı klasörlere açıldığı için bu diskte birleşme "
                f"değil, ama `STORES` ada göre anahtarlı: biri diğerinin "
                f"maskelerini gölgeler. Hangi parçaların çakıştığını "
                f"`tools/inspect_vtuav_vis.py` ile bulun.")
            sequences += vis_sequences
        if USE_BIRDSAI:
            sequences += birdsai_sequences(
                BIRDSAI_DATA, split="TrainReal", min_run=(CLIP_LEN - 1) * 2 + 1)
        if USE_ANTIUAV410:
            anti = list_sequences(ANTI_DATA, "train")[:ANTIUAV_LIMIT]
            sequences += [Sequence(
                name=f"antiuav410__{item.name}", split="unsplit",
                frames=item.frames,
                labels=SequenceLabels(
                    exist=item.labels.exist.copy(), boxes=item.labels.boxes.copy()))
                for item in anti]

        assert sequences, (
            "hiçbir kaynak açık değil ya da hiçbiri inmedi: USE_VTUAV, "
            "USE_VTUAV_VIS, USE_BIRDSAI hepsi boş sonuç verdi.")

        # Kaç karede hedef gerçekten yok. `object_score` başlığı yalnız
        # burada bir şey öğrenebilir, ve üç kaynaktan yalnız vtuav gerçek
        # `exist` taşıyor (aerial_video.py: vtuav_sequences ir.txt'ten okuyor,
        # vtuav_vis_sequences ve birdsai_sequences np.ones veriyor). Sayı
        # burada ölçülüyor çünkü loss ağırlığı aşağıda ona bakacak: sabit 1'e
        # karşı BCE, o başlığa koşulsuz ateşlemeyi öğretir.
        ABSENT_FRAMES = {}
        for _source in sorted({source_name(row) for row in sequences}):
            _rows = [row for row in sequences if source_name(row) == _source]
            _frames = sum(len(row) for row in _rows)
            _visible = sum(int(np.count_nonzero(row.labels.exist))
                           for row in _rows)
            ABSENT_FRAMES[_source] = _frames - _visible
        TOTAL_ABSENT = sum(ABSENT_FRAMES.values())
        print("absent (exist=False) frames per source:", ABSENT_FRAMES)

        SPLITS = split_flights(
            sequences, seed=SEED, val_fraction=0.20, test_fraction=0.10)
        print(f"{'part':<8}{'source':<14}{'flights':>9}{'tracks':>9}{'frames':>10}")
        for part, rows in SPLITS.items():
            for source in sorted({source_name(row) for row in rows}):
                selected = [row for row in rows if source_name(row) == source]
                print(f"{part:<8}{source:<14}{len({flight_name(r) for r in selected}):>9}"
                      f"{len(selected):>9}{sum(len(r) for r in selected):>10}")
        assert all(SPLITS.values()), "train/val/test split'lerinden biri boş."
        """),
        cell("markdown", r"""
        ## Gerçek yerel kontrast ve clip havuzu

        Her kaynak kendi içinde kontrast sırasına sokulur; alt %40 iki kez aday
        havuzuna girer. Ardından kaynak ağırlığı uygulanır. Böylece “VTUAV büyük
        olduğu için BIRDSAI ve düşük-kontrast örnekler hiç seçilmedi” durumu olmaz.
        """),
        cell("code", r"""
        def frame_local_contrast(sequence, index):
            image = cv2.imread(str(sequence.frames[index]), cv2.IMREAD_GRAYSCALE)
            box = sequence.labels.boxes[index]
            if image is None or not np.isfinite(box).all():
                return np.nan
            x0, y0, x1, y1 = box
            height, width = image.shape
            side = max(x1 - x0, y1 - y0, 2)
            pad = max(6, int(round(side)))
            ax0, ay0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
            ax1, ay1 = min(width, int(np.ceil(x1)) + pad), min(height, int(np.ceil(y1)) + pad)
            tx0, ty0 = max(0, int(x0)), max(0, int(y0))
            tx1, ty1 = min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1)))
            target = image[ty0:ty1, tx0:tx1].astype(np.float32)
            patch = image[ay0:ay1, ax0:ax1].astype(np.float32)
            if target.size < 4 or patch.size <= target.size:
                return np.nan
            ring = np.ones(patch.shape, dtype=bool)
            ring[ty0 - ay0:ty1 - ay0, tx0 - ax0:tx1 - ax0] = False
            background = patch[ring]
            return (float(abs(target.mean() - background.mean()) /
                          max(background.std(), 3.0))
                    if background.size >= 8 else np.nan)

        def sequence_contrast(sequence):
            visible = sequence.labels.visible_indices()[::CONTRAST_AUDIT_STRIDE]
            values = [frame_local_contrast(sequence, int(index)) for index in visible]
            values = np.asarray([value for value in values if np.isfinite(value)])
            return float(np.median(values)) if values.size else np.nan

        CONTRAST = {row.name: sequence_contrast(row) for row in sequences}
        LOW_NAMES = set()
        for source in sorted({source_name(row) for row in SPLITS["train"]}):
            rows = [row for row in SPLITS["train"]
                    if source_name(row) == source and np.isfinite(CONTRAST[row.name])]
            rows.sort(key=lambda row: CONTRAST[row.name])
            LOW_NAMES.update(row.name for row in
                             rows[:max(1, math.ceil(len(rows) * LOW_CONTRAST_QUANTILE))])
            print(source, "low-contrast cut",
                  CONTRAST[rows[max(0, math.ceil(len(rows) * LOW_CONTRAST_QUANTILE) - 1)].name]
                  if rows else "n/a")

        SOURCE_STRIDES = {"vtuav": 1, "vtuav_vis": 1,
                          "birdsai": 2, "antiuav410": 2}

        def make_clips(rows, jitter):
            clips = []
            for source in sorted({source_name(row) for row in rows}):
                selected = [row for row in rows if source_name(row) == source]
                clips += video_clips(
                    selected, length=CLIP_LEN, stride=SOURCE_STRIDES[source],
                    size=SIZE, min_visible=2, jitter=jitter, seed=SEED)
            return clips

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
        # Teacher havuzu VTUAV tracking karelerini maskeler; vtuav kapalıysa
        # eşleşecek dizi yok ve eşik "eksik veri" değil "istenmeyen kaynak"
        # anlamına gelirdi. Kapı o yüzden kolun içinde duruyor: açıkken hâlâ
        # box-only başlamayı reddediyor.
        TEACHER_STORES = pool_sequence_stores(
            TEMPORAL_POOL_ROOT, vtuav,
            {"vtuav_thermal", "vtuav_lt_thermal"},
            min_box_iou=TEACHER_MIN_BOX_IOU) if USE_VTUAV else {}
        teacher_mask_frames = sum(len(store) for store in TEACHER_STORES.values())
        assert not USE_VTUAV or teacher_mask_frames >= MIN_TEACHER_MASK_FRAMES, (
            f"Teacher pool eşleşmesi yalnız {teacher_mask_frames} kare verdi; "
            f"en az {MIN_TEACHER_MASK_FRAMES} bekleniyor. Training'i box-only "
            "başlatmak yerine edgetam-pool/vtuav_* arşivlerini kontrol edin.")
        STORES.update(TEACHER_STORES)
        STORES.update(VIS_STORES)

        if INSPECT_DATA:
            from tools.inspect_stage_c import render
            print("\ncontact sheets -- what Stage C is about to train on:")
            render(sequences, STORES, MIRROR / "inspect",
                   per_source=INSPECT_PER_SOURCE, span=INSPECT_SPAN,
                   windows=INSPECT_WINDOWS)

        print("\nmask supervision")
        for source in sorted({source_name(row) for row in sequences}):
            rows = [row for row in sequences if source_name(row) == source]
            labelled = sum(len(STORES[row.name]) for row in rows)
            visible = sum(int(row.labels.exist.sum()) for row in rows)
            print(f"{source:<14}{labelled:>7}/{visible:<7} visible frames "
                  f"({labelled / max(visible, 1):.1%})")

        for label, clips in (("train", TRAIN_CLIPS), ("val", VAL_CLIPS), ("test", TEST_CLIPS)):
            counts = {source: sum(source_name(c.sequence) == source for c in clips)
                      for source in sorted(SOURCE_WEIGHTS)}
            print(label, len(clips), counts)
        """),
        cell("markdown", r"""
        ## Stage C gerçekten gerekli mi? — eğitim öncesi video audit'i

        Stage B yalnız tek kare görür; bu nedenle statik IoU iyi olsa bile bir kez
        yanlış maskeyi belleğe yazınca takipçiyi geri getirmeyi öğrenmiş değildir.
        Yine de pahalı video eğitimini körlemesine başlatmıyoruz: her kaynaktan en
        düşük kontrastlı iki validation track'inin en zor 240 karelik bölümü önce
        yalnız Stage B ile çalıştırılır.

        Operasyonel kırmızı bayraklar: State Accuracy `<0.65`, kayıp-kare oranı
        `>10%` veya en uzun dropout `>24` kare. Bunlar evrensel benchmark eşikleri
        değil; tek-prompt uzun video kullanımında yeniden-prompt maliyetini görünür
        yapan başlangıç sınırlarıdır. Herhangi bir kaynak bu sınırı aşıyorsa Stage C
        önerilir. Kullanımınız her karede detector ile yeniden başlatıyorsa bu audit
        iyi sonuç verdiğinde Stage C opsiyonel olabilir.
        """),
        cell("code", r"""
        from src.accuracy import score_sequence
        from src.trackers import build_tracker
        from tools.eval_antiuav import track_sequence

        def precheck_segment(sequence):
            visible = sequence.labels.visible_indices()[::max(CONTRAST_AUDIT_STRIDE // 2, 1)]
            measured = [(int(index), frame_local_contrast(sequence, int(index)))
                        for index in visible]
            measured = [(index, value) for index, value in measured if np.isfinite(value)]
            hard = min(measured, key=lambda pair: pair[1])[0] if measured else int(visible[0])
            begin = max(0, hard - PRECHECK_FRAMES // 2)
            end = min(len(sequence), begin + PRECHECK_FRAMES)
            begin = max(0, end - PRECHECK_FRAMES)
            return Sequence(
                name=f"{sequence.name}__stage_c_precheck",
                split=sequence.split,
                frames=sequence.frames[begin:end],
                labels=SequenceLabels(
                    exist=sequence.labels.exist[begin:end].copy(),
                    boxes=sequence.labels.boxes[begin:end].copy()))

        def precheck_candidates():
            chosen = []
            for source in sorted({source_name(row) for row in SPLITS["val"]}):
                rows = [row for row in SPLITS["val"]
                        if source_name(row) == source
                        and row.labels.visible_indices().size]
                rows.sort(key=lambda row: (not np.isfinite(CONTRAST[row.name]),
                                           CONTRAST[row.name]))
                chosen += rows[:PRECHECK_PER_SOURCE]
            return chosen

        STOCK_CKPT = REPO / "third_party/EdgeTAM/checkpoints/edgetam.pt"
        PRECHECK_ARMS = [("stage_b", BASE_STAGE_B)]
        if PRECHECK_AGAINST_STOCK and STOCK_CKPT.is_file():
            PRECHECK_ARMS.insert(0, ("stock", STOCK_CKPT))
        elif PRECHECK_AGAINST_STOCK:
            print("!! stock checkpoint yok:", STOCK_CKPT,
                  "-- yalnız stage_b ölçülecek.")

        _segments = [precheck_segment(_row) for _row in precheck_candidates()]

        def run_precheck(label, checkpoint):
            config = {
                "model_cfg": "configs/edgetam.yaml", "checkpoint": str(checkpoint),
                "image_size": SIZE, "device": "cuda", "precision": "bfloat16",
                "mask_threshold": 0.0, "offload_video_to_cpu": False,
                "offload_state_to_cpu": False,
            }
            tracker = build_tracker("edgetam", **config)
            rows = []
            try:
                for segment in tqdm(_segments, desc=f"{label} video precheck"):
                    pred, gt = track_sequence(tracker, segment, "crop", SIZE)
                    score = score_sequence(segment.name, pred, gt)
                    lost = sum(score.dropouts.lengths)
                    name = segment.name.replace("__stage_c_precheck", "")
                    rows.append({
                        "arm": label, "name": name, "source": source_name(name),
                        "frames": score.frames,
                        "state_accuracy": score.state_accuracy,
                        "success_auc": score.success_auc,
                        "lost_frames": lost,
                        "lost_rate": lost / max(score.frames, 1),
                        "longest_dropout": max(score.dropouts.lengths, default=0),
                        "sequence_contrast": CONTRAST[name],
                    })
            finally:
                tracker.reset(); del tracker
                gc.collect(); torch.cuda.empty_cache()
            return rows

        PRECHECK_ARM_ROWS = {label: run_precheck(label, checkpoint)
                             for label, checkpoint in PRECHECK_ARMS}
        PRECHECK_ROWS = PRECHECK_ARM_ROWS["stage_b"]

        # Bir kol diğerini yenmiş mi: `state_accuracy` hedefin var/yok
        # durumunu ne kadar doğru bildiğidir, yani kaybolup geri gelmenin
        # ölçüsü; `longest_dropout` da geri gelene kadar geçen en uzun süre.
        PRECHECK_DELTAS = []
        if "stock" in PRECHECK_ARM_ROWS:
            _stock_by_name = {row["name"]: row for row in PRECHECK_ARM_ROWS["stock"]}
            for _row in PRECHECK_ROWS:
                _base = _stock_by_name.get(_row["name"])
                if _base is None:
                    continue
                PRECHECK_DELTAS.append({
                    "name": _row["name"], "source": _row["source"],
                    "state_accuracy": _row["state_accuracy"] - _base["state_accuracy"],
                    "success_auc": _row["success_auc"] - _base["success_auc"],
                    "lost_rate": _row["lost_rate"] - _base["lost_rate"],
                    "longest_dropout": _row["longest_dropout"] - _base["longest_dropout"],
                })
            print(f"\n{'source':<14}{'arm':<9}{'SA':>8}{'AUC':>8}{'lost%':>9}"
                  f"{'longest':>10}")
            for _row in PRECHECK_ROWS:
                for _label in ("stock", "stage_b"):
                    _one = (_stock_by_name[_row["name"]] if _label == "stock"
                            else _row)
                    print(f"{_one['source'] if _label == 'stock' else '':<14}"
                          f"{_label:<9}{_one['state_accuracy']:>8.3f}"
                          f"{_one['success_auc']:>8.3f}{_one['lost_rate']:>9.1%}"
                          f"{_one['longest_dropout']:>10}")
            _mean = lambda key: (sum(row[key] for row in PRECHECK_DELTAS)
                                 / max(len(PRECHECK_DELTAS), 1))
            _sa, _lost = _mean("state_accuracy"), _mean("lost_rate")
            print(f"\nstage_b - stock:  state_accuracy {_sa:+.4f}   "
                  f"lost_rate {_lost:+.1%}")
            PRECHECK_STOCK_WINS = _sa < 0 or _lost > 0
            if PRECHECK_STOCK_WINS:
                print("!! Stage B, bu klipler üzerinde stock'tan DAHA KÖTÜ takip "
                      "ediyor.\n"
                      "   Bu, 'Stage C'ye ihtiyaç var mı' sorusundan farklı bir "
                      "bulgudur: eğitim\n"
                      "   tek-kare maskeyi iyileştirirken video sürekliliğini "
                      "bozmuş demektir.\n"
                      "   Beklenen mekanizma: memory_attention/memory_encoder "
                      "donuk (finetune.py\n"
                      "   FROZEN_MODULES) ve stock encoder özelliklerine göre "
                      "eğitilmiş; encoder\n"
                      "   o dağılımdan uzaklaştıkça bellek eğitilmediği "
                      "özellikleri okuyor.")
        else:
            PRECHECK_STOCK_WINS = None

        PRECHECK_REASONS = []
        for _row in PRECHECK_ROWS:
            _flags = []
            if _row["state_accuracy"] < PRECHECK_MIN_STATE_ACCURACY:
                _flags.append("low_state_accuracy")
            if _row["lost_rate"] > PRECHECK_MAX_LOST_RATE:
                _flags.append("high_lost_rate")
            if _row["longest_dropout"] > PRECHECK_MAX_LONGEST_DROPOUT:
                _flags.append("long_dropout")
            if _flags:
                PRECHECK_REASONS.append({"name": _row["name"], "flags": _flags})
        PRECHECK_RECOMMEND_STAGE_C = bool(PRECHECK_REASONS)
        if "stock" not in PRECHECK_ARM_ROWS:
            print(f"\n{'source':<14}{'SA':>8}{'AUC':>8}{'lost%':>9}{'longest':>10}")
            for _row in PRECHECK_ROWS:
                print(f"{_row['source']:<14}{_row['state_accuracy']:>8.3f}"
                      f"{_row['success_auc']:>8.3f}{_row['lost_rate']:>9.1%}"
                      f"{_row['longest_dropout']:>10}")
        print("\nStage C recommendation:",
              "RUN — video continuity is below the gate"
              if PRECHECK_RECOMMEND_STAGE_C else
              "OPTIONAL — Stage B passed this small low-contrast audit")
        if PRECHECK_REASONS:
            print("reasons:", PRECHECK_REASONS)
        PRECHECK_REPORT = {
            "base_stage_b": str(BASE_STAGE_B),
            "arms": {label: str(path) for label, path in PRECHECK_ARMS},
            "arm_rows": PRECHECK_ARM_ROWS,
            "deltas": PRECHECK_DELTAS,
            "stock_wins": PRECHECK_STOCK_WINS,
            "thresholds": {
                "min_state_accuracy": PRECHECK_MIN_STATE_ACCURACY,
                "max_lost_rate": PRECHECK_MAX_LOST_RATE,
                "max_longest_dropout": PRECHECK_MAX_LONGEST_DROPOUT,
            },
            "recommend_stage_c": PRECHECK_RECOMMEND_STAGE_C,
            "reasons": PRECHECK_REASONS, "rows": PRECHECK_ROWS,
        }
        (MIRROR / "stage_c_precheck.json").write_text(
            json.dumps(PRECHECK_REPORT, indent=2) + "\n")
        if STOP_AFTER_PRECHECK:
            raise SystemExit(
                "STOP_AFTER_PRECHECK=True: stage_c_precheck.json yazıldı; "
                "kararı inceleyip devam etmek için False yapın.")
        """),
        cell("markdown", r"""
        ## Stage-B checkpoint'ini video belleği açık halde devam eğit

        Teacher forcing yoktur: her kare, modelin önceki kendi tahminini belleğe
        yazdığı gerçek deployment yolundan geçer. Bellek attention/encoder sabit
        kalır; mask decoder, IoU/object score ve görüntü encoder'ı düşük hızla
        uyarlanır.
        """),
        cell("code", r"""
        from sam2.build_sam import build_sam2_video_predictor
        from src.trackers._hydra_overrides import image_size_overrides
        from src.training.finetune import Rates, apply_freeze, summarise_freeze

        model = build_sam2_video_predictor(
            "configs/edgetam.yaml", str(BASE_STAGE_B), device="cuda",
            hydra_overrides_extra=image_size_overrides(SIZE))
        model.eval()
        print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M params")
        print(summarise_freeze(apply_freeze(model, "encoder"), model))

        trainable = {name for name, parameter in model.named_parameters()
                     if parameter.requires_grad}
        assert not any("memory_attention" in name or "memory_encoder" in name
                       for name in trainable), "memory coordinates must stay frozen"
        """),
        cell("markdown", r"""
        ## Zamanda yumuşak düşük-kontrast dönüşümü

        Parlaklık/kontrast kareden kareye rastgele titreştirilmez; klibin başı ile
        sonu arasında düzgün geçiş yapılır. Hedef kutusu çevre halkasının termal
        tonuna yaklaştırılır. Validation ve test bu dönüşümü görmez.
        """),
        cell("code", r"""
        from dataclasses import dataclass, replace
        import torch.nn.functional as F
        from src.training.antiuav import MEAN, STD
        from src.training.clip_loop import clip_losses
        from src.training.losses import Weights
        from src.training.schedule import CLIPS, Loop

        # `object_score` ölçülen kayboluş sayısına bakıyor, sabite değil.
        # Terim `losses.frame_loss`'ta her kareye koşulsuz uygulanıyor
        # (kaynağın gerçek `exist` taşıyıp taşımadığına bakan bir kapı yok),
        # yani hiç `exist=False` kare yokken 2.0 ağırlık "her karede hedef
        # var" diyen sabite karşı BCE demektir. `losses.py`'nin kendi modül
        # docstring'i sonucunu yazıyor: öyle bir başlık koşulsuz ateşlemeyi
        # öğrenir, `object_score_logits` negatife düşünce EdgeTAM bankaya
        # `no_obj_ptr` yazdığı ve sonraki kareler onu geri okuduğu için kendi
        # belleğini zehirler. Stage B aynı gerekçeyle terimi hiç kullanmıyor.
        OBJECT_SCORE_WEIGHT = 2.0 if TOTAL_ABSENT else 0.0
        print(f"object_score weight {OBJECT_SCORE_WEIGHT} "
              f"({TOTAL_ABSENT} absent frames measured)")
        if not TOTAL_ABSENT:
            print("   Açık hiçbir kaynak `exist=False` kare taşımıyor, o yüzden "
                  "terim kapatıldı.\n"
                  "   Bunun anlamı: bu koşu 'hedef gerçekten orada mı' "
                  "başlığına hiçbir şey öğretmiyor.\n"
                  "   Gerçek kayboluş yalnız VTUAV tracking'in ir.txt'inden "
                  "gelir (USE_VTUAV=True).")
        TRACK_WEIGHTS = Weights(focal=20.0, dice=1.0, iou=2.0,
                                object_score=OBJECT_SCORE_WEIGHT,
                                box_projection=1.0)

        def uniform(shape, limits, device, generator):
            low, high = (float(value) for value in limits)
            return low + (high - low) * torch.rand(
                shape, device=device, generator=generator)

        def temporal_pair(batch, limits, generator, device):
            shape = (batch, 1, 1, 1, 1)
            return (uniform(shape, limits, device, generator),
                    uniform(shape, limits, device, generator))

        def low_contrast_augment(batch, generator):
            images = batch.images
            device = images.device
            count, frames, _, height, width = images.shape
            mean = torch.as_tensor(MEAN, device=device).view(1, 1, 3, 1, 1)
            std = torch.as_tensor(STD, device=device).view(1, 1, 3, 1, 1)
            gray = (images * std + mean).clamp(0, 1).mean(2, keepdim=True)
            chosen = (torch.rand((count, 1, 1, 1, 1), device=device,
                                 generator=generator) < AUGMENT_PROB).float()
            phase = torch.linspace(0, 1, frames, device=device).view(1, frames, 1, 1, 1)

            c0, c1 = temporal_pair(count, GLOBAL_CONTRAST, generator, device)
            contrast = c0 + (c1 - c0) * phase
            center = gray.mean((-2, -1), keepdim=True)
            augmented = center + contrast * (gray - center)
            b0, b1 = temporal_pair(count, BRIGHTNESS_SHIFT, generator, device)
            augmented = augmented + b0 + (b1 - b0) * phase

            boxes = batch.boxes
            xx = torch.arange(width, device=device).view(1, 1, 1, width)
            yy = torch.arange(height, device=device).view(1, 1, height, 1)
            x0, y0, x1, y1 = [boxes[..., index].unsqueeze(-1).unsqueeze(-1)
                              for index in range(4)]
            valid = batch.exist.bool().unsqueeze(-1).unsqueeze(-1)
            inside = valid & (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
            pad = torch.maximum((x1 - x0).abs().clamp(min=4),
                                (y1 - y0).abs().clamp(min=4))
            outer = valid & (xx >= x0 - pad) & (xx < x1 + pad) & \
                    (yy >= y0 - pad) & (yy < y1 + pad)
            ring = (outer & ~inside).unsqueeze(2)
            ring_mean = ((gray * ring).sum((-2, -1), keepdim=True) /
                         ring.sum((-2, -1), keepdim=True).clamp(min=1))
            t0, t1 = temporal_pair(count, TARGET_CONTRAST, generator, device)
            factor = t0 + (t1 - t0) * phase
            suppressed = ring_mean + factor * (augmented - ring_mean)
            soft = F.avg_pool2d(inside.float().reshape(count * frames, 1, height, width),
                                11, stride=1, padding=5)
            soft = soft.reshape(count, frames, 1, height, width).clamp(0, 1) * chosen
            augmented = augmented * (1 - soft) + suppressed * soft

            sigma = uniform((count, 1, 1, 1, 1), SENSOR_NOISE, device, generator)
            augmented += chosen * sigma * torch.randn(
                gray.shape, device=device, generator=generator)
            blurred = F.avg_pool2d(augmented.reshape(count * frames, 1, height, width),
                                   3, stride=1, padding=1).reshape(
                                       count, frames, 1, height, width)
            blur = ((torch.rand((count, 1, 1, 1, 1), device=device,
                                generator=generator) < BLUR_PROB).float() * chosen)
            augmented = augmented * (1 - blur) + blurred * blur
            gray = (gray * (1 - chosen) + augmented * chosen).clamp(0, 1)
            return replace(batch, images=(gray.expand(-1, -1, 3, -1, -1) - mean) / std)

        @dataclass(frozen=True)
        class ContrastSplit:
            clips: list
            stores: dict
            augment: bool

        def contrast_stream(split, batch, seed, limit, device="cuda", workers=8, depth=2):
            stream = CLIPS.stream(split, batch, seed, limit, device, workers, depth)
            generator = torch.Generator(device=device)
            generator.manual_seed(SEED if seed is None else int(seed))
            for item in stream:
                yield low_contrast_augment(item, generator) if split.augment else item

        def tracking_loss(active_model, batch):
            return clip_losses(active_model, batch, weights=TRACK_WEIGHTS)

        CONTRAST_LOOP = Loop(stream=contrast_stream, loss=tracking_loss,
                             val_loss=tracking_loss)
        TRAIN_SPLIT = ContrastSplit(TRAIN_CLIPS, STORES, True)
        VAL_SPLIT = ContrastSplit(VAL_CLIPS, STORES, False)
        """),
        cell("code", r"""
        from src.training.loader import auto_batch_size

        apply_freeze(model, "encoder")
        BATCH = auto_batch_size(
            model, TRAIN_CLIPS[:BATCH_CEILING], STORES, device="cuda",
            maximum=BATCH_CEILING, reserve=0.15)
        ACCUM = max(1, math.ceil(32 / BATCH))
        print("batch", BATCH, "accum", ACCUM, "effective", BATCH * ACCUM)

        preview = next(contrast_stream(VAL_SPLIT, min(3, BATCH), SEED, 1))
        changed = low_contrast_augment(
            preview, torch.Generator(device="cuda").manual_seed(1234))
        mean = torch.as_tensor(MEAN, device="cuda").view(1, 1, 3, 1, 1)
        std = torch.as_tensor(STD, device="cuda").view(1, 1, 3, 1, 1)
        before = (preview.images * std + mean).clamp(0, 1).cpu()
        after = (changed.images * std + mean).clamp(0, 1).cpu()
        fig, axes = plt.subplots(2, before.shape[0], figsize=(4 * before.shape[0], 7))
        axes = np.asarray(axes).reshape(2, -1)
        for index in range(before.shape[0]):
            axes[0, index].imshow(before[index, 0].permute(1, 2, 0))
            axes[1, index].imshow(after[index, 0].permute(1, 2, 0))
            axes[0, index].axis("off"); axes[1, index].axis("off")
        axes[0, 0].set_ylabel("real"); axes[1, 0].set_ylabel("augmented")
        plt.tight_layout(); plt.show()
        del preview, changed, before, after
        gc.collect(); torch.cuda.empty_cache()
        """),
        cell("markdown", r"""
        ## Eğitim

        İlk aşama decoder/object-score başlığını, ikinci aşama termal görüntü
        özelliklerini düşük hızla açar. Checkpoint yalnız gerçek, augmentasyonsuz
        validation loss iyileştiğinde kaydedilir.
        """),
        cell("code", r"""
        from src.training.finetune import save_checkpoint
        from src.training.schedule import Schedule, run_stages

        schedule = Schedule(
            stages=(("head", 2, Rates(head=5e-5)),
                    ("encoder", 4, Rates(head=2e-5, neck=2e-5, trunk=5e-6))),
            batch=BATCH, accum=ACCUM, steps_per_epoch=STEPS_PER_EPOCH,
            val_batches=VAL_BATCHES, workers=LOADER_WORKERS,
            depth=PREFETCH_DEPTH, seed=SEED, patience=2,
            meta={
                "method": "aerial_temporal_contrast_finetune",
                "base": str(BASE_STAGE_B), "image_size": SIZE,
                "sources": SOURCE_WEIGHTS, "use_antiuav410": USE_ANTIUAV410,
                "use_vtuav": USE_VTUAV, "use_birdsai": USE_BIRDSAI,
                "use_vtuav_vis": USE_VTUAV_VIS,
                "absent_frames": ABSENT_FRAMES,
                "object_score_weight": OBJECT_SCORE_WEIGHT,
                "vtuav_archives": VTUAV_ARCHIVES,
                "vtuav_vis_parts": VTUAV_VIS_PARTS if USE_VTUAV_VIS else [],
                "teacher_mask_frames": teacher_mask_frames,
                "drawn_mask_frames": sum(len(store) for store in VIS_STORES.values()),
                "teacher_min_box_iou": TEACHER_MIN_BOX_IOU,
                "augment": {"prob": AUGMENT_PROB,
                            "global_contrast": GLOBAL_CONTRAST,
                            "target_contrast": TARGET_CONTRAST,
                            "brightness": BRIGHTNESS_SHIFT,
                            "noise": SENSOR_NOISE, "blur_prob": BLUR_PROB},
                "loss_weights": TRACK_WEIGHTS.__dict__,
            })
        result = run_stages(
            model, TRAIN_SPLIT, VAL_SPLIT, schedule, freeze=apply_freeze,
            save=lambda active_model, meta: save_checkpoint(active_model, CHECKPOINT, meta),
            progress=lambda stream, total, desc: tqdm(stream, total=total, desc=desc),
            loop=CONTRAST_LOOP)
        assert CHECKPOINT.is_file(), "training produced no checkpoint"
        print("best validation clip loss", result["best_val_loss"], "->", CHECKPOINT)
        del model
        gc.collect(); torch.cuda.empty_cache()
        """),
        cell("markdown", r"""
        ## Tracking A/B — kaynak bazında

        Stage B, yeni checkpoint ve yeni checkpoint+SAMURAI aynı held-out
        uçuşlarda çalışır. Seçim validation toplam State Accuracy ile yapılır;
        ayrılmış test uçuşlarına yalnız seçilen final kolu bir kez gider.
        """),
        cell("code", r"""
        from src.accuracy import score_sequence
        from src.trackers import build_tracker
        from tools.eval_antiuav import track_sequence

        SAMURAI = {"enabled": True, "kf_weight": 0.15,
                   "stable_frames": 15, "stable_iou": 0.3,
                   "memory_iou": 0.5, "memory_obj_score": 0.0,
                   "memory_kf_score": 0.0}

        def tracker_config(checkpoint, samurai=None):
            config = {
                "model_cfg": "configs/edgetam.yaml", "checkpoint": str(checkpoint),
                "image_size": SIZE, "device": "cuda", "precision": "bfloat16",
                "mask_threshold": 0.0, "offload_video_to_cpu": False,
                "offload_state_to_cpu": False,
            }
            if samurai is not None:
                config["samurai"] = samurai
            return config

        CONFIGS = {
            "stage_b": tracker_config(BASE_STAGE_B),
            "aerial_temporal": tracker_config(CHECKPOINT),
            "aerial_temporal+samurai": tracker_config(CHECKPOINT, SAMURAI),
        }

        def eval_subset(rows):
            selected = []
            for source in sorted({source_name(row) for row in rows}):
                seen_flights = set()
                for row in rows:
                    if source_name(row) != source or flight_name(row) in seen_flights:
                        continue
                    selected.append(row)
                    seen_flights.add(flight_name(row))
                    if len(seen_flights) >= EVAL_PER_SOURCE:
                        break
            return selected

        def run_arm(label, rows):
            tracker = build_tracker("edgetam", **CONFIGS[label])
            output = []
            try:
                for sequence in tqdm(rows, desc=label):
                    pred, gt = track_sequence(tracker, sequence, "crop", SIZE)
                    score = score_sequence(sequence.name, pred, gt)
                    output.append({
                        "name": sequence.name, "source": source_name(sequence),
                        "frames": score.frames, "state_accuracy": score.state_accuracy,
                        "success_auc": score.success_auc,
                        "dropout_lengths": list(score.dropouts.lengths),
                        "contrast": CONTRAST[sequence.name],
                    })
            finally:
                tracker.reset(); del tracker
                gc.collect(); torch.cuda.empty_cache()
            return output

        def weighted(rows, key):
            frames = sum(row["frames"] for row in rows)
            return sum(row[key] * row["frames"] for row in rows) / max(frames, 1)

        def summary(rows):
            return {"state_accuracy": weighted(rows, "state_accuracy"),
                    "success_auc": weighted(rows, "success_auc"),
                    "lost_frames": sum(sum(row["dropout_lengths"]) for row in rows),
                    "longest": max((max(row["dropout_lengths"], default=0)
                                    for row in rows), default=0)}

        VAL_SEQUENCES = eval_subset(SPLITS["val"])
        VAL_ROWS = {label: run_arm(label, VAL_SEQUENCES) for label in CONFIGS}
        print(f"\n{'arm':<26}{'source':<14}{'SA':>9}{'AUC':>9}{'lost':>9}{'longest':>10}")
        for label, rows in VAL_ROWS.items():
            for source in sorted({row["source"] for row in rows}):
                stats = summary([row for row in rows if row["source"] == source])
                print(f"{label:<26}{source:<14}{stats['state_accuracy']:>9.4f}"
                      f"{stats['success_auc']:>9.4f}{stats['lost_frames']:>9}"
                      f"{stats['longest']:>10}")
        SELECTED_LABEL = max(
            ("aerial_temporal", "aerial_temporal+samurai"),
            key=lambda label: summary(VAL_ROWS[label])["state_accuracy"])
        print("selected on validation:", SELECTED_LABEL)
        """),
        cell("code", r"""
        TEST_SEQUENCES = eval_subset(SPLITS["test"])
        TEST_ROWS = {
            "stage_b": run_arm("stage_b", TEST_SEQUENCES),
            SELECTED_LABEL: run_arm(SELECTED_LABEL, TEST_SEQUENCES),
        }
        print("\nheld-out test")
        for label, rows in TEST_ROWS.items():
            print(label, summary(rows))

        deploy_config = WORK / "edgetam_aerial_thermal_tracking_512.yaml"
        relative_checkpoint = "checkpoints/edgetam_aerial_thermal_tracking_512.pt"
        deploy = tracker_config(relative_checkpoint,
                                SAMURAI if SELECTED_LABEL.endswith("+samurai") else None)
        deploy_config.write_text(yaml.safe_dump(deploy, sort_keys=False))
        report_file = WORK / "aerial_tracking_log.json"
        report_file.write_text(json.dumps({
            "base_stage_b": str(BASE_STAGE_B), "checkpoint": str(CHECKPOINT),
            "stage_c_precheck": PRECHECK_REPORT,
            "selected_on_val": SELECTED_LABEL, "source_weights": SOURCE_WEIGHTS,
            "use_antiuav410": USE_ANTIUAV410, "vtuav_archives": VTUAV_ARCHIVES,
            "use_vtuav": USE_VTUAV, "use_birdsai": USE_BIRDSAI,
            "use_vtuav_vis": USE_VTUAV_VIS,
            "absent_frames": ABSENT_FRAMES,
            "vtuav_vis_parts": VTUAV_VIS_PARTS if USE_VTUAV_VIS else [],
            "teacher_mask_frames": teacher_mask_frames,
            "drawn_mask_frames": sum(len(store) for store in VIS_STORES.values()),
            "teacher_min_box_iou": TEACHER_MIN_BOX_IOU,
            "contrast": CONTRAST, "training": result,
            "validation": VAL_ROWS, "test": TEST_ROWS,
        }, indent=2) + "\n")

        shutil.copy2(CHECKPOINT, MIRROR / CHECKPOINT.name)
        shutil.copy2(deploy_config, MIRROR / deploy_config.name)
        shutil.copy2(report_file, MIRROR / report_file.name)
        print("saved to", MIRROR)
        """),
        cell("markdown", r"""
        ## VTUAV ve BIRDSAI before / after videoları

        Her kaynak için validation'da ölçülmüş dizilerden iki vaka seçilir:
        düşük-kontrast alt grubunda en iyi kazanç ve kaydedilmiş en kötü gerileme.
        Yeşil ground truth, kırmızı Stage B, camgöbeği seçilmiş final koldur. İki
        model aynı ilk kutuyu ve aynı sabit crop'u kullanır.
        """),
        cell("code", r"""
        from IPython.display import Video, display
        from src.accuracy import box_from_mask, per_frame_iou
        from src.prompts import BoxPrompt, PromptSet
        from tools.eval_antiuav import _prepare_crop

        def choose_video_cases():
            base = {row["name"]: row for row in VAL_ROWS["stage_b"]}
            final = {row["name"]: row for row in VAL_ROWS[SELECTED_LABEL]}
            cases = {}
            for source in sorted({row["source"] for row in base.values()}):
                names = [name for name, row in base.items()
                         if row["source"] == source and name in final]
                names.sort(key=lambda name: float(base[name]["contrast"]))
                low = names[:max(1, math.ceil(len(names) * 0.25))]
                delta = lambda name: (final[name]["state_accuracy"] -
                                      base[name]["state_accuracy"])
                best, worst = max(low, key=delta), min(names, key=delta)
                cases[f"{source}_low_contrast_best"] = best
                if worst != best:
                    cases[f"{source}_worst_regression"] = worst
            return cases

        def hardest_segment(sequence):
            visible = sequence.labels.visible_indices()[::5]
            measured = [(int(index), frame_local_contrast(sequence, int(index)))
                        for index in visible]
            measured = [(index, value) for index, value in measured
                        if np.isfinite(value)]
            if measured:
                hard_index, hard_value = min(measured, key=lambda pair: pair[1])
            else:
                hard_index, hard_value = int(visible[0]), float("nan")
            begin = max(0, hard_index - DEMO_PRE_ROLL)
            end = min(len(sequence), begin + DEMO_FRAMES)
            begin = max(0, end - DEMO_FRAMES)
            return Sequence(
                name=f"{sequence.name}__demo", split=sequence.split,
                frames=sequence.frames[begin:end],
                labels=SequenceLabels(
                    exist=sequence.labels.exist[begin:end].copy(),
                    boxes=sequence.labels.boxes[begin:end].copy())), begin, hard_index, hard_value

        def visual_track(config, segment, frames_dir, gt):
            start = int(segment.labels.visible_indices()[0])
            tracker = build_tracker("edgetam", **config)
            pred = np.full_like(gt, np.nan)
            masks = [None] * len(gt)
            try:
                tracker.prepare(frames_dir)
                tracker.set_prompts(PromptSet(boxes=[BoxPrompt(
                    obj_id=1, frame_idx=start,
                    xyxy=tuple(float(value) for value in gt[start]))]))
                for result_row in tracker.propagate():
                    if not 0 <= result_row.frame_idx < len(gt):
                        continue
                    mask = result_row.masks.get(1)
                    if mask is not None:
                        mask = np.asarray(mask, dtype=bool)
                        masks[result_row.frame_idx] = mask
                        pred[result_row.frame_idx] = box_from_mask(mask)
            finally:
                tracker.reset(); del tracker
                gc.collect(); torch.cuda.empty_cache()
            return pred, masks, start

        def draw_box(image, box, color):
            if np.isfinite(box).all():
                x0, y0, x1, y1 = (int(round(value)) for value in box)
                cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)

        def panel(gray, mask, pred, gt, title, color, iou, source_frame):
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if mask is not None and mask.shape == gray.shape:
                tint = np.zeros_like(image); tint[mask] = color
                image = cv2.addWeighted(image, 1.0, tint, 0.32, 0)
            draw_box(image, gt, (70, 220, 70)); draw_box(image, pred, color)
            cv2.rectangle(image, (0, 0), (SIZE, 52), (16, 16, 16), -1)
            cv2.putText(image, title, (12, 21), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"IoU {iou:.3f} | frame {source_frame}", (12, 43),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, (230, 230, 230), 1,
                        cv2.LINE_AA)
            return image

        def render_case(case, sequence):
            segment, source_begin, hard_index, hard_value = hardest_segment(sequence)
            frames_dir = WORK / "demo_frames" / case
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            frames_dir.mkdir(parents=True)
            frames_dir, gt = _prepare_crop(segment, SIZE, frames_dir)
            before_pred, before_masks, before_start = visual_track(
                CONFIGS["stage_b"], segment, frames_dir, gt)
            after_pred, after_masks, after_start = visual_track(
                CONFIGS[SELECTED_LABEL], segment, frames_dir, gt)
            start = max(before_start, after_start)
            before_iou, after_iou = per_frame_iou(before_pred, gt), per_frame_iou(after_pred, gt)

            raw = WORK / f"{case}_raw.mp4"
            output = WORK / f"{case}_before_after.mp4"
            writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"),
                                     DEMO_FPS, (2 * SIZE, SIZE))
            assert writer.isOpened()
            try:
                for index in range(start, len(gt)):
                    gray = cv2.imread(str(frames_dir / f"{index}.jpg"),
                                      cv2.IMREAD_GRAYSCALE)
                    assert gray is not None, f"missing rendered frame {index}"
                    left = panel(gray, before_masks[index], before_pred[index], gt[index],
                                 "BEFORE - Stage B", (60, 70, 235),
                                 before_iou[index], source_begin + index)
                    right = panel(gray, after_masks[index], after_pred[index], gt[index],
                                  f"AFTER - {SELECTED_LABEL}", (230, 190, 40),
                                  after_iou[index], source_begin + index)
                    writer.write(np.concatenate([left, right], axis=1))
            finally:
                writer.release()
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
                            "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                            str(output)], check=True)
            raw.unlink(missing_ok=True)
            target = MIRROR / "before_after_demo" / output.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output, target)
            return {"case": case, "sequence": sequence.name,
                    "source_range": [source_begin, source_begin + len(segment)],
                    "hard_frame": hard_index, "hard_contrast": hard_value,
                    "before_mean_iou": float(before_iou[start:].mean()),
                    "after_mean_iou": float(after_iou[start:].mean()),
                    "video": str(output), "drive_video": str(target)}

        VIDEO_RESULTS = []
        if RENDER_BEFORE_AFTER:
            by_name = {sequence.name: sequence for sequence in sequences}
            for case, name in choose_video_cases().items():
                print("rendering", case, name)
                row = render_case(case, by_name[name])
                VIDEO_RESULTS.append(row)
                print(row)
                display(Video(row["video"], embed=True, width=1024))
            video_report = MIRROR / "before_after_demo" / "video_report.json"
            video_report.write_text(json.dumps(VIDEO_RESULTS, indent=2) + "\n")
            print("videos saved to", video_report.parent)
        """),
        cell("markdown", r"""
        ## Karar

        Önce kaynak bazlı tabloya bakın. AFTER yalnız VTUAV'da yükselip
        BIRDSAI'da düşüyorsa model araç/pedestrian uçuşlarına aşırı uyumlanmıştır;
        `SOURCE_WEIGHTS` ile BIRDSAI payını artırın. İki kaynakta da State Accuracy
        artıyor, `longest` düşüyorsa bellek düzeltmesi havadan termal alana
        taşınmıştır. Test gerilerse checkpoint'i deploy etmeyin; validation'da iyi
        görünen tek bir video karar değildir.

        Bu model bir sınıf dedektörü değildir: ilk prompt/detector kutusundan sonra
        herhangi bir instance maskesini takip eden EdgeTAM'dır. HIT-UAV ve
        DroneVehicle sınıf çeşitliliği Stage B görünüşünü besler; Stage C kimliği
        sınıf adı yerine zaman içindeki aynı nesne olarak öğrenir.
        """),
    ]

    notebook = {
        "cells": cells,
        "metadata": template.get("metadata", {}),
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
