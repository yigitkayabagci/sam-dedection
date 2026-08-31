#!/usr/bin/env python3
"""Build notebook 30: an honest Stage-B versus Stage-C tracking video.

The training notebook already writes validation scores, but numbers alone are
awkward to inspect.  This notebook reuses those held-out results to choose a
low-contrast sequence, reruns both checkpoints around its hardest contrast
interval, and produces a synchronized side-by-side MP4 plus per-frame curves.
It also renders the worst regression so the demo cannot hide a failure by
showing only the nicest sequence.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks" / "29_thermal_contrast_tracking.ipynb"
OUTPUT = ROOT / "notebooks" / "30_tracking_before_after_demo.ipynb"


def source(text: str) -> list[str]:
    return dedent(text).strip("\n").splitlines(keepends=True)


def cell(kind: str, text: str) -> dict:
    row = {"cell_type": kind, "metadata": {}, "source": source(text)}
    if kind == "code":
        row.update({"execution_count": None, "outputs": []})
    return row


def main() -> None:
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    cells = [
        cell("markdown", r"""
        # 30 — Düşük kontrast tracking: gerçek before / after videosu

        Bu notebook eğitim yapmaz ve checkpoint değiştirmez. Tercihen `32`nin
        (yoksa `22`nin) Stage-B
        checkpoint'i ile `29`un temporal-kontrast checkpoint'ini aynı başlangıç
        kutusu, aynı termal kareler ve aynı crop üzerinde çalıştırır. Çıktı:

        - solda **BEFORE / Stage B**, sağda **AFTER / Stage C** olan senkron MP4;
        - her kare için ground-truth IoU ve 20-kare hareketli ortalama grafiği;
        - State Accuracy, success AUC, kayıp episode sayısı ve en uzun kayıp;
        - düşük-kontrast grubunda en çok iyileşen örneğin yanında, tüm validation
          içindeki **en kötü gerileme** videosu.

        Kaynak video, resmî
        [Anti-UAV410](https://github.com/HwangBo94/Anti-UAV410) validation
        bölümünden gelir. 410 termal dizi ve 438 binden fazla elle çizilmiş kutu
        içerdiği için yalnız göze bakmak yerine sonucu etiketle ölçebiliriz.

        > Önkoşul: Önce `29_thermal_contrast_tracking.ipynb` tamamlanmış ve
        > `/MyDrive/edgetam-stage-c/thermal_contrast_tracking` çıktıları oluşmuş
        > olmalıdır. Aynı Colab runtime'ında hemen arkasından çalıştırmak veri
        > indirmeyi tekrar etmez.
        """),
        cell("code", r"""
        # --- Runtime, repo, GPU -------------------------------------------------
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

        !bash scripts/setup_edgetam.sh 2>&1 | tail -5
        !pip install -q -r requirements.txt
        !pip install -q gdown
        !nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        """),
        cell("code", r"""
        # Bu kontratlar kırılmışsa pahalı inference'a başlamıyoruz.
        !python -m unittest tests.test_accuracy tests.test_antiuav_dataset \
            tests.test_samurai tests.test_fetch_antiuav410 2>&1 | tail -3
        """),
        cell("markdown", r"""
        ## Ayarlar ve Drive çıktıları

        Varsayılan demo, en zor kontrast karesinden 100 kare önce başlayıp en çok
        500 kare işler. Bu, uzun dönem validation skorunu yeniden hesaplamaz;
        `29`un zaten ürettiği tam-dizi skoruyla diziyi seçer, burada yalnız kritik
        bölgeyi tekrar koşup videoya çevirir.

        Belirli bir validation dizisini görmek isterseniz `FORCE_SEQUENCE` içine
        `29` raporundaki ismi yazın. `SHOW_WORST_REGRESSION=False` yaparsanız ikinci
        video atlanır.
        """),
        cell("code", r"""
        import gc, hashlib, json, math, shutil, subprocess
        import cv2
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        import yaml
        from IPython.display import Video, display

        SIZE                  = 512
        FPS                   = 20
        DEMO_FRAMES           = 500
        PRE_ROLL              = 100
        CONTRAST_SCAN_STRIDE  = 5
        FORCE_SEQUENCE        = None
        SHOW_WORST_REGRESSION = True

        DATA_DIR = Path("/content/data")
        WORK     = Path("/content/work/tracking_before_after")
        WORK.mkdir(parents=True, exist_ok=True)

        from google.colab import drive
        drive.mount("/content/drive")
        STAGE_C_DIR = Path("/content/drive/MyDrive/edgetam-stage-c/thermal_contrast_tracking")
        OUTPUT_DIR  = STAGE_C_DIR / "before_after_demo"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        BASE_CANDIDATES = [
            Path("/content/drive/MyDrive/edgetam-stage-b/aerial_thermal_stable/"
                 "edgetam_pool_aerial_thermal_stable_512.pt"),
            Path("/content/drive/MyDrive/edgetam-stage-b/thermal_deep/"
                 "edgetam_pool_thermal_deep_512.pt"),
        ]
        BASE_CHECKPOINT = next(
            (path for path in BASE_CANDIDATES if path.is_file()), BASE_CANDIDATES[0])
        AFTER_CHECKPOINT = STAGE_C_DIR / "edgetam_thermal_contrast_tracking_512.pt"
        LOG_FILE = STAGE_C_DIR / "contrast_tracking_log.json"
        for required in (BASE_CHECKPOINT, AFTER_CHECKPOINT, LOG_FILE):
            assert required.is_file(), f"Eksik: {required}. Önce notebook 29'u tamamlayın."

        def sha256(path, block=8 << 20):
            digest = hashlib.sha256()
            with Path(path).open("rb") as stream:
                for chunk in iter(lambda: stream.read(block), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        BASE_SHA, AFTER_SHA = sha256(BASE_CHECKPOINT), sha256(AFTER_CHECKPOINT)
        assert BASE_SHA != AFTER_SHA, "Before ve after checkpoint'leri aynı dosya."
        print("before", BASE_CHECKPOINT.name, BASE_SHA[:12])
        print("after ", AFTER_CHECKPOINT.name, AFTER_SHA[:12])
        """),
        cell("markdown", r"""
        ## Validation verisini hazırla

        Anti-UAV410 arşivi tek zip olarak yayımlandığı için yeni runtime ilk kez
        çalışırken yaklaşık 8.7 GiB indirir; fakat yalnız `val` üyelerini çıkarır.
        Aynı runtime'da notebook 29 verisi bulunursa bu hücre anında geçer.
        Arşivi kendi Drive'ınıza koyduysanız `DATASET_ZIP` yolunu ayarlayabilirsiniz.
        """),
        cell("code", r"""
        from tools.fetch_antiuav410 import dataset_root, describe, find_splits

        DATASET_ZIP = None  # örn. "/content/drive/MyDrive/datasets/Anti-UAV410.zip"
        splits = find_splits(DATA_DIR)
        if "val" not in splits:
            command = [sys.executable, "tools/fetch_antiuav410.py",
                       "--dest", str(DATA_DIR), "--splits", "val"]
            if DATASET_ZIP:
                command += ["--zip", str(DATASET_ZIP)]
            subprocess.run(command, check=True)

        splits = find_splits(DATA_DIR)
        assert "val" in splits, "Anti-UAV410 val split bulunamadı."
        DATA = dataset_root(splits)
        print(describe(splits))
        """),
        cell("markdown", r"""
        ## Videoyu sonuçtan seç — ama cherry-pick yapma

        `29`un tam validation JSON'ları karşılaştırılır. İlk demo, gerçek yerel
        kontrastı en düşük %25 içinden State Accuracy artışı en yüksek dizidir.
        İkinci demo ise tüm ortak diziler içindeki en kötü delta'dır. Bu seçim
        yalnız validation üzerinde yapılır; test split'ine bakılmaz.
        """),
        cell("code", r"""
        LOG = json.loads(LOG_FILE.read_text())
        AFTER_LABEL = LOG["selected_on_val"]
        assert AFTER_LABEL in ("temporal", "temporal+samurai")

        BASE_EVAL = STAGE_C_DIR / "eval_val_stage_b.json"
        AFTER_EVAL = STAGE_C_DIR / f"eval_val_{AFTER_LABEL.replace('+', '_')}.json"
        for required in (BASE_EVAL, AFTER_EVAL):
            assert required.is_file(), f"Eksik validation özeti: {required}"

        def indexed_rows(path):
            rows = json.loads(Path(path).read_text())["sequences"]
            return {row["name"]: row for row in rows}

        base_rows, after_rows = indexed_rows(BASE_EVAL), indexed_rows(AFTER_EVAL)
        contrast = LOG["val_sequence_contrast"]
        common = sorted(set(base_rows) & set(after_rows) & set(contrast))
        usable = [name for name in common if np.isfinite(float(contrast[name]))]
        assert usable, "Ortak ve kontrastı ölçülmüş validation dizisi yok."
        usable.sort(key=lambda name: float(contrast[name]))
        low_names = usable[:max(1, math.ceil(len(usable) * 0.25))]

        def validation_delta(name):
            return after_rows[name]["state_accuracy"] - base_rows[name]["state_accuracy"]

        BEST_NAME = max(low_names, key=validation_delta)
        WORST_NAME = min(common, key=validation_delta)
        if FORCE_SEQUENCE is not None:
            assert FORCE_SEQUENCE in common, f"{FORCE_SEQUENCE!r} validation özetinde yok."
            BEST_NAME = FORCE_SEQUENCE

        CASE_NAMES = {"low_contrast_best_gain": BEST_NAME}
        if SHOW_WORST_REGRESSION and WORST_NAME != BEST_NAME:
            CASE_NAMES["worst_regression"] = WORST_NAME

        print(f"selected after arm: {AFTER_LABEL}")
        print(f"low-contrast group ({len(low_names)}): {low_names}")
        print(f"{'case':<26}{'sequence':<28}{'contrast':>10}{'before SA':>12}"
              f"{'after SA':>11}{'delta':>10}")
        for case, name in CASE_NAMES.items():
            print(f"{case:<26}{name:<28}{float(contrast[name]):>10.3f}"
                  f"{base_rows[name]['state_accuracy']:>12.4f}"
                  f"{after_rows[name]['state_accuracy']:>11.4f}"
                  f"{validation_delta(name):>+10.4f}")
        """),
        cell("markdown", r"""
        ## En düşük kontrast aralığını bul ve iki modeli çalıştır

        `local contrast = |hedef ortalaması − çevre halka ortalaması| / halka std`.
        Hem sıcak-hedef hem soğuk-hedef için simetriktir. Crop penceresi bir kez
        belirlenir ve video boyunca sabit kalır; hareket eden crop ile tracking'e
        ground-truth sızdırılmaz. İki kola yalnız segmentin ilk görünür karesindeki
        aynı ground-truth kutusu prompt olarak verilir.
        """),
        cell("code", r"""
        from src.accuracy import box_from_mask, per_frame_iou, score_sequence
        from src.prompts import BoxPrompt, PromptSet
        from src.trackers import build_tracker
        from src.training.antiuav import Sequence, SequenceLabels, list_sequences
        from tools.eval_antiuav import _prepare_crop

        sequences = {s.name: s for s in list_sequences(DATA, "val")}
        missing = sorted(set(CASE_NAMES.values()) - set(sequences))
        assert not missing, f"Dataset'te seçilen diziler yok: {missing}"

        SAMURAI = {
            "enabled": True, "kf_weight": 0.15,
            "stable_frames": 15, "stable_iou": 0.3,
            "memory_iou": 0.5, "memory_obj_score": 0.0,
            "memory_kf_score": 0.0,
        }

        def tracker_config(checkpoint, samurai=False):
            config = {
                "model_cfg": "configs/edgetam.yaml",
                "checkpoint": str(checkpoint), "image_size": SIZE,
                "device": "cuda", "precision": "bfloat16",
                "mask_threshold": 0.0,
                "offload_video_to_cpu": False, "offload_state_to_cpu": False,
            }
            if samurai:
                config["samurai"] = SAMURAI
            return config

        BASE_CONFIG = tracker_config(BASE_CHECKPOINT)
        AFTER_CONFIG = tracker_config(
            AFTER_CHECKPOINT, samurai=(AFTER_LABEL == "temporal+samurai"))

        def frame_local_contrast(sequence, index):
            image = cv2.imread(str(sequence.frames[index]), cv2.IMREAD_GRAYSCALE)
            box = sequence.labels.boxes[index]
            if image is None or not np.isfinite(box).all():
                return np.nan
            x0, y0, x1, y1 = box
            height, width = image.shape
            bw, bh = max(x1 - x0, 2), max(y1 - y0, 2)
            pad = max(6, int(round(max(bw, bh))))
            ax0, ay0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
            ax1 = min(width, int(np.ceil(x1)) + pad)
            ay1 = min(height, int(np.ceil(y1)) + pad)
            tx0, ty0 = max(0, int(x0)), max(0, int(y0))
            tx1, ty1 = min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1)))
            target = image[ty0:ty1, tx0:tx1].astype(np.float32)
            patch = image[ay0:ay1, ax0:ax1].astype(np.float32)
            if target.size < 4 or patch.size <= target.size:
                return np.nan
            ring = np.ones(patch.shape, dtype=bool)
            ring[ty0 - ay0:ty1 - ay0, tx0 - ax0:tx1 - ax0] = False
            background = patch[ring]
            if background.size < 8:
                return np.nan
            return float(abs(target.mean() - background.mean()) /
                         max(background.std(), 3.0))

        def hardest_segment(sequence):
            visible = sequence.labels.visible_indices()[::max(CONTRAST_SCAN_STRIDE, 1)]
            measured = [(int(i), frame_local_contrast(sequence, int(i))) for i in visible]
            measured = [(i, value) for i, value in measured if np.isfinite(value)]
            assert measured, f"{sequence.name}: ölçülebilir görünür kare yok."
            hard_index, hard_contrast = min(measured, key=lambda pair: pair[1])
            begin = max(0, hard_index - PRE_ROLL)
            end = min(len(sequence), begin + DEMO_FRAMES)
            begin = max(0, end - DEMO_FRAMES)
            labels = SequenceLabels(
                exist=sequence.labels.exist[begin:end].copy(),
                boxes=sequence.labels.boxes[begin:end].copy())
            segment = Sequence(
                name=f"{sequence.name}_{begin}_{end}", split=sequence.split,
                frames=sequence.frames[begin:end], labels=labels)
            return segment, begin, hard_index, hard_contrast

        def run_arm(config, segment, frames_dir, gt):
            visible = segment.labels.visible_indices()
            assert visible.size, f"{segment.name}: prompt verilecek görünür kare yok."
            start = int(visible[0])
            tracker = build_tracker("edgetam", **config)
            pred = np.full_like(gt, np.nan)
            masks = [None] * len(gt)
            try:
                tracker.prepare(frames_dir)
                tracker.set_prompts(PromptSet(boxes=[BoxPrompt(
                    obj_id=1, frame_idx=start,
                    xyxy=tuple(float(v) for v in gt[start]))]))
                for result in tracker.propagate():
                    if not (0 <= result.frame_idx < len(pred)):
                        continue
                    mask = result.masks.get(1)
                    if mask is not None:
                        mask = np.asarray(mask, dtype=bool)
                        masks[result.frame_idx] = mask
                        pred[result.frame_idx] = box_from_mask(mask)
            finally:
                tracker.reset()
                del tracker
                gc.collect()
                torch.cuda.empty_cache()
            return {"pred": pred, "masks": masks, "start": start}

        def run_case(case, sequence):
            segment, source_begin, hard_index, hard_contrast = hardest_segment(sequence)
            frames_dir = WORK / "frames" / case
            # A rerun may select a different sequence or segment. EdgeTAM reads
            # every JPEG in the folder, so stale tail frames would silently
            # become part of the new video.
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            frames_dir, gt = _prepare_crop(segment, SIZE, frames_dir)
            print(f"\n{case}: {sequence.name} source [{source_begin}, "
                  f"{source_begin + len(segment)}) | min contrast "
                  f"{hard_contrast:.3f} @ frame {hard_index}")
            print("  BEFORE çalışıyor...")
            before = run_arm(BASE_CONFIG, segment, frames_dir, gt)
            print("  AFTER çalışıyor...")
            after = run_arm(AFTER_CONFIG, segment, frames_dir, gt)
            start = max(before["start"], after["start"])
            return {
                "case": case, "sequence": sequence.name, "segment": segment,
                "source_begin": source_begin, "hard_index": hard_index,
                "hard_contrast": hard_contrast, "frames_dir": frames_dir,
                "gt": gt, "before": before, "after": after, "start": start,
            }

        RUNS = {case: run_case(case, sequences[name])
                for case, name in CASE_NAMES.items()}
        """),
        cell("markdown", r"""
        ## Senkron video, IoU grafiği ve sayısal karar

        Yeşil kutu ground truth'tur. BEFORE tahmini kırmızı, AFTER tahmini camgöbeği
        gösterilir; yarı saydam alan gerçek maske tahminidir. Ground truth yalnız
        ölçüm/çizim içindir, prompt olarak segmentin ilk görünür karesi dışında
        kullanılmaz.
        """),
        cell("code", r"""
        def draw_box(image, box, color, thickness=2):
            if np.isfinite(box).all():
                x0, y0, x1, y1 = (int(round(v)) for v in box)
                cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)

        def draw_panel(gray, mask, pred_box, gt_box, title, color, iou, source_frame):
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if mask is not None and np.asarray(mask).shape == gray.shape:
                mask = np.asarray(mask, dtype=bool)
                tint = np.zeros_like(image)
                tint[mask] = color
                image = cv2.addWeighted(image, 1.0, tint, 0.32, 0)
            draw_box(image, gt_box, (70, 220, 70), 2)
            draw_box(image, pred_box, color, 2)
            cv2.rectangle(image, (0, 0), (SIZE, 54), (16, 16, 16), -1)
            cv2.putText(image, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"IoU {iou:.3f} | source frame {source_frame}",
                        (12, 45), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (230, 230, 230), 1, cv2.LINE_AA)
            return image

        def rolling_mean(values, window=20):
            values = np.asarray(values, dtype=np.float64)
            if values.size < window:
                return values.copy()
            return np.convolve(values, np.ones(window) / window, mode="same")

        def render_case(run):
            start, gt = run["start"], run["gt"]
            before, after = run["before"], run["after"]
            before_iou = per_frame_iou(before["pred"], gt)
            after_iou = per_frame_iou(after["pred"], gt)
            before_score = score_sequence(run["sequence"], before["pred"][start:], gt[start:])
            after_score = score_sequence(run["sequence"], after["pred"][start:], gt[start:])

            raw_video = WORK / f"{run['case']}_raw.mp4"
            final_video = WORK / f"{run['case']}_before_after.mp4"
            writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                     FPS, (2 * SIZE, SIZE))
            assert writer.isOpened(), f"VideoWriter açılamadı: {raw_video}"
            try:
                for index in range(start, len(gt)):
                    gray = cv2.imread(str(run["frames_dir"] / f"{index}.jpg"),
                                      cv2.IMREAD_GRAYSCALE)
                    assert gray is not None, f"Eksik demo karesi: {index}"
                    left = draw_panel(
                        gray, before["masks"][index], before["pred"][index], gt[index],
                        "BEFORE - Stage B", (60, 70, 235), before_iou[index],
                        run["source_begin"] + index)
                    right = draw_panel(
                        gray, after["masks"][index], after["pred"][index], gt[index],
                        f"AFTER - {AFTER_LABEL}", (230, 190, 40), after_iou[index],
                        run["source_begin"] + index)
                    writer.write(np.concatenate([left, right], axis=1))
            finally:
                writer.release()

            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_video),
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-pix_fmt", "yuv420p", str(final_video)], check=True)
            raw_video.unlink(missing_ok=True)

            indices = np.arange(start, len(gt)) + run["source_begin"]
            curve_file = WORK / f"{run['case']}_iou.png"
            fig, ax = plt.subplots(figsize=(11, 3.8))
            ax.plot(indices, rolling_mean(before_iou[start:]), color="#d62728",
                    lw=1.5, label="BEFORE 20-frame IoU")
            ax.plot(indices, rolling_mean(after_iou[start:]), color="#00a6a6",
                    lw=1.5, label="AFTER 20-frame IoU")
            ax.axvline(run["hard_index"], color="black", ls="--", lw=1,
                       label=f"min local contrast {run['hard_contrast']:.3f}")
            ax.set(xlabel="source frame", ylabel="IoU", ylim=(-0.02, 1.02),
                   title=f"{run['sequence']} — critical low-contrast segment")
            ax.grid(alpha=0.2); ax.legend(ncol=3, fontsize=8)
            fig.tight_layout(); fig.savefig(curve_file, dpi=150); plt.show()

            row = {
                "case": run["case"], "sequence": run["sequence"],
                "source_range": [run["source_begin"],
                                 run["source_begin"] + len(run["segment"])],
                "hard_frame": run["hard_index"],
                "hard_local_contrast": run["hard_contrast"],
                "before": {
                    "state_accuracy": before_score.state_accuracy,
                    "success_auc": before_score.success_auc,
                    "dropouts": before_score.dropouts.episodes,
                    "longest": before_score.dropouts.longest,
                },
                "after": {
                    "label": AFTER_LABEL,
                    "state_accuracy": after_score.state_accuracy,
                    "success_auc": after_score.success_auc,
                    "dropouts": after_score.dropouts.episodes,
                    "longest": after_score.dropouts.longest,
                },
                "video": str(final_video), "curve": str(curve_file),
            }
            return row

        RESULTS = [render_case(run) for run in RUNS.values()]
        print(f"\n{'case':<26}{'arm':<12}{'SA':>8}{'AUC':>9}"
              f"{'dropouts':>11}{'longest':>10}")
        for row in RESULTS:
            for arm in ("before", "after"):
                score = row[arm]
                print(f"{row['case']:<26}{arm:<12}{score['state_accuracy']:>8.4f}"
                      f"{score['success_auc']:>9.4f}{score['dropouts']:>11}"
                      f"{score['longest']:>10}")
        """),
        cell("code", r"""
        # Colab içinde oynat ve kalıcı olarak Drive'a kopyala.
        REPORT = WORK / "before_after_report.json"
        REPORT.write_text(json.dumps({
            "before_checkpoint": str(BASE_CHECKPOINT), "before_sha256": BASE_SHA,
            "after_checkpoint": str(AFTER_CHECKPOINT), "after_sha256": AFTER_SHA,
            "after_label": AFTER_LABEL, "results": RESULTS,
        }, indent=2) + "\n")

        for row in RESULTS:
            for key in ("video", "curve"):
                source_file = Path(row[key])
                shutil.copy2(source_file, OUTPUT_DIR / source_file.name)
            print(f"\n{row['case']} — {row['sequence']}")
            display(Video(row["video"], embed=True, width=1024))
        shutil.copy2(REPORT, OUTPUT_DIR / REPORT.name)
        print("\nDrive outputs:", OUTPUT_DIR)
        """),
        cell("markdown", r"""
        ## “Gerçekten çalıştı” kararı

        Tek bir güzel video yeterli kanıt değildir. Şu üç koşulu birlikte okuyun:

        1. `low_contrast_best_gain` kritik segmentinde AFTER State Accuracy/AUC
           artmalı, `longest` kayıp kısalmalıdır.
        2. `worst_regression` sonucu kabul edilebilir olmalı; büyük gerileme varsa
           deployment'tan önce o dizi ve benzerleri eğitime eklenmelidir.
        3. Asıl karar `29`un tüm validation ve tek-seferlik test tablosudur. Bu
           notebook görsel teşhis içindir; validation'dan seçilmiş bir videoyu test
           sonucu gibi sunmaz.

        AFTER yalnız maskeyi daha düzgün çiziyor ama IoU çöküşleri aynı yerde ve
        aynı uzunlukta kalıyorsa shape düzelmiş, tracking düzelmemiştir. AFTER
        düşük-kontrast çizgisinden sonra başka bir nesneye atlamıyor ve `longest`
        belirgin kısalıyorsa hedeflenen temporal düzeltme çalışmıştır.
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
