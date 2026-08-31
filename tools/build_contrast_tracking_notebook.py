#!/usr/bin/env python3
"""Build notebook 29: contrast-robust temporal continuation of notebook 22.

The existing 22 notebook is deliberately an image-only stage-B run.  This
builder reuses the well-tested Anti-UAV410 data/teacher plumbing from notebook
02, but starts from 22's checkpoint and adds the two pieces an image-only run
cannot provide: low-contrast temporal augmentation and tracking evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks" / "02_finetune_edgetam_512_thermal.ipynb"
OUTPUT = ROOT / "notebooks" / "29_thermal_contrast_tracking.ipynb"


def source(text: str) -> list[str]:
    return dedent(text).strip("\n").splitlines(keepends=True)


def cell(kind: str, text: str) -> dict:
    row = {"cell_type": kind, "metadata": {}, "source": source(text)}
    if kind == "code":
        row.update({"execution_count": None, "outputs": []})
    return row


def main() -> None:
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    old = template["cells"]

    cells = [
        cell("markdown", r"""
        # 29 — Düşük kontrastta kimlik kaymasını azaltan termal EdgeTAM

        Bu notebook tercihen `32_aerial_thermal_stage_b_stable.ipynb`, o yoksa
        korunan `22_thermal_deep_3_fixed.ipynb` çıktısını **başlangıç checkpoint'i**
        olarak alır ve onu kısa termal video klipleri üzerinde devam eğitir.
        Stage B tek kare maskesini öğrenir; ancak
        bellek bankasını hiç çalıştırmadığı için, düşük kontrastlı bir kareden
        sonra benzer görünen başka bir nesneye kaymayı ne eğitebilir ne de ölçebilir.

        Buradaki düzeltme üç parçalıdır:

        1. Gerçek veride hedef–yerel arka plan kontrastı düşük diziler daha sık
           örneklenir.
        2. Eğitim kliplerinde kontrast ve parlaklık **zamanda yumuşak** değiştirilir;
           hedef bölgesi ayrıca çevresindeki arka plan tonuna yaklaştırılır. Maske ve
           kutu geometrisi değişmez.
        3. Model kendi tahminini sonraki karelerin belleğine yazar; son değerlendirme
           aynı checkpoint'i stok bellek ve SAMURAI hareket/bellek kapısıyla ayrı ayrı
           ölçer.

        Araştırma dayanağı:

        - [SAM 2](https://arxiv.org/abs/2408.00714) video tahminini akış belleğiyle
          koşullandırır; bu yüzden tek kare skoru takip kanıtı değildir.
        - [SAMURAI](https://arxiv.org/abs/2411.11922) benzer görünümlü nesnelerde
          görünüş benzerliğinin uzamsal-zamansal tutarlılığı yenebildiğini ve kötü
          karelerin belleğe seçimsiz yazılmasının hatayı yaydığını gösterir.
        - [DAM4SAM (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Videnovic_A_Distractor-Aware_Memory_for_Visual_Object_Tracking_with_SAM2_CVPR_2025_paper.html)
          SAM2 için dikkat dağıtıcı-farkındalıklı bellek ve güvenilir güncellemenin
          gerekli olduğunu doğrudan ölçer.
        - [NLMTrack](https://arxiv.org/abs/2407.08265) termal görüntülerde düşük
          kontrast ve az dokunun, zamansal/koordinat bilgisini özellikle önemli
          yaptığını raporlar.

        **Girdi:** Drive'daki
        `edgetam-stage-b/aerial_thermal_stable/`
        `edgetam_pool_aerial_thermal_stable_512.pt`.

        **Çıktı:** `edgetam_thermal_contrast_tracking_512.pt`, val/test karşılaştırma
        JSON'ları ve SAMURAI'li dağıtım YAML'ı. Orijinal `22` notebook'u ve onun
        checkpoint'i değiştirilmez.
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

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        !nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        !df -h /content | tail -1
        """),
        old[2],
        old[3],
        cell("markdown", r"""
        ## Ayarlar

        `BASE_STAGE_B`, tercihen notebook 32'nin ürettiği checkpoint olmalıdır. Bu dosya
        bulunamazsa stok EdgeTAM'e sessizce düşülmez; aksi halde deney, kullanıcının
        iyi şekil tanıyan modelini düzeltmek yerine farklı bir modeli eğitmiş olur.

        Kontrast dönüşümü rastgele kare-kare titreşmez. Başlangıç ve bitiş değerleri
        çekilir, klip boyunca doğrusal geçiş uygulanır. Böylece termal kameranın AGC
        değişimini taklit ederken sahte temporal flicker öğretilmez.
        """),
        cell("code", r"""
        DATA_DIR = Path("/content/data")
        WORK     = Path("/content/work/thermal_contrast_tracking")
        SPLITS   = ("train", "val", "test")

        SIZE                   = 512
        CLIP_LEN, CLIP_STRIDE  = 8, 2
        TRAIN_SEQUENCES        = 80
        VAL_SEQUENCES          = 20
        TEST_SEQUENCES         = 16

        TEACHER_ID   = "facebook/sam2.1-hiera-large"
        LABEL_STRIDE = 3
        ZOOM, MIN_CROP = 4.0, 128

        STEPS_PER_EPOCH = 500
        VAL_BATCHES     = 32
        BATCH_CEILING   = 128
        LOADER_WORKERS  = min(2 * (os.cpu_count() or 4), 24)
        PREFETCH_DEPTH  = 2
        SEED            = 0

        # Gerçek düşük-kontrast dizi örneklemesi.
        CONTRAST_AUDIT_STRIDE = 40
        LOW_CONTRAST_QUANTILE = 0.40
        LOW_CONTRAST_REPEAT   = 2

        # Zamansal termal bozulma. 0.25 olasılıkla klip tamamen temiz kalır.
        AUGMENT_PROB          = 0.75
        GLOBAL_CONTRAST       = (0.35, 0.90)
        TARGET_CONTRAST       = (0.15, 0.65)
        BRIGHTNESS_SHIFT      = (-0.05, 0.05)
        SENSOR_NOISE          = (0.0, 0.018)
        BLUR_PROB             = 0.20

        # SAMURAI bellek kapısı; önce val'de stok bellekle A/B ölçülür.
        SAMURAI = {
            "enabled": True, "kf_weight": 0.15,
            "stable_frames": 15, "stable_iou": 0.3,
            "memory_iou": 0.5, "memory_obj_score": 0.0,
            "memory_kf_score": 0.0,
        }

        LABELS = WORK / "labels"
        CKPT   = REPO / "checkpoints"
        CHECKPOINT = CKPT / "edgetam_thermal_contrast_tracking_512.pt"
        for directory in (DATA_DIR, WORK, LABELS, CKPT):
            directory.mkdir(parents=True, exist_ok=True)

        import torch
        VRAM = (torch.cuda.get_device_properties(0).total_memory / 2**30
                if torch.cuda.is_available() else 0)
        TEACHER_BATCH = 8 if VRAM < 24 else 16 if VRAM < 48 else 32

        MIRROR = None
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            MIRROR = Path("/content/drive/MyDrive/edgetam-stage-c/thermal_contrast_tracking")
            MIRROR.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f"Drive bağlanamadı ({type(exc).__name__}): {exc}")

        BASE_STAGE_B_CANDIDATES = [
            Path("/content/drive/MyDrive/edgetam-stage-b/aerial_thermal_stable/"
                 "edgetam_pool_aerial_thermal_stable_512.pt"),
            Path("/content/drive/MyDrive/edgetam-stage-b/thermal_deep/"
                 "edgetam_pool_thermal_deep_512.pt"),
        ]
        BASE_STAGE_B = next(
            (path for path in BASE_STAGE_B_CANDIDATES if path.is_file()),
            BASE_STAGE_B_CANDIDATES[0])
        assert BASE_STAGE_B.is_file(), (
            f"Stage-B checkpoint bulunamadı: {BASE_STAGE_B_CANDIDATES}. "
            "Önce notebook 32'yi çalıştırın veya yolu doğru .pt dosyasına yöneltin.")

        print(f"{VRAM:.0f} GiB VRAM | teacher batch {TEACHER_BATCH}")
        print(f"base stage B: {BASE_STAGE_B}")
        print(f"output: {CHECKPOINT}")
        """),
        *old[6:18],
        cell("markdown", r"""
        ## Önce kontrastı ölç

        Ham parlaklık tek başına yeterli değildir: sıcak hedef de soğuk hedef de
        izlenebilir. Ölçülen büyüklük, hedef kutusunun ortalama yoğunluğu ile kutunun
        etrafındaki halka arasındaki mutlak farkın halka standart sapmasına oranıdır.
        Düşük değer, hedefin yerel arka plan içinde kamufle olduğunu gösterir.

        Bu ölçüm yalnız eğitim örnekleme ağırlığını ve rapor alt grubunu belirler;
        test etiketleri eğitim kararına girmez.
        """),
        cell("code", r"""
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt

        def frame_local_contrast(sequence, index):
            image = cv2.imread(str(sequence.frames[index]), cv2.IMREAD_GRAYSCALE)
            if image is None:
                return np.nan
            x0, y0, x1, y1 = sequence.labels.boxes[index]
            if not np.isfinite([x0, y0, x1, y1]).all():
                return np.nan
            h, w = image.shape
            bw, bh = max(x1 - x0, 2), max(y1 - y0, 2)
            pad = max(6, int(round(max(bw, bh))))
            ax0, ay0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
            ax1, ay1 = min(w, int(np.ceil(x1)) + pad), min(h, int(np.ceil(y1)) + pad)
            tx0, ty0 = max(0, int(x0)), max(0, int(y0))
            tx1, ty1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
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

        def sequence_contrast(sequence, stride=CONTRAST_AUDIT_STRIDE):
            visible = sequence.labels.visible_indices()[::max(int(stride), 1)]
            values = [frame_local_contrast(sequence, int(i)) for i in visible]
            values = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float32)
            return float(np.median(values)) if values.size else float("nan")

        TRAIN_CONTRAST = {s.name: sequence_contrast(s) for s in train}
        VAL_CONTRAST = {s.name: sequence_contrast(s) for s in val}
        usable = np.asarray([v for v in TRAIN_CONTRAST.values() if np.isfinite(v)])
        LOW_CONTRAST_CUT = float(np.quantile(usable, LOW_CONTRAST_QUANTILE))
        LOW_TRAIN_NAMES = {name for name, value in TRAIN_CONTRAST.items()
                           if np.isfinite(value) and value <= LOW_CONTRAST_CUT}

        print(f"train median local contrast: {np.median(usable):.3f}")
        print(f"lowest {LOW_CONTRAST_QUANTILE:.0%} cut: {LOW_CONTRAST_CUT:.3f}")
        print(f"oversampled sequences ({len(LOW_TRAIN_NAMES)}): {sorted(LOW_TRAIN_NAMES)}")

        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.hist(usable, bins=20, color="#cc6b49")
        ax.axvline(LOW_CONTRAST_CUT, color="black", ls="--", lw=1)
        ax.set_xlabel("|target mean - ring mean| / ring std")
        ax.set_ylabel("train sequences")
        ax.set_title("Gerçek termal dizilerde yerel kontrast")
        plt.tight_layout(); plt.show()
        """),
        cell("markdown", r"""
        ## Stage C: 22'nin checkpoint'inden temporal devam eğitimi

        Bellek yolu çalışır ve modelin kendi önceki tahminleri sonraki kareyi
        koşullandırır; teacher forcing yoktur. Bellek attention/encoder ağırlıkları
        yine dondurulur: amaç bellek koordinat sistemini bozmak değil, görüntü
        özelliklerini, maske/IoU başını ve nesne-güven skorunu termal kliplerde
        kalibre etmektir. Kötü belleğin yazılıp yazılmaması ayrıca SAMURAI kolunda
        ölçülür.
        """),
        cell("code", r"""
        from sam2.build_sam import build_sam2_video_predictor
        from src.trackers._hydra_overrides import image_size_overrides

        model = build_sam2_video_predictor(
            "configs/edgetam.yaml", str(BASE_STAGE_B), device="cuda",
            hydra_overrides_extra=image_size_overrides(SIZE),
        )
        model.eval()
        print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M params | "
              f"start = {BASE_STAGE_B.name}")
        """),
        old[20],
        cell("code", r"""
        # --- Video clips + real low-contrast oversampling ----------------------
        from src.training import open_masks

        def build(split, sequences, jitter):
            stores = {s.name: open_masks(LABELS / split / s.name / "pseudo_masks.npz")
                      for s in sequences}
            clips = sample_clips(sequences, length=CLIP_LEN, stride=CLIP_STRIDE,
                                 size=SIZE, frame_size=(width, height),
                                 jitter=jitter, seed=SEED)
            return clips, stores

        train_clips, train_stores = build("train", train, jitter=32)
        val_clips, val_stores = build("val", val, jitter=0)
        original_train_clips = len(train_clips)
        hard = [clip for clip in train_clips if clip.sequence.name in LOW_TRAIN_NAMES]
        train_clips = train_clips + hard * max(LOW_CONTRAST_REPEAT - 1, 0)

        labelled = sum(len(store) for store in train_stores.values())
        print(f"train {original_train_clips} -> {len(train_clips)} clips after "
              f"low-contrast x{LOW_CONTRAST_REPEAT} oversampling")
        print(f"val {len(val_clips)} clips | {labelled} teacher-labelled train frames")
        """),
        cell("markdown", r"""
        ## Zamansal düşük-kontrast dönüşümü

        Dönüşüm normalize edilmiş tensörü tekrar 0–1 yoğunluğa getirir. Global
        kontrast/parlaklık klip boyunca yumuşak değişir; hedef kutusunun içi çevre
        halkasının ortalamasına doğru çekilerek gerçek “hedef ile zemin aynı tona
        geldi” örneği oluşturulur. Hafif sensör gürültüsü ve blur eklenebilir.

        Geometriye dokunulmadığı için maskeler/kutular geçerli kalır. Validation ve
        test asla augment edilmez.
        """),
        cell("code", r"""
        from dataclasses import dataclass, replace
        import torch.nn.functional as F
        from src.training.antiuav import MEAN, STD
        from src.training.clip_loop import clip_losses
        from src.training.losses import Weights
        from src.training.schedule import CLIPS, Loop

        TRACK_WEIGHTS = Weights(focal=20.0, dice=1.0, iou=2.0,
                                object_score=2.0, box_projection=1.0)

        def uniform(shape, limits, device, generator):
            lo, hi = (float(v) for v in limits)
            return lo + (hi - lo) * torch.rand(
                shape, device=device, generator=generator)

        def temporal_pair(batch, limits, generator):
            shape = (batch, 1, 1, 1, 1)
            return (uniform(shape, limits, "cuda", generator),
                    uniform(shape, limits, "cuda", generator))

        def low_contrast_augment(batch, generator):
            images = batch.images
            device = images.device
            b, t, _, h, w = images.shape
            mean = torch.as_tensor(MEAN, device=device).view(1, 1, 3, 1, 1)
            std = torch.as_tensor(STD, device=device).view(1, 1, 3, 1, 1)
            gray = (images * std + mean).clamp(0, 1).mean(2, keepdim=True)

            chosen = (torch.rand((b, 1, 1, 1, 1), device=device,
                                 generator=generator) < AUGMENT_PROB).float()
            phase = torch.linspace(0, 1, t, device=device).view(1, t, 1, 1, 1)

            c0, c1 = temporal_pair(b, GLOBAL_CONTRAST, generator)
            contrast = c0 + (c1 - c0) * phase
            center = gray.mean((-2, -1), keepdim=True)
            augmented = center + contrast * (gray - center)

            b0, b1 = temporal_pair(b, BRIGHTNESS_SHIFT, generator)
            augmented = augmented + b0 + (b1 - b0) * phase

            # Hedefi çevresindeki termal tona yaklaştır. Kutunun yumuşatılmış
            # maskesi dikdörtgen kenar artefaktı oluşmasını engeller.
            boxes = batch.boxes
            xx = torch.arange(w, device=device).view(1, 1, 1, w)
            yy = torch.arange(h, device=device).view(1, 1, h, 1)
            x0, y0, x1, y1 = [boxes[..., i].unsqueeze(-1).unsqueeze(-1)
                              for i in range(4)]
            valid = batch.exist.bool().unsqueeze(-1).unsqueeze(-1)
            inside = valid & (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
            bw = (x1 - x0).abs().clamp(min=4)
            bh = (y1 - y0).abs().clamp(min=4)
            pad = torch.maximum(bw, bh)
            outer = valid & (xx >= x0 - pad) & (xx < x1 + pad) & \
                    (yy >= y0 - pad) & (yy < y1 + pad)
            ring = outer & ~inside
            ring5 = ring.unsqueeze(2)
            ring_mean = ((gray * ring5).sum((-2, -1), keepdim=True) /
                         ring5.sum((-2, -1), keepdim=True).clamp(min=1))

            tc0, tc1 = temporal_pair(b, TARGET_CONTRAST, generator)
            target_factor = tc0 + (tc1 - tc0) * phase
            suppressed = ring_mean + target_factor * (augmented - ring_mean)
            soft = F.avg_pool2d(inside.float().reshape(b * t, 1, h, w),
                                kernel_size=11, stride=1, padding=5)
            soft = soft.reshape(b, t, 1, h, w).clamp(0, 1) * chosen
            augmented = augmented * (1 - soft) + suppressed * soft

            sigma = uniform((b, 1, 1, 1, 1), SENSOR_NOISE, device, generator)
            noise = torch.randn(gray.shape, device=device, generator=generator)
            augmented = augmented + chosen * sigma * noise

            blurred = F.avg_pool2d(augmented.reshape(b * t, 1, h, w),
                                   kernel_size=3, stride=1, padding=1)
            blurred = blurred.reshape(b, t, 1, h, w)
            blur = ((torch.rand((b, 1, 1, 1, 1), device=device,
                                generator=generator) < BLUR_PROB).float() * chosen)
            augmented = augmented * (1 - blur) + blurred * blur

            gray = (gray * (1 - chosen) + augmented * chosen).clamp(0, 1)
            rgb = gray.expand(-1, -1, 3, -1, -1)
            return replace(batch, images=(rgb - mean) / std)

        @dataclass(frozen=True)
        class ContrastSplit:
            clips: list
            stores: dict
            augment: bool

        def contrast_stream(split, batch, seed, limit, device="cuda",
                            workers=8, depth=2):
            stream = CLIPS.stream(split, batch, seed, limit, device, workers, depth)
            generator = torch.Generator(device=device)
            generator.manual_seed(SEED if seed is None else int(seed))
            for item in stream:
                yield low_contrast_augment(item, generator) if split.augment else item

        def tracking_loss(model, batch):
            return clip_losses(model, batch, weights=TRACK_WEIGHTS)

        CONTRAST_LOOP = Loop(stream=contrast_stream, loss=tracking_loss,
                             val_loss=tracking_loss)
        TRAIN_SPLIT = ContrastSplit(train_clips, train_stores, True)
        VAL_SPLIT = ContrastSplit(val_clips, val_stores, False)

        # Görsel smoke test: solda gerçek, sağda aynı klibin augment edilmiş hali.
        preview = next(CLIPS.stream(VAL_SPLIT, 3, SEED, 1, "cuda", 2, 1))
        generator = torch.Generator(device="cuda").manual_seed(1234)
        changed = low_contrast_augment(preview, generator)
        mean = torch.as_tensor(MEAN, device="cuda").view(1, 1, 3, 1, 1)
        std = torch.as_tensor(STD, device="cuda").view(1, 1, 3, 1, 1)
        before = (preview.images * std + mean).clamp(0, 1).cpu()
        after = (changed.images * std + mean).clamp(0, 1).cpu()
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        for k in range(3):
            axes[0, k].imshow(before[k, 0].permute(1, 2, 0)); axes[0, k].axis("off")
            axes[1, k].imshow(after[k, 0].permute(1, 2, 0)); axes[1, k].axis("off")
        axes[0, 0].set_ylabel("real"); axes[1, 0].set_ylabel("augmented")
        plt.tight_layout(); plt.show()
        del preview, changed, before, after
        torch.cuda.empty_cache()
        """),
        old[22],
        old[23],
        old[24],
        old[25],
        cell("code", r"""
        # --- Loader + augmentation throughput ----------------------------------
        import time

        stream = contrast_stream(TRAIN_SPLIT, BATCH, SEED, 6, "cuda",
                                 LOADER_WORKERS, PREFETCH_DEPTH)
        held = next(stream)
        t0 = time.time()
        for batch in stream:
            del batch
        load = (time.time() - t0) / 5

        t0 = time.time()
        for _ in range(4):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tracking_loss(model, held)[0].backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step = (time.time() - t0) / 4

        print(f"load+augment {load * 1000:6.0f} ms/batch")
        print(f"compute      {step * 1000:6.0f} ms/batch")
        print("GPU-bound" if step > load else "input-bound: raise LOADER_WORKERS")
        del held; gc.collect(); torch.cuda.empty_cache()
        """),
        cell("markdown", r"""
        ## Eğitim

        İlk aşama başlık/IoU/object-score kalibrasyonunu yapar. İkinci aşama
        encoder'ı daha düşük hızla açar. `iou` ve `object_score` ağırlıkları 2'dir;
        çünkü bunlar yalnız rapor sayıları değil, aday maske seçimi ve bellek kapısının
        güven sinyalleridir. Validation gerçek görüntülerde, augmentasyonsuz yapılır.
        """),
        cell("code", r"""
        from src.training.finetune import save_checkpoint
        from src.training.schedule import Schedule, run_stages

        schedule = Schedule(
            stages=(("head", 2, Rates(head=1e-4)),
                    ("encoder", 3, Rates(head=5e-5, neck=5e-5, trunk=1e-5))),
            batch=BATCH, accum=ACCUM, steps_per_epoch=STEPS_PER_EPOCH,
            val_batches=VAL_BATCHES, workers=LOADER_WORKERS,
            depth=PREFETCH_DEPTH, seed=SEED, patience=2,
            meta={
                "method": "temporal_contrast_finetune", "image_size": SIZE,
                "dataset": "Anti-UAV410", "base": str(BASE_STAGE_B),
                "teacher": TEACHER_ID, "label_stride": LABEL_STRIDE,
                "augment": {
                    "prob": AUGMENT_PROB,
                    "global_contrast": GLOBAL_CONTRAST,
                    "target_contrast": TARGET_CONTRAST,
                    "brightness": BRIGHTNESS_SHIFT,
                    "noise": SENSOR_NOISE,
                    "blur_prob": BLUR_PROB,
                },
                "loss_weights": TRACK_WEIGHTS.__dict__,
            })

        result = run_stages(
            model, TRAIN_SPLIT, VAL_SPLIT, schedule,
            freeze=apply_freeze,
            save=lambda m, meta: save_checkpoint(m, CHECKPOINT, meta),
            progress=lambda stream, total, desc: tqdm(stream, total=total, desc=desc),
            loop=CONTRAST_LOOP,
        )
        assert CHECKPOINT.is_file(), "training produced no checkpoint"
        print(f"best val clip loss {result['best_val_loss']:.4f} -> {CHECKPOINT}")
        """),
        old[29],
        cell("markdown", r"""
        ## Takip değerlendirmesi: üç kol

        Tek-kare IoU yerine deployment yolunun tamamı çalıştırılır:

        - **stage B:** `22` checkpoint'i, stok FIFO bellek;
        - **temporal:** yeni checkpoint, stok FIFO bellek;
        - **temporal + samurai:** aynı yeni checkpoint, hareket-duyarlı aday seçimi
          ve yalnız güvenilir kareleri kabul eden bellek kapısı.

        Böylece ağırlık eğitiminin ve bellek politikasının katkıları birbirine
        karıştırılmaz. Rapor tüm validation yanında gerçek yerel kontrastı en düşük
        dörtte birlik grubu ayrıca gösterir.
        """),
        cell("code", r"""
        import subprocess, yaml

        def tracker_config(checkpoint, samurai=None):
            cfg = {
                "model_cfg": "configs/edgetam.yaml",
                "checkpoint": str(checkpoint),
                "image_size": SIZE, "device": "cuda", "precision": "bfloat16",
                "mask_threshold": 0.0,
                "offload_video_to_cpu": False, "offload_state_to_cpu": False,
            }
            if samurai is not None:
                cfg["samurai"] = samurai
            return cfg

        EVAL_CONFIGS = {
            "stage_b": WORK / "eval_stage_b.yaml",
            "temporal": WORK / "eval_temporal.yaml",
            "temporal+samurai": WORK / "eval_temporal_samurai.yaml",
        }
        EVAL_CONFIGS["stage_b"].write_text(yaml.safe_dump(
            tracker_config(BASE_STAGE_B), sort_keys=False))
        EVAL_CONFIGS["temporal"].write_text(yaml.safe_dump(
            tracker_config(CHECKPOINT), sort_keys=False))
        EVAL_CONFIGS["temporal+samurai"].write_text(yaml.safe_dump(
            tracker_config(CHECKPOINT, SAMURAI), sort_keys=False))

        def run_eval(label, split, limit):
            output = WORK / f"eval_{split}_{label.replace('+', '_')}.json"
            command = [sys.executable, "tools/eval_antiuav.py",
                       "--data", str(DATA), "--split", split,
                       "--tracker", "edgetam", "--config", str(EVAL_CONFIGS[label]),
                       "--mode", "crop", "--size", str(SIZE),
                       "--json", str(output)]
            if limit is not None:
                command += ["--limit", str(limit)]
            subprocess.run(command, check=True)
            return output

        VAL_FILES = {label: run_eval(label, "val", VAL_SEQUENCES)
                     for label in EVAL_CONFIGS}
        """),
        cell("code", r"""
        def load_rows(path):
            return json.loads(Path(path).read_text())["sequences"]

        def weighted(rows, key):
            frames = sum(row["frames"] for row in rows)
            return sum(row[key] * row["frames"] for row in rows) / max(frames, 1)

        def summary(rows):
            return {
                "state_accuracy": weighted(rows, "state_accuracy"),
                "success_auc": weighted(rows, "success_auc"),
                "lost_frames": sum(sum(row["dropout_lengths"]) for row in rows),
                "episodes": sum(len(row["dropout_lengths"]) for row in rows),
                "longest": max((max(row["dropout_lengths"], default=0)
                                for row in rows), default=0),
            }

        def print_group(title, rows_by_label, contrast):
            names = [name for name, value in contrast.items() if np.isfinite(value)]
            names.sort(key=lambda name: contrast[name])
            low = set(names[:max(1, int(np.ceil(len(names) * 0.25)))])
            print(f"\n{title}")
            print(f"{'arm':<20}{'group':<10}{'state acc':>11}{'AUC':>9}"
                  f"{'lost':>9}{'episodes':>10}{'longest':>9}")
            for label, rows in rows_by_label.items():
                for group, selected in (("all", rows),
                                        ("low-25%", [r for r in rows if r['name'] in low])):
                    s = summary(selected)
                    print(f"{label:<20}{group:<10}{s['state_accuracy']:>11.4f}"
                          f"{s['success_auc']:>9.4f}{s['lost_frames']:>9}"
                          f"{s['episodes']:>10}{s['longest']:>9}")
            print("low-contrast sequences:", sorted(low))

        VAL_ROWS = {label: load_rows(path) for label, path in VAL_FILES.items()}
        print_group("validation", VAL_ROWS, VAL_CONTRAST)

        # Teste yalnız val'de seçilen temporal bellek politikası gider.
        SELECTED_LABEL = max(("temporal", "temporal+samurai"),
                             key=lambda label: summary(VAL_ROWS[label])["state_accuracy"])
        print("\nselected on val:", SELECTED_LABEL)
        """),
        cell("markdown", r"""
        ### Test: bir kez

        Test, augmentation ayarı veya SAMURAI seçimi yapmak için kullanılmaz. Seçim
        validation state accuracy ile yukarıda yapılmıştır. Burada yalnız stage-B
        başlangıcı ve seçilmiş final kolu karşılaştırılır.
        """),
        cell("code", r"""
        assert "test" in splits, "test split indirilmedi"
        test_sequences = list_sequences(DATA, "test")[:TEST_SEQUENCES]
        TEST_CONTRAST = {s.name: sequence_contrast(s) for s in test_sequences}
        TEST_FILES = {
            "stage_b": run_eval("stage_b", "test", TEST_SEQUENCES),
            SELECTED_LABEL: run_eval(SELECTED_LABEL, "test", TEST_SEQUENCES),
        }
        TEST_ROWS = {label: load_rows(path) for label, path in TEST_FILES.items()}
        print_group("test (held out)", TEST_ROWS, TEST_CONTRAST)
        """),
        cell("code", r"""
        # --- Kalıcı çıktılar ----------------------------------------------------
        import shutil

        deploy = tracker_config(
            "checkpoints/edgetam_thermal_contrast_tracking_512.pt",
            SAMURAI if SELECTED_LABEL == "temporal+samurai" else None)
        DEPLOY_CONFIG = WORK / "edgetam_thermal_contrast_tracking_512.yaml"
        DEPLOY_CONFIG.write_text(yaml.safe_dump(deploy, sort_keys=False))

        summary_file = WORK / "contrast_tracking_log.json"
        summary_file.write_text(json.dumps({
            "base_stage_b": str(BASE_STAGE_B), "checkpoint": str(CHECKPOINT),
            "selected_on_val": SELECTED_LABEL,
            "low_contrast_cut": LOW_CONTRAST_CUT,
            "low_train_sequences": sorted(LOW_TRAIN_NAMES),
            "train_sequence_contrast": TRAIN_CONTRAST,
            "val_sequence_contrast": VAL_CONTRAST,
            "test_sequence_contrast": TEST_CONTRAST,
            "training": result,
            "validation": {k: summary(v) for k, v in VAL_ROWS.items()},
            "test": {k: summary(v) for k, v in TEST_ROWS.items()},
        }, indent=2) + "\n")

        if MIRROR is not None:
            shutil.copy(CHECKPOINT, MIRROR / CHECKPOINT.name)
            shutil.copy(DEPLOY_CONFIG, MIRROR / DEPLOY_CONFIG.name)
            shutil.copy(summary_file, MIRROR / summary_file.name)
            shutil.copy(WORK / "manifest.json", MIRROR / "manifest.json")
            for path in sorted(WORK.glob("eval_*.json")):
                shutil.copy(path, MIRROR / path.name)
            for split in ("train", "val"):
                shutil.copytree(LABELS / split, MIRROR / "labels" / split,
                                dirs_exist_ok=True)
            print("saved to", MIRROR)
        else:
            print("download before runtime ends:", CHECKPOINT, DEPLOY_CONFIG,
                  summary_file)
        """),
        cell("markdown", r"""
        ## Sonucu nasıl okuyacaksınız?

        En önemli satır `low-25%` grubudur.

        - `temporal`, stage B'den iyi ve kayıp episode'ları daha kısa ise düşük
          kontrast eğitimi işe yaramıştır.
        - `temporal+samurai` ayrıca iyiyse sorun yalnız temsil değil, kötü karenin
          belleğe yazılmasıdır; SAMURAI YAML'ı ile deploy edin.
        - Genel skor artıp `low-25%` artmıyorsa kontrast dönüşümü veri dağılımını
          yakalamıyordur. Önce görsel smoke test'i ve gerçek kamera histogramlarını
          karşılaştırın; `GLOBAL_CONTRAST` değerini körlemesine daha da düşürmeyin.
        - AUC iyi ama `lost/longest` yüksek kalıyorsa maske şekli düzelmiş, kimlik
          sürekliliği düzelmemiştir. Bu durumda daha fazla **video** ve benzer hedefli
          dizi gerekir; statik havuz eklemek tek başına doğru ekseni büyütmez.

        Final checkpoint mimariyi değiştirmez. TensorRT export edilecekse yeni
        checkpoint'ten yeniden export/kalibrasyon yapın; eski engine'lerle yeni
        checkpoint'i karıştırmayın.
        """),
    ]

    document = {
        "cells": cells,
        "metadata": template.get("metadata", {}),
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    OUTPUT.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)} with {len(cells)} cells")


if __name__ == "__main__":
    main()
