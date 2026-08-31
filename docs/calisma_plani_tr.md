# Çalışma planı: ne konuştuk, ne değişti, hangi notebook, nasıl test edilir

Bu dosya `docs/son_degisiklikler_tr.md`'nin operasyonel karşılığıdır. O
*ne yapıldığını* anlatır; bu *ne çalıştıracağını ve nasıl doğrulayacağını*.

---

## 0. Konuştuklarımızın özeti

Bu oturum boyunca sırayla şunlar konuşuldu ve hepsi koda döküldü:

| konu | senin söylediğin | sonuç |
|---|---|---|
| HIT-UAV `no_image` | "hituav'lara no image diyor, bir fix cell verir misin" | kök neden bulundu (arşiv her kareyi 4 kez taşıyor), `resolve_images_root` eklendi |
| SegFly | "segfly güvenilir, hepsini kullanalım" | `CLASS_WEIGHTS`'ten çıkarıldı; 15 007 / 5 378 farkının hata olmadığı kanıtlandı |
| büyük kutular | "ekranın %50'si gibi kutuları çıkaralım" | `MAX_AREA = 0.25` |
| ağırlıklar | "vtuav ve dronevehicle 0.8'e düşsün" | `THERMAL_WEIGHTS` güncellendi |
| `vtuav_lt` | "varsa kaldıralım" | `SKIP_POOLS`'a eklendi |
| hedef boyutu | "tiny ve huge değil, small/medium" | `MIN_AREA 64`, `MIN_SIDE 6`, `MAX_AREA 0.25` |
| **asıl problem** | "zıt kontrastta iyi, benzer kontrastta bırakıyor" | SCR ölçüldü, `photometric.py` + kontrast bandı raporu |
| boş kare eğitimi | "10 000 kareye 'burada bir şey yok' desek?" | `exist` etiketi VTUAV/RGBT234/LasHeR'den, crop penceresi yokluklarından bedava negatif |
| atlama / büyüme | "kutu araziye yayılıyor, atlıyor" | `stabiliser.py` — makullük kapıları + histerezis |
| yeniden arama | "kaybettiği yerde tekrar arayabilir miyiz" | NCC ile büyüyen arama penceresi |
| kamera kayması | "kamera trackle beraber kayıyor" | ego-hareket, Kalman'a **kontrol girdisi** olarak |
| klasik SOTA | "code based klasik yöntemlerle çözelim" | `stabiliser.py`, gerçek görüntüyle sınandı |
| Anti-UAV410 | "buna bu kadar takılmanı anlamıyorum, hedefim AERIAL" | 31'de varsayılan **kapalı**; `exist` kaynağı VTUAV/RGBT234/LasHeR'e taşındı |
| SAMURAI 768 | "ONNX'te sorun yok değil mi, verimli uygula, orijinalin üstüne çık" | doğrulandı + `ego_motion` eklendi + tek senkronizasyon |
| pretrain | "drive'a eriş, pretrain notebook'u hazırla, 1 tane de RGB" | **34** ve **35** |
| AeroVIS | "images ve mask yolunu MCP ile test et" | arşivin central directory'si okundu, `image_rel` düzeltildi, 8 test |

---

## 1. Yapılan değişikliklerin listesi

### Yeni dosyalar

| dosya | ne yapar |
|---|---|
| `src/training/photometric.py` | kontrast çökertme + polarite + gamma + gürültü |
| `src/trackers/stabiliser.py` | ego-hareket, makullük kapıları, durum makinesi, NCC yeniden yakalama (~700 satır) |
| `configs/edgetam_768_samurai.yaml` | 768 + SAMURAI + ego-hareket |
| `tests/test_photometric.py` | 11 test |
| `tests/test_stabiliser.py` | 27 test + 15 subtest |
| `notebooks/34_pretrain_thermal_aerial.ipynb` | termal + hava pretrain |
| `notebooks/35_pretrain_rgb_aerial.ipynb` | RGB pretrain |
| `docs/son_degisiklikler_tr.md` | değişiklik raporu |
| bu dosya | çalışma planı |

### Değişen dosyalar

| dosya | ne değişti |
|---|---|
| `src/training/pool_reader.py` | `resolve_images_root` (+ `image_rel` desteği), `_root_of_archive_paths` |
| `src/training/aerial.py` | `gates_tag` — önbellek anahtarına kapı ayarları |
| `src/training/image_loop.py` | `instance_contrast`, augmentasyonun collate'e bağlanması |
| `src/training/schedule.py` | `val_stream` — doğrulama augment edilmemiş akışı kullanır |
| `src/trackers/samurai.py` | ego-hareket kontrol girdisi, tek `.cpu()` senkronizasyonu |
| `src/trackers/edgetam_tracker.py` | `ego_motion` + `guard`, nesne başına verdict |
| `src/trackers/_hydra_overrides.py` | `check_image_size()` — 640/896 config okunurken reddedilir |
| `tools/train_encoder.py` | 5 yeni augmentasyon bayrağı |
| `tools/eval_instances.py` | kontrast bandına göre IoU raporu |
| `tools/build_stage_b_notebooks.py` | presetler, ağırlıklar, denetim bloğu, 34/35, referans checkpoint yolu |

---

## 2. Hangi notebook'u ne zaman çalıştıracaksın

### Aşama 0 — hazırlık (bir kez)

**0a. Havuzlar.** Drive'da hazır: `MyDrive/edgetam-pool/` altında.
AeroVIS havuzu `aerotrack/` alt klasöründe (`aerovis_train.zip` 372 MB,
`aerovis_heldout.zip` 77 MB). Hiçbir şey yapmana gerek yok.

**0b. AeroVIS kareleri.** Artık elle bir şey yapman gerekmiyor. Sürüm
arşivi Drive'da: `MyDrive/edgetam-pool/AeroVIS.zip` (12.63 GiB — dosyanın
kendisi ya da paylaşılan kopyaya bir kısayol, mount ikisini de çözer).
`POOL_ARCHIVES` bunu 20, 27, 28, 35 ve 21'de adıyla gösteriyor; havuzun
kayıtları hangi kareyi istediğini tek tek söylediği için arşivden yalnızca o
üyeler çıkarılır ve `/content/data/AeroVIS/AeroVIS/sequences/` altına iner —
`resolve_images_root` da tam orayı adlandırır.

Eskiden burada duran `gdown` + `unzip` adımı **artık gereksiz**: her oturumda
13.5 GiB'ı ağdan indirmek yerine kareler Drive'dan geliyor. Diskte tepe
kullanım yine yüksek olabilir, `21`'i çalıştırmadan önce `!df -h /content`.

Kareler bir kez diskteyse 20/27/28/35 hiçbir şeyi yeniden çıkarmaz
(`already there`), yani `21`'i bir kez çalıştırmak eğitim koşularını hazırlar.

**0c. `21_pool_data_readiness.ipynb`** — eğitim yapmaz. Her havuzun ne
kaydettiğini, kaç karesinin diskte olduğunu, geri kalanının neden olmadığını
ve bir koşunun alacağı tam `--pool` bayraklarını yazdırır. **Bir eğitim
başlatmadan önce buradan geç.**

### Aşama 1 — temel (pretrain)

| sıra | notebook | süre | çıktı |
|---|---|---|---|
| 1 | `34_pretrain_thermal_aerial.ipynb` | uzun (`EPOCHS [3, 40]`, `STEPS 1000`) | `edgetam-stage-b/pretrain_thermal_aerial_finetune/edgetam_pool_pretrain_thermal_aerial_512.pt` |
| 1′ | `35_pretrain_rgb_aerial.ipynb` | uzun | `edgetam-stage-b/pretrain_rgb_aerial_finetune/edgetam_pool_pretrain_rgb_aerial_512.pt` |

İkisi **paralel çalışabilir** (ayrı runtime) — birbirine bağlı değiller.
34 asıl hedef (hava + termal), 35 yanındaki ikinci RGB modeli.

**Not:** 34/35 şu an sıfırdan (`REFERENCE_CHECKPOINT = ""`) başlar. Zincirlemek
istersen sonraki aşamanın settings hücresinde `REFERENCE_CHECKPOINT`'i 34'ün
çıktısına çevir.

### Aşama 2 — Stage B (dar ve derin)

| sıra | notebook | ön koşul | neden |
|---|---|---|---|
| 2 | `32_aerial_thermal_stage_b_stable.ipynb` | — | 22'nin NaN'a giden LR profilini üç güvenli profille eleyip kararlı Stage B üretir. **22 yerine bunu tercih et.** |
| 2-alt | `22_thermal_deep.ipynb` | — | 32'yi çalıştıramıyorsan klasik kol |
| 3 | `27_thermal_deep_rgb_aerovis.ipynb` | 22'nin çıktısı + AeroVIS kareleri | 22 + RGB havuzları + AeroVIS; A/B referansı 22 |
| 3-alt | `28_rgb_deep_aerovis.ipynb` | AeroVIS kareleri | temiz RGB kontrolü |
| — | `23_thermal_deep_lora.ipynb` | — | 22 ile tek fark `METHOD = "lora"`; yöntem A/B'si isteğe bağlı |

`19` ve `20` kısa doğrulama kollarıdır (`EPOCHS [1, 3]`, `STEPS 400`). Bir
gecelik koşuya girmeden önce boru hattının uçtan uca çalıştığını görmek için
**iyi bir ilk adım** — özellikle AeroVIS düzeltmesini gerçek veride görmek için.

### Aşama 3 — takip (Stage C)

| sıra | notebook | ön koşul |
|---|---|---|
| 4 | `29_thermal_contrast_tracking.ipynb` | tercihen 32'nin checkpoint'i (yoksa 22) |
| 5 | `30_tracking_before_after_demo.ipynb` | **29 tamamlanmış olmalı**; aynı runtime'da hemen arkasından çalıştır |
| 6 | `31_aerial_thermal_tracking.ipynb` | 32'nin checkpoint'i + `MyDrive/VTUAV` arşivleri + VTUAV-VIS `train_001` |

**Stage C ne kadar kritik?** Stage B en kritik evredir — çünkü tek kare
maskesini o öğretir ve her şey onun üstüne biner. Ama senin bildirdiğin hata
(benzer kontrastta hedefi bırakma) **tek karede görünmez**: bellek bankası
devreye girmeden ortaya çıkmaz. Yani Stage B'yi düzeltmek gerekli ama yeterli
değil; 29/31 o hatanın ölçülebildiği tek yer. 30 ise eğitim yapmaz — sadece
gösterir.

### Önerilen minimum yol

```
21  →  19  →  32  →  29  →  30
```

AeroVIS/RGB de istiyorsan buna `35` (paralel) ve `27` ekle.

---

## 3. Nasıl test edilir

### 3a. Hemen, GPU'suz, şimdi

```bash
python3 -m pytest tests/ -q
```
→ `1009 passed, 9 skipped, 217 subtests`.

Tek tek:

| komut | ne kanıtlar |
|---|---|
| `pytest tests/test_pool_reader.py -q` | 80 test — AeroVIS yol çözümü dahil |
| `pytest tests/test_pool_reader.py -k AeroVIS -v` | 8 testin adı okununca ne kanıtlandığı görülür |
| `pytest tests/test_photometric.py -q` | 11 test — kontrast metriği ve augmentasyon |
| `pytest tests/test_stabiliser.py -q` | 27 test — ego-hareket, kapılar, yeniden yakalama |
| `pytest tests/test_samurai.py -q` | 52 test — Kalman, bellek kapısı, ego-hareket bağlantısı |
| `pytest tests/test_hydra_overrides.py -q` | 24 test — 640/896 reddi, 768 kabulü |

### 3b. AeroVIS yol düzeltmesini gerçek veride görmek

Notebook 20, 27, 28 veya 35'in **cell 1** çıktısında şu satırı ara:

```
aerovis_train: frames are under /content/data/AeroVIS/AeroVIS/sequences,
not /content/data/AeroVIS -- re-rooted.
```

ve hemen ardından hazırlık tablosunda:

```
pool                          records  probed  found   images root
aerovis_train                   39943     200    200   .../AeroVIS/sequences
   200 of them by joining the path the archive itself recorded, ...
```

`found` 200/200 **ve** "by joining" satırı varsa düzeltme çalışıyor.
`found` düşükse ya kareler eksik ya da kök yanlış — tablo hangisi olduğunu
söyler.

### 3c. Kontrast çalışmasını görmek

Her koşunun sonunda `eval_instances` **kontrast bandı tablosu** yazdırır:

```
contrast      instances    IoU
< 1.0             ....    ....
1.0 - 3.0         ....    ....
> 3.0             ....    ....
```

**Bakılacak sayı ortalama IoU değil, `< 1.0` satırı.** Senin problemin orada.
Ortalama IoU yükselirken `< 1.0` satırı düşüyorsa koşu yanlış yöne gitmiştir.

Elle de çalıştırabilirsin:

```bash
python tools/eval_instances.py \
  --pool /content/pool/_by_name/aerovis_heldout:/content/data/AeroVIS/AeroVIS/sequences:rgb \
  --checkpoint <ckpt> --split test --json outputs/eval.json
```

Ölçülmüş gerçek: HIT-UAV'da medyan SCR **0.91**, %53'ü 1'in altında. Yani o
veri kümesinde augmentasyon neredeyse etkisiz — kırılacak kestirme zaten yok.
Bunu tablo söyleyecek, varsayma.

### 3d. Takip düzeltmelerini görmek

**Sentetik / hız:**
```bash
python tools/benchmark_tracking.py --frames 500 --config configs/edgetam_768_samurai.yaml
```
SAMURAI + ego-hareketin FPS'e maliyetini ölçer. Beklenen: kare başına bir
senkronizasyon ve bir seyrek optik akış — encoder'ın yanında ~1 ms.

**Gerçek videoda kalite:** `30_tracking_before_after_demo.ipynb`. Eğitim
yapmaz, checkpoint değiştirmez. Ürettikleri:
- solda BEFORE / Stage B, sağda AFTER / Stage C senkron MP4
- kare başına IoU + 20-kare hareketli ortalama
- State Accuracy, success AUC, kayıp episode sayısı, en uzun kayıp
- en çok iyileşen örneğin yanında **tüm validation'daki en kötü gerileme**

O son madde önemli: ortalama iyileşirken bir sınıfta kötüleşme saklanabilir.
En kötü gerileme videosunu izlemeden "düzeldi" deme.

### 3e. ONNX / TensorRT'nin bozulmadığını görmek

```bash
python tools/export_edgetam_onnx.py --size 768
python tools/check_trt_parity.py
```
SAMURAI export yoluna hiç kurulmaz — doğrulandı, dosyada yalnızca iki yorum
satırı geçiyor. `check_image_size()` artık 640/896'yı config okunurken
reddeder, ilk ileri geçişte çökmek yerine.

---

## 4. Bilinen açıklar

1. **`src/training/aerovis.py` depoda yok.** Havuzları üreten modül Drive'da
   `repo_patch_aerovis.py` olarak duruyor. Mevcut notebook'ların hiçbiri ona
   ihtiyaç duymuyor (hepsi hasat edilmiş zip'leri okuyor). AeroVIS'i yeniden
   hasat etmen gerekirse depoya alınmalı.
2. **`wanted_frames` yalnızca `image`'i okur, `image_rel`'i değil.**
   `resolve_images_root`'ta kapatılan boşluğun aynısı `extract_frames`
   yolunda duruyor. Düz `unzip` bu yoldan geçmez, o yüzden 0b'deki hücre
   güvenli; ama `POOL_ARCHIVES` ile AeroVIS çıkarmaya kalkarsan "missing"
   diyebilir. (AeroVIS için zaten kazanç yok: havuz arşivin %97'sini istiyor.)
3. **34/35 zincirlenmiş değil.** İkisi de sıfırdan başlıyor. 34'ün çıktısını
   22/32'ye temel yapmak istersen `REFERENCE_CHECKPOINT`'i elle ver.
