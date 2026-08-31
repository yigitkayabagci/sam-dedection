# Termal yol haritası: hangi path, hangi ağırlık, nasıl entegre

AeroVIS ve RGB kolları (27, 28, 35) bu dosyada **yok** — hedef termal, onlar
sonraya bırakıldı. Buradaki her şey termal havuzlarla ve elindeki verilerle
çalışır.

---

## 1. Üç path

### Path A — sadece Stage B (en kısa yol)

```
21  →  19  →  32
```

| | notebook | ne için | çıktı |
|---|---|---|---|
| 21 | `21_pool_data_readiness.ipynb` | eğitim yok. Havuzların diskte olup olmadığını, hangi kapının ne kestiğini söyler | rapor |
| 19 | `19_thermal_stage_b_pool.ipynb` | `EPOCHS [1,3]`, `STEPS 400` — boru hattının uçtan uca çalıştığını 20 dakikada gösterir | tek kullanımlık checkpoint |
| 32 | `32_aerial_thermal_stage_b_stable.ipynb` | asıl Stage B. LR profilini kısa pilotla eler, VTUAV-LT zorunlu, hardening açık | **`edgetam_pool_aerial_thermal_stable_512.pt`** |

**32'yi 22 yerine tercih et.** Drive'daki `thermal_deep/run.json`'da head val
loss 0.2723 iken encoder açılınca 21.84 ve ardından NaN görülmüş; kaydedilen
checkpoint head aşamasında kalmış. 32 tam olarak bu kararsızlığı çözmek için
var: LR'yi batch'ten bağımsız 1.0 ölçekte tutuyor, trunk'ı en yavaş açıyor ve
üç güvenli profili aynı split üzerinde kısa pilotta karşılaştırıyor.

### Path B — pretrain dahil (optimize path)

```
21  →  19  →  34  →  32 (BASE_CHECKPOINT = 34'ün çıktısı)  →  29  →  30
```

| | notebook | çıktı |
|---|---|---|
| 34 | `34_pretrain_thermal_aerial.ipynb` | `edgetam-stage-b/pretrain_thermal_aerial_finetune/edgetam_pool_pretrain_thermal_aerial_512.pt` |
| 32 | üstteki checkpoint'ten devam | `edgetam-stage-b/aerial_thermal_stable/edgetam_pool_aerial_thermal_stable_512.pt` |

**Zincirleme artık mümkün** — daha önce değildi. `BASE_CKPT` stok
`edgetam.pt`'ye sabitlenmişti, yani 34'ün 40 epoch'u altındaki koşuya hiçbir
şey kazandırmıyordu. 32'nin ve 19/20/22/23/34/35'in ayar hücresine
`BASE_CHECKPOINT` eklendi:

```python
BASE_CHECKPOINT = "/content/drive/MyDrive/edgetam-stage-b/pretrain_thermal_aerial_finetune/edgetam_pool_pretrain_thermal_aerial_512.pt"
```

Dosya yoksa koşu **assert ile durur** — sessizce stoktan başlamaz.

> `BASE_CHECKPOINT` ile `REFERENCE_CHECKPOINT` farklı şeyler.
> Birincisi **nereden başlanacağı**, ikincisi yalnızca before/after'ın **neye
> karşı** ölçüleceği. Zincir için gereken birincisi.

`BASE_CHECKPOINT` verildiğinde before/after **pretrain'e karşı** ölçülür —
yani "bu koşu pretrain'in üstüne ne kattı", ikisinin stoka toplam katkısı
değil. Doğru soru budur.

### Path C — Stage B + Stage C (takip)

```
Path A veya B  →  29  →  30  →  31
```

| | notebook | ne yapar | ön koşul |
|---|---|---|---|
| 29 | `29_thermal_contrast_tracking.ipynb` | 32'nin checkpoint'ini düşük-kontrast temporal kliplerle devam eğitir; stage B / temporal / temporal+SAMURAI A/B'si | 32 |
| 30 | `30_tracking_before_after_demo.ipynb` | **eğitim yok.** Senkron before/after MP4, IoU eğrisi, State Accuracy, kayıp episode'ları — ve en iyi iyileşmenin yanında **en kötü gerileme** | 29, aynı runtime |
| 31 | `31_aerial_thermal_tracking.ipynb` | VTUAV ST/LT + VTUAV-VIS + BIRDSAI kimlikleriyle temporal eğitim. Anti-UAV410 varsayılan **kapalı** | 32 + `MyDrive/VTUAV` |

**Stage C ne kadar kritik?** Stage B en kritik evre — tek kare maskesini o
öğretir. Ama senin bildirdiğin hata (benzer kontrastta hedefi bırakma) **tek
karede görünmez**: bellek bankası devreye girmeden ortaya çıkmıyor. Stage B
gerekli ama yeterli değil; 29/31 o hatanın ölçülebildiği tek yer, 30 ise
sadece gösterir.

---

## 2. Ağırlıkları notebook'suz sisteme entegre etme

Bir checkpoint'i kullanmak için ekstra notebook gerekmiyor. Config'in
`checkpoint:` alanı tek bağlantı noktasıdır.

```bash
cp configs/edgetam_512_thermal_guard.yaml configs/benim_termal.yaml
# tek satır:  checkpoint: /yol/edgetam_pool_aerial_thermal_stable_512.pt
```

Sonra üç giriş noktasının hepsi aynı config'i alır:

```bash
# etiketli diziler üzerinde skor (State Accuracy, success AUC, dropout)
python tools/eval_antiuav.py --data <diziler> --config configs/benim_termal.yaml

# adaptif pencere ile takip
python tools/track_adaptive.py --data <diziler> --config configs/benim_termal.yaml

# hız / gecikme
python tools/benchmark_tracking.py --frames 500 --config configs/benim_termal.yaml
```

Python'dan:

```python
import yaml
from pathlib import Path
from src.trackers.registry import build_tracker

cfg = yaml.safe_load(Path("configs/benim_termal.yaml").read_text())
tracker = build_tracker("edgetam", **cfg)
```

`build_tracker(name, **cfg)` YAML'ı doğrudan kwarg olarak geçirir — yani
config'deki her anahtar bir constructor parametresidir. Bilinmeyen bir anahtar
sessizce yok sayılmaz, **başlangıçta TypeError verir.**

TensorRT'ye geçerken aynı config, `--tracker edgetam_trt` ve engine yolları:
`guard` artık TRT tracker'ın da imzasında (değildi — config'e koyunca
çöküyordu).

---

## 3. Şu anda, eğitim beklemeden ne deneyebilirsin

**Klasik katmanların üçü de eğitimsizdir.** Checkpoint'e dokunmazlar, stok
EdgeTAM üzerinde bugün çalışırlar. Yani Stage B bitmeden ölçebilirsin.

Ama bir boşluk vardı ve kapatıldı: **`stabiliser.py` 700 satırdı ve hiçbir
config onu açmıyordu.** Artık `configs/edgetam_512_thermal_guard.yaml` var —
640×512 termal için native boyut, üç katman da açık:

| katman | hangi hatayı yanıtlıyor |
|---|---|
| `samurai` | düşük kontrastlı bir kareden sonra hedef kayboluyor ve dönmüyor. `object_score_logits` negatife düşünce EdgeTAM belleğe görünüş yerine sabit "nesne yok" vektörü yazıyor; bellek kapısı üç sinyalin de kötü dediği kareyi yazmayı reddediyor |
| `ego_motion` | kamera trackle beraber kayıyor. Kameranın kendi hareketi Kalman'a **kontrol girdisi** olarak veriliyor |
| `guard` | kutu araziye yayılıyor / atlıyor. Alan, en-boy ve — arka planın hareketi çıkarıldıktan sonra — kat edilen mesafe üzerinde makullük kapıları, histerezis, ve kaybolduğu yerde şablonla yeniden yakalama |

**768 değil 512 kullan.** 768 config'inin kendi başlığı söylüyor: 768, 720p ve
üstü için. 640×512 termalde encoder bütçesinin tamamını kameranın hiç
ölçmediği pikselleri büyütmeye harcar.

### Bugün yapılabilecek dört deney

**Deney 1 — klasik katmanlar tek başına ne kazandırıyor?**
Stok EdgeTAM ile, eğitim yok:
```bash
python tools/eval_antiuav.py --data <diziler> --config configs/edgetam_512.yaml          # BEFORE
python tools/eval_antiuav.py --data <diziler> --config configs/edgetam_512_thermal_guard.yaml  # AFTER
```
Bu temiz bir ölçüm: iki koşunun ağırlıkları aynı, tek fark üç katman.

**Deney 2 — hangi katman ne kadarını yapıyor?**
Config'i kopyalayıp blokları tek tek çıkar: yalnız `samurai`; `samurai` +
`ego_motion`; üçü birden. `guard`'ı `ego_motion` olmadan denemenin anlamı yok
— sıçramayı arka planın hareketi çıkarıldıktan sonra ölçüyor, yoksa drone'un
yaw'ını hedef hareketi sanar.

**Deney 3 — kapılar çırpınıyor mu?**
Tracker `self.verdicts` içinde nesne başına kararı tutuyor. Bir dizide kaç
karenin `suspect`/`lost` okunduğuna bak. Sürekli `suspect` görüyorsan kapılar
gerçek hedefe ateş ediyor demektir — `max_area_ratio` ve `max_jump` gevşetilir.
Hiç görmüyorsan `caution_below` yükseltilir.

**Deney 4 — maliyeti:**
```bash
python tools/benchmark_tracking.py --frames 500 --config configs/edgetam_512_thermal_guard.yaml
```
Beklenen: kare başına bir senkronizasyon (birkaç düzine bayt) + bir seyrek
optik akış — onlarca ms süren encoder'ın yanında ~1 ms.

---

## 4. Bu turda yapılan değişiklikler

| değişiklik | neden |
|---|---|
| `configs/edgetam_512_thermal_guard.yaml` **(yeni)** | klasik katmanların üçü, termal native boyutta. Hiçbir config `guard` açmıyordu |
| `configs/edgetam_768_samurai.yaml` | `guard:` bloğu eklendi |
| `edgetam_trt_tracker.py` | `guard` imzaya eklendi — yoksa config'e koyunca TypeError |
| `build_stage_b_notebooks.py` | `BASE_CHECKPOINT` ayarı → pretrain zinciri |
| `build_stage_b_stable_notebook.py` | 32'ye de `BASE_CHECKPOINT`, ve `MAX_AREA 0.2` |
| `HARDER` preset | `MAX_AREA 0.25 → 0.2` |
| `tests/test_stabiliser.py` | 3 yeni test: her config'in `guard` bloğu `GuardConfig`'e uyuyor mu, termal config üç katmanı da taşıyor mu, **iki tracker da config'in her anahtarını kabul ediyor mu** |

### MAX_AREA 0.20 ne kesiyor

AeroVIS'in 1 411 468 instance'ında ölçüldü:

| eşik | atılan | oran |
|---|---|---|
| 0.25 | 603 | %0.043 |
| **0.20** | **1 463** | **%0.104** |
| 0.15 | 1 934 | %0.137 |

0.20'nin üstündeki 1 463'ün **1 434'ü UAVDT `vehicle`** (zaten 0.5'e
inceltilmiş kaba sınıf), 29'u person. Meşru sınıfların en büyüğü: kamyon
%13.6, tekne %8.7, araba %6.0, otobüs %4.7 — yani 0.20 hâlâ en büyük kamyona
1.47× pay bırakıyor ve hiçbir ince taneli sınıftan tek örnek kesmiyor.

**Ama bu ölçüm RGB hava verisinden.** Termal havuzların geometrisi farklı
olabilir; VTUAV gibi UAV-üstü tracking setlerinde hedef daha yakın olabilir.
Notebook artık her havuz için bu tabloyu kendi kayıtlarından yazdırıyor:

```
pool                     inst  p50 px     p50   p99.9     max  <min  >max
```

**`>max` sütunu binde mertebesinden tam yüzde mertebesine çıkarsa** o havuzda
`MAX_AREA` gerçek hedef kesiyor demektir — o zaman gevşet. İlk 21 veya 19
koşusunda bu tabloyu oku, sonra birlikte bakarız.
