# Jetson Orin AGX — EdgeTAM Benchmark & Quantization Planı

Hedef: EdgeTAM video tracking pipeline'ını Orin AGX üzerinde **çalışır hale
getirmek**, sonra **kademeli olarak hızlandırmak**, ve her adımda **hız +
doğruluk** ölçümünü yan yana koyan tek bir benchmark tablosu üretmek.

Sıra bilinçli olarak "önce doğru, sonra hızlı":

```
Faz 0  Ortam sağlığı        -> scripts/jetson_doctor.py
Faz 1  PyTorch baseline     -> referans FPS + referans maske (altın standart)
Faz 2  ONNX export          -> sayısal parite kontrolü
Faz 3  TRT float            -> fp32 / tf32 / fp16 / bf16
Faz 4  TRT INT8 (PTQ)       -> kalibrasyonlu, doğruluk kaybı ölçülü
Faz 5  İleri knob'lar       -> best / sparsity / DLA / CUDA graph
Faz 6  Kapsam genişletme    -> encoder dışı bloklar (opsiyonel)
Faz 7  Rapor                -> tek tablo, karar
```

Her fazın çıktısı bir sonraki fazın girdisidir. Bir faz yeşile dönmeden
sonrakine geçilmez — aksi halde hata kaynağını izole edemeyiz.

---

## 1. Donanım gerçekleri: Orin AGX'de ne mümkün, ne değil

Orin AGX GPU'su **Ampere, compute capability 8.7 (sm_87)**. Bu, quantization
matrisini doğrudan belirliyor. Var olmayan bir modu denemek için zaman
harcamayalım:

| Mod | Orin AGX (sm_87) | Not |
|---|---|---|
| FP32 | ✅ | Referans; en yavaş |
| TF32 | ✅ | Ampere'de TRT varsayılan olarak **açık**. "FP32 baseline" ölçerken `--noTF32` ile kapatmazsan aslında TF32 ölçersin |
| FP16 | ✅ | Tensor Core; ana kazanç burada |
| BF16 | ✅ | TRT ≥ 8.6. FP16'dan genelde **daha hızlı değil**, sadece daha geniş dinamik aralık |
| INT8 | ✅ | Tensor Core. Kalibrasyon **zorunlu** |
| INT8 + FP16 mixed | ✅ | TRT katman bazında seçer (`best`) |
| 2:4 Structured Sparsity | ✅ | Ancak ağırlıklar sparsity-aware eğitilmediyse kazanç ~0 |
| FP8 | ❌ | Ada (sm_89) / Hopper+ gerektirir. TRT build hata verir |
| INT4 / NVFP4 | ❌ | Blackwell sınıfı. Orin'de yok |
| DLA v2 (2 çekirdek) | ✅ | Sadece FP16/INT8. Desteklenmeyen katman → GPU fallback şart |

**Sonuç:** gerçek quantization matrisimiz `fp32 → tf32 → fp16 → bf16 → int8 →
int8+fp16(best)`, artı ortogonal knob'lar (sparsity, DLA, CUDA graph).
FP8/INT4 planın dışında — donanım desteklemiyor.

### Neyi quantize ediyoruz?

Şu an TRT'ye taşınan tek blok **image encoder** (RepViT trunk + FPN neck).
Sebep: per-frame maliyetin baskın kısmı orada ve statik shape'li
(`1x3x1024x1024`) — export'u temiz. Memory attention (2D Perceiver) ve mask
decoder PyTorch'ta kalıyor; dinamik shape'li ve object sayısına bağlılar.

Bu, **Amdahl tavanı** demek: encoder toplam sürenin %X'i ise, encoder'ı
sonsuz hızlandırsan bile toplam hızlanma `1/(1-X)` ile sınırlı. Bu yüzden
Faz 1'de encoder/geri-kalan ayrımını ölçüyoruz (`tools/benchmark.py`
`encoder_ms` / `rest_ms` kolonları) — o oran, INT8'e ne kadar emek vermeye
değeceğini söyler. Faz 6 bu tavanı kaldırmakla ilgili.

---

## 2. Ölçüm metodolojisi (bunu atlarsak tüm sayılar çöp)

Jetson'da ölçüm hijyeni, optimizasyonun kendisi kadar önemli. Kurallar:

1. **Güç modu sabit.** Her ölçümden önce:
   ```bash
   sudo nvpmodel -m 0        # MAXN
   sudo jetson_clocks         # saatleri tavana kilitle
   sudo nvpmodel -q           # doğrula
   ```
   Bunu yapmadan alınan iki ölçüm arasında %30-50 fark çıkabilir.
2. **Termal denge.** Arka arkaya build + benchmark koşarsan ikinci koşu
   throttle yiyebilir. Benchmark öncesi ~30 sn boşta bekle; `tegrastats`
   ile sıcaklığı kaydet (`--tegrastats` bayrağı bunu otomatik yapıyor).
3. **Warm-up hariç.** İlk 10-20 kare model yükleme + CUDA context + cuDNN
   algoritma seçimi içerir. `--warmup 10` varsayılanı bunu atar.
4. **Render ve disk I/O hariç.** `tools/benchmark.py` maske çizmez, video
   yazmaz. `cli.py --fps-chart` ile alınan FPS *uçtan uca* FPS'tir ve
   benchmark FPS'inden düşüktür — ikisini karıştırma.
5. **Kareler bir kez açılır.** Tüm varyantlar aynı JPG cache dizinini
   kullanır; decode maliyeti sabitlenir.
6. **Tekrar + medyan.** `--repeat 3`, medyan raporlanır. Tek koşuya bakma.
7. **Sessiz fallback yasak.** `configs/*_trt*.yaml` içinde benchmark için
   **`strict: true`**. Aksi halde engine yüklenemediğinde tracker sessizce
   PyTorch'a düşer ve "TRT sonucu" diye PyTorch sayısı raporlarsın. Bu, bu
   tür çalışmalarda en sık yapılan ölçüm hatasıdır.
8. **Doğruluk her zaman hızla birlikte.** Her varyant `--dump-masks` ile
   maskelerini kaydeder; `tools/compare_masks.py` FP32 PyTorch baseline'a
   karşı IoU verir. IoU'suz FPS tablosu karar aldırmaz.

### Raporlanan metrikler

| Metrik | Anlamı |
|---|---|
| `fps_median` | Warm-up sonrası, render hariç propagate FPS (medyan) |
| `latency_p50/p95_ms` | Kare başına gecikme dağılımı; p95 jitter'ı gösterir |
| `encoder_ms` | Image encoder'da geçen süre (CUDA-senkron ölçülür) |
| `rest_ms` | Memory attention + mask decoder + kalan |
| `peak_torch_mb` | `torch.cuda.max_memory_allocated` |
| `device_used_mb` | `cuda.mem_get_info` deltası (TRT arena dahil) |
| `mean_iou` | FP32 PyTorch baseline'a karşı ortalama maske IoU |
| `iou_below_0.9_pct` | IoU < 0.9 olan karelerin yüzdesi (drift yakalar) |
| `engine_mb`, `build_s` | Engine boyutu ve derleme süresi |

---

## 3. Fazlar

### Faz 0 — Ortam sağlığı

```bash
python3 scripts/jetson_doctor.py            # insan-okur rapor
python3 scripts/jetson_doctor.py --json > env.json
```

Doctor şunları doğrular: L4T/JetPack sürümü, güç modu, CUDA, cuDNN,
TensorRT (hem python modülü hem `trtexec`), PyTorch'un **Jetson wheel'i mi
yoksa PyPI x86/CPU wheel'i mi** olduğu, venv'in `--system-site-packages` ile
açılıp açılmadığı, onnx/onnxruntime, sam2 (EdgeTAM) importu ve checkpoint.

Her başarısız kontrol için ekrana somut bir `FIX:` satırı basar. Doctor
`BLOCKER` ile çıkarsa Faz 1'e geçme.

**Çıkış kriteri:** `torch.cuda.is_available() == True`, cihaz `sm_87`,
`import tensorrt` çalışıyor, `trtexec` bulunuyor, `import sam2` çalışıyor.

### Faz 1 — PyTorch baseline (altın standart)

Amaç: TRT'yi hiç karıştırmadan modelin doğru çalıştığını görmek ve
karşılaştıracağımız referans maskeleri üretmek.

```bash
# 1) Doğruluk referansı: fp32, deterministik
python3 tools/benchmark.py \
  --video samples/synthetic_car.mp4 \
  --prompt-file samples/synthetic_car_box.json \
  --variants pytorch_fp32 \
  --dump-masks runs/baseline_fp32.npz \
  --out runs/phase1

# 2) Hız referansları
python3 tools/benchmark.py ... --variants pytorch_fp32,pytorch_fp16,pytorch_bf16
```

Burada `encoder_ms / rest_ms` oranını not al — Faz 3-4'ün beklenen kazanç
tavanı bu.

**Çıkış kriteri:** üç precision da çöküyor değil, maskeler görsel olarak
doğru (`cli.py` ile bir çıktı videosu üretip bak), `baseline_fp32.npz` var.

### Faz 2 — ONNX export + parite

```bash
python3 tools/export_image_encoder_onnx.py \
  --checkpoint third_party/EdgeTAM/checkpoints/edgetam.pt \
  --output models/edgetam_image_encoder.onnx \
  --verify        # onnxruntime CPU ile FP32 parite (< 1e-3)
```

Notlar:
- Export **CPU'da ve FP32** yapılır. ONNX'i fp16'a çevirmeye çalışma —
  precision kararını TRT'ye bırak; QDQ olmayan bir fp16 ONNX, TRT'nin
  katman bazında karar verme yeteneğini elinden alır.
- ONNX **donanımdan bağımsız**. İstersen x86 makinede export edip `.onnx`
  dosyasını Orin'e kopyalayabilirsin. **`.engine` ise donanıma özgüdür**,
  mutlaka Orin'de üretilmeli.
- Statik shape (`dynamic_axes=None`) bilinçli: TRT statik shape'te daha
  agresif optimize ediyor ve CUDA graph'ı mümkün kılıyor.

**Çıkış kriteri:** `--verify` max abs diff < 1e-3.

### Faz 3 — TensorRT float varyantları

```bash
# FP16 (asıl beklenen kazanç)
python3 tools/build_trt_engine.py \
  --onnx models/edgetam_image_encoder.onnx \
  --engine models/enc_fp16.engine --precision fp16

# Dürüst FP32 baseline (TF32 kapalı!)
python3 tools/build_trt_engine.py ... --engine models/enc_fp32.engine --precision fp32

# TF32 (Ampere varsayılanı) ve BF16
python3 tools/build_trt_engine.py ... --precision tf32
python3 tools/build_trt_engine.py ... --precision bf16
```

Builder her engine'in yanına `<engine>.json` yazar: TRT sürümü, bayraklar,
build süresi, engine boyutu. Bu, sonradan "bu engine neyle üretilmişti"
sorusunu ortadan kaldırır.

Sonra benchmark:
```bash
python3 tools/benchmark.py ... --variants trt_fp32,trt_tf32,trt_fp16,trt_bf16 \
  --dump-masks-dir runs/phase3
python3 tools/compare_masks.py runs/baseline_fp32.npz runs/phase3/trt_fp16.npz
```

**Çıkış kriteri:** `trt_fp16` çalışıyor, `mean_iou > 0.98`, FPS > PyTorch fp16.
IoU beklenenden düşükse Faz 3'ün hata tablosuna bak (NaN/overflow bölümü).

### Faz 4 — INT8 post-training quantization

INT8'in tek kritik noktası **kalibrasyon**: TRT her tensöre bir dinamik
aralık atamalı. Kalibrasyonsuz `--int8` verirsen TRT tüm aralıkları 1.0
kabul eder ve çıktı çöp olur (boş maske / rastgele maske). Bu, "INT8
çalışmıyor" diye rapor edilen vakaların çoğunun sebebidir.

Kalibrasyon verisi **gerçek sahne kareleri** olmalı ve modele girenle
**bit-bit aynı preprocessing**'den geçmeli (1024x1024 PIL resize, /255,
SAM2 mean/std). `tools/calibration.py` bunu EdgeTAM'in kendi
`_load_img_as_tensor` akışını taklit ederek yapıyor.

```bash
python3 tools/build_trt_engine.py \
  --onnx models/edgetam_image_encoder.onnx \
  --engine models/enc_int8.engine \
  --precision int8 \
  --calib-video samples/synthetic_car.mp4 \
  --calib-num 200 \
  --calib-cache models/enc_int8.calib

# INT8 + FP16 karışık: TRT hangi katmanın INT8 kaldırdığına kendi karar verir
python3 tools/build_trt_engine.py ... --precision int8-fp16 --calib-cache models/enc_int8.calib
```

Kalibrasyon seti kuralları:
- 100-500 kare yeter; daha fazlası nadiren yardım eder.
- **Benchmark videosundan farklı** kareler kullan (kendi test setine
  kalibre etme). Aynı sahnenin başka bölümü ya da başka bir çekim ideal.
- Sahne çeşitliliği önemli: aydınlık/karanlık, farklı ölçek.
- `--calib-cache` üretildikten sonra tekrar build'lerde veri gerekmez,
  cache yeter (build çok hızlanır). Model/ONNX değişirse cache'i sil.

**Çıkış kriteri:** `trt_int8` çalışıyor ve `mean_iou` raporlandı. IoU düşükse
(< 0.95) bu bir başarısızlık değil, bir **bulgu**: kalibrasyon setini
büyüt/çeşitlendir, sonra `int8-fp16` karışık moda geç, sonra hassas
katmanları FP16'ya sabitle (Faz 5).

### Faz 4b — NVIDIA Model Optimizer (explicit QDQ) — koşullu

TensorRT'nin `IInt8EntropyCalibrator2` ile yaptığı **implicit quantization**,
TensorRT 10.1'den itibaren **deprecated**. Yerini **explicit quantization**
alıyor: ölçekler grafiğin içine QDQ (QuantizeLinear/DequantizeLinear)
düğümleri olarak gömülüyor, TensorRT tahmin yürütmüyor. NVIDIA Model
Optimizer (`nvidia-modelopt`) bu QDQ'lu ONNX'i üreten araç.

**Orin'de doğrudan çalışmıyor:** ModelOpt'un resmî sistem gereksinimi
x86_64. Ama ürettiği ONNX donanımdan bağımsız olduğu için iki makineli bir
akış kurulabilir:

```
x86 makine:   edgetam_image_encoder.onnx  --ModelOpt PTQ-->  ..._int8_qdq.onnx
   ↓ scp
Orin:         tools/build_trt_engine.py --precision int8 --onnx ..._int8_qdq.onnx
```

Builder QDQ düğümlerini otomatik algılıyor ve kalibratörü atlıyor — ek
bayrak gerekmiyor.

**Orin AGX'te ModelOpt'un hangi formatları işe yarar:** sadece **INT8**.
FP8 Hopper+, NVFP4 Blackwell gerektiriyor; sm_87'de ikisi de yok. Yani
ModelOpt'un tanıtım materyalindeki FP4/FP8 kazanımları bu donanımda geçerli
değil.

**Ne zaman değer:** ModelOpt doğruluk aracı, hız aracı değil. INT8'in
zaten hız kazandırdığı doğrulandıktan sonra, kaybedilen IoU'yu geri almak
için anlamlı. Faz 4'te basit kalibratörle INT8 hız kazancı çıkmazsa
(bkz. aşağıdaki launch-bound notu), ModelOpt kurulumuna girmek boşa emek.

**Root'suz alternatif (Orin üzerinde):** `onnxruntime.quantization` ile
`QuantFormat.QDQ` kullanarak aarch64'te de QDQ üretilebilir. ModelOpt kadar
TensorRT-dostu düğüm yerleşimi vermez ama x86 makine gerektirmez.

### Faz 5 — İleri knob'lar

| Knob | Komut | Beklenti |
|---|---|---|
| `best` (TRT hepsini dener) | `--precision best --calib-cache ...` | int8-fp16'ya yakın |
| 2:4 sparsity | `--sparsity` | Ağırlıklar sparse eğitilmediyse ~%0. Ölç ve geç |
| DLA | `--dla-core 0 --allow-gpu-fallback` | GPU'yu boşaltır, tek başına daha yavaş olabilir; çok-akışlı senaryoda değerli |
| CUDA graph | `configs/*.yaml: use_cuda_graph: true` | Küçük/statik grafta launch overhead'i siler, %5-15 |
| Katman hassasiyeti | `--fp16-layers <regex>` | INT8'te IoU'yu kurtarmak için ilk/son katmanları FP16'da tut |
| `builderOptimizationLevel` | `--opt-level 5` | Daha uzun build, biraz daha hızlı engine |

CUDA graph, statik shape'li encoder için ucuz bir kazanç: engine tek bir
graph olarak yakalanıp replay ediliyor, per-launch CPU overhead'i kalkıyor.
Orin'in CPU'su nispeten zayıf olduğu için Jetson'da x86'dan daha çok fark
eder.

### Faz 6 — Kapsam genişletme (opsiyonel, "başka case'ler")

Encoder'ı hızlandırdıktan sonra `rest_ms` baskın hale gelirse sıradaki
hedefler:

1. **Mask decoder** → ONNX/TRT. Statik'e yakın; prompt embedding + iki
   ufak transformer bloğu. Orta zorluk.
2. **Memory attention (2D Perceiver)** → en zoru. Memory bank uzunluğu
   kare sayısıyla değişiyor → dinamik shape profili (`min/opt/max`)
   gerekiyor. Sabit pencere (`num_maskmem`) varsayımıyla statikleştirmek
   pratik bir kaçamak.
3. **PyTorch tarafı ucuz kazançlar** (TRT'siz): `channels_last`,
   `torch.compile(mode="max-autotune")`, `sdpa` backend seçimi. Bunlar
   `rest_ms` için TRT'den daha az emekle kazanç verebilir — matrise
   `pytorch_compile` varyantı olarak eklendi.
4. **Girdi çözünürlüğü.** 1024 → 512 encoder maliyetini ~4x düşürür ama
   doğruluğu ciddi etkiler. Quantization değil ama aynı hız/doğruluk
   eğrisinde bir nokta; tabloya koymaya değer.

### Faz 7 — Rapor

```bash
python3 tools/benchmark.py --report runs/ --out runs/final_report.md
```

Tüm varyant JSON'larını tarayıp tek markdown tablo + CSV üretir. Karar
kuralı: **`mean_iou ≥ 0.98` kısıtı altında en yüksek `fps_median`.**

---

## 4. Hedef benchmark matrisi

| # | Varyant | Encoder | Precision | Not |
|---|---|---|---|---|
| 1 | `pytorch_fp32` | PyTorch | fp32 | Altın standart (IoU referansı) |
| 2 | `pytorch_fp16` | PyTorch | autocast fp16 | |
| 3 | `pytorch_bf16` | PyTorch | autocast bf16 | Repo'nun mevcut varsayılanı |
| 4 | `pytorch_compile` | PyTorch | bf16 + `torch.compile` | Faz 6 |
| 5 | `trt_fp32` | TensorRT | fp32 (TF32 kapalı) | Dürüst TRT tabanı |
| 6 | `trt_tf32` | TensorRT | tf32 | Ampere varsayılanı |
| 7 | `trt_fp16` | TensorRT | fp16 | Ana aday |
| 8 | `trt_bf16` | TensorRT | bf16 | |
| 9 | `trt_int8` | TensorRT | int8 (PTQ) | Kalibrasyonlu |
| 10 | `trt_int8_fp16` | TensorRT | int8+fp16 | Karışık |
| 11 | `trt_best` | TensorRT | best | TRT seçsin |
| 12 | `trt_fp16_sparse` | TensorRT | fp16 + 2:4 | Ölç ve muhtemelen ele |
| 13 | `trt_fp16_graph` | TensorRT | fp16 + CUDA graph | Launch overhead testi |
| 14 | `trt_int8_dla` | DLA | int8 + GPU fallback | Opsiyonel |

---

## 5. Bilinen sorunlar → çözüm

Orin'de bu pipeline'ı ayağa kaldırırken en sık çarpılan duvarlar. Doctor
scripti bunların çoğunu önden yakalar.

### 5.1 Kurulum / CUDA

| Belirti | Sebep | Çözüm |
|---|---|---|
| `torch.cuda.is_available() == False` | PyPI'dan `pip install torch` yapılmış — o wheel aarch64'te CPU-only | JetPack'e eşleşen NVIDIA wheel'i kur (`https://developer.download.nvidia.com/compute/redist/jp/` ya da jetson-ai-lab pip index). PyPI torch'u **önce kaldır** |
| `OSError: libcudart.so.12: cannot open shared object file` | CUDA runtime yolda değil / sürüm uyuşmuyor | `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`; `ls -l /usr/local/cuda` ile symlink'in doğru sürüme baktığını doğrula |
| `RuntimeError: operator torchvision::nms does not exist` | torch ↔ torchvision sürüm çifti uyumsuz | torchvision'ı **o torch sürümüyle** kaynaktan derle ya da eşleşen Jetson wheel'ini al |
| `numpy.core.multiarray failed to import` / ABI hataları | Jetson torch wheel'i numpy 1.x'e derlenmiş, ortamda numpy 2.x var | `pip install "numpy<2"` |
| venv içinde `ModuleNotFoundError: No module named 'tensorrt'` | JetPack TRT'yi apt ile `/usr/lib/python3.X/dist-packages` altına kurar; normal venv bunu görmez | venv'i `python3 -m venv --system-site-packages .venv` ile yeniden oluştur (en temiz yol). Alternatif: `tensorrt*` ve `libnvinfer*` dizinlerini venv'in `site-packages`'ına symlink'le |
| Mevcut `.venv`'i `--system-site-packages` ile yeniden kurdum ama hâlâ eski Python / `include-system-site-packages = false` | `python -m venv` var olan dizinin interpreter symlink'ini **değiştirmez**; ayrıca `ensurepip` başarısız olursa bayrak `false` kalır (venv onu pip kurulumundan sonra yazar) | `rm -rf .venv` sonra yeniden kur. `ensurepip is not available` derse: `sudo apt install python3.10-venv` |
| `import torch` → `ImportError: libcufile.so.0 / libcupti.so.12 / libcurand.so.10 ... cannot open shared object file` | JetPack kısmi kurulmuş: CUDA ~15 ayrı apt paketi ve matematik kütüphaneleri (`curand`, `cusolver`, `cusparse`, `nvrtc`, `cupti`, `cufile`) eksik. torch hepsine link'li olduğu için linker daha yükleme anında reddediyor — o fonksiyonları hiç çağırmasan bile | Eksiklerin tamamını tek seferde çıkar: `ldd .venv/lib/python3.10/site-packages/torch/lib/*.so \| grep 'not found' \| sort -u`. Sonra `sudo apt install -y cuda-libraries-12-6 cuda-cupti-12-6 libcufile-12-6`, ya da root'suz: `pip install nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-cupti-cu12 nvidia-cufile-cu12` + `LD_LIBRARY_PATH`'e o dizinleri ekle (`/usr/local/cuda/targets/aarch64-linux/lib` **başta** kalsın ki JetPack'in Tegra derlemeleri kazansın) |
| `find /usr/local/cuda -name 'libX.so*'` boş dönüyor ama kütüphane orada | `/usr/local/cuda` bir symlink, `find` symlink'leri takip etmez | `find -L ...` kullan ya da `readlink -f /usr/local/cuda` ile gerçek yola bak |
| `import sam2` çalışmıyor | EdgeTAM editable kurulmamış | `bash scripts/setup_edgetam.sh` |
| Hydra: `Cannot find primary config 'configs/edgetam.yaml'` | `model_cfg` bir **dosya yolu değil**, sam2 paketi içindeki Hydra config adı | Değeri olduğu gibi bırak; `cli.py` bilinçli olarak bu anahtarı path'e çevirmiyor. EdgeTAM'in editable kurulu olduğunu doğrula |
| RepViT ağırlıkları indirilmeye çalışılıyor / SSL hatası | timm trunk `pretrained=True` | `setup_edgetam.sh` bunu `False`'a çeviriyor; script'i çalıştır |
| Multi-object'te `view size is not compatible...` | Perceiver'da `expand().view()` | `setup_edgetam.sh` `.reshape()`'e patch'liyor |
| `decord` kurulamıyor | aarch64 wheel'i yok | Sorun değil: `--video-mode jpg` (auto zaten ona düşüyor). İstersen `eva-decord` kaynaktan |
| Rastgele `CUDA error: an illegal memory access` | GPU bellek baskısı / uzun video | `offload_state_to_cpu: true`, `offload_video_to_cpu: true`; `CUDA_LAUNCH_BLOCKING=1` ile gerçek satırı bul |

### 5.2 ONNX export

| Belirti | Sebep | Çözüm |
|---|---|---|
| `Exporting the operator ... is not supported` | Opset düşük | `--opset 18` (varsayılan). TRT 10 opset 11+ okur |
| `Expected 3 post-scalp FPN levels, got N` | Model config'inde farklı `scalp` | Wrapper'daki beklentiyi ayarla ya da config'i düzelt |
| Parite testi > 1e-3 | Model `eval()` değil / tracing sapması | `build()` zaten `eval()` çağırıyor; dropout/BN'yi doğrula, tracer uyarılarını oku |
| Export sırasında OOM | 1024x1024 tam grafik CPU'da izleniyor | Export'u x86 makinede yap, `.onnx`'i kopyala (donanımdan bağımsız) |

### 5.3 TensorRT build

| Belirti | Sebep | Çözüm |
|---|---|---|
| `trtexec: command not found` | Jetson'da PATH'te değil | `/usr/src/tensorrt/bin/trtexec` (build script'i bunu zaten deniyor) |
| `Unsupported ONNX data type` / parser hatası | TRT sürümü ONNX opset'ten eski | TRT sürümünü doctor ile gör, gerekirse `--opset 17` ile yeniden export et |
| `Error Code 2: OutOfMemory (no further information)` | Workspace veya sistem RAM'i yetmedi | `--workspace 2048` düşür; build sırasında başka iş çalıştırma; swap ekle |
| Build 30+ dakika sürüyor | Her build'de taktik araması sıfırdan | `--timing-cache models/timing.cache` (builder varsayılan olarak yazıyor); `--opt-level 3` |
| `The engine plan file is not compatible with this version of TensorRT` | Engine başka cihazda/TRT sürümünde üretilmiş | Engine'i **Orin'de** yeniden üret. JetPack yükseltmesinden sonra tüm engine'leri sil |
| FP16'da NaN / tamamen boş maske | Bir katmanda overflow | `--fp16-layers` ile o katmanları FP32'de tut; ya da `--precision bf16` (daha geniş aralık) |
| INT8 çıktısı çöp | Kalibratör verilmemiş → tüm dinamik aralıklar 1.0 | `--calib-video`/`--calib-dir` ile gerçek kalibrasyon çalıştır. `--int8`'i tek başına asla kullanma |
| INT8 build'i kalibrasyonu atlıyor | Eski/uyumsuz `--calib-cache` var | Cache'i sil, yeniden kalibre et |
| DLA'da `Layer ... is not supported on DLA` | DLA katman kümesi sınırlı | `--allow-gpu-fallback` şart; yine de karma yürütme yavaş olabilir |
| FP8 denemesi hata veriyor | sm_87'de FP8 yok | Matristen çıkar (bkz. §1) |

### 5.4 Runtime

| Belirti | Sebep | Çözüm |
|---|---|---|
| "TRT çalışıyor" ama FPS PyTorch ile aynı | `strict: false` iken engine yüklenemedi, sessizce PyTorch'a düşüldü | Benchmark configlerinde **`strict: true`**. Log'da `[edgetam_trt] falling back` satırını ara |
| `TRT execute_async_v3 failed` | Input shape set edilmemiş / adres bağlanmamış | Runtime bunu yapıyor; engine'in gerçekten tek girişli olduğunu `--inspect` ile doğrula |
| dtype mismatch / bozuk çıktı | Engine girişi fp16 beklerken fp32 tensor bağlanmış | Runtime artık girişi engine'in beklediği dtype'a çeviriyor. Engine'i `--inputIOFormats` olmadan kur (I/O fp32 kalsın) |
| `conv_s0` çağrısında dtype hatası | TRT fp16 çıktı + autocast kapalı (precision fp32) | `trt_output_dtype: float32` (varsayılan `auto` bunu zaten yapıyor) |
| CUDA graph capture başarısız | Yakalama sırasında dinamik allocation | Runtime kalıcı buffer kullanıyor; yine de olursa `use_cuda_graph: false` ile devam eder (fail-soft) |

### 5.5 Ölçüm

| Belirti | Sebep | Çözüm |
|---|---|---|
| Aynı komut iki farklı FPS veriyor | Güç modu / throttle | `nvpmodel -m 0` + `jetson_clocks`; `--repeat 3` medyan |
| FPS beklenenden çok düşük | JPG decode ve overlay ölçüme dahil | `tools/benchmark.py` kullan (render yok); `cli.py` uçtan uca ölçer |
| İlk koşu hep yavaş | Warm-up | `--warmup 10` |
| TRT hızlandı ama toplam FPS az arttı | Amdahl: encoder toplamın küçük parçası | `encoder_ms`/`rest_ms` oranına bak, Faz 6'ya geç |

---

## 6. TensorRT çıktısı repoyla uyuşmuyorsa: hangi dosya?

Engine'in gerçekte ne ürettiği ile repodaki tüketici kodun beklediği şey
ayrıştığında, düzeltilecek yer neredeyse her zaman şu dört dosyadan biri.
`python3 tools/build_trt_engine.py --inspect models/enc_fp16.engine` çıktısı
(giriş/çıkış adları, shape'ler, dtype'lar) bu tabloya girmek için yeterli.

| Belirti | Kök sebep | Düzeltilecek yer |
|---|---|---|
| `Expected exactly one input` | Engine'in birden fazla girişi var (export sırasında fazladan tensör kaçmış) | `tools/export_image_encoder_onnx.py` → `ImageEncoderWrapper.forward` tek tensör grubu döndürmeli |
| Çıkış sayısı 3 değil | `scalp` farklı ya da FPN seviye sayısı değişmiş | Önce `export_image_encoder_onnx.py` → `ImageEncoderWrapper.forward` (`len(feats) != 3` kontrolü), sonra `src/trackers/edgetam_trt_tracker.py` → `trt_forward` içindeki `features[-1]` / `backbone_fpn` sözlüğü |
| Çıkış **sırası** ters (en yüksek çözünürlük sonda) | ONNX output_names sırası | `export_image_encoder_onnx.py` → `output_names`; runtime sırayı isimden değil indeksten alıyor (`_trt_runtime._output_names`) |
| `Engine tensor 'x' has TensorRT dtype ... no torch equivalent` | Engine I/O formatı standart dışı (ör. `--inputIOFormats=fp16:chw`) | Engine'i standart FP32 I/O ile yeniden kur. Gerçekten gerekiyorsa `src/trackers/_trt_runtime.py` → `_build_dtype_map` |
| Maskeler boş/gürültü ama build başarılı | INT8 kalibrasyonsuz, ya da giriş dtype'ı yanlış bağlanmış | Kalibrasyon: `tools/build_trt_engine.py --calib-video ...`. Bağlama: `_trt_runtime.infer` girişi zaten çeviriyor, `--inspect` ile giriş dtype'ını doğrula |
| `conv_s0` / `conv_s1` dtype hatası | Engine fp16 çıkarıyor, autocast kapalı | `configs/*.yaml` → `trt_output_dtype: float32` (veya `precision`'ı bfloat16 yap) |
| `AttributeError: set_input_shape` / `execute_async_v3` | TensorRT 8.x API'si (JetPack 5) | `src/trackers/_trt_runtime.py` → `set_binding_shape` / `execute_async_v2` karşılıklarına çevir |
| `create_builder_config` / `set_memory_pool_limit` yok | Aynı: TRT 8.x builder API'si | `tools/build_trt_engine.py` → `build()` |
| Pozisyonel encoding shape'i tutmuyor | Neck'in `position_encoding`'i FPN seviyeleriyle eşleşmiyor | `edgetam_trt_tracker.py` → `trt_forward` içindeki `neck_pos(f)` döngüsü |
| `--inspect` "not compatible with this version" | Engine başka TRT/GPU'da üretilmiş | Engine'i cihazda yeniden kur; kod değişikliği değil |

Uyuşmazlığı bana getirirken en faydalı üç çıktı:
`build_trt_engine.py --inspect <engine>`, `--dry-run` ile ONNX katman listesi,
ve hatanın tam traceback'i.

---

## 6.5 ÖLÇÜM SONUCU: darboğaz CPU, sıralama değişti

`tools/encoder_probe.py` Orin AGX üzerinde şunu ölçtü:

| image_size | eager | CUDA graph | hızlanma |
|---|---|---|---|
| 1024 | 41.53 ms | **29.58 ms** | 1.40× |
| 768 | 35.46 ms | 15.69 ms | 2.26× |
| 512 | 33.33 ms | **6.54 ms** | 5.10× |

Graph süresi GPU'nun boşluksuz çalışma süresidir ve **pikselle neredeyse
birebir ölçeklenir** (oranlar 1.00 / 0.94 / 0.88). Yani GPU tarafı sağlıklı.

Eager süresi ise girdi küçüldükçe **~33 ms'lik bir tabana** yaklaşıp durur.
O taban, CPU'nun ~300 kernel'i sıraya koyma maliyetidir; kernel sayısı
girdi boyutuyla değişmediği için sabittir. 512'de GPU işini 6.5 ms'de
bitirir, kalan 26.8 ms boyunca **boş bekler**.

```
GPU'nun ihtiyacı :  29.6 ms (1024) · 15.7 (768) · 6.5 (512)   pikselle ölçeklenir
CPU'nun ihtiyacı :  ~33 ms sabit                              kernel sayısı belirler
eager süre       :  ikisinin kusurlu örtüşmesi
```

**Planın sırası bu yüzden değişti:**

1. **Önce kernel sayısını azalt** — CUDA graph (launch'ı siler) ve TensorRT
   katman füzyonu (kernel'leri birleştirir). Bağlayıcı kısıt budur.
2. **Sonra quantization** — INT8 GPU süresini azaltır, ama bağlayıcı kısıt
   CPU iken görünmez. Ancak 1. adım tamamlandıktan sonra ölçülebilir hale
   gelir.

Bugün doğrudan INT8'e gitmek, 29.6 ms'lik GPU işini yarıya indirip 33 ms'lik
CPU tabanının arkasında kaybetmek olurdu. `pytorch_graph` varyantı bu
kazancı TensorRT'siz ve doğruluk kaybı olmadan alıyor: 1024'te encoder
41.5 → 29.6 ms, tam kare 147.9 → 135.9 ms, uçtan uca 1.09×.

## 6.6 Hedef: 40 ms inference — bütçe

Çalışmanın hedefi kare başına **40 ms inference** (25 FPS). Faz 1 ölçümü
(1920×1080, 1 nesne, PyTorch bf16):

```
inference = encoder 41.1 + kalan bloklar 28.7 = 69.9 ms      gereken: 1.75x
```

Kritik kısıt: **kalan bloklara dokunulmazsa encoder'ın 11.3 ms'nin altına
inmesi gerekir.** Ölçülmüş CUDA graph encoder'ı 29.6 ms'ye getiriyor;
oradan 11.3'e inmek TensorRT'den gerçekçi olmayan bir beklenti.

| Adım | encoder | kalan | inference |
|---|---|---|---|
| bugün | 41.1 | 28.7 | 69.9 |
| + CUDA graph *(ölçüldü)* | **29.6** | 28.7 | 58.3 |
| + TRT fp16 encoder (temkinli) | ~20 | 28.7 | 48.7 |
| + TRT fp16 encoder (iyimser) | ~15 | 28.7 | 43.7 |
| **+ kalan bloklar da TRT** | ~20 | **~18** | **38** ✓ |

Sonuç: **40 ms hedefi, memory attention + mask decoder + memory encoder'ın
da TensorRT'ye taşınmasını zorunlu kılıyor.** §7'de "getirisi belirsiz"
diye sıralanan iş, bu hedefle birlikte opsiyonel olmaktan çıktı.

Faz 3 bu bütçeyi kesinleştirecek: TRT füzyonu encoder'ı 29.6'dan kaça
indiriyor? O sayı kalan blokların ne kadar hızlanması gerektiğini —
ve INT8'in gerekip gerekmediğini — belirliyor.

## 6.7 ÖLÇÜM SONUCU (Faz 4): INT8 işe yaramadı, çözünürlük yaradı

500 sentetik kare, 1 nesne, 3 tekrar, warmup 25. `precision: bfloat16`
her satırda sabit, yani tek değişken `.engine` dosyası.

| varyant | infer ms | enc ms | memattn | memenc | dec | other | fps (loop) | engine MB |
|---|---|---|---|---|---|---|---|---|
| trt_fp16 (1024) | 49.26 | 19.00 | 8.72 | 6.81 | 11.21 | 12.36 | 16.13 | 11.8 |
| trt_int8 (1024) | 50.00 | **19.92** | 9.46 | 6.85 | 11.12 | 12.29 | 15.97 | 7.8 |
| trt_int8_fp16 | 48.75 | 18.78 | 8.77 | 6.82 | 11.17 | 12.33 | 16.28 | 8.5 |
| trt_best | 47.52 | 18.81 | 8.71 | 6.79 | 11.07 | 12.32 | 16.61 | 8.5 |
| **trt_fp16_512** | **31.11** | **4.70** | 7.25 | 5.43 | 10.97 | 13.84 | **22.08** | 11.9 |

### INT8 neden hiçbir şey vermedi

Quantization gerçekten oldu: engine 11.8 MB → 7.8 MB, ağırlıklar küçüldü.
Süre değişmedi. Üç satırın sıralaması sebebi söylüyor:

- `int8` (INT8 + FP32 fallback) **en yavaş**: INT8 kernel'ı olmayan katman
  FP32'ye düşüyor, FP16'dan pahalı.
- `int8-fp16` en hızlı INT8 satırı: aynı katmanlar bu kez FP16'ya düşüyor.
- Ama saf `fp16`'ya karşı kazancı **%1.2**, yani gürültü.
- `best`, havuza bf16+tf32 ekleyip `int8-fp16` ile aynı yeri buluyor:
  daha geniş arama uzayı yeni bir şey bulamadı.

İlk hipotez **reformat patlaması**ydı: INT8↔float her sınırında TensorRT bir
reformat katmanı (tensörü oku, formatı çevir, geri yaz) koyar ve
bandwidth-bound bir kartta bu, INT8'in kazandırdığını geri alır. Ölçüm bunu
**doğrulamadı**:

| engine | katman | reformat/copy |
|---|---|---|
| enc_fp16 | 272 | 96 (%35) |
| enc_int8 | 263 | **87 (%33)** |

INT8 engine'de reformat *daha az*. Yani "ekstra dönüşüm katmanları" hipotezi
yanlış, o sayıyla açıklanmıyor.

Geriye kalan ve verilerle uyumlu açıklama: **TensorRT INT8 kernel'larını
ölçtü ve çoğunu reddetti.** Autotuner katman başına aday kernel'ları gerçek
donanımda süreleyip en hızlısını seçer; INT8 adayı FP16'yı yenemediğinde
seçilmez. Bu bir hata değil, autotuner'ın doğru çalışması. Bunu kesinleştiren
sayı katman başına precision dağılımıdır ve o, engine DETAILED profiling
verbosity ile kurulduğunda saklanır (artık varsayılan):

```bash
python3 tools/build_trt_engine.py --precision int8 \
  --onnx models/edgetam_image_encoder.onnx --engine models/enc_int8_v2.engine \
  --calib-cache models/enc_int8.calib
python3 tools/build_trt_engine.py --inspect models/enc_int8_v2.engine
```

Yan bulgu, asıl bulgudan daha önemli olabilir: **fp16 engine'in katmanlarının
%35'i saf veri taşıma.** Encoder grafiğinin üçte biri hesap değil kopya. Bu,
bandwidth-bound teşhisini bağımsız olarak destekliyor ve iki şeyi işaret
ediyor: (1) çözünürlüğü düşürmek neden bu kadar iyi çalıştı, (2) engine'in
FP32 çıkışı neden israf. Encoder kare başına 88 MB FP32 yazıyor
(256×256×256 + 128×128×256 + 64×64×256). `--io-fp16` bunu yarıya indirir ve
grafiğin sonundaki genişletme reformat'ını siler.

### Çözünürlük neden yaradı

`19.00 / 4.70 = 4.04`, ideal ölçekleme `(1024/512)² = 4.00`. TensorRT
encoder'ı **piksel sayısıyla neredeyse mükemmel ölçekleniyor**. PyTorch'ta
512'ye inmek işe yaramamıştı çünkü orada encoder launch-bound'du (bkz. §6.5);
TRT füzyonu kernel sayısını düşürünce darboğaz gerçek işe kaydı ve knob
çalışır hale geldi. **Aynı knob, farklı backend, tamamen farklı cevap.**

Buradaki ders, ağırlık byte'ını küçültmenin (INT8) değil **aktivasyon
byte'ını** küçültmenin (çözünürlük) bu modelde kazandırdığıdır.

### Hedef karşısındaki durum

- 1024'te en iyi: **47.5 ms** — 40 ms hedefinin üstünde.
- 512'de: **31.11 ms** — hedefin altında, ama doğruluğu doğrulanmadı.

512'nin bedava olmadığı yerler: RoPE tabloları 64×64 için eğitildi, biz
`q_sizes`'ı 32×32'ye override ediyoruz; maske çözünürlüğü düşüyor (ince
yapılar); memory bank 32×32 feature tutuyor, yani uzun takipte kayıp tek
kare IoU'sundan fazla olabilir. Etiketli veri gelince ilk ölçülecek şey bu.

Yan fayda: peak torch bellek 6970 MB → 1839 MB (3.8x), 500 karelik memory
bank feature haritasıyla birlikte küçüldüğü için.

### Hangi sayı 40 ms ile karşılaştırılır

Üç farklı sayı var ve karıştırmak kolay:

```
pre 36.38 ms   <- init_state icinde, loop'un disinda
--- propagate loop: 45.27 ms/kare = 22.08 FPS ---
    infer  31.11   <- 40 ms hedefi bu sayiya konduysa TUTTU
    post    0.32
    other  13.84   <- CPU: python dongusu, state bookkeeping, D2H kopya
```

- **inference = 31.11 ms** → hedefin altında.
- **kare süresi = 45.27 ms (22.08 FPS)** → hedef "kare başına 40 ms" ise 5 ms
  eksiğimiz var.
- **uçtan uca = 81.65 ms (12.2 FPS)** → preprocess dahil.

Fark tamamen `other`, yani GPU değil CPU. Bu yüzden bir sonraki iş TensorRT
değil, launch ve Python maliyetini kesmek.

### Encoder işi bitti

512'de encoder inference'ın **%10**'u. Kalan bütçe: mask decoder 10.97,
memory attention 7.25, memory encoder 5.43, ve `other` 13.84 ms. `other`
1024'te 12.3, 512'de 13.8 — image size ile ölçeklenmiyor, yani CPU tarafı
(Python döngüsü, state bookkeeping, threshold + D2H kopya). Encoder'ı daha
fazla optimize etmenin getirisi kalmadı; §7'deki iş listesi artık asıl iş.

## 6.8 "Max" pipeline: her şey açık

INT8 rafa kalktığına göre kalan tüm hızlanma kaynaklarını tek bir varyantta
topluyoruz. Yeni matris satırları:

| varyant | encoder | encoder graph | diğer bloklar |
|---|---|---|---|
| `trt_fp16` | TRT füzyon | yok | eager PyTorch |
| `trt_fp16_graph` | TRT füzyon | var | eager PyTorch |
| `pytorch_graph_all` | yok | var (torch) | **CUDA graph** |
| `trt_fp16_max` | TRT füzyon | var | **CUDA graph** |
| `trt_fp16_512_max` | TRT füzyon @512 | var | **CUDA graph** |

Mekanizma: `src/trackers/_cuda_graph.py`. Bir modülün `forward`'ını sarar,
girdi imzası (şekil + dtype + tensör olmayan argümanlar) başına bir graph
yakalar ve sonraki çağrılarda girdileri sabit tampona kopyalayıp `replay()`
eder. Yakalayamadığı her şeyde eager'a düşer.

İki tasarım kararı ölçümden geliyor:

- **`min_hits=2`:** bir imza ilk görüldüğünde yakalanmaz. EdgeTAM'in memory
  bank'i ilk birkaç karede büyür, o geçici şekillerin her biri bir kez
  görünür; onları yakalamak bütçeyi asıl kararlı şekil gelmeden tüketirdi.
- **Çıkışlar klonlanır, ama aynı nesne olanlar aynı nesne kalır.** Graph her
  replay'de aynı belleğe yazar ve EdgeTAM bir karenin feature'larını bir
  sonraki hesaplanırken elinde tutar; klonlamazsak kareler sessizce aynı
  belleği paylaşır. Kimlik korunmazsa da SAM2'nin `vision_features is
  backbone_fpn[-1]` beklentisi bozulur ve 16 MB iki kez kopyalanır.

Neden bu blokların TensorRT'ye taşınmasından önce deneniyor: encoder'da
ölçtüğümüz orana göre TRT füzyonunun CUDA graph üzerine kattığı ek kazanç
512'de 6.54 → 4.70 ms, yani 1.4x. Graph'in eager üzerine kattığı ise
33.33 → 6.54, yani 5.1x. **Kazancın büyük kısmı graph'ten geliyor, füzyondan
değil**, ve graph ONNX export gerektirmiyor. Kalan bloklar için de aynı
sıralamayı izliyoruz: önce ucuz olanı ölç, ihtiyaç kalırsa export et.

## 6.9 ÖLÇÜM SONUCU (Faz 6): graph kazandı, ama saatler kilitli değil

### Önce ölçüm hatası: aynı varyant üç farklı sayı verdi

`trt_fp16_512` üç ayrı oturumda, aynı engine ve aynı kodla:

| oturum | loop fps | infer ms | enc ms | dec ms |
|---|---|---|---|---|
| Faz 4 | 22.08 | 31.11 | 4.70 | 10.97 |
| Faz 5 | 19.99 | 35.11 | 5.55 | 12.54 |
| Faz 6 | 19.97 | 35.38 | 5.47 | 12.78 |

**%13 sapma, sabit bir TensorRT binary'siyle.** `trt_fp16` de aynı: Faz 4'te
encoder 19.00 ms, Faz 6'da 24.01 ms. Kod değişmedi, engine değişmedi.

**GPU saati değil.** `jetson_clocks` üç oturumda da açıktı. Sapmanın blok
dağılımı zaten saat düşüşüne uymuyor, çünkü saat düşüşü bütün GPU bloklarını
aynı oranda vurur:

| ölçüm | Faz 4 → Faz 6 | ne ile ölçülüyor |
|---|---|---|
| pre | +%1.5 | perf_counter (saf CPU) |
| memenc | +%3.5 | CUDA event |
| enc | +%16 | CUDA event |
| dec | +%16 | CUDA event |
| memattn | +%26 | CUDA event |

Tek biçimli değil. Sıralama "GPU matematiği" ile değil, **bloktaki küçük
kernel sayısı** ile örtüşüyor.

Sebep bu: `AttrTimer` GPU işini stream üzerine kaydedilen CUDA event'leriyle
ölçüyor. Event çifti iki event'in stream'de çalıştığı an arasını verir, yani
**CPU bir sonraki kernel'ı yetiştiremediği için GPU'nun boş beklediği süre de
bloğun içinde sayılır.** Launch-bound bir pipeline'da bu doğru davranıştır
(graph kazancını görünür yapan da budur), ama bir yan etkisi vardır: ölçülen
blok süresi o an CPU'nun kernel besleme hızına bağlıdır. CPU tarafında ne
değiştiyse (başka bir süreç, çekirdek göçü, GC baskısı) küçük kernel'lı
bloklara orantısız yansır.

Elimizdeki veriyle hangi CPU olayının sorumlu olduğu ayırt edilemiyor.
Ayırt edecek iki şey eklendi:

1. **`repeat %` sütunu:** aynı config'in tekrarları arasındaki en hızlı/en
   yavaş farkı, medyanın yüzdesi olarak. Bir varyant %5 öndeyse ama aynı
   config kendi içinde %10 oynuyorsa, o %5 gürültüdür. `frame sd` bunu
   söylemez, o tek run içindeki kare dağılımıdır.
2. **`--tegrastats`:** yavaş run'da GPU% düşük ve güç benzerse, GPU besleme
   sorunu doğrulanır.

Bağımsız olarak düzeltilen bir sıra biası daha vardı: eski harness bir
varyantın bütün tekrarlarını bitirip diğerine geçiyordu, yani listedeki ilk
varyant her zaman aynı makine durumunu alıyordu. Artık tekrarlar
**round-robin ve her turda kaydırılmış sırayla** koşuyor (`rotated()`).

**Kural: sadece aynı run içindeki satırlar karşılaştırılabilir.** Farklı
tablolardan sayı çekip yan yana koymak geçersiz.

### Faz 6 içi karşılaştırma (aynı run, geçerli)

**512 grubu:**

| | infer | enc | memattn | memenc | dec | other | loop fps |
|---|---|---|---|---|---|---|---|
| trt_fp16_512 | 35.38 | 5.47 | 9.17 | 5.62 | 12.78 | 14.41 | 19.97 |
| **trt_fp16_512_max** | **15.90** | 5.66 | **3.39** | **1.96** | **1.81** | 11.69 | **35.78** |

Blok bazında graph kazancı: **mask decoder 7.1x**, memory encoder 2.9x,
memory attention 2.7x. Inference 2.2x, kare süresi 50.01 → 27.93 ms.

**1024 grubu:**

| | infer | enc | memattn | memenc | dec | other |
|---|---|---|---|---|---|---|
| trt_fp16 | 59.02 | 24.01 | 12.19 | 7.92 | 11.78 | 12.40 |
| trt_fp16_graph | 61.23 | 25.24 | 12.45 | 8.51 | 12.01 | 13.01 |
| trt_fp16_max | 57.08 | 27.10 | 12.68 | 9.41 | **4.90** | 7.84 |
| pytorch_graph_all | 76.96 | 47.17 | 12.40 | 8.63 | **5.24** | 6.56 |

1024'te decoder yine 2.4x kazanıyor ama memory attention ve memory encoder
kazanmıyor. Encoder'ın 24.01 → 25.24 → 27.10 gidişi tam olarak çalışma
sırasını takip ediyor (bu run'da hâlâ blok sırayla koşuluyordu), yani graph'e
değil sıraya ait. Net kazanç %3, ve `repeat %` sütunu olmadan bunun gürültüden
ayırt edilebilir olduğu söylenemez. **1024 grubunun yeniden ölçülmesi
gerekiyor.**

### Kural: graph, bloğun GPU işi CPU launch maliyetinden küçükken kazandırır

Aynı blok, iki çözünürlük, iki farklı cevap:

| blok | 1024 kazanç | 512 kazanç |
|---|---|---|
| mask decoder | 2.4x | 7.1x |
| memory encoder | yok | 2.9x |
| memory attention | yok | 2.7x |

Girdi büyüdükçe kernel başına gerçek iş artıyor, launch payı küçülüyor,
graph'in kesecek bir şeyi kalmıyor. Bu, encoder'da §6.5'te bulunan kuralın
aynısı: **512 launch-bound, 1024 compute-bound.**

### Kalan bütçe ve full-TensorRT'nin tavanı

`trt_fp16_512_max`, kare 27.93 ms:

```
enc      5.66
memattn  3.39
memenc   1.96
dec      1.81
(diger)  ~3.1
--------------
infer   15.90
post     0.35
other   11.69   <- karenin %42'si, ve GPU degil CPU
```

Kalan blokları TensorRT'ye taşımanın 512'de üst sınırı bellidir: GPU işinin
tamamı sıfırlansa kare 27.93'ten ~12 ms'ye iner, gerçekçi kazanç 4-6 ms.
Graph zaten launch payını aldı, TRT'ye kalan sadece füzyon farkı (encoder'da
ölçülen oran 1.4x).

1024'te durum farklı: orada GPU blokları 57 ms ve graph çoğuna dokunamıyor,
yani full-TensorRT'nin asıl getirisi **1024'te**.

`other` 11.69 ms'ye ne TensorRT ne graph dokunuyor. Python döngüsü, EdgeTAM
state bookkeeping, threshold ve D2H kopya. Kare bütçesinde en büyük tek
kalem artık bu.

## 7. Encoder tavanını kırmak

TensorRT bugün sadece image encoder'ı kapsıyor. Faz 1 çıktısındaki
`encoder_ms / rest_ms` bölünmesi, geri kalanı kovalamanın değip
değmeyeceğini söyler:

```
Toplam hızlanma tavanı = 1 / (1 - encoder_payı)
```

`encoder_share_pct` %50 ise, encoder'ı sonsuz hızlandırsan bile uçtan uca
en fazla 2x alırsın. `tools/benchmark.py` artık her bloğu ayrı ölçüyor
(`enc / memattn / memenc / dec / other` kolonları), yani tavanı hangi
bloğun tuttuğunu tahmin etmiyoruz — okuyoruz.

Sıra, (kazanç / emek) oranına göre:

**1. Bedava olanlar (kod zaten yazıldı ya da tek satır)**
- Maske eşiği GPU'da alınıp `bool` transfer ediliyor (eskiden fp32 idi):
  kare başına cihaz→host trafiği 4x azaldı. `other_ms` kolonunda görünür.
- `use_cuda_graph: true` — TRT'nin kernel launch overhead'i. Orin'in CPU'su
  görece zayıf olduğu için burada x86'dan daha çok kazanç var.
- `offload_state_to_cpu: false` (64 GB unified memory'de gerek yok).

**2. TensorRT'siz, orta emek**
- `compile_image_encoder: true` (matriste `pytorch_compile`): aynı bloğu
  TRT'siz hızlandırma denemesi, karşılaştırma noktası olarak değerli.
- Aynı knob'u `memory_attention` ve `sam_mask_decoder` için de aç —
  `torch.compile` orada launch overhead'ini eritir. `src/trackers/
  edgetam_tracker.py` içinde `compile_image_encoder` ile aynı desende.
- Girdi çözünürlüğü 1024 → 512: encoder FLOP'u ~4x düşer. Quantization
  değil ama aynı hız/doğruluk eğrisinde bir nokta; IoU ile ölçüp tabloya koy.

**3. Daha fazla bloğu TensorRT'ye taşımak (asıl tavan kırma)**
Zorluk sırasına göre, çünkü hepsi aynı değil:

| Blok | Zorluk | Neden |
|---|---|---|
| `memory_encoder` | Kolay | Girdileri sabit shape'li (pix_feat + mask), tek yönlü |
| `sam_mask_decoder` | Orta | Batch = nesne sayısı → dinamik batch profili (`min=1, opt=1, max=8`) gerekir |
| `memory_attention` (2D Perceiver) | Zor | Memory bank uzunluğu kare indeksiyle büyüyor. Ya `min/opt/max` dinamik profil, ya da `num_maskmem` penceresini sabitleyip statikleştirmek |

Pratik yol: `memory_encoder` → `sam_mask_decoder` → `memory_attention`.
Her biri `tools/export_image_encoder_onnx.py`'nin aynı desenini izler
(dict yerine tuple döndüren bir wrapper + statik/profilli export), ve
`edgetam_trt_tracker.py`'deki `_patch_image_encoder` ile aynı şekilde
monkey-patch'lenir. Yani altyapı hazır; iş her bloğun gerçek imzasını
çıkarmakta.

Bunlara girmeden önce **Faz 1 ölçümünü görmek şart**: eğer `memattn_ms`
zaten küçükse, en zor bloğu export etmek boşa emek olur.

---

## 8. Komut referansı

```bash
# --- Faz 0
python3 scripts/jetson_doctor.py
sudo nvpmodel -m 0 && sudo jetson_clocks

# --- Faz 1
python3 tools/benchmark.py --video samples/synthetic_car.mp4 \
    --prompt-file samples/synthetic_car_box.json \
    --variants pytorch_fp32,pytorch_fp16,pytorch_bf16 \
    --dump-masks-dir runs/phase1 --out runs/phase1

# --- Faz 2
python3 tools/export_image_encoder_onnx.py --verify \
    --output models/edgetam_image_encoder.onnx

# --- Faz 3
for p in fp32 tf32 fp16 bf16; do
  python3 tools/build_trt_engine.py --onnx models/edgetam_image_encoder.onnx \
      --engine models/enc_$p.engine --precision $p
done

# --- Faz 4
python3 tools/build_trt_engine.py --onnx models/edgetam_image_encoder.onnx \
    --engine models/enc_int8.engine --precision int8 \
    --calib-video samples/synthetic_car.mp4 --calib-num 200 \
    --calib-cache models/enc_int8.calib

# --- Faz 3-5 ölçüm
python3 tools/benchmark.py --video samples/synthetic_car.mp4 \
    --prompt-file samples/synthetic_car_box.json \
    --variants all --dump-masks-dir runs/all --out runs/all

# --- Doğruluk
python3 tools/compare_masks.py runs/phase1/pytorch_fp32.npz runs/all/trt_int8.npz

# --- Rapor
python3 tools/benchmark.py --report runs/ --out runs/final_report.md
```
