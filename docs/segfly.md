# SegFly

ECCV 2026. Fraunhofer IVI + TU Münih. `docs/datasets.md`'de aşama B'nin hacim
kaynağı olarak listeleniyor; bu dosya o satırın arkasındaki inceleme.

Buradaki her sayı **indirilen veriden ölçüldü**, yayından okunmadı: HF'den 761
parquet shard'ın 16'sı (3,1 GB) çekildi, iki modaliteyi, iki kamerayı ve üç
irtifayı da kapsayan **747 kare** açıldı. Yayınla çelişen yerler ayrıca
işaretlendi.

## 1. Ne çözüyor

Havadan RGB-T semantik segmentasyonda üç darboğaz var: elle etiketleme pahalı
olduğu için setler küçük kalıyor, çeşitlilik dar, ve hazır dronelarda RGB ile
termal donanımsal olarak senkron olmadığı için hizalama tutmuyor.

SegFly'in önerisi bir veri seti değil, bir **üretim yöntemi**: `2D-3D-2D`.

1. RGB karelerin **%2,84'ü** — 20 606 karenin **586'sı**, hepsi 50 m'de —
   elle etiketleniyor.
2. Bu etiketler SfM/MVS ile kurulan **semantik 3B nokta bulutuna** kaldırılıyor.
3. Bulut bütün görüntülere geri yansıtılıyor → RGB etiketlerinin %97'si ve
   termal etiketlerin %100'ü otomatik üretiliyor.

Aynı geometri hizalama için de kullanılıyor: RGB ve termal dizileri **ayrı ayrı**
3B'ye kaldırılıyor, iki bulut **ICP** ile çakıştırılıyor, sonra RGB kareler
termal kameranın bakış açısına warp ediliyor. Donanım senkronu yok.

Yazarların bildirdiği doğruluklar: RGB etiket %91, termal etiket %88, kayıt
(registration) %87,05.

**Asıl katkı budur** — 35 613 kare değil, 586 elle çizilmiş maskeden 35 613
kare üretebilen boru hattı. Veri seti o boru hattının çıktısı ve kanıtı.

Kaynak görüntüler kendi çekimi değil: **OccuFly** (CVPR 2026, aynı grup, havadan
semantik sahne tamamlama). SegFly onun ham uçuşlarını + 586 elle etiketi
kullanıyor. `docs/datasets.md`'deki çakışma tablosunda bu yüzden "türetilmiş"
işaretli.

## 2. Nasıl dağıtılıyor

| | ölçülen |
|---|---|
| satır | **35 613** (20 606 RGB + 15 007 termal) |
| boyut | **191,44 GB**, 761 parquet shard |
| split | **tek `train` split** — val/test yok |
| lisans | CC BY-NC-SA 4.0 (ticari kullanım yok) |

Sütunlar: `image`, `label`, `RGB_aligned`, `scene`, `altitude`, `modality`.

**İlk tuzak:** `load_dataset("markus-42/SegFly")` tek bir `train` split döndürür
ve **içinde test sahneleri de vardır**. Bölme `scene` sütunundan elle
kurulmalı, yoksa test üstünde eğitmiş olursun:

| modalite | train | val | test |
|---|---|---|---|
| RGB | scene 01–05 | 06, 07 | 08, 09 |
| termal | scene 03–05 | — | 09 |

**İkinci tuzak:** GitHub'daki klasör ağacı (`RGB/scene_01/30m/images/...`) HF'de
**yok**; orada sadece parquet var. `tools/export_hf_dataset.py` bu yüzden var.

## 3. Semantik veri nasıl saklanıyor

`label`, **8-bit tek kanallı PNG** (PIL `mode=L`), paletsiz. **Piksel değeri
doğrudan sınıf ID'sidir.**

Ölçüm, `scene_05/40m/000000.png`:

```
mode=L  size=(4000,3000)  palette=None
uniques = [0, 3, 7, 8]      →  %5,1 unlabeled · %1,0 dirt · %79,0 vegetation · %14,9 tree
```

Sınıf ID'leri **bitişik değil**, 0–36 aralığında boşluklu — 15 sınıf + ignore:

| ID | sınıf | | ID | sınıf |
|---:|---|---|---:|---|
| 0 | Unlabeled (ignore) | | 13 | **Vehicle** |
| 1 | Road | | 14 | Water |
| 2 | Walkway | | 16 | Building |
| 3 | Dirt | | 17 | Roof |
| 4 | Gravel | | 33 | Parking Lot |
| 6 | Grass | | 34 | Construction |
| 7 | Vegetation | | 36 | **Truck** |
| 8 | Tree | | | |
| 9 | Ground Obstacle | | | |

Bu tablo `src/training/aerial.py`'deki `SPECS["segfly"]` ile birebir aynı.
**Bitişik varsaymak bu depoda bir kez yapıldı ve `things` grass/vegetation/
tree/ground_obstacle'ı takip hedefi seçti** — 6,7,8,9 gerçekte manzara.

### 747 karede ölçülen sınıf dağılımı

| ID | sınıf | kaç karede | piksel payı |
|---:|---|---:|---:|
| 6 | grass | 648 | %29,87 |
| 8 | tree | 536 | %12,97 |
| 7 | vegetation | 695 | %11,64 |
| 33 | parking_lot | 377 | %7,93 |
| 16 | building | 343 | %7,15 |
| 17 | roof | 439 | %5,48 |
| 2 | walkway | 424 | %4,86 |
| 3 | dirt | 557 | %4,84 |
| 1 | road | 281 | %3,95 |
| 14 | water | 86 | %2,72 |
| 4 | gravel | 330 | %2,24 |
| 9 | ground_obstacle | 450 | %2,05 |
| 0 | unlabeled | 110 | %2,03 |
| **13** | **vehicle** | **265** | **%1,21** |
| 34 | construction | 399 | %0,64 |
| **36** | **truck** | **226** | **%0,33** |

İki "şey" sınıfı toplam piksellerin **%1,54'ü**. Set ezici çoğunlukla
manzaradır; araç ince bir kenar durumudur.

## 4. Asıl soru: car1 / car2 / car3 diye erişebilir miyim?

**Hayır.** SegFly saf semantik segmentasyondur. Dosyada araç kimliği yok, ne
ayrı bir kanalda, ne ayrı bir dosyada, ne de metadata'da.

Bir otopark karesinde ölçüm:

```
araç bölgesindeki farklı değerler: [13]
```

Tek skaler. Sahnedeki bütün arabalar aynı 13'tür — "her araba araba".
Yazarların kendi Firefly modeli de semantik çıktı veriyor, örnek değil.

### Connected component ile kurtarma ve sınırı

Tek çare aynı sınıfın bağlı bileşenlerine ayrılması — bu depoda
`src/training/aerial.py::decompose` bunu yapıyor. **Kısmen çalışıyor.**

50 m'den bir otopark karesi (`scene_04/50m`): sağlam görünen 46 bileşen,
ama en büyükleri tek araba olamayacak kadar iri. 50 m'de tek araba ~300×150 px
(~38 000 px); ölçülen bloklar 312×484 ve 110 510 px — yani **yan yana park
etmiş üç araba tek bloğa kaynamış**.

En kötü örnek `scene_09/30m`: gözle sayılabilen **altı araç** (bir panelvan +
beş otomobil) **tek bileşen** oluyor, 301 761 px.

**Bu blok `InstanceGates`'in dört kapısının dördünü de geçiyor:**

```
alan          = 301 761 px
bbox          = 469 x 813
fill          = 0,791   (kapı 0,25'in altını atıyor)
kare oranı    = 0,0251  (kapı 0,9'un üstünü atıyor)
→ hepsini geçer
```

`fill` kapısı "iki nesne ince bir köprüyle birleşmiş" durumunu yakalamak için
konmuş. **SegFly'deki baskın füzyon biçimi o değil:** sıralı park etmiş
araçlar kompakt, bbox'ını iyi dolduran bir blok üretir. Kapı bunu göremez.

### Füzyonun büyüklüğü

Her karenin EXIF'inden irtifa + kamera okunup GSD hesaplandı, oradan tek
aracın beklenen piksel alanı çıkarıldı (4,5 × 1,9 m). 4 818 bileşen üstünde:

| bileşen | sayı | pay | araç piksellerinin |
|---|---:|---:|---:|
| 1 araç | 4 427 | %91,9 | %79,1 |
| ~2 kaynamış | 142 | %2,9 | %9,1 |
| 3–4 kaynamış | 160 | %3,3 | %7,4 |
| 5+ kaynamış | 89 | %1,8 | %4,4 |

**Bileşenlerin %8,1'i çok-araçlıdır ve araç piksellerinin %20,9'unu tutar.**
4 818 bileşene karşılık tahmini 6 138 gerçek araç → **bir-araç-bir-blok
isabeti ~%78,5**.

Yani örnek *sayısının* dörtte biri değil ama örnek *pikselinin* beşte biri
yanlış hedefte. Ve yanlış olanlar büyük, göze çarpan, otopark sahneleridir —
promptlanması en olası olanlar.

### `watershed` modu ne kadar kurtarıyor

`decompose(..., mode="watershed")` köprüleri ayırmayı deniyor. Ölçüm:

| kare | components | watershed |
|---|---|---|
| scene_04/50m | 44 örnek, en büyük 110 510 | 52 örnek, en büyük 62 494 |
| scene_09/30m | 25 örnek, en büyük 301 761 | 29 örnek, en büyük 179 502 |

İyileştiriyor ama kapatmıyor: 179 502 px hâlâ ~dört araç. Kısmi bir onarım,
çözüm değil.

### Sonuç

SegFly'i **hacim** için kullan (aşama A distilasyonu etiket okumaz, orada
35 613 karenin hepsi sayar). **Örnek ayrımı** gereken yerde tek başına yetmez;
`docs/datasets.md`'nin VTUAV-VIS'i "listedeki tek örnek maskeli havadan RGB-T
set" diye işaretlemesinin sebebi tam olarak budur.

## 5. İndirince çıkan, hiçbir yerde yazmayan şeyler

### 5.1 Palette dışı ID'ler — ve bu depodaki bir açık

Yayınlanan tablo 16 ID veriyor. Gerçek veride **22** var. Fazladan olanlar:

| ID | kaç karede | ne |
|---:|---:|---|
| 5 | 252 | Rock — 9'a haritalanmalıydı |
| 12 | 84 | Bicycle — 0'a haritalanmalıydı |
| 21 | 49 | Cable — 0'a |
| **35** | **46** | **Crane — 0'a** |
| 11 | 21 | Person — 0'a |
| **10** | **1** | isimsiz |

Yani yayıncının kendi yeniden haritalaması sızdırmış. Hiçbiri "şey" sınıfı
olmadığı için **eğitilen şeyi değiştirmiyor**; ama palet kontrolünü yanıltıyor.

`SPECS["segfly"].ignore` bu depoda `(0, 5, 11, 12, 21)` idi — **35 ve 10
eksikti**, çünkü liste yalnızca termal export'tan okunmuştu ve ikisi de RGB
diliminde yaşıyor. Ölçülen histogram `verify()`'a verildiğinde:

```
!! 2 value(s) not in the palette: [10, 35]
```

Tam da `ignore` listesinin engellemek için var olduğu yanlış alarm. Bu
incelemede `(0, 5, 10, 11, 12, 21, 35)` olarak düzeltildi ve
`tests/test_export_hf.py` ölçülen histogramla birlikte pinlendi.

### 5.2 Termal görüntüler ham radyometrik veri taşıyor

`image`, termal tarafta 640×512 **MPO** (`n_frames=2`), `mode=RGB` — ama üç
kanal birebir aynı, yani DJI'nin `WhiteHot` paletiyle render edilmiş 8-bit gri.

**Asıl veri JPEG'in APP3 (`0xE3`) segmentlerinde duruyor:**

```
APP3 yükü = 655 360 bayt = 640 × 512 × 2   (uint16, little-endian)
render ile korelasyon = 0,924
```

Bu DJI R-JPEG ham termal dizisi. Değerler ~18 400–19 700 aralığında ve
basit bir K×10 / K×100 dönüşümüne oturmuyor; mutlak sıcaklık için DJI Thermal
SDK'nın kalibrasyon parametreleri gerekiyor. Yine de **8 bit yerine ~14 bit
dinamik aralık** demek.

`load_dataset()` ile okuyan herkes bunu kaybediyor — PIL sadece 0. kareyi
verir. Termal encoder'a girdi seçilirken bilinmesi gereken bir şey.

### 5.3 Kare başına tam uçuş metadatası

Her JPEG'de EXIF + DJI XMP duruyor:

```
GpsStatus=RTK  RtkFlag=50  GpsLatitude=+48.840660  GpsLongitude=+11.348389
AbsoluteAltitude=+506.477  RelativeAltitude=+29.983
GimbalPitchDegree=-70.00  GimbalYawDegree=+20.70  GimbalRollDegree=+0.00
FlightYawDegree=+21.30  UTCAtExposure=2025-08-22T10:07:00.247066
DroneModel=M3T  CameraSerialNumber=5L4SM3Q02AB09V
```

RTK GPS + gimbal açıları, yani kare başına 6-DoF poz pratikte mevcut.
Konum Bavyera (Manching/Ingolstadt civarı), çekimler 2025 Nisan–Ağustos.
`RelativeAltitude` klasör adını doğruluyor (30m → +29,983 / +30,040 ...).

### 5.4 İki farklı kamera, iki farklı çözünürlük

Proje sayfası "DJI Phantom 4 RTK and Mavic 3 Enterprise Series UAVs" diyor,
ama hangi sahnenin hangisiyle çekildiğini ve çözünürlüklerin farklı olduğunu
hiçbir yerde söylemiyor. Ölçüm:

| sahne | kamera | RGB çözünürlük |
|---|---|---|
| 03, 04, 05, 09 | DJI **M3T** (Mavic 3 Thermal, WideCamera) | 4000×3000 |
| **07** | DJI **FC6310R** (Phantom 4 RTK) | **5472×3648** |

scene_07 bir **val** sahnesi. Yani doğrulama bölümü, eğitim bölümünden farklı
bir kamerayla ve farklı çözünürlükte çekilmiş. Termal sadece M3T'den geliyor
(640×512, InfraredCamera).

### 5.5 Kareler nadir değil

"Havadan" genelde tepeden-aşağı varsayılır. Gimbal pitch dağılımı:

| sahne / irtifa | RGB | termal |
|---|---|---|
| scene_03 30m | — | −70° |
| scene_03 40m | −75° | −75° |
| scene_04 30m | — | −90° |
| scene_04 50m | −90° | −75° |
| scene_05 30m | −90° | −70° |
| scene_05 40m | −90° | −75° |
| scene_07 50m | −75° | — |
| scene_09 30m | **−70°** | — |
| scene_09 40m | — | −90° |
| scene_09 50m | — | −75° / −90° |

**−70° ile −90° arasında değişiyor.** scene_09 30m RGB (−70°, eğik) bir
**test** sahnesi. Nadir varsayan bir ön işleme (kare artırma, ölçek priori,
homografi) burada sessizce yanlış olur.

## 6. Yeniden üretim

```bash
pip install huggingface_hub pyarrow pillow numpy scipy
```

```python
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq, io, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# ~500 MB'lık tek bir shard yeter; termal shard'lar ~60 MB (indeks 80, 200, 320…)
p = hf_hub_download("markus-42/SegFly", "data/train-00560-of-00761.parquet",
                    repo_type="dataset", local_dir=".")
row = pq.read_table(p).to_pylist()[37]

lab = np.array(Image.open(io.BytesIO(row["label"]["bytes"])))
print(lab.dtype, np.unique(lab))            # uint8, boşluklu sınıf id'leri
print(np.unique(lab[lab == 13]))            # -> [13]: örnek kimliği yok
```

Füzyonu görmek için:

```python
from scipy import ndimage
veh = lab == 13
cc, n = ndimage.label(veh, structure=np.ones((3, 3)))
areas = ndimage.sum(veh, cc, range(1, n + 1))
print(n, sorted(areas)[-5:])                # ~46 bileşen, en büyüğü ~110 000 px
```

Ham termal (termal bir shard'da):

```python
b = row["image"]["bytes"]; app3 = bytearray(); i = 2
while i < len(b) - 1:
    if b[i] != 0xFF: break
    m = b[i + 1]
    if m in (0xD8, 0xD9): i += 2; continue
    ln = int.from_bytes(b[i + 2:i + 4], "big")
    if m == 0xE3: app3 += b[i + 4:i + 2 + ln]
    if m == 0xDA: break
    i += 2 + ln
raw = np.frombuffer(bytes(app3[:640 * 512 * 2]), "<u2").reshape(512, 640)
```

## 7. Bağlantılar

- Kod / doküman: https://github.com/markus-42/SegFly
- Proje sayfası: https://markus-42.github.io/publications/2026/segfly/
- arXiv: https://arxiv.org/abs/2603.17920
- Veri seti: https://huggingface.co/datasets/markus-42/SegFly
- Firefly modeli: https://huggingface.co/markus-42/SegFly-Firefly
  (DINOv3 ViT-B/16 + MLP head; termal varyantta Rein adapter; girdi 640;
  15 sınıf, **semantik** çıktı)
- Kaynak set OccuFly: https://markus-42.github.io/publications/2026/occufly/
  (havadan semantik sahne tamamlama; **semantik voxel grid**, örnek yok —
  yani örnek etiketi oradan da gelmiyor)

Görev tanımında verilen Google Drive bağlantısı
(`18VG2UUA6tzkjw5XYHSiQQ5w6zF2YCroC`) bu oturumdan açılamadı: anonim istek
Google giriş sayfasına düşüyor, bağlı Drive hesabında da dosya görünmüyor
("Requested entity was not found"). Yukarıdaki her şey HF ve GitHub
kaynaklarından ölçüldü.
