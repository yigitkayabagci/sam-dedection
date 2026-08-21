# Datasets

Aday veri setleri — **incelendi**. Ayrıntılı değerlendirme, karşılaştırma
tablosu ve her set için kısa notlar raporda:
`report/bolumler/veri_setleri.tex` (Bölüm 5, "Veri Setleri ve
Değerlendirme").

Eleme ölçütü: (1) gerçekten indirilebilir ve açık mı, (2) yoğun (piksel
düzeyinde) maske GT'si var mı, (3) havadan mı, (4) video mu. Gerekçe:
J&F metriği yoğun maske GT'si ister; Anti-UAV410'un kutu etiketleri onu
veremez.

## Öncelikli

| Veri seti | Yıl · Yayın | Neden |
|---|---|---|
| [Kust4K](https://figshare.com/articles/dataset/_b_Kust4K_b_b_b_b_A_Large-scale_Multimodal_UAV_Dataset_for_Robust_Urban_Traffic_Scenes_Semantic_Segmentation_b_/29476610) ([makale](https://www.nature.com/articles/s41597-025-05994-7)) | 2025 · Sci. Data | 4 024 hizalı RGB-TIR çifti (2 514 gündüz / 1 510 gece), **640×512** — projenin native çözünürlüğü, **9 sınıf**. Palet arşivin kendi `visual.py`'sinden okundu ve gerçek indirmeye karşı uçtan uca doğrulandı; makaleden tahmin edilen eski palet **yanlıştı** (aşağıya bak) |
| [MVUAV](https://jiwei0921.github.io/MVUAV) ([NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/78e839f96568985d18463044a064ea0f-Abstract-Conference.html)) | 2024 · NeurIPS | 413 havadan RGB-T video, 53 828 kare, 2 183 etiketli kare, 36 sınıf, gündüz/gece |
| [Caltech Aerial RGB-T](https://github.com/aerorobotics/caltech-aerial-rgbt-dataset) ([CaltechDATA](https://data.caltech.edu/records/cks6g-ps927)) | 2024 · ECCV | **Yayınla indirme uyuşmuyor, indirilen doğrulandı.** İki arşiv: `labeled_rgbt_pairs.zip` (4,29 GB) → **2 282 kayıtlı çift, 960×600**, stereo rektifiye, termal EO çerçevesine yansıtılmış, ölçülen artık kayma **1–4 px** — listedeki **en iyi hizalanmış** RGB-T. `labeled_thermal_singles.zip` (4,14 GB) → 3 076 maske, termal 640×512, ama içindeki `color/` **kayıtlı eş değil** (819×512, ~%5'i tamamen siyah) → `rgb=None`. Makale 4 195 maske diyor, **yayın 3 076 veriyor**. Örnek kaynağı olarak zayıf: `decompose` 3 076 maskeden 1 357 örnek çıkarıyor, medyan alan 145 px. Değeri **aşama A** ve doğal arazi/su/gece alanı. CC BY-NC-SA |
| [AeroVIS / AeroTrack](https://github.com/Dmygithub/AeroTrack) | 2026 · arXiv | 117 video / 49 204 kare, **örnek maskesi + kimlik izi** (8 279 iz) — EdgeTAM'ın çıktı tipine en yakın set |
| [UAVScenes](https://github.com/sijieaaa/UAVScenes) | 2025 · ICCV | 23 uçuş dizisi, **120 K karenin her biri** elle etiketli, 2448×2048, 19 sınıf — listedeki en yoğun zamansal etiketleme |
| [SegFly](https://github.com/markus-42/SegFly) ([HF](https://huggingface.co/datasets/markus-42/SegFly)) | 2026 · ECCV | 20 606 RGB + **15 007 hizalı RGB-T çifti**, termal 640×512, 30/40/50 m irtifa. Termal sadece sahne 3, 4, 5, 9'da. **Aşama B'nin hacim kaynağı** — Kust4K'nın ~4 katı kare. İki uyarı: (1) HF'de **parquet** olarak dağıtılıyor, klasör değil → `tools/export_hf_dataset.py`; (2) sınıf id'leri **boşluklu** ve "şey" sınıfı sadece ikisi: `vehicle=13`, `truck=36` — insan/otobüs/motosiklet sınıfı **yok** |
| [MVSeg](https://jiwei0921.github.io/Multispectral-Video-Semantic-Segmentation/) | 2023 · CVPR | 738 RGB-T video, 3 545 maske, 26 sınıf — MVUAV'ın yer seviyesi kardeşi, aynı biçim |
| [VTUAV (DUT-VTUAV)](https://zhang-pengyu.github.io/DUT-VTUAV/) | 2022 · CVPR | 500 dizi / ~1,7 M hizalı 1920×1080 RGB-T kare; **100 videoluk maske bölümü** ayrı indirilebilir, J&F için yeterli. **Encoder çalışmasının ana kaynağı, iki ayrı sebeple.** Aşama A: distilasyon etiket okumuyor, o yüzden 1,7 M çiftin tamamı sayıyor — listedeki diğer setlerin toplamından fazla. Aşama B: **VIS bölümü listedeki tek örnek (instance) maskeli havadan RGB-T set**, yani `decompose`'un semantikten yeniden kurmaya çalıştığı şeyin kendisi (`mode="labels"`). Uyarı: iki modalite piksel piksel hizalı değil ve maskelerin bir kısmı propagate edilmiş. Bkz. `docs/encoder_mimari.md` §3, §8 |

## İkincil

| Veri seti | Yıl · Yayın | Kısıt |
|---|---|---|
| [VDD](https://github.com/RussRobin/VDD) | 2023 · arXiv | 4000×3000, 50–120 m, kentsel+endüstriyel+kırsal üçünü birden kapsıyor; ama 400 statik RGB görüntü |
| [MESSI](https://github.com/messi-dataset/messi-dataset) | 2025 · TMLR | 2 525 görüntü, 5472×3648, 16 sınıf. Değeri hacmi değil **irtifa ekseni**: aynı sahne 30/50/70/100 m + 120→10 m iniş dizileri |
| [SkyScapes](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/dlr-skyscapes) | 2019 · ICCV | Listenin en yoğun etiketi: 31 sınıf, 13 cm/px, 5616×3744. Ama **16 görüntü** — eğitim seti değil, ince/küçük yapı stres testi (görüntü başına ~77 adet 512² pencere) |
| [FlyAwareV2](https://github.com/LTTM/FlyAwareV2) | 2026 · SPIC | Büyük kısmı CARLA sentetiği, termal yok, tam sürüm ~296 GB |
| [Multi-View UAV](https://huggingface.co/datasets/Peter341/Multi-View-UAV-Dataset) | 2025 · arXiv | 357 690 kare ama 400×300 (EdgeTAM'ın 512 girdisinin altında) ve tamamı CARLA |

## Ortak kaynaklar — çakışma riski

On üç setin altısı kendi çekimi değil, daha eski açık setlerin yeniden
etiketlenmiş hali. Detay: raporda Bölüm 5.4.

| Set | Kaynak |
|---|---|
| AeroVIS | VisDrone, UAVDT, SeaDronesSee (tamamı türetilmiş; kutular SAM3 ile maskeye çevrilmiş) |
| FlyAwareV2 | gerçek: VisDrone (eğitim) + UAVid (test), ~2 K kare · sentetik: CARLA/SynDrone ~288 K kare |
| VDD | kendi 400 görüntüsü özgün; birlikte gelen IDD paketi UDD ve UAVid'i yeniden etiketliyor |
| MVSeg | OSU, INO, KAIST, RGBT234 |
| SegFly | OccuFly (CVPR 2026) — ham görüntünün tamamı + elle çizilmiş 586 etiket |
| UAVScenes | MARS-LVIG (SLAM veri seti) — ham uçuş verisi; UAVScenes pozları ve 120 K etiketi ekliyor |

Kendi çekimi: Kust4K, MVUAV, Caltech Aerial RGB-T, VTUAV, MESSI, SkyScapes
(Multi-View UAV = CARLA sentetiği).

**Çakışan ikililer:**

- **VisDrone → AeroVIS + FlyAwareV2** — tek gerçek kare düzeyinde risk;
  bu ikisi eğitim/değerlendirme çifti olarak kullanılamaz.
- **UAVid → FlyAwareV2 (test) + VDD/IDD** — IDD'de eğitip FlyAwareV2
  testinde ölçmek aynı kareleri iki kez görmek.
- **CARLA → FlyAwareV2 + Multi-View UAV** — kare değil simülatör ortak;
  sızıntı yok ama aynı render imzası.
- **MARS-LVIG → UAVScenes** — listedeki başka hiçbir set kullanmıyor,
  çakışma yok.
- **OccuFly → SegFly** — listedeki başka hiçbir set OccuFly kullanmıyor,
  bugün çakışma yok; ama OccuFly değerlendirmeye alınırsa SegFly tarafsız
  olmaktan çıkar.
- **RGBT234 → MVSeg** — bugün zararsız; ileride RGBT234'te takip
  ölçülürse örtüşür.

**Sonuç:** öncelikli sekizin dördü türetilme (AeroVIS, UAVScenes, SegFly,
MVSeg) ama kaynak aileleri tamamen ayrık → **öncelikli sekizi birlikte
kullanılabilir.**
Güvenli kurulum: Kust4K + MVUAV'da eğitip J&F'i AeroVIS'te ölçmek.

## Elenenler

- [UAV-VisLoc](https://github.com/IntelliSensing/UAV-VisLoc) — GT = GPS
  koordinatı; maske ya da kutu yok, görev görsel konumlandırma.
- [CPD-UAV](https://www.mdpi.com/2504-446X/10/6/447) — 1 061 görüntü,
  yalnızca kutu, video yok.
- [CrossLoc](https://crossloc.github.io/) (CVPR '22) — 7 000+ görüntü ama
  görev görsel konumlandırma, semantik etiketler sentetik render'dan,
  çözünürlük 720×480 (EdgeTAM'ın 512 girdisinin altında).
- `https://www.scilit.com/publications/d95b5f3cf21eb74fa105e75f47956c1a`
  — sayfa erişime kapalı (HTTP 403), hangi yayına işaret ettiği
  doğrulanamadı.

**Not:** Ham listedeki figshare bağlantısı ile Nature Scientific Data
makalesi **aynı veri setidir** (Kust4K); tek satıra indirildi.

## İndirme: hepsi otomatik

`tools/fetch_datasets.py` üç setin de bağlantısını içinde taşıyor; Drive'a elle
arşiv koymak gerekmiyor. Notebook 07 ve 08 bu dosyayı çağırıyor.

```
python tools/fetch_datasets.py kust4k    --dest /content/data/Kust4K
python tools/fetch_datasets.py vtuav_vis --dest /content/data/VTUAV_VIS
python tools/fetch_datasets.py segfly    --dest /content/data/SegFly
```

Her bağlantı **canlı sunucuda** doğrulandı, çünkü üçü de "bariz" yöntemi
kıracak şekilde sunuluyor:

| Set | Bariz yöntem neden çalışmıyor | Ne kullanılıyor |
|---|---|---|
| **Kust4K** | Makale sayfasındaki `ndownloader/articles/29476610/versions/3` bağlantısı **HTTP 202 + boş gövde** dönüyor: figshare zip'i *o an üretmeye başlıyor* ve dosya var olmadan yanıt veriyor. İndirici sıfır baytlık dosya yazıp "başarılı" diyor | Dosya bazlı uç noktalar (`ndownloader.figshare.com/files/<id>`) — hazır, range isteğine 206 dönüyor, yayıncının md5'i ile geliyor. Sadece termal isteniyorsa 1,66 GB'lık `RGB.zip` atlanabiliyor |
| **VTUAV** | Maske bölümü bir Drive **klasörü** ve içindeki zip'ler 8–17 GB. Birkaç yüz MB'ın üstünde Drive dosyayı değil, "virüs taraması yapılamıyor" formunu sunuyor — üstelik 200 koduyla. `curl` bu HTML'i diske yazıyor | `gdown`, formu yeniden gönderiyor |
| **SegFly** | HF'de parquet, klasör değil — spec'in glob'ları hiçbir şey bulamıyor | `tools/export_hf_dataset.py` PNG düzenine çeviriyor |

Arşivler indirildikten sonra siliniyor (tepe disk yarıya iniyor), indirmeler
kaldığı yerden devam ediyor, md5 olan yerde doğrulanıyor.

**Kust4K arşivleri düz:** `TIR.zip`, `Seg_annos.zip` ve `RGB.zip`'in üçü de
aynı 4 024 dosya adını en üst seviyede taşıyor. Yan yana açılsalar birbirlerini
ezerlerdi; fetcher her birini kendi klasörüne (`tir/`, `label/`, `rgb/`) açıyor.

**VTUAV maske bölümü** sekiz zip: `training/train_001..003` ve
`test/test_001..005`, toplam ~120 GB. Varsayılan sadece `train_001` (8,5 GB) —
14 dizi, 26 059 kayıtlı çift, 875 RGB maskesi. Zip'in içi:
`bike_009/rgb/000000.jpg`, `bike_009/mask/rgb/000000.png`; maskeler
**modalite başına ayrı klasörde** ve her 30. karede bir, değerler `{0, 255}`.

## İki palet yanlıştı — ikisi de aynı sebeple

Hem Kust4K hem SegFly için palet önce makaleden tahmin edilmişti ve ikisi de
yanlış çıktı. Sonuç ikisinde de aynı: **`things` yanlış sınıfları seçiyor**,
yani model bir prompt'a çalıyla cevap vermeyi öğreniyor.

| | Tahmin | Gerçek | Sonuç |
|---|---|---|---|
| **Kust4K** | 3 = vegetation varsayıldı | Böyle bir sınıf yok; id 3 = **motorcycle** | Üstündeki her id bir kaydı, id 8'e yer kalmadı. `things` id 6'yı (= **tree**) aldı, id 3'ü (**motorcycle**) kaçırdı |
| **SegFly** | id'ler ardışık varsayıldı | id'ler **boşluklu** | Grass / Vegetation / Tree / Ground Obstacle takip hedefi oldu |

Kust4K'nın doğru paleti (arşivdeki `visual.py` → `get_palette()`, dizideki sıra
= sınıf id'si):

```
0 unlabelled   1 road    2 building   3 motorcycle   4 car
5 truck        6 tree    7 human      8 traffic_facilities
```

`things = (motorcycle, car, truck, human)` → id `{3, 4, 5, 7}`.

Gerçek indirmede doğrulandı: 4 024 karenin hepsi eşleşiyor, 60 örneklenen
haritada palet dışı **tek bir değer yok**, dört "şey" sınıfının dördü de
mevcut. Ayrıca frekanslar da paleti destekliyor — id 6 kırk kareden 40'ında
görünüyor, ki bu *tree*'ye uyuyor, *truck*'a değil.

Ders: **paleti ezberden yazma.** İkisi de makul görünen tahminlerdi.
`probe_classes` indirilen maskelerde gerçekten hangi değerlerin olduğunu
söylüyor; notebook'ların bunun için bir hücresi var.

## Mevcut referans

[Anti-UAV410](https://github.com/HwangBo94/Anti-UAV410) (2023 · TPAMI) —
410 termal video, 438 K kutu, 640×512. Projede zaten kullanılıyor;
kutu etiketli olduğu için J&F üretemez, bu yüzden raporun veri seti
bölümünde yer almıyor (kod tarafındaki kullanımı için bkz.
`docs/new_report.md`).
