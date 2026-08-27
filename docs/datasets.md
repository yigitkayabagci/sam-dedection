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

## Kontaminasyon: hangi model neyi görmüş

Bir seti **held-out değerlendirme** olarak kullanmak, o setin öğretmenimizin ya
da öğrencimizin eğitim verisinde **olmamasını** gerektiriyor. Birincil
kaynaklardan (makale tabloları, model kartları) çıkarıldı, çıkarımdan değil:

| model | MOSE gördü mü | kanıt |
|---|---|---|
| **SAM 2 / 2.1 yayınlanan checkpoint** | **temiz** | Makale §D.2.2: *"Yayınlanan modelimiz SA-V manual + Internal ve SA-1B ile eğitildi."* Tablo 17'nin `‡` satırları (MOSE val 71.8 / 73.5) GitHub README'deki yayınlanan checkpoint sayılarıyla birebir; MOSE **içeren** karışımın 76.6 / 77.9 satırları yayınlanmadı. İki bağımsız teyit: MOSEv2 kendi tablosunda `SAM2-L (ZS)` ile `SAM2-L`'yi ayırıyor; SAM 3 Tablo 5 SAM 2.1'in MOSEv2 sayısını `47.9†` = *zero-shot* işaretliyor |
| `sam2.1_hiera_b+_MOSE_finetune.yaml` | — | Sadece **örnek tarif**. `training/README.md`: *"checkpoint'lerimizi MOSE üzerinde ince ayarlamak için basit bir örnek sunuyoruz."* Yayınlanan checkpoint'in tarifi değil |
| **SAM 3** | **kontamine** | LaTeX Tablo 25: SA-Co/VIDEO-EXT medya havuzu = DAVIS2017, **MOSEv2**, YTVOS2019. Tablo 17: SA-Co/VIDEO-EXT **Train ✓**, aşama 2–4. Model kartı da listeliyor. Ayrıca makale SAM 2.1'in sayısını zero-shot işaretlerken SAM 3'ünkini işaretlemiyor |
| **DINOv3 / LVD-1689M** | **belirlenemez** | ~17 B Instagram görüntüsünden kümeleme + retrieval; ham açık set olarak sadece ImageNet-1k/22k ve Mapillary SLS adı geçiyor. MOSE ya da herhangi bir video benchmark'ı adı yok. LVD-1689M yayınlanmadı, indeksi yok → birebir kare kontrolü **imkânsız**. Etkisi düşük: DINOv3 donuk, aşama A'da sadece RGB-T çifti görüyor, MOSE etiketine hiç dokunmuyor |
| **EdgeTAM** — *bizim öğrencimiz* | **MOSEv1 gördü**, ama MOSEv2 val temiz | Makale §4.1: *"SA-V, SA-1B'nin %10'u, **DAVIS, MOSE, YTVOS** ile eğitiyoruz."* Ampirik bölme testi: MOSEv1-train (1 507) ⊂ MOSEv2-train, ve **MOSEv1-train ∩ MOSEv2-valid = 0** — 433 MOSEv2 val videosunun hepsi yeni |

**Pratik sonuç:** MOSEv2 val bugün hem öğretmenimiz (SAM 2.1) hem öğrencimiz
(EdgeTAM) için temiz. **Ama öğretmeni SAM 3'e çevirirsek bu biter** — SAM 3
MOSEv2'yi eğitimde görmüş. Değerlendirmeyi dürüst tutan şey SAM 2.1'de kalmak.

## MOSEv2 — sadece aşama C için

| | |
|---|---|
| ne | 5 024 video, 468 251 kare, 10 074 nesne, 701 976 maske, 200 kategori |
| modalite | **sadece RGB**, **yer seviyesi** — havadan/İHA/termal geçmiyor |
| indirme | HF, anonim, ranged GET **206**, gated değil. **84,8 GB** |
| lisans | **CC BY-NC-SA 4.0** — ticari kullanım yok |
| aşama A | **hayır** — RGB-only, öğrencinin termal girdisi yok |
| aşama B | **hayır** — havadan değil |
| aşama C | **evet, listedeki en iyi aday** |

Aşama C'ye uygun olmasının sebebi tam da hafıza yolunun var olma sebebi:
%61,8 kaybolma, %50,3 yeniden görünme, yoğun örtüşme, maskelerin %50,2'si
karenin %1'inden küçük.

İki operasyonel uyarı: **val bölümü sadece ilk karenin maskesini içeriyor**
(yerel J&F için CodaBench sunucusu ya da train'den ayrılmış bir dilim gerekiyor),
ve **train'in 1 507 videosu MOSEv1-train** — EdgeTAM onları zaten görmüş, yeni
sinyal değil.

## SAM2DV — elendi

IEEE DataPort'taki SAM2DV **yeni bir veri seti değil**, zaten indirdiğimiz
DroneVehicle üzerine bir **etiket katmanı**: 17 990 + 1 469 örnek, DroneVehicle
train/val bölümleriyle birebir. Yazar DroneVehicle'ın yönlü kutularını SAM2'ye
verip çıkanı kaydetmiş, kendisi de *"insan etiketi değil, makine üretimi bir
anotasyon kaynağı"* ve *"ikili (binary) segmentasyon türevi"* diyor.

Üç sebeple elendi:

1. **$40/ay IEEE DataPort aboneliği** arkasında. Sayfadaki dosya girdisi
   `href="#"`, istek atılacak bir uç nokta yok, S3 kovası 403 dönüyor.
   *"github.com/AirLab/SAM2DV'de yayınlandı"* iddiası **404**; bağlantılı
   `AirAI-Lab/SAM2DV` var ama **boş depo**.
2. Video değil → aşama C'ye yaramıyor.
3. **Zaten ikisi de bizde.** `tools/fetch_datasets.py` DroneVehicle'ın 28 442
   çiftini XML kutularıyla indiriyor, `tools/make_masklets.py` kutuları maskeye
   çeviriyor — üstelik **kalibrasyonla** (`--calibrate`), yani güvenmeden önce
   IoU'yu ölçerek. Kendi üretmek daha yeni bir öğretmen, kontrol edilebilir
   örnek ayrımı, kalibrasyon sayısı ve SAM2DV'nin atladığı 8 980 çiftlik test
   bölümü demek — bedeli bir çıkarım geçişi ve sıfır dolar.

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

## Havuz kaynakları — kutu etiketli setler (defter 13/14)

Yukarıdaki eleme ölçütü (2) — piksel maskesi — bu setlerde **bilerek**
aranmıyor: maske havuzu boru hattı (`docs/mask_pool_plan.md`,
`src/training/pool.py`) kutuyu güçlü bir öğretmene prompt olarak verip
maskeyi kendisi üretiyor. Ölçüt burada: (1) gerçekten indirilebilir mi,
(2) kutu yoğunluğu/kalitesi, (3) modalite.

| Set | Ne | İndirme (2026-08 doğrulandı) | Defter |
|---|---|---|---|
| [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) | 6 471 train havadan RGB, 10 sınıf, yoğun küçük hedef | Resmî dağıtım kişi-başı Drive linkleri (Colab'da kota); HF aynası `banu4prasad/VisDrone-Dataset` düz dosya, `snapshot_download` | 13 |
| [HIT-UAV](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset) | 2 898 termal 640×512, 24 899 kutu (insan/araba/bisiklet/diğer-araç), 80–130 m, CC-BY-4.0 | **Git reposu verinin kendisi** (~0,4 GB, codeload zip); COCO json `normal_json/annotations/` | 14 |
| [RGBT234](https://github.com/mmic-lcl/Datasets-and-benchmark-code) | 234 hizalı RGB-T video / 233,8 K çift; **modalite başına ayrı kutu** (`visible.txt` + `infrared.txt`) | Resmî: Baidu-only. HF aynası `xche32/rgbt234`: tek 7,67 GB tar.gz, range destekli | 14 (masklet) |
| [LasHeR](https://github.com/BUGPLEASEOUT/LasHeR) | 1 224 hizalı dizi / 730 K+ çift, aynı düzen | Resmî: Baidu/TeraBox. HF aynası `xche32/lasher`: 5 parçalı tar.gz **~224 GB** — Colab diskini aşar; fetcher akış halinde, dizi seçmeli açıyor (`--sequences`) | 14 (ek; varsayılan değil) |

Üç not:

1. **Aynalar üçüncü şahıs.** Defterlerin probe hücreleri geleni *sayarak*
   doğruluyor (kare/kutu/sınıf histogramı, RGBT234'te dizi sayısı ≥ 200
   asserti); raporlara öğretmenle birlikte kaynak da yazılıyor.
2. **AeroVIS çakışması** (yukarıdaki tabloya ek): 13'ün havuzu VisDrone'dan
   üretildiği için, bu havuzla eğitilen bir model AeroVIS'te **ölçülemez** —
   kare kare aynı veri. Temiz video değerlendirmesi UAVScenes/MVUAV'da kalır.
3. **DroneVehicle** iki defterde de kullanılabilir (RGB yarısı 13'te isteğe
   bağlı, termal yarısı 14'te varsayılan) — zaten `fetch_datasets.py`'deydi;
   OBB poligonları dik zarfa çevrilir, 100 px beyaz bant okuma anında kesilir.

```
python tools/fetch_datasets.py visdrone --dest /content/data/VisDrone
python tools/fetch_datasets.py hituav   --dest /content/data/HIT_UAV
python tools/fetch_datasets.py rgbt234  --dest /content/data/RGBT234
```

## İkinci tarama: hizalı havadan RGB-T ve termal, **örnek bazlı** etiketli

Bu bölümün ölçütü yukarıdakinden farklı — ve bilerek. Yoğun (semantik)
maske aranmıyor, çünkü havuz boru hattı (`docs/mask_pool_plan.md`) maskeyi
kutudan kendisi üretiyor. Aranan: (1) havadan, (2) hizalı RGB-T ya da salt
termal, (3) etiket **nesne bazlı** — kutu ya da örnek maskesi, sınıf haritası
değil, (4) forma/hesaba takılmadan indirilebilir.

Aşağıdaki her satır 2026-08'de **canlı sunucuda** doğrulandı: Drive
klasörlerinin merkezi dizini `Range` isteğiyle okundu (arşivi indirmeden içini
saymanın yolu), LILA'nın blob'ları ve Zenodo'nun API'si sorgulandı. Sayılar
makaleden kopyalanmadı; makale ile indirmenin uyuşmadığı iki yer aşağıda
ayrıca yazılı.

### Bağlandı — `tools/fetch_datasets.py` bu dördünü indiriyor

| Set | Modalite | Etiket | Doğrulanan indirme | Lisans |
|---|---|---|---|---|
| [RGBTDronePerson](https://nnnnerd.github.io/RGBTDronePerson/) (arşivde adı *WHU-DroneDual*) · ISPRS 2023 | hizalı havadan RGB-T, 640×512 | 6 125 çift, **70 880 kutu**, 4 sınıf (person / rider / crowd / uncertain); hem COCO json hem kare başına Pascal-VOC xml | Drive `18zm2Ca…`, **1 573 829 771 B**, onaylı URL 206 dönüyor (resume var). Zip'in içi `train/{annotation,thermal,visible}` + `val/{…}`: 4 900 + 1 225 çift, 12 250 jpg, 6 125 xml | CC BY 4.0 |
| [VTUAV-det](https://nnnnerd.github.io/RGBTDronePerson/) (aynı ekibin VTUAV yeniden etiketlemesi) | hizalı havadan RGB-T, **1920×1080** | 11 392 train + 5 378 test kare, **124 869 kutu**, aynı 4 sınıf | Drive `1TLmMOQ…`, **6 314 183 652 B**, 206. İçi `train/{anno,ir,rgb}` + `test/{anno,ir,rgb}` | (sayfa CC BY 4.0 diyor) |
| [BIRDSAI](https://lila.science/datasets/conservationdrones) · WACV 2020 | **salt termal** (LWIR), havadan, gece | 48 gerçek dizi / ~62 K kare, ~166 K kutu **+ iz kimliği**, MOT csv (`frame,id,x,y,w,h,class,species,occlusion,noise`) | LILA Azure blob, **2 271 707 323 B** (TrainReal) + **1 761 788 520 B** (TestReal), 206. TrainReal 32 dizi / 40 661 kare / 32 csv; TestReal 16 / 21 336 / 16 | **CDLA-Permissive-1.0** — listedeki tek ticari-serbest set |
| [AIResQ Benchmark](https://zenodo.org/records/17405074) · Sci. Data 2026 | salt termal, havadan (DJI) | 1 988 kare + 1 988 YOLO txt, tek sınıf (person) | Zenodo, **1 609 176 235 B**, yayıncı md5'i var, 206. İçi `Benchmark/{images,labels}` | CC BY 4.0 |

```
python tools/fetch_datasets.py rgbtdroneperson --dest /content/data/RGBTDronePerson
python tools/fetch_datasets.py vtuavdet        --dest /content/data/VTUAV_det
python tools/fetch_datasets.py birdsai         --dest /content/data/BIRDSAI
python tools/fetch_datasets.py airesq          --dest /content/data/AIResQ
```

Dördü birlikte ~87 K yeni kutu-etiketli kare getiriyor ve bunun ~79 K'sı
termal tarafta — havuz defteri 14'ün asıl ihtiyacı olan şey.

### Makale ile indirmenin uyuşmadığı iki yer

**AIResQ 9 788 kare değil, 1 988.** Makalenin öne çıkardığı 2048×1536'lık
6 506 + 3 282 görüntülük asıl set Zenodo'da `access_right: restricted` —
API boş `files` listesi döndürüyor, erişim için yazarlara başvurmak
gerekiyor. Açık olan tek kayıt 640×512'lik benchmark ve **içinde 1 988
görüntü var**. Küçük ama temiz; sadece adı vaat ettiği şey değil.

**NII-CU 9,1 GB değil, 26,2 GB.** Detay bir alttaki tabloda.

### Ölçülen iki sayı — ikisi de prompt kararını değiştiriyor

**RGBTDronePerson'da iki modalitenin kutuları 11,7 px ayrı.** Set "piksel
düzeyinde hizalı" diye tanıtılıyor, ama `sub_train_thermal.json` ile
`sub_train_visible.json` aynı 1 010 karenin aynı kişilerini iki kez
etiketliyor ve fark ölçülebiliyor: aynı sınıf, en yakın merkez, 40 px kapı ile
11 648 örneğin %94,5'i eşleşiyor, eşleşenlerin **merkez farkı medyan 11,7 px**
(p90 = 21,7 px), yalnız %15,4'ü 5 px içinde. Aynı dosyadaki hedeflerin medyan
büyüklüğü **√alan = 11–12 px**, yani fark bir hedef çapı kadar. Bu bir
kayıt hatası mı yoksa iki ayrı çizim mi — sonuç aynı: **termal kutu, görünür
karede kullanılabilir bir prompt değil.** Ayrıca hedeflerin %93'ü 16 px
altında; kutu-prompt'lu bir öğretmen için bu setin zoom-crop'suz hiçbir
şansı yok.

**VTUAV-det'te tek kutu kümesi iki modalitede paylaşılıyor.** Zip'teki xml
`<folder>rgb</folder>` diyor ve `train/00001.jpg` için (769,217)-(793,258)
veriyor; yanındaki `train_ir.json` aynı kare için `[768,216,24,41]` veriyor —
kapsayıcı/dışlayıcı sınır farkı dışında **aynı dikdörtgen**. Yani etiket bir
modalitede çizilip diğerine olduğu gibi taşınmış. VTUAV'ın iki yarısının
piksel piksel hizalı olmadığı zaten biliniyordu (bkz. `docs/encoder_mimari.md`
§3); VTUAV-det bunu düzeltmiyor, **sessizce devralıyor**. Buna karşılık
hedefler kutu-prompt için yeterince büyük: medyan √alan train'de 69 px,
val'de 48 px — RGBTDronePerson'daki 11 px'in yanında başka bir dünya.

Pratik sonuç: **termal havuzun prompt kaynağı VTUAV-det, RGBTDronePerson
değil.** RGBTDronePerson'ın değeri prompt değil, defter 14'ün kalibrasyon
tablosuna "modaliteler arası kutu transferi ne kadar bozuyor" sorusunun
ölçülmüş bir cevabını koymak.

### Doğrulandı ama bağlanmadı — ve neden

| Set | Ne | Engel |
|---|---|---|
| [NII-CU MAPD](https://www.nii-cu-multispectral.org/) · J. Field Robotics 2022 | 5 880 hizalı RGB (3840×2160) + FIR (640×512) çift, 20–50 m, 45° eğim, lens distorsiyonu düzeltilmiş + homografi ile bindirilmiş, kişi kutusu, CC BY-NC-SA 3.0 | Dağıtım bir **Dropbox klasörü**. Dosya bazlı link (`…/NII_CU_MAPD_dataset.zip?rlkey=…&dl=1`) *"No Access"* dönüyor; klasör linki `dl=1` ile çalışıyor ama iki arşivi birden **26 212 695 407 B tek akış** olarak veriyor — `Range` yok, resume yok, üyeler akış halinde (boyutları başta bilinmiyor), üstelik istemediğimiz 15,2 GB'lık ham videoyu da içeriyor. 9,1 GB'lık asıl arşivi tek başına almanın bir yolu bulunamadı |
| [Anti-UAV-RGBT](https://huggingface.co/datasets/CornBac0n/Anti-UAV-RGBT) | 318 eşli video: dizi başına `visible.mp4` + `infrared.mp4` + modalite başına json, ayrıca `label_new/{train,val,test}.json`; HF aynası 6,7 GB, anonim | Bakış **yerden göğe** — hedef gökyüzündeki İHA, havadan sahne değil. Projede zaten Anti-UAV410 var |
| [WiSARD](https://sites.google.com/uw.edu/wisard/) · IROS 2022 | 15 453 eşzamanlı görsel-termal çift (toplam ~56 K etiketli görüntü), YOLO kutusu, tek sınıf (human), termal 640×512, izin verici lisans, 40,54 GB tek Drive dosyası | Çiftler yalnız **zamansal** senkron. Makale görsel-termal kaydı (registration) *çözülmemiş problem* olarak sunuyor — hizalı RGB-T sayılmaz |
| [RGBT-Tiny](https://github.com/XinyiYing/RGBT-Tiny) · TPAMI 2025 | 115 hizalı dizi / 93 K kare / **1,2 M kutu + iz kimliği**, 7 sınıf, 8 sahne tipi — listedeki en zengin RGB-T kutu seti | Erişim **form arkasında** (Google/Microsoft Forms). Otomatik indirilemiyor; istenirse elle alınıp `staged()` yoluna konabilir |
| [VTSaR](https://github.com/zxq309/VTSaR) · TGRS 2024 | Hizalı görünür-IR havadan kişi seti (A-VTSaR) + sentetik yarısı (AS-VTSaR) | **Baidu-only** (`pan.baidu.com`, kod `qqru`); Colab yanıtlayamaz, HF aynası bulunamadı |
| [AVIID-1/2/3](https://github.com/silver-hzh/Averial-visible-to-infrared-image-translation) · J. Remote Sensing 2023 | 3 363 hizalı havadan görünür-IR çift, 434×434 ve 512×512 | **Baidu-only**; ayrıca görev görüntü çevirisi — **kutu etiketi yok** |
| [VEDAI](https://huggingface.co/datasets/ckyrkou/vedai) | Havadan araç tespiti, yönlü kutu | Modalite RGB + **NIR**, termal değil; HF aynası 52 MB, yayının tamamı değil |

### Yer seviyesi olduğu için elenenler

LLVIP, M3FD, KAIST-multispectral, CVC-14, CAMEL — hepsi hizalı RGB-T ve
kutu etiketli, HF aynaları da var (`Frencis/LLVIP_RGBT`, `Frencis/M3FD_RGBT`,
`nonameplease/M3FD_Detecion`), ama hiçbiri havadan değil. Aynı sebeple MVSeg
zaten yukarıda "MVUAV'ın yer seviyesi kardeşi" olarak duruyor.
`Frencis/NIICU_RGBT` havadan, ama 129 GB'lık sıkıştırılmamış TIFF dökümü —
resmî 9,1 GB'lık arşivin yanında anlamsız.

### Kutuyu maskeye çevirmiş hatlar

Sorunun ikinci yarısı — "bazıları pipeline kurup kutuyu maskeye çevirmiş
olabilir" — üç örnekle karşılandı, üçü de aynı deseni izliyor:

| Hat | Girdi → çıktı | Durum |
|---|---|---|
| **AeroVIS** | VisDrone + UAVDT + SeaDronesSee kutuları → SAM 3 → örnek maskesi + kimlik izi | Yukarıda öncelikli listede; **defter 13'ün havuzuyla eğitilen model burada ölçülemez** (aynı VisDrone kareleri) |
| **SAM2DV** | DroneVehicle yönlü kutuları → SAM 2 → ikili maske | Yukarıda gerekçesiyle elendi: IEEE DataPort aboneliği arkasında, video değil, ve aynısını `make_masklets.py` kalibrasyonla üretiyor |
| [**UAVDB**](https://arxiv.org/html/2409.06490v6) | yörünge **noktası** → Patch Intensity Convergence ile kutu → SAM 2 → örnek maskesi | Deseni doğruluyor ama bize uymuyor: RGB, ve hedef yine gökyüzündeki İHA |

Üçünün ortak dersi bu repodaki karara denk düşüyor: kutuyu maskeye çeviren
tarafın elinde kalibrasyon yoksa çıktı denetlenemez bir etiket katmanı
oluyor. `src/training/pool.py`'nin dört kapısı ve `--calibrate` yolu tam da
bunun için var.

### Çakışma riski — yeni satırlar

- **VTUAV → VTUAV-det.** Aynı çekim. Defter 07/08 VTUAV'da eğitiyorsa
  VTUAV-det bir **değerlendirme** seti olamaz; prompt kaynağı olarak
  kullanılabilir, ölçüm için kullanılamaz.
- **RGBTDronePerson, BIRDSAI, AIResQ, NII-CU** listedeki hiçbir setle kaynak
  paylaşmıyor — dördü de kendi çekimi.
- **Anti-UAV-RGBT ↔ Anti-UAV410.** İkisi de Anti-UAV yarışma ailesinden;
  410'un termal dizileri RGB-T sürümündekilerle örtüşebilir. İkisi birden
  kullanılacaksa dizi adları karşılaştırılmalı.

## VTUAV takip bölümü — DroneVehicle'in tarifi buraya taşınmıyor

Maske bölümü (100 video, `vtuav_vis`) zaten kullanılıyor. Bu bölüm **takip**
arşivleri hakkında: 500 dizi, 1920×1080 RGB-T, dizi başına tek hedef.
2026-08'de proje sayfasının Drive linkleri üzerinden ölçüldü — merkezi
dizinler `Range` isteğiyle okundu, kutu dosyaları tek tek indirildi.

### Üç şey, ölçülerek

**1. "RGB sürümü" ayrı bir veri değil, aynı baytlar.** `train_ST_001.zip`'i
iki Drive klasöründen de aldım: örneklenen her RGB üyesi **CRC32 ve
sıkıştırılmış boyutta birebir eşleşti**. RGB arşivi = RGB-T arşivinden `ir/`
ve `ir.txt` silinmiş hâli. İkisini birden indirmek parça başına ~9 GiB saf
tekrar. **Yalnız RGB-T sürümü indirilmeli**; RGB hasadı aynı çıkarımın
`rgb/` yarısını okur.

**2. Kareler 1/10 etiketli.** Dizi düzeni `<dizi>/rgb/000000.jpg`,
`<dizi>/ir/…`, `<dizi>/rgb.txt`, `<dizi>/ir.txt`. Kare kimlikleri 0..n-1
boşluksuz ve kutu dosyaları 20 dizinin **yirmisinde de** `ceil(n/10)` satır
taşıyor → **satır k = kare 10k**. `train_ST_001`: 20 dizi, 37 419 kare
çifti, **3 750 etiketli satır**, 15,4 GiB. Yani arşivin %90'ı bu boru hattı
için ölü ağırlık; `fetch_datasets.tracked_members` yalnız etiketli kareleri
ve tek modaliteyi açıyor (~0,8 GiB).

**3. DroneVehicle'ın paylaşılan-kutu numarası burada işlemiyor.** Aynı 3 750
satırda:

| | |
|---|---|
| `rgb.txt` ile `ir.txt` **birebir aynı** satır | **%12,2** (DroneVehicle'da %53,0) |
| merkez farkı | medyan **8,38 px**, p75 15,8, p90 29,4, maks 94,2 |
| 5 px içinde | %30,5 |
| hedef büyüklüğü | medyan √alan **76,7 px**, %98,5'i ≥32 px |
| yalnız-bir-modalitede satır | **0** (rgb-only 0, ir-only 0) |

Son satır yapısal: hedef tek bir fiziksel nesne, iki dosyada da etiketli ya
da hiçbirinde. Yani VTUAV'da ne **aynalanacak** maske var (%88 satırda
kutular farklı) ne de **only** hasadı. Doğru kurgu: her modalite kendi
karesinde, kendi kutusuyla, ayrı bir öğretmen geçişi — ve tam da bu yüzden
iki defter (16, 17) **aynı anda iki runtime'da** koşabiliyor.

Not: 8,38 px'lik fark 77 px'lik bir hedefte ~%11; RGBTDronePerson'daki
11,7 px'in 12 px'lik hedefte olması ile karıştırılmamalı. VTUAV'ın kayması
görecelı olarak çok daha küçük — ama yine de "tek maske iki modaliteye"
demek için yeterli değil.

### Parçalar dizi adına göre alfabetik — bu bir tuzak

| parça | boyut | dizi | ~etiketli satır | nesne türleri |
|---|---:|---:|---:|---|
| `train_ST_001` | 15,42 GiB | 20 | 3 750 | animal, bike, bus |
| `train_ST_005` | **7,32 GiB** | 20 | 1 931 | car, elebike |
| `train_ST_008` | 13,07 GiB | 28 | 4 054 | pedestrian (28'in 24'ü), car |
| `train_ST_011` | 11,57 GiB | 10 | 3 034 | car, pedestrian, truck |
| `train_LT_001` | 16,49 GiB | 12 | 5 617 | bus, car |
| `train_LT_004` | 13,91 GiB | 9 | 4 051 | pedestrian, truck |

Ardışık parça almak iki kategorilik bir havuz demek. Defterlerin `ARCHIVES`
varsayılanı bu yüzden bir **serpme**: `ST_001 + ST_005 + ST_008 + ST_011` =
47,4 GiB → ~12 800 etiketli kare, animal/bike/bus/car/elebike/pedestrian/truck.
LT yarısı zaten yalnız dört parça, o yüzden 24 ve 25 hepsini alıyor ve soru
ortaya çıkmıyor.

Train yarısının tamamı **214,5 GiB** (11 ST + 4 LT parça). `fetch_datasets.py`
tarifinde (`vtuav_track`) her parça **varsayılan olarak kapalı**; id'ler ve
gerçek boyutlar orada, ama hiçbiri kazara inmiyor.

### LT ile ST arasındaki fark, bu boru hattı açısından

LT = long-term, ST = short-term: ayrım **takip görevinin** türü, verinin değil.
Aynı sensör, aynı 1920×1080 hizalı RGB-T çiftler, aynı
`<dizi>/{rgb,ir}/` + `<dizi>/{rgb,ir}.txt` düzeni. Bu boru hattı hiçbir
takip protokolü okumadığı için ayrım büyük ölçüde anlamsız — **iki gerçek
sonucu** dışında:

**1. LT'de hedef kadrajdan çıkıyor, ve o satırlar öyle işaretli.** Bir kutu
satırı sıfır/negatif genişlik-yükseklik ya da NaN taşıyorsa hedef görünmüyor
demektir. `boxes.vtuav_frames` bu satırları **düşürüyor** — sıfır alanlı ya da
sayı-olmayan bir dikdörtgen, promptable bir öğretmene verilebilecek en
faydasız şey. Her iki yazım da ele alınıyor: `nan <= 0` **False** olduğu için
yalnız genişliği sınamak NaN'i öğretmene prompt olarak sızdırıyordu.

**2. Adım (stride) LT'de ölçülmüş değil.** "Satır k = kare 10k" `train_ST_001`
üzerinde doğrulandı, başka hiçbir yerde değil — ve bir eksik varsayılan adım
gürültüyle patlamıyor, sessizce **kare 9'u kare 10'un kutusuyla** etiketleyip
Drive'a yanlış maskeler yazıyor. Bu yüzden sayı artık her dizinin kendi kare
ve satır sayısından türetiliyor (`boxes.annotated_stride`): `ceil(kare/adım) ==
satır` kısıtı gerçek uzunlukta bir dizide tek bir cevap bırakıyor (2 000 kare +
200 satır yalnız 10'u kabul eder), 10 o cevaplar arasındaysa tercih ediliyor —
yani ST hasadı bit bit aynı kalıyor — ve **hiçbir tek cevap çıkmayan dizi
sayıları basılarak atlanıyor**. Aynı fonksiyonu üç yer de kullanıyor: arşivi
açan `tracked_members`, indeksleyen `vtuav_frames`, ve defterlerin 2. hücresi
(sonda) — yani prob ne yazdırıyorsa koşu onu yapıyor.

Defterlerin 2. hücresi bunun için var: 15 GiB açılmadan **önce** her arşivin
merkezi dizinini okuyup dizi başına kare, satır, o iki sayının ima ettiği adım
ve hedefi "yok" işaretleyen satır oranını basıyor. Merkezi dizin dosyanın
sonunda, yani saniyeler.

### Bu bölümde maske yok — hiç

VTUAV'ın çizilmiş instance maskeleri **ayrı** bir yayın: VIS bölümü, 100 video.
Takip bölümünün 500 dizisi (ST + LT) kare başına tek bir `x y w h` taşıyor ve
başka hiçbir şey. 16/17/24/25'in var olma sebebi tam olarak bu: kutuyu maskeye
çeviren şey öğretmen.

## DroneVehicle — tam sayım (evet, kullanıyoruz; evet, kutu)

Kısa cevabı: **evet, üç yerde kullanılıyor** — aşama A'da hizalı çift kaynağı
(`SPECS["dronevehicle"]`), defter 14'te termal havuzun varsayılan kutu
kaynağı, defter 13'te RGB yarısı `INCLUDE_DRONEVEHICLE` ile isteğe bağlı. Ve
etiketi **kutu**, üstelik **yönlü kutu** (oriented box), semantik maske değil.

Aşağısı tahmin değil: `train.zip`'in ZIP64 merkezi dizini `Range` isteğiyle
okundu, **35 980 XML'in tamamı** (iki modalite × 17 990 kare) indirilip
ayrıştırıldı, dört kare de gerçekten kod çözülüp piksel kontrol edildi.
2026-08.

### Ne var

| | |
|---|---|
| kaynak | HF aynası `McCheng/DroneVehicle`, `train.zip` = **8 880 004 598 B**, 206 (resume var), ZIP64 |
| içerik | `train/{trainimg, trainimgr, trainlabel, trainlabelr}` — dördü de **17 990** dosya → 17 990 hizalı RGB-TIR çift, **iki modalite de ayrı ayrı etiketli** |
| kare | 840×712, dört yanda 100 px bant → **içeride 640×512** (projenin native çözünürlüğü) |
| etiket tipi | VOC-vari XML, `<polygon>` `x1..y4` = **yönlü kutu**. 293 495 poligon + **22 917 düz `<bndbox>` (%7,2)** — iki biçim de gerçek trafik taşıyor |
| hacim | termal **316 412** kutu, RGB **286 794** kutu (yalnız train; makalenin 953 087'si üç bölmenin toplamı) |
| sınıflar | car 270 350 · truck 15 833 · bus 11 334 · **freight car 11 192** · van 7 701 |

### Ölçek — sorduğun kısım

Kutu büyüklüğü, √alan olarak (100 px bant çıktıktan sonraki 640×512 karede):

```
p05 27,2   p10 30,9   medyan 44,2   p90 68,6   p95 83,9   maks 300  px
<16 px: %0,03      16–32 px: %11,8      >=32 px: %88,1
```

Sınıf bazında medyan / p90: car 42,4 / 59,0 · van 47,0 / 80,0 ·
freight car 66,8 / 119,5 · truck 67,3 / 122,3 · bus 85,2 / 125,1.

Bu, listedeki **en iyi prompt ölçeği**. Karşılaştırma: RGBTDronePerson'da
medyan 11 px (%93'ü 16 px altında), VTUAV-det'te 48–69 px. DroneVehicle'da
hedeflerin **%88'i 32 px'in üstünde** — kutu-prompt'lu bir öğretmenin
zoom-crop'a en az muhtaç olduğu set bu.

Bir uyarı, yönlülükten geliyor: kutuların **%92,8'i döndürülmüş**, ve SAM'ın
prompt kodlayıcısı açıyı ifade edemediği için eksen-hizalı zarf
promptlanıyor. Zarf, gerçek döndürülmüş dikdörtgenin **medyan 1,46 katı**
alanını kaplıyor (p75 2,12×, p90 2,37×) — yani prompt sıkılığında ödenen
sabit bir bedel var. Kapılar maskeyi kendi genişliğine göre ölçtüğü için bu
doğruluğu değil, sıkılığı etkiliyor; `box_iou` eşiği ayarlanırken akılda
tutulmalı.

### İrtifa — kısmen filtrelenebilir

Uçuş klasörü adlarının bir kısmı irtifayı ve gimbal açısını **adında
taşıyor**: `310_80m45night1`, `39_120m45_2`, `319_120m_30_3`,
`321_100_15_3`. Tam sayım:

| | |
|---|---|
| irtifa/açı kodlayan kare | **2 243 / 17 990 (%12,5)** |
| irtifa | 120 m: 955 · 80 m: 673 · 100 m: 615 |
| gimbal | 45°: 1 087 · 30°: 866 · 15°: 290 |
| gece etiketli | 328 |
| kalan %87,5 | jenerik adlar: `image` (3 556), `A` (2 199), `modalB` (1 114), `B` (834), `5-20`, `6-5`… |

Yani makalenin "80–120 m, 15/30/45°, gündüz+gece" ifadesi doğru, ama
**kare bazında irtifaya göre bölmek yalnız sekizde bir için mümkün.** İrtifa
ekseninde kontrollü bir deney isteniyorsa bu 2 243 kare o deneyin veri
kümesi; geri kalanı irtifası bilinmeyen havuz.

### Hizalama — gerçekten kayıtlı

İki yarı ayrı ayrı etiketlenmiş ama bağımsız değil. Termal kutuyu aynı
sınıftan en yakın RGB kutusuyla eşleştirdim (40 px kapı, 17 990 karenin
tamamı):

| | |
|---|---|
| byte-birebir aynı dikdörtgen | **%53,2** (168 253 kutu) |
| 40 px içinde eşleşen | %89,4 toplam |
| eşleşenlerin merkez farkı | medyan **0,00 px** · p75 3,9 · p90 8,1 · p95 11,6 |
| 10 px içinde | %93,2 |
| **RGB karşılığı hiç olmayan termal kutu** | **%10,6 — 33 383 kutu** |

Son satır setin var olma sebebi: termalin gördüğü, görünürün göremediği
araçlar. Ayrıca 512 RGB karesi tamamen boşken yalnız 33 termal kare boş.
**Sonuç: hasat termal yarıdan yapılmalı**, ve `--prompt pair` (maskeyi RGB'de
üretip termalde kapılamak) burada RGBTDronePerson'ın aksine gerçekten
savunulabilir — orada iki modalitenin kutuları medyan 11,7 px ayrıydı,
burada 0.

### Beyaz bant ve kanal sayısı — ikisi de ölçüldü

Dört kare (iki modalite × iki uçuş) indirilip kod çözüldü: hepsi 840×712,
100 px bandın **%99,3–99,9'u tam 255** (kalanı JPEG çınlaması), içeride
640×512. `DatasetSpec.border = 100` ve `inset()` doğru.

Termal jpg'lerin bir kısmı **üç kanallı** geliyor (17 990 termal XML'in
1 346'sı `<depth>3</depth>` diyor) — okuyucular `IMREAD_UNCHANGED` + gri
dönüşüm kullandığı için bu zaten karşılanıyor, ama kendi loader'ını yazan
biri buna takılır.

### Paylaşılan kutular — defter 15'in beslendiği alt küme

Yukarıdaki "%53,2 byte-birebir aynı" satırı bir gözlem değil, bir alt küme.
Ölçütü sıkılaştırıp (zarf değil, **sekiz poligon koordinatının tamamı** artı
sınıf) tam sayım yapıldığında:

| | |
|---|---|
| iki etiket dosyasının aynı poligonu yazdığı kutu | **167 644 / 316 412 (%53,0)** |
| bu kutuları taşıyan kare | **13 129 / 17 990 (%73,0)** |
| kare başına | medyan 4, ortalama 9,3, p90 25 |
| büyüklük | medyan √alan **43,8 px** (tüm termal kutularda 44,2) — boyutça yanlı bir örnek değil |
| sınıflar | car 141 220 · truck 9 010 · bus 7 409 · freight car 5 842 · van 4 163 |

Bu kutularda iki modalitenin anlaşmazlığı **ölçümle değil, inşa gereği
sıfır** — etiketçi aynı koordinatları iki kez yazmış. İki yarı da kayıtlı
olduğu için RGB piksellerinden üretilen maske, aynı koordinatlarda termal kare
için de doğru maske. `boxes.dronevehicle_shared_frames` bu alt kümeyi
`image = RGB`, `pair = termal` olarak veriyor (`dronevehicle_frames`'in
tersi, bilerek) ve `pool.label_pool(..., mirror=...)` tek öğretmen
geçişinin çıktısını iki havuza birden yazıyor: termal havuzun bedeli bir
dosya kopyası.

### Yalnız-bir-modalitede kutular — üçüncü ve dördüncü havuz

Paylaşılan alt kümenin tamamlayıcısı tek parça değil, iki farklı parça.
Termal kutuların **~%36'sının** RGB tarafında biraz farklı çizilmiş bir
karşılığı var (aynı araç, başka dikdörtgen) — bunları ayrıca etiketlemek tek
nesneye iki maske koymak olur, o yüzden **hiçbir hasada girmiyorlar.** Geri
kalanın ise karşılığı **hiç yok**: diğer dosyada aynı sınıftan, merkezi 40 px
içinde bir kutu bulunmuyor.

| | termal-only | rgb-only |
|---|---:|---:|
| kutu | **33 383 (%10,6)** | **3 797 (%1,3)** |
| taşıyan kare | 4 420 (%24,6) | 2 146 (%11,9) |
| kare başına | medyan 0, ortalama 1,9, p90 5 | medyan 0, ortalama 0,2, p90 1 |
| medyan √alan | 45,7 px | 36,7 px |
| ≥32 px | %87,0 | %65,6 |
| sınıflar | car 26 911 · freight 2 607 · truck 2 293 · bus 988 · van 584 | car 3 284 · van 160 · truck 148 · freight 129 · bus 76 |

Asimetri **dokuz kat**: termal kameranın gördüğü ve görünürün kaçırdığı araç
sayısı, tersinin dokuz katı. Gece. Setin var olma sebebi tam olarak bu sayı.

`boxes.dronevehicle_only_frames(root, modality=...)` bu kutuları veriyor.
**Prompt yalnız kendi modalitesinde** olabilir (`prompt="self"`, `mirror` yok):
diğer yarı hedefi zaten göstermiyor, oraya promptlamak öğretmenden o
koordinatlarda ne varsa onu segmentlemesini istemek olur. `pair` referans
olarak taşınıyor, prompt kaynağı olarak değil.

Böylece havuz dört klasör oluyor ve dördü de ayrı duruyor —
`dronevehicle_rgb` + `dronevehicle_thermal` (paylaşılan, tek maske iki dosya),
`dronevehicle_thermal_only`, `dronevehicle_rgb_only`. Tek klasöre karıştırmak
aynalanmış bir maskeyi termalde promptlanmış birinden ayırt edilemez kılardı;
kayıtların `prompt` alanı (`self` / `mirror`) bu ayrımı taşıyor.

**Üç bölme de aynı düzeni kullanıyor.** `val.zip` (723 321 423 B, 1 469 çift)
`val/{valimg,valimgr,vallabel,vallabelr}`, `test.zip` (4 430 343 899 B,
8 980 çift) `test/{testimg,…}` — merkezi dizinlerden okundu. Toplam
17 990 + 1 469 + 8 980 = **28 439**, yayının sayısı. Okuyucular `*img` /
`*imgr` / `*label` / `*labelr` kalıbına baktığı için üçü de ek kod
istemiyor; `ARCHIVES`'a eklemek yetiyor.

**Kaggle'daki YOLOv11-OBB sürümü bu iş için kullanılamaz.** (Kaggle API'sinden
okundu: `redzapdos123/dronevehicle-dataset-yolov11-obb`, 1,04 GB,
CC BY-NC-SA 4.0.) Üç sebeple: bölme başına **tek düz `images/` klasörü** var —
RGB ile termal ayrılmıyor, dolayısıyla eşleştirilemiyor; **etiket görüntü
başına tek** dosya, yani "iki modalite aynı poligonu yazmış mı" sorusu
sorulamıyor; ve beş sınıf **ikiye indirilmiş** (`small-vehicle`,
`large-vehicle`). YOLO-OBB eğitimi için düzgün bir paket, havuz kaynağı için
değil. Bu boru hattının ihtiyacı `trainimg`/`trainimgr` + `trainlabel`/
`trainlabelr` dördünü birden taşıyan orijinal arşiv.

### Bulunan tek gerçek kusur: sınıf adı iki yazımda

`feright_car` **7 419** kutu, `feright car` **3 773** kutu — aynı sınıf, iki
yazım. Ayrıca bir `feright`, bir `truvk` (RGB tarafında) ve bir `*`. Elle
dokunulmasa `class_histogram` aynı aracı iki satırda raporluyor ve ada göre
süzen herhangi bir kod **freight car'ların üçte birini** sessizce kaybediyor.
`boxes.DRONEVEHICLE_ALIASES` bunları tek ada indiriyor; `*` bilerek
dokunulmadan bırakıldı, ki probe onu göstersin.

## Mevcut referans

[Anti-UAV410](https://github.com/HwangBo94/Anti-UAV410) (2023 · TPAMI) —
410 termal video, 438 K kutu, 640×512. Projede zaten kullanılıyor;
kutu etiketli olduğu için J&F üretemez, bu yüzden raporun veri seti
bölümünde yer almıyor (kod tarafındaki kullanımı için bkz.
`docs/new_report.md`).
