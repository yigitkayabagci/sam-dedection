# RGB havadan kaynaklar — obje bazlı etiket (kutu / örnek maskesi)

`docs/datasets.md`'nin ana listesi **yoğun semantik maske** arıyordu (J&F
ölçütü). Bu doküman tam tersini arıyor: **obje bazlı** etiket — ya kutu, ya
örnek (instance) maskesi. Sebep, maske havuzu boru hattı
(`docs/mask_pool_plan.md`, `src/training/pool.py`): kutuyu güçlü bir
öğretmene prompt olarak verip maskeyi kendimiz üretiyoruz, dolayısıyla
"maskesi yok" bir eleme sebebi değil — **kutusu yok** eleme sebebi.

Ölçüt: (1) RGB, (2) havadan, (3) obje bazlı etiket, (4) betikle indirilebilir.

Aşağıdaki her indirme **2026-08-25'te canlı uca karşı** kontrol edildi.
Doğrulama yöntemi ve ne bulunamadığı §8'de.

---

## 1. Hedef bant: ~20–60 m, iri nesne — ve ölçülen sıralama

Proje bilerek **yüksek irtifadan / uydudan** çalışmıyor ve **tiny nesne**
hedeflemiyor. Bu, listeyi irtifaya göre değil **nesnenin kaç piksel
göründüğüne** göre sıralamayı gerektiriyor — ve ikisi aynı şey değil.

Aşağıdaki tablo tahmin değil: her satır o setin **kendi etiket
dosyalarından** örneklenerek hesaplandı (kutu setlerinde kutu alanı,
maske setlerinde bağlı bileşen alanı; `ignored` sınıfı ve 100 px²
altındaki artıklar atıldı).

| Set | irtifa | çözünürlük | **medyan √alan** | **< 32 px** | < 16 px | kare başına | karar |
|---|---|---|---:|---:|---:|---:|---|
| **AU-AIR** | 2,8–30,6 m | 1920×1080 | **81 px** | **%11,8** | %1,3 | 3 | **✓ en iri** |
| AeroVIS `sd_` dilimi | yüksek, deniz | çeşitli | 84 px | %21,1 | %0,1 | — | ✓ ama 2 sınıf, su |
| **CODrone 60 m** | 60 m | 3840×2160 | **60 px** | %20,4 | %2,3 | 39 | **✓** |
| **CODrone 30 m** | 30 m | 3840×2160 | 47 px | %28,9 | %2,4 | 21 | **✓** |
| CODrone 100 m | 100 m | 3840×2160 | 44 px | %30,2 | %4,4 | 26 | sınırda |
| **DroneVehicle — RGB** *(zaten repoda)* | 80–120 m · dik + 15/30/45° | 840×712 (→640×512) | **40 px** | **%19,0** | — | 12 | **✓ sınırda ama geçer** — §4.3 |
| AeroScapes | 5–50 m | 1280×720 | 35 px | %40,0 | %22,9 | 1–2 | zayıf: seyrek + düşük çözünürlük |
| VisDrone2019-DET *(bugün havuzda)* | yüksek | ≤1920×1080 | 32 px | **%51,0** | %17,5 | 56 | **✗ bant dışı** |
| **AeroVIS bütünü** | — | çoğu ≤1080p | **31 px** | **%52,1** | %10,6 | — | **✗ bant dışı** |
| UAVid | 50 m **eğik** | 3840×2160 | 28 px | %58,7 | — | 78 | ✗ eğik bakış uzağı küçültüyor |
| AeroVIS `ud_` dilimi (UAVDT) | yüksek | 1024×540 | **24 px** | **%75,1** | %23,7 | — | **✗✗** |
| DOTA · DIOR · SODA-A · iSAID · AI-TOD-v2 · xView | uydu / çok yüksek | — | — | — | — | — | **✗ kapsam dışı** |

### 1.1 Dersi: irtifa yanıltıcı bir filtre

Tablodaki üç satır bunu tek başına gösteriyor. **CODrone 100 m'de (44 px)
nesneler, UAVid'in 50 m'sinden (28 px) daha iri** — çünkü CODrone 4K ve
neredeyse dik bakıyor, UAVid ise 4K ama **eğik**, ve eğik bakışta karenin
uzak yarısı otomatik olarak tiny oluyor. Aynı şekilde **AU-AIR 20 m'de
1080p ile (81 px), CODrone'un 30 m'deki 4K'sını (47 px) geçiyor**, çünkü
belirleyici olan irtifa değil **yer örnekleme aralığı** (irtifa ÷ piksel
başına odak) ve bakış açısı.

Pratik sonuç: **"şu irtifadaki setleri al" diye filtreleme.** Ölçüt
doğrudan medyan √alan olmalı; bu doküman her aday için o sayıyı veriyor.
Öneri eşiği: **medyan ≥ 45 px ve < 32 px oranı ≤ %30**. Bunu geçen
kaynaklar: AU-AIR, CODrone (30 ve 60 m), AeroVIS'in `sd_` dilimi — ve
**DroneVehicle'ın RGB yarısı**, medyanda kıl payı kalsa da (40 px) tiny
oranında rahat geçerek (%19,0). DroneVehicle 80–120 m'de uçuyor, yani
"irtifa filtresi" onu elerdi; ölçüm elemiyor, çünkü karesi 640×512 dar
alan görüyor. Tablodaki en net "irtifaya bakma" kanıtı bu satır.

## 1.2 Özet — ne eklenmeli

| Öncelik | Set | Neden |
|---|---|---|
| **1** | [CODrone](#2-codrone--yeni-birincil-aday), **30 m + 60 m dilimleri** | 7 457 kare 3840×2160, medyan 60/47 px, kare başına 39/21 kutu. Havuzun hacmi buradan gelir; 100 m dilimi (2 547 kare) isteğe bağlı bırakılır |
| **2** | [AU-AIR](#41-au-air--en-alçak-irtifa-ama-tek-kavşak) | Bandın **en iri** nesnesi (medyan 81 px, <32 px yalnızca %11,8) ve kare başına irtifa okuması. Ama tek kavşak → seyrelterek, hacim için değil kalibrasyon/aşama C için |
| **3** | [SegFly](https://github.com/markus-42/SegFly) — **zaten repoda** | 30/40/50 m, tam hedef bant, 20 606 RGB, **gerçek maske**. Kalibrasyonun yeni doğal yeri; kısıtı `things`in sadece `vehicle` + `truck` olması |
| 4 | [Okutama-Action](#42-bant-içi-diğer-adaylar) · [CARPK](#42-bant-içi-diğer-adaylar) · [Semantic Drone](#42-bant-içi-diğer-adaylar) | 10–45 m 4K insan · ~40 m 90 K araba · 5–30 m 6000×4000. Üçü de bant içi, üçünün de erişimi betikle değil |
| **kalıyor** | [DroneVehicle](#43-dronevehicle--visdrone-değil-ve-iki-modalitede-tek-geçiş-fikri) — **zaten repoda** | RGB medyan 40 px / %19 tiny → bandı geçiyor. Asıl değeri tek geçişte **iki maske**: hizalı RGB-IR, ama akış **IR etiketinden** sürülmeli, RGB'den değil (§4.3) |
| **düştü** | VisDrone-DET, UAVDT, SODA-A, DIOR, DOTA, **iSAID**, AI-TOD-v2 | Ölçüldü, hepsi tiny/uydu rejiminde. iSAID'in 655 K elle çizilmiş örneği cazipti ama **uydu görüntüsü** — kalibrasyonu bant dışına taşırdı |

**Tek cümle:** havuzu **CODrone 30+60 m** üzerine kur, **AU-AIR**'ı
seyrelterek ve aşama C için ekle, kalibrasyonu **SegFly**'a taşı,
**DroneVehicle'ı tut ama IR etiketinden sür**, **VisDrone'u RGB
havuzundan çıkar**.

## 2. CODrone — yeni birincil aday

[GitHub](https://github.com/AHideoKuzeA/CODrone-A-Comprehensive-Oriented-Object-Detection-benchmark-for-UAV)
· [arXiv 2504.20032](https://arxiv.org/pdf/2504.20032) · CC BY-NC-SA 4.0

Drive dosyası uzaktan açıldı, merkezî dizini okundu — sayılar makaleden
değil **arşivin kendisinden**:

| | |
|---|---|
| arşiv | `CODrone.zip`, **7 233 513 058 B (6,74 GiB)**, Drive id `1FQ6mUaOr_kATDaH7N2bObD5SRRkV7qJy`, ranged GET **206** → `gdown`/`drive_download` çalışır |
| içerik | 30 025 girdi: **10 004 JPG + 10 004 DOTA-biçimi `.txt` + 10 004 VOC `.xml`** |
| bölünme | `train/` 5 002 · `val/` 2 000 · `test/` 3 002 görüntü — **üçünün de etiketi var** |
| çözünürlük | 3840×2160 (örneklenen en büyük kare 4,4 MB) |
| irtifa | **60 m: 4 316 · 30 m: 3 141 · 100 m: 2 547** (dosya adından: `..._60m_90c_...`) |
| kamera | 90°: 5 044 · 30°: 4 960 |
| gündüz/gece | gece ≈ **3 736** kare (%37) — Kust4K'nın D/N ekseninin görüntü tarafındaki karşılığı |
| sahne | 47 farklı yer (`huanqiucheng`, `pujiang`, `xidian`, `yuemeihu`, …) |
| sınıflar | 30 dosyalık örneklemde 994 kutu: `car` 411, `motor` 203, `people` 150, `truck` 72, `boat` 56, `traffic-sign` 39, **`ignored` 28**, `bus` 11, `traffic-light` 10, `bicycle` 8, `tricycle` 5, `bridge` 1 |
| yoğunluk | görüntü başına kutu (aynı örneklem): min 3, medyan 21, maks 116 |

Etiket satırı DOTA düzeni — sekiz poligon koordinatı, sınıf adı, zorluk:

```
0 25 75 25 75 55 0 55 car 0
856 1008 864 1126 604 1141 596 1024 truck 0
```

**Neden bu repo için kolay:** `src/training/boxes.py` DroneVehicle'ın yönlü
kutularını zaten dik zarfa çeviriyor; CODrone aynı poligon → zarf yolunu
kullanır, yeni okuyucu gerekmez. İki uyarı: **`ignored` sınıfı düşülmeli**
(DroneVehicle'ın 100 px beyaz bandıyla aynı refleks), ve `traffic-sign` /
`traffic-light` / `bridge` "şey" değil sabit tesis — `things` seçimi
Kust4K'da olduğu gibi **histogramla problanmalı**, ezberden yazılmamalı.

---

## 3. AeroVIS — açıldı ve sayıldı

[AeroTrack repo](https://github.com/Dmygithub/AeroTrack) ·
[arXiv 2607.08075](https://arxiv.org/abs/2607.08075) (UAV-OVVIS) · kod MIT,
**veri akademik / ticari olmayan kullanım**

`docs/datasets.md` bunu "yayınlanacak" diye değil, gerçek bir dosya olarak
listeliyordu; artık **arşivin içi okundu**:

| | |
|---|---|
| arşiv | `AeroVIS.zip`, **13 563 857 996 B (12,63 GiB)**, Drive id `1DMLagGZMPntrvxk5W0PsaIoybsE7WX56`, ranged GET **206** |
| içerik | 49 326 girdi: `aero_vis.json` (**332,6 MB**) + `sequences/{video}/{kare}.jpg` **49 204 JPG** + `data.md` |
| ölçek | **117 video · 49 204 kare · 8 279 iz · 9 sınıf** (makaleyle birebir) |
| kare/dizi | min 49, medyan 436, maks 768 |
| biçim | **YTVIS protokolü**: iz başına `bboxes`, **RLE `segmentations`**, `areas`, `track_id`. 80 adet `iscrowd=1` birincil değerlendirmeden dışlanmış |
| sınıf başına iz | person 2 413 · car 2 967 · truck 149 · bus 47 · bicycle 284 · motorcycle 663 · tricycle 269 · boat 41 · vehicle 1 446 |
| metrik | HOTA (ana), yanında DetA / AssA / mAP / Mask J |

`data.md` bir ayrımı açıkça yazıyor: **`car` (VisDrone, ince taneli) ile
`vehicle` (UAVDT, kaba taneli) ayrı değerlendiriliyor** — yani tek bir
"araç" sınıfı yok, iki kaynak iki sınıf olarak duruyor.

### 3.0 Nesne boyutu — "kullanılabilir mi"nin cevabı

Daha önce verilen cevap **"eğitime değil, ölçüme"** idi. Hedef bant
netleşince o cevap yetmiyor: `aero_vis.json` indirilip **1 411 532 maske
alanının tamamı** ölçüldü.

| dilim | maske | medyan √alan | < 32 px | < 16 px | > 64 px |
|---|---:|---:|---:|---:|---:|
| `vd_` (VisDrone) | 1 015 836 | 34 px | %44,5 | %6,0 | %15,7 |
| `ud_` (UAVDT) | 370 283 | **24 px** | **%75,1** | %23,7 | %5,2 |
| `sd_` (SeaDronesSee) | 25 413 | **84 px** | %21,1 | %0,1 | **%56,0** |
| **bütün** | **1 411 532** | **31 px** | **%52,1** | %10,6 | %13,7 |

Çözünürlük de düşük: 117 videonun 35'i 1024×540, 4'ü 960×540, 10'u
1344×756. Yalnızca 27'si tam 1080p.

**Cevap:** AeroVIS **bütünüyle bant dışı**. Maskelerinin yarıdan fazlası
32 px'in altında; onun üstünde ölçülen bir J&F skoru, esas olarak
"tiny nesnede ne kadar iyisin" sorusunun cevabı olur — yani tam olarak
odaklanmak istemediğin şey. Kullanılabilecek üç yolu var:

1. **`sd_` dilimi (26 dizi / 8 170 kare / 25 413 maske, medyan 84 px).**
   Bant içindeki tek parçası, ve listedeki **tek video örnek-maskesi +
   kimlik** kaynağı olarak kalıyor. Kısıtı dar: yalnızca `boat` (41 iz) ve
   `person` (34 iz), tamamı su üstü.
2. **Boyut kovalı J&F.** Tüm sette ölç ama skoru √alan kovasına göre kır ve
   raporda **≥ 45 px kovasını** öne al. Bu, kıyaslanabilirliği korurken
   sayının hangi rejimden geldiğini gizlemez.
3. **Hiç kullanma.** Bant içi video değerlendirmesi için AU-AIR (kimlik
   yok ama kutu var) ya da UAVScenes/MVUAV daha dürüst.

Kontaminasyon tarafında bir rahatlama var: VisDrone ve UAVDT artık havuza
zaten girmeyecekleri için (§1, ölçüldü ve elendi), §3.3'teki "AeroVIS'i
yakma" kısıtı pratikte kendiliğinden kalkıyor.

### 3.1 Nasıl üretilmiş

`data.md`'nin kendi şeması: **VisDrone · SeaDronesSee · UAVDT → SAM3 +
Manual Review → AeroVIS.** Makale de aynısını söylüyor: kaynak setlerin
kare düzeyindeki **elle çizilmiş kutuları** konum ve kimlik önselı olarak
SAM3'e veriliyor, çıkan maske adayları zamansal düzenleme + kalite
filtresi + **elle yeniden gözden geçirmeden** geçiyor.

Yani AeroVIS, kullanıcının sorduğu "birileri boru hattı kurup kutuyu maskeye
çevirmiş olabilir" tarifinin tam örneği — ve **bu reponun 13/14 numaralı
defterlerinin yaptığı işin aynısı**, farkı elle yeniden inceleme adımı.

### 3.2 Kaynak dağılımı — makalede yok, arşivde var

Dizi adlarının öneki kaynağı ele veriyor (`vd_`, `ud_`, `sd_`). Sayıldı:

| Kaynak | Dizi | Kare | Pay |
|---|---:|---:|---:|
| **VisDrone** (`vd_`) | 52 | 21 758 | %44,2 |
| **UAVDT** (`ud_`) | 39 | 19 276 | %39,2 |
| **SeaDronesSee** (`sd_`) | 26 | 8 170 | %16,6 |
| toplam | **117** | **49 204** | |

### 3.3 Kontaminasyon — mevcut kural bir yerde fazla sıkı, iki yerde eksik

`docs/datasets.md` bugün şunu diyor: *"13'ün havuzu VisDrone'dan üretildiği
için, bu havuzla eğitilen bir model AeroVIS'te ölçülemez — kare kare aynı
veri."* İkinci yarısı düzeltilmeli:

- VisDrone'un resmî README'si koleksiyonu **"288 video klip → 261 908 kare
  *ve* 10 209 statik görüntü"** diye tanımlıyor; statik görüntüler ile video
  kareleri **ayrı koleksiyon**. Defter 13 **VisDrone-DET** (statik) kullanıyor,
  AeroVIS ise **VisDrone-MOT/VID** (video) klipleri. Yani **kare düzeyinde
  örtüşme yok** — örtüşme sahne/kampanya düzeyinde (aynı 14 şehir, aynı
  platform). Bu bir alan yakınlığı, birebir sızıntı değil.
- Buna karşılık iki gerçek tehlike listede yoktu:
  **UAVDT-M havuza girerse AeroVIS'in 39 dizisi / 19 276 karesi (%39,2)
  birebir yanar**, **VisDrone-MOT girerse 52 dizi / 21 758 kare (%44,2)**,
  **SeaDronesSee girerse 26 dizi / 8 170 kare (%16,6)**. Üçü birden girerse
  AeroVIS'ten geriye ölçülecek bir şey kalmaz.

**Pratik kural:** AeroVIS ölçüm seti olarak kalacaksa, aşama C'nin masklet
kaynağı **VisDrone-MOT / UAVDT / SeaDronesSee olamaz**. Bu üçü zaten
listedeki en kolay video-kutu setleri — bu yüzden §6'da ayrı duruyorlar,
"indirilebilir ama bedeli şu" etiketiyle.

---

## 4. Kutu etiketli havadan RGB — görüntü

Aşama B havuzunu (`notebooks/13_rgb_mask_pool.ipynb`) besleyenler. Hepsinin
ucu **ranged GET 206** döndürüyor (kaldığı yerden devam eden indirme
çalışır), hiçbiri gated değil.

| Set | Ne | İndirme (doğrulandı 2026-08-25) | Kutu biçimi |
|---|---|---|---|
| ~~**VisDrone2019-DET**~~ *(bugün havuzda — **çıkarılmalı**)* | 6 471 train / 548 val, 10 sınıf, **medyan 32 px / %51'i < 32 px** | HF `banu4prasad/VisDrone-Dataset`, `snapshot_download` | YOLO txt |
| **CODrone** | 10 004 kare, 3840×2160, 12 sınıf, irtifa/açı ekseni | Drive `1FQ6mUaOr...`, **6,74 GiB** tek zip | DOTA poligon + VOC XML |
| ~~**SODA-A**~~ **(bant dışı)** | 2 513 kare, 872 069 yönlü kutu — ama görüntü başına ~350 örnek, tanımı gereği tiny | HF `torchgeo/soda-a`: `Images.zip` **10 960 MB** + `Annotations.zip` **55,4 MB** | yönlü kutu (JSON) |
| ~~**DIOR**~~ **(uydu)** | 23 463 görüntü 800×800, 192 472 kutu, 20 sınıf | HF `torchgeo/dior`: `Images_trainval.zip` 3 883 MB + `Images_test.zip` 3 528 MB + `Annotations_trainval.zip` 12,1 MB | VOC XML |
| ~~**DOTA v1.0 / v1.5 / v2.0**~~ **(uydu)** | v2.0: 11 268 görüntü, 1,79 M örnek, 18 sınıf | HF `isaaccorley/dota` (ungated, toplam 22,5 GB): `dotav1.0_images_train.tar.gz` 10 183 MB, `dotav2.0_images_train.tar.gz` 7 720 MB, etiketler ≤ 5 MB | DOTA poligon |
| ~~**SeaDronesSee**~~ *(yüksek irtifa)* | denizde arama-kurtarma, 5 sınıf (`swimmer`, `boat`, `jetski`, `life_saving_appliances`, `buoy`) | Resmî dağıtım **kayıt istiyor**; HF aynası `dronefreak/SeaDronesSee` ungated, **7 416 MB**, 20 964 dosya, hazır `data.yaml` (train 3 shard + val 1 shard; test GT'siz) | YOLO txt |
| **AU-AIR** | 32 823 kare 1920×1080, **132 031 kutu**, 8 sınıf, 2,8–30,6 m irtifa, kare başına GPS/IMU/hız | Drive `1pJ3xfKtHi...` **2,16 GiB** (`auair2019data.zip`) + `1boGF0L6ol...` **4,0 MB** (etiket zip'i, `annotations.json`) — ikisi de 206 | özel JSON (`top/left/width/height` + `class`) |
| ~~**AI-TOD-v2**~~ **(tanımı gereği tiny)** | 28 036 görüntü, 752 745 örnek, ortalama nesne **12,7 px** | [GitHub](https://github.com/Chasel-Tsui/AI-TOD-v2) — link sayfası, aynası yok | COCO |
| **VEDAI** | 1 246 görüntü, 3 640 nesne, 8 araç sınıfı, **RGB + IR eş kayıtlı** | HF `ckyrkou/vedai`: `vedai.zip` **52,3 MB** (512² alt küme) | yönlü kutu |

> **Üstü çizili satırlar hedef bandın dışında** (§1). Listede kalmalarının
> sebebi, öğretmenin *tavanını* değil **tabanını** ölçen stres testi
> olabilmeleri: kutu-IoU kapısının 20 px'lik hedefte ne yaptığını görmek
> isteyen bir koşu için hâlâ en hızlı yol. Eğitim havuzuna girmezler.
>
> **DOTA/DIOR/SODA-A "havadan" ama İHA değil.** Görüntüler Google
> Earth / uydu / yüksek irtifa uçak. EdgeTAM'in hedef alanı 30–130 m İHA;
> bunlar **alan çeşitliliği** ve **öğretmen stres testi** olarak değerli,
> alan-içi eğitim verisi olarak değil. İHA irtifasında olanlar: VisDrone,
> CODrone, SeaDronesSee, UAVDT.

### 4.1 AU-AIR — en alçak irtifa, ama tek kavşak

[arXiv 2001.11737](https://arxiv.org/abs/2001.11737) (ICRA 2020) ·
[veri indeksi](https://fmi-data-index.github.io/au_air.html) ·
[python kütüphanesi](https://github.com/bozcani/auairdataset)

Her iki Drive dosyası da indirildi/açıldı; aşağıdaki her sayı **etiket
dosyasının kendisinden** (`annotations.json`, 32 823 kayıt), makaleden değil.

| | |
|---|---|
| arşivler | `auair2019data.zip` **2 318 071 652 B (2,16 GiB)**, 32 823 JPG; etiket zip'i **4 039 375 B**. İkisi de ranged GET **206** |
| çözünürlük | 1920×1080 — JPEG SOF başlığından teyit edildi, etiketteki değerle uyuşuyor |
| örnek | **132 031 kutu** / 32 823 kare, **boş kare yok** |
| sınıf dağılımı | Car **102 619 (%77,7)** · Van 9 995 · Truck 9 545 · Human 5 158 · Trailer 2 538 · Bicycle 1 128 · Bus 729 · **Motorbike 319** |
| yoğunluk | kare başına kutu: min 1, **medyan 3**, ortalama 4,0, maks 56 |
| nesne boyutu | √alan: p05 **24 px**, medyan **81 px**, p95 249 px. 32×32'nin altı yalnızca **%11,8**, 16×16'nın altı **%1,3** |
| irtifa | **2,8 – 30,6 m**, p05 10,2 · medyan **20,4** · p95 29,7 (kare başına, mm cinsinden kayıtlı) |
| süreklilik | 8 uçuş; kare indeksleri **ardışık** (baskın fark = 1), dizi içi doluluk %51–99 |
| coğrafya | 56,2061–56,2071 K, 10,1878–10,1910 D → **≈ 111 m × 198 m'lik tek bir alan** (Aarhus) |
| zaman | 29 Ağu + 5–6 Eyl 2019, saat 09–15 — **gece yok** |
| platform | Parrot Bebop 2 |
| lisans | `annotations.json`'ın `licenses` bloğu: **CC BY-NC-SA 2.0** / CC BY-NC 2.0. Veri indeksi sayfasının "CC BY 4.0" demesi **yanlış** |

**Değerli olduğu iki yer.** Birincisi irtifa: listedeki her şey 30 m ve
üstü (CODrone 30/60/100, SegFly 30/40/50, HIT-UAV 80–130, DroneVehicle
~80–100). AU-AIR medyanı **20,4 m** ve nesne medyanı **81 px** — öğretmenin
en kolay rejimi. VisDrone/SODA-A'da kutu-IoU kapısının küçük hedefte
zayıfladığı ölçülmüştü; AU-AIR bunun karşı ucu, yani kapı eşiklerinin
tavanını gösteren set. İkincisi **kare başına irtifa okuması**: kalibrasyon
tablosu bugün sınıf × boyut; AU-AIR aynı tabloya bedava bir **irtifa sütunu**
ekler, çünkü ölçüyü etiketin kendisi taşıyor. Listedeki başka hiçbir set
bunu vermiyor.

**Neden birincil havuz kaynağı değil.** 32 823 kare kulağa VisDrone'un beş
katı gibi geliyor; değil. Hepsi **tek bir kavşağın** 8 uçuşu, ardışık video
kareleri. Aşama B için bu 32 823 örnek değil, **8 sahne** — havuza olduğu
gibi girerse aynı kavşak on binlerce kez tekrarlanır ve `split_frames`'in
tuttuğu ayrımlar anlamsızlaşır. Üstelik kare başına medyan 3 kutu var
(CODrone'da 21): öğretmen kodlayıcısını 32 823 kez çalıştırıp yalnızca
132 K maske alırsın — CODrone aynı bütçeyle ~5 kat daha fazla kutu
işler. Ek olarak JPEG'ler ağır sıkıştırılmış (1920×1080 için medyan
**72 KB** ≈ 0,28 bit/piksel), yani maske sınırı `box_iou` kapısının
varsaydığından yumuşak olabilir — ölçülmeli, varsayılmamalı.

**Kullanım tarifi.**

1. **Aşama C masklet kaynağı olarak — evet, doğrudan.** Ardışık kare +
   kare-başına kutu, `masklets.py`'nin tam istediği şey. Eksik olan tek şey
   **kimlik**: `bbox` yalnızca `class` taşıyor, iz numarası yok. Çözüm
   zaten yazılmış — ilk karede kutuyla promptla, sonraki her karede
   `gate_boxes` ile o karenin kutusuna karşı kapıla; RGBT234 rotasının
   aynısı, tek farkı kapı kutusunun ikinci modaliteden değil aynı
   görüntüden gelmesi.
2. **Aşama B havuzuna — seyrelterek.** Uçuş başına her N'inci kare
   (saniyede ~1 kare ⇒ ~1 100 kare) yeter; komşu kareler zaten
   birbirinin kopyası.
3. **Kalibrasyon probu olarak — evet.** Gerçek maskesi yok, o yüzden IoU
   veremez; ama kabul oranını irtifaya göre kırar.
4. **Kontaminasyon: yok.** AeroVIS'in üç kaynağının hiçbiri değil, mevcut
   setlerin hiçbiriyle ortak kare taşımıyor. Ölçüm tarafında da serbest —
   ama maskesi olmadığı için J&F değil, yalnızca kutu metriği verir.

```python
AUAIR = Recipe(
    name="auair",
    note="AU-AIR: 32 823 kare 1920x1080, 132 031 kutu, 8 sınıf, 2,8-30,6 m "
         "irtifa; ardışık video kareleri, tek kavşak (Aarhus)",
    parts=(
        Part("images", drive="1pJ3xfKtHiTdysX5G3dxqKTdGESOBYCxJ",
             size=2_318_071_652),
        Part("annotations", drive="1boGF0L6olGe_Nu7rd1R8N7YmQErCb0xA",
             size=4_039_375),
    ),
)
```

### 4.2 Bant içi diğer adaylar

Hedef banda irtifasıyla giren ama her biri bir bedelle gelen dört set.
İkisi ölçüldü, ikisinin erişimi betikle değil.

| Set | irtifa · çözünürlük | ne | ölçüm / kısıt |
|---|---|---|---|
| **Okutama-Action** | **10–45 m**, 45°/90°, **4K** | 43 dizi / **77 365 kare**, kutu + **kimlik** + 12 eylem sınıfı, DJI Phantom 4 | Bandın ortası ve 4K → nesne iri olmalı. Ama **tek konu**: tek bir beysbol sahasında yürüyen/el sıkışan insanlar, araç yok. Erişim [GitHub](https://github.com/miquelmarti/Okutama-Action) üzerinden form |
| **CARPK** | **~40 m**, DJI Phantom 3 | 4 otoparktan **89 777 araba kutusu** / 1 448 görüntü | Bant tam ortada, sınıf tek (`car`), sahne 4. HF aynası `ShantyCam/carpk`: **`carpk.zip` 2 081,6 MB**, ranged GET **206** — doğrulandı |
| **Semantic Drone (TU Graz)** | **5–30 m**, **6000×4000** | 400 açık eğitim görüntüsü, 20 sınıf semantik + insan kutusu | Listedeki en yüksek çözünürlük ve en alçak irtifa → nesneler devasa. Ama **400 görüntü** ve dağıtımı [kurum sayfası](https://ivc.tugraz.at/research-project/semantic-drone-dataset/) / Kaggle |
| **AeroScapes** | **5–50 m**, 1280×720 | 3 269 görüntü + maske, 12 sınıf (**Person, Bike, Car, Drone, Boat, Animal** = 6 "şey") | **Ölçüldü ve zayıf çıktı**: thing bileşenlerinin medyanı **35 px**, %40'ı 32 px altında, **kare başına medyan 1–2 nesne**. İrtifa bant içi ama 720p bunu yiyor. HF `dronefreak/Aeroscapes`, **757 MB**, hazır `data.yaml` |
| **UAVid** | 50 m **eğik**, 3840×2160 | 420 görüntü (200/70/150) + maske, 8 sınıf | **Ölçüldü ve elendi**: medyan **28 px**, %58,7'si 32 px altında — 4K'ya rağmen, çünkü eğik bakışta karenin uzak yarısı tiny. HF `dronefreak/UAVid-2020`, 6,47 GB |

**UAVid aynasında bir palet tuzağı daha.** `dronefreak/UAVid-2020` maskeleri
resmî RGB paletiyle değil **sınıf indeksiyle** (0–7) dağıtıyor, ve indeks
sırası resmî sıra **değil**: `data.yaml`'a göre `3 = Static Car`,
`6 = Human`, `7 = Moving Car`. Resmî sıraya güvenip 5/6/7 okuyan bir probe
**sıfır bileşen** buluyor — bu ölçümde bire bir yaşandı. Kust4K ve SegFly
paletlerinin dersi burada üçüncü kez geçerli: **paleti setin kendi
dosyasından oku.**

### 4.3 DroneVehicle — VisDrone değil, ve iki modalitede tek geçiş fikri

[GitHub](https://github.com/VisDrone/DroneVehicle) · RA-L / T-CSVT 2022 ·
`fetch_datasets.py`'de **zaten var** (`dronevehicle`)

Önce karışıklığı temizleyelim: **DroneVehicle, VisDrone değildir.** Aynı
laboratuvarın (Tianjin Üniversitesi AISKYEYE) deposu olduğu ve GitHub yolu
`VisDrone/DroneVehicle` diye geçtiği için karışıyor; ama başka bir çekim,
başka bir sensör (RGB **+ termal**), başka bir etiket (5 araç sınıfı,
yönlü kutu). Bu dokümanda ikisi hep ayrı satırlar. **AeroVIS'in üç
kaynağından da biri değil** → kontaminasyon yok.

| | |
|---|---|
| ölçek | **28 439 hizalı RGB-IR çifti** (56 878 görüntü), **953 087 yönlü kutu** |
| sınıflar | car · truck · bus · van · freight car |
| kutu, modalite başına | **RGB 452 570** · **IR 500 517** — sayılar *farklı*, sebebi §aşağıda |
| kare | 840×712, çevresinde 100 px beyaz bant → içerik **640×512** (XML `<size>`'dan doğrulandı) |
| irtifa · açı | **80–120 m**, dik + 15°/30°/45° eğik |
| ışık | gündüz · gece · **karanlık gece** |
| indirme | HF `McCheng/DroneVehicle` — `val.zip` **723 321 423 B**, ranged GET **206**; train 8,88 GB, test 4,43 GB |

**Ölçüm** (val'den 70 çift, 840 RGB + 1 149 IR kutu):

| | medyan √alan | < 32 px | > 64 px |
|---|---:|---:|---:|
| RGB | **40 px** | %19,0 | %12,7 |
| IR | 40 px | %19,9 | — |

RGB p05 24 · p25 34 · p75 52 · p95 77 px; kare başına medyan 12 kutu.
Yani hedef bandı **geçiyor** — VisDrone-DET'ten (32 px / %51) belirgin
şekilde daha iyi, CODrone 30 m'nin (47 px / %28,9) biraz altında.

#### "RGB'de etiketlerken termali de kaydedeyim" — fikir doğru, yönü ters

Bu zaten boru hattının **(a) rotası** (`docs/mask_pool_plan.md` §2) ve
defter 14'ün ilk çıktısı tam olarak "(a) mı (b) mi" ölçümü. DroneVehicle
bu ölçüm için Kust4K'nın yanına konabilecek **ikinci ve çok daha büyük**
yüzey. Ama iki düzeltmesi var, ikisi de ölçüldü.

**1. Maskeyi taşıma, promptu taşı.** Aynı kareye ait RGB ve IR kutuları
merkezden eşleştirildiğinde eşleşmelerin **%72,1'i tam 0,0 px** kayıyor —
ama bu sensör hizasının kanıtı *değil*, etiketçinin çoğu kutuyu
modaliteler arası **kopyaladığının** kanıtı. Bilgi taşıyan kısım yeniden
çizilen %27,9: orada kayma **medyan 5,2 px, p90 12,6 px, maks 36,1 px**.
40 px'lik bir araçta p90 kayma nesnenin **~%32'si** demek — RGB'de üretilip
termale olduğu gibi yapıştırılan bir maske o kuyrukta aracın kenarını
kesiyor. Repoda çaresi zaten yazılı: **prompt RGB kutusundan, kapı termal
kutusundan** (`masklets.VideoSequence.gate_boxes`, RGBT234 rotası). Kayan
çift orada düşer.

**2. RGB'den sürme — IR'den sür.** Asıl mesele hiza değil, **kapsam**.
Örneklemde **1 149 IR kutusunun 319'unun (%27,8) RGB karşılığı yok**;
tersi neredeyse hiç (10/840). Yayınlanan tam sayılar da aynısını söylüyor:
IR 500 517'ye karşı RGB 452 570 — car 389 779 → 428 086, freight car
13 400 → 17 173 (**+%28,2**), truck 22 123 → 25 960 (+%17,4). Bunlar
gece ve karanlık gece araçları: RGB karesinde **görünmüyorlar**, o yüzden
etiketleri de yok.

Sonuç: "RGB'de etiketle, termali de kaydet" dediğinde **termal hedeflerin
yaklaşık dörtte birini baştan kaybediyorsun** — üstelik tam olarak termal
dalın var olma sebebi olanları. Doğru yön tersi:

```
IR etiketi üzerinde döngü kur (üst küme, 500 517 kutu)
  ├─ RGB'de tolerans içinde eşi VAR  → kalibrasyonun kazanan rotasıyla
  │                                    maskeyi üret, ikisine birden yaz
  └─ RGB'de eşi YOK (gece nesnesi)   → öğretmeni doğrudan termalde
                                       promptla (rota b)
```

Bu, önerilen akışın kaybettiği ~48 000 örneği geri alıyor ve RGB maskesini
mümkün olan her yerde yine üretiyor. Sıfır ek indirme: iki etiket klasörü
(`*label/` ve `*labelr/`) aynı arşivde.

**Bir de sınıf adı tuzağı:** freight car sınıfı dosyalarda **iki farklı
yazımla** geçiyor — `feright_car` *ve* `feright car` (ikisi de "freight"
yazım hatasıyla). Örneklemde RGB'de 13 + 7, IR'de 41 + 8. Tek yazıma
normalize edilmezse sınıf haritası bu nesnelerin bir kısmını sessizce
düşürür — Kust4K/SegFly/UAVid palet derslerinin dördüncüsü.

---

## 5. Gerçek örnek maskesi — kalibrasyon ve held-out

Havuz boru hattının **kalibrasyon** adımı (`pool.py`, Kust4K rotası) bugün
semantik maskeyi `decompose` ile örneklere ayırıyor. Elle çizilmiş gerçek
örnek sınırı olan bir set bunu doğrudan yapar — ve havadan RGB'de bu
setler **az**:

| Set | Etiket | Ne | İndirme |
|---|---|---|---|
| **iSAID** | **elle çizilmiş örnek maskesi** | **655 451 örnek / 2 806 görüntü / 15 sınıf**; train 1 411 · val 458 · test 937. Görüntüler DOTA-v1.0'ın kendisi | Etiketler: HF `isaaccorley/isaid` (`isaid_annotations_train.tar.gz` 159,4 MB + val 52,7 MB, **CC BY-NC 4.0**). Görüntüler: HF `isaaccorley/dota` v1.0 train+val (13,5 GB). Tek-paket alternatif: HF `ariG23498/iSAID` (10,8 GB parquet, `image`/`ins`/`seg` kolonları, **ama satır başına tek `category_id`** — dönüşüm şüpheli, önce probla |
| **NWPU VHR-10 (instance)** | elle çizilmiş | 650 görüntü, 10 sınıf — küçük ama temiz; RSPrompter'ın değerlendirme setlerinden | [RSPrompter](https://github.com/KyanChen/RSPrompter) üzerinden |
| **DRespNeT** | **poligon** | 2023 Türkiye–Suriye depremi 1080p İHA görüntüsü, **28 sınıf** (hasarlı bina, girişler, moloz, araç, sivil, kurtarma ekibi) | [figshare](https://figshare.com/s/66d3116a0de5b7d827fb) (YOLOv8-seg zip) · [arXiv 2508.16016](https://arxiv.org/pdf/2508.16016). **Bu sandbox'tan doğrulanamadı** (§8) |
| **AeroVIS** | SAM3 + elle inceleme | §3 | Drive, doğrulandı |

**Kalibrasyon tavsiyesi hedef bant yüzünden değişti: iSAID değil, SegFly.**

iSAID'in cazibesi hâlâ geçerliydi — 655 K sınır insan eli, ve öğretmeni
kendi çıktısına karşı skorlama döngüsüne düşmüyor (AeroVIS'in maskesi
SAM3 üretimi; SAM 3 öğretmenini ona karşı ölçmek IoU'yu değil iki SAM 3
koşusunun tekrarlanabilirliğini ölçer). Ama iSAID **uydu görüntüsü**:
kalibrasyonu tam da kaçınmak istediğimiz rejimde yapmak, ölçülen kabul
oranını hedef bantta geçersiz kılar. Kalibrasyon **çalışılacak rejimde**
olmalı.

Bant içinde kalan seçenekler ve neden bu sıra:

1. **SegFly — 30/40/50 m, zaten indiriliyor.** Hedef bandın birebir
   kendisi, gerçek maske, 20 606 RGB kare. Kısıtı `things`in sadece
   `vehicle` (13) + `truck` (36) olması — insan/motosiklet kalibre
   edilemez. Bugünkü Kust4K rotasının yanına konması gereken ilk şey.
2. **Semantic Drone (5–30 m, 6000×4000)** — AU-AIR'ın irtifasını maskeyle
   eşleyen tek set; 400 görüntü az ama kalibrasyon zaten hacim işi değil.
3. **AU-AIR'ın kendisi maske vermiyor**, ama **irtifa sütununu** o taşıyor
   (§4.1): kabul oranını irtifaya göre kırmak için gereken ölçü onda.
4. **iSAID → bant dışı stres testi.** Silinmiyor, ama rolü "kalibrasyon"
   değil "öğretmen küçük nesnede nerede kırılıyor" sorusu.

---

## 6. Video + kimlikli kutu — aşama C masklet kaynağı

`src/training/masklets.py` bunları doğrudan okuyabilir (kare başına kutu +
kimlik → masklet). **Üçü de AeroVIS'in kaynağı** — §3.3'teki bedel geçerli.

| Set | Ne | İndirme (doğrulandı) | AeroVIS bedeli |
|---|---|---|---|
| **UAVDT** | 100 dizi, ~80 K kare 1080×540, **0,84 M kutu / 2 700 araç**, 14 öznitelik (hava, irtifa, kamera açısı, örtülme) | HF `vanthanh/UAVDT-Benchmark-M`: `UAV-benchmark-M.zip` **6 790 MB** + `UAV-benchmark-MOTD_v1.0.zip` **245,7 MB** | 39 dizi / **19 276 kare** yanar |
| **VisDrone-MOT** | 288 klip / 261 908 karenin MOT bölümü | HF `vanthanh/VisDrone2019-MOT`: train 8 081 MB + val 1 602 MB + test-dev 2 294 MB | 52 dizi / **21 758 kare** yanar |
| **SeaDronesSee-MOT** | 22 klip, 3840×2160 | Resmî, **kayıt gerekli** | 26 dizi / **8 170 kare** yanar |
| **Okutama-Action** | 43 dizi / **77 365 kare** 4K, kutu + kimlik + 12 eylem sınıfı, tepeden insan | [GitHub](https://github.com/miquelmarti/Okutama-Action) — form ile | **yok** |
| **Stanford Drone** | 8 sahne / 60+ video, ~20 000 hedef (yaya, bisiklet, araba, golf arabası), tepeden sabit İHA | [Kaggle](https://www.kaggle.com/datasets/aryashah2k/stanford-drone-dataset), academictorrents | **yok** |
| **AU-AIR** | 8 uçuş / **32 823 ardışık kare** 1920×1080, 132 031 kutu, ama **kimlik yok** (§4.1) | Drive, **2,16 GiB**, doğrulandı | **yok** |

Okutama-Action, Stanford Drone ve AU-AIR, **AeroVIS'e dokunmadan** aşama C'ye
kare-başına kutu getiren üç set. Bedelleri farklı: Okutama form arkasında,
Stanford yalnızca üçüncü şahıs aynada, AU-AIR ise betikle iniyor ama kimlik
taşımıyor — izi ilk kareden propagasyonla kurup her karede kutuyla kapılamak
gerekiyor (§4.1).

---

## 7. Kutuyu maskeye çevirmiş olanlar — veri ve boru hattı

Kullanıcının sorduğu asıl kategori. İkiye ayrılıyor: **çıktısı hazır veri**
ve **kodu hazır boru hattı**.

### 7.1 Hazır ürünler

| Ne | Kaynak → çıktı | Ölçek | Erişim |
|---|---|---|---|
| **SAMRS** — SOTA / SIOR / FAST ([GitHub](https://github.com/ViTAE-Transformer/SAMRS), [arXiv 2305.02034](https://arxiv.org/html/2305.02034), NeurIPS'23 D&B) | DOTA-V2.0 + DIOR (**H-Box** prompt) ve FAIR1M-2.0 (**R-Box** prompt) → SAM maskeleri | **105 090 görüntü / 1 668 241 örnek** — mevcut RS segmentasyon setlerinin 10 katından fazla. Hem örnek maskesi hem kutu taşıyor | **Sadece OneDrive + Baidu**, HF aynası **yok** (arandı, sıfır sonuç). Colab'dan betikle çekilemez; büyük diskli makinede elle. `Generate Dataset/` klasörü üretim kodunu içeriyor → **kendin üretmek de mümkün**, girdiler (DOTA v2 / DIOR) §4'te ungated |
| **AeroVIS** | VisDrone + UAVDT + SeaDronesSee kutuları → SAM3 + elle inceleme | 117 video / 49 204 kare / 8 279 iz | Drive, **doğrulandı** (§3) |
| **UAVDB** ([GitHub](https://github.com/wish44105/UAVDB), [arXiv 2409.06490](https://arxiv.org/pdf/2409.06490)) | yörünge **noktası** → PIC ile kutu → **SAM2** ile maske | çok görünüşlü İHA takip verisi | Açık; **ama bakış açısı ters** — sabit yer kamerası, hedef İHA'nın kendisi. Havadan RGB değil |
| **SAM2DV** | DroneVehicle OBB → SAM2 | 17 990 + 1 469 örnek | Repoda zaten **elendi** (IEEE DataPort ücretli, `docs/datasets.md`) |

**SAMRS notu:** boru hattımızla en yakın akraba bu. Ders çıkarılacak yeri
prompt seçimi: FAIR1M için **yönlü** kutu, DOTA/DIOR için **dik** kutu
kullanmışlar. CODrone yönlü kutu veriyor ve biz onu dik zarfa çeviriyoruz —
SAMRS'in bulgusu, yönlü kalmanın sıkı nesnelerde daha iyi olduğu yönünde;
kalibrasyon geçişinde **iki prompt tipini yarıştırmak** ölçülebilir bir soru.

### 7.2 Kodu hazır boru hatlar

Havuz kodumuz (`src/training/pool.py`) zaten var; bunların değeri kod değil,
**kapı ve prompt tasarımı** fikirleri:

| Repo | Ne veriyor |
|---|---|
| [opengeos/segment-geospatial](https://github.com/opengeos/segment-geospatial) (`samgeo`) | Coğrafi raster için kutu/nokta/metin promptu → maske; **[SAM3 kutu-prompt örneği](https://samgeo.gishub.org/examples/sam3_box_prompts/)** ve SAM2 karşılığı hazır defter olarak duruyor. Öğretmen API'sini teyit etmenin en hızlı yolu |
| [KyanChen/RSPrompter](https://github.com/KyanChen/RSPrompter) | SAM'e RS için **öğrenilmiş prompt**; WHU-building / NWPU VHR-10 / SSDD üzerinde ölçülmüş. Sabit kutu promptunun tavanını gösteriyor |
| [zhen6618/OBBInstanceSegmentation](https://github.com/zhen6618/OBBInstanceSegmentation) (OBSeg, [arXiv 2401.08174](https://arxiv.org/abs/2401.08174)) | **Yönlü kutu prompt kodlayıcı** — SAM'in kabul etmediği OBB'yi prompta çeviriyor. CODrone/DroneVehicle poligonlarını zarfa düşürmeden kullanmanın yolu |
| [LiWentomng/BoxInstSeg](https://github.com/LiWentomng/BoxInstSeg) · [boxlevelset](https://github.com/LiWentomng/boxlevelset) (Box2Mask, ECCV'22 / TPAMI) | **SAM'siz** kutu-denetimli örnek segmentasyonu (level-set). Öğretmen kapılarının çok şey reddettiği yerde bağımsız ikinci görüş |
| [PM-VIS](https://arxiv.org/pdf/2404.13863) | **Kutu-denetimli video** örnek segmentasyonu — aşama C'nin masklet üretiminin literatürdeki karşılığı |
| [haochenheheda/segment-anything-annotator](https://github.com/haochenheheda/segment-anything-annotator) · [Dezt/yoloSAMtools](https://github.com/Dezt/yoloSAMtools) · `ultralytics.data.annotator.auto_annotate` | Elle inceleme adımı için araç. AeroVIS'in "manual review" aşamasını taklit etmek isteyen için |

---

## 8. Doğrulama — ne yapıldı, ne yapılamadı

**Yapılan.** HF uçları `Range: bytes=0-1023` ile denendi, hepsi **206**
döndü. İki Drive dosyası (`AeroVIS.zip`, `CODrone.zip`) **uzaktan zip
olarak açıldı**: onay formu yeniden gönderildi, `content-range`'den boyut
alındı, merkezî dizin ranged GET'le okundu ve dosya listesi/sayıları
oradan çıkarıldı. §2 ve §3'teki her sayı bu okumanın çıktısı — makaleden
alınmadı. CODrone sınıf histogramı, arşivin içinden 30 rastgele `.txt`
etiket dosyası okunarak üretildi.

Boyut ölçümleri (§1) aynı disiplinle: CODrone irtifa kovası başına 34
`annfile`, VisDrone-DET için aynanın 60 YOLO etiketi, AeroScapes için 18
maske, UAVid için 6 maske (bağlı bileşen, `scipy.ndimage.label`), AU-AIR
için etiket dosyasının **tamamı** (132 031 kutu), AeroVIS için
`aero_vis.json`'ın **tamamı** (332,6 MB indirildi, 1 411 532 maske alanı).
VisDrone'un YOLO etiketi normalize olduğu için piksel boyutu 1920×1080
varsayımıyla hesaplandı — yani **iyimser taraf**; setin bir kısmı
1360×765 ve orada nesneler daha da küçük.

DroneVehicle için `McCheng/DroneVehicle`'ın `val.zip`'i uzaktan zip olarak
açıldı ve 70 karenin **her iki** etiket klasörü (`vallabel/` = RGB,
`vallabelr/` = IR) okundu: 840 RGB + 1 149 IR yönlü kutu. Modaliteler arası
kayma, kutu merkezleri 40 px eşiğiyle açgözlü eşleştirilerek ölçüldü;
eşleşmeyenler "yalnız şu modalitede" sayıldı. XML'lerin bir kısmında
kaçırılmamış `&` var, ayrıştırıcı bunu onarıyor.

**Yapılamayan — üçü de bu sandbox'ın kısıtı, setin kusuru değil:**

| Ne | Ne oldu |
|---|---|
| `codeload.github.com` | **403**. Repoda hâlihazırda çalışan HIT-UAV bağlantısı da aynı 403'ü veriyor → ajan proxy'sinin engeli. Colab'da bu yol açık; CODrone/AI-TOD-v2 gibi GitHub barındırmalı setler oradan denenmeli |
| DRespNeT figshare paylaşımı | **202 + boş gövde** (JS ile üretilen sayfa). Kust4K'da görülen figshare davranışının aynısı: dosya-bazlı `ndownloader/files/<id>` ucu gerekiyor, ama paylaşım linkinden dosya id'si okunamadı |
| SAMRS OneDrive/Baidu | OneDrive linki 301 ile klasör görünümüne gidiyor; betikle liste alınamıyor. Baidu zaten hesap istiyor |
| SeaDronesSee resmî | [macvi.org/dataset](https://macvi.org/dataset) kayıt istiyor — bu yüzden HF aynası tercih edildi |

---

## 9. `fetch_datasets.py`'ye nasıl bağlanır

Üçü de mevcut `Recipe` alanlarıyla, yeni mekanizma gerektirmeden yazılır
(`Part.drive` ve `snapshot`/`url` yolları hazır):

```python
CODRONE = Recipe(                      # havuzun yeni gövdesi
    name="codrone",
    note="CODrone: 10 004 İHA karesi 3840x2160, 12 sınıf yönlü kutu, "
         "30/60/100 m irtifa x 30/90 derece kamera, ~%37 gece",
    parts=(Part("all", drive="1FQ6mUaOr_kATDaH7N2bObD5SRRkV7qJy",
                size=7_233_513_058),),
)
# Tek zip, irtifaya göre bölünemiyor -- ayrım okuma anında, dosya adından:
#   re.search(r"_(\d+)m_", stem).group(1) in {"30", "60"}
# 30+60 m = 7 457 kare (medyan 47 / 60 px); 100 m dilimi (2 547) opsiyonel.

AUAIR = Recipe(                        # bandın en iri nesnesi + irtifa sütunu
    name="auair",
    note="AU-AIR: 32 823 kare 1920x1080, 132 031 kutu, 2,8-30,6 m irtifa; "
         "ardışık video kareleri, tek kavşak -- aşama B'ye seyrelterek",
    parts=(
        Part("images", drive="1pJ3xfKtHiTdysX5G3dxqKTdGESOBYCxJ",
             size=2_318_071_652),
        Part("annotations", drive="1boGF0L6olGe_Nu7rd1R8N7YmQErCb0xA",
             size=4_039_375),
    ),
)

AEROVIS = Recipe(                      # yalnızca ölçüm, ve yalnızca sd_ dilimi
    name="aerovis",
    note="AeroVIS: 117 video / 49 204 kare / 8 279 iz, YTVIS RLE + track_id. "
         "Bütünü bant dışı (medyan 31 px); sd_ dizileri bant içi",
    parts=(Part("all", drive="1DMLagGZMPntrvxk5W0PsaIoybsE7WX56",
                size=13_563_857_996),),
)

CARPK = Recipe(                        # ~40 m, tek sınıf, ucuz
    name="carpk",
    note="CARPK: 1 448 görüntü / 89 777 araba kutusu, ~40 m, 4 otopark",
    parts=(Part("all",
                url="https://huggingface.co/datasets/ShantyCam/carpk/"
                    "resolve/main/carpk.zip",
                size=2_081_600_000),),
)
```

iSAID/DOTA tarifi bilerek yazılmadı: §5'te kalibrasyon iSAID'den SegFly'a
taşındı, ve SegFly `fetch_datasets.py`'de zaten var.

Boyutlar canlı `content-length`/`content-range`'den; `Part.md5` bu
kaynakların hiçbirinde yayıncı tarafından verilmiyor, bu yüzden boş —
Kust4K'daki gibi md5 doğrulaması yapılamaz, karşılığında defterin **probe
hücresi** sayarak doğrular (CODrone: 10 004 görüntü, `ignored` sınıfının
varlığı ve irtifa histogramının 3 141/4 316/2 547 çıkması; AU-AIR: 32 823
kare / 132 031 kutu; AeroVIS: 117 dizi / 49 204 kare; CARPK: 89 777 kutu).

Probe hücresine bu dokümanın asıl ölçütü de girmeli: **medyan √alan ve
< 32 px oranı**. Yeni bir kaynak eklendiğinde önce o iki sayı basılmalı;
eşik (§1.1) medyan ≥ 45 px ve < 32 px ≤ %30.

Sınıf id'leri **hiçbirinde ezberden yazılmamalı** — Kust4K ve SegFly
paletlerinin ikisinin de yanlış tahmin edildiği dersi burada da geçerli.
