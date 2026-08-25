# `(görüntü, prompt, maske)` veri setleri — aşama B için ne var, ne yok

Aşama B'nin (`src/training/image_loop.py`) yediği tek şey üçlüdür: bir kare,
o karedeki **tek bir nesneyi** seçen bir prompt (kutu ya da tık), ve o
nesnenin maskesi. `docs/datasets.md` setleri "yoğun maske GT'si var mı"
diye eliyor. Bu dosya bir eksen daha ekliyor — ve pratikte belirleyici olan
o:

> **Maske, nesne başına mı ayrılmış?**

Semantik bir harita "burası araç" der. Aşama B'nin sorusu "bu prompt hangi
aracı gösteriyor" — ve bir otoparkta o iki cümle aynı şey değildir.

Buradaki her sayı **indirilen dosyadan ölçüldü**. Ölçülmeyen (kaynağa
erişilemeyen) her satır açıkça öyle işaretli.

---

## 1. Üç kalite sınıfı

| sınıf | ne verir | aşama B'ye maliyeti |
|---|---|---|
| **A — yerli örnek maskesi** | dosya zaten `car1 / car2 / car3` ayırıyor | sıfır: kutu = prompt, maske = hedef |
| **B — semantik harita** | "burası araç" | `decompose` ile yeniden kurma; **bitişik nesneler kaynar** |
| **C — sadece kutu** | kutu | öğretmen geçişi (defter 13/14, `src/training/pool.py`) |

Reponun bugünkü aşama B verisi neredeyse tamamen **B sınıfı**: Kust4K ve
SegFly semantik, örnekleri `decompose` yeniden kuruyor. Ölçülen kayıp
küçük değil — SegFly'de bir-araç-bir-blok isabeti **%78,5**, araç
piksellerinin **%20,9'u** çok-araçlı bloklarda. `datasets.md`'nin VTUAV-VIS'i
"listedeki tek örnek maskeli havadan RGB-T set" diye işaretlemesinin sebebi
bu; ve VTUAV'ın çizilmiş maske sayısı **875**.

Aşağıdaki asıl bulgu şu: **A sınıfı havadan RGB'de bol, havadan termalde
pratikte yok.**

---

## 2. iSAID — A sınıfının en büyüğü, ama başka bir ölçekte

[iSAID](https://captain-whu.github.io/iSAID/) (CVPR-W 2019), DOTA-v1.0
görüntüleri üzerine elle çizilmiş **örnek** maskeleri. İndirilip açıldı
(`isaaccorley/isaid` aynası, 200 MB, ungated, anonim indirilebiliyor).

**val bölümünden ölçülen:**

```
458 görüntü · 116 649 örnek maskesi · 15 sınıf · iscrowd = 0 (hepsi)
görüntü başına medyan 56, maksimum 7 436 örnek
```

Dağıtım üç dosya tipi veriyor ve üçüncüsü tam olarak SegFly'de olmayan şey:

| dosya | ne |
|---|---|
| `Annotations/iSAID_val.json` | COCO: poligon `segmentation` + `bbox` + sınıf |
| `Semantic_masks/…_instance_color_RGB.png` | semantik (P0003'te 4 renk = 3 sınıf + arka plan) |
| **`Instance_masks/…_instance_id_RGB.png`** | **örnek kimliği**: her nesne ayrı bir RGB rengi — P0003'te **60 farklı değer** (59 örnek + arka plan) |

Yani `car1 / car2 / car3` sorusunun cevabı burada **evet**, üstelik dosya
biçiminde.

### Ne kadar fark ediyor — aynı ölçüm, iSAID üstünde

SegFly'de sorulan soru burada da soruldu: elle çizilmiş örnekleri tek bir
semantik haritaya eritip connected-component ile geri kurmayı denersen ne
kaybedersin? Beş yoğun karede (`P2733`, `P2097`, `P2504`, `P1786`, `P2462`):

```
3 523 çizilmiş araç  ->  2 889 bileşen     (isabet %82,0)
949 örnek (%26,9) başka bir örnekle aynı bloğa düşüyor
en kötü kare: 751 araç -> 466 bileşen (%62,1); en büyük blok ~15 araç
```

SegFly'de ölçülen %78,5 ile aynı büyüklük sınıfı. **Fark, iSAID'de bu kaybı
ödemek zorunda olmaman** — ayrım dosyada zaten var.

### Ama: ölçek rejimi bu projenin rejimi değil

| sınıf | örnek | medyan alan | medyan uzun kenar |
|---|---:|---:|---:|
| Small_Vehicle | 84 096 | 67 px | **13 px** |
| ship | 12 453 | 457 px | 38 px |
| Large_Vehicle | 9 124 | 506 px | 43 px |
| plane | 2 621 | 1 387 px | 69 px |
| tennis_court | 788 | 5 572 px | 118 px |

```
tüm sınıflar: medyan uzun kenar 18 px · örneklerin %75,6'sının uzun kenarı < 32 px
```

Görüntüler uydu (Google Earth + JL-1 + GF-2), İHA değil. SegFly'de 50 m'den
bir araba ~300×150 px; iSAID'de **11×11 px**. Aynı nesne, iki büyüklük
mertebesi arası fark.

**Sonuç:** iSAID'i "havadan araç segmentasyonu" verisi diye stage B'ye
katmak, modele bu projenin hiç görmeyeceği bir ölçek öğretir. Değeri başka
yerde:

1. **Kalibrasyon / ölçüm seti.** `pool.calibrate_spec` bugün Kust4K'nın
   *yeniden kurulmuş* örneklerine karşı ölçüyor — yani belirsiz bir hedefe.
   iSAID'in 116 649 val maskesi **elle çizilmiş**, yani havuzun ürettiği
   maskenin şeklini dürüst bir hedefe karşı ölçmenin tek yolu.
2. **Küçük hedef rejimi.** Bu proje zaten küçük hedefle uğraşıyor
   (`zoom_window` bunun için var). iSAID, "6 piksellik kutudan maske"
   sorusunun en büyük etiketli örneklemi.
3. Aşama A'ya (distilasyon etiket okumaz) hacim olarak girebilir.

**Uyarılar:** lisans akademik kullanım (ticari yasak); `isaaccorley/isaid`
**anotasyonu** veriyor ama **görüntüyü vermiyor** — DOTA-v1.0'ı ayrıca
indirmek gerekiyor (Baidu + Google Drive, resmî sayfadan).

### İkinci ayna: görüntü + maske bir arada

`ariG23498/iSAID` DOTA avına çıkmadan çalışabilen tek yol. Bir shard
(710 MB) indirilip açıldı, sayılar HF datasets-server'dan doğrulandı:

```
sütunlar: id · image · ins · seg · category_id · category_name
771 satır (tahmini tam: 1 312) · 6,03 GB · tek train split
image/ins/seg üçü de tam çözünürlük (ölçülen örnek: 3875x5502), tile YOK
ins  = örnek maskesi (o karede 445 farklı renk)
seg  = semantik (5 renk)
```

Satır başına bir **(görüntü × sınıf)** çifti — kare başına değil. Sınıf
kapsaması **15'in 7'si**: Small_Vehicle 383, Large_Vehicle 316, ship 31,
plane 21, Harbor 18, Swimming_pool 1, storage_tank 1.

Yani: kolay ama kısmi. Tam set isteniyorsa `isaaccorley/isaid`
anotasyonu + DOTA görüntüleri; hızlı bir deneme isteniyorsa bu.

---

## 3. SAMRS — bu reponun defter 13'ünün zaten yapılmış hali

[SAMRS](https://github.com/ViTAE-Transformer/SAMRS) (NeurIPS 2023 D&B),
uzaktan algılama tespit setlerinin kutularını SAM'e prompt olarak verip
maske üretmiş. Yani **`src/training/pool.py`'nin tam olarak yaptığı iş**,
uydu tarafında, 2023'te yapılmış ve yayınlanmış.

Yayından (indirilemedi, aşağıya bak):

| alt set | kaynak | görüntü | sınıf | boyut | prompt |
|---|---|---:|---:|---|---|
| SOTA | DOTA-V2.0 | 17 480 | 18 | 1024² | H-Box |
| SIOR | DIOR | 23 463 | 20 | 800² | H-Box |
| FAST | FAIR1M-2.0 | 64 147 | 37 | 600² | RH-Box |
| **toplam** | | **105 090** | | | **1 668 241 örnek** |

**Neden doğrudan kullanılamadı:** dağıtım OneDrive + Baidu. HF'de aynası
yok (`search=samrs` → sıfır sonuç). Yani `fetch_datasets.py`'ye bir tarif
yazılabilir ama otomatik indirme yolu bugün yok.

**Yine de okunması gereken bir sonuç var:** SAMRS'ın kendi kabul/red
disiplini bu reponun dört kapısından (`labels.Gates`) zayıf ve kalibrasyon
adımı yok. Defter 14'ün ilk çıktısının bir *ölçüm* olması (termal mi RGB mi
promptlansın) SAMRS'ın atladığı adım. Aynı işi yapan iki boru hattından
bu reponunki daha ihtiyatlı — bu, 41ac1ab'nin savunulabilir tarafı.

---

## 4. Metin promptlu setler — SAM 3 için, EdgeTAM için değil

"image + prompt + mask" ifadesinin ikinci okuması: prompt = **metin**.
Havadan/uzaktan algılamada bunun iki seti var ve ikisi de indirilebiliyor.

### RRSIS-D — indirildi ve doğrulandı

`VoyageWang/rrsis-d` (5,07 GB: `JPEGImages.zip` + `instances.json` +
`refs(unc).p`). `instances.json` (51,8 MB) indirildi, açıldı, RLE'leri
decode edildi:

```
17 402 görüntü · 17 402 anotasyon · 20 sınıf · 800×800
görüntü başına TAM OLARAK BİR anotasyon (distinct image_id = 17 402)
RLE temiz decode oluyor; 400 örnekte geçersiz yok
gerçek maske pikseli: medyan 11 298 px (iSAID'in tam tersi — büyük nesneler)
```

**İki tuzak, ikisi de ölçülerek bulundu:**

1. **`bbox` COCO değil.** `[411, 53, 642, 279]` ve maskenin gerçek yayılımı
   `x[408,647] y[56,279]` — yani biçim **`x1,y1,x2,y2`**, COCO'nun
   `x,y,w,h`'si değil. COCO okuyucusuna verilirse kutular sessizce bozulur.
2. **`area` alanı piksel alanı değil.** Kayıtlı `area` medyanı gerçek maske
   pikselinin **~233 katı**. İlk örnekte 10 633 943 yazıyor — 800×800 = 640 000
   olan bir görüntüde. Bu alana göre filtreleme yapan her kod yanlış çalışır.

**Ama EdgeTAM'a yaramaz.** EdgeTAM SAM 2 mimarisi: prompt encoder'ı nokta,
kutu ve maske alır — **metin almaz**. Bu set ancak öğretmen SAM 3 iken ve
metin promptu üzerinden anlamlıdır. Ayrıca görüntüler yine uydu (GSD 0,5–30 m).

`JessicaYuan/RefSegRS` (2,77 GB tek zip) aynı ailenin küçük kardeşi;
indirilebilir olduğu doğrulandı, içine bakılmadı.

---

## 5. Termal: A sınıfı yok, ve bu ölçülmüş bir yokluk

Havadan **termal** tarafta örnek maskeli bir set aranırken bulunan her şey:

| aday | ne çıktı | neden olmadı |
|---|---|---|
| SegFly | semantik | `vehicle=13`, sahnedeki bütün arabalar aynı 13 |
| Kust4K | semantik | aynı |
| MVUAV, Caltech Aerial RGB-T | semantik | aynı |
| HIT-UAV | **sadece kutu** (2 898 kare) | maske yok — havuzun girdisi |
| DroneVehicle | **sadece kutu** (28 442 çift) | maske yok — havuzun girdisi |
| Anti-UAV410 | sadece kutu | maske yok |
| **AUVD-Seg300** | **örnek maskesi var** (YOLO poligon) | **300 görüntü**, ve *gökyüzüne bakan* anti-UAV — havadan değil |
| UAVDB | maskeler **SAM2 ile üretilmiş** | bizim havuzumuzun aynısı, üçüncü şahıs kalitesiyle |

**Sonuç:** havadan termalde elle çizilmiş örnek maskesi pratikte yok.
Defter 14'ün var olma sebebi tam olarak budur ve araştırma bunu
çürütmüyor — doğruluyor.

### Havuza yeni girdi: MONET

Aramada çıkan ve `docs/datasets.md`'de **olmayan** tek ciddi aday:

[MONET](https://github.com/fabiopoiesi/monet_dataset) (CVPR-W 2023, FBK) —
**~53 000 termal drone karesi, 162 000 elle çizilmiş kutu**, insan + araç,
kırsal sahneler, **COCO biçiminde**, ve her kare drone metadatasıyla
(irtifa, hız, GPS, attitude) zaman damgalı hizalı.

Neden ilginç: HIT-UAV 2 898 kare. MONET **18 katı**, aynı modalitede, aynı
kutu-tabanlı boru hattına `boxes.coco_frames` ile değişiklik gerektirmeden
girer, ve irtifa metadatası kalibrasyonda boyut kovası olarak doğrudan
kullanılabilir.

**Engel:** dağıtım FBK SharePoint üzerinden ve programatik istek **HTTP 403**
dönüyor (302 → `onedrive.aspx` → 403). Yani Baidu'yla aynı sınıf problem:
tarayıcı gerekiyor, `fetch_datasets.py` tarifi bugün yazılamaz. Kullanıcı
elle indirip Drive'a koyarsa `local` tarifi zaten bunu karşılıyor.

---

## 6. Kullanılmaması gerekenler

Arama sırasında "tam set" gibi görünüp değerlendirilince örneklem çıkanlar —
bir kere ölçüldü, bir daha bakılmasın:

| HF deposu | görünen | gerçek |
|---|---|---|
| `dronefreak/UAVid-2020` | UAVid | **42 dosya**, 0,51 GB — örneklem |
| `dronefreak/Aeroscapes` | Aeroscapes | **43 dosya**, 0,01 GB — örneklem |

İki iSAID aynası bunlardan farklı ve ikisi de kullanılabilir, farklı
sebeplerle: `isaaccorley/isaid` **tam** train + val anotasyonu (152 MB +
50 MB, görüntü yok); `ariG23498/iSAID` görüntü + örnek maskesi bir arada
(6,03 GB) ama 15 sınıfın 7'si.

---

## 7. Karar

Aşama B'nin veri sorusu tek cümleyle: **havadan RGB'de A sınıfı bol,
havadan termalde yok, ve bu projenin ihtiyacı termal.**

| ne için | ne kullan | neden |
|---|---|---|
| termal aşama B hacmi | **defter 14'ün havuzu** (HIT-UAV + DroneVehicle) | alternatifi yok — ölçüldü |
| termal hacmi büyütmek | **MONET** (elle indirme) | 53 K kare, aynı boru hattı, 18× HIT-UAV |
| havuz maskesinin *şeklini* ölçmek | **iSAID val** (116 649 çizilmiş maske) | bugünkü kalibrasyon yeniden kurulmuş hedefe karşı |
| RGB aşama B | defter 13'ün havuzu + (opsiyonel) iSAID | iSAID'in ölçeği farklı, karıştırılırsa etiketlenmeli |
| metin promptu | RRSIS-D — **ama sadece SAM 3 ile** | EdgeTAM metin promptu almıyor |
| aşama A (etiketsiz) | hepsi | distilasyon etiket okumuyor |

Bir şey **yapılmamalı**: iSAID'i "havadan araç" diye Kust4K/SegFly'ın yanına
etiketsiz karıştırmak. 13 piksellik medyan uzun kenar, bu projenin 300×150
piksellik hedefiyle aynı dağılım değil ve `datasets.md`'nin `role` alanı
(train/val/test) bunu ayırmak için var.

---

## 8. Yeniden üretim

```bash
pip install pycocotools scipy pillow numpy
```

```bash
# iSAID anotasyonları (ungated, anonim, 50 MB)
curl -sSL -o isaid_val.tar.gz \
  https://huggingface.co/datasets/isaaccorley/isaid/resolve/main/isaid_annotations_val.tar.gz
tar xzf isaid_val.tar.gz
```

```python
import json, numpy as np
from pycocotools import mask as mu
from scipy import ndimage

d = json.load(open("val/Annotations/iSAID_val.json"))
names = {c["id"]: c["name"] for c in d["categories"]}
print(len(d["images"]), "görüntü", len(d["annotations"]), "örnek maskesi")

# örnek kimliği dosyada: her nesne ayrı renk
from PIL import Image
a = np.array(Image.open("val/Instance_masks/images/P0003_instance_id_RGB.png"))
ids = (a[..., 0].astype(np.int64) << 16) | (a[..., 1].astype(np.int64) << 8) | a[..., 2]
print("farklı örnek kimliği:", len(np.unique(ids)))     # 60 = 59 örnek + arka plan
```

```python
# RRSIS-D: bbox x1y1x2y2, ve `area` piksel alanı DEĞİL
import json
from pycocotools import mask as mu
d = json.load(open("rrsis_instances.json"))
a = d["annotations"][0]
rle = a["segmentation"][0] if isinstance(a["segmentation"], list) else a["segmentation"]
if isinstance(rle["counts"], str):
    rle = {"size": rle["size"], "counts": rle["counts"].encode()}
print("gerçek piksel:", mu.decode(rle).sum(), "| kayıtlı area:", a["area"])
```

## 9. Bağlantılar

- iSAID: https://captain-whu.github.io/iSAID/ · ayna
  https://huggingface.co/datasets/isaaccorley/isaid · görüntüler DOTA-v1.0
  https://captain-whu.github.io/DOTA/
- SAMRS: https://github.com/ViTAE-Transformer/SAMRS ·
  https://arxiv.org/abs/2305.02034 (OneDrive/Baidu)
- RRSIS-D: https://github.com/Lsan2401/RMSIN · ayna
  https://huggingface.co/datasets/VoyageWang/rrsis-d
- RefSegRS: https://huggingface.co/datasets/JessicaYuan/RefSegRS
- MONET: https://github.com/fabiopoiesi/monet_dataset ·
  https://arxiv.org/abs/2304.05417 (SharePoint, 403)
- AUVD-Seg300: https://github.com/chen-yuzhi/YOLO11-AU-IR ·
  https://doi.org/10.1371/journal.pone.0330074
