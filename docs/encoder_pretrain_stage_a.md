# Aşama A — dört kol, tek çıktı

`docs/encoder_training_todo.md` aşama A'yı tek bir yol olarak planlamıştı:
hizalı RGB-T çiftleri üzerinde bir RGB temel modelini termal encoder'a distile
et. Bu belge o planı **dört kollu bir seçime** çeviriyor ve seçimi neyin
kararlaştıracağını söylüyor.

Kod: `src/training/pretrain.py` (hub), `src/training/automask.py` (maske kolu),
`tools/pretrain_stage_a.py` (giriş noktası).
Defterler: `notebooks/15_pretrain_automask.ipynb` (maske kolu),
`notebooks/16_pretrain_vitb.ipynb` ve `notebooks/17_pretrain_convnext.ipynb`
(öğretmen kolu — **tek string** farkla, eğiten her kod hücresi bayt bayt aynı).

---

## 0. Dört kol

| `ARM` | encoder'ı ne denetliyor | hangi veriyi yiyebiliyor |
|---|---|---|
| `none` | hiçbir şey — stok EdgeTAM ağırlıkları, kopyalanıyor | yok |
| `mask` | modelin **maskesiz** kare üzerindeki kendi öznitelikleri | **her şey**; tek modalite yeter |
| `distil` | donmuş bir öğretmenin eşlenik girdideki öznitelikleri | öğretmen tarafında bir görüntü şart |
| `both` | önce `distil`, üstüne `mask` | `distil` ne yiyebiliyorsa |

**Dördü de aynı dosyayı yazıyor:** sıradan bir EdgeTAM state dict'i, yanında
`stage_a.json` manifestosu. Aşama B'ye geçiş tek bayrak:

```bash
python tools/train_encoder.py --base <bu dosya> ...
```

`none` de bir dosya yazıyor ve bu göründüğünden önemli: **farklı bir kod
yolundan geçen taban çizgisi kayan taban çizgisidir.** Aşama B stok ağırlıkları
bir yerden, ön-eğitilmişleri başka yerden yüklerse, er geç kimsenin niyet
etmediği bir şey de değişir.

---

## 1. Asıl soru: öğretmen RGB, dağıtım termal

Elde üç tür külliyat var ve üçü farklı muamele istiyor:

| modalite | öğretmen ne görüyor | öğrenci ne görüyor | ne olduğu |
|---|---|---|---|
| `paired` | RGB yarısı | hizalı termal yarısı | yayınlanmış yol; tek gerçek çapraz-modal dayanak |
| `rgb` | renkli görüntü | **aynı görüntünün parlaklığı** | vekil: aynı tek-kanal-girer yapısı, yanlış sensör |
| `thermal` | termal kare | aynı termal kare | RGB öğretmen kendi modalitesi dışında |

`rgb` kolunda öğrenciye tam renk vermek, sensörün sahip olmadığı bir kanalı
kullanmayı öğretmek olurdu — dağıtılan encoder hayatı boyunca renk görmeyecek.
Öğrenciye **luminansı** vermek görevin şeklini koruyor: tek kanal girer,
üç-kanallı semantik çıkar.

### `thermal` kolu ölçülmeden kullanılmaz

`pretrain.modality_gap` bu ölçüm. Hizalı çiftlerde öğretmenin RGB yarısındaki
öznitelik haritasını termal yarısındaki haritasıyla karşılaştırıyor ve iki
referansa göre okuyor: **chance** (iki alakasız kare) ve **floor** (aynı kare,
hafif parlaklık değişimiyle). Üçüncü satır **luminance**, yalnızca rengi
kaybetmenin maliyetini söylüyor — termal boşluğunun aslında termalle ilgili
olmayan kısmı.

Beklenecek cevap yayınlanmış ve **ikiye ayrılıyor**. Caltech'in havadan RGB-T
seti (arXiv 2403.08997, ECCV 2024) tam bu projenin alanı — 120 m altı drone,
eş-kayıtlı yarımlar — ve aynı kareler üzerinde iki şeyi birden ölçmüş:

* **Temel modelin *çıktıları* termalde çöküyor.** SAM'in termal yarıdaki örnek
  maskeleri, kendi RGB maskelerine karşı **AP 0.018** alıyor; aynı görüntünün
  *gri tonlamalı* hâlinde 0.538. Açık-sözlük segmenterler 0.13–0.27 mIoU
  kaybediyor. Bu yüzden bir öğretmenin **tahminini** termalden distile etmek
  değersiz: bunu deneyen tek çalışma, **hiç distile etmemekten 2.9 mAP50 daha
  kötü** sonuç bildiriyor (arXiv 2606.11572).
* **Öznitelikleri büyük ölçüde hayatta kalıyor.** Aynı sette **donmuş** DINOv2,
  doğrusal olmayan bir başlıkla termal segmentasyonda **0.706 mIoU** alıyor;
  uçtan uca termalde eğitilmiş en iyi ağ 0.725, termale özel FTNet ise 0.613.

Bu aşama öznitelik kopyalıyor ve hiçbir tahmine dokunmuyor — sorunun sorulmaya
değer olmasının sebebi bu.

---

## 2. Öğretmen seçimi

Birden fazla kez ölçülmüş tek bulgu: **öğretmen kapasitesi öğrenciye
eşlenmeli, maksimize edilmemeli.** EdgeCrafter (arXiv 2603.18739, TMLR 2026)
öğrenciyi sabit tutup öğretmeni değiştiriyor: DINOv3-S (21 M) **54.0** COCO AP,
DINOv3-B (86 M) **54.3**, DINOv3-L (300 M) **52.6**. Ters-U, tepe ~86 M'de.
Proteus (ICLR 2025) 21 M'lik bir öğrenci için aynı şekli daha dik veriyor:
86 M → 85.8, 300 M → 82.2, 1.1 B → 80.6, 12 veri setinin 11'inde tutarlı.

**Bunun yerleştirmediği şey.** İkisi de tek koşu. Yankıladıkları klasik
kapasite-boşluğu sonucu bir *logit* distilasyonu bulgusu; öznitelik tabanlı bir
hedefi test eden tek çok-seed'li çalışma (Cho & Hariharan, AT+KD) eğilimin
düzleştiğini ve optimumun **çok daha büyük** bir öğretmene kaydığını buluyor.
TinyMIM tam bu öğrenci ölçeğinde (5.7 M) 86 M'lik öğretmenin 21 M'liği
yendiğini bildiriyor. Yani kullanışlı okuma: **öğrencinin 5–15 katı arası, 50
kat tehlikeli bölge** — 6 M'lik bir encoder için **30 M – 90 M**.

| öğretmen | param | 6 M'ye oran | ADE20k mIoU (DINOv3'ün kendi tablosu) | stride |
|---|---|---|---|---|
| ConvNeXt-T | 29 M | 4.8× | 42.7 | 32 |
| **ConvNeXt-S** *(defter 17)* | **50 M** | **8×** | **44.8** | 32 |
| ConvNeXt-B | 89 M | 15× | 46.3 | 32 |
| ViT-S/16 | 21 M | 3.5× | 47.0 | 16 |
| ViT-S+/16 | 29 M | 4.8× | **48.8** | 16 |
| **ViT-B/16** *(defter 16)* | **86 M** | **14×** | **51.8** | 16 |
| ViT-L/16 | 300 M | 50× | 54.9 | 16 |

Son iki sütun birlikte okununca: **eşit parametrede DINOv3'ün ViT öğrencileri
ConvNeXt öğrencilerinden gözle görülür şekilde daha güçlü yoğun öznitelik
taşıyor** — 29 M'de 48.8'e karşı 42.7. Üstelik ViT-B/16 burada *daha ucuz*,
çünkü stride-16 haritası 512'de zaten öğrencinin 32×32 gridi; stride-32 bir
ConvNeXt aynı gridi üretmek için 1024'te koşmak zorunda.

Bu yüzden sıra net: **önce 16 (ViT-B), sonra 17 (ConvNeXt-S).** 17 bir
alternatif değil bir **kontrol**: o sayıların söyleyemediği tek şey,
**konv→konv** aktarımının evrişimsel bir öğrenciye ViT→konv'dan daha iyi geçip
geçmediği — EdgeTAM'ın gövdesi RepViT, ve yayınlanmış her öğretmen-boyutu
sonucu bir transformer'a ölçülmüş. Bir string ve bir yeniden koşu; ikisi de
aynı çıktıyı yazdığı için aşama B doğrudan sıralıyor.

Kesin olan: **ViT-L/16 ve üstü kullanma.** 50 kat, iki ölçülmüş çalışmanın da
kaybettiği oran.

### ConvNeXt öğretmen neden 1024, ViT neden 512

ConvNeXt'in son aşaması stride **32**, öğrencinin haritası stride **16**. Aynı
512 verilirse öğretmen 16×16 üretir, öğrenci 32×32; kayıp öğretmeni
büyütür — dört öğrenci hücresini tek öğretmen hücresiyle denetler, ki hedef
yirmi piksel genişken tam da kaybedilmemesi gereken çözünürlük odur.
`TEACHER_SIZE = None` bu yüzden `(512 / 16) × 32 = 1024`'e çözülüyor.

Stride **ölçülüyor**, config'den okunmuyor: DINOv3'ün ConvNeXt config'lerinde
`patch_size` alanı hiç yok, dolayısıyla onu okuyan her şey bir varsayılan alıyor
ve iki kat yanılıyor. `distill.probe_geometry` gövdeyi 128 ve 256'da koşup token
sayısının nasıl büyüdüğünden stride'ı çözüyor.

---

## 3. Kayıplar

| `LOSS` | ne yapıyor | kanıt |
|---|---|---|
| `cosine` *(varsayılan)* | pozisyon başına kosinüs uzaklığı | deponun kendisi; ölçek-bağımsız ve çapraz-modalitede geri gittiği bildirilmemiş tek basit hedef |
| `feature` | beyazlatılmış smooth-L1, β = 2.0 | kanonik yayınlanmış öznitelik-distilasyonu tarifi (arXiv 2205.14141) — ama **aynı modalitede** doğrulanmış |
| `frequency` | düşük frekansta sert, yüksekte doyuran | DINO ailesinden RGB→termal distilasyonda ölçülmüş (arXiv 2606.11572): **64.1 mAP50**, kosinüs 62.1, düz MSE 61.1, hiç distile etmemek 61.7 |

Son satırın çoğu bir uyarı: **düz tam-bant MSE, hiç distile etmemenin altında
kaldı**; yanıt seviyesi distilasyon 2.9 daha da altında. Ölçülmüş sebep, iki
modalitenin özniteliklerinin yüksek bantta düşük banta göre **~2.4 kat** daha
fazla ayrışması — termal görüntünün ince dokusu gerçekten farklı, yerleşimi
değil.

`GRAM` ilişkisel bir terim ekliyor: pozisyonların kendisini değil,
**pozisyonlar arası benzerlik matrisini** eşliyor. DINOv3 bunu kullanıyor (Gram
anchoring, ağırlık 2) ve bağımsız bir üç-koşulu çalışmada ilişkisel distilasyon
düz benzerlik distilasyonunu **+2.1 – +2.7 zero-shot mIoU** geçiyor. Ayrıca
hizasızlığa dayanan tek mekânsal hedef, çünkü *p* pozisyonunda ne olduğunu hiç
sormuyor — yalnızca *p* ile *q*'nun benzeyip benzemediğini.

`JITTER` bunun diğer yarısı. Bu arşivler piksele hizalı değil: DroneVehicle'ın
yazarları kutuların **%20'sinden fazlasının** 3 px veya 3 dereceden fazla
saptığını bildiriyor. Ölçülmüş hasar eğrisi (AR-CNN, ICCV 2019): 1–2 px bedava,
4 px birkaç puan, 6 px ~%65 göreli. O çalışmanın çaresi kaymayı *yok etmek*
değil **altında eğitmekti** — kaymalar arası varyansı 11.55'ten 1.24'e
düşürdü. `JITTER = 2` o çare.

---

## 4. Maske kolu, ve dürüst beklenti

Yöntem: encoder'ın bir kopyası temiz kareyi görüp hedef haritayı üretiyor;
eğitilen encoder 16-piksellik hücrelerinin yarısı ortalama piksele boyanmış
kareyi görüyor ve o haritayı **boyanmış pozisyonlarda** yeniden üretmek zorunda.
Maske hedefin kendi öznitelik haritasından seçiliyor — karenin ortalama
yönünden en uzak hücreler — ki bu, RepViT gövdesinde okunacak bir dikkat
haritası olmadığı için AttMask'ın "sınıf token'ının baktığını maskele"sinin
evrişimsel karşılığı.

**Seyrek evrişim yok.** ConvNeXt V2'nin FCMAE'si ona ihtiyaç duydu çünkü maskeli
bir görüntü üzerinde yoğun evrişim sızdırıyor. O argüman hedef *piksel* olduğunda
en çok ısırıyor; burada hedef bir öznitelik haritası ve decoder yok, yani sızıntı
çok daha az kazandırıyor — alternatifin maliyeti ise deponun sahibi olmadığı bir
gövdeyi yeniden yazmak.

**Ama sızıntı sıfır değil ve bir parçasının adı var.** RepViT'in geç
aşamalarında squeeze-and-excitation blokları var; SE kapısı harita üzerinde
global bir ortalama. Maskeli girdi üzerinde bu ortalama görünür orana göre
sapıyor, ve aşama B'de maske hiç yok. Buradaki hafifletmeler ucuz olanlar:
dolgu ortalama piksel (mevcut en bilgisiz değer) ve `MASK_RATIO` MAE'nin
0.75'i yerine 0.5. Bu kol kazanırsa, havuzlamayı yalnızca görünür pozisyonları
sayacak şekilde yamamak (SparK'ın yaptığı) denenecek ilk şey.

### Ölçülmüş önsel

| ölçülen | sonuç |
|---|---|
| ConvNeXt V2'nin FCMAE'si **değiştirilmemiş** mimaride (kendi Tablo 14'ü) | −0.1 ile +0.1 top-1. Başlıktaki +1.0, yanında GRN katmanını birlikte tasarlamaktan geliyor |
| **5.7 M**'lik bir ViT'i MAE ile ön-eğitmek (TinyMIM) | sıfırdan eğitmeye karşı **−0.6** |
| öğrenilmiş maske vs rastgele, diğer her şey sabit (HPM'in izolasyon ablasyonu) | +0.46; ve +0.72'sinin +0.26'sı sadece yardımcı tahminciyi eklemekten |
| **4.8 M** parametrede havadan termal segmentasyon (Caltech RGB-T, ECCV 2024) | sıfırdan 0.687, **ImageNet 0.725**, 40 k termal görüntüde öz-denetim 0.714 |

Son satır oturup düşünülecek olan: bu projenin model ölçeğinde ve alanına
yakın, sıradan RGB ön-eğitimi termal öz-denetimi yendi. Bu kolu anlamsız
yapmıyor — onu bir **ölçüm** yapıyor, ve ölçtüğü şey dağıtımın kendi
modalitesindeki termal-only bir külliyatın, diğer kolun yapısal olarak
erişemediği bir şey katıp katmadığı.

`MASK_MODE = "green"` burada saygı duyulacak ayar. ColorMAE (ECCV 2024):
bant-geçiren filtrelenmiş gürültü, parametre yok, ek ileri geçiş yok, ve kendi
kontrollü karşılaştırmasında önceki üç yılın öğrenilmiş maskeleyicilerine denk
geldi. `hint`, aşama B'nin sayısında `green`'i yenemezse öz-üretilen maske
hiçbir şey kazandırmamıştır ve dürüst olan bunu yazmaktır.

---

## 5. Külliyat eklemek

Bir külliyat bir satır. Ya `aerial.SPECS`'in zaten bildiği bir spec adı:

```python
Corpus("kust4k", "/content/data/Kust4K", "paired", spec="kust4k")
```

ya da glob'ları açıkça yaz — kimsenin spec yazmadığı bir set için:

```python
Corpus("visdrone", "/content/data/VisDrone", "rgb", rgb="**/images/*.jpg")
```

Başka hiçbir şeye dokunmak gerekmiyor: kırpma diskteki gerçek resimlerden
türetiliyor, eşleme anahtarı `list_pairs`'ın zaten doğru yaptığı, ve indeks
hücresi külliyat başına ne bulduğunu basıyor — hiçbir şeyle eşleşmeyen bir glob
için gürültülü bir `!!` dahil, ki bu işin yanlış gitme biçimi budur.

Literatürden doğrudan uygulanan bir uyarı: AnyThermal, **yalnızca** havadan
veriyle distile edilen bir varyantın kentsel sahnelerde *distile edilmemiş RGB
modelden daha kötü* çıktığını bildiriyor. Dar distilasyon kapsamadığı şeye zarar
veriyor. **Külliyatları karıştır.**

---

## 6. Aşama B'nin verisi aşama A'ya verilebilir mi?

**Evet, ve normal olan bu.** Aşama A hiçbir etiket okumuyor — `paired` kolunda
öğretmen RGB yarısına, öğrenci termal yarısına bakıyor; `rgb` kolunda öğrenci
aynı görüntünün luminansına bakıyor; `mask` kolunda hiçbir öğretmen yok. Yani
bu aşamanın yiyebileceği şey **çözülebilen her görüntü**, ve aşama B'nin
kullandığı setler de buna dahil. SAM 2'nin kendi tarifi de böyle: statik
görüntü aşama C'de bile atılmıyor, videoyla dönüşümlü besleniyor.

Üç gerçek uyarı var, ve üçü de "yapma" değil "şunu bil" biçiminde.

### 6.1 Aşama B'nin *not verdiği* kareler

En önemlisi ve tek ciddi olanı. Aşama B `split_frames` ile **dizi bazında**
train/val/test ayırıyor. Aşama A o test dizilerinin karelerini de görürse,
aşama B'nin `test/instance_iou` sayısı şişer — etiket sızıntısı değil ama
**transdüktif ön-eğitim**, ve ölçülmüş bir etkisi var: SCOTT'un kendi tablosu
1 020 → 8 189 görüntüde 74.25 → 97.15 gidiyor ve bunun büyük kısmı eğitim
setinin test setini içermesinden geliyor.

Bu deponun kuralı zaten net: **yeniden kurulmuş setler eğitir, gerçek maskeli
setler not verir** (`aerial.Source.role`). Not veren set **VTUAV VIS** — gerçek
örnek maskesi taşıyan tek set. Yani pratik kural:

> Her şeyi ver, **ama VTUAV VIS'i aşama A'ya koyarken bunu raporda söyle** —
> ya da koyma. DroneVehicle, VisDrone, HIT-UAV, SegFly, Kust4K, RGBT234,
> LasHeR: hiçbiri not vermiyor, hepsi serbest.

En temiz sayıyı istiyorsan VTUAV VIS'i aşama A'nın külliyatından çıkar. Bu
kadar. (Dizi-bazlı bir dışlama filtresi yazmak mümkün ama iki tarafın grup
anahtarını *tam olarak* aynı hesaplaması gerekiyor, ve sessizce hiçbir şey
dışlamayan bir filtre bu deponun en sevmediği hata biçimi.)

### 6.2 Ardışık kareler neredeyse aynı görüntü

Bu setlerin çoğu videodan geliyor. VTUAV 500 dizi × ~3 400 kare; ilk 5 000
çift **iki uçuş** demek. `Corpus(..., limit=N)` bunu çözüyor: külliyat boyunca
**yayarak** örnekliyor, baştan kesmiyor. 20 000 karelik bir `limit` genelde
20 000 karelik bir külliyattan daha çeşitli.

### 6.3 Dar külliyat kapsamadığı şeye zarar veriyor

AnyThermal'ın ölçümü: **yalnızca** havadan veriyle distile edilen varyant,
kentsel sahnelerde *distile edilmemiş RGB modelden daha kötü* çıkıyor. Aynı
çalışmanın sonucu — "çeşitlilik ölçekten daha kritik" — bu aşamanın külliyat
listesinin neden karışık olması gerektiği. Havadan termal + havadan RGB +
yerden RGB-T karışımı, saf havadan termalden iyi.

### 6.4 Peki hangi setler?

| set | modalite | aşama A'da rolü |
|---|---|---|
| DroneVehicle | `paired` | 28 442 çift, 640×512 (100 px beyaz bant kırpıldıktan sonra), gündüz **ve gece** — çapraz-modal dayanağın çoğu |
| Kust4K | `paired` | 4 024 çift, en temiz kayıt — **modalite probunun evi** |
| SegFly / Caltech RGB-T | `paired` | çeşitlilik; Caltech donanım-senkronize havadan |
| RGBT234 / LasHeR | `paired` | yerden RGB-T; havadan değil, ama §6.3'ün istediği çeşitlilik |
| VisDrone | `rgb` | havadan RGB, termal ikizi yok — `rgb->gray` vekil yolu, ve genel kapasitenin kaynağı |
| HIT-UAV | `thermal` | 2 898 havadan termal kare — küçük ama dağıtımın kendi modalitesi |
| Anti-UAV410 | `thermal` | 410 dizi, minik İHA'lar; maske kolunun doğal yemi |
| VTUAV VIS | `paired` | 1.7 M çift — **ama aşama B bunun üzerinde not veriyor**, §6.1 |

Yani mentorun haklı: aşama B'nin verisini de ver. Tek istisna not veren set.

---

## 7. Ne zaman bitmiş sayılır (§8 iki-model planı için aşağıda)

Aşama A'nın ürettiği kayıp bir **vekil**, ve iki kolun vekilleri aynı ölçeği
bile paylaşmıyor. Karar veren sayı şu üç koşunun tablosu:

| aşama A | aşama B `test/instance_iou` |
|---|---|
| `none` | — |
| `mask` (defter 15) | — |
| `distil`, ViT-B (defter 16) | — |
| `distil`, ConvNeXt-S (defter 17) | — |

Diğer her şey — veri setleri, program, seed, prompt karışımı — sabit tutulmak
zorunda; yoksa sayılar karşılaştırılabilir değil ve karşılaştırma bu aşamayı
koşmanın tek sebebiydi.

---

## 8. İki model: RGB ve termal, ayrı ayrı beslenmesi

Dağıtımda göreve göre bir RGB bir termal encoder arasında geçiliyorsa, bu
**iki ayrı ön-eğitim** demek — ve aralarındaki tek fark öğrencinin hangi yarıyı
gördüğü. `STUDENT` ayarı bunu tek satıra indiriyor:

```python
STUDENT = "thermal"   # veya "rgb"
```

`for_student` külliyat listesini o öğrenciye göre yeniden yönlendiriyor:

| külliyat | `STUDENT="thermal"` | `STUDENT="rgb"` |
|---|---|---|
| `paired` | öğretmen RGB, öğrenci hizalı termal | ikisi de RGB yarısı (`rgb->rgb`) |
| `rgb` | öğretmen renk, öğrenci luminans | ikisi de renk (`rgb->rgb`) |
| `thermal` | ikisi de termal kare | **düşürülüyor** ve hangileri düşürüldüğü basılıyor |

RGB öğrenci için her yol `rgb->rgb` oluyor: öğretmen ve öğrenci **aynı**
augmentasyonlu görüntüyü görüyor ve öğrenci öğretmenin haritasını yeniden
üretmeyi öğreniyor. Bu, arXiv 2205.14141 ve Proteus'un kanonik tarifi — yani
RGB tarafı, termal taraftan **daha iyi doğrulanmış** bir problem üzerinde
çalışıyor.

Termal-only külliyatlar RGB öğrenci için düşürülüyor, yönlendirilmiyor: maske
koluna verilebilirlerdi (öğretmen istemiyor) ama görüntüler termal olurdu, ve
renkte dağıtılan bir encoder'ın hiç görmeyeceği bir radyometri üzerinde
çözülmüş bir pretext görevinden faydası yok.

### 8.1 Termal encoder — külliyat bütçesi

Bugün `tools/fetch_datasets.py`'nin indirebildikleriyle:

| külliyat | modalite | kare/çift | çözünürlük | disk | not |
|---|---|---|---|---|---|
| VTUAV `train_001` | `paired` | **26 059 çift** | 1920×1080 | 9,1 GB | tek başına aşama A'yı doyuruyor; **hiç yaya yok** — `train_003` yanına gelmeli |
| DroneVehicle `train` | `paired` | **17 990 çift** | 640×512 (100 px bant kesildikten sonra) | 8,9 GB | gündüz **ve gece**; projenin native çözünürlüğü, hiç yeniden örnekleme yok |
| SegFly | `paired` | **15 007 çift** | termal 640×512 | 22 GB | ICP ile hizalanmış, yayıncının kendi kaydı %87 diyor |
| Kust4K | `paired` | **2 864** (4 024 − dataset'in bozuk dediği 1 160) | 640×512 | 2,5 GB | listedeki en temiz kayıt — **modalite probunun evi** |
| Caltech `pairs` | `paired` | **2 282 çift** | 960×600 | 4,3 GB | **listedeki en iyi hizalı**, ölçülen artık kayma 1–4 px; doğal arazi/su/gece |
| RGBT234 | `paired` | 233,8 K çift | — | 7,7 GB | yer seviyesi — havadan değil, ama çeşitliliğin kaynağı |
| LasHeR | `paired` | 730 K+ çift | — | ~224 GB | akışlı, dizi-seçmeli (`--sequences`); varsayılan değil |
| VisDrone | `rgb` → luminans | 6 471 | — | 1,8 GB | vekil yol; **AeroVIS kontaminasyonu, §8.3** |
| SegFly RGB | `rgb` → luminans | 3 000 (20 606'nın dilimi) | 4000×3000 | 23 GB | |
| HIT-UAV | `thermal` | 2 898 | 640×512 | 0,4 GB | havadan, 80–130 m |
| Caltech `singles` | `thermal` | 3 076 | 640×512 | 4,1 GB | içindeki `color/` **kayıtlı eş değil**, `rgb=None` |
| Anti-UAV410 | `thermal` | 410 dizi / 438 K kutu | 640×512 | 8,7 GB | ayrı fetcher (`fetch_antiuav410.py`) |

Makul bir ilk bütçe — **~64 000 hizalı çift + ~6 000 termal-only + ~9 500 RGB
vekil**, toplam ~50 GB disk:

```python
CORPORA = [
    Corpus("vtuav",        DATA_ROOT / "VTUAV_VIS",   "paired", spec="vtuav_vis", limit=20000),
    Corpus("dronevehicle", DATA_ROOT / "DroneVehicle","paired", spec="dronevehicle"),
    Corpus("segfly",       DATA_ROOT / "SegFly",      "paired", spec="segfly",    limit=12000),
    Corpus("kust4k",       DATA_ROOT / "Kust4K",      "paired", spec="kust4k"),
    Corpus("caltech",      DATA_ROOT / "Caltech",     "paired", spec="caltech_rgbt"),
    Corpus("visdrone",     DATA_ROOT / "VisDrone",    "rgb",    rgb="**/images/*.jpg"),
    Corpus("segfly_rgb",   DATA_ROOT / "SegFly_RGB",  "rgb",    rgb="**/*.jpg"),
    Corpus("hituav",       DATA_ROOT / "HIT_UAV",     "thermal",thermal="**/normal_json/**/*.jpg"),
]
```

**Ve dört set daha var, ama bu dalda değil.**
`claude/aerial-rgb-thermal-data-qhjpxd` dalının `fetch_datasets.py`'si 14 tarif
taşıyor, bu dalınki 10. Fark, hepsi **havadan** ve üçü **termal** olan dört
set:

| tarif | ne | kare | lisans |
|---|---|---|---|
| `rgbtdroneperson` | WHU-DroneDual, hizalı RGB-T, 640×512 | **6 125 çift**, 70 880 kutu | CC BY 4.0 |
| `vtuavdet` | VTUAV'ın aynı ekipçe yeniden kutulanmış hâli, 1920×1080 | **11 392 + 5 378** kare | CC BY 4.0 |
| `birdsai` | havadan gece termal, LWIR | **48 dizi / ~62 K kare**, ~166 K kutu + iz | **CDLA-Permissive-1.0** — listedeki tek ticari-serbest set |
| `airesq` | havadan termal, 640×512 | **1 988 kare** | CC BY 4.0 |

Dördü birlikte **~87 K yeni kutu-etiketli kare, ~79 K'sı termal tarafta**.
Aşama A etiket okumadığı için buradaki değer kutular değil **kareler**, ve
`birdsai`'nin 62 K havadan gece termal karesi bu listedeki en büyük tek
termal-only kaynak.

İki uyarı, o dalın kendi ölçümlerinden: **RGBTDronePerson "piksel hizalı" diye
duyuruluyor ama iki modalitenin kutuları medyan 11,7 px ayrı** ve medyan hedef
boyu zaten √alan 11–12 px — kayma bir hedef çapına eşit, yani hizalı çift
kaynağı olarak *kullanılmamalı*. **VTUAV-det'in kutuları tek modalitede
çizilip diğerine aynen taşınmış**, yani VTUAV'ın bilinen hizasızlığını sessizce
miras alıyor.

Bu dört tarifi buraya almak iki commit'lik bir cherry-pick
(`6a350f4`, `5ac1575`) ve `fetch_datasets.py` + `boxes.py` + testlerden ibaret.

### 8.2 RGB encoder — külliyat bütçesi

Aynı liste, `STUDENT = "rgb"`. Termal-only satırlar düşüyor, kalan her şey
`rgb->rgb` oluyor ve **paired setlerin RGB yarıları** devreye giriyor.

**Ama RGB tarafında bir seçim ölçütü var ve termal tarafta yok.**
`docs/rgb_aerial_kaynaklar.md` (dal `claude/rgb-aerial-data-sources-e73aml`)
adayları paperlardan değil **arşivleri açıp kutuları ölçerek** sıralıyor, ve
ölçüt irtifa değil piksel:

> **medyan √alan ≥ 45 px** ve **< 32 px oranı ≤ %30**

Bu ölçüte göre listenin yarısı eleniyor — **VisDrone dahil**:

| külliyat | medyan √alan | < 32 px | karar |
|---|---|---|---|
| AU-AIR | **81 px** | %11,8 | ✓ listedeki en iri nesneler |
| AeroVIS `sd_` dilimi | 84 px | %21,1 | ✓ ama 2 sınıf, hep su |
| **CODrone 60 m** | **60 px** | %20,4 | ✓ |
| **CODrone 30 m** | 47 px | %28,9 | ✓ |
| CODrone 100 m | 44 px | %30,2 | sınırda, opsiyonel |
| DroneVehicle RGB | 40 px | %19,0 | ✓ sınırda ama geçiyor |
| AeroScapes | 35 px | %40,0 | ✗ 720p bandı yiyor |
| **VisDrone2019-DET** | **32 px** | **%51,0** | **✗ bant dışı** |
| AeroVIS (bütün) | 31 px | %52,1 | ✗ maskelerinin yarısı 32 px altı |
| UAVid | 28 px | %58,7 | ✗ eğik bakış uzağı küçültüyor |
| AeroVIS `ud_` (UAVDT) | 24 px | %75,1 | ✗✗ |

Uydu setleri (DOTA, DIOR, iSAID, SODA-A, AI-TOD-v2, xView) kapsam dışı: havadan
ama İHA değil.

Bunun sonucu, RGB ön-eğitim bütçesinin **bugün fetcher'da olan setlerle
kurulamayacağı**. Sıralama:

| külliyat | bant-içi kare | çözünürlük | fetcher'da mı |
|---|---|---|---|
| **CODrone** (30 + 60 m dilimi) | **7 457** (60 m: 4 316 · 30 m: 3 141) | 3840×2160 | **hayır** — tarif yazıldı, koda girmedi |
| **AU-AIR** | 32 823 ham → **~1 100** (uçuş başına ~1 fps) | 1920×1080 | **hayır** — tarif yazıldı |
| **UAVScenes** | **120 000** (23 uçuş, her kare elle etiketli) | 2448×2048 | **hayır** — aday |
| VTUAV RGB yarısı | 26 059 (`train_001`) | 1920×1080 | evet |
| SegFly RGB | 20 606 (fetcher 3 000'lik dilim) | 4000×3000 | evet |
| DroneVehicle RGB yarısı | 17 990 (`train`) | 640×512 | evet |
| MVUAV | 53 828 | — | **hayır** |
| Kust4K RGB yarısı | 2 864 | 640×512 | evet |
| Caltech RGB yarısı | 2 282 | 960×600 | evet |
| MESSI | 2 525 | 5472×3648 | hayır — irtifa ekseni (30/50/70/100 m + iniş) |
| CARPK | 1 448 | — | hayır |
| VDD | 400 | 4000×3000 | hayır |
| SkyScapes | 16 | 5616×3744 | hayır — stres testi, eğitim seti değil |
| ~~VisDrone-DET~~ | ~~6 471~~ | — | evet, **ama ölçüt elemesi** |

Sentetiklere dikkat: Multi-View UAV'ın 357 690 karesi **400×300** (girdinin
altında) ve tamamı CARLA; FlyAwareV2'nin 288 K'sı yine CARLA, gerçek kısmı
~2 K. Hacim büyük, sinyal değil.

**Bugün fetcher'la kurulabilecek en iyi RGB bütçesi** ≈ **69 000 kare**:
VTUAV RGB (26 059) + SegFly RGB (20 606, fetcher 3 000 çekiyor) +
DroneVehicle RGB (17 990) + Kust4K RGB (2 864) + Caltech RGB (2 282). CODrone
ve AU-AIR tarifleri yazılırsa +8 500 bant-içi kare, UAVScenes eklenirse
+120 000.

### 8.3 Kontaminasyon — ve burada bir düzeltme var

`docs/datasets.md`'nin bu daldaki hâli şunu diyordu: *"13'ün havuzu
VisDrone'dan üretildiği için, bu havuzla eğitilen bir model AeroVIS'te
ölçülemez — kare kare aynı veri."* **Bu fazla sıkıymış ve ölçülerek
düzeltilmiş.** VisDrone'un resmî tanımı "288 video klip → 261 908 kare **ve**
10 209 statik görüntü"; ikisi **ayrı koleksiyon**. Defter 13 VisDrone-**DET**'i
(statik), AeroVIS ise VisDrone-**MOT**'u (video) kullanıyor → örtüşme kare
düzeyinde değil, sahne/kampanya düzeyinde.

Asıl tehlike başkaymış, ve listede yoktu:

| havuza girerse | AeroVIS'ten yanan |
|---|---|
| **VisDrone-MOT** | 52 dizi / 21 758 kare (**%44,2**) |
| **UAVDT-M** | 39 dizi / 19 276 kare (**%39,2**) |
| **SeaDronesSee** | 26 dizi / 8 170 kare (**%16,6**) |

Üçü birden girerse AeroVIS'ten ölçülecek bir şey kalmaz. Yani: **AeroVIS ölçüm
seti olarak kalacaksa, aşama C'nin masklet kaynağı bu üçü olamaz.** Ve bu üçü
listedeki en kolay video-kutu setleri, yani kısıt gerçek.

Diğer kurallar değişmedi:

- **UAVid → FlyAwareV2 (test) + VDD/IDD.**
- **VTUAV → VTUAV-det.** Aynı çekim; VTUAV'da eğitiliyorsa VTUAV-det bir
  değerlendirme seti olamaz, prompt kaynağı olabilir.
- **CARLA → FlyAwareV2 + Multi-View UAV** — kare değil, render imzası ortak.
- **OccuFly → SegFly** — bugün çakışma yok; OccuFly değerlendirmeye alınırsa
  SegFly tarafsız olmaktan çıkar.
- **MARS-LVIG → UAVScenes** — listede başka kullanan yok, temiz.

Ve **DroneVehicle VisDrone değildir** — AeroVIS'in üç kaynağından hiçbiri de
değil. RGB tarafında kontaminasyonsuz, bant-içi ve fetcher'da olan tek büyük
set o.

**Not:** `claude/aerotrack-dataset-review-5jcxzy` dalında AeroVIS'i yeniden
ölçen daha yeni bir `rgb_aerial_kaynaklar.md` var (1 378 603 hazır
`(görüntü, kutu, maske)` üçlüsü, medyan kenar 39,5 px, %34,1 COCO-small) ve
AeroVIS'in rolü konusunda iki dal **aynı fikirde değil**. Karar vermeden önce
o dosya okunmalı.

### 8.4 Termal modeli RGB ile beslemek mantıklı mı?

**Evet, ve bu bir taviz değil — ölçülmüş bir kazanç.**

| ölçüm | sonuç |
|---|---|
| FLIR termal detection, ViT-L: rastgele başlatma vs DINOv2 **RGB** ağırlıkları | 39,8 → **70,5 mAP50** |
| Caltech havadan termal segmentasyon, **4,8 M** encoder: sıfırdan / **ImageNet RGB** / 40 k termalde öz-denetim | 0,687 / **0,725** / 0,714 |
| SODA-IR + MFNet-IR, ViT-B: termal-only MIM vs RGB IN1K ön-eğitimi | 61,5 / 43,0 → **69,6 / 50,3** |

Üçü de aynı yöne bakıyor: **termal görevde bile RGB ön-eğitimi, termal
öz-denetimden iyi.** Sebep basit — termal görüntülerin taşıdığı bilgi kare
başına daha az (InfMAE'nin kendi ölçümü: Inf30 entropi 6,44 vs ImageNet 7,19),
ve dünyada RGB verisi kat kat fazla.

Ama **nasıl** beslendiği önemli, ve burada iki yanlış yol var:

1. **Öğrenciye tam renk verme.** Dağıtılan encoder hayatı boyunca renk
   görmeyecek. `rgb->gray` yolu öğrenciye luminansı veriyor, öğretmene rengi —
   görevin şeklini koruyor.
2. **Öğretmenin *tahminini* distile etme.** Termal→RGB-öğretmen yanıt seviyesi
   distilasyonu ölçülmüş: **hiç distile etmemekten 2,9 mAP50 kötü**. Öznitelik
   kopyala, tahmin değil — bu aşama zaten öyle yapıyor.

Ve bir de üçüncü, daha ince tehlike: **dar külliyat.** AnyThermal, *yalnızca*
havadan veriyle distile edilen varyantın kentsel sahnelerde distile edilmemiş
RGB modelden **daha kötü** çıktığını bildiriyor. Yani RGB'yi karıştırmanın
faydası çeşitlilikten geliyor; RGB'yi karıştırıp külliyatı yine tek sahneye
sıkıştırmak faydayı yok eder.

### 8.5 Sıra: RGB modeli önce eğit, sonra termalin öğretmeni yap

İki model eğitiliyorsa doğal sıra şu:

1. **RGB encoder'ı önce ön-eğit** (`STUDENT="rgb"`, defter 16). Aynı modalite,
   en çok veri, en iyi doğrulanmış tarif. Aşama B'yi de RGB tarafında koş.
2. **Termal encoder'ı ön-eğitirken öğretmen olarak DINOv3 yerine kendi RGB
   encoder'ını dene.** EdgeCrafter'ın ölçtüğü şey tam bu: öğretmeni göreve
   adapte etmek **+1,0 AP**, ve COCO'ya adapte etmek +0,8 AP. Havadan
   görüntüye adapte edilmiş bir RGB encoder, havadan termal için genel bir
   DINOv3'ten daha yakın bir öğretmen — ve kapasite oranı da doğru (6 M → 6 M,
   yani 1×; bu bandın *altında*, o yüzden DINOv3-ViT-B ile yan yana koşulmalı,
   yerine değil).

Bu ikinci adım bugün bir satır değil — `distill.build_teacher` transformers
checkpoint'i bekliyor, EdgeTAM checkpoint'ini değil. Ama ölçülmeye değer ve
kodda küçük bir ekleme.
