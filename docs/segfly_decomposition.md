# SegFly: semantik haritadan örnek maskeye — ölçüldü

`docs/encoder_training_todo.md` §6'nın **birinci açık sorusu** şuydu:

> **Bağlantılı bileşen ayrıştırması temiz örnek veriyor mu?** §4.1'deki ilk
> kontrol. Bütün B aşaması buna bağlı; en ucuz ve en riskli adım bu.
> *Ölçüm hazır* — notebook 07 hücre 11–12, GPU'suz. **Cevap hâlâ yok.**

Bu rapor o cevabı SegFly için veriyor. Kısa hâli: **kaynaşma korkusu gerçek
çıkmadı, ama havuz yine de bozuk — sebebi ayrıştırma değil, `truck` sınıfı.**
Ve iki tane sessiz hata var: RGB havuzu hiç eğitime girmemiş, termal havuz
ise not verdiği yerde durmaması gereken bir yerde duruyor.

Tarih: 2026-08-31. Ölçen: `tools/segfly_decompose_audit.py` (bu commit).

---

## 1. Ne ölçüldü, neyin üstünde

Üç ayrı kaynak, üçü de birbirini kontrol ediyor:

| kaynak | ne verdi | nereden |
|---|---|---|
| **Hasat edilmiş havuzlar** | 5 378 termal + 1 969 RGB karenin her örneğinin kutusu, alanı, sınıfı | Drive `edgetam-pool/segfly_thermal`, `edgetam-pool/segfly_rgb` — 10 shard `.jsonl` + iki `manifest.json` |
| **Aşama B koşularının kararları** | havuzun eğitimde ne kadar yer tuttuğu ve sınıf başına test IoU'su | Drive `edgetam-stage-b/thermal/verdict.json`, `edgetam-stage-b/thermal_rgb/verdict.json` |
| **Yayıncının kendi etiket haritaları** | `aerial.decompose()`'un iki modda gerçek çıktısı ve **reddedilen bileşenler** | HF `markus-42/SegFly`, parquet'ten yalnız `label` kolonu — 140 termal (640×512; scene_03/30 m, scene_09/40 m, scene_09/50 m) + 141 RGB (5472×3648; scene_01/40 m, scene_06/40 m, scene_07/30 m) |

Üçüncüsü önemli, çünkü havuz kayıtları **kabul edilen** örnekleri taşıyor;
reddedilenler kayıtta yok. Kaynaşma sinyali — `fill` kapısının kaç bileşeni
attığı — ancak `decompose`'u yeniden koşarak ölçülebiliyordu.

**Havuz manifest'inin söyledikleri** (`segfly_thermal/manifest.json`):

```
method  decompose:watershed
gates   min_area 48 · min_side 4 · max_area 0.9 · fill 0.25
things  vehicle (13), truck (36)
frames  5 378        instances  20 145   (vehicle 15 044 · truck 5 101)
```

RGB tarafı: 1 969 kare, **69 521** örnek (truck 45 643 · vehicle 23 878).

---

## 2. Kaynaşma: korkulan şey olmadı

`aerial.py`'nin modül docstring'i ayrıştırmanın gerekçesini kurarken şunu
diyor: *"does connected-component decomposition yield clean instances, or do
two parked cars fuse into one?"* — ve `reject_reason` bunun ölçüsünü
`fill` kapısına bağlıyor: kendi kutusunun dörtte birini dolduran bileşen
genelde ince bir piksel köprüsüyle birleşmiş iki nesnedir.

Yayıncının 140 termal haritası üzerinde, repo'nun kendi `decompose()`'u,
repo'nun kendi kapılarıyla:

| | `components` | `watershed` |
|---|---:|---:|
| kabul edilen örnek | 344 | 361 |
| reddedilen bileşen | 99 | 119 |
| reddin sebebi `min_area` | **99** | **119** |
| reddin sebebi **`fill`** | **0** | **0** |
| örnek üretmeyen kare | 41 / 140 (%29,3) | 41 / 140 |
| kabul edilenlerin `fill` medyanı | 0,723 | 0,719 |

**`fill` üzerinden reddedilen tek bir bileşen yok.** Havuzun kabul ettiği
20 145 örneğin dağılımı da aynı şeyi söylüyor: `fill` medyanı 0,719, ve
yalnız %2,7'si 0,35'in altında — yani kapının kıyısında toplanma yok.

Geometri de destekliyor (havuzdaki 20 145 örnek, kare içi kutu ilişkileri):

| | termal | RGB |
|---|---:|---:|
| kutusu bir başkasıyla IoU ≥ 0,30 örtüşen | %3,4 | %2,5 |
| kutusu bir başkasının içinde (≥ %90) | %3,9 | %5,7 |
| 2 px içinde komşusu olan | %22,6 | %37,4 |
| en yakın komşuya uzaklık (medyan) | 14 px | 14 px |

Nesneler birbirine değiyor (termalde dörtte biri, RGB'de üçte biri), ama
**değmek kaynaşmak değil** — ayrıştırma onları ayrı bileşen olarak
tutabiliyor.

RGB tarafında `fill` reddi sıfır değil ama küçük: 141 haritada 10 299
bileşenin **135'i (%1,31)** `fill`'den düştü. Bu hâlâ `docs/encoder_neden_boyle.md`
ölçütüyle "birkaç yüzde — normal" bandında.

**Sonuç: §6'nın 1. sorusunun SegFly için cevabı "evet, temiz."** Aşama B'nin
üzerine kurulduğu varsayım bu sette tutuyor. Riskli adım sanılan yer riskli
değilmiş; risk başka yerde çıktı.

---

## 3. `watershed` mi `components` mı — ve repo üçe bölünmüş

Ölçülen fark küçük: watershed termalde **+17 örnek (%4,9)** getiriyor ve
karşılığında 20 bileşeni daha `min_area`'nın altına düşürüyor. `fill` reddi
iki modda da sıfır olduğu için watershed'in çözmek üzere var olduğu problem
bu sette zaten yok.

RGB tarafında watershed **pahalı ve ters yönde çalışıyor**. 141 haritalık koşu
10 dakikada bitmedi (`components` aynı iş için saniyeler alıyor); 40 haritalık
alt kümede karşılaştırma şöyle:

| 40 RGB haritası (5472×3648) | `components` | `watershed` |
|---|---:|---:|
| kabul edilen örnek | 3 777 | 4 335 (+%14,8) |
| `min_area` reddi | 356 | 495 |
| **`fill` reddi** | **37** | **213** |
| `fill` reddinin tüm bileşenlere payı | %0,9 | **%4,2** |

Yani watershed burada kaynaşma çözmüyor, **kaynaşmış görünen parça
üretiyor**: bir bileşeni böldüğünde ortaya çıkan iki parçadan biri kendi
kutusunu doldurmuyor ve `fill` kapısından düşüyor. 20 MP'lik haritada hem
yavaş hem de kazandırdığından fazlasını attırıyor.

Buna karşılık repo üç ayrı şey söylüyor:

| nerede | ne diyor |
|---|---|
| havuz `manifest.json` (Drive) | `decompose:watershed` — havuz **watershed** ile hasat edilmiş |
| `tools/build_notebooks.py:182`, `:340`, `:347` | `segfly:…:thermal:components:train` — defter 07–11 **components** okuyor |
| `tools/build_notebooks.py:720` (defter metni, hücre 5) | tabloda `segfly … watershed`, `kust4k … components` |

Yani hasadın kullandığı mod, eğitimin kullandığı moddan farklı, ve defterin
kendi metni üçüncü bir şey anlatıyor. Sayı küçük olduğu için sonucu
değiştirmiyor, ama üç yerde üç cevap olması kendi başına düzeltilecek bir şey.

---

## 4. Asıl bulgu: `truck` bir kamyon gibi davranmıyor

Havuzdaki iki "şey" sınıfının boyutları:

| | termal `vehicle` | termal `truck` | RGB `vehicle` | RGB `truck` |
|---|---:|---:|---:|---:|
| örnek | 15 044 | 5 101 | 23 878 | 45 643 |
| alan medyanı (px) | **2 108** | **226** | 1 429 | 588 |
| uzun kenar medyanı | 70 | 24 | 70 | 41 |
| 32 px altı | %13,6 | %59,6 | %30,5 | %37,5 |
| 200 px'den küçük | %8,1 | **%46,3** | %20,8 | %23,6 |

Bir kamyon bir otomobilden **büyüktür**. Burada `truck`, `vehicle`'ın alan
olarak **dokuzda biri** ve neredeyse yarısı 200 pikselin altında. 30–50 m
irtifada, 640×512 termalde bir araba ~2 100 px iken 226 px'lik bir nesne
kamyon olamaz.

Üç kontrol daha yapıldı, üçü de aynı yöne gösteriyor:

**(a) Palet doğru.** Yayıncının HF veri kartındaki tablo `36 | Truck |
[128,128,64]` diyor; `SPECS["segfly"]` bununla birebir aynı. Bu, Kust4K ve
SegFly'da iki kez yaşanan palet hatasının üçüncüsü **değil**.

**(b) Parçalanmış araç değiller.** Her `truck` bileşeninin 2 px'lik halkası
neyden oluşuyor (282 bileşen; örneklenen üç shard'ın id 36 taşıyanı yalnız
scene_09'un ikisi — 40 m'de 139, 50 m'de 143 bileşen):

| halkanın çoğunluğu | pay |
|---|---:|
| vegetation | %36,5 |
| walkway | %26,6 |
| dirt | %7,4 |
| building | %7,1 |
| tree | %6,4 |

ve **bileşenlerin yalnız %0,4'ünün halkasında tek bir `vehicle` pikseli var.**
Yani bunlar bir aracın kopmuş parçaları değil; çimenin ve kaldırımın ortasında
duran bağımsız küçük lekeler. Şekilleri de ince/uzun değil (en-boy oranı
medyan 1,43), yani direk/kablo gibi bir sınıfın yanlış id'si de değiller.

RGB yarısında aynı ölçüm daha da keskin (scene_01/06/07, 40 harita, 3 791
`truck` bileşeni): halkanın çoğunluğu **tree %52,7 · vegetation %42,6** —
ikisi birlikte **%95,3**. Bir `vehicle` pikseline değen bileşen oranı %1,2.
Yani RGB tarafında `truck` etiketi ezici çoğunlukla **ağaç örtüsünün içinde**
duruyor.

**(c) Ham haritada da öyle.** Kapılardan önce, yayıncının bayt'ları üzerinde:

| ham bileşen (kapısız) | termal (140 harita) | RGB (141 harita) |
|---|---|---|
| id 13 `vehicle` | 161 bileşen, medyan **742 px** | 1 547 bileşen, medyan **1 680 px** |
| id 36 `truck` | 282 bileşen, medyan **126 px** | 8 752 bileşen, medyan **577 px** |

İki modalitede de aynı örüntü: kamyon, arabadan **küçük** ve **kalabalık** —
termalde 1,8 kat, RGB'de 5,7 kat daha çok bileşen. İkisinin birlikte
göründüğü scene_09'da 113 `vehicle` bileşenine karşı 282 `truck` var.

En makul okuma, setin kendi makalesinin söylediği şeyle uyuşuyor: etiketlerin
yalnız **%2,84'ü elle çizilmiş**, gerisi 2D–3D–2D geometrik yayılımla
üretilmiş. `truck` id'si yayılımın gürültüsünü toplamış görünüyor. Bu iddia
**etiket geometrisine** dayanıyor, göze değil — kesinleştirmek için 20–30
karelik bir görsel panel gerekiyor (bkz. §8).

Bunun havuzdaki ağırlığı küçük değil: **5 378 termal karenin 1 194'ü (%22,2)
yalnızca `truck` taşıyor** — o karelerde eğitilen tek hedef bu sınıf. RGB
tarafında ise `truck`, hedeflerin **%65,7'si**, ve sahne bazında scene_01'de
%82,7, scene_02'de %84,3, scene_06'da %82,6.

---

## 5. Bunun eğitimdeki bedeli — tahmin değil, ölçüm

`edgetam-stage-b/thermal/verdict.json`, SegFly havuzunu eğitimin ikinci en
büyük kaynağı yapmış: **7 268 train penceresi** (kaggle_uav_thermal'ın 22 176'sından
sonra en büyüğü), ve **906 test penceresi**.

Test skorları, sınıf başına, `edgetam_pool_thermal_512.pt` sonrası:

| kaynak / sınıf | box IoU (önce → sonra) | point IoU (önce → sonra) |
|---|---|---|
| `pool/kust4k_broken_thermal/truck` | 0,927 → 0,941 | 0,667 → 0,887 |
| `pool/dronevehicle_thermal_only/car` | 0,855 → 0,918 | 0,582 → 0,876 |
| `kust4k/car` | 0,825 → 0,868 | 0,666 → 0,798 |
| … | | |
| **`pool/segfly_thermal/vehicle`** | 0,641 → **0,776** | 0,386 → **0,486** |
| **`pool/segfly_thermal/truck`** | 0,561 → **0,692** | 0,202 → **0,229** |

İki satır da tablonun **en altı**. Ve `point` sütunu asıl söyleyen: model bu
havuzda eğitildikten *sonra* bile, `segfly/truck` hedeflerini bir nokta
prompt'undan 0,229 IoU ile yeniden üretebiliyor. Bir sonraki en kötü sayı
`kust4k_thermal/human` 0,389 — yani SegFly'ın kamyonu, listenin en zor sınıfı
olan insandan da **çok** aşağıda.

Bir hedefi, o hedefte eğitilmiş bir model bile bulamıyorsa, sorun modelde
değil: hedef görüntüde görünen bir şeye karşılık gelmiyor demektir. §4'ün
bulgusunun bağımsız doğrulaması bu.

**Ve bu hedefler not da veriyor.** `POOL_ROLE = "all"` olduğu için
`segfly_thermal` test bölmesine de giriyor: skorlanan 11 443 örneğin
**3 060'ı (%26,7)** SegFly'dan (755 truck + 2 305 vehicle). Bu, projenin kendi
kuralına aykırı — `docs/encoder_neden_boyle.md` ve defter metni açıkça
"yeniden kurulmuş örnekler eğitir, **asla skorlanmaz**; VTUAV/Kust4K not
verir" diyor. Bugünkü mean IoU rakamı, dörtte biri gürültülü bir referansla
ölçülmüş durumda.

---

## 6. RGB havuzu hiç eğitime girmedi — yol hatası

`edgetam-stage-b/thermal_rgb/verdict.json`, `unusable` listesinde:

```
segfly_rgb: 1969 records and not one usable frame ({'no_image': 1969})
```

Sebebi tek satır. Havuz kayıtları görüntüyü şurada arıyor:

```
/content/data/SegFly_rgb/rgb/scene_01_30m_020176.jpg
```

Defterin `IMAGES` satırı ise havuzun kökünü `SegFly` yapıyor
(`tools/build_stage_b_notebooks.py:143` → `["segfly_rgb", "", "SegFly", []]`),
yani `/content/data/SegFly`. `Relocator` son ekleri sırayla deniyor ve
`rgb/scene_01_30m_020176.jpg`'yi orada arıyor — o klasör **var**, ama içinde
termal çiftlerin 640×512 kayıtlı RGB'si duruyor, scene_01'in 5472×3648 karesi
değil. Hiçbir son ek tutmuyor, 1 969 kaydın tamamı düşüyor.

İki ek ayrıntı aynı satırda:

- İkinci alan (fetch adı) **boş**: defter RGB dilimini zaten indirmiyor.
- `tools/build_notebooks.py:359` aynı veri için `SegFly_RGB` kökünü
  kullanıyor. İki defter iki farklı yol adı taşıyor.

Bedeli: 1 969 kare, **69 521 örnek**, ve `report/nb11_beklenti.tex:251`'e göre
~23 GB'lık bir indirme — hasat edildi, Drive'a yazıldı, ve tek bir eğitim
penceresi üretmedi.

---

## 7. "Hacim kaynağı" iddiası iki düzeltme istiyor

`docs/encoder_mimari.md:207` SegFly için *"hacmi getiriyor (Kust4K'nın ~4 katı
kare)"* diyor. Ölçülen iki şey bunu daraltıyor:

1. **15 007 termal karenin yalnız 5 378'i (%35,8) hiç örnek veriyor.** Geri
   kalanında `vehicle` ya da `truck` pikseli yok. Kust4K'nın 4 024 karesine
   karşılık gerçek sayı 5 378 — dört katı değil, **1,3 katı**.
2. **Havuzdaki karelerin %85,2'sinin bir önceki kare id'si de havuzda.**
   SegFly bağımsız fotoğraf değil, ardışık uçuş kareleri; 20 145 örnek
   20 145 ayrı nesne değil, aynı araçların tekrar tekrar sayılması. (RGB
   diliminde bu oran %92,5 — `spread` sahneleri gezmiş ama shard içinde
   ardışık kareler almış.)

Sahne kapsaması ise dokümanla uyuşuyor ve doğru: termal yalnız scene_03, 04,
05, 09'da var; RGB dilimi dokuz sahnenin hepsini ve üç irtifayı da görüyor.

---

## 8. Ne yapmalı — sırayla

1. **`things` içinden `truck`'ı düşür** (`SPECS["segfly"].things = ("vehicle",)`).
   Bedeli 5 101 termal + 45 643 RGB hedef; kazancı, eğitimin ve notun en
   gürültülü kaynağının gitmesi. Ara çözüm: `truck` için ayrı, yüksek bir
   `min_area` — ama §4(b) bunların büyük kardeşlerinin parçası olmadığını
   gösterdiği için eleme muhtemelen sınıfın tamamını götürür.
2. **`segfly_thermal`'ı `role="train"` yap.** Yeniden kurulmuş örnekler not
   vermemeli; bugün test penceresinin üçte birini o taşıyor.
3. **RGB yolunu düzelt ya da havuzu sil.** `build_stage_b_notebooks.py`'nin
   `IMAGES` satırında kök `SegFly_RGB` olmalı ve fetch adı doldurulmalı;
   yoksa 69 521 örnek Drive'da ölü duruyor.
4. **Modu tek yerde karara bağla.** Termalde watershed'in getirisi %4,9;
   RGB'de hem yavaş hem `fill` reddini %0,9'dan %4,2'ye çıkarıyor. Öneri:
   her yerde `components`, ve defter metnini buna göre düzelt.
5. **Hasat reddettiklerini kaydetsin.** `verdict.json`'daki
   `"rate": 1.0, "rejected": {}` bir ölçüm değil, ölçümün yokluğu — havuz
   yazıcısı `decompose`'un döndürdüğü `rejects` sözlüğünü atıyor. Bu rapor
   için o sayıyı elde etmenin tek yolu 178 GiB'lik parquet'e geri dönmekti.

## Ölçülmeyenler

- **Piksele bakılmadı.** Yalnız etiket haritaları indirildi; görüntüler değil.
  §4'ün "`truck` kamyon değil" okuması geometriye dayanıyor. Kesin cevap için
  defter 07'nin 15. hücresindeki panelin SegFly için koşulması gerekiyor.
- **`truck`'ın halka analizi tek sahneden.** Örneklenen üç termal shard'ın
  yalnız ikisinde (scene_09/40 m ve /50 m) id 36 var; havuz sayıları
  `truck`'ın scene_03'te de bulunduğunu söylüyor (%30,5 pay), orası
  örneklenmedi. Sonucu genişletmek scene_03'ten iki shard daha ister.
- **Kust4K ve Caltech'in aynı ölçümü.** §6'nın 3. sorusu (Kust4K'nın "şey"
  sınıfları kaç örnek veriyor) hâlâ açık; bu rapor yalnız SegFly'ı kapatıyor.

## Tekrarlanması

```bash
# 1. Havuz kayıtları (Drive): edgetam-pool/segfly_{thermal,rgb}/*.jsonl
# 2. Yayıncının etiket haritaları + decompose, her iki mod:
python tools/segfly_decompose_audit.py --shards 63 410 692 --out /tmp/segfly_audit
# 3. Aşama B kararları (Drive): edgetam-stage-b/{thermal,thermal_rgb}/verdict.json
```

`--shards` numaraları termal shard'lar; RGB için `475 550 50` (5472×3648,
`components` ile koşun). Shard'lar `label` kolonu için range isteğiyle
okunuyor, tam indirme gerekmiyor.
