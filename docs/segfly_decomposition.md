# SegFly decomposition: reddedilenlerin tablosu

Bu rapor, `report/bolumler/veri_setleri.tex` §`sec:segfly-instance`'in ve
`notebooks/26_segfly_instance_audit.ipynb`'in **açıkta bıraktığı sayıyı**
dolduruyor. O bölüm dönüşümün boru hattını kuruyor ve tek bir kareyi
(`train/0212.parquet`, 17. termal satır) panelle gösterip şunu diyor:

> Full-data mode kararı, reject table ve manually reviewed panel sample
> birlikte okunarak verilmelidir.

Panel vardı, **reject table yoktu**. Bu rapor onu 140 termal + 141 RGB
etiket haritası üzerinde çıkarıyor, ve aynı ölçüm
`docs/encoder_training_todo.md` §6'nın 1. açık sorusunu da kapatıyor:

> **Bağlantılı bileşen ayrıştırması temiz örnek veriyor mu?** … Cevap hâlâ yok.

Kısa hâli: **kaynaşma korkusu, ölçülebildiği kadarıyla gerçek çıkmadı** —
köprülü hâli hiç yok, yan yana hâli hedeflerin en fazla %1,6'sı (§2b; şekille
daha fazlası dışlanamıyor). Ama aynı ölçüm başka bir şey buldu: `truck`
sınıfı bir kamyon gibi davranmıyor, ve SegFly havuzunun aşama B'deki bütün
sorunları oradan geliyor.

Tarih: 2026-08-31. Koşan kod: `tools/segfly_decompose_audit.py`.

---

## 1. Ne ölçüldü, neyin üstünde

| kaynak | ne verdi | nereden |
|---|---|---|
| **Hasat edilmiş havuzlar** | 5 378 termal + 1 969 RGB karenin her örneğinin kutusu, alanı, sınıfı | Drive `edgetam-pool/segfly_thermal`, `edgetam-pool/segfly_rgb` — 10 shard `.jsonl` + iki `manifest.json` |
| **Aşama B koşularının kararları** | havuzun eğitimdeki ağırlığı ve sınıf başına test IoU'su | Drive `edgetam-stage-b/thermal/verdict.json`, `…/thermal_rgb/verdict.json` (defter 19 ve 20, 2026-08-26) |
| **Yayıncının kendi etiket haritaları** | `aerial.decompose()`'un iki modda çıktısı ve **reddedilen bileşenler** | HF `markus-42/SegFly`, parquet'ten yalnız `label` kolonu — 140 termal (640×512; scene_03/30 m, scene_09/40 m ve /50 m) + 141 RGB (5472×3648; scene_01/40 m, scene_06/40 m, scene_07/30 m) |

Üçüncüsü zorunluydu: havuz kayıtları **kabul edilen** örnekleri taşıyor,
reddedilenler kayıtta yok. Kaynaşma sinyali — `fill` kapısının kaç bileşeni
attığı — ancak `decompose`'u yeniden koşarak elde edilebiliyordu. Parquet
kolonlu olduğu için bu pahalı değil: `label`, iki megabaytlık görüntü
kolonunun yanında küçük bir kolon, range isteğiyle okunuyor.

Havuz manifest'i (`segfly_thermal/manifest.json`):

```
method  decompose:watershed
gates   min_area 48 · min_side 4 · max_area 0.9 · fill 0.25
things  vehicle (13), truck (36)
frames  5 378        instances  20 145   (vehicle 15 044 · truck 5 101)
```

RGB tarafı: 1 969 kare, **69 521** örnek (truck 45 643 · vehicle 23 878).

---

## 2. Kaynaşma: korkulan şey olmadı

`aerial.py`'nin docstring'i ayrıştırmanın gerekçesini kurarken soruyu böyle
koyuyor: *"does connected-component decomposition yield clean instances, or do
two parked cars fuse into one?"* — ve ölçüyü `fill` kapısına bağlıyor: kendi
kutusunun dörtte birini dolduran bileşen genelde ince bir piksel köprüsüyle
birleşmiş iki nesnedir.

140 termal harita, repo'nun kendi `decompose()`'u, repo'nun kendi kapıları:

| | `components` | `watershed` |
|---|---:|---:|
| kabul edilen örnek | 344 | 361 |
| reddedilen bileşen | 99 | 119 |
| reddin sebebi `min_area` | **99** | **119** |
| reddin sebebi **`fill`** | **0** | **0** |
| örnek üretmeyen kare | 41 / 140 (%29,3) | 41 / 140 |
| kabul edilenlerin `fill` medyanı | 0,723 | 0,719 |

**`fill` üzerinden reddedilen tek bir termal bileşen yok** — reddin tamamı
`min_area`, yani "hedef olamayacak kadar küçük". Havuzun kabul ettiği 20 145
örneğin dağılımı da aynı yöne gidiyor: `fill` medyanı 0,719, yalnız %2,7'si
0,35 altında, yani kapının kıyısında birikme yok.

Geometri de destekliyor (havuzun 20 145 + 69 521 örneği, kare içi kutu
ilişkileri):

| | termal | RGB |
|---|---:|---:|
| kutusu bir başkasıyla IoU ≥ 0,30 örtüşen | %3,4 | %2,5 |
| kutusu bir başkasının içinde (≥ %90) | %3,9 | %5,7 |
| 2 px içinde komşusu olan | %22,6 | %37,4 |
| en yakın komşuya uzaklık (medyan) | 14 px | 14 px |

Nesneler birbirine değiyor — termalde dörtte biri, RGB'de üçte biri — ama
**değmek kaynaşmak değil**; ayrıştırma onları ayrı bileşen olarak tutuyor.
RGB'de `fill` reddi sıfır değil ama küçük: 141 haritada 10 299 bileşenin
**135'i (%1,31)**.

§`sec:segfly-instance`'in "Connected component = object" satırına verdiği
**Koşullu** notu, bu ölçüm koşul lehine çeviriyor — ama tamamen kaldırmıyor.
Sebebi bir sonraki bölüm.

---

## 2b. `fill`'in göremediği kaynaşma, ve nereye kadar dışlanabildiği

Yukarıdaki sıfır tek başına "kaynaşma yok" demek **değil**. `fill` yalnız
**köprülü** kaynaşmayı yakalar: iki nesne ince bir bağla birleşince kutu
büyür ve boşalır. Yan yana **tam kenardan** yapışmış iki araç ise düzgün ve
dolu bir dikdörtgen verir — `fill`'i yüksek çıkar, kapıdan geçer. Ve o
şekil, gerçek bir minibüsün şekliyle **birebir aynıdır**; maske geometrisi
ikisini ayıramaz. Defter 07'nin 16. hücresi ve §`sec:segfly-instance` bunu
zaten söylüyordu ("full-edge contact'ı ayıramaz"); buradaki iş, geriye ne
kadar belirsizlik kaldığını sayıya bağlamak.

İki ek test yapıldı.

**Aşındırma.** Kabul edilen her `vehicle` bileşeni kenarından 1 ve 2 piksel
yontuldu; aralarında ince bir bağ olan her çift bu işlemde dağılır.

| | |
|---|---|
| 1 px aşındırınca ≥2 anlamlı parçaya ayrılan | **0 / 127** |
| 2 px aşındırınca ayrılan | **0 / 127** |

Yani `fill`'in sıfırı bir kapı artefaktı değil; zayıf bağlı bileşen de yok.

**Şekil profili.** Yan yana kaynaşmanın imzası bellidir: *bir* kenar ikiye
katlanır, diğeri normal kalır. Havuzun 15 044 `vehicle` örneğinde:

| | | |
|---|---:|---|
| alanı medyanın 1,6 katından büyük | 3 900 · %25,9 | ilk bakışta ürkütücü |
| …bunlardan **iki kenarı da** büyük | 1 615 · %10,7 | kaynaşma bir kenarı büyütür, ikisini değil → gerçek büyük araç |
| …bunlardan **bir kenarı 2×, diğeri normal** | **235 · %1,6** | kaynaşma profiline uyan tek grup |

Üçüncü satırın `fill` medyanı **0,839**, normal grubunki 0,762 — yani
şüpheli grup daha *dolu*. Aralarında boşluk kalmış iki aracın `fill`'i
düşük olurdu; yüksek olması kaynaşma aleyhine delil.

**Söylenebilecek olan:** köprülü kaynaşma yok (0/127), yan yana kaynaşma
olsa olsa hedeflerin **%1,6'sı** kadar, ve o grubun `fill` profili de bunun
aleyhine. Kalan belirsizlik şekille kapatılamaz.

**O %1,6 düşürülmek istenirse**, ucuzdan pahalıya:

| yöntem | ne yapar | bedeli |
|---|---|---|
| alan üstü kesme (medyanın 2,5 katı) | kaba; minibüsleri de atar | %9,8 hedef |
| **şekil profili kesmesi** (bir kenar 2×, diğeri normal) | hedefli; büyük araca dokunmaz | **%1,6 hedef** |
| öğretmenle çapraz kontrol (SAM'a lekenin içinden nokta) | tek gerçek çözüm — görüntüye bakar, tam kenardan yapışığı da ayırır | bir GPU geçişi |
| gerçek örnek maskesi olan set (VTUAV VIS) | problemi ortadan kaldırır | — |

Üçüncüsü repoda zaten var: `tools/analyze_segfly_instances.py --sam-teacher`,
kapı olarak değil inceleme kuyruğu olarak — çünkü SAM de bir sözde-etiket,
düşük uyuşma "hangisi yanlış" demiyor. Dördüncüsü ise projenin verdiği
karar: SegFly eğitir, not vermez.

---

## 3. `components` mi `watershed` mı — ve defter metni hâlâ üçüncü şeyi diyor

Termalde watershed **+17 örnek (%4,9)** getiriyor, karşılığında 20 bileşeni
daha `min_area`'nın altına düşürüyor. `fill` reddi iki modda da sıfır olduğu
için watershed'in çözmek üzere var olduğu problem bu sette zaten yok.

RGB'de ise **ters yönde çalışıyor**. 141 haritalık koşu 10 dakikada bitmedi
(`components` aynı işi saniyelerde yapıyor); 40 haritalık alt kümede:

| 40 RGB haritası (5472×3648) | `components` | `watershed` |
|---|---:|---:|
| kabul edilen örnek | 3 777 | 4 335 (+%14,8) |
| `min_area` reddi | 356 | 495 |
| **`fill` reddi** | **37** | **213** |
| `fill` reddinin tüm bileşenlere payı | %0,9 | **%4,2** |

Watershed burada kaynaşma çözmüyor, **kaynaşmış görünen parça üretiyor**: bir
bileşeni böldüğünde çıkan parçalardan biri kendi kutusunu doldurmuyor ve
`fill`'den düşüyor. §`sec:segfly-instance` bunu zaten öngörüyordu ("long
vehicle'ı over-split edebilir"); buradaki tablo onun sayısı.

**Kalan tutarsızlık.** Eğitim yolu artık her yerde `components`
(`tools/build_notebooks.py:182`, `:340`, `:347`;
`tools/build_stage_b_notebooks.py:298`). Ama:

- Drive'daki havuz `decompose:watershed` ile hasat edilmiş — yani havuzun
  hedefleri ile `--dataset segfly:…:components` yolunun hedefleri **aynı
  değil**;
- defter metni (`tools/build_notebooks.py:720` → defter 07–11, hücre 5)
  tablosunda hâlâ `segfly … watershed` yazıyor.

Ölçülen fark küçük olduğu için sonucu çevirmiyor, ama metin düzeltilmeli.

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
olarak dokuzda biri ve neredeyse yarısı 200 pikselin altında. 30–50 m
irtifada 640×512 termalde bir araba ~2 100 px iken 226 px'lik bir nesne
kamyon olamaz.

Üç kontrol, üçü de aynı yöne:

**(a) Palet doğru.** Yayıncının HF veri kartı `36 | Truck | [128,128,64]`
diyor; `SPECS["segfly"]` ve `analyze_segfly_instances.SEGFLY_COLORS`
bununla birebir aynı. Bu, Kust4K ve SegFly'da iki kez yaşanan palet
hatasının üçüncüsü **değil**.

**(b) Parçalanmış araç değiller.** Her `truck` bileşeninin 2 px'lik halkası
neyden oluşuyor (282 termal bileşen; örneklenen üç shard'ın id 36 taşıyanı
yalnız scene_09'un ikisi — 40 m'de 139, 50 m'de 143):

| halkanın çoğunluğu | termal | RGB (3 791 bileşen) |
|---|---:|---:|
| vegetation | %36,5 | %42,6 |
| walkway | %26,6 | %0,3 |
| **tree** | %6,4 | **%52,7** |
| dirt | %7,4 | %2,6 |
| building | %7,1 | — |

ve **halkasında tek bir `vehicle` pikseli olan bileşen oranı termalde %0,4,
RGB'de %1,2.** Yani bunlar bir aracın kopmuş parçaları değil; çimenin,
kaldırımın ve RGB'de ağaç örtüsünün içinde duran bağımsız küçük lekeler.
Şekilleri ince/uzun da değil (en-boy oranı medyan 1,43), yani direk/kablo
gibi bir sınıfın yanlış id'si de değiller.

**(c) Ham haritada da öyle.** Kapılardan önce, yayıncının bayt'ları üzerinde:

| ham bileşen (kapısız) | termal (140 harita) | RGB (141 harita) |
|---|---|---|
| id 13 `vehicle` | 161 bileşen, medyan **742 px** | 1 547 bileşen, medyan **1 680 px** |
| id 36 `truck` | 282 bileşen, medyan **126 px** | 8 752 bileşen, medyan **577 px** |

İki modalitede de aynı örüntü: kamyon, arabadan **küçük** ve **kalabalık** —
termalde 1,8 kat, RGB'de 5,7 kat daha çok bileşen. İkisinin birlikte
göründüğü scene_09'da 113 `vehicle` bileşenine karşı 282 `truck`.

En makul okuma, §`sec:segfly-instance`'in "SegFly semantic pseudo-GT: Orta"
satırıyla uyuşuyor: etiketlerin yalnız %2,84'ü elle çizilmiş, gerisi
2D–3D–2D yayılımla üretilmiş ve yayın doğrulamasını **60 manuel termal
görüntüyle** yapmış. `truck` id'si yayılımın gürültüsünü toplamış görünüyor.
Bu iddia **etiket geometrisine** dayanıyor, göze değil — kesinleştirmek için
defter 26'nın panelini `truck` bileşenlerine odaklayarak koşmak gerekiyor.

Havuzdaki ağırlığı küçük değil: **5 378 termal karenin 1 194'ü (%22,2)
yalnızca `truck` taşıyor.** RGB'de `truck` hedeflerin %65,7'si; sahne bazında
scene_01 %82,7, scene_02 %84,3, scene_06 %82,6.

---

## 5. Bunun eğitimdeki bedeli — ve neden `SKIP_POOLS`'a girdiği

`edgetam-stage-b/thermal/verdict.json` (defter 19, 2026-08-26) SegFly
havuzunu eğitimin ikinci en büyük kaynağı yapmış: **7 268 train penceresi**
(yalnız kaggle_uav_thermal'ın 22 176'sı önünde), ve **906 test penceresi**.

`edgetam_pool_thermal_512.pt` sonrası sınıf başına test IoU'su:

| kaynak / sınıf | box (önce → sonra) | point (önce → sonra) |
|---|---|---|
| `pool/kust4k_broken_thermal/truck` | 0,927 → 0,941 | 0,667 → 0,887 |
| `pool/dronevehicle_thermal_only/car` | 0,855 → 0,918 | 0,582 → 0,876 |
| `kust4k/car` | 0,825 → 0,868 | 0,666 → 0,798 |
| … | | |
| **`pool/segfly_thermal/vehicle`** | 0,641 → **0,776** | 0,386 → **0,486** |
| **`pool/segfly_thermal/truck`** | 0,561 → **0,692** | 0,202 → **0,229** |

İki satır da tablonun en altı, ve `point` sütunu asıl söyleyen: model bu
havuzda eğitildikten *sonra* bile `segfly/truck` hedeflerini bir nokta
prompt'undan 0,229 IoU ile üretebiliyor. Sıradaki en kötü sayı
`kust4k_thermal/human` 0,389 — yani SegFly'ın kamyonu, listenin en zor
sınıfından da çok aşağıda. Bir hedefi, o hedefte eğitilmiş model bile
bulamıyorsa sorun modelde değil.

Ağaçtaki durum, bunun zaten görülmüş olduğunu söylüyor: **defter 22 ve 32
`SKIP_POOLS`'a `segfly_thermal`'ı koyuyor** ve SegFly'ı havuz olarak değil
`--dataset segfly:…:components:**train**` olarak okuyor — yani not vermiyor.
Bu rapor o kararın sayısal gerekçesi.

**Ama defter 19 ve 20'de düzeltilmemiş.** İkisinde de `SKIP_POOLS = []` ve
`POOL_ROLE = "all"`; yani havuz hâlâ hem eğitiyor hem not veriyor. Defter
19'un koşusunda skorlanan 11 443 örneğin **3 060'ı (%26,7)** SegFly'dan
(755 truck + 2 305 vehicle) — projenin kendi kuralına aykırı: yeniden
kurulmuş örnekler eğitir, **asla skorlanmaz**.

---

## 6. RGB havuzu: 69 521 örnek hasat edildi, o koşuda kullanılamadı

`edgetam-stage-b/thermal_rgb/verdict.json`, `unusable` listesinde:

```
segfly_rgb: 1969 records and not one usable frame ({'no_image': 1969})
```

Kayıtlar `/content/data/SegFly_rgb/rgb/…` diyordu, defter ise havuzun kökünü
`SegFly` yapıyordu; `Relocator` hiçbir son eki tutturamadı ve 1 969 kaydın
tamamı düştü.

**Bu artık düzelmiş durumda** — `b6e7b68` (`Say why a pool's frames could not
be found, and stop miswiring SegFly's RGB`) hem `why_no_image`'i ekledi hem
satırı `["segfly_rgb", "segfly_rgb", "SegFly_RGB", []]` yaptı, ve defterler
yeniden üretildi. Burada durmasının sebebi, §4'ün RGB sayılarının **o
kullanılamayan hasattan** gelmesi: havuz Drive'da duruyor, içeriği bu rapora
girdi, ama hiçbir eğitim penceresi üretmedi. Yeniden koşulduğunda 45 643
`truck` hedefi eğitime girecek — §4 nedeniyle bu istenmeyen bir şey.

---

## 7. "5 378 / 15 007" funnel'ının bir adımı ölçüldü

`bf3e20d` SegFly'ın 15 007 karenin 5 378'i olarak eğitime ulaşmasına dört
aday sebep sayıyor: yarım okunmuş 14,7 GiB arşiv, eşleşemeyen yarılar,
içinde araç olmayan kareler, ve 0,6'lık `CLASS_WEIGHTS`. Bu ölçüm
üçüncüsünü sayıya bağlıyor:

- Yayıncının haritalarında, kapılardan sonra, **kare başına örnek üretmeme
  oranı %29,3** (140 haritanın 41'i). Yani "içinde araç yok" tek başına
  %36'lık kaybı açıklamıyor; sahne dağılımına göre değişiyor ve geri kalanı
  diğer üç adaydan geliyor. Defter 21'in 3. hücresi hangisi olduğunu
  yazdırabilir.

İkinci bir düzeltme, aynı iddianın hacim tarafında: **havuzdaki karelerin
%85,2'sinin bir önceki kare id'si de havuzda.** SegFly bağımsız fotoğraf
değil, ardışık uçuş kareleri; 20 145 örnek 20 145 ayrı nesne değil, aynı
araçların tekrar tekrar sayılması. (RGB diliminde %92,5 — `spread` sahneleri
gezmiş ama shard içinde ardışık kareler almış.)

Sahne kapsaması dokümanla uyuşuyor: termal yalnız scene_03, 04, 05, 09'da;
RGB dilimi dokuz sahnenin hepsini ve üç irtifayı da görüyor. Resmî split de
öyle — termal train = 03/04/05, val/test = 09.

---

## 8. Ne yapmalı — sırayla

1. **`things` içinden `truck`'ı düşür** (`SPECS["segfly"].things = ("vehicle",)`).
   Bedeli 5 101 termal + 45 643 RGB hedef; kazancı, eğitimin en gürültülü
   kaynağının gitmesi. Ara çözüm `truck`'a ayrı ve yüksek bir `min_area`,
   ama §4(b) bunların büyük kardeşlerinin parçası olmadığını gösterdiği için
   eleme muhtemelen sınıfın tamamını götürür.
2. **Defter 19 ve 20'yi 22/32 ile aynı hizaya getir**: `SKIP_POOLS`'a
   `segfly_thermal`, ya da en azından `POOL_ROLES["segfly_thermal"] =
   "train"`. Bugün o iki defterde test penceresinin üçte biri SegFly'ın.
3. **Mod tutarsızlığını kapat.** `tools/build_notebooks.py:720`'deki defter
   metni `watershed` diyor, kod her yerde `components`; ve Drive'daki havuz
   watershed ile hasat edilmiş, yani `--dataset` yolundan farklı hedefler
   taşıyor. §3'ün tablosu `components` lehine.
4. **RGB havuzunu yeniden hasat etmeden kullanma.** Yol düzeldi (§6), ama
   havuzun içeriği %65,7 `truck`; olduğu gibi açmak §4'ün problemini
   eğitime dört katıyla sokar.
5. **`decompose` havuzu reddettiklerini de yazsın.** `verdict.json`'daki
   `"rate": 1.0, "rejected": {}` bir ölçüm değil, ölçümün yokluğu.
   `e35f6f8` öğretmenli havuzlar için okumaları kaydetmeye başladı;
   `decompose:watershed` ile yazılan havuz hâlâ `decompose`'un döndürdüğü
   `rejects` sözlüğünü atıyor. Bu rapor için o sayının tek kaynağı 178
   GiB'lik parquet'e geri dönmekti.

## Ölçülmeyenler

- **Piksele bakılmadı.** Yalnız etiket haritaları indirildi. §4'ün "`truck`
  kamyon değil" okuması geometriye dayanıyor; kesin cevap defter 26'nın
  panelini `truck` bileşenlerine odaklayarak koşmayı ister.
- **`truck`'ın halka analizi termalde tek sahneden.** Örneklenen üç termal
  shard'ın yalnız ikisinde (scene_09/40 m ve /50 m) id 36 var; havuz
  sayıları `truck`'ın scene_03'te de olduğunu söylüyor (%30,5 pay).
- **Kust4K ve Caltech'in aynı ölçümü.** `encoder_training_todo.md` §6'nın
  3. sorusu hâlâ açık; bu rapor yalnız SegFly'ı kapatıyor.

## Tekrarlanması

```bash
# Yayıncının etiket haritaları + decompose, her iki mod (indirme gerekmez;
# `label` kolonu range isteğiyle okunur):
python tools/segfly_decompose_audit.py --shards 63 410 692 --out /tmp/segfly_audit
python tools/segfly_decompose_audit.py --shards 475 550 50 --modes components

# Havuz kayıtları:      Drive edgetam-pool/segfly_{thermal,rgb}/*.jsonl
# Aşama B kararları:    Drive edgetam-stage-b/{thermal,thermal_rgb}/verdict.json
# Tek karelik panel ve SAM çapraz kontrolü: notebooks/26_segfly_instance_audit.ipynb
```
