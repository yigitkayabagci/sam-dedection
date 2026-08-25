# Semantic kurtarma — etiket haritasından *örnek* çıkarmak

Defter `15_semantic_mask_pool.ipynb`, modül `src/training/semantic.py`,
CLI `tools/make_semantic_pool.py`.

Bu, repodaki **üçüncü** etiketleyici ve kutuya ihtiyaç duymayan tek olanı:

```
13   tespit kutusu   ->  öğretmen maskesi   (havadan RGB)
14   tespit kutusu   ->  öğretmen maskesi   (termal, rota önce ölçülüyor)
15   semantik harita ->  örnek              (hiç kutu yok)
```

---

## 1. Problem, sayıyla

`aerial.decompose` semantik haritayı sınıf başına bağlı bileşenlere ayırıyor.
Bir etiket haritasının tek başına yapabileceği tek şey bu, ve tek bir hatası
var — ama o hata ayrıntı değil:

| set | ölçülen |
|---|---|
| **SegFly** | bir-araç-bir-blok isabeti **%78,5**; araç piksellerinin **%20,9'u** çok-araçlı bileşende |
| **iSAID** (çizilmiş örnekler, aynı test) | 3 523 çizilmiş araç → 2 889 bileşen (**%82,0**); örneklerin **%26,9'u** ortak blokta |

Yan yana park etmiş iki araba **birbirine değiyor**. Aralarındaki sınır bir
*sınıf* sınırı değil, dolayısıyla haritada yok, dolayısıyla tek bileşen,
dolayısıyla aşama B bir arabaya verilen prompt'a iki arabayı kaplayan maskeyle
cevap vermeyi öğreniyor.

Eşikle çözülmüyor: `InstanceGates.fill` "ince köprüyle birleşmiş iki nesne"
için yazıldı, ve gerçek bir **altı araçlık blok fill 0,79** ile dört kapının
dördünü de geçiyor.

**Harita onları ayıramaz çünkü iki tane olduğunu hiç bilmedi. Görüntü biliyor.**

## 2. Fikir

Bir **kutu**, park etmiş araç sırasının etrafına çizilince sırayı içerir ve
öğretmen sırayla cevap verir. Bir **tık**, tek kaputun üstüne konunca tek
arabayla cevap verir.

Roller 13/14'ün tam tersi:

| | prompt nereden | ne doğruluyor |
|---|---|---|
| 13 / 14 | insanın **kutusu** | geometri (`box_iou`) |
| **15** | haritanın **bileşeni** | insanın **haritası**, piksel piksel (`purity`) |

İkinci sütun bu rotanın sahip olduğu ve kutu havuzunun asla olamayacağı şey:
etiket haritası **çizilmiş**, yani geri gelen maskenin doğru sınıf olup
olmadığını söyleyebiliyor. Arabanın altındaki yolu da yutmuş bir maske temiz
tek parça ve doğru yerde — hiçbir geometrik kontrol sorunu görmez, çizim görür.

## 3. Mekanik

```
bileşen -> kaç nesne? -> o kadar iç nokta -> her tık ayrı segment -> kapılar
```

**Kaç nesne?** `unit_areas`: sınıf başına bileşen alanlarının **medyanı**.
Alternatif GSD priorü (metre/piksel × arabanın gerçek boyu) irtifa ister ve
SegFly gibi 30/40/50 m karıştıran bir sette yanlıştır. Medyan çalışır çünkü
**bileşenlerin çoğu zaten tek nesne** (SegFly %91,9; iSAID'in çizilmiş
araçlarının %91,5'i tek başına), ve kaynamış azınlık medyanı ortalamayı
çektiği kadar çekmez.

> **Birim tahmini kare başına değil veri seti ölçeğinde yapılmalı.** Kare
> başına tam ihtiyaç duyulan yerde çöküyor: tek bileşeni kaynamış bir çift
> olan karede medyan o çiftin kendisidir, seed sayısı 1 çıkar, hiçbir şey
> ayrılmaz. Bu, uygulamada bulunup düzeltilen bir hatadır — `estimate_units`
> ve `measure_rescue`'nun ilk geçişi bunun içindir.

**Noktalar nereye?** `seed_points`: bileşenin *sırtı* üzerinde
farthest-point sampling. Sırt = mesafe dönüşümü en derin değerinin en az
yarısı kadar olan pikseller.

- **İç olmalı**, çünkü iki arabanın dikişine yakın bir tık ikisini birden
  döndürmeye en yatkın prompt'tur.
- **Yayılmış olmalı.** İlk deneme "tepe al, çevresini bastır" idi ve tam bu
  modülün var olduğu şekilde başarısız oldu: park etmiş araç sırası uzun ve
  ince tek bir bölgedir, mesafe dönüşümü **kısa eksenle** sınırlıdır, ondan
  türetilen bastırma yarıçapı bir arabanın uzunluğundan küçüktür ve her seed
  ilk araca düşer. Farthest-point sampling ölçeğini iç yarıçaptan değil
  bölgeden alır.

**Kapılar** (`SemanticGates`) — bilerek `labels.Gates` + bir alan değil. O set
`box_iou` etrafında kurulu ve tıkın kutusu yok:

| kapı | ne | varsayılan |
|---|---|---|
| `purity` | maskenin piksellerinin ne kadarı haritada doğru sınıf | 0,65 |
| `containment` | maskenin ne kadarı kendi bileşeninin içinde | 0,70 |
| `unit_area` | maske alanı / bir nesnenin tahmini alanı | (0,30, **1,75**) |
| `component` | tek nesne tek parça | 0,80 |
| `teacher_iou` | öğretmenin kendi skoru — zayıf, simetri için | 0,0 |

`unit_area`'nın üst sınırı **1,75, 2 değil** — buradaki tek ölçüm olmayan,
yargı olan sayı. İki birim tam olarak en yaygın füzyondur, yani 2'de bir sınır
düzeltilmeye çalışılan durumu kabul eder. Bedeli gerçekten büyük tek nesnedir
(araba medyanına karşı bir kamyon), ve bu takas bilinçli: iki hata simetrik
değil. Düşen bir örnek bir eksik eğitim örneğidir; **kabul edilen iki-araçlık
bir maske modele tam olarak bu rotanın silmek için var olduğu hatayı öğretir.**

## 4. Kararı veren ölçüm

Modül daha iyi olduğunu **iddia etmiyor**. `measure_rescue` karar veriyor:

1. çizilmiş örnek taşıyan bir set al (iSAID; VTUAV'ın VIS bölümü),
2. onları ima ettikleri semantik haritaya **düzleştir** — ayrımı atarak, ki bu
   tam olarak üretimde görülen girdidir,
3. o harita üzerinde **iki rotayı da** koştur: düz `decompose` ve kurtarma,
4. her ikisini de düzleştirildikleri örneklere birebir eşle.

Hiçbir rota ayrımı görmüyor; ayrım yalnızca skorlandıkları gerçekte var.
Kurtarma orada `components`'ı recall'da geçmiyorsa, kimsenin kontrol
edemediği Kust4K'da da geçmeyecek.

Geçilmesi gereken çıta "hiçlikten iyi" değil, **`decompose`'dan iyi** — ki o
hiç GPU harcamıyor.

## 5. Ne zaman yanlış rota

Üçü de ölçülmüş, kimse yeniden keşfetmesin:

- **Hiçbir şey kaynamamışsa.** Defterin probe hücresinin son sütunu testtir.
  Seyrek setlerde bileşenler zaten tek nesne ve kurtarma yalnızca precision
  kaybettirir. **Caltech'in `pairs` arşivi ölçüldü: 289 karede 138 örnek,
  karelerin yalnızca %14,9'u bir şey veriyor** — çoğu açık arazide tek başına.
- **Nesneler çok küçükse.** Tıkın içine düşeceği bir iç bölge gerekir. Kabaca
  20 px altında hedefin sırtı yoktur, `seed_points` tek nokta döndürür, ve bu
  `decompose`'un fazladan adımlarla yapılmış halidir.
- **Harita zaten örnekse.** VTUAV VIS ve iSAID ayrımı *taşıyor*;
  `--mode labels` doğrudan okur ve burada hiçbir şey koşmamalı.

Ve bu defterle ilgili olmayan bir tane daha: termalde **hacim** gerekiyorsa
14'ün kutu havuzları hâlâ daha büyük kaynaktır — tek başına HIT-UAV 0,4 GB'da
24 899 kutu taşıyor, ki bu buradaki bütün semantik termal setlerin toplamından
fazla süpervizyondur.

## 6. Caltech Aerial RGB-T — ölçüldü, verdikt

Kullanıcı sorusu "termal için Caltech gibi semantic'leri kullanabilir miyim,
zor veya bozuksa girmek istemiyorum" idi. Cevap: **bozuk değil, ama örnek
kaynağı olarak değmez.**

Arşivin merkezi dizini ranged GET ile okundu, 289 anotasyon çıkarıldı ve
`decompose` gerçekten koşturuldu:

| | `labeled_rgbt_pairs.zip` | `labeled_thermal_singles.zip` |
|---|---|---|
| boyut | 4,29 GB | 4,14 GB |
| içerik | `color/` `thermal8/` `thermal16/` `annotations/` `thermal_ann_overlay/` — her biri **2 282** | `color/` `masks/` `masks_vis/` `masks_overlays/` `thermal8/` `thermal16/` — her biri **3 076** |
| kayıt | RGB-T **hizalı**, 960×600, artık kayma 1–4 px | `color/` kayıtlı eş **değil** → `rgb=None` |
| repodaki spec | `caltech_rgbt` | `caltech` |
| örnek verimi | **289 karede 138 örnek, karelerin %14,9'u** | repo ölçümü: 3 076 maskeden 1 357 örnek, yalnızca 430 maskeden |

`pairs` ölçümünün detayı:

```
138 örnek / 289 kare  ->  tam sete ekstrapolasyon ~1 090 örnek / ~340 kare
sınıflar: vehicles 80, person 58
alan: medyan 328 px (p10 104, p90 1 187)
birim alan (medyan): vehicles 415 px, person 284 px
1,75 birimden büyük bileşen: 42/138 = %30,4   <- kurtarmanın hedefi
```

Sahne içeriği neden böyle olduğunu söylüyor: **su %38,9, çıplak zemin %21,4,
kayalık %13,8, çalı %9,5, ağaç %6,0, gökyüzü %5,3** — araç piksellerin
**%0,030'u**, insan %0,224'ü. Bu doğal arazi ve su; otopark yok, yoğun kentsel
araç yok. Ayrıca kadrajda **gökyüzü** olması nadir çekim olmadığını gösteriyor.

`thermal16/` klasörü 16-bit TIFF taşıyor — SegFly'ın R-JPEG'indeki ham
radyometrik veriye karşılık gelen şey, burada doğrudan dosya olarak. Aşama A
ve gece/doğal arazi alanı için değeri orada.

**Verdikt:** Caltech'i `docs/datasets.md`'nin zaten dediği yerde bırak — aşama
A ve alan çeşitliliği. Örnek kaynağı olarak 4,29 GB'a ~1 090 küçük örnek,
karelerin %85'i boş. Karşılaştırma: HIT-UAV 0,4 GB'da 24 899 kutu. Bu topa
girmeye değmez.
