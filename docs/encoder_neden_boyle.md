# Aşama A: segmentasyon verisi olmadan encoder eğitmek

**Soru:** Amacımız promptlanabilir segmentasyon. Peki neden eğitim verimizin
büyük kısmı segmentasyon maskesi taşımıyor — hatta neden bilerek maskesi
olmayan veri seti arıyoruz? Bu mantıklı mı?

Bu belge o soruyu cevaplıyor. Yaptığımız işin gerekçesi, dayandığı kanıt ve —
en önemlisi — **hangi kısmının hâlâ ölçülmemiş olduğu**.

---

## 1. Kısa cevap

Eğitim tek parça değil, **iki ayrı aşama** ve ikisi birbirinin tersi veriyi
istiyor:

| | Aşama A — distilasyon | Aşama B — promptlanabilir segmentasyon |
|---|---|---|
| ne öğreniyor | *"termal bir sahne neye benzer"* | *"bu prompt hangi nesneyi gösteriyor"* |
| okuduğu | hizalı (termal, RGB) çiftleri | görüntü + **maske** |
| etiket | **hiç yok** | maske, tek denetim sinyali |
| eğitilen | `image_encoder` | `image_encoder` + `mask_decoder` |
| öğretmen | SAM 2.1 Hiera-B+ (donuk); DINOv3 ikinci koşu | yok — hedef gerçek maske |
| kayıp | kosinüs mesafesi | 20·focal + dice + IoU |
| elimizdeki veri | ~150 000 kare | ~4 000 maske |

Son satır bütün meselenin özeti. **Maskeli veri, maskesiz veriden iki kat
küçük.** Eğer sadece maskeli veriyle eğitseydik, encoder'ın göreceği dünya
4 000 kareden ibaret olurdu.

Aşama A "segmentasyon verisi olmadığı için mecburen yapılan bir taviz" değil.
Tam tersi: **etiket okumadığı için** herkesin kullanamadığı veriyi
kullanabiliyor, ve o veri kat kat daha fazla.

---

## 2. Neden encoder'ı ayrı eğitiyoruz

EdgeTAM (SAM 2 forku) dört parça:

```
    görüntü ──▶ image_encoder ──▶ öznitelik ──┬──▶ mask_decoder ──▶ maske
                 4.92 M                       │          ▲
                                              │   prompt (kutu / nokta)
                                              ▼
                            memory_attention ◀── hafıza ◀── memory_encoder
                                2.96 M                        1.62 M
```

`image_encoder` + `mask_decoder` bir **durağan görüntü segmentleyici**. Hafıza
yolu onu **takipçi** yapan şey. İkisi farklı şekilde bozulur ve farklı veri
ister.

Bu ayrım bizim icadımız değil — **SAM 2'nin kendi tarifi.** SAM 2 önce statik
görüntüyle görüntü ön-eğitimi yapıyor, sonra videoya geçiyor, ve uzun dizilere
geçerken **image encoder'ı tamamen donduruyor**. Yani encoder'ın kalitesi video
aşamasından önce belirleniyor; sonra düzeltme şansı yok.

Şu an sadece encoder'la uğraşmamızın sebebi bu.

---

## 3. Aşama A tam olarak ne yapıyor

```
   RGB yarısı  ──▶  SAM 2.1 (donuk) ──▶  öznitelik haritası ─┐
                                                             ├─▶ kosinüs mesafesi
   termal yarısı ──▶ bizim encoder  ──▶  öznitelik haritası ─┘
```

Aynı sahnenin iki hâli. Öğretmen RGB'ye bakıyor, öğrenci termale. Öğrenci,
**termalden bakarak RGB modelinin temsilini üretmeyi** öğreniyor.

Hiçbir yerde etiket okunmuyor. `distill.py`'deki veri tipinin kendi tanımı:

> *"Registered thermal and RGB halves of the same scene. **No labels.**"*

**Neden işe yarıyor:** öğretmen web ölçeğinde eğitilmiş; kenar, nesnelik,
doku, şekil gibi kavramları taşıyor. Termal görüntü aynı dünyayı farklı bir
fizikle görüyor ama **aynı nesneleri** içeriyor. Kayıtlı çift bize "bu termal
desen, şu RGB kavramına karşılık geliyor" eşlemesini etiketsiz veriyor.

**Neden öğretmen SAM 2.1:** görev sınıfsız tek nesne takibi, yani hiçbir yerde
"bu ne" sorulmuyor. SAM 2'nin öznitelikleri tam olarak bunu taşıyor: sınıf
bilgisi olmadan nesnenin nerede bittiğini. Üstüne EdgeTAM **zaten** SAM 2'nin
distilasyonu, dolayısıyla aşama A checkpoint'in doğduğu hedefi sürdürüyor.
Ayrıntı ve DINOv3 karşılaştırması: `docs/encoder_mimari.md`.

**Neden mimariyi bozmuyor:** distilasyon yalnızca `image_encoder`'ın ağırlık
*değerlerini* değiştiriyor. Aynı modüller, aynı state-dict anahtarları, aynı
ONNX grafiği, aynı TensorRT motorları. Projeksiyon başlığı eğitim sonunda
atılıyor. SAM 2'nin mimarisine dokunulmuyor.

---

## 4. "Peki bu gerçekten mantıklı mı?"

Dürüst cevap: **fikir sağlam temelli, ama bizim özel durumumuzda ölçülmedi.**

### Lehine olan kanıt

- **AnyThermal** (yayınlanmış sonuç): DINOv2 sınıfı bir RGB temel modeli,
  hizalı RGB-T çiftleri ve **hiç etiket olmadan** bir termal encoder'a
  öğretmenlik edebiliyor. Bizim kurulum bunun doğrudan uygulaması.
- **SAM 2'nin kendi tarifi** encoder'ı önce ayrı eğitip sonra donduruyor.
- **Veri asimetrisi** tartışmasız: 150 000'e karşı 4 000.

### Aleyhine olan, ya da en azından belirsiz olan

- **Alan farkı.** AnyThermal yer seviyesi termalde çalıştı. Havadan bakış
  farklı: nesneler küçük (VTUAV'da hedef karenin %0.2–2'si), perspektif
  tepeden, arka plan homojen. Bunun transferi *varsayım*, kanıt değil.
- **Öğretmen–öğrenci mimari farkı.** Öğretmen bir ViT (SAM 2'nin Hiera'sı ya
  da DINOv3), bizim encoder konvolüsyonel RepViT. Öznitelikler farklı
  uzaylarda; projeksiyon başlığı bunu kapatmaya çalışıyor ama tam eşleşme
  garantisi yok. SAM 2.1 öğretmeninde bu fark en küçüğü: hedef tensör zaten
  256 kanal ve stride 16, yani öğrencinin ürettiğinin aynı biçimi.
- **Hizalama.** VTUAV'ın iki modalitesi piksel piksel hizalı değil.
  `tolerance=1` (3×3 komşulukta en iyi eşleşme) bunu tolere ediyor, ama
  kusursuz değil.

### Bu yüzden ölçüyoruz, tartışmıyoruz

Notebook'lar **dört ayrı ağırlık** üretip aynı test setinde puanlıyor:

| koşu | ne | neyi cevaplıyor |
|---|---|---|
| `stock` | hiç eğitilmemiş EdgeTAM | taban çizgisi |
| `finetune` | sadece aşama B | aşama A olmadan ne kadar? |
| `lora` | sadece aşama B, PEFT | az parametre yeter mi? |
| `distilled+ft` | aşama A **sonra** aşama B | **aşama A ne kattı?** |

`distilled+ft` ile `finetune` arasındaki fark, aşama A'nın değerinin tanımı.
Tek fark başlangıç ağırlıkları — tarif, veri, seed, her şey aynı.

**Aşama A kazanmazsa, harcanan GPU saati olur ve bunu sayıyla göreceğiz.**
Bu belgenin amacı da bunu peşinen savunmak değil, ölçülebilir hâle getirmek.

---

## 5. Aşama B: maskesiz veriyi maskeliye çevirmek

Aşama B maske istiyor ama havadan setlerin çoğu **semantik** segmentasyon:
piksel → sınıf, yani "bütün arabalar araba". Takipçinin istediği tam tersi:
"şu araba, yanındaki değil".

Semantik hedefle eğitmek **zarar verir** — yan yana iki arabanın özniteliklerini
birbirine yaklaştırır, oysa takip için ayrışmaları gerekiyor.

Çözüm veri setini değiştirmek değil, **hedefi** değiştirmek:

```
   semantik harita          sadece "şey" sınıfları        sınıf başına
   ┌───────────────┐        ┌───────────────┐             bağlantılı bileşen
   │ yol  yol  yol │        │ ·    ·    ·   │             ┌───┐┌───┐
   │ yol  ARB  ARB │  ───▶  │ ·   ARB  ARB  │  ────────▶  │ 1 ││ 2 │
   │ bina bina ağaç│        │ ·    ·    ·   │             └───┘└───┘
   └───────────────┘        ("stuff" atılır)             iki örnek, iki prompt
```

SAM 2 de semantik harita görmüyor: görüntü başına tek tek maske örnekleyip her
birini ayrı prompt ediyor. Biz o şekli elimizdeki veriden üretiyoruz.

**Bu yeniden kurmanın bilinen sınırı var ve ölçülüyor.** Tampon tampona park
etmiş iki araba tek bileşen olur; hiçbir kapı bunu ayıramaz. Notebook füzyon
oranını sayıyor. **%15'i geçerse** SAM 3'ü ayırıcı olarak devreye almak
gündeme gelir — görünüşe bakarak karar verdiği için maske geometrisinin
yapamadığını yapabilir. Altındaysa masrafı karşılığını vermiyor.

**Bir set bu sorundan tamamen muaf:** VTUAV'ın VIS bölümü kare başına gerçek
örnek maskesi veriyor (`mode="labels"`). Yeniden kurmaya gerek yok.

### Ölçümün dürüstlüğü: `role`

Yeniden kurulmuş bir hedefte puan almak, modeli değil **yeniden kurmayı** da
ölçmek olur: ayrıştırma iki arabayı birleştirdiyse, onları doğru ayıran model
*yanlış* diye işaretlenir.

Bu yüzden her veri setinin bir rolü var:

| rol | besledigi | kime |
|---|---|---|
| `train` | sadece eğitim | yeniden kurulmuş örnekler (SegFly, Kust4K) |
| `all` | eğitim + puan | gerçek çizilmiş maske (VTUAV VIS) |

Yani **test sayısı her zaman birinin elle çizdiği maskeye karşı.**

---

## 6. Kullandığımız veri ve neden

| set | çift | maske | aşama | neden |
|---|---:|---:|---|---|
| **VTUAV VIS** | 26 059 | 875 | A + B | tek gerçek örnek maskeli havadan RGB-T set; 1920×1080 |
| **SegFly** | 15 007 | var | B | hacim + irtifa çeşitliliği (30/40/50 m) |
| **Kust4K** | 4 024 | 4 024 | B | 640×512 native, gündüz/gece, en ince etiket |
| **DroneVehicle** | 28 442 | **yok** | **sadece A** | en büyük A kaynağı; 640×512 native, gündüz+gece |

Son satır bu belgenin ana fikrinin somut hâli: **DroneVehicle'ın etiketi
yönlü kutu, maske değil. Segmentasyon için kullanılamaz — bu yüzden bizim
yaptığımız işi yapan kimse onu kullanmıyor.** Aşama A etiket okumadığı için
bizim en büyük tek kaynağımız.

---

## 7. Doğrulanmış olanlar

Bu projede birden fazla kez, "makaleden okunan" bir bilgi gerçek arşivle
uyuşmadı. Aşağıdakiler **canlı sunucudan ölçüldü**, kaynaktan okunmadı:

| ne | bulunan |
|---|---|
| Kust4K paleti | makaleden tahmin **yanlıştı**; id 6 = *ağaç*, takip hedefi sanılıyordu. Doğrusu arşivin kendi `visual.py`'sinden, 4 024 karede uçtan uca doğrulandı |
| SegFly paleti | id'ler boşluklu; tahmin çim/ağaç/bitki örtüsünü hedef yapıyordu |
| VTUAV maske yolu | `mask/ir` ve `mask/rgb` ayrı; değer 1 değil **255** — spec 1 diyordu, her maske boş okunacaktı |
| VTUAV kare adları | her dizi `000000`'dan başlıyor → bir uçuşun görüntüsü başka uçuşun maskesiyle eşleşebiliyordu |
| VTUAV içerik | `train_001`'de **hiç yaya yok** (sadece araç); yayalar 002 ve 003'te |
| DroneVehicle çerçevesi | 840×712, dört kenarda **tam 100 px beyaz**; kırpınca **640×512** — tam native çözünürlük |
| DroneVehicle erişimi | Baidu değil, HF aynası: anonim HTTPS, HTTP 206, 28 MB/s |

Son ikisi olmasa öğretmen model patch token'larının üçte birini beyaz alana
harcayacaktı.

---

## 8. Dürüst sınırlar

1. **Aşama A'nın havadan termalde işe yaradığı ölçülmedi.** Kanıt yer seviyesi
   termalden geliyor. `distilled+ft` koşusu bunu ölçmek için var.
2. **Öğretmen–öğrenci mimari farkı** (ViT ↔ RepViT) köprüleniyor ama
   kanıtlanmıyor.
3. **Yeniden kurulmuş örnekler gürültülü.** Bilerek kabul ediliyor; ölçüm
   sadece gerçek maskede yapıldığı için *sayı* temiz kalıyor.
4. **Test seti dar.** `train_001` tek başına 14 dizi → 1 test dizisi. Üç
   arşivle 50 dizi → 5. İki koşuyu karşılaştıracaksan bu fark önemli.
5. **Aşama C (video/hafıza) hiç başlamadı.** Encoder sağlamlaşmadan
   başlanmaması bilinçli.

---

## 9. Deney tasarımı

Üç notebook, aynı kaynaktan üretiliyor, **tam dört hücrede** ayrışıyorlar:

| | aşama A | aşama B | cevapladığı soru |
|---|---|---|---|
| **07** | VTUAV | üç set | referans koşu |
| **08** | VTUAV | sadece VTUAV | ekstra iki set ne kattı? |
| **09** | **DroneVehicle** | üç set | aşama A çözünürlük mü ister, native piksel + çeşitlilik mi? |

Her biri 07'ye karşı **tek bir değişkeni** izole ediyor. Üçü de aynı VTUAV
dizilerini test için ayırıyor (kaynak başına tohum, listedeki sıraya göre değil),
yani sayılar doğrudan karşılaştırılabilir.

---

## 10. Tek cümlelik özet

> Amacımız segmentasyon, ama segmentasyon etiketi pahalı ve az. Encoder'ın
> öğrenmesi gereken şeyin çoğu — bir sahnenin neye benzediği — etiket
> gerektirmiyor. O yüzden encoder'ı önce etiketsiz ve bol veriyle, sonra
> etiketli ve az veriyle eğitiyoruz; ve aşama A'nın gerçekten katkı verip
> vermediğini savunmak yerine ölçüyoruz.
