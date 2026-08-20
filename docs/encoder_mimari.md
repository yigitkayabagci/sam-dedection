# Encoder Mimarisi — format, basitçe ve eksiksiz

`docs/encoder_training_todo.md` **ne yapılacağını** planlıyor. Bu belge
**nasıl kurulduğunu** anlatıyor: hangi parça neyi eğitiyor, veri hangi şekle
giriyor, kayıp hangi terimlerden oluşuyor, ve hangi dosya nereden sorumlu.

Çalışan hâli: `notebooks/07_encoder_aerial_rgbt.ipynb` (üç veri seti) ve
`notebooks/08_encoder_vtuav_only.ipynb` (sadece VTUAV VIS).

---

## 0. Kısa cevap: hangi model nerede, şu an ne yapıyoruz

**Evet — şu anki tek amaç encoder'i eğitmek.** Video ve hafıza yolu (aşama C)
bu işin dışında; oraya encoder sağlamlaştıktan sonra geçilecek.

Üç model adı geçiyor ve **üçü çok farklı roller**. En sık karışan yer burası:

| model | bizim için ne | ne zaman çalışıyor | eğitiliyor mu |
|---|---|---|---|
| **SAM 2 / EdgeTAM** | **Modelin kendisi.** EdgeTAM, SAM 2'nin küçültülmüş forku (13,9 M) ve `sam2` paketi olarak kuruluyor. Eğittiğimiz ağırlıklar bunun | Her zaman — aşama A ve B'nin ikisinde de | **Evet.** Zaten iş bu |
| **DINOv3** | **Öğretmen**, model değil. RGB yarısına bakıp öznitelik üretiyor; bizim encoder termal yarısına bakıp aynı özniteliği üretmeye çalışıyor | Yalnızca **aşama A**. Bitince atılıyor | Hayır — donuk, sadece okunuyor |
| **SAM 3** | **Modelde yok, eğitim döngüsünde yok.** Tek yeri çevrimdışı **masklet üretimi** (aşama C verisi, `tools/make_masklets.py`) — orada da varsayılan öğretmen SAM 2.1, SAM 3 tek string uzakta | Sadece çevrimdışı, eğitimden önce bir kez | Hayır — donuk, sadece okunuyor |

**DINOv3 mimariyi bozmuyor.** Distilasyon `image_encoder`'ın **ağırlık
değerlerini** değiştiriyor, başka hiçbir şeyi değil: aynı modüller, aynı
state-dict anahtarları, aynı ONNX grafiği, aynı TensorRT motorları. Projeksiyon
başlığı eğitim sonunda çöpe gidiyor. Üstelik aşama A sadece bir **başlangıç
noktası**; aşama B ardından SAM 2'nin kendi hedefiyle (prompt'lu segmentasyon,
focal + dice + IoU) gerçek maskeler üzerinde eğitiyor. DINOv3 öznitelikleri
işe yaramasa aşama B onları geri çeker; kaybedilen tek şey GPU saati olur.
Notebook'taki `distilled+ft` koşusu tam da bunu ölçmek için var.

**SAM 3 neden yok.** Düşünülen iş şuydu: semantik haritayı örneklere ayırmak
için `components`/`watershed` yerine SAM 3'ü kullanmak — çünkü o, görünüşe
bakarak karar veriyor, maske geometrisine değil. İki bitişik arabayı ayırmak
tam olarak bunu gerektiriyor. Ertelendi, çünkü **bu masrafın gerekli olup
olmadığı henüz ölçülmedi**. Tetikleyici notebook'un füzyon sayısı: bitişik
nesnelerin tek bileşende birleşme oranı ~%15'i geçerse SAM 3 (ya da SAM 2.1)
ayırıcı olarak devreye girer. Altındaysa maliyeti karşılığını vermiyor.

Not: VTUAV VIS'te bu sorun **hiç doğmuyor** — maskeleri zaten örnek bazlı
(`mode="labels"`), yani `decompose`'un yeniden kurmaya çalıştığı şeyin kendisi.
Ayırıcı sorusu sadece semantik setler (Kust4K, SegFly) için geçerli.

---

## 1. EdgeTAM dört parça

```
    görüntü ──▶ image_encoder ──▶ öznitelik ──┬──▶ mask_decoder ──▶ maske
                 (RepViT gövde                │          ▲
                  + FPN neck)                 │          │ prompt (kutu / nokta)
                    4.92 M                    │   sam_prompt_encoder
                                              │
                                              ▼
                            memory_attention ◀── hafıza bankası ◀── memory_encoder
                                2.96 M                                  1.62 M
                            ──── tekrarlayan kısım: sadece kareler arası ────
```

**image_encoder + mask_decoder** bir *durağan görüntü segmentleyicisi*.
**Hafıza yolu** onu *takipçi* yapan şey. İkisi farklı şekilde bozulur, farklı
veri ister, ve SAM 2 onları bu sırayla eğitir — önce statik görüntüyle
görüntü ön-eğitimi, sonra video. SAM 2 daha uzun dizilere geçerken
**image encoder'ı tamamen donduruyor**. Yani bu ayrım bizim icadımız değil,
tarifin kendisi.

---

## 2. Üç aşama

| | A — ön-eğitim | **B — bu notebook** | C — video (sonraki) |
|---|---|---|---|
| **eğitilen** | `image_encoder` | `image_encoder` + `mask_decoder` | `memory_attention` + `memory_encoder` |
| **donuk** | geri kalan her şey | bütün hafıza yolu | **image encoder** |
| **veri** | hizalı RGB-T çiftleri, **etiketsiz** | statik görüntü + yoğun maske | video + örnek kimliği |
| **hedef** | donuk RGB öğretmenin öznitelikleri | prompt başına bir nesnenin maskesi | aynısı, kareler boyunca |
| **kayıp** | kosinüs mesafesi | 20·focal + dice + IoU-L1 | + object-score BCE |
| **freeze çağrısı** | `apply_freeze(model, "backbone")` | `apply_freeze(model, "encoder")` | C'nin kendi aşaması (henüz yok) |
| **kod** | `src/training/distill.py` | `src/training/image_loop.py` | `src/training/clip_loop.py` (hazır) |

Bu kurulumun çoklu-öğretmen alternatifine karşı değerlendirmesi, literatürün
neyi desteklediği ve neyi çürüttüğü: **`docs/encoder_arastirma.md`**. Oradan
koda giren iki şey — aşama B'de **çapa terimi** (`--anchor-weight`, aşama A'nın
geri alınmasını engelliyor) ve distilasyonda **moment terimi**
(`--moments`) — aşağıda ilgili bölümlerde.

Aşama sırası keyfi değil, gerekçesi TODO §1'de.

---

## 3. Asıl mesele: etiket çoğu sette yanlış tipte

Kust4K, SegFly, Caltech Aerial — hepsi **semantik** segmentasyon: piksel →
sınıf, yani "bütün arabalar araba". Takipçinin istediği bunun tersi:
**"şu araba, yanındaki değil"**.

**Bir set istisna** ve varsayılan olarak `DATASETS`'te: **VTUAV'ın VIS
bölümü** kare başına **örnek (instance) maskesi** veriyor, 1920×1080'de. Orada
aşağıdaki hiçbir şey çalışmıyor — `mode="labels"` her sıfırdan farklı değeri
doğrudan bir hedef olarak okuyor. Aşağısı, elinde sadece semantik harita olan
setler için; ki hâlâ çoğunluk onlar.

### 3.0 Üç `mode`, üç farklı anotasyon

| mode | harita ne | ne oluyor |
|---|---|---|
| `components` | semantik | sınıf sınıf bağlantılı bileşen — §3'teki yeniden kurma |
| `watershed` | semantik | aynısı + ince köprüyle birleşmiş nesneleri ayırma |
| `labels` | **zaten örnek** | hiçbir ayrıştırma yok; her sıfırdan farklı değer bir hedef |

Semantik hedefle eğitmek zarar verir — yan yana iki arabanın özniteliklerini
**birbirine yaklaştırır**, oysa takip için ayrışmaları gerekiyor.

Veri setini değiştirmek çözmüyor (neredeyse bütün yoğun havadan setler
semantik). **Hedefi** değiştirmek çözüyor. SAM 2 de zaten semantik harita
görmüyor: görüntü başına tek tek maske örnekleyip her birini ayrı prompt
ediyor.

```
   semantik harita                sadece "şey" sınıfları       sınıf başına
   ┌───────────────┐              ┌───────────────┐            bağlantılı
   │ yol  yol  yol │              │ ·    ·    ·   │            bileşen
   │ yol  ARB  ARB │   filtre     │ ·   ARB  ARB  │  ───────▶  ┌───┐┌───┐
   │ bina ARB  ARB │  ─────────▶  │ ·   ARB  ARB  │            │ 1 ││ 2 │
   │ bina bina ağaç│  ("stuff"    │ ·    ·    ·   │            └───┘└───┘
   └───────────────┘   atılır)    └───────────────┘          iki örnek,
                                                             iki prompt,
                                                             iki hedef
```

Dört adım (`src/training/aerial.py`):

1. **Sadece "şey" (thing) sınıfları.** Kust4K'nın 9 sınıfının 4'ü: motosiklet,
   araba, kamyon, insan. Yol/bina/ağaç/trafik-tesisi "stuff" — örneği yok, takip
   hedefi değil. (Bu paleti tahmin etmek iki kez yanlış çıktı; bkz.
   `docs/datasets.md` "İki palet yanlıştı".)
2. **Bağlantılı bileşen, sınıf sınıf.** Birleşim üzerinden değil: bir arabaya
   değen bir insan, birleşimde iki sınıfı kapsayan tek bileşen olurdu.
3. **Kapılar** (`InstanceGates`) — nesne olamayacak kadar küçük, olamayacak
   kadar büyük, ya da kendi kutusunu dolduramayacak kadar seyrek.
4. **Bileşen başına bir kutu prompt'u**, hedef o bileşenin maskesi.

### Bunun yarattığı risk ölçülüyor, varsayılmıyor

Tampon tampona park etmiş iki araba bir piksel kenarı paylaşır ve **tek
bileşen** olur; hiçbir kapı bunu ayıramaz. Notebook'un 11. hücresi, tek bir
GPU-saati harcanmadan önce bunun ne sıklıkta olduğuna sayı koyuyor. O sayı
bütün B aşamasının git/gitme kararı.

`InstanceGates.fill` (alan / kutu alanı) bunun sinyali: kendi kutusunun
dörtte birini dolduran bir bileşen genelde **ince bir piksel köprüsüyle
birleşmiş iki nesnedir**. `summarise()` bunu reddedilen bileşenlerin payı
olarak raporluyor.

### 3.1 Kaynaşma kötüyse: dört çıkış yolu, maliyet sırasıyla

**1. O veri setinde `mode=watershed` — bedava, kısmi.** *İnce köprüyle* birleşmiş nesneler
tek bileşen ama uzaklık dönüşümünde (distance transform) iki tepe ve arada bir
vadi; tepelerden tohumlayıp dışa doğru taşırmak onları ayırıyor.
`DATASETS`'te o setin dördüncü alanını değiştir
(`kust4k:...:thermal:watershed`), indeks hücresinden itibaren yeniden koş,
`fill` reddi sayısını karşılaştır. Her mode kendi indeksini önbelleğe aldığı
için geri dönmek bedava. **Denemeden önce sınırını bilmek gerekiyor:** tam
kenar boyunca bitişik iki dikdörtgen, daha büyük bir dikdörtgene döşenir ve
onun uzaklık dönüşümünde **vadi yoktur**. Maske geometrisiyle çalışan hiçbir
yöntem bunları ayıramaz. Köprü vakasını kurtarır, başka bir şeyi kurtarmaz.
(`src/training/aerial.py:split_bridges`, ve `tests/test_aerial.py` **her iki**
sonucu da — kurtardığını ve kurtaramadığını — ayrı ayrı test ediyor.)

**2. Promptable bir öğretmeni ayırıcı olarak kullanmak — bitişik nesneleri
ayırabilen tek yol.** SAM 2.1 zaten bu depoda (`src/training/labels.py`,
Anti-UAV410'un kutularını maskeye çeviriyor). Her bileşenin içine bir **nokta**
prompt'u ver, çıkan maskeyi semantik sınıfla uyuştuğu yerde kabul et: karar
**görünüşe** göre veriliyor, maske geometrisine göre değil, yani piksel kenarı
paylaşan iki araba ona hâlâ iki araba. Bileşen başına bir öğretmen geçişi,
çevrimdışı ve bir kez. Henüz yazılmadı — istersen yazarım.

**3. Daha sert ele, daha az veri kabul et.** `fill`, `min_area` ve sınıf başına
bir `max_area` zaten şüphelileri atıyor; `FILL`'i 0.4'e çekmek daha fazlasını
atar. Daha az örnek, hepsi güvenilir. En ucuzu ve çoğu zaman yeterli — 20 000
temiz örnek, 40 000 yarı-kaynaşmış örnekten iyidir.

**4. Zaten örnek veren bir set kullan — ve biri zaten veriyor.** **VTUAV'ın
VIS bölümü** havadan, RGB-T, 1920×1080 ve örnek maskeli; `mode=labels` onu
doğrudan okuyor, hiçbir ayrıştırma çalışmıyor. Tam bu sebeple varsayılan
`DATASETS`'te. **AeroVIS** örnek maskesi *artı* kimlik veriyor (YTVIS biçimi)
ama sadece RGB; **iSAID** havadan büyük örnek-segmentasyon seti (655 K örnek),
bu aşama için RGB kabul edilebilirse. Ayrıştırma, yoğun *termal* havadan
setlerin **çoğu** semantik olduğu için var — gerçek bir örnek etiketine tercih
edildiği için değil.

---

## 3.5 Tek veri seti yetmez — `Source` neden örnek başına

`--dataset` tekrarlanabilir ve pencereler birleşiyor:

```
--dataset vtuav_vis:/data/VTUAV_VIS:thermal:labels:all       1920×1080, örnek maskesi
--dataset segfly:/data/SegFly:thermal:components:train        640×512, 15 007 kare
--dataset kust4k:/data/Kust4K:thermal:components:train        640×512, 4 024 kare
```

Üç setin iş bölümü: **VTUAV** gerçek örnek maskesiyle hem eğitiyor hem not
veriyor; **SegFly** hacmi getiriyor (Kust4K'nın ~4 katı kare, üç irtifa,
sahne çeşitliliği); **Kust4K** en ince etiketi getiriyor (4 "şey" sınıfı —
motosiklet/araba/kamyon/insan; SegFly'da sadece `vehicle` ve `truck` var).

Sebebi encoder'ın ne taşıdığı: **genel** görsel öznitelik. Tek veri seti = tek
sensör, tek şehir, tek etiketleme alışkanlığı. Kust4K'nın 4 024 karesi bir
head'i ince ayarlamak için makul, bir gövdeyi oynatmak için ince.

Bu yüzden `Source` (spec + gates + mode + modalite) **her `Sample`'ın üstünde**
duruyor, run'ın üstünde değil: bir batch VTUAV'ın örnek maskeleriyle Kust4K'nın
ayrıştırılmış semantik maskelerini **aynı adımda** taşıyabiliyor. Alternatif —
setler arasında sırayla geçmek — trunk'ın her seferinde bir setin
istatistiklerine oturmasına izin verirdi.

Yan fayda: bir sessiz hata sınıfı kapanıyor. İndeksi bir mode ile kurup
maskeleri başka bir mode ile okumak bileşen etiketlerini yeniden numaralandırır
ve her örnek başka bir örneğin maskesiyle eşleşir — kayıp yine de sonlu görünür.
İkisi birlikte taşındığı için artık uyuşmazlık **mümkün değil**; üstelik
`save_index` damgalıyor ve `load_index` reddediyor.

**Hangi set neyi besliyor: `role`.** Beşinci alan. `train` = sadece eğitim,
`eval` = sadece val/test, `all` = normal 80/10/10. Sebebi ölçümün ne demek
olduğu: **semantik haritadan yeniden kurulmuş** örnekler taşıyan bir setin
"ground truth"u kendisi belirsiz — `decompose` iki arabayı kaynaştırdıysa
doğrusu tek blob, ve onları **doğru ayıran** model **yanlış** sayılır. O sayı
modeli değil ayrıştırmayı ölçer. Yani Kust4K `train`, VTUAV VIS `all`: gürültülü
hedeften öğrenmek normal, gürültülü hedefle **not almak** değil.

**Bölme (split) veri seti başına katmanlı.** 4 024 karelik bir set, 100 dizilik
bir setin yanında tek permütasyonla neredeyse tamamen `test`'e düşebilir. Ayrıca
bölme *isim* üzerinden değil indeks girdisi üzerinden: `000123.png` setlerin
çoğunda var, isimle eşleme bir setin karesinde eğitip başkasınınkinde ölçerdi.

---

## 4. Bir batch nasıl kurulur

Encoder 512'de ~38.7 GFLOP ve **prompt'a bağlı değil**. O yüzden görüntü bir
kez encode edilir, pencere içindeki her örnek aynı özniteliklere karşı prompt
edilir:

```
   pencere (512×512 native piksel)      image_encoder'dan tek geçiş
   ┌─────────────────────┐                        │
   │   [araba A]         │                        ▼
   │           [araba B] │  ──────▶  öznitelik ──┬──▶ decoder ← kutu A → maske A
   │   [insan]           │                       ├──▶ decoder ← kutu B → maske B
   └─────────────────────┘                       └──▶ decoder ← kutu İ → maske İ
```

Hem daha ucuz hem daha iyi sinyal: batch artık **aynı sahneyi paylaşan** birden
çok nesne içeriyor — kaybın yapması gereken ayrım tam olarak bu.

Teknik olarak: `forward_image` bir kez çalışır, düzleştirilmiş öznitelikler
(SAM 2'nin HWxNxC düzeni) batch ekseninde `index_select` ile örnek başına
seçilir. SAM 2 zaten **nesneleri** batch'liyor — her satır kendi prompt'unu,
kendi pointer'ını, kendi skorunu taşır ve dikkat satırları karıştırmaz — yani
N örnek, N takip edilen nesnenin bindiği makinenin aynısına biner.

**Padding hesaplanmaz, maskelenmez.** Pencere başına örnek sayısı değişiyor;
`valid` gerçek satırları öznitelikler çoğaltılmadan *önce* seçiyor, dolayısıyla
boş bir slot hiç maliyet çıkarmıyor.

---

## 5. Tek kare ileri geçiş — memory yolu hiç çalışmıyor

`src/training/image_loop.py`, `clip_loop.py`'nin kardeşi ve **ortak kare için
aynı kod yolunu** kullanıyor. Tek kare zaten bir klibin 0. karesi demek:

```python
model.track_step(
    frame_idx=0, is_init_cond_frame=True,
    current_vision_feats=..., point_inputs=box_prompt(...),
    output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
    num_frames=1, run_mem_encoder=False,      # kimse okumayacak
)
```

SAM head'ine doğrudan uzanmak yerine `track_step`'ten geçmenin sebebi: ilk
karenin öznitelikleri burada da çıkarımdaki gibi koşullanıyor
(`directly_add_no_mem_embed` config'i ne derse desin), ve bakımı gereken ikinci
bir görüş oluşmuyor.

- `run_mem_encoder=False` → memory encoder hiç çalışmaz.
- Kayıtlı hafıza olmadığı için memory attention'ın okuyacağı bir şey yok.

---

## 6. Kayıp — hangi terim var, hangisi neden yok

`src/training/losses.py:instance_loss`:

```
kayıp = 20 · focal + 1 · dice + 1 · IoU-L1
```

SAM 2'nin görüntü ön-eğitimi hedefinin aynısı. `focal` tasarım gereği
baskın: havadan bir aracın kaç piksel tuttuğunu ölçeğine yansıtan tek terim o.

**`object_score` terimi sıfır ağırlıklı değil, hiç yok.** O terim
Anti-UAV410'un kare başına `exist` bayrağına karşı BCE ve statik sette öyle bir
etiket yok — prompt edilen her örnek tanım gereği mevcut. Terimi bırakmak
"hiçbir şey öğretmemek" değil, **koşulsuz ateşlemeyi öğretmek** olurdu; ve
`object_score_logits`'in yersiz ateşlemesi tam olarak videoda hafıza bankasını
zehirleyen şey. Etiketin gerçek olduğu yerde (video, aşama C) eğitilecek.

**`box_projection` de yok**, ters sebeple: o, öğretmen maskesi kalite kapısından
geçemeyen kareler için yedek. Burada maske *zaten* ground truth; geri
düşülecek bir şey yok.

---

## 7. Pencere ve bölme (split)

- **Bölme kare üzerinden, asla pencere üzerinden.** Bir görüntünün iki penceresi
  piksellerinin çoğunu paylaşır; pencere seviyesinde bölmek neredeyse aynı
  görüntüleri çizginin iki yanına koyar ve "held-out" sayı anlamını kaybeder.
- **İsim sırasına göre, dosya sırasına göre değil.** Bir uçuşun ardışık kareleri
  diskte yan yana duruyor.
- Her kare `PER_IMAGE` pencere alır, rastgele seçilen bir örneğe (anchor)
  ortalanır; pencereye **tamamen** giren diğer örnekler bedava prompt olarak
  gelir.
- Kenardan kesilen örnek **kırpılmaz, düşürülür**: piksellerinin sahip olmadığı
  bir genişliği iddia eden kutu, hedefsizlikten daha kötü bir hedeftir.
- **`native`**: 640×512 kaynakta 512 pencere = yeniden örnekleme yok, sensörün
  kendi pikselleri — Orin'in `crop512` girdi modunun aynısı. Notebook bunun
  oranını raporluyor; düşükse küçük nesneler eğitim görmeden önce yeniden
  ölçeklemede yok ediliyor demektir.

---

## 7.5 Üç ayrı "öğretmen" var — karıştırılmasın

Bu projede "teacher" kelimesi **üç farklı işi** anlatıyor ve üçü birbirinden
bağımsız. En sık karışan yer burası:

| rol | kopyalanan şey | ne zaman | hangi model |
|---|---|---|---|
| **1. Sahte etiket** (mask distillation) | öğretmenin **çıktı maskesi** | Anti-UAV410 — sadece kutu etiketi var, maske yok | **SAM 2.1 Hiera Large** ✓ zaten kullanılıyor (`labels.py:Sam2Teacher`) |
| **2. Öznitelik distilasyonu** (aşama A) | öğretmenin **iç öznitelik haritası** | RGB-T çiftleri — **hiç etiket yok** | **DINOv3-ViT-B/16** (varsayılan); DINOv2 ve SAM 2.1 tek dize uzaklıkta |
| **3. Öğretmen yok** (aşama B) | — hedef veri setinin **gerçek maskesi** | Kust4K vb. — yoğun maske zaten var | yok |

Yani "neden SAM 2.1 large kullanmıyoruz" sorusunun cevabı: **kullanıyoruz** —
1. rolde, notebook 01/02'de. Soru aslında 2. rol için geçerli.

---

## 8. Aşama A — modalite distilasyonu (etiketsiz)

`src/training/distill.py`. AnyThermal'ın sonucu: DINOv2/v3 sınıfı bir RGB temel
modeli, **hizalı RGB-T çiftleri ve hiç etiket olmadan** bir termal encoder'a
öğretmenlik edebiliyor. Öğretmen RGB yarısına, öğrenci termal yarısına bakar.
Semantik harita hiç okunmadığı için §3'teki problem burada doğmuyor.

**Neden MAE değil.** MAE, SAM 2'nin Hiera gövdesinin ön-eğitim yöntemi, o yüzden
akla ilk gelen o. Ama transfer olmuyor: MAE **token** maskeliyor ve ViT için
tasarlandı; EdgeTAM'ın gövdesi evrişimsel RepViT, maskelenecek token ızgarası
yok. Öznitelik distilasyonunun böyle bir şartı yok — öğretmenle öğrencinin
mimarileri tanım gereği ilgisiz, hizalanması gereken tek şey bir öznitelik
haritası, ve 1×1 projeksiyon herhangi iki kanal sayısını hizalar.
**TODO'daki açık soru 2'nin cevabı bu:** evet, hedef RepViT gövdesine takılıyor,
çünkü distilasyon kaybının öğrenci tarafı mimariden bağımsız.

### Çift nereden gelecek: VTUAV'ın yeri burası

Aşama A **hiç etiket okumuyor**, yani bir veri setinin anotasyonu bu aşama için
alakasız; sadece **çift sayısı** önemli. Bu, veri seti listesini baştan sıralıyor:

| kaynak | hizalı RGB-T çifti | aşama A'da kullanılır mı? |
|---|---:|---|
| **VTUAV** (tam set) | **~1 700 000** (500 dizi, 1920×1080) | **evet — açık ara en büyüğü** |
| VTUAV VIS bölümü | 100 dizi (~340 000) | evet, **ve aşama B'nin de seti** |
| MVUAV | ~53 800 | evet |
| SegFly | 15 007 | evet |
| Kust4K | 4 024 | evet, ve aşama B'nin seti bu |
| *AnyThermal'ın kendi eğitim seti* | *16 943* | *ölçek için* |

Yani "VTUAV maskesiz, az gelmez mi?" sorusunun cevabı tersine: **maskesiz olmak
burada hiçbir şeye mal olmuyor, ve bu notebook'un aşama B'de eğittiği setten yüz
kat fazla çifti var.** Kutu etiketleri ve 100 videoluk maske bölümü bu aşamanın
okuduğu şeyler değil.

VTUAV için iki ayar başka hiçbir yerde olmadığı kadar önemli, ikisi de 1920×1080
video olmasından:

- **`--crop 0.474` (= 512/1080).** 16:9 bir kareyi 512×512'ye sıkıştırmak hem en
  boy oranını bozuyor hem de küçük bir aracı encoder'ın hiç göremeyeceği boyutun
  altına indiriyor — yani encoder, dağıtımın hiç üretmediği görüntü
  istatistikleriyle ön-eğitilmiş olurdu. Kırpma **her iki yarıdan aynı
  normalize konumdaki** kareyi alıyor, böylece hizalama korunuyor; bu oranda da
  native piksel oluyor.
- **`--pairs`.** Kap, seti **baştan kesmiyor**, üzerine yayarak örnekliyor.
  Videodan türeyen bir sette bu incelik değil zorunluluk: VTUAV'ın ilk 5 000
  çifti **iki uçuş**, yani kesen bir kap iki sahnede eğitip beş bin örnek diye
  raporlardı.
- **`--tolerance 1`.** VTUAV'ın iki modalitesi piksel piksel hizalı **değil**
  (yazarların kendi notu). `tolerance=0`'da öğrenci pozisyonu *p* sadece
  öğretmen pozisyonu *p* ile eşleşiyor — hizasızsa öğretmen öğrencinin
  bakmadığı yere bakmış olur ve her pozisyon sistematik hata taşır.
  `tolerance=1` her öğrenci pozisyonunu çevresindeki 3×3 komşuluğun **en iyi**
  öğretmen pozisyonuyla eşleştiriyor, yani bir öznitelik hücresine (stride
  16'da 16 kaynak piksel) kadar kaymayı yutuyor. Bedeli uzamsal kesinlik, o
  yüzden hizalı setlerde 0 kalıyor.

Sorun etiket değil **disk**: 1.7 M kare 1920×1080'de bir Colab runtime'ına
sığmaz. Bir alt küme indir — birkaç dizilik parça bile on binlerce çift eder ki
bu AnyThermal'ın bütün eğitim setinin üstünde.

Nereye distile ediliyor: `image_encoder`'ın en üst çıktısı — mask decoder'ın
gerçekten tükettiği ve ONNX encoder grafiğinin ürettiği [B, 256, S/16, S/16]
tensörü. Ara bir gövde katmanına distile etmek, decoder'ın sadece hiç
eğitilmemiş katmanların arkasından gördüğü bir şeyi iyileştirmek olurdu.

Projeksiyon **iskele**: aşama bitince atılıyor. Çıkan checkpoint sıradan bir
EdgeTAM state dict'i.

### "SAM tabanlı bir modeliz, DINO öğretmeni bozmaz mı?"

**Yapısal olarak bozamaz.** Distilasyon sadece `image_encoder` içindeki
**ağırlık değerlerini** değiştiriyor: aynı modüller, aynı state-dict
anahtarları, aynı ONNX grafiği, aynı motorlar. Projeksiyon atılıyor. Üstelik
aşama A yalnızca bir **başlangıç noktası** — arkasından gelen aşama B, SAM 2'nin
kendi hedefiyle (promptable segmentasyon, focal + dice + IoU) **gerçek
maskeler** üzerinde eğitiyor. DINO öznitelikleri işe yaramazsa aşama B onları
geri çekiyor ve tek kayıp GPU saati oluyor.

**Gerçek risk yapısal değil, optimizasyona ait:** aşama A encoder'ı oynatırken
mask decoder donuk, yani decoder hiç eğitilmediği özniteliklerle karşılaşabilir.
Bu yüzden trunk 5e-5'te, EMA var, ve aşama B **önce head-only** aşamasıyla
başlıyor — encoder tekrar oynamadan önce decoder yeniden uyum sağlıyor.
`distilled+ft` koşusu `finetune`'dan **kötü** çıkarsa teşhis budur:
`--trunk-lr` düşür ya da `EPOCHS = (2, 3)` yap.

**Ve SAM 2.1'i öğretmen yapmanın gerçekten iyi bir gerekçesi var:**

| | **DINOv3-ViT-B/16** (varsayılan) | DINOv2-base | SAM 2.1 Hiera-Large |
|---|---|---|---|
| öznitelik ne kodluyor | **semantik**, ve özellikle **yoğun** olanı — sürümün bütün derdi bu | semantik | **sınıftan bağımsız sınır** — her şeyi segmentler, *ne* olduğunu bilmez |
| 512 girdide ızgara | 512/16 = **32×32 — öğrencinin ızgarasının aynısı**, hiç yeniden örnekleme yok | 518/14 = 37×37 → 32'ye interpolasyon | 1024/16 = 64×64 → 32'ye interpolasyon |
| hedef tensörle uyum | 768-d → 256'ya projekte | 768-d → 256'ya projekte | `fpn_hidden_states[-1]` **zaten** 256-d, stride 16 |
| öğrenciyle ilişkisi | ilgisiz model | ilgisiz model | EdgeTAM **zaten** SAM 2'nin distilasyonu; aynı hedefi modalite boşluğu ekleyerek sürdürmek |
| adım maliyeti | 512 girdi, base | 518 girdi, base | 1024 girdi, Hiera-Large — EdgeTAM'ın var olma sebebi olan model |
| erişim | **kapalı (gated)** — şartları bir kez kabul et, `HF_TOKEN` ver | açık | açık |

**Neden artık DINOv3 varsayılan.** Bu aşamanın kopyaladığı şey **yoğun** bir
öznitelik haritası, pozisyon pozisyon — ve yoğun öznitelik kalitesi tam olarak
DINOv3 sürümünün konusu: selefinin eğitim ölçeklendikçe yoğun tahminde
bozulduğu belgeli, DINOv3 bunu düzeltmek için çıktı. 16-piksel yaması ikinci ve
daha küçük bir kazanç: 512'de 32×32 ızgara üretiyor, yani **öğrencinin
ızgarası** — öğretmenin haritası kayıptan önce hiç yeniden örneklenmiyor.
(AnyThermal'ın yayınlanmış sonucu DINOv2 kullanıyor; seçenek bu yüzden tek
string uzakta duruyor ve `facebook/dinov2-base` hesap istemiyor.)

DINO öğretmeni lehine genel argüman: **termal encoder'ın eksiği semantik, sınır
değil** — soğuk arka plandaki sıcak nesne çoğu zaman RGB'den *daha kolay* bir
kenar. SAM 2.1 lehine argüman: dördüncü sütunun tamamı.

**Hiçbiri ölçülmedi.** O yüzden argüman değil, anahtar:

```python
TEACHER = "facebook/dinov3-vitb16-pretrain-lvd1689m"   # varsayılan (gated)
TEACHER = "facebook/dinov2-base"                       # açık, hesap gerekmez
TEACHER = "facebook/sam2.1-hiera-large"                # SAM 2'nin kendi encoder'ı
```

`build_teacher` sınıfı **id'den** seçiyor, ayrı bir bayraktan değil: SAM 2.1
checkpoint'i DINOv2'nin 518'inde de yüklenir, koşar ve hiç eğitilmediği
interpolasyonlu pozisyon gömmelerinden öznitelik üretir — sessizce yanlış.
Girdi boyutu öğretmenle birlikte seçiliyor (ViT 518, SAM 2.1 1024).

**Ve hangisi olursa olsun, hiçbir şey yapmamayı yenmek zorunda.** Taban
çizgisi: stok checkpoint'ten başlayan aşama B. Notebook üçüncü bir koşu yapıp
sadece `--base`'i değiştirerek bunu ölçüyor. Kosinüs mesafesi **öğretmenler
arasında karşılaştırılamaz** — sadece test bölmesindeki sayı karşılaştırılabilir.

---

## 9. İki yöntem, tek döngü

`--method finetune | lora`. Aralarındaki fark **tam olarak iki şey**: hangi
parametreler çözülüyor, ve checkpoint nasıl yazılıyor (LoRA bir kopyayı
birleştiriyor). Aşamalar, batch sırası, tohum, kayıplar, one-cycle programı,
gradyan kırpma, EMA, doğrulama dilimi, checkpoint saklama kuralı — hepsi
`src/training/schedule.py`'de bir kez duruyor ve ikisi de oradan geçiyor.
Yoksa karşılaştırma iki yöntemi değil iki notebook'u ölçerdi.

| | kısmi fine-tune | LoRA (r=16) |
|---|---|---|
| ne hareket eder | head, sonra head + image encoder | o katmanların yanında `B @ A` |
| eğitilen | ~9.3 M | ~0.3 M |
| düzenlileştirme | hafıza yolunu dondurmak | rank'in kendisi + aynı dondurma |
| ne teslim edilir | bir checkpoint | **aynı checkpoint** — adapter'lar yazmadan önce birleştirilir |

Son satır testi adil yapan şey. `k × k` evrişim ile `1 × 1` evrişimin
bileşkesi **bir `k × k` evrişimdir**, yani faktörizasyon RepViT gövdesi için de
tam; "LoRA sadece `nn.Linear`'a ulaşır" itirazı, yönteme değil Linear-only bir
uygulamaya yapılan itiraz.

**Bilerek eşitlenmeyen tek şey öğrenme oranı.** LoRA `B = 0`'dan başlıyor,
güncellemesinin gideceği yol daha uzun; her yayınlanmış tarif ona bir mertebe
fazlasını veriyor. Fine-tune'un oranını dayatmak, LoRA'yı değil kötü ayarlanmış
bir LoRA'yı ölçmek olurdu. İki oran da `tools/train_encoder.py:RATES`'te, tek
yerde, görünür.

---

## 10. Değerlendirme — ve bu sayının ölçmediği

`tools/eval_instances.py`: kutu gir, maske çık, IoU. `test` bölmesinde, aynı
tohumlu kare seviyesi bölmeden, hiçbir koşunun görmediği kareler.

**Ortalamayı değil küçük-nesne sütununu okuyun.** Havadan bir setin ortalama
IoU'su kamyonlarının hâkimiyetinde; bu hattın var olma sebebi olan hedefler ise
yirmi piksellik olanlar.

**Ve hepsi bir vekil (proxy).** Tek prompt'lu kare skorlanıyor, dolayısıyla bu
projenin düzeltmeye çalıştığı hatayı **göremez** — o hata hafıza bankası
gerektiriyor, hafıza bankası da aşama C. Daha iyi bir encoder, daha iyi bir
takipçi için **ön koşuldur, kanıt değildir.**

---

## 11. Hangi dosya neden sorumlu

| dosya | sorumluluk |
|---|---|
| `src/training/aerial.py` | veri seti düzeni, palet, semantik → örnek ayrıştırma, kapılar, pencere geometrisi, piksel okuma |
| `src/training/image_loop.py` | tek kare ileri geçiş, batch, ragged padding, örnek IoU, image-mode akış |
| `src/training/distill.py` | öğretmen, projeksiyon, kosinüs kaybı, ön-eğitim döngüsü |
| `src/training/losses.py` | `instance_loss` (statik) ve `frame_loss` (video) — hedef tek dosyada |
| `src/training/schedule.py` | iki aşama, EMA, checkpoint kuralı — `Loop` ile iki veri moduna da hizmet ediyor |
| `src/training/finetune.py` | `STAGES` / `apply_freeze` — `backbone`, `head`, `encoder` |
| `src/training/lora.py` | PEFT; `STAGES` tablosunu fine-tune ile paylaşıyor |
| `src/training/masklets.py` | video öğretmeniyle kutu → masklet: parçalı yayılım, kare başına kutu kapısı, kalibrasyon — aşama C'nin veri fabrikası (arastirma §8) |
| `tools/make_masklets.py` | masklet hattının giriş noktası; `--calibrate` VIS'in çizilmiş maskelerine karşı ölçüyor |
| `tools/fetch_aerial.py` | Drive/arşiv/URL → yerel disk + palet doğrulaması |
| `tools/pretrain_encoder.py` | aşama A giriş noktası |
| `tools/train_encoder.py` | aşama B giriş noktası, `--method finetune\|lora` |
| `tools/eval_instances.py` | held-out örnek IoU |
| `notebooks/07_encoder_aerial_rgbt.ipynb` | hepsinin tek oturumdaki tarifi |
| `configs/edgetam_512_aerial{,_lora}.yaml` | çıkan checkpoint'lerin dağıtım config'i |

---

## 12. Bittiğinde elde ne var, sonra ne olacak

Sıradan bir EdgeTAM checkpoint'i: aynı anahtarlar, aynı yükleyici, aynı ONNX
export, aynı motor derlemesi. **Hafıza yolu hâlâ stok.**

Export zinciri değişmiyor — ama encoder ONNX grafiği mask decoder'ın
`conv_s0`/`conv_s1` projeksiyonlarını içine katlıyor ve bu aşama decoder'ı da
eğitiyor, o yüzden ikisini de yeniden export edip parity'yi dört modülde de
koşturun.

**Sonraki (aşama C):** bu encoder'ı **dondur**, `memory_attention` ve
`memory_encoder`'ı aç, örnek kimlikli videoyla eğit (AeroVIS / UAVScenes),
SAM 2'nin yaptığı gibi statik veriyle dönüşümlü besleyerek.
`FROZEN_MODULES` bugün bunun tersini yapıyor, yani `STAGES`'e kendi girdisi
eklenecek. `clip_loop.py` hazır.

Aşama C'nin veri tarafı da hazır: `tools/make_masklets.py`, VTUAV'ın kutu
etiketli ~400 dizisini bir video öğretmeniyle (varsayılan SAM 2.1; SAM 3 tek
string) masklet'e çeviriyor ve çıktıyı `labels.py`'ın deposunun aynısına
yazıyor. Harcamadan önce ölçü: `--calibrate`, VIS bölümünün çizilmiş
maskelerine karşı masklet IoU'sunu basıyor — o sayı zayıfsa fabrika
çalıştırılmaz. Ayrıntı ve literatür: `docs/encoder_arastirma.md` §8.
