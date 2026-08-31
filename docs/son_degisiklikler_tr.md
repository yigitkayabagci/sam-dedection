# Son değişiklikler — ne yapıldı, hangi notebook neden var

Bu dosya `91ac412`'den `a5dcf0e`'ye kadar olan işi tek yerde toplar: hangi
kod eklendi, hangi notebook niye eklendi, ve hangi iddia gerçekten ölçüldü.
Ölçülmemiş bir şey burada iddia olarak yazılmaz — her sayının nereden geldiği
belirtilir.

---

## 1. AeroVIS: hem görüntü hem maske yolu doğrulandı

**Soru:** AeroVIS havuzları gerçekten çözülüyor mu — hem `image` hem
`pseudo_masks.npz` yolu için?

**Cevap: evet, ikisi de çalışıyor.** Ama görüntü yolu, çalışması *amaçlanan*
mekanizma üzerinden değil, yedek mekanizma üzerinden çalışıyordu. Bu düzeltildi.

### Arşivin gerçek düzeni (tahmin değil, okundu)

Drive'daki `AeroVIS.zip` 13.5 GiB. İçindekileri indirmeden okumak için
arşivin *central directory*'si byte-aralığı isteğiyle çekildi (dosyanın son
5.6 MB'ı), ardından yalnızca `aero_vis.json` girdisi açıldı:

```
AeroVIS/aero_vis.json
AeroVIS/data.md
AeroVIS/fig3.png
AeroVIS/sequences/{sd_001..vd_052}/{frame}.jpg     117 klasör, 49 204 kare
```

ve `aero_vis.json`'ın ilk satırları:

```json
"videos": [{"id": 1, "name": "vd_001",
            "file_names": ["vd_001/0000001.jpg", "vd_001/0000002.jpg", ...
```

Yani `file_names` — ki `aerovis.write_pool` bunu `image_rel` olarak saklıyor —
**`AeroVIS/sequences`'e göredir**, arşivin açıldığı `AeroVIS` klasörüne göre
değil. `IMAGE_ROOTS` ise `/content/data/AeroVIS`'i gösteriyor, yani doğru
kökün *iki* seviye üstünü.

`data.md`'den çıkan iki ek gerçek:

* AeroVIS maskeleri **SAM3 + manuel inceleme** ile üretilmiş. Yani "teacher
  olarak SAM3 kullanalım mı" sorusunun cevabı kısmen zaten elimizde: bu
  havuzun maskeleri zaten SAM3 kalitesinde ve üstüne insan eli değmiş.
* 117 video · 49 204 kare · 8 279 track · 9 sınıf.

`aerovis_selection.json`'ın kendi raporu: train 39 943 kare / 1 095 567
instance / **`missing_image: 0`**, heldout 7 978 kare / 234 929 instance /
`missing_image: 0`. Yani hasat anında tek bir kare bile kaybolmamış.

### Bulunan sorun

`Relocator.direct(image_rel)` — AeroVIS için özellikle yazılmış olan, arama
yapmadan doğrudan birleştiren yol — yapılandırılmış kökün altında **47 921
karenin hepsinde ıskalıyordu**. Kareler yine de bulunuyordu, ama hasat anındaki
mutlak yolun son parçalarını eşleyen yedek yol üzerinden.

Bu neden kırılgan: **117 dizinin 91'inde `0000001.jpg` adlı bir dosya var.**
Kayıtlı mutlak yol yerel ağaçla ortak bir kuyruk paylaşmayı bıraktığı anda
(başka bir mirror, arşivin bir klasör farklı açılması) geriye 91 dosyayla
eşleşen bir isim kalır — ve `Relocator` beraberliği tahmin etmek yerine
reddeder. O durumda havuzun tamamı `no_image` okurdu.

### Yapılan düzeltme

`resolve_images_root` artık `image_rel`'i her şeyden **önce** yanıtlıyor
(`_root_of_archive_paths`). Arşive göreli bir yol tartılacak kanıt değildir —
bir kök ya ona eklenir ya eklenmez:

* her prob aynı fikirde olmalı (4/5 yanıtlayan kök, arşivin kökü değildir);
* iki kök birden eklenirse (ağaç iki kez açılmış) soru mevcut sıralamaya geri
  verilir, tahmin yapılmaz;
* `image_rel` taşımayan havuzlar hiç etkilenmez, eski yoldan devam eder.

Sonuç: kök `.../AeroVIS/sequences` olarak isimlendiriliyor, `direct` her kareyi
bir `join` ile yanıtlıyor, isim indeksi hiç kurulmuyor.

### Testler (`tests/test_pool_reader.py::AeroVISPathsTest`, 8 test)

Fikstür yukarıdaki gerçek düzenin birebir kopyası. Kanıtlananlar:

| test | ne gösteriyor |
|---|---|
| `..._name_that_repeats_across_sequences...` | isim çakışması varsayım değil, fikstürde de gerçek |
| `..._named_from_the_archive_relative_path` | yapılandırılmış kök hiçbir şeyi çözmüyor; `sequences` bulunuyor |
| `..._when_the_harvest_staged_it_elsewhere` | hasat yolu tamamen farklıyken de aynı cevap |
| `..._both_paths_resolve_and_the_masks_belong_to_the_frames` | **6/6 kare + 6/6 maske deposu**, hiçbiri yanlış diziye işaret etmiyor, maskeler decode ediliyor ve kayıttaki shape ile birebir |
| `..._means_no_frame_is_matched_by_name` | `found_by_name = 0`, `ambiguous = 0`, `misses = 0` |
| `..._still_reported_as_missing` | eksik indirme hâlâ eksik olarak raporlanıyor, arama başka bir şey bulmuyor |
| `..._two_copies_of_the_tree_hand_the_question_back` | ağaç iki kez açılmışsa tahmin yok; kareler yine doğru ağaçtan geliyor |
| `..._without_image_rel_still_reads_the_recorded_paths` | diğer havuzlar bozulmadı |

Ayrıca notebook'ların hazırlık tablosu (`READY`) düzeltildi: eskiden yalnızca
`image`'i probe ediyordu, yani okuyucunun *tercih ettiği* yol tamamen
ıskalarken tablo "bulundu" diyordu. Artık `_read_frame` ne yapıyorsa onu probe
ediyor ve kaç karenin `join` ile yanıtlandığını yazdırıyor.

### Bilinen açık

`src/training/aerovis.py` bu depoda **yok** — havuzları üreten modül Drive'da
`repo_patch_aerovis.py` olarak duruyor ve hasat notebook'u onu oradan yüklüyor.
Mevcut notebook'lardan hiçbiri bu modüle ihtiyaç duymuyor (hepsi zaten hasat
edilmiş `aerovis_train.zip` / `aerovis_heldout.zip` havuzlarını okuyor), o
yüzden dokunulmadı. AeroVIS'i yeniden hasat etmek gerekirse modülün depoya
alınması gerekir.

---

## 2. Havuz kökleri: HIT-UAV `no_image` sorunu

`7c451df`, `bf3e20d`, `90ff13e`.

**Kök neden:** HIT-UAV havuzu Kaggle mirror'ından hasat edilmiş
(`manifest.json`: `"source": "kaggle:pandrii000/..."`), ama koşu GitHub
arşivini indiriyor — o arşiv her kareyi **4 kez** taşıyor (`normal_json`,
`rotate_json` ve iki `JPEGImages` ağacı). İsimle eşleme beraberlikle
sonuçlanıyor, `Relocator` reddediyor, 2 866 karenin hepsi `no_image` okuyor —
tam olarak "veri hiç indirilmemiş" gibi görünüyor.

İkinci hata: `IMAGE_ROOTS` içindeki glob deseni **indirmeden önce**
çözülüyordu, bu yüzden `fetch` arşivi gerçekten `**` adlı bir klasöre
açıyordu.

**Çözüm:** `resolve_images_root` — havuzun kayıtlarını probe edip kökü diskten
okuyor; adayları (kayıtlı yolla paylaşılan kuyruk, çözülen prob sayısı,
derinlik) sırasıyla sıralıyor, beraberlikleri byte karşılaştırmasıyla ayırıyor
(`_same_frames`), ve yapılandırılmış kökten daha iyisini bulamazsa onu
değiştirmiyor. Ayrıca indirme her zaman veri setinin düz klasörüne gidiyor;
desen yalnızca okumayı daraltıyor.

**Sessiz maske eşleştirme hatası:** `decompose` tutulan bileşenleri yeniden
numaralandırıyor (`min_area=8 → [1,2,3]`, `min_area=48 → [1,2]`), yani eski bir
önbellek instance'ları **komşusunun maskesiyle** eşleştiriyordu. `gates_tag`
artık önbellek anahtarında.

**SegFly 15 007 / 5 378:** hata değil. Havuzun kendi manifest'i böyle diyor —
15 007 karenin yalnızca ~5 378'inde vehicle/truck var. Notebook artık huniyi
yazdırıyor.

---

## 3. Kontrast: asıl problem

`3c9ba02`. Kullanıcının bildirdiği hata: **zıt kontrastta takip iyi, benzer
kontrastta hedefi bırakıyor.**

**Ölçüm önce yapıldı, varsayılmadı.** HIT-UAV'ın 500 karesindeki 4 421 hedef
üzerinde sinyal/karmaşa oranı (SCR): medyan **0.91**, %53'ü 1'in *altında*,
yalnızca %15'i 3 ve üzeri. İnsanlar görünür olanlar (1.58); arabalar (0.55) ve
bisikletler (0.34) zaten karmaşanın içinde.

Bu, "termal hedef soğuk zeminde sıcaktır" varsayımının bu veri için **yanlış**
olduğunu gösterdi ve modül dokümanları buna göre düzeltildi.

**`src/training/photometric.py`** — dört ayar:

* `collapse` + `noise`: **tek başına kontrast düşürmek hiçbir şey yapmaz.** Bir
  pencereyi ortalamasına doğru ölçeklemek hedefin sinyalini *ve* zeminin
  değişimini aynı sayıya böler; oran değişmez, model aynı problemi daha karanlık
  görür. Oranı düşüren, ölçeklemenin *ardından* eklenen sensör gürültüsüdür.
  `tests/test_photometric.py` bunu iki testle kanıtlıyor.
* `invert`: polarite çevirmesi. White-hot ve black-hot aynı sahne; yalnızca
  birini görmüş model kuralın yarısını öğrenmiş. Maskeye dokunmaz. Renkli
  pencerelere **hiç uygulanmaz** (RGB'nin başka bir polaritesi yok).
* `gamma`: doğrusal olmayan esnetme — yerel kontrastı histogramı bütün olarak
  kaydırmadan değiştirir, farklı bir sensörün transfer eğrisi gibi.

**`instance_contrast`** (`image_loop.py`): hedefin içindeki ortalama ile
etrafındaki halkanın ortalaması arasındaki fark, halkanın standart sapmasına
bölünür. Normalize edilmiş tensör üzerinde hesaplanır ve affine dönüşüme
duyarsızdır (test edildi), yani ImageNet istatistiklerini raporlamaz.

**`tools/eval_instances.py`** artık IoU'yu kontrast bandına göre ayırıyor
(`<1`, `1–3`, `>3`). Bir koşu artık "zor durumda ne kadar iyiyim" sorusuna
kendi tablosuyla cevap verir — augmentasyonun zorluk yarattığına *güvenmek*
yerine.

Dürüst sonuç: agresif ayar HIT-UAV medyanını 0.85'ten 0.79'a taşıdı, kolay
kuyruğu %15'ten %10'a indirdi. Yani bu veri kümesinde augmentasyon neredeyse
etkisiz — çünkü kırılacak kestirme zaten yok. Değerini havuz belirler; koşu
kendi tablosunu okumalı.

---

## 4. Klasik CV koruma katmanı

`3a4f57a`, `82ef877`, `db4f9e7`. **`src/trackers/stabiliser.py`** (~700 satır).

Kullanıcının bildirdiği üç hata: kontrast bozulunca kutunun geniş araziye
yayılması, atlama, ve hedefi bir daha bulamama. Bunlar SAMURAI'den bağımsız,
kod tabanlı klasik yöntemlerle ele alındı:

* **Ego-hareket** — hedefin komşuluğu dışlanmış bir ızgarada seyrek
  Lucas-Kanade, robust medyan, faz korelasyonu yedeği, ve dokusu olmayan karede
  dürüst bir "ölçüm yok".
* **Makullük kapıları** — alan oranı, en-boy oranı, sıçrama mesafesi, karenin
  yüzdesi. "Ekranın %25'ini kaplayan hedef" reddedilir.
* **Durum makinesi** — TRACKING / SUSPECT / LOST / REACQUIRED. Histerezis var:
  tek kare kötü diye hedef bırakılmaz, tek kare iyi diye geri alınmaz.
* **NCC ile yeniden yakalama** — kaybedilen yerin etrafında büyüyen bir arama
  penceresi; şablon neredeyse sabitse eşleme **reddedilir** (düz şablon her
  yerde 1.0 puan alır).

**Gerçek görüntü bunları çürüttü ve düzeltildi:**

| iddia | gerçek görüntüde ne oldu | ne yapıldı |
|---|---|---|
| faz korelasyonu iyi bir yedek | güvenle sıfır döndürdü, LK 106–0 kazandı | `_supported()` — tam çözünürlükte hizalama artığı doğrulaması, kötü durumlar 9 → 0 |
| yarı çözünürlükte doğrulama yeterli | iyi tahminlerin dörtte birini attı | tam çözünürlük |
| HIT-UAV'da hedef soğuk zeminde sıcak | medyan SCR 0.91 — yanlış | `caution_below` 2.0 → 1.0, presetler yumuşatıldı |
| aynı karelerde LK sıfır verir | düz karede 159 piksel kayma bildirdi | dağılım + doku kapıları |

`edgetam_tracker.py` artık verdict'e göre davranıyor: reddedilen maske **boş**
döner — piksel uydurulmaz. Sürekli yanlış bir tracker'da koruma "çırpınıyordu";
`driven` modu bunu çözdü. 27 test.

---

## 5. SAMURAI 768 + ego-hareket

`dd76b71`. **`configs/edgetam_768_samurai.yaml`**.

* **ONNX export'ta sorun yok** — doğrulandı: export yolu SAMURAI'yi hiç
  kurmuyor, dosyada yalnızca iki yorum satırı geçiyor. Motor şekli değişmiyor,
  çünkü SAMURAI'nin dokunduğu her şey bu projenin bilerek PyTorch'ta bıraktığı
  bellek defteri (`docs/tensorrt_fp16.md`).
* **768 neden geçerli:** `_hydra_overrides.check_image_size()` eklendi. Kural
  spatial perceiver'ın 16×16 pencerelemesinden gelir: 768/16 = 48, 48 de 16'nın
  katı. 640, 896, 720, 1000 artık config okunurken reddediliyor — ilk ileri
  geçişte çökmek yerine.
* **Yayınlanmış SAMURAI'nin üstüne çıkan kısım:** SAMURAI'nin Kalman filtresi
  görüntü koordinatlarında yaşar ve kameranın kendi hareketi için terimi yoktur.
  Drone'da bu istisna değil, normal durumdur: bir derece yaw tüm sahneyi
  kaydırır, hiç hareket etmemiş hedef ivmeleniyor görünür, sabit-hız modeli o
  ivmeyi öğrenir — ve filtre bu hatayı aday maskeleri *puanlamak* için kullandığı
  için hata filtrenin içinde kalmaz. Ego-hareket artık **kontrol girdisi**
  olarak veriliyor.
* **Verimlilik:** `rescore` artık kare başına tek `.cpu()` senkronizasyonu
  yapıyor (kutular/IoU'lar/skorlar cihazda birleştiriliyor).

---

## 6. Notebook'lar

### Yeni: 34 ve 35 — pretrain kolları

Bunlar 22/27/28'in varyantı değil; **ayrı bir aşama**. Amaç: elde ne varsa tek
bir geniş temel eğitimde toplamak, sonra dar/derin koşuları onun üstünden
başlatmak.

| | notebook | ne yapıyor | neden ayrı |
|---|---|---|---|
| 34 | `34_pretrain_thermal_aerial.ipynb` | **Termal + hava** pretrain. `EPOCHS [3, 40]`, `STEPS 1000`, `PATIENCE 5`. Bütün termal havuzlar, `MIN_BOX_IOU 0.75` ile tek standarda yeniden kesiliyor (altı ayrı hasadın altı ayrı kapısına güvenmek yerine). SegFly kendi çizili haritalarıyla giriyor, ondan hasat edilen havuzla değil. `vtuav_lt` dışarıda. | Asıl hedef: **AERIAL + THERMAL**. Bu kol o hedefin temeli. |
| 35 | `35_pretrain_rgb_aerial.ipynb` | **RGB** pretrain, bilerek ayrı. AeroVIS tek başına 39 943 kare / 1 095 567 instance, **dizi bazında** ayrılmış held-out ile (18 dizi, 7 978 kare). Stage-B kollarındaki 10 000'lik AeroVIS sınırı burada 40 000'e çıkarılıyor — orada garnitür, burada ana yemek. `POLARITY_FLIP = 0.0` (polarite termal bir sensör konvansiyonu; `harden` zaten renkli pencereye uygulamaz, 0 yazmak dürüstlük içindir). `car` 400 450 ve `vehicle` 307 555 instance olduğu için 0.5 ile inceltiliyor — yoksa koşu her şeyden önce "hedef = araba" öğrenir. VisDrone **dışarıda**, çünkü AeroVIS VisDrone'u içeriyor; ikisini karıştırmak held-out dizileri başka bir anotasyon üzerinden eğitime sızdırır. | RGB, asıl hedefin yanında ikinci model. Karışması istenmiyordu. |

Numaralar 34/35 çünkü 29–32 birleştirmede karşı taraftan geldi (aşağıya bakın).

### Birleştirmeden gelen: 29–32

`4286a7f`. Yalnızca `.stamps.json` çakıştı. Benim 29/30 numaralı
notebook'larım karşı tarafın 29–32'siyle çarpıştığı için benimkiler 34/35'e
taşındı ve builder haritası güncellendi. Her iki tarafın `schedule.py`
değişikliklerinin hayatta kaldığı doğrulandı.

* **29** `29_thermal_contrast_tracking.ipynb` — 32'nin checkpoint'ini düşük
  kontrastlı temporal kliplerle sürdürür; stage B / temporal / temporal+SAMURAI
  A/B'si ve düşük kontrast alt-grup raporu.
* **30** `30_tracking_before_after_demo.ipynb` — senkron before/after MP4, IoU
  eğrisi, dropout ölçüleri. En iyi kazancın yanında **en kötü gerilemeyi** de
  üretir.
* **31** `31_aerial_thermal_tracking.ipynb` — VTUAV ST/LT + VTUAV-VIS + BIRDSAI
  kimlikleriyle temporal devam eğitimi. Anti-UAV410 varsayılan **kapalı** —
  hedef alan hava görüntüleri, Anti-UAV410 iyi bir veri ama hedef değil.
* **32** `32_aerial_thermal_stage_b_stable.ipynb` — VTUAV-LT'yi zorunlu yapar,
  hardening ekler, üç encoder LR profilini kısa pilotta eler.

### Güncellenen: 19, 20, 22, 23, 27, 28

Hepsi `tools/build_stage_b_notebooks.py`'den üretiliyor, hepsi 9 hücre, yorum
yok. Bu turda değişenler:

* `HARDER` preset: `MIN_AREA 64`, `MIN_SIDE 6`, `MAX_AREA 0.25`,
  `CONTRAST_COLLAPSE 0.25`, `POLARITY_FLIP 0.25`, `GAMMA_JITTER 0.25`,
  `SENSOR_NOISE 2.0`. `MAX_AREA 0.25` istenen "ekranın yarısını kaplayan
  kutuları at" kuralıdır — hedef small/medium, tiny ve huge değil.
* `THERMAL_WEIGHTS`: dronevehicle 0.45/0.7, vtuav_thermal ve vtuav_rgb
  **0.8**, car/truck 0.7. **SegFly ağırlıklardan çıkarıldı** — güvenilir
  olduğu için tamamı kullanılıyor. HIT-UAV da aynı gerekçeyle bozulmuyor.
* `SKIP_POOLS`'a `vtuav_lt` eklendi.
* İndirme klasörü ile okuma kökü ayrıldı (`images_for` artık 5 değer
  döndürüyor), indirme sonrası yeniden köklendirme geçişi eklendi.
* Denetim bloğu: havuz probe tablosu, veri seti hunisi, havuz başına SCR
  dağılımı — ve artık kaç karenin arşiv yoluyla `join` edildiği.

---

## 7. Şu anki durum

* `python3 -m pytest tests/ -q` → **1009 passed, 9 skipped, 217 subtests**.
* Yeni test dosyaları: `tests/test_photometric.py` (11),
  `tests/test_stabiliser.py` (27), `tests/test_pool_reader.py`'de
  `AeroVISPathsTest` (8).
* Dal: `claude/thermal-stage-b-training-43ktcl`.

### Sırada ne var

1. **34**'ü çalıştır (termal + hava pretrain) — asıl hedefin temeli.
2. Onun checkpoint'ini `REFERENCE_CHECKPOINT` olarak **22**'ye ver.
3. **35**'i ayrı runtime'da çalıştır (RGB, ikinci model).
4. Takip kolları için **32 → 29 → 30** sırası; 31 VTUAV kimlikleri hazırsa.
5. Her koşudan sonra `eval_instances`'ın **kontrast bandı tablosunu** oku —
   ortalama IoU, düşük kontrastta ne olduğunu saklayabilir.
