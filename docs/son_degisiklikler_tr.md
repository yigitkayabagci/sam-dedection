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

* `python3 -m pytest tests/ -q` → **1163 passed, 9 skipped, 379 subtests**.
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


---

## 8. `memgate` — maske ile memory bank arasındaki kapı

**İtiraz haklıydı:** guard, `propagate_in_video` kareyi *yield ettikten sonra*
çalışıyor. Yani kare çoktan memory bank'a yazılmış, encoder onu okumaya
başlamış oluyor. Guard çıktının etiketini değiştirebilir; kötü bir karenin
hedefin "hatırlanan görünümü" olmasını engelleyemez.

`memgate` tam olarak o noktada duruyor:

    JPEG -> [prefilter] -> image encoder -> memory attention (bank'ı okur)
         -> mask decoder -> [memgate] -> memory'ye yazma -> yield -> [guard]

### İki kapı, ikisi de sadece kutu aritmetiği

| ayar | neyi reddediyor | ölçüm |
|---|---|---|
| `memory_jump: 2.5` | merkezi, kabul edilen son kutunun kendi uzun kenarının 2.5 katından fazla uzağa düşen maske | 26 px hedefte 90 px sıçrama: kapı yokken sonraki 16 karenin 16'sı bank'a giriyor, varken 8'i |
| `memory_area_ratio: 2.5` | yerinde durup şişen maske (kabul edilmiş alanların koşan medyanının 2.5 katı) | kenarları x2, x3, x4 olan balon: kapısız 10/10 kare bank'a giriyor, kapılı 0/10 |

İkisi **OR**: biri tek başına kareyi reddeder — sıçrayan maske büyümemiş olsa
da, yerinde şişen maske hiç kıpırdamamış olsa da. `memory_area_ratio` bir
**alan** oranıdır, kenar oranı değil: 2.5, kenarları bir karede ~1.6 kat büyüyen
kutuda ateşlenir; kenarları ikiye katlanan kutu 4 kat alan eder, fazlasıyla
içeride. 2.0 denendi ve kenarları 1.5 kat (alan 2.25) büyümeyi de reddediyordu —
kameraya yaklaşan hedef için fazla dar bulundu.

Dürüst durumlarda bedeli yok, aynı ölçümlerde: gerçek manevra (16 px/kare'ye
hızlanma) 16/16 kalıyor, gerçek 2.6x yaklaşma 40/40 kalıyor. `memory_patience:
8` art arda sekiz reddin ardından geçmişi yeniden tohumluyor — yoksa gerçekten
yer değiştirmiş bir hedef klibin sonuna kadar dışarıda kalırdı.

### İçinde SAMURAI yok

`kf_weight: 0` → maskeyi hâlâ SAM 2'nin kendi `argmax`'ı seçiyor. Bu kapının
**kabul ettiği** bir karenin maskesi `plain`'inkiyle bit bit aynı. Kalman
filtresini okuyan kimse kalmadığı için filtre hiç çalıştırılmıyor
(`SamuraiConfig.uses_filter`): kare başına **33 µs**, SAMURAI'nin 144 µs'una
karşı. Aynı nedenle `--all-pointers` ile yeniden export gerektirmiyor.

### Yol boyunca bulunan gerçek hata

TensorRT yolunda `_reselect`, motoru `obj_ptr_all` çıktısı olmadan gördüğünde
**kareyi yargılamadan** erken dönüyordu. Kapı kayıt göremeyince `keep()` her
kareye "evet" diyor — yani sıradan export edilmiş bir motor setinde memory
kapısı sessizce hiç çalışmıyordu. `rescore` artık kontrolden önce çağrılıyor.
`tests/test_samurai.py::AcceleratedPathTest` bunu tutuyor.

### Koşu neyi yazıyor

Reddedilen kare çıktıda görünmez — değişen şey, *sonraki* karelerin bank'tan ne
okuduğudur. O yüzden kapının bir şey yapıp yapmadığının tek kanıtı
`memory.json`:

```
[memory] frame 214 refused: jumped 6.3 sizes>2.5 (iou 0.88, obj 4.10,
         kf 0.00, area x1.1, jump 6.3 sizes)
[memory] kept 11/430 frames out of the memory bank (jumped x9, area x2)
```

İlk bakılacak alan `refused`. **Sıfırsa o koşu `plain`'in kendisidir** — başka
bir isim altında.

### Nasıl koşulur

```bash
python3 tools/run_records.py --records ~/Videos/records --modes crop768 \
  --weights pool_deep --backend trt --policy plain,memgate
```

İki klasör çıkar: `<record>/crop768` ve `<record>/crop768__memgate`. İkisinde de
`tracked.mp4`, `track.png`, `track.json`, `latency.png`; ikincisinde ayrıca
`memory.json`. `SUMMARY.md` iki satırı yan yana koyar.

Kontrastı ayrı ölçmek için ikinci komut, aynı biçimde:

```bash
python3 tools/run_records.py --records ~/Videos/records --modes crop768 \
  --weights pool_deep --backend trt --policy plain,prefilter
```

`prefilter.json`'da ilk bakılacak alan `stretched`; 0 ise eşik bu görüntünün
kendi aralığının altında kalmış demektir ve o koşu da `plain`'dir.


---

## 9. Kırılmanın sebebini ayırmak: encoder mi, memory mi?

Sorulan soru buydu: düşük kontrast encoder'ı mı çökertiyor, yoksa memory'de
duran karelerin matrisleri yeni renge/pozlamaya uymadığı için mi patlıyor?
İkisi videoda **aynı** görünüyor — kutu hedeften kopup benzer bir şeye
yapışıyor — ama tedavileri zıt. Ayrım ölçülebilir, ve artık ölçülüyor.

### İki sayı

Her kare için, encoder'a giren pikseller üzerinden:

| sayı | ne demek | hangi arıza |
|---|---|---|
| **span** — karenin kullandığı gri seviye aralığı (1.–99. persentil) | gece termalde 255'in 60 seviyesi. /255 ve ImageNet normalizasyonundan sonra encoder'ın eğitildiği bandın onda biri | küçükse: **encoder kareyi okuyamıyor** |
| **drift** — bu karenin p1/p99'unun, memory bank'ın tuttuğu son 7 karenin medyanından uzaklığı | pozlama kaydı: gain değişti, güneş bulut arkasına girdi, sensör yeniden aralıklandı | büyükse: **bank başka bir pozlamayı tutuyor** |

Yavaş sürüklenme sayılmaz — bank onu takip eder, drift küçük kalır. Sayan şey
**basamak**: bank'ın içi eski pozlamada kodlanmışken kareler artık başka yerde.

### Nasıl koşulur

```bash
python3 tools/run_records.py --records ~/Videos/records --modes crop768 \
  --weights pool_deep --backend trt --policy plain,memgate \
  --photometry --tag vis3_deneme1
```

`--photometry` **hiçbir pikseli değiştirmez**; sadece ölçer, kare başına
0.23 ms. Her koşu klasörüne `photometry.json` yazar.

Sonra:

```bash
python3 tools/diagnose_break.py frame_output/vis3_deneme1/<record>/crop768_pool_deep_memgate
```

Çıktı şuna benzer:

```
   40/40 frames held, longest gap 0, 2 jumps, max share 0.9%
   span median 28.0 levels (40/40 under 70), exposure drift median 1.0, max 110.0

          frames   span- drift+  what happened
              20    27.0  110.0  jumped 6.7 widths
                    BOTH — the frame is dark and the exposure moved
```

Dört karar var ve dördü de bir sonraki adımı söyler:

* **ENCODER** (span küçük, drift yok) → `--policy prefilter`, ya da eğitimde
  fotometrik augmentasyon (`photometric.py`).
* **MEMORY** (span iyi, drift büyük) → `--policy memgate`, daha kısa bank, ya
  da memory yolunu pozlama değişimleriyle eğitmek.
* **BOTH** → ikisi de.
* **NEITHER** (pikseller değişmemiş) → sorun kontrast değil; benzer görünen
  bir distractor. Bu da bir sonuç: yanlış yeri düzeltmekten kurtarır.

### Yol boyunca bulunan ikinci gerçek hata

`--prefilter` kare başına **10 ms** harcıyordu ve bunun 7.1 ms'i sadece
`np.percentile` idi — yani ölçüm, ölçtüğü şeyi değiştiriyordu. 256 kutuluk
histogram aynı iki sayıyı bir gri seviye içinde **0.23 ms**'de veriyor; germe
işlemi de float çarpım yerine `cv2.LUT` ile bit bit aynı sonucu 3.0 ms yerine
0.23 ms'de veriyor. `--prefilter` artık ~10 ms değil ~0.86 ms.

Ayrıca `track.json`'daki `jumps` yanlış birimdeydi: hedefin kendi genişliği
`sqrt(share) * 100` ile hesaplanıyordu, oysa maske ızgarasının kenarı 768.
Eşik yedi kat küçük çıkıyor, sıradan hareket sıçrama sayılıyordu. Artık gerçek
kenar kullanılıyor ve `track.json` kare kare satır tutuyor.

### Çıktı klasörü: `--tag`

```
frame_output/vis3_deneme1/<record>/crop768_pool_deep/
frame_output/vis3_deneme1/<record>/crop768_pool_deep_memgate/
frame_output/vis3_deneme1/SUMMARY.md
```

Aynı tag ile tekrar koşarsan üstüne yazar (bir deneme = bir klasör). Sayacı sen
çevirirsin: `--tag vis3_deneme2`. **Hedef seçimi tag'lenmez** —
`frame_output/<record>/prompts.json`'da kalır, yeni tag seni tekrar seçiciye
göndermez.

## 768'de çalışmak: iki ayrı 768 var

`crop768`, kaydın içinden **768x768 doğal piksel** pencere alır ve modele 768
olarak verir. `crop512` de 512 alıp 512 verir. Yani ikisinde de yeniden
örnekleme yok: N piksellik bir hedef her ikisinde de N pikseldir. 768'in aldığı
şey **detay değil, görüş alanı** — 2.25 kat sahne, 2.25 kat token (stride 16'da
32x32 yerine 48x48). Kaynak 768'i taşımıyorsa (640x512 termalde olduğu gibi)
kırpma kareye kenetlenir, tüm kare 768'e gerilir ve o 2.25 kat maliyet
interpolasyona ödenir.

### Dağıtım tarafı

`--weights aerial_stable` 768'i reddediyordu, çünkü tabloda 512'den başka
girdi yoktu. Artık var:

* `configs/edgetam_768_aerial_stable.yaml` (PyTorch)
* `configs/edgetam_trt_768_aerial_stable.yaml` (TensorRT, kendi export dizini)

Bu, 512'de eğitilmiş ağırlıkları **bilerek** kendi boyutunun 1.5 katında
koşturur; `run_records` özeti eğitim boyutunu satırın yanına yazdığı için sayı
"eğitim işe yaradı mı" sorusuna değil, "bu ağırlıklar 768'de ne yapıyor"
sorusuna cevap verir. Motorlar şekle özgüdür: 768 için ayrı export gerekir,
komutlar TRT config'inin başlığında.

### Eğitim tarafı

32'nin `SIZE` değeri pencereyi **kaynak piksel** olarak keser. `windows_for`
doğal kırpmayı yalnız kare her iki eksende de `SIZE` kadarken alır; altında
tüm kareyi en-boy oranıyla birlikte yeniden boyutlandırır. 512'de sınır tam
termal setlerin oturduğu yerdir (640x512 → yeniden örnekleme yok); `SIZE`'ı
yükseltmek hepsini sessizce upsample'a çevirir. 32'nin 3. hücresi artık bunu
kaynak kaynak sayıp yazdırıyor — varsayılan 512'de hiçbir şey basmaz.

### Üçüncü seçenek: 768 yerine uyarlanır pencere

Hedef küçük olduğu için 768 istiyorsan, 768 bunu vermez — kırpma modları
ölçek değiştirmez, sadece görüş alanını büyütür. Küçük hedefi büyüten şey
`src/trackers/adaptive.py`: pencereyi hedefin kendi boyutunun sabit bir katına
kırpar ve **512 olarak** besler, yani 10 piksellik bir dron modele 100 piksel
gelir — 4-10 kat büyütme, 512 fiyatına. Bedeli, pencere her yer değiştirdiğinde
memory bank'in sıfırlanıp son maskeden yeniden prompt edilmesi (SAM 2 memory'yi
*girdi* koordinatlarında tutar), o yüzden pencere bir segment sınırı gibi
davranır ve nadiren oynar. Sürücüsü `tools/track_adaptive.py`.

## 1280x768 kaynak: 768 artık meşru, iki ayar eklendi

Kayıtlar **1280x768** olduğu için `crop768` gerçek bir doğal pencere alır
(kısa kenar tam 768) — yukarıdaki 640x512 uyarısı bu kayıtlar için geçerli
değil. Hedef 768'de geliştirmekse eğitim tarafında iki şey değişti.

### 1. Pencere kareyi dolduramıyorsa artık kare kalıyor

`windows_for` doğal kırpmayı kare iki eksende de `SIZE` kadarken alıyordu;
altında **tüm kareyi** alıp `size x size`'a geriyordu. 640x512 bir kare 768'e
giderken yatayda 1.2, dikeyde 1.5 geriliyordu — maske kafasına öğretilen şey
tam olarak hedefin şekli olduğu için bu ucuz bir hata değil.

Artık kırpma, karenin verebildiği **en büyük kareye** iniyor: 640x512 için 512
kare, çapa üstünde ortalanmış, sonra 768'e büyütülüyor. Aynı interpolasyon
maliyeti, sıfır deformasyon. Çapa hiçbir kareye sığmıyorsa eski tüm-kare
yedeği hâlâ devrede. **512 eğitimlerinde hiçbir şey değişmez** (640x512 zaten
512 pencereyi doğal veriyor), yani alınmış koşular etkilenmiyor.

`Sample.window` ne alındığını yazar: `window=1536` isteyip 1080'lik kaynakta
1080 alınmışsa kayıtta 1080 görünür.

### 2. `EFFECTIVE_BATCH` — hız tablosu batch'e bağlı

32'de `ACCUM` 1'e sabitti ve `--lr-scale 1` hız tablosunu sabit tutuyor. `SIZE`
768 olunca token sayısı 2.25 kat artıyor, otomatik prob daha küçük bir batch
buluyor ve **aynı hızlar daha gürültülü bir gradyanda** koşuyor — hiçbir şey
söylemeden. Yeni knob:

```python
EFFECTIVE_BATCH = 0     # 0 = kapalı, bugüne kadarki her koşu
```

512 koşusunun `run.json`'daki `batch x accum` değerini buraya yazarsan 768
koşusu farkı biriktirmeyle kapatır ve ne yaptığını basar. Kapalıyken davranış
aynen eskisi gibidir.

### 768 stage B için ayar listesi

| knob | 512 | 768 |
|---|---|---|
| `SIZE` | 512 | 768 |
| `MAX_AREA` 0.06 karşılığı | 125x125 | **188x188** |
| `EFFECTIVE_BATCH` | 0 | 512 koşusunun `batch x accum`'ı |
| `RUN` | çakışmaması için elle | checkpoint adı `SIZE`'ı taşır, çakışmaz |

3. hücrenin kaynak sayımı hangi setlerin 768'i doğal veremediğini yazar:
VTUAV / VTUAV-VIS (1920x1080) ve segfly (4000x3000) verir, 640x512 termal
setler vermez.

## 32'nin bütçesi: 30 epoch, patience 10, bir basamak daha alçak LR

Encoder kararsız olduğu için tam koşu bütçesi büyütüldü. Gerekçe `schedule.py`
docstring'inde zaten yazılıydı: OneCycle `total_steps`'i stage'in epoch
bütçesinden kuruyor, yani **bütçeyi büyütmek LR inişini yavaşlatmanın kendisi**.

| knob | eski | yeni |
|---|---|---|
| `EPOCHS` | `[2, 12]` | `[2, 30]` |
| `PATIENCE` | 0 (kapalı) | 10 |
| `LR_PROFILES` | cautious / stable / thermal | **gentle** / cautious / stable / thermal |

`gentle = (1e-5, 1e-5, 2e-6)` — trunk hızı `cautious`'ın yarısından da düşük.
Pilot dört profili de aynı split ve seed ile deneyip en düşük encoder val
loss'u seçer, non-finite olanı eler.

`PATIENCE=10` bir ayar knob'u değil emniyet ağı: erken durmak LR'yi yüksek ve
inişi yarım bırakır, o yüzden eşik bütçenin üçte biri. Gerçek bir plato
kesilir, normal dalgalanma kesilmez.

Uyarı: pilot 150 adımlık bir vekildir. Yüksek bir rate 150 adımda iyi görünüp
30 epoch'ta patlayabilir; seçileni `lr_pilot.json`'dan okuyun. Zorla düşürmek
isterseniz `RUN_LR_PILOT = False` + `LR_PROFILES["gentle"]`.
