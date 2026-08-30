# Termal düşük-kontrast tracking geliştirme raporu

**Proje:** EdgeTAM tabanlı havadan termal instance tracking  
**Tarih:** 30 Ağustos 2026  
**Kapsam:** Bu çalışma sırasında yapılan analizler, eklenen notebooklar, veri
bağlantıları, eğitim sırası, doğrulama yöntemi, çıktı dosyaları ve bilinen
sınırlamalar.

## 1. Yönetici özeti

Başlangıçtaki davranış şuydu: model hedefin şeklini, hedef ile arka planın
termal tonu belirgin biçimde farklıyken iyi çıkarıyor; hedef ve arka plan benzer
kontrasta geldiğinde hedefi bırakabiliyor, yakın bir sıcak/soğuk bölgeye veya
benzer görünümlü başka nesneye kayabiliyordu.

Bu tek bir “görüntüyü daha parlak yapma” problemi değildir. İki ayrı problem
üst üste binmektedir:

1. **Statik görünüş problemi:** Model hedefi şekil/doku yerine sıcaklık farkı
   üzerinden tanımaya yatkın olabilir.
2. **Temporal kimlik problemi:** Bir kötü karede oluşan yanlış maske video
   belleğine yazılırsa hata sonraki karelere taşınabilir. Tek-kare segmentasyon
   başarısı bu davranışı ölçmez.

Bu nedenle çözüm iki aşamalı tasarlandı:

```text
Termal statik havuzlar + maskeler
              │
              ▼
  32 — kararlı Stage B: şekil, termal görünüş,
       statik düşük-kontrast hardening + LR pilotu
              │
              ▼
  31 — Stage C: havadan video, düşük kontrast,
       gerçek kimlik ve bellek üzerinden eğitim
              │
              ▼
  Stage B / Stage C / Stage C+SAMURAI A/B testi
              │
              ▼
  Kaynak-bazlı metrikler + before/after videoları
```

Ana üretim yolu son denetimden sonra **32 → 31** olarak güncellenmiştir.
Kullanıcının 22 dosyası korunmuş referanstır. Notebook 29 ve 30,
Anti-UAV410 üzerinde ayrı bir düşük-kontrast deney/diagnostik koludur. Notebook
28 ise RGB-only kontrol modelidir; termal final modelin devamı değildir.

## 2. Problemin teknik teşhisi

### 2.1 Tek karede iyi maske, videoda iyi tracking anlamına gelmez

Notebook 22 görüntü encoder'ını ve maske üretim yolunu statik örneklerle
uyarlar. Ancak video belleğini gerçek bir sekans içinde çalıştırmaz. Dolayısıyla
şu hata zincirini doğrudan eğitemez:

```text
düşük kontrastlı kare
  → düşük object score / yanlış maske
  → yanlış bilginin belleğe yazılması
  → sonraki karelerde yanlış bölgeye dikkat
  → uzun dropout veya başka nesneye kimlik kayması
```

Bu nedenle Stage C'de model kendi önceki tahminini belleğe yazar. Ground-truth
maskesi belleğe zorla verilmez; yani teacher forcing kullanılmaz. Eğitim yolu,
deployment sırasında modelin yaşayacağı hataları görür.

### 2.2 Kontrast kısayolu

Termal veri setlerinin önemli bölümü sıcak hedef/soğuk zemin veya bunun tersi
gibi kolay örnekler içerir. Model bu durumda “araç/insan şekli” yerine “çevreden
farklı parlak blob” kısayolunu öğrenebilir.

Çözüm olarak hedefin yerel kontrastı şu şekilde ölçülür:

```text
| hedef kutusu ortalaması − çevre halka ortalaması |
---------------------------------------------------
          çevre halkanın standart sapması
```

Mutlak fark kullanıldığı için hem white-hot hem black-hot hedeflerde simetriktir.
Düşük skorlu sekanslar eğitim clip havuzunda daha sık örneklenir.

### 2.3 Distractor ve bellek kirlenmesi

Benzer kontrastlı başka bir nesne, görünüş benzerliği açısından doğru hedeften
daha çekici olabilir. SAMURAI kolu bu nedenle aynı yeni checkpoint'i farklı bir
bellek politikasıyla ölçer: hareket tutarlılığı, IoU, object score ve Kalman
bilgisini kullanarak hangi karelerin belleğe alınacağını filtreler.

Bu ayrım önemlidir:

- **Temporal checkpoint iyileşiyorsa:** ağırlık eğitimi faydalıdır.
- **Aynı checkpoint SAMURAI ile ayrıca iyileşiyorsa:** sorun yalnız görünüş
  temsili değil, bellek güncelleme politikasıdır.

### 2.4 Domain yönü

Anti-UAV410'da kamera çoğunlukla yerde, hedef havadadır. Asıl deployment ise
kameranın UAV üzerinde olduğu havadan görüntüdür. Bu yüzden Anti-UAV410 faydalı
bir düşük-kontrast testi olsa da ana Stage C veri kaynağı yapılmamıştır.
Notebook 31'de varsayılan olarak kapalıdır.

## 3. Notebook haritası

| Notebook | Rol | Çözdüğü problem | Ana çıktı |
|---|---|---|---|
| `22_thermal_deep_3_fixed.ipynb` | Korunan Stage B referansı | Termal şekil/görünüş; önceki Drive koşusunun teşhisi | `edgetam_pool_thermal_deep_512.pt` |
| `32_aerial_thermal_stage_b_stable.ipynb` | **Ana Stage B** | VTUAV-LT zorunluluğu, kontrast hardening, kararlı discriminative LR ve LR pilotu | `edgetam_pool_aerial_thermal_stable_512.pt` |
| `29_thermal_contrast_tracking.ipynb` | Opsiyonel Stage C deney kolu | Anti-UAV410 üzerinde düşük-kontrast bellek/kimlik kayması | `edgetam_thermal_contrast_tracking_512.pt` |
| `30_tracking_before_after_demo.ipynb` | 29'un diagnostik videosu | 29 gerçekten çalışıyor mu; en iyi kazanç ve en kötü gerileme | Senkron MP4, IoU grafiği, JSON |
| `31_aerial_thermal_tracking.ipynb` | **Ana Stage C** | Havadan termal videoda düşük kontrast, kimlik sürekliliği ve bellek | `edgetam_aerial_thermal_tracking_512.pt` |
| `28_rgb_deep_aerovis.ipynb` | Ayrı kontrol | RGB-only encoder karşılaştırması | RGB checkpoint; termal zincire girmez |

## 4. Notebook 22 — Stage B: şekil ve termal görünüş

### 4.1 Ne yapar?

Kullanıcının sağladığı `22_thermal_deep_3_fixed.ipynb` dosyası bu çalışma
sırasında değiştirilmemiştir. Ana görevi statik termal görüntülerde instance
maskesi ve termal görünüş öğrenmektir.

Başlıca kaynakları:

- DroneVehicle termal teacher-mask havuzu,
- VTUAV short-term termal teacher-mask havuzu,
- VTUAV long-term termal teacher-mask havuzu mevcutsa,
- HIT-UAV,
- Kaggle aerial thermal havuzu,
- SegFly termal semantic maskeleri,
- Kust4K çizilmiş maskeleriyle held-out değerlendirme.

Önemli ayarlar:

- Görüntü boyutu: `512×512`
- Uzun eğitim bütçesi: `[2, 24]` epoch aşamaları
- Her epoch: `800` step
- Patience: `4`
- Teacher maskesi için ikinci kalite kesimi: `box IoU ≥ 0.80`
- `LR_HEAD=0` head'in kapalı olduğu anlamına gelmez; yalnız encoder aşamasındaki
  explicit override'ın kapalı olduğu anlamına gelir. Head-only aşaması yine
  eğitilir, encoder aşamasında head tablo değerini kullanır.
- `LR_NECK=1e-4` ve `LR_TRUNK=1e-4` absolute override'dır. Otomatik
  `LR_SCALE` bunlara uygulanmaz; fakat head'in tablo LR'lerini büyütür.

### 4.2 Ne yapmaz?

- Video belleğini çalıştırmaz.
- Kimlik kaymasını ölçmez.
- Uzun dropout episode'larını doğrudan optimize etmez.
- İlk prompt/detector ihtiyacını ortadan kaldıran bir detector değildir.

Bu nedenle 22 tek başına final tracking çözümü değildir. Ayrıca Drive'daki
önceki koşu encoder açıldığında kararsız olduğu için artık ana 31 başlangıcı
değil, 32'nin karşılaştırma referansıdır.

### 4.3 Çıktı

```text
/content/drive/MyDrive/edgetam-stage-b/thermal_deep/
  edgetam_pool_thermal_deep_512.pt
```

22 çalışırken basılan veri tablosunda `vtuav_lt_thermal` satırının sıfırdan
büyük olduğunun kontrol edilmesi gerekir. Bu pool sabit notebook'ta zorunlu
eşik listesinde değildir; erişilemezse notebook başka zorunlu kaynaklar varsa
devam edebilir. LT satırı `unusable` veya `0` ise eğitim durdurulmalı ve Drive
shortcut/arşiv erişimi düzeltilmelidir.

## 5. Notebook 29 — Anti-UAV410 düşük-kontrast temporal deney kolu

### 5.1 Amaç

Notebook 29, tercihen 32 (yoksa 22) checkpoint'ini Anti-UAV410 kısa video clip'leri üzerinde devam
eğitir. Ana deployment modeli olmak zorunda değildir; kontrast ve bellek
problemini kontrollü bir benchmark üzerinde incelemek için oluşturulmuştur.

### 5.2 Uygulanan düzeltmeler

- Gerçek sekans kontrastı ölçülür.
- Eğitim sekanslarının en düşük kontrastlı `%40` bölümü iki kez örneklenir.
- Clip uzunluğu `8`, temporal stride `2`dir.
- Eğitimde kontrast değişimi kareden kareye sıçramaz; clip başı ve sonu arasında
  düzgün interpolasyon yapılır.
- Global kontrast `0.35–0.90` aralığına düşürülür.
- Hedef bölgesi çevre halka tonuna `0.15–0.65` katsayısıyla yaklaştırılır.
- Küçük brightness shift, sensör gürültüsü ve blur uygulanır.
- Validation ve test augmentasyonsuz gerçek görüntülerde yapılır.
- SAM2.1 teacher ile kabul edilen seyrek maskeler kullanılır; maskesi olmayan
  kareler box-projection ve object-score ile eğitime devam eder.

### 5.3 Eğitim aşamaları

- `head`: 2 epoch, `lr=1e-4`
- `encoder`: 3 epoch; head/neck `5e-5`, trunk `1e-5`
- Her epoch `500` step
- Batch boyutu GPU üzerinde gerçek forward/backward ile otomatik ölçülür.

### 5.4 Değerlendirme kolları

1. `stage_b`: seçilen Stage B checkpoint'i + standart FIFO bellek
2. `temporal`: yeni checkpoint + standart FIFO bellek
3. `temporal+samurai`: aynı yeni checkpoint + güvenilir bellek kapısı

Seçim validation State Accuracy ile yapılır. Test split'ine yalnız Stage B ve
validation'da seçilmiş final kolu gider.

### 5.5 Çıktılar

```text
/content/drive/MyDrive/edgetam-stage-c/thermal_contrast_tracking/
  edgetam_thermal_contrast_tracking_512.pt
  contrast_tracking_log.json
  eval_val_stage_b.json
  eval_val_temporal.json
  eval_val_temporal_samurai.json
  eval_test_*.json
  edgetam_thermal_contrast_tracking_512.yaml
```

YAML, validation'da `temporal+samurai` seçildiyse SAMURAI ayarlarını içerir;
yalnız `temporal` seçildiyse standart bellek politikasını korur.

## 6. Notebook 30 — before/after kanıt notebook'u

Notebook 30 eğitim yapmaz. 22 ve 29 checkpoint'lerini aynı kareler, aynı ilk
ground-truth kutusu ve aynı sabit crop üzerinde çalıştırır.

Cherry-pick riskini azaltmak için iki vaka üretir:

1. Validation düşük-kontrast `%25` grubu içindeki en yüksek State Accuracy
   kazancı,
2. Tüm ortak validation sekanslarındaki en kötü gerileme.

Üretilenler:

- Sol taraf `BEFORE — Stage B`, sağ taraf `AFTER — Stage C` senkron MP4,
- Ground-truth, tahmin kutuları ve tahmin maskeleri,
- Kare bazında IoU,
- 20-kare hareketli ortalama IoU grafiği,
- State Accuracy, Success AUC, dropout episode sayısı ve en uzun dropout,
- JSON karar raporu.

Çıktı klasörü:

```text
/content/drive/MyDrive/edgetam-stage-c/thermal_contrast_tracking/
  before_after_demo/
```

Notebook 30 yalnız `32 → 29` deney kolunu görselleştirir. Ana `32 → 31`
modelinin videolarını görmek için 31'in kendi video hücresi kullanılmalıdır.

## 7. Notebook 31 — ana havadan termal Stage C

### 7.1 Neden 31 oluşturuldu?

Anti-UAV410 düşük-kontrast problemi için değerlidir ancak kamera yönü asıl
kullanımdan farklıdır. Kullanıcının amacı yalnız drone detect etmek değil,
havadan termal görüntülerde prompt verilen herhangi bir instance'ı takip
etmektir. Bu nedenle ana Stage C, havadan çekilmiş termal video kaynaklarıyla
yeniden tasarlanmıştır.

### 7.2 Kullanılan video kaynakları

#### VTUAV tracking RGB-T sürümü

Varsayılan arşivler:

```text
train_ST_001.zip
train_ST_008.zip
train_LT_001.zip
train_LT_002.zip
train_LT_003.zip
train_LT_004.zip
```

- ST001: animal/bike/bus çeşitliliği,
- ST008: ağırlıklı pedestrian,
- Dört LT parçası: kaybolma ve yeniden görünme davranışı.

Yalnız annotation bulunan termal kareler çıkarılır. Target gerçekten görüntü
dışındaysa satır silinmez; `exist=False` olarak object-score eğitimine katılır.

#### VTUAV-VIS instance-mask video sürümü

Varsayılan yalnız `train_001` kullanılır:

- Yaklaşık `9.1 GiB` indirme,
- Yaklaşık `875` maskeli kare,
- `--frames masked` ile yalnız maskesi olan kareler ve gerekli metadata
  çıkarılır.

Maskeler yaklaşık her 30 kaynak karede bir bulunduğu için bu kol uzun-zaman
maskeli denetim verir. Kutu, tüm-frame `txt` satırıyla yanlış hizalanmaması için
doğrudan maskenin sınırlarından çıkarılır.

`train_003` pedestrian çeşitliliğini artırmak için daha sonra eklenebilir;
ilk smoke test'te üç büyük arşivi birden indirmek önerilmez.

#### BIRDSAI

- UAV üzerindeki gece TIR kamera,
- İnsan ve hayvan track ID'leri,
- Aynı uçuş içindeki bütün track ID'leri aynı train/val/test bölümünde tutulur,
- `noise=1` interpolasyon satırları yanlış negatif yapılmaz; track run'ını
  böler,
- Occlusion satırları varsayılan olarak korunur.

#### Anti-UAV410

`USE_ANTIUAV410=False` varsayılandır. Açılırsa ağırlıklar normalize edildiğinde
yaklaşık `%9` düzenleyici pay alır; ana aerial kaynakları outvote edemez.

### 7.3 Kaynak ağırlıkları

Anti-UAV kapalıyken 12.000 eğitim clip'i şu oranlarda oluşturulur:

| Kaynak | Pay |
|---|---:|
| VTUAV tracking ST/LT | `%40` |
| VTUAV-VIS gerçek maskeli video | `%30` |
| BIRDSAI | `%30` |

Büyük bir veri setinin diğer kaynakları yok etmemesi için her kaynak önce kendi
içinde clip üretir, ardından ağırlıklı örnekleme yapılır.

### 7.4 Maske denetimi

31 üç farklı denetim seviyesini birlikte kullanır:

1. **Gerçek çizilmiş maskeler:** VTUAV-VIS.
2. **Kabul edilmiş teacher maskeleri:** notebook 17/25'in
   `vtuav_thermal` ve `vtuav_lt_thermal` havuzları.
3. **Maskesiz kareler:** box-projection + object-score.

Teacher pool kayıtları `sekans adı + frame stem` üzerinden video karelerine
bağlanır. Maske yalnız ihtiyaç anında RLE/PNG'den açılır; tüm maskeler RAM'e
yüklenmez. Stage B ile tutarlı olarak `box IoU ≥ 0.80` ikinci kalite kesimi
uygulanır.

Notebook eşleşmiş teacher maskesi sayısı `1000`den azsa eğitimi durdurur. Böylece
pool bağlantısı bozulduğunda sessizce box-only eğitime geçilmez.

### 7.5 Düşük-kontrast eğitimi

- Her kaynak kendi içinde kontrasta göre sıralanır.
- En düşük kontrastlı `%40` sekans daha sık örneklenir.
- Clip uzunluğu `8`dir.
- Global kontrast, hedef-kontrastı, brightness, sensör gürültüsü ve blur clip
  boyunca zamanda yumuşak değiştirilir.
- Validation/test gerçek, augmentasyonsuz görüntülerdir.
- Geometri, maskeler ve sınıf kimliği değişmez.

Burada amaç görüntüyü inference sırasında yapay olarak “daha kontrastlı” yapmak
değil, modelin kontrast çöktüğünde de şekil ve zamansal kimlik kullanmasını
öğretmektir.

### 7.6 Bellek ve öğrenme oranları

Teacher forcing kullanılmaz. Model kendi tahminini sonraki karelerin belleğine
yazar.

Bellek attention/encoder koordinatları sabit tutulur; aksi halde az veriyle
temel video geometrisinin bozulma riski vardır. Eğitilen kollar:

- `head`: 2 epoch, `5e-5`
- `encoder`: 4 epoch; head/neck `2e-5`, trunk `5e-6`
- Her epoch `500` step

### 7.7 Değerlendirme ve video

31 aşağıdaki kolları aynı held-out uçuşlarda karşılaştırır:

1. Stage B,
2. Aerial temporal checkpoint,
3. Aerial temporal checkpoint + SAMURAI.

Sonuçlar kaynak bazında ayrı basılır. Böylece yalnız VTUAV'da yükselip
BIRDSAI'da düşen bir model aggregate ortalamanın arkasına saklanamaz.

Her kaynak için validation'dan:

- düşük-kontrast grubundaki en iyi kazanç,
- en kötü gerileme

seçilerek senkron before/after video oluşturulur.

### 7.8 Çıktılar

```text
/content/drive/MyDrive/edgetam-stage-c/aerial_thermal_tracking/
  edgetam_aerial_thermal_tracking_512.pt
  edgetam_aerial_thermal_tracking_512.yaml
  aerial_tracking_log.json
  before_after_demo/
    *_before_after.mp4
    video_report.json
```

## 8. Eklenen kod altyapısı

### `src/training/aerial_video.py`

Yeni modül aşağıdaki işleri yapar:

- `vtuav_sequences`: VTUAV termal karelerini kutularla hizalar; absent satırları
  korur.
- `vtuav_vis_sequences`: sparse VIS maskelerini aynı stem'li termal karelerle
  eşler ve kutuyu maskeden çıkarır.
- `birdsai_sequences`: MOT track ID'lerini temiz, kesintisiz run'lara ayırır.
- `split_flights`: aynı fiziksel uçuşun train/val/test arasında sızmasını önler.
- `weighted_clip_sample`: kaynak ağırlıklarını deterministik uygular.
- `ImageMaskStore`: çizilmiş PNG maskelerini lazy okur.
- `PoolMaskStore`: teacher RLE maskelerini lazy açar.
- `pool_sequence_stores`: statik teacher pool kayıtlarını video frame
  indeksleriyle birleştirir ve `min_box_iou` filtresi uygular.

### `src/training/__init__.py`

Yeni aerial video adaptörleri eğitim notebooklarından tek import noktasıyla
kullanılacak şekilde dışa aktarıldı.

### Notebook builder'ları

Notebook JSON'unu elle düzenlemek yerine tekrar üretilebilir builder dosyaları
eklendi:

- `tools/build_contrast_tracking_notebook.py`
- `tools/build_tracking_demo_notebook.py`
- `tools/build_aerial_tracking_notebook.py`
- `tools/build_stage_b_stable_notebook.py`

### Testler

`tests/test_aerial_video.py` şu kontratları doğrular:

- VTUAV strided kare/kutu hizalaması,
- absent satırların korunması,
- BIRDSAI noise satırının yanlış negatif yapılmaması,
- aynı uçuş track'lerinin tek split'te kalması,
- kaynak ağırlıklı clip sayıları,
- VTUAV-VIS kutularının maskeden çıkarılması,
- teacher pool maskelerinin sekans/frame ile eşleşmesi,
- stricter `box IoU` filtresi.

Son kontrolde `803` testlik CPU paketi çalıştırıldı. Sandbox koşusunda `800`
test geçti; yalnız eksik geçici `gdown` ve localhost test sunucusu izni nedeniyle
üç ortam hatası çıktı. Bağımlılık geçici `/private/tmp` dizinine kurularak ve
localhost izni verilerek bu üç test ayrıca tekrarlandı ve geçti. Ek olarak yeni
notebook/fail-fast kapsamındaki `48` hedefli test geçti. Notebook 29/30/31/32'nin
sırasıyla `28/8/14/10` kod hücresinde her isim kullanılmadan önce tanımlıdır.

## 9. Google Drive doğrulaması

Google Drive bağlantısı üzerinden salt-okunur inceleme yapıldı.

### VTUAV long-term teacher pool

```text
MyDrive/edgetam-pool/vtuav_lt_thermal/
  vtuav_lt_thermal.zip       ≈ 35.2 MB
  pool_index.jsonl           ≈ 7.4 MB
```

İndeks içeriği:

- `18.123` frame kaydı,
- `43` sekans,
- `15.777` kabul edilmiş SAM3 maskesi,
- Harvest kabul oranı: `%87.1`.

Kabul edilen maskelerin sınıf dağılımı:

| Sınıf | Maske |
|---|---:|
| pedestrian | 8.418 |
| car | 4.509 |
| tricycle | 1.539 |
| bus | 1.040 |
| truck | 253 |
| elebike | 18 |

Bu sayılar harvest kabulüdür. 22 ve 31'de uygulanan `box IoU ≥ 0.80` kesimi
sonrasında kullanılan sayı daha düşük olabilir; notebook gerçek son sayıyı
çalışma sırasında basar.

### VTUAV short-term pool

`vtuav_thermal` manifest'i 11 short-term training arşivinin tamamını ve
SAM2.1 teacher kullanımını doğruladı. Pool sekans bazlı `.zip + .jsonl`
çiftlerinden oluşuyor.

### VTUAV kaynak arşivleri

`MyDrive/VTUAV` altında ST ve LT zip'lerine Google Drive shortcut'ları mevcut.
31, Colab Drive mount shortcut'ı gerçek dosya gibi çözebilirse doğrudan oradan
okur. Çözemezse resmî Drive kimliklerinden seçili arşivleri sırayla indirip
yalnız `tracked_ir` üyelerini çıkarır.

Seçili altı tracking zip'inin toplam sıkıştırılmış boyutu yaklaşık `95 GiB`dır.
Shortcut fallback'i tetiklenirse ağ ve geçici disk bütçesi buna göre
planlanmalıdır.

### VTUAV-VIS resmî arşivleri

- `train_001.zip`: yaklaşık `9.1 GiB`
- `train_002.zip`: yaklaşık `16.1 GiB`
- `train_003.zip`: yaklaşık `17.9 GiB`

Varsayılan yalnız `train_001`dir.

## 10. Önerilen Colab çalışma sırası

### 10.1 Ön hazırlık

1. Bu repo değişikliklerini kullandığınız GitHub branch'ine commit/push edin.
   Yeni notebooklar başlangıçta repo branch'ini klonladığı için yalnız yerel
   dosyanın Colab'a yüklenmesi yeterli değildir.
2. Google Drive'da aşağıdaki yolları doğrulayın:

   ```text
   MyDrive/edgetam-pool/vtuav_thermal
   MyDrive/edgetam-pool/vtuav_lt_thermal
   MyDrive/VTUAV
   ```

3. Colab GPU runtime açın. Büyük zip ve video eğitimi için yüksek diskli/A100
   runtime tercih edilir.

### 10.2 Ana üretim yolu

#### Adım 1 — Notebook 32 (önerilen Stage B)

`notebooks/32_aerial_thermal_stage_b_stable.ipynb` dosyasını Run all çalıştırın.

Kontrol edin:

- Zorunlu pool tablosunda bütün required kaynaklar `ok` olmalı.
- `vtuav_lt_thermal` en az `5.000` kabul edilmiş instance vermeli.
- Test pencereleri boş olmamalı.
- `lr_pilot.json` içinde seçilen profil finite olmalı.
- `run.json` içinde `nonfinite_*` durma nedeni olmamalı.
- Tercihen en az bir encoder epoch'unda `saved=true` olmalı. Yoksa checkpoint
  güvenli biçimde head-only en iyi ağırlıkta kalmıştır; statik test tablosu
  artmadan Stage C'ye geçilmemelidir.
- Final `.pt` Drive'a kopyalanmış olmalı.

22'nin checkpoint'i silinmez ve karşılaştırma için saklanır. 31 önce 32'nin yeni
checkpoint'ini arar; bulamazsa eski 22 checkpoint'ine fallback yapar ve uyarı basar.

#### Adım 2 — Notebook 31

`notebooks/31_aerial_thermal_tracking.ipynb` dosyasını Run all çalıştırın.

İlk koşuda varsayılanları koruyun:

```python
USE_VTUAV_VIS = True
VTUAV_VIS_PARTS = ["train_001"]
USE_ANTIUAV410 = False
TEACHER_MIN_BOX_IOU = 0.80
STOP_AFTER_PRECHECK = False
```

Veri hazırlığı sonunda şu kontrolleri yapın:

- Train/val/test bütün kaynaklarda nonzero olmalı.
- `teacher_mask_frames >= 1000` assertion'ı geçmeli.
- `mask supervision` tablosunda VTUAV ve VTUAV-VIS nonzero olmalı.
- Clip tablosu yaklaşık `%40/%30/%30` kaynak dağılımını göstermeli.
- `stage_c_precheck.json` içindeki State Accuracy, lost-rate ve longest-dropout
  satırlarını kaynak bazında okuyun. Yalnız audit yapmak istiyorsanız
  `STOP_AFTER_PRECHECK=True` ayarlayın; notebook bu JSON'u yazdıktan sonra durur.

Eğitim sonunda:

- Validation'da seçilen kolu okuyun.
- Kaynak-bazlı VTUAV, VTUAV-VIS ve BIRDSAI satırlarını ayrı değerlendirin.
- Held-out test gerilerse modeli deploy etmeyin.
- En iyi ve en kötü before/after videolarının ikisini de izleyin.

### 10.3 Opsiyonel Anti-UAV deney yolu

```text
32 → 29 → 30
```

Bu kol şu soruya cevap verir: “Düşük-kontrast temporal eğitim ve SAMURAI,
Anti-UAV410 benchmark'ında dropout/kimlik kaymasını azaltıyor mu?”

29'un checkpoint'ini doğrudan 31'in başlangıcı yapmayın. Ground-camera domain
shift'i nedeniyle doğru karşılaştırma, iki ayrı 32 tabanlı branch'tir:

```text
32 ──→ 29  (Anti-UAV ablation)
 │
 └──→ 31  (aerial final)
```

### 10.4 Notebook 28

28, RGB-only AeroVIS kontrolüdür. 22 checkpoint'iyle devam etmez ve final termal
model için gerekli değildir. Yalnız “aynı eğitim bütçesinde RGB kolu ne yapıyor?”
sorusunu cevaplamak istiyorsanız ayrı çalıştırın.

## 11. Sonuçları nasıl okumalı?

### State Accuracy

Hedef görünürken IoU'yu, hedef yokken modelin gerçekten “yok” demesini ödüllendirir.
Model sürekli bir kutu üreterek skor satın alamaz.

### Success AUC

Yalnız görünür karelerde overlap başarısını gösterir. Pratikte görünür karelerin
ortalama IoU'sudur.

### Dropout episode / lost frames / longest

Bu proje için en önemli sinyallerdendir. Ortalama IoU, modelin bir kere kayıp
200 kare boyunca dönememesini saklayabilir. `longest` bunun doğrudan ölçüsüdür.

### Karar tablosu

| Gözlem | Yorum | Sonraki hareket |
|---|---|---|
| SA ve AUC artıyor, longest düşüyor | Hem maske hem kimlik sürekliliği iyileşti | Final aday |
| AUC artıyor, longest yüksek kalıyor | Şekil daha iyi; bellek/kimlik hâlâ zayıf | Daha fazla gerçek video, bellek kapısı |
| Temporal iyi, SAMURAI daha iyi | Güvenilir bellek seçimi önemli | SAMURAI YAML ile deploy |
| Yalnız VTUAV iyileşiyor, BIRDSAI düşüyor | Kaynak/sınıf overfit | BIRDSAI ağırlığını artır |
| Genel artıyor, düşük-kontrast grup artmıyor | Augmentation gerçek kamera dağılımını yakalamıyor | Kamera histogramlarını ölç, ayarları yeniden kalibre et |
| Test geriliyor | Validation seçimi genelleşmedi | Deploy etme |

## 12. AI tabanlı ve AI tabanlı olmayan çözümler

### Uygulanan AI tabanlı parçalar

- SAM2.1/SAM3 teacher ile kutudan pseudo-instance maskesi,
- EdgeTAM video belleği altında temporal fine-tuning,
- Gerçek VTUAV-VIS maskeleriyle doğrudan maske loss,
- SAMURAI hareket/güven tabanlı bellek seçimi.

### Uygulanan klasik/veri tabanlı parçalar

- Yerel target-background kontrast ölçümü,
- Düşük-kontrast sekans oversampling,
- Zamanda yumuşak contrast/brightness dönüşümü,
- Hedefi çevre termal tonuna yaklaştırma,
- Sensör gürültüsü ve blur,
- Kaynak ağırlıklı clip örnekleme,
- Uçuş-bazlı veri split'i,
- Box-IoU kalite filtresi.

### Bilerek inference yoluna eklenmeyenler

CLAHE, histogram equalization veya kare-bazlı otomatik kontrast şu anda final
tracker girişine zorunlu bir preprocessing olarak eklenmedi. Kare bazında
bağımsız uygulanmaları temporal flicker yaratabilir ve encoder'ın eğitim
dağılımını değiştirebilir.

Bunlar ileride ayrı A/B kolları olarak denenebilir:

- Percentile normalization + zamansal EMA,
- Temporal olarak sabitlenmiş CLAHE parametreleri,
- Raw / normalized / CLAHE üç-kollu validation,
- Ayrı öğrenilmiş thermal enhancement ön-ağı.

Ancak aynı checkpoint üzerinde held-out video testi yapılmadan deployment'a
eklenmemelidir.

## 13. Bilinen sınırlamalar ve riskler

1. **Tam GPU eğitimi bu yerel ortamda yapılmadı.** Kod, veri metadata'sı,
   maskeler, notebook bağımlılıkları ve testler doğrulandı; gerçek kalite kararı
   Colab çıktılarından verilecektir.
2. **VTUAV-VIS sparse'tır.** Maskeli kareler yaklaşık 30 kaynak kare aralıklıdır.
   Dense kısa-zaman hareketi BIRDSAI ve tracking split tamamlar.
3. **Teacher maskesi insan maskesi değildir.** Bu nedenle 0.80 box-IoU filtresi
   ve en az 1000 eşleşme assertion'ı kondu. VTUAV-VIS gerçek maskeleri ayrı bir
   güvenilir kol sağlar.
4. **Stage B görünüş sızıntısı:** 22 bazı VTUAV tracking karelerini görmüş
   olabilir. 31 split'i Stage C içinde uçuş sızıntısını önler, ancak VTUAV
   skorunu “tamamen yeni görünüş” olarak yorumlamamak gerekir. BIRDSAI ve 31'de
   ilk kez kullanılan VTUAV-VIS kaynak-bazlı satırları genelleme açısından
   ayrıca okunmalıdır.
5. **`train_001` sınıf dağılımı sınırlıdır.** İlk doğrulamadan sonra özellikle
   pedestrian için `train_003` eklenebilir.
6. **Tracker detector değildir.** İlk kutu/point prompt operatörden veya ayrı bir
   detector'dan gelmelidir.
7. **Yeni checkpoint yeni engine ister.** TensorRT/INT8 kullanılıyorsa yeni
   checkpoint'ten yeniden export ve kalibrasyon yapılmalıdır.
8. **Drive shortcut riski:** Colab mount shortcut'ları çözmezse 31 resmî
   arşivleri yeniden indirir; yaklaşık 95 GiB tracking indirmesi oluşabilir.

## 14. Değiştirilen veya eklenen dosyalar

Yeni dosyalar:

- `notebooks/29_thermal_contrast_tracking.ipynb`
- `notebooks/30_tracking_before_after_demo.ipynb`
- `notebooks/31_aerial_thermal_tracking.ipynb`
- `notebooks/32_aerial_thermal_stage_b_stable.ipynb`
- `src/training/aerial_video.py`
- `tests/test_aerial_video.py`
- `tools/build_contrast_tracking_notebook.py`
- `tools/build_tracking_demo_notebook.py`
- `tools/build_aerial_tracking_notebook.py`
- `tools/build_stage_b_stable_notebook.py`

Güncellenen dosyalar:

- `src/training/__init__.py`
- `src/training/schedule.py`
- `tools/train_encoder.py`
- `tests/test_schedule.py`
- `notebooks/README.md`
- `notebooks/.stamps.json`

Korunan dosya:

- `22_thermal_deep_3_fixed.ipynb` değiştirilmedi.

## 15. Nihai öneri

İlk gerçek koşu şu sırada yapılmalıdır:

```text
1. notebooks/32_aerial_thermal_stage_b_stable.ipynb
2. 31_aerial_thermal_tracking.ipynb
3. stage_c_precheck.json ve aerial_tracking_log.json incelemesi
4. kaynak-bazlı before/after videoları
5. yalnız held-out test iyileşiyorsa deploy/export
```

29/30 ayrı bir benchmark kolu, 28 ise RGB kontrolüdür. Final termal ağırlık
zincirine eklenmemelidir.

## 16. Teknik kaynaklar

- [VTUAV proje sayfası](https://zhang-pengyu.github.io/DUT-VTUAV/)
- [VTUAV GitHub](https://github.com/zhang-pengyu/DUT-VTUAV)
- [BIRDSAI makalesi](https://openaccess.thecvf.com/content_WACV_2020/html/Bondi_BIRDSAI_A_Dataset_for_Detection_and_Tracking_in_Aerial_Thermal_Infrared_Videos_WACV_2020_paper.html)
- [Anti-UAV410](https://github.com/HwangBo94/Anti-UAV410)
- [SAM 2](https://arxiv.org/abs/2408.00714)
- [Resmî SAM2.1 MOSE fine-tune yapılandırması](https://github.com/facebookresearch/sam2/blob/main/sam2/configs/sam2.1_training/sam2.1_hiera_b%2B_MOSE_finetune.yaml)
- [Resmî SAM2 video eğitim modeli](https://github.com/facebookresearch/sam2/blob/main/training/model/sam2.py)
- [EdgeTAM resmî deposu](https://github.com/facebookresearch/EdgeTAM)
- [PyTorch OneCycleLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR)
- [SAMURAI](https://arxiv.org/abs/2411.11922)
- [DAM4SAM](https://openaccess.thecvf.com/content/CVPR2025/html/Videnovic_A_Distractor-Aware_Memory_for_Visual_Object_Tracking_with_SAM2_CVPR_2025_paper.html)
- [HIT-UAV](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset)
- [DroneVehicle](https://github.com/VisDrone/DroneVehicle)

## 17. Ek denetim — Stage B LR/epoch ve Stage C gerekliliği

Bu bölüm, Drive'daki gerçek eğitim kayıtları okunduktan ve notebook zinciri
yeniden denetlendikten sonra eklenmiştir.

### 17.1 Drive'daki `thermal_deep` koşusunun gerçek sonucu

Drive klasörü:

```text
MyDrive/edgetam-stage-b/thermal_deep/
  run.json
  edgetam_pool_thermal_deep_512.pt
  verdict.json
  score_stock_box.json
  score_thermal_deep_box.json
```

`run.json` şu eğriyi kaydetmiştir:

| Aşama | Epoch | Validation loss | Kaydedildi mi? |
|---|---:|---:|---|
| head | 0 | `0.2723096` | evet |
| encoder | 0 | `21.8418603` | hayır |
| encoder | 1 | `NaN` | hayır |
| encoder | 2 | `NaN` | hayır |

Koşu ayarları `batch=128`, `steps_per_epoch=400`, `lr_scale=4.0`, yaklaşık
`34.972` train frame ve `231.289` train instance idi. Bu kayıt mevcut sabit
22'nin `[2,24] × 800` bütçesinden önceki `[1,3] × 400` koşusudur; dolayısıyla
iki notebook bire bir aynı değildir. Ancak yüksek LR yapısı ortaktır ve encoder
açıldığı anda görülen sıçrama, daha uzun epoch'un tek başına çözüm olmadığını
gösterir.

Checkpoint seçimi global en iyi validation loss'u koruduğu için NaN encoder
ağırlığı Drive'a yazılmamıştır; dosya head-only aşamasında kalmıştır. Bu ayrım
önemlidir: aşağıdaki kazanımlar encoder adaptasyonunun değil, esas olarak mask
decoder/object head adaptasyonunun sonucudur.

| Box-prompt test metriği | Stock | Drive checkpoint | Fark |
|---|---:|---:|---:|
| Mean IoU | `0.7584` | `0.7866` | `+0.0282` |
| IoU ≥ 0.50 | `0.9441` | `0.9651` | `+0.0210` |
| IoU ≥ 0.75 | `0.6043` | `0.6847` | `+0.0804` |
| Small mean IoU | `0.6892` | `0.7441` | `+0.0549` |
| Large mean IoU | `0.7759` | `0.7973` | `+0.0214` |

Sonuç: Stage B fikri işe yarıyor, fakat kaydedilmiş bu model “başarılı encoder
fine-tune” olarak adlandırılmamalıdır. Head belirgin kazanım sağlamış; encoder
LR'si sayısal olarak dağılmıştır.

### 17.2 Sabit 22'de hangi veriler nasıl kullanılıyordu?

22 tek-kare instance eğitimidir. Başlıca denetim kaynakları:

- Kust4K çizilmiş semantic haritalarının connected-component instance'ları,
- SegFly termal semantic haritalarının component instance'ları,
- DroneVehicle thermal ve thermal-only teacher-mask pool'ları,
- HIT-UAV teacher-mask pool'u,
- Kaggle aerial thermal teacher-mask pool'u,
- Kust4K teacher-mask pool'ları,
- VTUAV short-term thermal teacher-mask pool'u,
- bulunursa VTUAV long-term thermal teacher-mask pool'u.

Pool içindeki maske ile görüntü ayrı tutulur; pool maskeyi/RLE'yi, `IMAGE_ROOTS`
gerçek kareyi verir. Teacher maskeleri ikinci kez `box IoU ≥ 0.80` filtresinden
geçer. Frame-level split ve aynı sahne/registered-pair leakage kontrolleri
uygulanır. Bir görüntü encoder forward'ı üzerinde en fazla sekiz instance prompt'u
hesaplanır; eğitim prompt'u exact box ile jittered box karışımıdır.

Sabit 22'nin açıkları:

- `vtuav_lt_thermal` keşfedilebilir ama zorunlu minimum listesinde değildir.
- Düşük-kontrast collapse, polarity flip, gamma ve sensör gürültüsü aktif değildir.
- `LR_SCALE`, batch/16 ile büyüyüp 4'te sınırlanır.
- Encoder aşamasında neck ve trunk aynı `1e-4` absolute LR'yi kullanır; pretrained
  trunk için discriminative LR hiyerarşisi kaybolur.
- `[2,24] × 800` büyük bir bütçedir; `patience=4`, OneCycle toplam 24 epoch'a
  kurulmuşken döngünün çok erken bir bölümünde durabilir.

### 17.3 Neden epoch sayısını yalnız artırmadık?

Bu repoda epoch, veri setini tam bir kez dolaşmak değil, sabit sayıda örneklenen
batch'tir. Örneğin batch 128 ve 800 step, bir epoch'ta 102.400 pencere demektir.
24 encoder epoch'u 19.200 optimizer adımıdır. Veri yaklaşık on binlerce frame
iken bu değer büyük tekrar üretir.

PyTorch `OneCycleLR`, LR'yi her batch'te değiştirir ve toplam adımı
`epoch × steps_per_epoch` üzerinden kurar. Bu nedenle:

- LR kararsızsa epoch artırmak NaN'i düzeltmez.
- Kısa patience, planlanan LR inişi bitmeden stage'i kesebilir.
- En iyi epoch validation checkpoint seçimiyle korunabildiği için önce finite,
  kontrollü bir toplam adım bütçesi daha güvenlidir.

Resmî SAM2.1 MOSE video fine-tune tarifi sekiz karelik video batch'i kullanır,
görüntü encoder LR'sini ana LR'den daha düşük tutar ve cosine decay uygular.
Bu değerleri EdgeTAM'e körlemesine kopyalamadık; fakat pretrained vision trunk'ı
en hızlı parametre grubu yapmama kararı aynı prensiple uyumludur.

### 17.4 Yeni notebook 32'de yapılan Stage B düzeltmeleri

`32_aerial_thermal_stage_b_stable.ipynb`, 22'nin veri keşfi ve leakage
kontrollerini koruyan ayrı bir notebook'tur.

Veri ve kalite değişiklikleri:

- `vtuav_lt_thermal ≥ 5.000 instance` artık zorunludur.
- VTUAV ST ve LT pool'ları `0.8` kaynak ağırlığıyla seyreltilir; yakın ardışık
  karelerin diğer kaynakları ezmesi azaltılır.
- Minimum instance alanı `48 → 64`, minimum kenar `4 → 6` yapılır.
- Görüntünün `%25`inden büyük hedefler dışarı alınır; aerial küçük/orta instance
  dağılımına odaklanılır.
- Eğitim pencerelerinin `%40`ında yerel kontrast en az `0.15` kata kadar
  düşürülebilir.
- `0.25` polarity flip, `0.30` gamma jitter ve 8-bit seviyede `5.0` sensör
  gürültüsü eklenir.
- Validation/test augmentasyonsuz kalır.

LR ve bütçe değişiklikleri:

```python
BATCH_CEILING = 128
LR_SCALE = 1.0
EPOCHS = [2, 12]
STEPS = 500
PATIENCE = 0
```

Encoder LR adayları:

| Profil | Head | Neck | Trunk |
|---|---:|---:|---:|
| cautious | `2e-5` | `2e-5` | `5e-6` |
| stable | `5e-5` | `5e-5` | `1e-5` |
| thermal | `5e-5` | `5e-5` | `2e-5` |

Her profil aynı split/seed ile `1 head + 2 encoder`, epoch başına `150` adımlık
pilot görür. Non-finite profil elenir; kalanların en düşük finite encoder
validation loss'u tam koşunun LR'sini seçer. Pilotun görevi final accuracy
seçmekten çok kararsız bölgeyi ucuza elemek olduğundan test split'i kullanılmaz.

Tam koşu `2 × 500` head ve `12 × 500` encoder adımıdır. En iyi checkpoint
validation ile korunur. `run.json` artık gerçekten uygulanan head/neck/trunk
LR'lerini `effective_rates` altında yazar.

Eğitim çekirdeğine ayrıca iki koruma eklenmiştir:

1. Train veya validation loss non-finite olursa stage anında durur; kalan
   epoch'lar NaN ağırlık üzerinde harcanmaz.
2. Son finite en iyi checkpoint değiştirilmez. Notebook non-finite durmayı
   başarılı encoder eğitimi gibi sunmaz ve hata verir.

32 çıktısı:

```text
MyDrive/edgetam-stage-b/aerial_thermal_stable/
  edgetam_pool_aerial_thermal_stable_512.pt
  run.json
  lr_pilot.json
```

### 17.5 Stage C video eğitimi ne kadar gerekli?

Kullanıcının tarif ettiği hata “tek karede maske kötü”den fazlasıdır: kontrast
eşitlendiğinde takip başka bölgeye kayıyor ve sonraki karelerde orada kalabiliyor.
Bu, modelin kendi yanlış tahminini belleğe yazdığı zamansal geri besleme
problemidir. Statik Stage B bu döngüyü çalıştırmadığı için tek-prompt uzun video
deployment'ında Stage C doğrudan gerekli problemdir.

Stage C'nin daha az gerekli olduğu iki durum vardır:

1. Her karede ayrı detector çalışıyor ve tracker sık sık doğru kutuyla yeniden
   başlatılıyorsa temporal bellek hatasının ömrü kısadır.
2. Stage B, gerçek düşük-kontrast held-out videolarda kabul edilebilir dropout
   sınırlarını zaten karşılıyorsa video fine-tune'un maliyeti kazanımdan büyük
   olabilir.

Bu kararı ölçmek için 31'e eğitimden önce Stage B video precheck'i eklendi.
Her aerial kaynaktan en düşük kontrastlı iki validation track'inin en zor 240
karelik bölümü ölçülür. Başlangıç karar sınırları:

| Ölçü | Stage C kırmızı bayrağı |
|---|---:|
| State Accuracy | `< 0.65` |
| Lost-frame oranı | `> %10` |
| En uzun dropout | `> 24 kare` |

Herhangi bir kaynak eşiği aşarsa `recommend_stage_c=true` yazılır. Sonuç
`stage_c_precheck.json` dosyasına kaydedilir. Yalnız audit istenirse
`STOP_AFTER_PRECHECK=True` seçilir ve eğitim başlamadan notebook kontrollü
biçimde durur. Bu eşikler evrensel bilimsel optimum değil, operasyonel başlangıç
sınırlarıdır; kamera FPS'i ve kabul edilen yeniden-prompt süresine göre
değiştirilmelidir.

31, Stage B başlangıcını şu sırayla arar:

1. 32'nin `aerial_thermal_stable` checkpoint'i,
2. bulunamazsa eski 22 `thermal_deep` checkpoint'i.

İkinci durumda bilinen encoder kararsızlığı nedeniyle açık uyarı basılır.

### 17.6 Güncellenmiş çalışma ve karar sırası

```text
25 (VTUAV-LT thermal pool hazır değilse)
  ↓
32 — LR pilotu + kararlı statik Stage B
  ↓
run.json / lr_pilot.json denetimi
  ↓
31 — Stage B video precheck
  ├─ kabul edilebilir + detector her karede reinit → Stage C opsiyonel
  └─ düşük SA / yüksek lost-rate / uzun dropout → Stage C'yi çalıştır
       ↓
Stage B vs Stage C vs Stage C+SAMURAI validation
       ↓
yalnız seçilen kolun held-out test'i ve before/after videoları
       ↓
test de iyiyse export / TensorRT / INT8 yeniden üret
```

29/30, Anti-UAV410 ground-camera benchmark kolu olarak kalır. Onlar da artık
önce 32 checkpoint'ini arar, yoksa 22'ye fallback yapar. 29 checkpoint'i 31'in
başlangıcı yapılmaz; iki Stage C kolu aynı Stage B'den ayrı dallanır.

### 17.7 Bu değişikliklerden sonra kesin olarak ne söylenebilir?

- Drive'daki eski Stage B checkpoint statik box-prompt testinde stock modeli
  iyileştirmiştir.
- Aynı koşunun encoder aşaması başarısızdır; “24 epoch daha iyi olur” sonucu
  çıkarılamaz.
- Yeni LR/bütçe tarifi sayısal kararsızlığı önlemek ve gerçek encoder kazanımını
  ayırmak için hazırlanmıştır; GPU koşusu tamamlanmadan nihai IoU artışı garanti
  edilemez.
- Stage C, tarif edilen uzun süreli kimlik kayması için teorik olarak doğru
  aşamadır; eklenen precheck ve held-out A/B bunun bu veri üzerinde pratikte
  gerekli ve faydalı olup olmadığını ölçer.
