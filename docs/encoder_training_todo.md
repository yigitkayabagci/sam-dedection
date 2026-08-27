# Encoder Eğitimi ve Pretraining — TODO

Amaç: EdgeTAM'ın **image encoder**'ını termal + havadan veriyle iyileştirmek,
sonra iyileşmiş encoder'la bütün mimariyi birleştirip video üzerinde memory
attention tarafına geçmek.

Buradaki hiçbir şey **ölçülmüş** değil. Bu bir **araştırma ve iş planı**;
ölçülen sonuçlar `docs/EXPERIMENT_LOG.md`'ye, rapora girecek olanlar
`report/`'a yazılır.

A ve B aşamalarının kodu artık yazıldı — **kurulumun** nasıl olduğu
`docs/encoder_mimari.md`'de, çalışan hâli
`notebooks/07_encoder_aerial_rgbt.ipynb`'de. Yazılmış olmak ölçülmüş olmak
değil: aşağıdaki her madde hâlâ kendi taban çizgisine karşı sayı bekliyor.

Veri setleri: `docs/datasets.md` ve raporda Bölüm 5.

---

## 0. Plan — üç aşama

| Aşama | Ne eğitilir | Hangi veri | Durum |
|---|---|---|---|
| **A. Pretraining** | image encoder, etiketsiz | hizalı RGB-T çiftleri | **kod yazıldı** (`src/training/distill.py`), ölçülmedi |
| **B. Encoder + head** | image encoder + mask decoder | **statik** yoğun maskeli setler | **kod yazıldı** (`src/training/image_loop.py`), ölçülmedi |
| **C. Video** | memory attention / encoder | video setleri | B ölçülmeden başlanmaz |

Aşama sırası keyfi değil — gerekçesi §1'de.

---

## 1. Neden bu sıra doğru

**SAM 2'nin kendi eğitim tarifi tam olarak bu.** İki aşamalı:

1. **Görüntü ön-eğitimi:** SA-1B üzerinde, **statik görüntülerle**, promptable
   segmentasyon hedefiyle. Encoder (MAE ile ön-eğitilmiş Hiera) burada oturur.
2. **Video eğitimi:** SA-V + açık video setleri; ve burada bile eğitim
   **statik görüntüyle videoyu dönüşümlü** besliyor, kaynak büyüklüğüyle
   orantılı örnekleyerek.

Yani "statik veri encoder için yeterli" sezgisi doğru, üstelik statik veri
aşama C'de de kullanılmaya devam ediyor — atılmıyor.

Ters yönde de doğrulanıyor: SAM 2 uzun video için ince ayar yaparken (8 → 16
kare) **image encoder'ı donduruyor.** Yani encoder statik veriyle, hafıza yolu
videoyla eğitilen iki ayrı problem olarak ele alınıyor.

**Bu repo zaten aynı ayrımı yapıyor.** `src/training/finetune.py`:

```python
FROZEN_MODULES = ("memory_attention", "memory_encoder", "spatial_perceiver")
STAGES = {
    "head":    HEAD_MODULES,                          # mask decoder + prompt encoder
    "encoder": HEAD_MODULES + ("image_encoder",),     # + trunk
}
```

ve modül dokümanındaki gerekçe: *"It operates on abstract features, not pixels.
Thermal-versus-RGB is an **encoder** problem; there is no reason to expect the
memory path to need it."*

Yani aşama A ve B için gereken dondurma mekanizması **hazır**.

---

## 2. Elde ne var, ne yok

### Yeniden kullanılabilir

| Dosya | Ne veriyor |
|---|---|
| `src/training/finetune.py` | `apply_freeze(model, "encoder")`, iki aşamalı LR, EMA, checkpoint yazma |
| `src/training/losses.py` | SAM 2 tarifi: focal + dice + IoU L1 |
| `src/training/loader.py` | prefetch, `auto_batch_size` (gerçek forward/backward ile ölçüyor) |
| `src/training/labels.py` | öğretmen modelle etiket üretimi — **zoom-crop** ve **kalite kapıları** fikri |
| `tools/train_thermal.py` | eğitim giriş noktası, `--method finetune\|lora` |
| `tools/export_edgetam_onnx.py` → `build_trt_engines.py` → `check_trt_parity.py` | eğitim sonrası dağıtım zinciri |

### Eksik olan — artık yazıldı

Üçü de aşağıdaki gibi çözüldü; hiçbiri ölçülmedi.

| eksikti | nerede karşılandı |
|---|---|
| **statik (tek kare) eğitim yolu** — tek kare encode → prompt → maske → loss, memory bank devreye girmeden | `src/training/image_loop.py`. `clip_loop.py`'nin kardeşi ve ortak kare için **aynı** kod yolunu kullanıyor: tek kare zaten bir klibin 0. karesi, yani `track_step(is_init_cond_frame=True, run_mem_encoder=False)`. Görüntü bir kez encode edilir, pencere içindeki her örnek aynı özniteliklere karşı prompt edilir |
| **veri okuyucu** — Kust4K / SegFly / Caltech için ortak arayüz | `src/training/aerial.py`. Bildirimsel `DatasetSpec` (düzen + palet + hangi sınıf "şey"), `probe_classes` ile indirmeye karşı doğrulanıyor. Örnek ayrıştırma, kapılar ve pencere geometrisi de burada; `crop_window`/`map_boxes` video yoluyla ortak, böylece iki taraf birbirinden ayrışamıyor |
| **`object_score` statik sette karşılıksız** | `losses.instance_loss` — terim sıfır ağırlıklı değil, **hiç yok**. Sıfır ağırlık da yanlış olurdu: sabit 1'e karşı BCE "hiçbir şey öğretme" değil "koşulsuz ateşle" öğretir, ve videoda hafıza bankasını zehirleyen tam olarak o. `box_projection` de yok — orada maske zaten ground truth, geri düşülecek bir şey yok |

Yanında gelen üç şey daha: `schedule.run_stages` artık `Loop` alıyor (klip modu
ve görüntü modu **aynı** aşamalar, EMA ve checkpoint kuralından geçiyor),
`STAGES`'e `backbone` eklendi (ön-eğitimin kapsamı — head'siz), ve
`loader.prefetch` / `auto_batch_size` veri modundan bağımsız hâle getirildi.

---

## 3. TODO A — Pretraining (kavramsal araştırma)

Soru: encoder'ı doğrudan EdgeTAM checkpoint'inden mi ince ayarlamalı, yoksa
önce termale özel bir ön-eğitim mi yapmalı?

### A1. Modalite distilasyonu — en umut verici yol

**AnyThermal** (arXiv 2602.06203, CMU AirLab): DINOv2 gibi bir RGB görsel
temel modelinin özniteliklerini, **hizalı RGB-T çiftleri** üzerinden bir
termal encoder'a distile ediyor. Göreve özel eğitim yok; iç mekân, kentsel,
havadan ve arazi ortamlarında çalışıyor, aşağı akış görevlerinde %36'ya varan
kazanç raporluyor.

Kritik nokta: **bu yöntem etiket istemiyor.** Öğretmen RGB yarısına, öğrenci
termal yarısına bakıyor; semantik etiket hiç devreye girmiyor. Yani §4'teki
semantik-vs-örnek problemi bu aşamada hiç doğmuyor.

Daha da iyisi, **girdi formatı elimizdekiyle birebir aynı.** Hizalı RGB-T
çifti bütçesi:

| Kaynak | Hizalı çift | Not |
|---|---|---|
| MVUAV | ~53 800 | yalnızca 2 183'ü etiketli — **burada fark etmez** |
| SegFly-RGB-T | 15 007 | 640×512 |
| Kust4K | 4 024 | 640×512 |
| Caltech Aerial RGB-T | havadan çekimler | donanım senkronize |

Karşılaştırma için: AnyThermal'ın kendi eğitim seti TartanRGBT **16 943**
çift. Yani sadece havadan veriyle bile o mertebenin üstüne çıkılıyor.

**Cevaplandı (§6 soru 2):** evet, takılıyor. Distilasyon kaybının **öğrenci
tarafı mimariden bağımsız** — hizalanması gereken tek şey bir öznitelik
haritası, ve 1×1 projeksiyon herhangi iki kanal sayısını hizalar. Öğretmenin
ViT olması öğrenciyi bağlamıyor. `src/training/distill.py` bunu
`image_encoder`'ın en üst çıktısına (mask decoder'ın tükettiği ve ONNX
grafiğinin ürettiği [B, 256, S/16, S/16] tensörü) uyguluyor; projeksiyon aşama
bitince atılıyor ve çıkan checkpoint sıradan bir EdgeTAM state dict'i.

**Yapılacak:** koş ve A3 taban çizgisine karşı ölç. Kust4K'nın 4 024 çifti bu
aşama için az olabilir — dürüst testi MVUAV'ın 53 828 çifti.

### A2. MAE / maskeli görüntü modelleme — muhtemelen uygun değil

SAM 2'nin Hiera gövdesi MAE ile ön-eğitilmiş, o yüzden akla ilk gelen bu.
Ama **EdgeTAM'ın gövdesi RepViT, yani evrişimsel** (`finetune.py`'nin kendi
ifadesiyle *"the domain shift here lives in a convolutional RepViT trunk"*).
MAE token maskelemeye dayanıyor ve ViT için tasarlandı; evrişimsel gövdede
doğrudan karşılığı yok.

**Yapılacak:** ConvNeXt/RepViT için maskeli ön-eğitim varyantı var mı diye
bak; yoksa bu yolu kapat ve gerekçesini yaz.

### A3. Ön-eğitim yok — yenilmesi gereken taban çizgisi

Doğrudan EdgeTAM checkpoint'inden `apply_freeze(model, "encoder")` ile ince
ayar. **Her ölçüm buna karşı raporlanmalı**, yoksa ön-eğitimin bir şey
kattığı iddia edilemez.

---

## 4. TODO B — Encoder'ı statik veriyle eğitmek

### 4.1 Asıl mesele: semantik etiket yanlış hedef

Veri setlerinin neredeyse tamamı **semantik segmentasyon**: piksel → sınıf,
yani "bütün arabalar araba". EdgeTAM'ın ihtiyacı olan şey bu değil —
**örnek** ayrımı: "şu araba, yanındaki değil".

Semantik hedefle eğitmek aktif olarak zarar verir: yan yana iki arabanın
özniteliklerini birbirine **yaklaştırır**, oysa takip için ayrışmaları
gerekir.

**Çözüm veri setini değiştirmek değil, aynı veri üzerinde hedefi
değiştirmek.** SAM 2 zaten böyle eğitiliyor: semantik harita hiç görmüyor,
görüntü başına **64 maskeyi tek tek örnekleyip** her birini ayrı ayrı prompt
ediyor (görüntünün %90'ından fazlasını kaplayan maskeler eleniyor).

Uygulama:

1. Semantik haritayı **bağlantılı bileşenlere** ayır.
2. Yalnızca **"şey" (thing) sınıflarını** al — Kust4K'da 9 sınıfın 4'ü:
   motosiklet, araba, kamyon, insan. "Şey olmayan" (stuff) sınıflar — yol,
   bina, ağaç, trafik tesisi — takip hedefi değil, atılır ya da tek parça
   bırakılır.
3. Her bileşene bir nokta/kutu prompt'u ver, o bileşenin maskesini hedef al.
4. SAM 2'nin kaybını uygula (`losses.py`'deki focal + dice + IoU, `exist`
   terimi kapalı).

**Ölçülebilir ilk kontrol — ucuz ve hemen yapılabilir:** bağlantılı bileşen
ayrıştırması temiz örnek veriyor mu? Yan yana duran iki araç tek bileşene
kaynıyorsa hedef yine bozuk olur. Bileşen sayısı ve alan dağılımını çıkar;
gerekirse en büyük %N'i ele.

*Bunun aracı hazır:* `index_frames` + `summarise` semantik haritaları okuyup
(görüntü decode etmeden, GPU'ya dokunmadan) sınıf başına örnek sayısını, boyut
dağılımını ve hangi kapının neyi reddettiğini raporluyor. Kaynaşmanın sinyali
`InstanceGates.fill` — kendi kutusunun dörtte birini dolduran bileşen genelde
ince bir piksel köprüsüyle birleşmiş iki nesnedir. Notebook 07'nin 11. ve 12.
hücreleri sayıyı ve görüntüyü birlikte veriyor; **hiçbir GPU-saati harcanmadan
önce**. `decompose` bileşenleri sınıf sınıf buluyor, birleşim üzerinden değil —
bir arabaya değen bir insan böylece iki örnek kalıyor.

**AeroVIS'te bu adım hiç gerekmiyor** — zaten örnek maskesi + kimlik veriyor
(YTVIS biçimi). Ama o video, aşama C'ye ait.

### 4.2 Hangi veri, hangi sırayla

| Öncelik | Set | Neden |
|---|---|---|
| 1 | **Kust4K** | 640×512 termal, native çözünürlük, sıfır yeniden ölçekleme; 4 "şey" sınıfı |
| 2 | **SegFly-RGB-T** | hizalı çift 640×512, 15 007 örnek, 15 sınıf |
| 3 | **Caltech Aerial RGB-T** | FLIR ADK 640×512, gerçek uçuş; ama sınıflar doğa odaklı, küçük hedef yok |
| stres testi | **SkyScapes** / **MESSI** | ince ve küçük yapı; MESSI ayrıca irtifa ekseni (bkz. rapor §3.4) |

### 4.3 Dikkat edilecek teknik noktalar

- **BatchNorm.** `clip_loop.py` modeli eğitim sırasında `eval()` modunda
  tutuyor; gerekçesi RepViT'in batch norm'larının TensorRT export'unda
  katlanmış istatistiklerden sapmaması. **Statik döngü için de aynısı
  geçerli** — aksi halde motor checkpoint'le eşleşmeyi bırakır.
- **Encoder ONNX grafiği mask decoder'ın `conv_s0`/`conv_s1` projeksiyonlarını
  içine katlıyor** (`export_image_encoder_onnx.py` notu). Yani head ile
  encoder birlikte eğitilirse encoder grafiği de değişir; ikisi ayrı ayrı
  düşünülemez.
- **Sadece encoder değişirse yalnızca o motor yeniden derlenir**, diğer üçü
  aynı kalır. Ama `check_trt_parity.py` her durumda yeniden koşmalı.
- **Çözünürlük.** Eğitim hangi `image_size` ile yapılırsa motor da o şekilde
  derlenmeli; `_hydra_overrides.py` `q_sizes`'ı kilitli tutuyor
  (rapor §3.4).

---

## 5. TODO C — Video aşaması (B bitmeden başlanmaz)

- Eğitilmiş encoder'ı **dondur**, memory attention / memory encoder'ı aç.
  `finetune.py`'nin `FROZEN_MODULES`'ü bugün tersini yapıyor, yani yeni bir
  aşama tanımı gerekecek.
- SAM 2'nin tarifi gereği bu aşamada **statik veriyle video dönüşümlü**
  beslenmeli — B'nin verisi atılmıyor.
- Veri: **AeroVIS** (örnek maskesi + kimlik, göreve en yakın),
  **UAVScenes** (her kare etiketli, kesintisiz J&F'in mümkün olduğu tek set),
  **MVUAV** (25 karede bir).
- **Dördüncü kaynak artık kod:** VTUAV'ın ~400 dizisinin tek anotasyonu kare
  başına bir kutu, ama `tools/make_masklets.py` bunları bir video
  öğretmeniyle (varsayılan SAM 2.1, gated değil; SAM 3 tek string uzakta)
  masklet'e çeviriyor. Çıktı `labels.py`'ın deposunun aynısı, yani aşama C
  iki kaynağı tek arayüzden okuyor. Harcamadan önce ölçü var: `--calibrate`,
  VIS bölümünün **çizilmiş** maskelerine karşı masklet IoU'sunu basıyor.
  Ayrıntı `docs/encoder_mimari.md` sonu ve `docs/encoder_arastirma.md` §8.
  **Arşivi bütün açmak şart** (`--frames all`): `masklets.find_sequences`
  konum ile kare numarasını bir tutuyor (depo kare numarasıyla anahtarlanıyor,
  `calibrate` de çizilmiş PNG'nin adındaki sayıyla arıyor), o yüzden `tracked_*`
  ya da `masked` ile açılmış — kareleri 10'ar/30'ar atlayan — bir ağacı artık
  **okumayı reddediyor**. Havuz defterlerinin (16/17/24/25) `tracked_*`
  çıkarımı masklet'e yaramaz; bu ikisi ayrı çıkarım istiyor. Aşama C'nin
  gerçek işi burada: konum ile kare numarasını `VideoSequence` içinde ayırmak,
  ki 1/10 örneklenmiş bir ağaç da masklet üretebilsin.
- `clip_loop.py` hazır: teacher forcing yok, model kendi yazdığı hafızayı
  okuyor — düzeltilmek istenen hata modunun tam olarak ortaya çıkacağı
  kurulum.

---

## 6. Açık sorular

1. **Bağlantılı bileşen ayrıştırması temiz örnek veriyor mu?** §4.1'deki ilk
   kontrol. Bütün B aşaması buna bağlı; en ucuz ve en riskli adım bu.
   *Ölçüm hazır* — notebook 07 hücre 11–12, GPU'suz. Cevap hâlâ yok.
2. ~~**AnyThermal'ın distilasyon hedefi RepViT gövdesine takılıyor mu?**~~
   **Cevaplandı: evet.** Distilasyon kaybının öğrenci tarafı mimariden
   bağımsız; hizalanması gereken tek şey bir öznitelik haritası. Gerekçe §3
   A1'de, uygulaması `src/training/distill.py`'de. Geriye kalan soru artık
   "takılıyor mu" değil, **"bir şey kazandırıyor mu"** — o A3'e karşı ölçülecek.
3. **Kust4K'nın "şey" sınıfları yeterli örnek üretiyor mu?** 4 024 görüntüde
   kaç araç/insan bileşeni var — bilinmiyor, sayılmalı. *Sayan kod hazır*
   (`summarise(index, spec)`), sayı hâlâ yok.
4. **Statik eğitim gerçekten J&F'i iyileştiriyor mu?** Encoder'ı statik veriyle
   eğitip video setinde ölçmek, aşamalar arası transferin işe yaradığı
   varsayımına dayanıyor. Bu varsayım A3 taban çizgisine karşı ölçülmeli.
   **Bu soruyu notebook 07 cevaplayamaz** — `eval_instances.py` tek prompt'lu
   kare skorluyor, J&F ise hafıza bankası istiyor. Vekil ile asıl metriğin
   karıştırılmaması gereken yer burası.
5. **Veri sızıntısı.** Eğitim ve değerlendirme setleri kaynak paylaşmamalı;
   hangi ikililerin çakıştığı raporda Bölüm 5.3'te. Güvenli kurulum:
   Kust4K + MVUAV'da eğit, J&F'i AeroVIS ya da UAVScenes'te ölç.
6. **PEFT mi tam fine-tune mu?** Notebook 06 bunu klip modunda soruyor,
   notebook 07 aynı soruyu görüntü modunda soruyor — ve cevaplar aynı olmak
   zorunda değil: burada hareket eden 9.3 M parametreye karşı çok daha çeşitli
   bir sahne kümesi var. İki ölçüm ayrı ayrı raporlanmalı.

---

## Kaynaklar

- SAM 2 — arXiv 2408.00714 (iki aşamalı tarif, dönüşümlü statik/video eğitimi)
- AnyThermal — arXiv 2602.06203, <https://anythermal.github.io/>
- Veri setleri — `docs/datasets.md`, rapor Bölüm 5
- Mevcut eğitim kararları — `src/training/finetune.py` modül dokümanı
- **Kurulumun kendisi** — `docs/encoder_mimari.md` (bu planın uygulanmış hâli)
- **Çalıştırmak için** — `notebooks/07_encoder_aerial_rgbt.ipynb`
