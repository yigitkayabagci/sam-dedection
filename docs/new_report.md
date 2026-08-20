# Termal Drone Uzmanlaştırması — Ne Yapıldı, Ne Yapılacak

Bu belge, bu dalda yapılan işin tamamının Türkçe özeti ve çalıştırma kılavuzu.
Amaç: sohbet geçmişine erişemesen bile buradan devam edebilmek.

**Durum: bir ölçüm yapıldı, gerisi hazır ama ölçülmedi.** Yapılan ölçüm
LoRA'ya karşı kısmi fine-tuning (§2, `docs/lora_vs_finetune.md`) — sonucu
deponun ilk argümanını çevirdi ve varsayılan checkpoint'i değiştirdi. Geri
kalan tablolardaki rakamların tamamı `docs/tensorrt_fp16.md`'deki mevcut fp16
çalışmasından geliyor; onların üzerine kurulan her şey hâlâ plan.

---

## 0. Hızlı başlangıç — senin sıradaki adımların

```
1. Anti-UAV410'u indir  →  Drive'a koy          (notebook 01 anlatıyor)
2. notebook 01  →  pseudo-mask etiketleri üret   (~1-2 sa, A100)
3. notebook 02  →  fine-tune, checkpoint çıkar   (~2-4 sa)
3b. notebook 06 →  LoRA vs fine-tune  ✅ koşuldu   (karar: LoRA, §2)
4. notebook 03  →  INT8 kalibrasyon + PTQ
5. notebook 04  →  SAMURAI  (bunu 1'den önce de çalıştırabilirsin)
6. notebook 05  →  adaptif ROI (en riskli parça, en sona bıraktım)
```

**Notebook 04 hiçbir şeye bağımlı değil.** SAMURAI eğitimsiz, ağırlık
değiştirmiyor, yeni engine istemiyor. Veri indirmesi uzarsa ya da en hızlı
ölçülebilir sonucu istiyorsan oradan başla.

---

## 1. Dataset: Anti-UAV410

[github.com/HwangBo94/Anti-UAV410](https://github.com/HwangBo94/Anti-UAV410) —
410 termal-kızılötesi video, 438K elle etiketlenmiş kutu, **640×512**,
train/val/test ayrımlı. Yazarların Google Drive'ındaki tek bir 8.7 GB'lık zip;
`tools/fetch_antiuav410.py --dest /content/data` onu indirir, açar ve çıkan
yerleşimi doğrular. Dataset eğitimi yapan makinenin **yerel diskine** iner,
Drive'a değil: eğitim yüz binlerce küçük JPEG'i rastgele sırayla okur ve Drive
FUSE mount'u bunları GPU'nun tükettiğinden bir mertebe yavaş servis eder.

### Neden bu dataset

| özellik | senin mimarine etkisi |
|---|---|
| 640×512 mono termal | **512×512 merkez crop = native piksel, sıfır resize.** `tools/run_records.py`'deki `crop512` modunun aynısı |
| kare başına `exist` bayrağı | `object_score_logits` için doğrudan supervision — hafıza kirlenmesi hatanın tam sinyali |
| video (tek kare değil) | hafıza bankası modelin kendisi; tek karelik set onu ne eğitir ne ölçer |
| Occlusion / Out-of-View öznitelikleri | hata modun artık anekdot değil, etiketli değerlendirme dilimi |
| Tiny/Small boyut öznitelikleri | notebook 05'in latency'sine değip değmediğini bu söyler |

Elenen alternatifler: **HIT-UAV** (drone platform, hedef değil; tek kareler),
**LSOTB-TIR** (genel termal, drone'a özel değil), **Anti-UAV-RGBT** (RGB akışı
gereksiz). **CST Anti-UAV** ikincil, daha zor test seti olarak öneriliyor.

### Tek kanal mı?

**İçerik olarak evet** — termal, tek bant, renk yok, R=G=B.
**Dosya kodlaması doğrulanmadı** — JPEG dağıtılıyor ve termal veri JPEG'e
genelde 3 kanal (R=G=B) yazılıyor. İndirince kontrol et:

```bash
python3 -c "import cv2,sys; im=cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED); print(im.shape, im.dtype)" <kare>.jpg
```

Kodda fark etmiyor: `src/training/antiuav.py:load_window` iki durumu da aynı
yere indiriyor. Zaten `17500f6` commit'in ("detect grayscale-as-3-channel by
pixel value, not ndim") bu tuzağı bir kez yaşamış.

> **Dikkat edilecek asıl uyumsuzluk.** Senin kayıtların `*.tif*`. Eğer bunlar
> **16-bit radyometrik TIFF** ise gerçek bir domain gap var: Anti-UAV410 8-bit
> JPEG (dinamik aralık bir kez sabitlenmiş), seninkiler ise `to_rgb8` ile
> **kare kare min/max normalize** ediliyor — yani kontrast sürekli değişiyor.
> Kendi `frames/` klasöründen bir kareyle yukarıdaki komutu çalıştır ve
> `dtype`'a bak. Sabit-kontrast veriyle eğitip değişken-kontrastta çalıştırmak
> ölçülmeden varsayılacak bir şey değil.

---

## 2. Karar: LoRA — ölçüldü, ilk argüman yanlış çıktı

Bu bölüm "sonuç ne çıkarsa ona göre yeniden yazılacak" diyordu. Ölçüldü, ve
sonuç deponun ilk argümanının tersi: **512 termal için varsayılan artık LoRA**
(`configs/edgetam_512_lora.yaml`). Tam tablo, gerekçe ve sınırlar:
`docs/lora_vs_finetune.md`.

**test split, 25 dizi — hiçbir eğitim bunu görmedi:**

| | state acc | success AUC | epizot | kayıp kare | medyan uzunluk |
|---|---:|---:|---:|---:|---:|
| stock | 0.3840 | 0.3800 | 48 | 12 270 | 8.0 |
| finetune | 0.5626 | 0.5593 | 43 | 7 070 | 56.0 |
| **lora** | **0.5669** | **0.5650** | 54 | **6 815** | **31.0** |

**Maliyet tarafı:** finetune 9.318 M eğitilebilir parametre, 36 dk, 77.6 GiB,
en iyi val 0.1468. LoRA 1.064 M — fine-tune'un oynattığının **%11.42**'si, 131
katman, r=16 — 41 dk, 77.7 GiB, en iyi val 0.1461.

### Önce ilk satır okunur

Uyarlamanın kendisi stock'a göre **+0.183 state accuracy** ve hedefin kayıp
olduğu kare sayısında **%44 azalma** demek. İki yöntem de bunun neredeyse
tamamını veriyor. LoRA mı fine-tune mu sorusu, birinci dereceden bir kazancın
içindeki ikinci dereceden bir karar — bu sıra kaybedilmemeli.

### Doğruluk farkı berabere

0.0043 state accuracy, tek seed. Notebook'un, sayılar görülmeden yazılmış
kuralı: 0.01'in altındaki bir farka inanmak için iki-üç seed gerekir. Val
kaybında da aynı hikâye (0.1461 / 0.1468) — ayrışma yok. Dürüst ifade "LoRA
daha doğru" değil, **"aynı doğruluk, parametrenin %11'iyle"**.

### Berabere olmayan şey: dropout'un şekli

`docs/samurai.md`'nin anlattığı arıza bu: kötü bir karede bankaya
`no_obj_ptr` yazılıyor, sonraki `num_maskmem` kare onu geri okuyor ve mimaride
bunu geri alan hiçbir şey yok. **Uzun dropout = hafıza zehirlenmesi; kısa
dropout = sadece kötü bir kare.**

- toplam kör kalınan süre: 6 815 / 7 070 kare — %3.6, marjinal
- medyan epizot: **31 / 56 kare** — 30 fps'te ~1.0 s'e karşı ~1.9 s
- ortalama epizot: 126 / 164 kare — kuyruk da daha hafif
- epizot sayısı: 54 / 43 — LoRA hedefi **daha sık** kaybediyor

Yani fine-tune daha seyrek düşüyor ama neredeyse iki katı süre yerde kalıyor;
LoRA daha sık göz kırpıyor ve daha çabuk toparlıyor. Anti-UAV'de operasyonel
maliyet "kaç saniye kör kalındı" olduğu için beraberlik LoRA lehine bozuldu.
Bu, bu uygulamaya dair bir yargı — yöntemler hakkında genel bir doğru değil.

Mekanizma olarak da tutarlı: rank-16 bir güncelleme bir katmanı o kadar uzağa
taşıyamaz, dolayısıyla uyarlanmış encoder altındaki genel amaçlı özniteliklere
yakın kalır ve termal ipuçları belirsizleştiğinde onlara geri düşer.
Beraberliğin işaret ettiği mekanizma bu; ayrıca ölçülmedi.

### Beklenen iki fayda tutmadı

Bunu açıkça yazmak gerek, çünkü LoRA genelde bu ikisi için seçiliyor:

- **Bellek kazandırmadı.** Aynı peak, aynı batch — ikisi de 32, ve batch
  `auto_batch_size` ile ölçüldü, sabitlenmedi. LoRA'nın kaçındığı optimizer
  state 8.25 M parametre üzerinde iki moment, ~63 MB; peak ise 32 klip × 8 kare
  × 512²'lik aktivasyon. 77 GiB'ın yanında 63 MB hiçbir şey. Optimizer state'in
  gerçekten baskın olduğu bir GPU'da hikâye başka olurdu; bu o GPU değil.
- **Hız kazandırmadı.** %14 daha yavaş: adapter'lar ileri geçişte her uyarlanan
  katmanın yanına ikinci bir konvolüsyon koyuyor, geri geçiş yine trunk'ın
  tamamından geçmek zorunda. Ucuzlayan tek şey ağırlık güncellemesi, ki pahalı
  olan zaten o değildi.

Geriye üçüncü sebep kalıyor ve işe yarayan o: **güncelleme rank ile sınırlı** —
dondurma politikasının ifade edemediği bir düzenlileştirme.

### Tablodaki bir satır zaten yanlıştı

"LoRA `nn.Linear` hedefler, domain kayması konvolüsyonel trunk'ta" — bu
*Linear-only bir implementasyona* itiraz, yöntemin kendisine değil. `k × k` bir
konvolüsyonun `1 × 1` ile bileşkesi yine `k × k` bir konvolüsyondur; düşük
ranklı çarpanlara ayırma RepViT trunk'ında da **tam** geçerli.
`src/training/lora.py` konvolüsyonları da adapte ediyor, sadece gruplu
(depthwise) olanları atlıyor — onların ağırlığı blok-köşegen, yoğun bir `B @ A`
ona bir güncelleme temsil edemez. Kaç tanesini atladığını da raporluyor.
Uyarlanan 131 katmanın büyük kısmı bu yüzden trunk'ta.

### Ne söylemiyor

- **Tek seed.** 0.0043'ü "LoRA daha doğru" diye okumak için `--seed 1`,
  `--seed 2` ile tekrar gerekir.
- **60 dizi.** Asıl risk örnek sayısı değil sahne çeşitliliği: 60 dizi = 60
  arka plan, tek sınıf, tek sensör. Fine-tune'un kullanacak daha fazla
  kapasitesi var ve sahne eklendikçe farkı kapatması beklenir — yani bu karar
  60 sahnelik bir bütçeye özgü, bütçe büyürse yeniden bakılmalı.
- **Termal, 512 crop, Anti-UAV410.** RGB'ye, başka bir sensöre ya da tam kare
  640×512 girdiye hiçbir şey aktarılmıyor; her biri yeniden ölçüm ister.
- **r = 16 taranmadı.** Yaygın bir varsayılan olarak seçildi ve çalıştı.

### Orijinal ağırlıklar diskte duruyor

`tools/train_thermal.py --method lora` artık iki dosya yazıyor: merge edilmiş
checkpoint (54 MB, deployment artefaktı) ve **adapter'ın kendisi** (birkaç MB).
`configs/edgetam_512_lora_adapter.yaml` stock EdgeTAM'ı yükleyip adapter'ı
açılışta merge ediyor — aynı model, aynı maskeler, milisaniyelik ek yük. Kazanç
dosya sisteminde:

- upstream'in checkpoint'i bit bit aynı kalıyor, "bu proje neyi değiştirdi"
  sorusunun cevabı 54 MB değil birkaç MB;
- ikinci bir domain (RGB, başka sensör, başka irtifa bandı) ikinci bir 54 MB
  değil, aynı tabanın yanında ikinci bir adapter oluyor;
- `lora_scale` stock (0.0) ile eğitildiği hâl (1.0) arasında karışım veriyor —
  domainin kısmen tuttuğu görüntüler için gerçek bir düğme, ve 1.0 dışındaki
  her değer ölçülmemiş bir deney.

Bunu abartmamak lazım: **merge edildikten sonra LoRA checkpoint'inin
ağırlıkları da fine-tune kadar değişmiştir** — aynı tensörler, aynı yerler.
Sınırlı olan değişimin *rank*'i, değişimin varlığı değil. "Orijinali bozmamak"
iki ayrı argüman: adapter'ı ayrı tutmak *diskte ne durduğuna* dair, rank ise
*ne öğrenildiğine* dair. İkisi de ayrı ayrı geçerli.

ONNX export ve engine build hâlâ merge edilmiş dosyadan geçiyor; TensorRT
adapter diye bir şey görmüyor.

### Nasıl koşuluyor

`tools/train_thermal.py --method {finetune,lora}` iki yöntemi **tek koddan**
çalıştırıyor: aynı klipler, aynı sıra, aynı loss, aynı iki aşama, aynı
one-cycle, aynı EMA, aynı validation dilimi (`src/training/schedule.py`).
`--method` yalnızca iki şeyi değiştiriyor — hangi parametrelerin gradyan aldığı
ve checkpoint'in nasıl yazıldığı. Tek kasıtlı fark **learning rate**: LoRA
`B = 0`'dan başlıyor, gidecek yolu daha uzun; her yayınlanmış reçete ona bir
mertebe fazla veriyor. İkisi de `tools/train_thermal.py:RATES` içinde, tek
yerde. `notebooks/06_lora_vs_finetune.ipynb` ikisini arka arkaya koşturup
yukarıdaki tabloyu ve dizi başına farkı çiziyor.

### Hep donuk: tüm hafıza yolu

`memory_attention`, `memory_encoder`, `spatial_perceiver`, öğrenilmiş hafıza
token'ları. Üç bağımsız gerekçe:

1. **Rekürrent döngünün yazma portu.** Kendi INT8 ölçümün gösterdi: yalnız
   memory encoder'ı nicemlemek 0.9397'ye düşürdü ve — diğer her modülden
   farklı — hata **klip boyunca sürekli düşüş** şeklindeydi. Çünkü çıktısı
   bankaya yazılıyor, sonraki yedi kare geri okuyor.
2. **ONNX yeniden yazımının en karmaşık kısmı orada** (sabit slotlar,
   tile'lanmış RoPE, additive mask). O ağırlıkları yeniden eğitmek checkpoint
   ile engine arasında sessiz uyumsuzluk davetiyesi.
3. **Piksel değil soyut öznitelik işliyor.** Termal-RGB kayması encoder problemi.

### Eğitilen

| aşama | modüller | LR |
|---|---|---|
| A | `sam_mask_decoder` (+ IoU ve object-score head) + `obj_ptr_proj` | 1e-4 |
| B | + `image_encoder` (trunk / neck ayrı) | trunk 1e-5, neck 5e-5 |

EMA 0.999, cosine, val state-accuracy üzerinden early stop.

### İki ayrıntı

- **Model `eval()` modunda eğitiliyor.** RepViT batch-norm dolu; istatistiklerin
  kayması TensorRT'nin ağırlıklara katladığı istatistiklerle uyuşmazlığa ve
  öncesinde alınmış INT8 kalibrasyonunun geçersizleşmesine yol açar. Ayrıca
  SAM 2 `training` modunda `object_score_logits` döndürmüyor.
- **Teacher forcing yok.** Her kare *modelin kendi yazdığı* hafızaya
  koşullanıyor — bankaya ground truth beslemek, hiç kendi hatasını görmemiş bir
  model eğitmek olurdu.

### Etiket problemi

Anti-UAV410 **kutu** veriyor, EdgeTAM **maske** üretiyor. Çözüm: **SAM 2.1-Large
öğretmeni**, GT kutuyla prompt'lanıp **4× zoom'lanmış crop** üzerinde çalışıyor
(zoom şart — hedeflerin çoğu birkaç piksel), maske geri haritalanıyor. Dört kapı:

| kapı | eşik |
|---|---|
| öğretmenin kendi IoU'su | ≥ 0.7 |
| maske-kutu vs GT-kutu IoU | ≥ 0.6 |
| alan oranı (maske / kutu) | 0.15 – 1.3 |
| en büyük bağlı bileşen | ≥ 0.8 |

Geçemeyen kareler `exist` supervision'ını koruyup BoxInst tarzı projeksiyon
loss'una düşüyor. **Kabul oranı ölçülen sayı olarak raporlanıyor** ve *hangi*
kapının reddettiği ne düzelteceğini söylüyor (çoğu `area` ise zoom yanlış,
çoğu `teacher_iou` ise öğretmen bu görüntüde yetersiz).

> **Ortam tuzağı.** EdgeTAM kendini `sam2` paketi olarak kuruyor (SAM 2 fork'u).
> Meta'nın `sam2`'si ile aynı ortamda yaşayamazlar. Notebook 01 öğretmeni
> `transformers` üzerinden çalıştırıyor; notebook 02+ EdgeTAM kuruyor.
> **Diskteki maske deposu aralarındaki tek arayüz. Notebook 01 için ayrı
> runtime kullan.**

---

## 3. Precision: hangi modüle ne, ve şu an nerede duruyoruz

### Sorunun cevabı: PTQ hepsine, QAT sadece kalırsa image encoder'a

Bunun temeli senin kendi tablon (`docs/tensorrt_fp16.md`):

| modül | FLOP payı | absmax (TensorRT varsayılanı) | yüzdelik kırpma |
|---|---:|---:|---:|
| memory attention | **61.9 %** | 0.9998 | 0.9998 |
| memory encoder | 8.8 % | **0.9999** | 0.9397 |
| SAM head | 2.5 % | 0.9993 | 0.9990 |
| image encoder | 26.8 % | **0.8170** | 0.9936 |

İki kalın hücre **zıt yönleri** gösteriyor — tüm strateji buradan çıkıyor:

- **Image encoder min/max altında çöküyor.** RepViT derinlemesine-ayrılabilir;
  kanal başına aralıklar mertebe farkıyla değişiyor, TensorRT ise aktivasyonu
  *tensör başına* ölçekliyor. Birkaç aykırı değer ölçeği geriyor, dağılımın
  gövdesi çözünürlüksüz kalıyor. → **entropy kalibrasyonu**
- **Memory encoder tam tersi — kırpma ona zarar veriyor.** Yuvarlama dağılır,
  kırpma **sistematiktir**; memory encoder'ın çıktısı saklanan hafızanın
  kendisi, sapma bankaya girip yedi kare geri okunuyor. → **max, asla kırpma**
- **Memory attention aritmetiğin %62'si ve INT8'i neredeyse hiç hissetmiyor.**
  Modeldeki en iyi oran. Bankayı sadece *okuyor*, yazma sorununu almıyor.

**Uygulandı mı?** Araç ve plan yazıldı ve test edildi; **hiçbir engine
derlenmedi, hiçbir sayı ölçülmedi.** Planı görmek için:

```bash
python tools/quantize_edgetam.py plan
```

### Kalibrasyon verisi gerçek olmak zorunda

Image encoder kolay: termal kare besle. Diğer üçü `pix_feat`, doldurulmuş
`memory`, sigmoid'lenmiş `mask_for_mem` alıyor — bu dağılımları kimse yazamaz,
gürültüyle kalibre etmek **var olmayan bir ağın** ölçeklerini üretir.

Bu yüzden kalibrasyon **gerçek bir takip koşusundan, engine sınırında, engine'in
gerçekten gördüğü düzende** kaydediliyor (`src/trackers/calibration.py`).
Kaydedici `TRTEngine` arayüzü sunan her şeyi sarıyor — yani `reference_engines`
(PyTorch, TensorRT'siz, **varsayılan bu olmak zorunda:** kalibrasyon INT8
engine *var olmadan önce* yapılır) veya Orin'deki gerçek fp16 engine'ler.

İki ince nokta: örnekler **klip boyunca stride'lı** alınıyor (hafıza bankası ilk
~16 karede doluyor, o geçiciyi kalibre etmek yanlış duruma ayar yapmaktır); ve
**yanlış çözünürlükte alınmış yakalama reddediliyor** (512 grafiğini 1024
aktivasyonla kalibre etmek gayet geçerli ama tamamen yanlış bir engine üretir).

### PTQ / QAT / distilasyon iş bölümü

| | ne için | ne zaman |
|---|---|---|
| **PTQ** | kalibre ölçek, eğitim yok | **varsayılan ve genelde işin sonu** |
| **QAT** | sahte nicemlemeyle fine-tune, ağırlıklar yuvarlamayı emsin | **sadece PTQ'nun eksik bıraktığı modül** — pratikte image encoder. Fine-tune programının ~%10'u |
| **distilasyon** | öğretmen çıktısını eşleme | **iki farklı yerde**, aşağıda |

Distilasyon iki uçta ve ikisi alakasız:
1. **SAM 2.1-L → EdgeTAM, etiket için** (`src/training/labels.py`)
2. **fp16 EdgeTAM → INT8 EdgeTAM, QAT sırasında** (`src/training/qat.py`) —
   öğretmen *nicemleme öncesi aynı ağ*, çünkü hedef onu birebir üretmek. Kısa
   programın çalışmasının sebebi de bu.

QAT loss'u **KL divergence**, soft-target cross-entropy değil: ikisi öğretmenin
kendi entropisi kadar farklı, o da "mükemmel uyum" tabanını keyfi bir pozitif
sayıya koyar. KL olarak **sıfır = birebir aynı**.

**QAT hafıza yolunu açmıyor.** `insert_quantizers`, `memory_attention` ve
`memory_encoder`'ı iki kez sorulmadan reddediyor. Bir hafıza-yolu modülü
kapıdan geçemezse doğru cevap onu fp16'da bırakmak.

**FP8 yok:** Orin sm_87, FP8 tensor core'u yok.

### Kabul kapısı

Modül başına: Anti-UAV410 val **state accuracy** düşüşü ≤ 1.0 puan **ve** fp16
engine'e karşı maske IoU ≥ 0.99. Geçemeyen modül tek başına fp16'ya dönüyor,
diğerleri hızını koruyor — her engine bağımsız.

```bash
python tools/quantize_edgetam.py capture  --data <anti-uav410> --calib outputs/calib/
python tools/quantize_edgetam.py quantize --outdir models512/ --qdq-outdir models512_int8/ --calib outputs/calib/
python tools/build_trt_engines.py --outdir models512_int8/ --precision auto --max-batch 4
python tools/check_trt_parity.py  --outdir models512_int8/ --image-size 512
python tools/eval_antiuav.py --data <anti-uav410> --split val --tracker edgetam_trt \
    --config configs/edgetam_trt_int8_512.yaml
```

`--precision auto` önemli: INT8 build'ler **strongly typed** (TensorRT
kalibre Q/DQ düğümlerine uysun diye), ama strongly typed altında Q/DQ'su
*olmayan* bir grafik fp32'de koşar. Global `--precision int8` fp16'da tutulan
modülleri sessizce **yavaşlatırdı**. `auto` her modülün spec'ine yazılan
seçimi okuyor.

**Saklanmayan maliyet:** strongly typed engine IO tiplerini grafikten alıyor,
o da fp32 — yani fp16 build'in `--inputIOFormats` ile kazandığı sınır trafiği
geri veriliyor. INT8 hesabının bunu kapatıp kapatmadığı ölçüm işi, modül başına.

Ayrıntı: `docs/quantization_int8.md`.

---

## 4. SAMURAI — asıl koymak istediğin şey

### Hedeflediği hata

EdgeTAM her karede `object_score_logits` üretiyor. Sıfırın altına düşünce maske
sıfırlanıyor **ve bankaya gerçek görünüm yerine sabit "nesne yok" vektörü
(`no_obj_ptr`) yazılıyor**. Sonraki yedi kare onu geri okuyor. Sıfırlanan maske
sorun değil — o kare zaten kötüydü. **Sorun hafıza: tek zor kare onu kirletiyor
ve mimaride bunu geri alan hiçbir şey yok.** Bu precision değil *politika*
problemi; fp16, INT8, fine-tuning hiçbiri dokunmuyor.

### Değişen iki karar

| | stock SAM 2 | SAMURAI |
|---|---|---|
| 3 aday maskeden hangisi | en yüksek IoU-head skoru | `0.15·s_kf + 0.85·s_mask` |
| hangi kareler hafızaya girer | son N, koşulsuz | `iou > 0.5` **ve** `object_score > 0` **ve** `kf_score > 0` |

Ortadaki eşik, EdgeTAM'ın `no_obj_ptr` yazmaya başladığı noktanın ta kendisi.
**Önemli olan yarısı bu: kirli kare hiç yazılmıyor.**

### Neden bedava

Hafıza bankası defter tutması, projenin **bilinçli olarak PyTorch'ta bıraktığı**
kısım (`docs/tensorrt_fp16.md`). SAMURAI'nin değiştirdiği tam olarak orası:

- **Hiçbir engine şekil değiştirmiyor** — filtreleme hangi slotların *canlı*
  olduğunu değiştiriyor, sabit-slot tasarımı 0..`num_maskmem` arasını zaten
  kabul ediyor
- **Kalman filtresi numpy'da 8 durum** — kare başına mikrosaniye, CPU'da
- **Maske yeniden seçimi bir yeniden puanlama** — SAM 2 maskeyi, object
  pointer'ı ve hafıza yazımını *tek* bir `argmax`'tan türetiyor, o değerleri
  değiştirmek üçünü birlikte taşıyor

Tek ölçülebilir maliyet TensorRT'de ve sadece seçim engine'inkinden farklıysa:
128²→512² bilinear upsample, ~0.2 ms.

### Kullanımı

```bash
# PyTorch
python tools/eval_antiuav.py --data <anti-uav410> --split val --limit 20 \
    --tracker edgetam --config configs/edgetam_samurai_512.yaml --json results/samurai.json

# TensorRT (önerilen deployment)
python tools/export_edgetam_onnx.py --outdir models512/ --image-size 512 \
    --checkpoint checkpoints/edgetam_thermal_512.pt --all-pointers --verify
python tools/build_trt_engines.py --outdir models512/ --max-batch 4
python cli.py --tracker edgetam_trt --config configs/edgetam_trt_samurai_512.yaml ...
```

`--all-pointers` neden gerekli: sam_head engine adayını içeride seçiyor. Farklı
aday seçilecekse **o adayın object pointer'ı da lazım** — bankaya yazılan
pointer saklanan maskeyi tarif etmek zorunda. Bayrak yoksa tracker bir kez
uyarıp *sadece maske yeniden seçimini* kapatıyor; **hafıza kapısı çalışmaya
devam ediyor**, yani mevcut engine'ler de faydanın çoğunu alıyor.

### Nasıl ölçülür — ortalama IoU'ya bakma

Bir 60-karelik kayıp ile altmış 1-karelik kayıp **aynı ortalamayı** verir ve
tamamen farklı hatalardır. Hafıza kirlenmesi kesinlikle birincisi.
`tools/eval_antiuav.py` **dropout episode**'larını raporluyor: görünür hedefin
kaç kez ve ne kadar süreyle kaybedildiği.

| gördüğün | anlamı |
|---|---|
| episode sayısı ↓, süreler ↓ | hafıza kapısı çalışıyor — aranan kazanç bu |
| episode ↓, süreler aynı | sadece maske yeniden seçimi işe yaradı; kapı gevşek, `memory_iou`'yu yükselt |
| episode aynı, süreler ↓ | toparlanma iyileşti, ki asıl mesele o |

### Kapatma garantisi

`tests/test_edgetam_trt_integration.py::test_samurai_with_a_permissive_config_is_stock_edgetam`
hiçbir şeyi reddetmeyen eşikler + `kf_weight: 0` ile stock EdgeTAM'ı **piksel
piksel** ürettiğini doğruluyor. Bu garanti olmadan "SAMURAI işe yaradı" ile
"SAMURAI alakasız bir şeyi değiştirdi" ayırt edilemez.

Ayrıntı: `docs/samurai.md`.

---

## 5. SAM2Long — varyant, karşılaştırmak için

Aynı problemi öbür taraftan çözüyor: daha iyi seçmek yerine **seçmeyi
reddediyor**. N rakip hafıza yolu taşıyor, hangisine inanacağına sonda karar
veriyor. Eğitimsiz, uzun videoda daha iyi yayınlanmış sayılar.

**Uygulandı ama varsayılan olarak kapalı** (`src/trackers/sam2long.py`,
`configs/edgetam_sam2long_512.yaml`) — çünkü maliyeti gerçek: memory attention
karenin %61.9'u ve **yol başına bir kez** koşuyor. Üç yol, ölçülen ~10 ms'lik
kareyi ~22 ms'ye taşır. 20-30 ms tavanının içinde ama marjın çoğunu tek tekniğe
harcamak.

**Öneri: varsayılan SAMURAI, SAM2Long ise "12 ms fazladan bir şey satın alıyor
mu" sorusunu ölçmek için açtığın anahtar.** İkisi birlikte çalışabiliyor —
SAMURAI yol *içinde* seçip neyi hatırlayacağını filtreliyor, SAM2Long *yollar
arasında* seçiyor — ve config ikisini birden açıyor.

### Nasıl çalışıyor (yeni inference döngüsü yok)

SAM 2 zaten her batch satırına kendi hafıza bankasını, object pointer'ını ve
object score'unu veriyor; attention satırları asla karıştırmıyor. Yani
**N yol = N kez aynı kutuyla prompt'lanmış N nesne**. Tracker prompt'u
çoğaltıyor, satırlar farklı aday maske alarak ayrışıyor, çıkış en yüksek
kümülatif skorlu satıra indirgeniyor.

Ayrışma bilinçli olarak nadir: yollar sadece en iyi iki aday
`uncertainty_margin` içindeyse **veya** object score `object_score_floor`
altına düştüyse (yani `no_obj_ptr` yazılmak üzereyken, yanılmanın en pahalı
olduğu an) bölünüyor. Makalenin kısıtlı ağaç araması bu, ve hipotezlerin
gürültüye dağılmasını engelleyen şey de bu.

**Yeniden üretilmeyen:** makale ayrıca nesne-oluşum istatistikleriyle her yolun
*hangi* hafızaları tutacağını rafine ediyor. Burada her yol stock SAM 2 hafıza
seçimini (veya SAMURAI açıksa onunkini) kullanıyor.

```bash
python tools/eval_antiuav.py --data <anti-uav410> --split val --limit 12 \
    --tracker edgetam --config configs/edgetam_sam2long_512.yaml --json results/sam2long.json
```

TensorRT'de tek gereksinim: `--max-batch >= pathways × objects` (3 yol × 4 nesne
= 12).

---

## 6. Adaptif inference (notebook 05) — en riskli parça

**SAHI olduğu gibi uygulanmadı.** SAHI bir *detector* tekniği: kareyi dilimleyip
her dilimde detector koşuyor. Burada detector yok, bir prompt'lanmış hedef ve
bir hafıza bankası var. Her karede tile'lamak tracker işi için detector fiyatı
ödemek olur.

Tracker karşılığı: **hedefi takip eden ve kendini ona göre boyutlandıran
pencere**. 10 piksellik drone modele 40 piksel olarak gidiyor.

**Baştan söylenen tehlike:** SAM 2 hafıza özniteliklerini *girdi
koordinatlarında* saklıyor. Crop oynayınca saklanan her hafıza farklı referans
çerçevesinde kalıyor. Bu yüzden pencere **segment sınırı** — boyut merdiveni +
konum ızgarası + kenar marjı ile sabit duruyor, oynadığında takip yeni pencerede
taze bankayla, son maskeden yeniden prompt'lanarak başlıyor.

Notebook 05 doğrulukla birlikte **100 karede kaç pencere hareketi** raporluyor.
Doğruluk düşerse doğru cevap sabit `crop512`, marj da SAM2Long'a gider.

Tile'lama yine de yerini koruyor ama sadece ait olduğu yerde: **hedef
kaybedildikten sonra yeniden yakalama** (`Reacquisition`, `tile_grid`), otuz
karede bir. Sürücüye henüz bağlanmadı — **önce SAMURAI'nin kayıpları zaten
düzeltip düzeltmediğine bak**; episode süreleri kısaldıysa sweep'in yeniden
yakalayacak bir şeyi yok, saf maliyet.

---

## 7. Dosya haritası

### Yeni kaynak

| dosya | ne |
|---|---|
| `src/accuracy.py` | state accuracy, success AUC, **dropout episode**'ları, mask IoU |
| `src/training/antiuav.py` | sekans keşfi, `IR_label.json`, 512 crop geometrisi, klip örnekleme |
| `src/training/labels.py` | SAM 2.1-L öğretmeni, zoom'lu crop, 4 kalite kapısı, RLE maske deposu |
| `src/training/clip_loop.py` | autograd altında kısa-klip BPTT (`track_step` üzerinden) |
| `src/training/losses.py` | focal+dice, IoU L1, **object-score BCE**, kutu projeksiyonu |
| `src/training/finetune.py` | dondurma politikası, param grupları, EMA, checkpoint |
| `src/training/qat.py` | modelopt QAT sarmalayıcı + distilasyon KL'i |
| `src/trackers/calibration.py` | gerçek engine girdilerini kaydetme |
| `src/trackers/samurai.py` | Kalman, maske yeniden seçimi, hafıza kapısı |
| `src/trackers/sam2long.py` | N hafıza yolu (varyant) |
| `src/trackers/adaptive.py` | KF-çapalı adaptif ROI + SAHI tile ızgarası |

### Yeni araçlar

| araç | ne |
|---|---|
| `tools/eval_antiuav.py` | tracker'ı Anti-UAV410 üzerinde koşturup skorlar |
| `tools/quantize_edgetam.py` | `plan` / `capture` / `quantize` |
| `tools/track_adaptive.py` | segmentli adaptif pencere sürücüsü |

### Config'ler

| config | ne zaman |
|---|---|
| `edgetam_512.yaml` | **baseline** — karşılaştırma buna karşı |
| `edgetam_512_lora.yaml` | **512 termal varsayılanı** — LoRA, merge edilmiş |
| `edgetam_512_lora_adapter.yaml` | aynı ağırlıklar: stock checkpoint + adapter |
| `edgetam_512_thermal.yaml` | kısmi fine-tune; artık varsayılan değil (§2) |
| `edgetam_samurai_512.yaml` | + SAMURAI, PyTorch |
| `edgetam_trt_samurai_512.yaml` | **önerilen deployment** |
| `edgetam_trt_int8_512.yaml` | INT8 karışık engine seti |
| `edgetam_sam2long_512.yaml` | SAM2Long varyantı (deneysel) |

### Testler

244 test, hepsi geçiyor, **sadece numpy + torch ile** (EdgeTAM yok, GPU yok):

```bash
python -m unittest tests.test_antiuav_dataset tests.test_accuracy tests.test_pseudo_labels \
    tests.test_training_losses tests.test_clip_loop tests.test_quantization \
    tests.test_samurai tests.test_sam2long tests.test_adaptive
```

Eski üç süit daha fazla ortam istiyor (OpenCV / pytest / EdgeTAM); onlar için
`python -m pytest tests/ -v`.

---

## 8. Dürüst sınırlar

- **§2 dışında buradaki hiçbir sayı yeni bir ölçüm değil.** LoRA/fine-tune
  karşılaştırması gerçek bir çalışma; mevcut fp16 tablosunun altındaki her şey
  hâlâ ondan türetilmiş plan.
- **§2'nin kendisi de tek seed.** Doğruluk farkı gürültünün içinde; karar
  dropout epizotlarının uzunluğuna dayanıyor, doğruluk farkına değil.
- **Adaptif inference (notebook 05) işe yaramayabilir.** Hafıza bankasıyla
  çalışmak yerine ona karşı çalışıyor. Notebook o ihtimali açıkça ölçüyor.
- **SAM2Long makalenin tamamı değil** — nesne-oluşum farkındalıklı hafıza rafine
  etmesi yok.
- **Çok nesneli SAM2Long/SAMURAI**: hafıza kapısı kare başına ve tüm satırlar
  için ortak, bu yüzden "herhangi bir nesne için kötüyse kare saklanmıyor"
  kuralı kullanılıyor. Tek hedefte sorun değil, kalabalıkta muhafazakâr.
- **`transformers` SAM2 API'si versiyona duyarlı.** `Sam2Teacher.mask_for` tek
  yerde toplandı; API değişirse tek düzenleme.
- **Rapor (`report/`) elle sürülmedi** — senin isteğin üzerine.
