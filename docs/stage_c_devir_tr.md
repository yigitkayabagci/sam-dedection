# Stage C devir notu — 31 numaralı notebook

Bu doküman, `31_aerial_thermal_tracking.ipynb`'i devralacak biri için yazıldı.
İki soruyu cevaplıyor: **Stage C ne yapıyor**, ve **eğittiği veriyi gözle nasıl
kontrol ederim**. Sonunda da sıradaki iş var: 32'nin pretrain tabanlı koşusu
bitince onun çıktısını 31'e vermek.

Repo: `yigitkayabagci/sam-dedection`, dal
`claude/thermal-stage-b-training-43ktcl`. Önce `CLAUDE.md` okunmalı; oradaki
kurallar bu dokümandan üstündür.

---

## 0. Zincirin tamamı

```
34_pretrain_thermal_aerial      geniş havuzda maske eğitimi (adı "pretrain", kendisi Stage B)
        │                                  çıktı: edgetam_pool_pretrain_thermal_aerial_512.pt
        ▼
32_aerial_thermal_stage_b_stable  Stage B  tek promptlu KARE üzerinde maske eğitimi
        │                                  BASE_CHECKPOINT = <34'ün çıktısı>
        │                                  çıktı: edgetam_pool_aerial_thermal_stable
        │                                          _from_pretrain_thermal_aerial_512.pt
        ▼
31_aerial_thermal_tracking        Stage C  VİDEO: bellek bankası devrede
                                           BASE_STAGE_B = <32'nin çıktısı>
```

**34 bir Stage A değil.** Adı öyle diyor ama notebook kendi ilk hücresinde
`stage B only: no stage A, no distillation` basıyor: stock EdgeTAM'den
başlıyor, `METHOD="finetune"`, ve etiketli maske havuzlarında eğitiyor. Aynısı
35 (RGB kolu) için de geçerli. `LR_HEAD = 0` kafayı dondurmaz — `train_encoder`
içindeki `scaled_rates` docstring'i yazıyor: *"A zero override means 'leave this
part alone', so the flag's own default is inert"* — yani kafa tablodaki
varsayılanla eğitilir (head aşamasında 1e-4, encoder aşamasında 5e-5). 35'in
`run.json`'undaki `effective_rates` bunu doğruluyor.

Pratik sonucu: zincir A→B→C değil, **geniş Stage B → dar Stage B → Stage C**.
Bu meşru bir müfredat, ama "encoder'ı etiketsiz veriyle ısıttık" diye
okunmamalı; 34'ün çıktısı da maskeyle eğitilmiş bir checkpoint'tir.

**Stage B tek kare ölçer, Stage C ise takibi.** Bu ayrım bu projenin merkezinde:
`image_loop.instance_iou` docstring'i açıkça söylüyor — tek promptlu bir kare
"bu projenin var olma sebebi olan arızayı göremez, o yalnız bellek bankası
devredeyken ortaya çıkar". Stage B'nin mean IoU'su yükselirken takip
bozulabilir, ve bunun olduğuna dair bir gözlem var (bkz. §4).

---

## 1. Stage C'de ne eğitiliyor, ne eğitilmiyor

`src/training/finetune.py`:

```python
FROZEN_MODULES = ("memory_attention", "memory_encoder", "spatial_perceiver")
FROZEN_PARAMS  = ("maskmem_tpos_enc", "no_mem_embed", "no_mem_pos_enc",
                  "no_obj_ptr", "no_obj_embed_spatial")
STAGES = {"head": HEAD_MODULES, "encoder": HEAD_MODULES + ("image_encoder",), ...}
```

31 `apply_freeze(model, "encoder")` çağırıyor. Yani eğitilen: **image encoder +
sam_mask_decoder + sam_prompt_encoder + obj_ptr_proj**. Eğitilmeyen: **bellek
yolunun tamamı** — ve bu koşullu değil, `FROZEN_MODULES` her aşamada geçerli.

Eğitim bunu tek adımda açmıyor; 19. hücrenin `Schedule`'ı iki aşamalı:

```python
stages=(("head",    2, Rates(head=5e-5)),
        ("encoder", 4, Rates(head=2e-5, neck=2e-5, trunk=5e-6)))
```

Yani önce 2 epoch **yalnız kafa** (image encoder de donuk), sonra 4 epoch
encoder açık ve trunk 5e-6 gibi çok düşük hızda. Toplam 6 epoch, `patience=2`.
Trunk'ın bu kadar yavaş olması §4'teki mekanizma B'ye karşı örtük bir frendir —
ama ölçülmüş bir fren değil.

Donmuş olmasının repoda yazılı üç gerekçesi var (aynı dosyanın modül
docstring'i):

1. Bellek encoder'ı özyinelemeli bir döngünün yazma portu. Ölçüldü: yalnız onu
   quantize etmek mean IoU 0.9397 verdi ve hasar diğer modüllerin aksine
   **klip boyunca sürekli düşüş** şeklindeydi.
2. ONNX yeniden yazımı en karmaşık olanı; ağırlıkları eğitmek checkpoint-engine
   uyuşmazlığı davet ediyor.
3. "Soyut özellikler üzerinde çalışır, piksel üzerinde değil. Termal-RGB bir
   *encoder* problemidir; bellek yolunun buna ihtiyaç duymasını beklemek için
   bir sebep yok."

**3. gerekçe şu an tartışmalı** — §4'e bakın.

### Stage C'nin Stage B'de olmayan şeyi

`clip_loop` gerçek deployment yolundan geçiyor: teacher forcing yok, her kare
modelin kendi önceki tahminini belleğe yazıyor. Loss ağırlıkları (31, 16. hücre):

```python
TRACK_WEIGHTS = Weights(focal=20.0, dice=1.0, iou=2.0,
                        object_score=2.0, box_projection=1.0)
```

`object_score` burada **2.0** ve gerçek `exist` etiketine karşı denetleniyor.
Stage B'de bu terim `instance_loss`'tan **kasten** çıkarılmış, çünkü statik
sette `exist` etiketi yok — `losses.py` gerekçeyi yazıyor: sabit 1'e karşı BCE,
kafaya koşulsuz ateşlemeyi öğretir, ve `object_score_logits` negatife düştüğünde
EdgeTAM bankaya `no_obj_ptr` yazıp sonraki kareler onu geri okuduğu için yanlış
ateşleyen bir kafa kendi belleğini zehirler.

Maske denetimi kare kare karar veriliyor: `STORES[dizi_adı][kare_no]` varsa
mask loss, yoksa box-projection + object-score. Teacher'ın reddettiği maske
"boş hedef" sayılmıyor.

---

## 2. Stage C hangi veriyi kullanıyor

31'in 8. hücresi, tam liste. Her kaynağın bir `USE_*` anahtarı var ve
**varsayılan kol artık yalnız vtuav_vis**:

| kaynak | ne | denetim | `SOURCE_WEIGHTS` | varsayılan |
|---|---|---|---|---|
| **vtuav** — VTUAV ST/LT, `tracked_ir` | UAV üstünde termal tek-hedef tracking | kutu; maske yalnız 17/25 teacher havuzu `box_iou ≥ 0.80` ile eşleşen karelerde | 0.40 | `USE_VTUAV=False` |
| **vtuav_vis** — VTUAV-VIS, `--frames masked` | resmi instance maskeleri, ~her 30. kare | **gerçek maske** | 0.30 | `USE_VTUAV_VIS=True` |
| **birdsai** — `split="TrainReal"` | gece TIR, gerçek track id | kutu | 0.30 | `USE_BIRDSAI=False` |
| **antiuav410** | yer kamerası, yukarı bakıyor | — | 0.10 | `USE_ANTIUAV410=False` |

`weighted_clip_sample` kaynağı olmayan ağırlığı düşürüp kalanları yeniden
normalize ediyor (`aerial_video.py:469-473`), yani kolu `SOURCE_WEIGHTS` değil
bu anahtarlar seçiyor: yalnız vtuav_vis açıkken payı %100 olur.

`VTUAV_VIS_PARTS` sekiz resmî arşivin hepsini sayıyor (`train_001..003`,
`test_001..005`). **Her parça kendi alt klasörüne açılıyor.** VTUAV dizileri
hedefin türüne göre adlandırılmış — `train_003` üçüncü *tren* videosudur,
train split'i değil — ve bu yüzden `test_001.zip` içinden `train_003/` çıkar.
Sekizi tek klasöre açmak, iki arşiv aynı ada rastlarsa kareleri diskte
birleştirirdi. Ayrıca `STORES` ada göre anahtarlı olduğu için çakışan iki dizi
maskeleri gölgeler; 8. hücre bunu ad tekrarı üzerinden assert ediyor.

### Yalnız vtuav_vis koşmanın iki bedeli

Bunlar ölçülmüş yapısal özellikler, ayar değil:

1. **Hiç kayboluş yok.** `--frames masked` yalnız maskeli kareleri ve
   eşleniklerini çıkarıyor (`fetch_datasets.masked_members`) ve maskeli kare
   demek hedefin orada olduğu kare demek. `ir.txt` bu unpack'te hiç
   çıkarılmıyor. Yani `exist` sabit True'dur ve bunu değiştirmenin yolu yok.
   `object_score` ağırlığı bu yüzden ölçülen `absent` sayısına bağlandı; sıfırsa
   terim kapanıyor.
2. **Kareler ~30 kaynak kare arayla.** `CLIP_LEN = 8` olduğu için bir klip
   ~240 kaynak kareye yayılıyor: bellek bankası 12 saniyelik sıçramalarla
   eğitiliyor, deployment'taki hareketle değil. `vtuav_vis_sequences`
   docstring'i bunu zaten söylüyor: *"not a substitute for the denser tracking
   split"*.

Gerçek kayboluş denetimi isteniyorsa tek yolu `USE_VTUAV=True`.

Anti-UAV410 kapalı olması bir tercih değil, `CLAUDE.md`'nin kuralı: kamerası
yerde, hedefi havada — bu projenin (dronedan aşağı bakan) tam tersi
perspektifi. Açmayı önermeyin.

VTUAV arşivleri (`VTUAV_ARCHIVES`): `train_ST_001`, `train_ST_008`,
`train_LT_001..004`. Bilinen çıkarma sayıları: 3790 + 4092 + 5616 + 5059 +
4233 + 4074 = **26 864 kare**.

### En önemli bulgu: kaybolma denetimi tek kaynaktan geliyor

`src/training/aerial_video.py` içinde `exist=` yalnız üç yerde geçiyor:

| fonksiyon | satır | `exist` |
|---|---|---|
| `vtuav_sequences` | 99 | `ir.txt`'ten okunuyor — **gerçek kaybolma** |
| `vtuav_vis_sequences` | 215 | `np.ones(...)` — her çıkarılan kare maskeli, kayboluş yok |
| `birdsai_sequences` | 389 | `np.ones(...)` — track boşluklarda **bölünüyor**, işaretlenmiyor |

Yani `object_score` terimi (ağırlık 2.0) yalnız `exist=False` karelerinde bir
şey öğrenebilir, ve o kareler **sadece %40'lık vtuav kolunda** var. Kalan %60
o kafaya hiçbir şey öğretmiyor. Bu, §4'teki soruyla doğrudan ilgili.

---

## 3. Veriyi gözle inceleme

`tools/inspect_stage_c.py`. 31'de varsayılan **açık**:

```python
INSPECT_DATA = True        # 4. hücre
INSPECT_PER_SOURCE = 4
INSPECT_SPAN = 12
INSPECT_WINDOWS = 2
```

10. hücrede, `STORES` kurulduktan hemen sonra çalışır ve **Drive'a** yazar:
`edgetam-stage-c/aerial_thermal_tracking_from_<taban>/inspect/<kaynak>/`.

Notebook dışından da koşar:

```bash
python tools/inspect_stage_c.py \
    --vtuav /content/data/VTUAV_aerial_stage_c \
    --vtuav-vis /content/data/VTUAV_VIS_aerial_stage_c \
    --birdsai /content/data/BIRDSAI \
    --pool /content/pool/aerial_stage_c \
    --out /content/inspect
```

### Sayfalarda ne görülüyor

- 🟡 **sarı maske/kutu** — bu karede gerçek maske var, mask loss orada çalışıyor
- 🟢 **yeşil kutu** — maske yok, denetim box-projection
- 🔴 **kırmızı çerçeve** — `exist` False; **kutu hiç çizilmiyor**, çünkü son
  bilinen kutuyu çizmek anotasyonun "yok" dediği bir hedefi varmış gibi
  gösterirdi

### İki tasarım kararı ve gerekçeleri

**Pencereler `exist` değişimlerinin etrafına konuyor.** 1500 karelik bir dizide
uniform örnekleme kareleri sıradan takipte harcar ve genelde hiçbir kayboluş
göstermez. Araç kaybolma ve geri gelme anlarını ayrı pencereler olarak alır,
her kaynakta en çok geçiş içeren dizileri öne çıkarır, hiç geçişi olmayan dizi
uniform örneklemeye düşer (atlanmaz).

**Maske ile kutu ayırt ediliyor.** Bir koşu kare sayısında tamamen denetimli
görünüp tam da önemli karelerde box-only olabilir. `index.json` her dizi için
`masks` sayısını yazar.

`index.json` ayrıca kaynak başına **absent** (kaybolma) sayısını yazar — §2'deki
bulguyu her koşuda tekrar ölçer. Hiçbir kaynakta absent yoksa araç bunu uyarı
olarak basar.

Görsel bir kontrol **ölçüm değildir**: 2/98 germe bakmak içindir,
`aerial.load_image` eğitimde farklı normalize eder.

---

## 4. Açık soru: stock, kaybolup gelmede daha iyi

Kullanıcının gözlemi: hedef kaybolup geri geldiğinde **stock EdgeTAM,
fine-tune edilmiş checkpoint'ten daha iyi takip ediyor**. Bu bir izlenim, ve
ölçülmesi gerekiyor — ama iki mekanizma da yapıda mevcut:

**A — `object_score` denetimsiz kaldı.** Stage B `sam_mask_decoder`'ı eğitiyor
(LR_HEAD 1e-4) ama `object_score` terimini vermiyor. "Burada bir şey var mı"
diyen kafanın altındaki özellik dağılımı kayıyor, o çıktıya hiçbir denetim
gelmeden. Stock'unki en azından kendi özellikleriyle tutarlı.

**B — donmuş bellek, kaymış encoder.** `memory_attention` stock encoder
özelliklerine karşı eğitildi ve donuk. Stage B/C encoder'ı o dağılımdan
uzaklaştırıyor; bellek eğitilmediği özellikleri okuyor. Encoder ne kadar çok
hareket ederse uyumsuzluk o kadar büyür — **tek-kare IoU yükselirken tracking
bozulur**. Bu, §1'deki 3. gerekçeyle doğrudan çelişir.

### Ölçüm hazır: iki kollu precheck

31'in 12. hücresi eğitimden önce, düşük kontrastlı validation dizilerinde
`state_accuracy`, `success_auc`, `lost_rate`, `longest_dropout` ölçer. Artık
**aynı segmentler üzerinde iki kol** koşuyor:

```python
PRECHECK_AGAINST_STOCK = True      # 4. hücre, varsayılan True
STOP_AFTER_PRECHECK    = False     # 4. hücre, varsayılan False
```

`STOP_AFTER_PRECHECK`'in varsayılanı **`False`**'tur, yani notebook precheck'ten
sonra eğitime devam eder. Yalnız audit isteyen bir koşuda bu satırı elle `True`
yapın; o zaman 12. hücre raporu yazdıktan sonra `SystemExit` ile durur.

Segmentler iki tracker kurulmadan **önce bir kez** hesaplanır, yani karşılaştırma
birebir aynı karelerde. Ortalama delta basılır; stage_b stock'tan kötüyse açık
bir uyarı ve yukarıdaki mekanizma adıyla yazılır. Kapının kendisi
(`PRECHECK_RECOMMEND_STAGE_C`) hâlâ stage_b kolunu okur — onun kararı "Stage C
gerekli mi", stock kıyası değil. Rapor: `MIRROR/stage_c_precheck.json`
(`arm_rows`, `deltas`, `stock_wins`).

**Bu ölçüm eğitim gerektirmiyor, bir GPU geçişi.** Yeni sohbetin ilk işi bu
olmalı.

İki uyarı, sonucu okurken:

- **Ölçüm 512'de alınıyor**, çünkü 31'in `SIZE`'ı 512. İki kol da aynı boyutta
  koştuğu için *karşılaştırma* adildir, ama çıkan sayılar dağıtım sayısı
  değildir: `CLAUDE.md`'ye göre bir 768 koşusu yalnız 768'deki stock'a karşı
  puanlanır.
- **Örneklem altı dizi** (`PRECHECK_PER_SOURCE = 2` × üç kaynak), her biri 240
  kare. `PRECHECK_STOCK_WINS = _sa < 0 or _lost > 0` bir tolerans bandı
  taşımıyor, yani −0.0001'lik bir ortalama delta da uyarıyı bastırır. Tek
  ortalamaya değil, `arm_rows` içindeki altı satırın dağılımına bakın.

### Eğer stock kazanıyorsa — sıradaki adımlar, ucuzdan pahalıya

1. **`ANCHOR_WEIGHT`** (`tools/train_encoder.py --anchor-weight`): encoder'ın
   aşama başındaki modelden ne kadar uzaklaştığını cezalandırır — mekanizma
   B'yi doğrudan sınırlar, ek forward maliyeti yok.
2. **`NEIGHBOUR_WEIGHT`** (32'de): kalabalıkta maskenin komşuları yutmasını
   cezalandırır. Ayrı bir arıza ama aynı ailede; ağırlık 0'da bile leak
   ölçülüp basılıyor, önce sayıya bakın.
3. **Stage C'de bellek yolunu açmak**: kökü hedefler, ama §1'deki üç gerekçeyi
   üstlenir (özellikle ölçülmüş "klip boyunca sürekli düşüş" riski ve
   ONNX/engine uyuşmazlığı). Yalnız 1 ve 2 denendikten sonra.

---

## 5. Sıradaki iş: pretrain tabanlı Stage B çıktısını 31'e vermek

Kullanıcının hedefi bu. 32, `BASE_CHECKPOINT` dolu olduğunda çıktısının adına
`_from_<pretrain>` eki koyar, yani beklenen yol:

```
/content/drive/MyDrive/edgetam-stage-b/aerial_thermal_stable_from_pretrain_thermal_aerial/
    edgetam_pool_aerial_thermal_stable_from_pretrain_thermal_aerial_512.pt
```

31'de tek satır:

```python
BASE_STAGE_B_OVERRIDE = "<yukarıdaki tam yol>"
```

`MIRROR` ve `CHECKPOINT` tabanın adını otomatik taşır
(`aerial_thermal_tracking_from_<taban>`), yani iki farklı Stage B çıktısından
koşulan iki Stage C birbirini **ezmez**. Eski `thermal_deep` (22'nin çıktısı)
fallback'i listeden kaldırıldı — odak `aerial_thermal`.

### Dikkat: bu koşu bir A/B değil

32'nin yeni koşusu aynı anda **altı** şeyi değiştiriyor: pretrain tabanı,
`MAX_AREA` 0.2→0.06, 12→30 epoch, patience 0→10, LR merdiveni (`thermal`
çıkarıldı — 12 epoch'luk koşu pilotta onu seçmiş ve encoder validation'ı
0.1908'de dip yapıp epoch 9'da 0.2721'e fırlamıştı), ve temizlenmiş SegFly.
Sonuç iyi çıkarsa hangisinin yaptığı ayrılamaz. Bu bilinçli bir karar;
`tools/compare_stage_a.py` farklı `gates`/`neighbour_weight` taşıyan kolları
sıralamayı reddeder, yani yanlışlıkla haksız bir sıralama üretilmez.

---

## 6. Bilinen tuzaklar

- **Drive mount düşmesi** (`OSError: [Errno 107] Transport endpoint is not
  connected`): VTUAV parçaları başkasının Drive'ındaki kısayollar ve
  `tracked_ir` filtresi binlerce rastgele seek yapıyor. 31 artık düşünce
  yeniden bağlanıp thread sayısını azaltarak (16→8→4→1) tekrar deniyor. Tek
  thread'de de tutmazsa parçayı yerel diske kopyalayın.
- **VTUAV-VIS quota**: `train_001/002/003` 8.5/15/17 GB. `fetch_datasets.py`
  artık bir parça reddedilince diğerlerini denemeye devam ediyor ve neyin
  indiğini söylüyor; 31 de parça parça çekip `.done` işareti bırakıyor, yani
  retry sadece eksik parçaya mal oluyor. Yine reddedilirse zip'i Drive'da
  "Bir kopyasını oluştur" ile `MyDrive/datasets/` altına alın — `staged()` onu
  ağa gitmeden bulur.
- **`assert all(SPLITS.values())`**: bir kaynak hiç inmemişse split boş kalır
  ve burada durur. Mesajı okuyup eksik veriyi tamamlayın, assert'i gevşetmeyin.
- **`MIN_TEACHER_MASK_FRAMES = 1000`**: teacher havuzu eşleşmesi bunun altında
  kalırsa 31 durur. Box-only başlamak yerine `edgetam-pool/vtuav_*` arşivlerini
  kontrol edin.

---

## 7. Bu repo'nun kendine uyguladığı kurallar

Yeni sohbetin bunlara uyması bekleniyor (`CLAUDE.md`):

- **İddia etmeden önce ölç.** `docs/` içindeki her eşik geldiği sayıyı
  adlandırıyor. "HIT-UAV hedefleri soğuk zemine karşı sıcaktır" varsayıldı,
  sonra medyan SCR **0.91** ölçüldü ve docstring'ler düzeltildi.
- **Gerçek hedefe ateş eden bir kapı, kapısızlıktan kötüdür.**
- **Üretilen notebook'lar `tools/build_*.py`'den gelir — `.ipynb`'yi asla elle
  düzenlemeyin**, builder'ı düzenleyip yeniden üretin.
- `22_thermal_deep_3_fixed.ipynb` kullanıcının kendi korunan dosyası; yeniden
  üretilmez.
- **Türkçe** çalışma dili: `docs/*_tr.md` ve cevaplar.
- Dağıtım hedefi **1280x768 → modele 768**. Bir 768 koşusu 768'deki stock'a
  karşı puanlanır; 512 sayısı asla onun baseline'ı değildir.
