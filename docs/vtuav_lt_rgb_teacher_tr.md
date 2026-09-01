# VTUAV-LT'yi SAM 3 ile maskelemek — ne zaten var, hangi kapı hangi şartı karşılıyor, RGB için ne eksik

Soru şuydu: VTUAV-LT arşivleri kare başına yalnız **kutu** taşıyor (`rgb.txt` /
`ir.txt`), maske yok. Bunları SAM 3 ile maskeleyip notebook 37'nin (RGB Stage C)
verisine katalım — ama üç şartla: (a) sadece tahmine gerçekten güvendiğimizde,
"0.8 kutu doğruluğu" gibi bir bar; (b) **çok büyük** maskeler elensin, hedef
küçük/orta nesne takibi; (c) veriyi kötüleştirmesin.

Kısa cevap: **bu makine zaten kurulu ve büyük ölçüde koşulmuş.** Aşağıda her
şeyin gerçek adı, gerçek varsayılanı ve kodda nerede olduğu var. Eksik olan
şey hasat değil, **notebook 37'nin bu havuzu hiç okumaması**.

---

## 1. Zaten var olan

### 1.1 İki ayrı SAM 3 yolu — ikisi de bağlı

| yol | sınıf | ne yapar | sürücü |
|---|---|---|---|
| **görüntü** öğretmeni | `Sam3Teacher` — `src/training/labels.py:387` | kare başına bağımsız: kutunun etrafında zoom-crop → tek maske + `iou_scores` | `src/training/pool.py:label_pool`, `tools/make_mask_pool.py`, **notebook 24** |
| **video** öğretmeni | `Sam3VideoTeacher` — `src/training/masklets.py:369` | parçanın ilk karesine kutu prompt'u, maskeyi sonraki karelere yayar | `src/training/masklets.py:masklet_sequence`, `tools/make_masklets.py` |

İkisinde de sınıf **model id'sinden** seçilir, ayrı bir bayrak yok:
`labels.py:432` ve `masklets.py:410` → `"sam3" in model_id.lower()`.

**Gereksinimler** (ikisi için de aynı, `labels.py:414-421` ve
`masklets.py:393-401`): `transformers>=5.0` (`Sam3TrackerModel` /
`Sam3TrackerVideoModel` orada geldi), **gated repo** — `facebook/sam3` model
sayfasının bir kez kabul edilmesi ve `HF_TOKEN` (`build_image_teacher`'ın hata
metni bunu yazar, `labels.py:446-454`), GPU + `bfloat16`. Yedek öğretmen
`facebook/sam2.1-hiera-large`: gated değil, Apache-2.0.

### 1.2 Kutu → maske tam olarak nasıl oluyor (görüntü yolu)

1. `labels.zoom_window` (`labels.py:49`) kutunun uzun kenarının `zoom=4.0`
   katı, en az `min_size=128` px kenarlı **kare crop** çıkarır. Sebep: 20 px'lik
   bir hedefi 1920x1080 karede segmentlemek zor, aynı hedefi 128 px'lik crop'ta
   kolay.
2. `_ImageTeacher.mask_for` (`labels.py:287`) crop'u ve **crop koordinatlarına
   taşınmış kutuyu** processor'a verir, `post_process_masks` ile maskeyi geri
   alır, `iou_scores`'un en yükseğini seçer → `(mask, teacher_iou)`.
3. `labels.measure` (`labels.py:148`) dört okuma üretir, `labels.reject_reason`
   (`labels.py:112`) **sabit sırayla** ilk düşen kapıyı döndürür.
4. Kabul edilen maske tam kare koordinatına geri yerleştirilip RLE olarak
   `pseudo_masks.npz`'e, kutu/sınıf/dört okuma `record.json`'a yazılır
   (`pool.py:462-506`).

### 1.3 VTUAV-LT için hazır uçtan uca sürücü: **notebook 24**

`tools/build_vtuav_pool_notebooks.py:251-255` şu varyantı üretir:

```
notebooks/24_vtuav_lt_rgb_pool.ipynb
  POOL         = "vtuav_lt_rgb"        MODALITY = "rgb"
  EXTRACT_MODE = "tracked_rgb"         DATA_ROOT = /content/data/VTUAV_lt_rgb
  ARCHIVES     = train_LT_001..004.zip
  MIRROR_DIR   = /content/drive/MyDrive/edgetam-pool/vtuav_lt_rgb
  TEACHER      = "facebook/sam3"       BOX_IOU = 0.6   ZOOM = 4.0   MIN_SIZE = 128
```

(notebook dosyasının 1. hücresi; builder'da `:240-261` ve `:280-306`.)

Yani **"VTUAV-LT'yi SAM 3 ile RGB'de maskele" için yazılması gereken tek satır
kod yok** — notebook 24 tam olarak bu. Kardeşi 25 aynı işi `ir` yarısı için
yapar; 16/17 ise short-term (ST) parçaları içindir.

`tracked_rgb` çıkarımı arşivin yalnız etiketli karelerini açar: kare kimlikleri
0..n-1 boşluksuz ve kutu dosyası `ceil(n/10)` satır taşıdığı için **satır k =
kare 10k**, 15,4 GiB'lık parça diskte ~0,8 GiB'a iner
(`docs/datasets.md:383-389`, `boxes.vtuav_frames` docstring `boxes.py:600-616`).

### 1.4 Video (masklet) yolu VTUAV-LT'de neden pratikte kapalı

`masklets.find_sequences` **strided** bir ağacı okumayı reddeder
(`masklets.py:138-153` ve `:191-198`): `tracked_rgb` ile açılmış bir ağaçta her
10. kare vardır, konuma göre eşleme her kutuyu 10 kat kaydırır. Yani video
öğretmenini VTUAV-LT'de koşmak **arşivi tam açmak** demek (parça başına
15,4 GiB, on katı disk ve on katı teacher-saati). Bu yüzden pratik yol görüntü
öğretmeni + havuz (notebook 24), video masklet'i değil.

---

## 2. Kapılar: hangisi hangi şartı karşılıyor

Kapılar **iki ayrı zamanda** uygulanır ve karıştırılmamalı:

| ne zaman | nerede | kapılar |
|---|---|---|
| **hasat** (teacher koşarken, geri alınamaz) | `labels.reject_reason` (`labels.py:112`), `pool.label_pool` üzerinden | `teacher_iou`, `box_iou`, `area`, `component` |
| **okuma** (eğitim indeksi kurulurken, GPU'suz, tekrar kesilebilir) | Stage B: `pool_reader.index_pool` (`pool_reader.py:1000`, kapı `:1107`) · Stage C: `aerial_video.pool_sequence_stores` (`aerial_video.py:240`) | `min_box_iou` (ikisinde de), `InstanceGates` (**yalnız Stage B**) |

### 2.1 Şart (a) — "0.8 kutu doğruluğu"

Bu şartın gerçek karşılığı **`box_iou`**, ve **kullanıcının 0.8'i zaten var olan
bir varsayılan**:

- `TEACHER_MIN_BOX_IOU = 0.80` — notebook 31'in ayar hücresi,
  `tools/build_aerial_tracking_notebook.py:136`. Stage C'de teacher havuzu
  okunurken uygulanır: `:458-461`.
- `MIN_BOX_IOU = 0.8` — Stage B kolları 22/27/28,
  `tools/build_stage_b_notebooks.py:350`, `:379`, `:549`. Pretrain kolları
  34/35'te `0.75` (`:428`, `:473`).
- `tools/inspect_stage_c.py:250` da `--min-box-iou` için `0.80` varsayıyor.
- Hasat sırasındaki taban ise `Gates.box_iou = 0.6` (`labels.py:99`), notebook
  24 de bunu kullanıyor (`BOX_IOU = 0.6`). Builder bu seçimi gerekçelendiriyor
  (`build_vtuav_pool_notebooks.py:71-87`): 0.5 DroneVehicle'ın yönlü
  kutularından gelen başkasının sayısıydı, VTUAV'ın kutuları eksen hizalı, o
  yüzden kütüphane varsayılanı 0.6'da bırakıldı.

**Neyi ölçtüğü önemli.** `box_iou`, maskenin **çevreleyen kutusu** ile insanın
çizdiği kutu arasındaki IoU'dur (`labels.py:152-157`, `box_from_mask` +
`box_iou`). Yani "maske doğru mu" değil, "maskenin kapladığı alan
annotation'ın gösterdiği yerde ve o büyüklükte mi". Maske-vs-gerçek-maske IoU'su
**değildir** — bunu ölçebilecek tek şey çizilmiş maske, o da VTUAV-LT'de yok.

**Hasat 0.6, okuma 0.8: neden ikisi birden var.** `record.json` her instance için
dört okumayı da saklar (`pool.py:488-497`, `GATE_READINGS` `pool.py:612`), bu
yüzden 0.8'e sıkmak teacher'ı yeniden koşmak değil, indeks üzerinde bir geçiş
(`pool_reader.index_pool(min_box_iou=...)`, docstring `:1029-1035`). **Yalnız
yukarı doğru**: 0.6'da hasat edilmiş havuz 0.5'in ne tutacağını bilmiyor.

**Teacher'ın kendi güveni bu şartın cevabı değil.** `Gates.teacher_iou = 0.7`
(`labels.py:98`) SAM'ın kendi `iou_scores`'udur ve `Gates`'in kendi docstring'i
(`labels.py:81-90`) bunun **ölçülmüş olarak zayıf** olduğunu yazar: sentetik
12-36 px hedeflerde SAM 2'nin kendi skoru ortalama **0.92-0.94** iken gerçek IoU
**0.56-0.67** — yani **+0.25 ila +0.38 aşırı güven**, ve büyük checkpoint'lerde
daha kötü. Kullanıcının "0.8"ini `teacher_iou`'ya bağlamak bu yüzden yanlış
olurdu; doğru yer `box_iou`.

**Ve 0.8'in bir bedeli var, ölçülmüş bir yönde.** `pool.gate_report` docstring'i
(`pool.py:713-725`) şunu söyler: bir box-IoU eşiği **en çok küçük hedefi**
vurur — 20 px'lik bir nesne ile etrafına çizilen dikdörtgen arasındaki birkaç
piksellik boşluk, aynı boşluğun 200 px'lik nesnede maliyetinden çok daha fazla
IoU götürür. Yani sınıf-nötr görünen bir kesim, aslında bir **küçük hedef
filtresidir** — ki bu projenin yargılandığı eksen tam olarak küçük hedef. Şart
(a) ile şart (b) burada birbirine karşı çalışıyor. Karar, tahmin edilerek değil,
her havuz için `summarise_gates`'in bastığı 0.5/0.6/0.7/0.8/0.9 tablosuna
bakılarak verilmeli (notebook 24 hücre 5 bunu zaten basıyor).

### 2.2 Şart (b) — "çok büyük maskeler elensin"

Burada **iki farklı "büyük"** var ve yalnız biri kullanıcının kastettiği:

**(i) Kutusuna göre büyük** — `Gates.area = (0.15, 1.3)` (`labels.py:100`),
hasat sırasında. Ölçtüğü şey `area_ratio` = maske pikselleri / kutu alanı
(`labels.py:154`). Alt sınır parça-maskeyi, üst sınır **arka plana taşmayı**
yakalar. Bu bir *mutlak boyut* kapısı değildir: kareyi dolduran bir hedefin
maskesi kutusunu tam doldurduğu sürece `area_ratio ≈ 1.0` verir ve bu kapıdan
sorunsuz geçer.

**(ii) Kareye göre büyük** — kullanıcının istediği bu, ve adı
`InstanceGates.max_area` (`src/training/aerial.py:1218`), `reject_reason`
içinde `instance.area > gates.max_area * frame_area` olarak uygulanır
(`aerial.py:1459-1460`).

Varsayılanlar ve koşulan değerler:

| ayar | `min_area` | `min_side` | `max_area` | `fill` | nerede |
|---|---:|---:|---:|---:|---|
| kütüphane varsayılanı | 48 | 4 | 0.9 | 0.25 | `aerial.py:1216-1219` |
| `PLAIN` preset | 48 | 4 | 0.9 | 0.25 | `build_stage_b_notebooks.py:202` |
| **`HARDER` preset** (19/20/22/23/27/28/34/35) | **64** | **6** | **0.2** | 0.25 | `build_stage_b_notebooks.py:224` |

`HARDER` doğrudan kullanıcının kendi cümlesinden geldi:
`docs/calisma_plani_tr.md:16` — *"ekranın %50'si gibi kutuları çıkaralım"* →
`MAX_AREA`; `:18` — *"tiny ve huge değil, small/medium"* → `MIN_AREA 64`,
`MIN_SIDE 6`. (Not: `docs/son_degisiklikler_tr.md:284-287` bu değeri **0.25**
diye yazıyor, kodda **0.2**. Koşan sayı koddaki 0.2'dir; o dosya bu çalışmanın
kapsamı dışında olduğu için düzeltilmedi.)

**Ama kritik boşluk:** `InstanceGates` **yalnız Stage B havuz indeksinde**
uygulanıyor (`pool_reader.py:1107`). Stage C'nin havuz yolu
`aerial_video.pool_sequence_stores` (`aerial_video.py:240-303`) **yalnız
`verdict is None` ve `min_box_iou`'ya bakar** — `min_area`, `min_side`,
`max_area`, `fill`'in hiçbiri orada yok. `src/training/clip_loop.py` ve
`video_clips` içinde de boyut filtresi yok (arandı, bulunamadı).

Yani: **bugünkü hâliyle Stage C'de (31 ve 37) hiçbir boyut kapısı yoktur.**
Şart (b) Stage B'de karşılanıyor, Stage C'de karşılanmıyor. VTUAV tracking'in
kare başına tek hedefi olduğu ve hedeflerin medyanı √alan **76,7 px**
(`%98,5'i ≥32 px`, `docs/datasets.md:399`) olduğu için pratikte "kareyi dolduran
maske" beklenmiyor — ama bu bir **beklenti**, uygulanan bir kapı değil.

### 2.3 Şart (c) — "veriyi kötüleştirmesin"

Yapısal güvence gerçek ve ölçülebilir:

- **Reddedilen kare boş maske değil, maskesizdir.** `label_pool` yalnız kabul
  edilenleri store'a yazar; okurken eksik anahtar "burada maske denetimi yok"
  demektir. `antiuav.clip_masks` (`antiuav.py:384-409`) o kareye `None` koyar ve
  eğitim döngüsü kutu terimine düşer — *"boş maske (burada bir şey yok)
  demekten farklı bir şey"*, docstring'in kendi ifadesiyle.
- **Sıkılaştırma geri dönüşlü, gevşetme değil.** Dört okuma kayıtta durduğu için
  0.6 → 0.8 bir indeks geçişi; 0.6 → 0.5 imkânsız (`pool_reader.py:1029-1035`).
- **Teacher karıştırmak yasak.** Builder bir fallback teacher'ı bilerek
  kaldırdı (`build_vtuav_pool_notebooks.py:143-150`): havuzlar tek eğitim
  setinde karışacağı için komşusundan farklı bir öğretmenle üretilmiş havuz,
  kimsenin seçmediği ve koşunun göremeyeceği bir değişkendir.
- **Eğitimden önce göz denetimi var.** `tools/inspect_stage_c.py` kontakt
  sayfaları basar; 31/37 `INSPECT_DATA = True` ile bunu çağırır.
- **Sert alt sınır.** 31 hücre 10'da `MIN_TEACHER_MASK_FRAMES = 1000`
  (`build_aerial_tracking_notebook.py:137`, assert `:463-466`): eşleşme bu
  sayının altında kalırsa koşu **durur**, sessizce box-only başlamaz.

Karşı tarafta duran gerçek: **teacher maskesi insan maskesi değildir**
(`docs/thermal_contrast_tracking_gelisim_raporu_tr.md:683`). 0.8 filtresi ve
1000-kare assert'i tam olarak bu yüzden kondu.

---

## 3. VTUAV-LT **RGB** için tam olarak ne eksik

Önce yanlış varsayımı düzeltelim: **RGB yolu eksik değil.** `vtuav_rgb`
(notebook 16) ve `vtuav_lt_rgb` (notebook 24) havuzları tanımlı, isimlendirilmiş
ve Drive hedefleri belli. `boxes.vtuav_frames` `modality="rgb"`'yi zaten alıyor
(`boxes.py:598`), `aerial_video.vtuav_sequences` de (`aerial_video.py:63`,
`:76-77`). Modalite argümanı, glob, havuz adı — **üçü de var**.

Gerçekten eksik olan iki şey:

### 3.1 Hasat koşulmamış olabilir (veri işi, kod işi değil)

`docs/` yalnız **termal** havuzların Drive'da doğrulandığını kaydediyor:
`edgetam-pool/vtuav_lt_thermal` ve `vtuav_thermal`
(`thermal_contrast_tracking_gelisim_raporu_tr.md:463-495`). `vtuav_lt_rgb` /
`vtuav_rgb` için hiçbir doğrulama kaydı yok — **ölçülmedi**, var olup olmadığı
bu depodan bilinemez. Yapılacak iş: notebook 24'ü koşmak (ya da Drive'da
`edgetam-pool/vtuav_lt_rgb/vtuav_lt_rgb.zip` + `pool_index.jsonl` var mı diye
bakmak). Kod değişikliği: **yok**.

### 3.2 Notebook 37 bu havuzu hiç okumuyor — asıl eksik bu

37 bilinçli olarak tek kaynaklı: yalnız VTUAV-VIS, yalnız yazarların çizdiği
maskeler (`tools/build_rgb_tracking_notebook.py:5-13`). 31'den türetilirken
teacher havuzu yolu **çıkarıldı**: `:429-433` `teacher_mask_frames` ve
`teacher_min_box_iou` alanlarını kayıtlardan siliyor, `:399-405` ise 31'in
havuz bloğunu içeren hücreyi `CONTRAST = {row.name:` satırından kesip yerine
tek kaynaklı `CLIPS`'i koyuyor.

VTUAV-LT RGB'yi 37'ye sokmak için gereken **tam düzenlemeler** (hiçbiri
yapılmadı; hepsi `tools/build_rgb_tracking_notebook.py` içinde, `.ipynb`'de
değil):

1. **`SETTINGS` bloğu** (`build_rgb_tracking_notebook.py:103-189`) — ekle:
   `VTUAV_ARCHIVES = ["train_LT_001.zip" … "train_LT_004.zip"]`,
   `VTUAV_DATA = DATA / "VTUAV_lt_rgb_stage_c"`,
   `TEMPORAL_POOL_ROOT = Path("/content/pool/aerial_rgb_tracking")`,
   `VTUAV_DRIVE`, `POOL_DRIVE = Path("/content/drive/MyDrive/edgetam-pool")`,
   `TEACHER_MIN_BOX_IOU = 0.80`, `MIN_TEACHER_MASK_FRAMES = 1000`; ve
   `SOURCE_WEIGHTS`'e `"vtuav"` payı (31'de 0.40, `:126`).
2. **`FETCH` bloğu** (`:201-226`) — 31'in iki bloğunu RGB'ye çevirerek ekle:
   arşiv çıkarımı `extract(archive, VTUAV_DATA, frames="tracked_rgb")`
   (31'de `frames="tracked_ir"`, `:261`) ve havuz zip'lerinin açılması
   (31: `:290-307`), pool adı **`"vtuav_lt_rgb"`** (31'de
   `("vtuav_thermal", "vtuav_lt_thermal")`, `:292`).
3. **`SPLIT` bloğu** (`:240-276`) — import'a `pool_sequence_stores` ve
   `vtuav_sequences` ekle; `vtuav = vtuav_sequences(VTUAV_DATA,
   modality="rgb")` ve `sequences = vtuav + train_sequences + held_sequences`.
   Dikkat: `split_flights(..., hold_out=held_sequences)` çağrısına VTUAV
   dizilerinin de girmesi gerekir, yoksa hiçbir split'e düşmezler.
4. **`CLIPS` bloğu** (`:278-339`) — `STORES = empty_stores(sequences)`'tan
   sonra, `STORES.update(VIS_STORES)`'tan **önce**:
   ```
   TEACHER_STORES = pool_sequence_stores(
       TEMPORAL_POOL_ROOT, vtuav, {"vtuav_lt_rgb"},
       min_box_iou=TEACHER_MIN_BOX_IOU)
   ```
   ve 31'deki `MIN_TEACHER_MASK_FRAMES` assert'i (`:462-466`).
   Ayrıca `make_clips` burada tek kaynak varsayıyor (`:300-302`); 31'in
   kaynak-başına `SOURCE_STRIDES` döngüsüne (`:436-446`) dönmek gerekir.
5. **`tools/inspect_stage_c.py`** — iki satır: `:266` `modality="ir"` sabit
   (yorumu da "VTUAV tracking her hâlükârda termal okunur" diyor) ve `:283-284`
   havuz adları `{"vtuav_thermal", "vtuav_lt_thermal"}` sabit. RGB için bir
   `--vtuav-modality` bayrağı ve havuz adlarının parametreleşmesi lazım.
6. **Kayıt alanları** — `:411-433`'teki `replace(...)` çağrıları
   `teacher_mask_frames` / `teacher_min_box_iou` alanlarını sildiği için, bu
   alanlar geri gelecekse o replace'ler de güncellenmeli (aksi hâlde builder
   `RuntimeError: expected 1 occurrence(s)` ile durur — `:47-58`).

**`src/` içinde değişiklik gerekmiyor.** `pool_sequence_stores` modaliteden
bağımsız: tek koşulu `source_name(sequence) == "vtuav"` (`aerial_video.py:258`),
ve `vtuav_sequences` varsayılan `prefix="vtuav"` ile RGB dizilerine de aynı adı
verir.

### 3.3 Bu yolun **sessiz** tuzağı — mutlaka `datasets=` verin

`pool_sequence_stores`'un eşleştirme anahtarı `(flight, frame.stem)`
(`aerial_video.py:262-264`). VTUAV'ın RGB ve IR yarıları **aynı dizi adı ve aynı
kare adını** taşır (`bus_017/rgb/000000.jpg` ve `bus_017/ir/000000.jpg`). Yani
`datasets` argümanı **`None` bırakılırsa** (varsayılanı budur,
`aerial_video.py:243`) ve aynı `TEMPORAL_POOL_ROOT` altında termal havuz da
açılmışsa, **termal maskeler RGB karelere iliştirilir** ve hiçbir yerde hata
vermez. VTUAV'da iki modalitenin kutuları satırların yalnız **%12,2**'sinde
birebir aynı, merkez farkı medyan **8,4 px** (`docs/datasets.md:394-399`) — yani
bu sessizce yanlış bir denetim olurdu. 31 bu yüzden havuz adlarını açıkça
veriyor (`build_aerial_tracking_notebook.py:458-461`); 37'de de `{"vtuav_lt_rgb"}`
açıkça yazılmalı.

---

## 4. Öğretmen kalitesi hakkında **ölçülmüş** olan ne

Uydurulmuş sayı yok; aşağıdaki her satırın kaynağı yazılı.

| ölçüm | değer | kaynak |
|---|---|---|
| `vtuav_lt_thermal` havuzu, kayıt sayısı | 18.123 kare, 43 dizi | `docs/thermal_contrast_tracking_gelisim_raporu_tr.md:470-475` |
| aynı havuz, kabul edilen SAM 3 maskesi | **15.777** | aynı |
| aynı havuz, **hasat kabul oranı** | **%87,1** (`BOX_IOU=0.6` ile) | aynı |
| aynı havuz, sınıf dağılımı | pedestrian 8.418 · car 4.509 · tricycle 1.539 · bus 1.040 · truck 253 · elebike 18 | `:477-486` |
| SAM'ın kendi güven skorunun sapması (sentetik 12-36 px) | skor 0.92-0.94, gerçek IoU 0.56-0.67 → **+0.25…+0.38 aşırı güven** | `src/training/labels.py:81-90` |
| VTUAV hedef büyüklüğü | medyan √alan **76,7 px**, %98,5'i ≥32 px | `docs/datasets.md:399` |
| VTUAV iki modalitenin kutu uyuşması | **%12,2** birebir; merkez farkı medyan **8,4 px**, p90 29,4 | `docs/datasets.md:394-399` |
| `MAX_AREA` kalibrasyonu (AeroVIS üzerinde) | medyan instance 967 px (~31x31), karenin %0,065'i; en büyük car %6,0 · truck %13,6 · bus %4,7; %25 üstü yalnız UAVDT'nin kaba `vehicle` sınıfı, 1.411.468 instance'ın **603'ü (%0,04)** | `tools/build_stage_b_notebooks.py:1409-1418` |
| SAM 3 ↔ SAM 2.1 video farkı (yayınlanan) | SA-V **+5.6**, LVOSv2 **+8.9** J&F (ikisi de SAM 3'ün kendi tablosunda Test-only) | `src/training/masklets.py:372-381`, `docs/encoder_arastirma.md:242` |

### Ölçülmeyenler — "ölçülmedi"

- **`vtuav_lt_thermal`'ın 0.80 kesiminden sonra kaç karesi kaldığı: ölçülmedi.**
  Raporun kendisi bunu söylüyor (`:488-490`): 15.777 sayısı hasat kabulüdür,
  `box IoU ≥ 0.80` sonrası sayıyı notebook koşarken basar.
- **`vtuav_lt_rgb` / `vtuav_rgb` havuzlarının kabul oranı, kare sayısı, sınıf
  dağılımı: ölçülmedi.** Depoda ve `docs/`'ta tek bir sayı yok.
- **Teacher maskesinin çizilmiş maskeye karşı IoU'su (kalibrasyon):
  ölçülmedi.** Mekanizma iki yerde hazır — `masklets.calibrate`
  (`masklets.py:548`, `--calibrate`) VIS'in çizilmiş maskeleriyle, ve
  `make_mask_pool.py calibrate` Kust4K'nın çizilmiş haritalarıyla — ama
  `docs/`'ta çıktısı yok. Yani bugün elimizdeki tek kalite sayısı **kabul
  oranı**, ki `Gates` docstring'inin kendi uyarısıyla *"maskelerin ne kadar kötü
  olduğuna bir alt sınırdır, ne kadar iyi olduğunun ölçüsü değil"*
  (`labels.py:86-90`).
- **Gece/düşük ışık RGB'de kabulün ne olduğu: ölçülmedi.** Altyapı var — her
  kayıt `luma` ve `target_luma` taşıyor, notebook 24 hücre 5 `summarise_luma`
  basıyor, `MIN_LUMA` kapısı **varsayılan kapalı** (`MIN_LUMA = 0.0`) çünkü
  makul görünen bir sayı seçmek ölçmenin yerine geçmez
  (`build_vtuav_pool_notebooks.py:40-47`). LT dizileri uzun ve gece kareleri
  içeriyor; RGB kolunun asıl risk yeri burası.
- **Teacher tutarsızlığı:** `vtuav_thermal` (ST) **SAM 2.1** ile,
  `vtuav_lt_thermal` (LT) **SAM 3** ile hasat edilmiş
  (`:495` ve `:474`). Bu farkın eğitime etkisi **ölçülmedi**. RGB tarafında aynı
  hatayı yapmamak için 16 ve 24 aynı öğretmenle koşulmalı.

---

## 5. Kötü bir maske eğitime girerse ne olur

Somut zincir:

1. Maske kutuya oturuyor ama nesnenin **etrafındaki zemini** de kapsıyor
   (`area_ratio` 1.3'ün altında kaldığı sürece geçer). Decoder "hedef =
   nesne + altındaki asfalt" öğrenir; bu tam olarak sahada gördüğümüz
   *"kutu araziye yayılıyor"* davranışıdır (`docs/calisma_plani_tr.md:22`).
2. Stage C'de bu bir kareyle sınırlı kalmaz: maske belleğe yazılır ve sonraki
   karelere taşınır. Tek kare denetimi olan Stage B'de aynı hata daha ucuzdur.
3. `box_iou` bunu yakalayamaz: taşma kutunun içinde kaldığı sürece maskenin
   çevreleyen kutusu doğru yerdedir. Bunu yakalayan `area_ratio` üst sınırı ve
   `component` (`labels.py:100-101`) — ve `summarise_gates`'in **her kapıyı
   bağımsız** sorması, çünkü `reject_reason` yalnız **ilk** düşen kapıyı
   döndürür ve bu, veriye dair yanıltıcı bir tablo verir
   (`build_vtuav_pool_notebooks.py:59-69`, `pool.gate_report` docstring
   `pool.py:700-712`).
4. Ve deponun kendi kuralı: **gerçek bir hedefte tetiklenen kapı, kapısız
   olmaktan kötüdür** (CLAUDE.md). 0.8'i her yere uygulamak, §2.1'deki ölçülmüş
   sebeple, en çok küçük hedefi eler — yani şart (b)'yi kurtarayım derken şart
   (a) küçük hedef verisini kesebilir.

---

## 6. Öneri

**Yapılabilir ve büyük ölçüde zaten yapılmış. Sıra şu:**

1. **Önce bak, koşma.** Drive'da `edgetam-pool/vtuav_lt_rgb/` var mı? Varsa
   `pool_index.jsonl` üzerinden kabul oranı ve sınıf dağılımı, termal
   kardeşinin %87,1'i ile yan yana okunur. Bedeli sıfır.
2. **Yoksa notebook 24'ü koş** — kod değişikliği gerekmez. Aynı öğretmen
   (`facebook/sam3`), aynı `BOX_IOU = 0.6`, aynı `ZOOM/MIN_SIZE`. Hücre 5'in
   bastığı `summarise_gates` tablosu 0.5/0.6/0.7/0.8/0.9 kesimlerinin **her
   birinin** ne kadar veri götürdüğünü söyler; 0.8 kararı orada verilir,
   burada değil.
3. **Gece kontrolü**: `summarise_luma` çıktısına bak. Karanlık kareler kabulü
   düşürmüyorsa `MIN_LUMA` kapalı kalsın; düşürüyorsa **RGB kolunda** bir eşik
   düşünülebilir (termalde asla — orada düşük okuma *soğuk* hedef demektir).
4. **Sonra 37'ye bağla**, §3.2'deki altı düzenlemeyle, `datasets={"vtuav_lt_rgb"}`
   açıkça verilerek ve 1000-kare assert'i ile.
5. **Kalibrasyonu bir kez gerçekten yap.** VTUAV-VIS'in çizilmiş RGB maskeleri
   elimizde ve `masklets.calibrate` hazır. "Kabul oranı %87" bir kalite ölçüsü
   değil; teacher-vs-çizim IoU'su ölçüdür ve bu depo hiç ölçmedi. En değerli
   tek ölçüm bu.
6. **Şart (b) için dürüst olalım:** Stage C'de boyut kapısı **yok**. Ya
   `pool_sequence_stores`'a `InstanceGates` eklenmeli (bu belgenin kapsamı
   dışında, ayrı bir iş), ya da "VTUAV tek hedefli ve medyan hedef 77 px, o
   yüzden pratikte gerekmiyor" gerekçesi **açıkça yazılmalı** — sessizce
   varsayılmamalı.

**Karşı görüş, kayda geçsin:** VTUAV-LT RGB, projenin ana hattı değil.
CLAUDE.md'nin ilk cümlesi termalin öncelik olduğunu söylüyor ve 37 zaten
**yazarların kendi maskeleriyle** koşan tek RGB Stage C — kimsenin tartışmadığı
denetim. Teacher havuzu eklemek o özelliği bozar: sonuç "35 + Stage C RGB'yi
bozmadı mı" sorusuna cevap verirken artık "hangi yarısı teacher maskesiydi"
sorusunu da taşır. Bu yüzden ekleme yapılacaksa **kaynak-bazlı ayrı satır**
olarak raporlanmalı (31'in `source_name` bazlı tablosu bunu zaten yapıyor),
tek bir ortalamada eritilmemeli.
