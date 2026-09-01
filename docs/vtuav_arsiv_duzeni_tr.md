# VTUAV arşivlerinin düzeni — hangi zip'in içinde ne var

Bu proje VTUAV'ın **üç ayrı biçimini** kullanıyor ve bunlar sürekli birbirine
karışıyor. Bu dosya, her birinin disk üzerindeki ağacını, hangi dosyanın neyi
taşıdığını ve hangi okuyucu fonksiyonun onu tükettiğini tek yerde toplar.

**Kural:** her iddianın yanında nasıl doğrulandığı yazar. İşaretler:

| işaret | anlamı |
|---|---|
| `[kod]` | çalışan koddan okundu; dosya ve satır verilir |
| `[yorum]` | koddaki bir yorumda, arşivin central directory'sinin okunmasıyla saptandığı belirtilmiş |
| `[foto]` | kullanıcı arşivi açıp klasörü kendisi gösterdi |
| **[DOĞRULANMADI]** | bu dalda hiçbir kanıt bulunamadı; varsayımdır |

**Hiçbir arşiv bu belge için indirilmedi.** Dosya sayıları ve boyutlar yalnızca
depodaki kodda kayıtlı olan ölçümlerden alınmıştır; onların dışında sayı
uydurulmadı.

---

## 0. Önce isimlendirme tuzağı

VTUAV'da bir dizi klasörünün adı `<hedef türü>_<sayaç>` biçimindedir
(`bike_009`, `pedestrian_192`, `train_003`) **ve sayaç her arşivde yeniden
başlar.**

Sonuç: `train_003` hem `test_001.zip`'in *içindeki bir dizinin adı*, hem de
ayrı bir *arşivin adıdır*. Bunlar aynı şey değildir.

* Kullanıcının fotoğrafladığı `train_004/` klasörü, `test_001.zip`'in içindeki
  bir **dizidir** — bir split değil. `[foto]`
* Aynı tehlike `fetch_datasets.py`'de açıkça yazılıdır: "`test_001.zip`
  contains a sequence called `train_003` while `train_003.zip` is a different
  archive entirely." `[yorum: tools/fetch_datasets.py:217` altındaki
  VTUAV_VIS bloğunun yorumu`]`

Bu daldaki iki savunma da yerinde:

1. **`Part.into`** — `tools/fetch_datasets.py:71`'de tanımlı, "" ise arşivin
   kendi düzeni korunur. VTUAV-VIS'in sekiz parçasının **hepsi** kendi
   `into="train_001"` … `into="test_005"` değerini taşır
   `[kod: tools/fetch_datasets.py:217-268]`. `extract()` bunu
   `dest / into` olarak uygular `[kod: tools/fetch_datasets.py:1031]`.
   Böylece iki arşivin aynı adlı dizileri tek klasöre karışmaz.
2. **Parça adının dizi adına yazılması** — `vtuav_vis_sequences` içinde:
   `part = sequence_dir.parent` ve arşivler kendi klasörlerine açılmışsa ad
   `f"{part.name}_{sequence_dir.name}"` olur; düz (flat) açılmışsa eski ad
   korunur, yani daha önce ölçülmüş hiçbir şey kaymaz
   `[kod: src/training/aerial_video.py:208-222]`.
   Nedeni `split_flights`'ın docstring'inde de tekrarlanır: "Part.into and the
   part-qualified names from `vtuav_vis_sequences` are what make two archives'
   identically-named sequences distinguishable in the first place."
   `[yorum: src/training/aerial_video.py:432]`

**Tracking (ST/LT) arşivlerinde `into` yoktur** `[kod:
tools/fetch_datasets.py:627-664]` — parçalar düz açılır. Gerekçe yorumda:
"The parts are ordered by sequence name, so each carries only two to four
object kinds" `[yorum: tools/fetch_datasets.py:623-626]`. Yani ST/LT
parçalarının dizi adları birbiriyle çakışmıyor kabul edilir; bu **ayrı bir
çakışma taraması yapılarak doğrulanmadı** — parçaların adlarının global
sıralandığı yorumdan çıkarılmıştır. **[DOĞRULANMADI: ST/LT parçaları arasında
aynı adlı dizi bulunmadığı]**

---

## 1. VTUAV-VIS — maske (video instance segmentation) sürümü

Arşivler: `train_001.zip`, `train_002.zip`, `train_003.zip`,
`test_001.zip` … `test_005.zip`. Kaynak: yazarların Drive klasörü
`11E-WPkCPVL49hOKRdCzfgQULmGU8pyz8` `[kod: tools/fetch_datasets.py:215-216]`.

### Ağaç (arşivin kökünde doğrudan dizi klasörleri)

```
<dizi>/rgb/000000.jpg          1920x1080 RGB kare
<dizi>/ir/000000.jpg           1920x1080 termal kare
<dizi>/mask/rgb/000000.png     RGB maskesi, mode L, değerler {0, 255}
<dizi>/mask/ir/000000.png      termal maske
<dizi>/rgb.txt                 kare başına `x y w h` takip kutusu
<dizi>/ir.txt
```

Kanıt:

* Bu listenin tamamı (`bike_009/...` örnekleriyle) `SPECS["vtuav_vis"]`'in
  üstündeki yorumda yazılıdır ve orada "The globs below were read off the
  archive, not guessed… by listing the central directory of
  `training/train_001.zip` over range requests" denir
  `[yorum: src/training/aerial.py:377-395]`.
* `masked_members`, maske üyelerini `len(parts) >= 4 and parts[-3] == "mask"`
  ile seçer — yani yol tam olarak `<dizi>/mask/<modalite>/<dosya>`'dır
  `[kod: tools/fetch_datasets.py:923-948]`.
* Kullanıcı `test_001.zip` içindeki `train_004/` klasörünü açtı: `ir/`,
  `ir.txt`, `mask/` (içinde `ir/` ve `rgb/`), `rgb/`, `rgb.txt`. `[foto]`

### **Maskeler burada.** Kutular da burada.

VTUAV-VIS, bu üç biçim içinde **çizilmiş maske taşıyan tek biçimdir**
(`mask/ir` ve `mask/rgb`) — ve ayrıca `ir.txt` / `rgb.txt` kutularını da
taşır. `[yorum: src/training/aerial.py:377-395]` + `[foto]`

Maske değeri **255'tir, 1 değil**; spec bunu `classes={"background": 0,
"target": 255}` olarak tutar ve yorum "this spec said 1, which would have made
every mask read as empty" diye düzeltmeyi anlatır
`[kod: src/training/aerial.py:407-416]`.

### Ne kadar var (kodda kayıtlı ölçümler)

`train_001.zip` üzerinde `[yorum: src/training/aerial.py:381-392]`:

| | |
|---|---|
| kare (her modalite) | 26 059 |
| RGB maskesi | 875 |
| IR maskesi | 874 |
| toplam maske / toplam görüntü | 1 749 / 52 118 |
| maskeli kare aralığı | her 30. kare |
| ön plan oranı | kare alanının %0.2–2.0'ı |

Parça bazında `[yorum: tools/fetch_datasets.py:219-232]`:

| parça | boyut | dizi | maske | hedef türleri |
|---|---|---|---|---|
| train_001 | 9.1 GB | 14 | 875 | bike 1, bus 4, c-vehicle 1, car 8 |
| train_002 | 16.1 GB | 18 | 1 408 | car 4, elebike 3, excavator 2, pedestrian 9 |
| train_003 | 17.9 GB | 18 | 1 778 | pedestrian 15, train 1, tricycle 1, truck 1 |

**`train_001` içinde hiç yaya yoktur** — tek başına eğitilirse encoder'a
"hedef bir araçtır" öğretilir `[yorum: tools/fetch_datasets.py:229-235]`.
`test_001..005` parçalarının içerik dökümü kodda **yoktur**; yalnızca tahmini
boyutları (`Part.size`, 14–16.4 GB) kayıtlıdır
`[kod: tools/fetch_datasets.py:255-267]`.
**[DOĞRULANMADI: test parçalarının dizi/maske sayıları ve hedef türleri]**

İki uyarı yazarların makalesinden aktarılmıştır, bu depoda ölçülmemiştir: iki
modalite **tam registre değildir** ve her maske elle çizilmemiştir, bir kısmı
propagasyonla üretilmiştir `[yorum: src/training/aerial.py:399-405]`.
**[DOĞRULANMADI: burada bağımsız olarak ölçülmedi]**

### Kim okur

| okuyucu | ne yapar |
|---|---|
| `fetch_datasets.masked_members` (`tools/fetch_datasets.py:923`) | `--frames masked`: yalnız maskeleri ve maskeli karelerin iki modalitedeki eşlerini çıkarır |
| `SPECS["vtuav_vis"]` (`src/training/aerial.py:407`) | `thermal="**/ir/*.jpg"`, `rgb="**/rgb/*.jpg"`, `masks="**/mask/{modality}/*.png"` — stage A/B görüntü yolu |
| `aerial_video.vtuav_vis_sequences` (`src/training/aerial_video.py:169`) | `mask/<modalite>` klasörlerini gezer, video dizisi + `ImageMaskStore` üretir |

**Önemli ayrıntı:** `--frames masked` ile açılan bir ağaçta `ir.txt` /
`rgb.txt` **yoktur**. `masked_members`'ın tuttuğu üyeler ya
`len(parts) >= 4` (maske) ya da `len(parts) >= 3` (kare) koşulunu sağlar;
`<dizi>/ir.txt` `len(parts) == 2` olduğu için hiçbir dala girmez
`[kod: tools/fetch_datasets.py:941-948]`. `vtuav_vis_sequences` kutuları bu
yüzden maskenin kendisinden türetir (`_box_from_mask`) — "so an all-frame
`*.txt` can never be misaligned with a masked-only extraction"
`[kod: src/training/aerial_video.py:160-181]`.

Ardışık iki maskeli kare arasında **30 kaynak karesi** vardır; bu bilinçli
uzun-ufuk denetimi sağlar ama yoğun takip verisinin yerine geçmez
`[yorum: src/training/aerial_video.py:174-178]`.

---

## 2. VTUAV tracking ST/LT — kutu (takip) sürümü

Arşivler: `train_ST_001.zip` … `train_ST_011.zip`, `train_LT_001.zip` …
`train_LT_004.zip` `[kod: tools/fetch_datasets.py:627-664]`. Notebook 31 ve
notebook 16/17/24/25 bunları kullanır.

### Ağaç (yine arşivin kökünde doğrudan dizi klasörleri)

```
<dizi>/rgb/000000.jpg        1920x1080, kare id'leri 0..n-1, boşluksuz
<dizi>/ir/000000.jpg
<dizi>/rgb.txt               ceil(n / 10) satır, `x y w h`
<dizi>/ir.txt                ceil(n / 10) satır, `x y w h`
```

**Maske klasörü yoktur.** Bu üç biçim içinde yalnızca kutu taşıyan biçim
budur.

Kanıt:

* `vtuav_frames`'in docstring'i: "Layout, read off `train_ST_001.zip`'s
  central directory rather than the paper: `<sequence>/rgb/000000.jpg`,
  `<sequence>/ir/000000.jpg`, `<sequence>/rgb.txt`, `<sequence>/ir.txt`."
  `[yorum: src/training/boxes.py:598-608]`
* `tracked_members` yalnızca iki şekli tanır: `len(parts) == 2` ve `.txt` ile
  biten dosyalar, ya da `len(parts) == 3` ve `parts[1] in ("rgb", "ir")`. Bir
  `mask/` dalı ne aranır ne de tutulur `[kod: tools/fetch_datasets.py:1019-1028]`.
* `vtuav_frames`'in hata mesajı beklenen düzeni yazar: "expected
  `<sequence>/{modality}.txt` beside `<sequence>/{modality}/`"
  `[kod: src/training/boxes.py:700-702]`.
* `aerial_video.vtuav_sequences` de aynı ikiliyi arar: `root.rglob(f"{modality}.txt")`
  ve yanında `box_file.parent / modality` klasörü `[kod: src/training/aerial_video.py:63-101]`.

`SPECS["vtuav"]` bir `masks="**/mask/{modality}/*.png"` glob'u **tanımlar**,
ama aynı bloğun yorumu bunun neden boş döneceğini açıklar: "Its annotation is
a tracking box, and only a 100-video subset carries masks -- so `list_frames`
will find almost nothing here" `[kod+yorum: src/training/aerial.py:337-357]`.
Yani spec'teki glob, o 100 videoluk alt küme (VIS sürümü) aynı ağacın altına
açıldığında iş görsün diye vardır; ST/LT arşivinin kendisinde karşılığı yoktur.

### Kutuların anlamı

* Satır `k`, kare `k * stride`'a aittir; ST_001'in 20 dizisinin hepsinde
  `stride = 10`, yani **on karede dokuzu etiketsizdir**
  `[yorum: tools/fetch_datasets.py:955-960]`, `[kod: src/training/boxes.py:538]`.
* Satır biçimi `x y w h`; genişliği veya yüksekliği pozitif olmayan ya da NaN
  taşıyan satır **hedefin görüş dışı olduğu** anlamına gelir. `read_boxes` bu
  satırı silmez — indeksler kaymasın diye yerinde tutar ve `exist=False`
  bayrağını verir `[kod: src/training/masklets.py:111-134]`. `exist`
  etiketlerinin geldiği yer budur.
* Sınıf, dizi adının önekidir (`bus_017` → `bus`)
  `[kod: src/training/boxes.py:679-681]`.
* **İki modalitenin kutuları farklıdır.** `train_ST_001`'in 3 750 etiketli
  satırında `rgb.txt` ile `ir.txt` satırlarının yalnızca %12.2'si aynıdır;
  merkez sapmasının medyanı 8.4 px (p90 29.4), medyan hedef 77 px. Yalnız-RGB
  ve yalnız-IR satır sayısı sıfırdır — hedef ya iki dosyada da vardır ya
  hiçbirinde `[yorum: src/training/boxes.py:625-638]`.

### Ne kadar var (kodda kayıtlı ölçümler)

`train_ST_001` üzerinde `[yorum: tools/fetch_datasets.py:620-622]`:
20 dizi, 37 419 kare çifti, 3 750 etiketli satır; arşivin kendisi 15.4 GiB.
Bütün eğitim yarısı **214.5 GiB**'dir ve `fetch_datasets`'te **hiçbir parça
varsayılan olarak indirilmez** (`default=False`) `[kod: tools/fetch_datasets.py:628-664]`.

Parça başına hedef türü dağılımı `[yorum: tools/fetch_datasets.py:623-626]`:
ST_001 animal/bike/bus, ST_005 car/elebike, ST_008 pedestrian (28 dizinin
24'ü), ST_011 car/pedestrian/truck. Ardışık parçalar tek kategoriye yığılır;
havuz kurarken dağıtmak gerekir.

**RGB-only arşivleri indirmeyin.** Yazarların ayrı yayımladığı RGB sürümü,
RGB-T sürümünün `ir/` ve `ir.txt` çıkarılmış halidir: `train_ST_001` iki
klasörden de örneklendiğinde her RGB üyesi hem CRC32 hem sıkıştırılmış boyut
olarak eşleşmiştir `[yorum: tools/fetch_datasets.py:612-618]`.

### LT parçaları hakkında dürüst not

Yukarıdaki düzen ve `stride = 10` **yalnızca `train_ST_001` üzerinde**
doğrulanmıştır. Kod bunu açıkça söyler: "That 10 was measured on one
short-term part; the long-term parts are a separate download and this function
is the first thing that touches them" `[yorum: tools/fetch_datasets.py:973-981]`
ve "the long-term parts are a separate download whose layout nobody in this
repo has opened" `[yorum: src/training/boxes.py:610-617]`.

**[DOĞRULANMADI: LT arşivlerinin iç düzeni ve stride'ı.]** Kod bu boşluğu
kapatmak yerine ölçmeyi seçer: `boxes.annotated_stride` /
`_sequence_stride`, her dizinin stride'ını kendi kare ve satır sayısından
türetir; tek bir cevap çıkmayan dizi sayıları basılarak **atılır**, tahminle
etiketlenmez `[kod: src/training/boxes.py:551-596`, `tools/fetch_datasets.py:993-1015]`.

### Kim okur

| okuyucu | ne yapar |
|---|---|
| `fetch_datasets.tracked_members` (`tools/fetch_datasets.py:951`) | `--frames tracked_ir` / `tracked_rgb` / `tracked`: yalnız kutu dosyasının adlandırdığı kareleri çıkarır; iki `.txt` her zaman tutulur |
| `boxes.vtuav_frames` (`src/training/boxes.py:598`) | havuz hasadı için `BoxFrame` listesi; anahtar `<dizi>/<stem>` |
| `aerial_video.vtuav_sequences` (`src/training/aerial_video.py:63`) | Stage C video dizileri; `exist` + kutular `read_boxes`'tan |

**Tek modaliteli çıkarma tek ağaç ister.** `tracked_ir` her iki `.txt`'yi
tutup yalnız `ir/` karelerini yazdığı için, aynı klasöre `tracked_rgb` de
açılırsa `rgb.txt` dolu ama `rgb/` boş görünür. `vtuav_frames` bu durumu ayrı
bir hata mesajıyla yakalar ve "give each one its own root (e.g.
`<root>_<modality>`)" der `[kod: src/training/boxes.py:693-699]`. Notebook
16/17/24/25 zaten `VTUAV_rgb`, `VTUAV_ir`, `VTUAV_lt_rgb`, `VTUAV_lt_ir` diye
ayrı `DATA_ROOT` kullanır `[kod: tools/build_vtuav_pool_notebooks.py:240-261]`.

---

## 3. Hasat edilmiş maske havuzları (`edgetam-pool/vtuav_thermal`, `vtuav_lt_thermal`)

Bunlar bir *indirme* değil, **notebook 17 ve 25'in ürettiği** üçüncü bir
biçimdir: ST/LT kutuları bir öğretmen modele prompt olarak verilir, çıkan
maskeler kare kare saklanır.

| havuz | notebook | modalite | kaynak arşivler |
|---|---|---|---|
| `vtuav_rgb` | 16 | rgb | ST_001, ST_005, ST_008, ST_011 |
| `vtuav_thermal` | 17 | ir | ST_001, ST_005, ST_008, ST_011 |
| `vtuav_lt_rgb` | 24 | rgb | LT_001..LT_004 |
| `vtuav_lt_thermal` | 25 | ir | LT_001..LT_004 |

`[kod: tools/build_vtuav_pool_notebooks.py:240-261]`

### Ağaç

```
<POOL_ROOT>/<havuz>/<dizi>/<kare stem>/record.json
<POOL_ROOT>/<havuz>/<dizi>/<kare stem>/pseudo_masks.npz
<POOL_ROOT>/pool_index.jsonl
```

Kanıt:

* `label_pool` çıktı kökünü `Path(out_dir) / dataset` yapar
  `[kod: src/training/pool.py:398]`; kare klasörü `_frame_dir(out_root, frame.key)`
  ile açılır ve `key` yol parçalarına bölünür `[kod: src/training/pool.py:265-267, 420]`.
* VTUAV için `frame.key = f"{sequence.name}/{stem}"`
  `[kod: src/training/boxes.py:679]` — yani `<dizi>/<kare stem>`.
* Dosya adları sabit: `RECORD_FILE = "record.json"`,
  `INDEX_FILE = "pool_index.jsonl"` `[kod: src/training/pool.py:53-54]`,
  `MASK_STORE = "pseudo_masks.npz"` `[kod: src/training/labels.py:40]`.

Drive'a şu biçimde yazılır: `edgetam-pool/<havuz>/<havuz>.zip` — zip üyeleri
`POOL_ROOT`'a göre görecelidir, yani içinde `<havuz>/<dizi>/<stem>/record.json`
vardır — ve yanına `pool_index.jsonl` kopyalanır
`[kod: tools/build_vtuav_pool_notebooks.py:607-620]`.

### `record.json` ne taşır

`key`, `dataset`, `prompt`, `image` (hasat anındaki **mutlak** yol), `shape`,
`luma`, `target_luma`, `teacher`, ve `instances` listesi; her instance
`{"i": kutunun kare içindeki satır numarası, "class", "box", …kapı ölçümleri…,
"verdict"}` `[kod: src/training/pool.py:472-497]`. `verdict is None` kabul
edilmiş demektir `[kod: src/training/aerial_video.py:288-296]`. Ayna
(mirror) kayıtlarında `prompt="mirror"` ve `mirror_of` alanları bulunur
`[kod: src/training/pool.py:298-320]`.

### `pseudo_masks.npz` ne taşır

Run-length kodlanmış maskeler, **kare içindeki instance indeksiyle**
anahtarlanmış (dizi içindeki kare indeksiyle değil — detection verisinin zaman
ekseni yoktur) `[yorum: src/training/pool.py:26-31]`. Yazma sırası çökme
sözleşmesidir: önce `record.json`, en son `.npz`; bir kare ancak store'u
varsa bitmiştir `[yorum: src/training/pool.py:32-35]`.

**Pikseller havuzda değildir.** Store birkaç kilobayt run-length'tir; kare
hâlâ veri kümesinin kendi diskindedir, bu yüzden her okuyucu bir `images_root`
alır ve hasat anındaki mutlak yolu yeniden köklendirir
`[yorum: src/training/pool_reader.py:14-21]`.

### Kim okur

| okuyucu | ne yapar |
|---|---|
| `pool_reader.index_pool` (`src/training/pool_reader.py`) | havuzu Stage B'nin `FrameIndex` listesine çevirir, yolları yeniden köklendirir |
| `aerial_video.pool_sequence_stores` (`src/training/aerial_video.py:240`) | `record.json`'ları gezip maskeleri Stage C video dizilerine bağlar; eşleştirme `key`'in `<dizi>/<stem>` parçalarıyla yapılır |
| `pool.write_index` / `pool_report` / `gate_report` | `pool_index.jsonl` ve kabul/ret raporları |

---

## 4. Bir arşivi kendiniz doğrulama komutu

`tools/inspect_vtuav_vis.py` bir arşivi **açmadan** yargılar: ilk geçiş yalnız
ZIP central directory'sini okur, ikinci geçiş `--sample` kadar maske/kare
çiftini belleğe açar. Diske hiçbir şey yazmaz
`[yorum: tools/inspect_vtuav_vis.py:1-20]`.

**Yalnızca yerel dosya yolu alır.** URL de, Drive dosya id'si de kabul etmez:
argüman `type=Path`'tir `[kod: tools/inspect_vtuav_vis.py:259]`, dosya
`zipfile.ZipFile(archive)` ile açılır `[kod: tools/inspect_vtuav_vis.py:38-40]`
ve boyut `archive.stat().st_size` ile okunur
`[kod: tools/inspect_vtuav_vis.py:166]`. Colab'de arşivler Drive mount'unda
olduğu için verilecek yol mount yoludur:

```bash
python3 tools/inspect_vtuav_vis.py \
    /content/drive/MyDrive/VTUAV/test_001.zip \
    --modality ir --sample 40
```

Birden çok arşiv aynı anda verilebilir (`archives` `nargs="+"`)
`[kod: tools/inspect_vtuav_vis.py:259]`:

```bash
python3 tools/inspect_vtuav_vis.py \
    /content/drive/MyDrive/VTUAV/train_001.zip \
    /content/drive/MyDrive/VTUAV/test_001.zip --modality ir
```

Çıktının bu belge açısından önemli satırları:

* `top level: N entr(ies)` — arşivin kökündeki dizi adları. Dosya adı ile
  üyeler çelişirse ("the file is called `test_001` and its members say
  `train_003`") uyarı basılır ve **üyelere güvenilir**
  `[kod: tools/inspect_vtuav_vis.py:172-177]`.
* `folders inside a sequence:` — `ir`, `rgb`, `mask/ir`, `mask/rgb` gibi
  klasörlerin dosya sayıları. **Bir arşivin maske mi kutu mu taşıdığını
  gösteren satır budur:** `mask/...` satırı yoksa o arşiv tracking (kutu)
  arşividir `[kod: tools/inspect_vtuav_vis.py:183-190]`.
* `masks with a frame beside them` ve örneklenen karelerin
  boyut/sinyal-gürültü dağılımı.

İsteğe bağlı `--against <manifest.json>` sızıntı denetimi yapar: verilen
dosyadan düz bir `sequences` listesi okur ve arşivdeki dizilerle kesişimini
basar `[kod: tools/inspect_vtuav_vis.py:146-160]`.
**[DOĞRULANMADI: VTUAV havuzları için böyle bir `manifest.json` üreten kod yok.]**
Notebook 16/17/24/25 yalnız `pool_index.jsonl` yazar
`[kod: tools/build_vtuav_pool_notebooks.py:617-619]`; `sequences` anahtarlı
`manifest.json` yazan yerler notebook 01 ve 29'dur ve orada `sequences` bir
liste değil, split→dizi sözlüğüdür — o dosya `--against`'e verilirse kesişim
dizi adlarıyla değil split adlarıyla yapılır. VTUAV havuzu için bu listenin
elle (ya da `pool_index.jsonl`'daki `key` alanlarının ilk parçasından)
üretilmesi gerekir.

---

## 5. Tek bakışta: maske mi, kutu mu?

| biçim | arşiv/klasör | maske | kutu | `exist` |
|---|---|---|---|---|
| VTUAV-VIS | `train_00x.zip`, `test_00x.zip` | **var** — `mask/ir`, `mask/rgb` | var (`ir.txt`, `rgb.txt`; `--frames masked` ile açılınca çıkmaz) | yok — `vtuav_vis_sequences` hepsini `True` verir `[kod: src/training/aerial_video.py:225]` |
| VTUAV tracking ST/LT | `train_ST_00x.zip`, `train_LT_00x.zip` | **yok** | **var** — `ir.txt`, `rgb.txt`, `x y w h`, satır k = kare 10k | **var** — dejenere/NaN satır = hedef görüş dışı `[kod: src/training/masklets.py:126-134]` |
| Hasat havuzu | `edgetam-pool/vtuav_thermal`, `vtuav_lt_thermal` (+ `_rgb`) | **var** — öğretmen üretimi, `pseudo_masks.npz` | `record.json` içinde prompt kutusu olarak | yok (havuzun zaman ekseni yok) |

Not: bir maskenin kaynağı önemlidir. VTUAV-VIS maskeleri **insan çizimidir**
(bir kısmı propagasyon); havuz maskeleri bir **öğretmen modelin** kapılardan
geçmiş çıktısıdır ve `record.json` hangi öğretmen olduğunu yazar.
