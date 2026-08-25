# Pseudo-mask havuzları — RGB ve termal dal planı

Amaç: EdgeTAM'ı hem RGB hem termal havadan görüntüde geliştirmek için,
**kutu etiketli** veri setlerini güçlü bir öğretmen modelle (SAM 3, yedeği
SAM 2.1) **maske havuzlarına** çevirmek. Havuzlar `(görüntü, prompt, maske)`
üçlüsü üretir — SAM 2'nin kendi eğitim tarifi — ve iki aşamayı besler:

| Havuz | Besleyeceği aşama | Neden |
|---|---|---|
| statik görüntü + kutu → maske | **Aşama B** (tek kare promptable segmentasyon, `src/training/image_loop.py`) | B'nin bugünkü verisi Kust4K/SegFly'ın *yeniden kurulmuş* örnekleri + VTUAV'ın 875 çizilmiş maskesi; havuz buna on binlerce gerçek-kutu-tabanlı örnek ekler |
| video + kare-başına kutu → masklet | **Aşama C** (video, `src/training/masklets.py` deposu) | C'nin eksik malzemesi kimlikli kare-başına maske; VTUAV için makine hazır, RGBT234/LasHeR aynı boru hattına girer |

İki defter, iki dal:

- `notebooks/13_rgb_mask_pool.ipynb` — **RGB dalı.** Kolay olan: öğretmenler
  RGB'de eğitildi, havadan RGB kutu verisi bol.
- `notebooks/14_thermal_mask_pool.ipynb` — **termal dal (asıl hedef).** İki
  rota, ikisi de ölçülerek: (a) hizalı RGB-T çiftinde maskeyi RGB'de üretip
  termale taşı, (b) öğretmeni doğrudan termal kareye kutuyla promptla.

İkisi de `tools/build_pool_notebooks.py`'den üretilir (probe defterinin
kalıbı), hücreleri `tools/check_notebook.py`'den ve stamp kontrolünden geçer.
Orkestrasyon `tools/make_mask_pool.py`'de; defterler tarif, kod `src/`'de ve
GPU'suz test ediliyor — reponun geri kalanıyla aynı sözleşme.

---

## 1. Öğretmen: SAM 3, ve neden

Kullanıcı sorusu "Meta'nın SOTA'sı hangisi" idi; cevap ölçüldü, varsayılmadı:

| aday | durum (2026-08) | karar |
|---|---|---|
| **SAM 3** (`facebook/sam3`) | transformers ≥ 5.0'da tam entegre: görüntü için `Sam3TrackerModel`/`Sam3TrackerProcessor` (SAM 2 ile aynı API — kutu promptu, `iou_scores`, `post_process_masks`), video için `Sam3TrackerVideoModel`. Gated repo; lisansı Apache değil | **varsayılan öğretmen.** Yayınlanan farklar görüntü PVS'te SAM 2.1'in üstünde, videoda SA-V +5.6 / LVOSv2 +8.9 J&F |
| SAM 2.1 (`facebook/sam2.1-hiera-large`) | ungated, Apache-2.0, transformers 4.56+ | **tek satırlık yedek.** Hesap istemez; MOSEv2 değerlendirmesini dürüst tutan öğretmen de bu (aşağıda) |
| SAM 3.1 (`facebook/sam3.1`) | var ama **checkpoint-only**: `library_name: checkpoint`, transformers entegrasyonu yok, GitHub paketi `numpy<2`'ye pinli (bizim ortam numpy 2.x). Başlık özelliği Object Multiplex = kare başına çok nesnede *hız*; kutu-başına-crop çalışan bir öğretmene hiçbir şey kazandırmaz | **elendi** (repo bunu commit `915be02`'de zaten ölçmüştü) |

İki kontaminasyon kuralı havuzlara taşınır (`docs/datasets.md`'deki tablo):

1. **SAM 3 MOSEv2'yi eğitimde gördü.** Havuz üretiminde sorun değil (havuz
   *eğitim* verisi), ama MOSEv2 val'de ölçülecek her şey SAM 2.1
   öğretmeninde kalmalı. Her maske deposu raporu öğretmenin model id'sini
   yazar; hangi maskenin hangi öğretmenden geldiği dosyadan okunur.
2. **AeroVIS = VisDrone + UAVDT + SeaDronesSee** (kutuları SAM3 ile maskeye
   çevrilmiş). VisDrone'dan havuz üretirsek AeroVIS o modelin held-out
   değerlendirmesi olamaz — kare kare aynı veri. Temiz değerlendirme adayı
   olarak UAVScenes/MVUAV kalır; bu defterlerin işi eğitim havuzu, ölçüm
   seti değil.

Öğretmen sınıfı model id'den seçilir (`"sam3"` geçiyorsa SAM 3), ayrı bir
bayrak checkpoint'le çelişemez — `masklets.teacher_class_for` ile aynı kalıp.

## 2. Veri: ne bulundu, ne doğrulandı

Hepsi canlı uç noktaya karşı kontrol edildi (2026-08); "bariz yöntem"in
neden kırıldığı ve ne kullanıldığı `tools/fetch_datasets.py` içinde belgeli.

### RGB dalı (13)

| set | ne | indirme | kutu biçimi |
|---|---|---|---|
| **VisDrone2019-DET** | 6 471 train + 548 val havadan RGB, 10 sınıf, yoğun küçük hedef | HF aynası `banu4prasad/VisDrone-Dataset` — düz dosyalar (`images/` + `labels/`), `snapshot_download` ile bölüm bölüm | YOLO txt (normalize cxcywh) → piksel xyxy |
| **DroneVehicle — RGB yarısı** | 28 442 çiftin RGB'si, gündüz+gece | fetcher'da zaten var (`dronevehicle`), 100 px beyaz bant `border` ile düşülüyor | VOC-vari XML'de yönlü kutu (poligon) → dik çerçeve zarfı |

VisDrone sınıf kimlikleri defterde **histogramla problanır** (Kust4K/SegFly
paletlerinin ikisinin de ezberden yanlış yazıldığı ders: `probe_classes`
kalıbı kutulara da uygulanır — sınıf id dağılımı beklenenle karşılaştırılır).

### Termal dal (14)

| set | ne | indirme | rota |
|---|---|---|---|
| **HIT-UAV** | 2 898 termal kare (640×512), 24 899 kutu, insan/araba/bisiklet/diğer-araç, 80–130 m irtifa, CC-BY-4.0 | GitHub reposunun kendisi veriyi taşıyor: codeload zip, **~408 MB**, `normal_json/` = COCO | (b) doğrudan termal |
| **DroneVehicle — termal yarısı** | 28 442 çiftin TIR'ı, OBB kutulu | fetcher'da var | (b) doğrudan termal; RGB yarısı hizalı olduğundan (a) ile kıyaslanabilir |
| **RGBT234** | 234 hizalı RGB-T video, 233,8 K çift, **modalite başına ayrı kutu** (`visible.txt` + `infrared.txt`) | resmî dağıtım Baidu-only; HF aynası `xche32/rgbt234` **tek 7,67 GB tar.gz** | (a) video: maske RGB'de, kapı termal kutuda |
| **VTUAV** | 1,7 M hizalı 1920×1080 çift, kare-başına kutu | fetcher'da var; `tools/make_masklets.py` bu rotayı zaten koşuyor | (a) video — mevcut makine |
| **LasHeR** | 1 224 hizalı dizi / 730 K+ çift, modalite başına kutu | resmî: Baidu/TeraBox. HF aynası `xche32/lasher` = 5 parçalı tar.gz, **~224 GB** — Colab diskini aşar | (a) video; **varsayılan değil.** Akış halinde (arşivi diske yazmadan) dizi-seçmeli çıkarma destekleniyor, ama ağ maliyeti 224 GB okumak; büyük diskli makineye belgelendi |
| **Kust4K** | 4 024 hizalı çift + gerçek semantik maske | fetcher'da var (~2,7 GB: tir+rgb+label) | **kalibrasyon** — aşağıda |

RGBT234 ve LasHeR **yer seviyesi** RGB-T'dir (kullanıcının verdiği iki
kaynak); havuzda havadan setlerle karışımı, encoder'ın modalite ekseni için
çeşitlilik sayılır — ama etiketde `dataset` alanı durur, aşama B/C istediğini
seçerek okur.

## 3. Mekanik: tek boru hattı, iki dal

Anti-UAV410 etiketleyicisinin ispatlanmış üç kararı aynen taşınır
(`src/training/labels.py`): **zoom-crop** (küçük hedefte öğretmene kolay
problem), **kapılar** (kabul oranı ölçülen bir sayı), **RLE depo** (aynı
`pseudo_masks.npz`, aşama B/C okuyucuları değişmeden okur).

Yenisi üç modül:

- `src/training/boxes.py` — kutu setlerinin ortak okuyucusu. `BoxFrame`
  (görüntü yolu + xyxy kutular + sınıflar + varsa hizalı eş yolu); YOLO
  (VisDrone), COCO (HIT-UAV), DroneVehicle XML (OBB→zarf) okuyucuları;
  sınıf histogramı probu.
- `src/training/pool.py` — havuz geçişi: kare başına kutuları zoom-crop'la
  öğretmene ver, `Gates`'ten geçir, kabul edileni görüntü-başına `.npz`'ye
  (anahtar = örnek indeksi) yaz; `pool_index.jsonl`'e görüntü/örnek/sınıf/
  kutu/karar satırı ekle; resume (var olan `.npz` atlanır); Kust4K
  kalibrasyonu (aşağıda); markdown özet.
- `masklets.py` uzantısı — `find_rgbt_sequences` (LasHeR/RGBT234 düzeni:
  `visible/ infrared/ visible.txt infrared.txt`) ve `VideoSequence`'a
  isteğe bağlı `gate_boxes`: prompt RGB kutusuyla, **kapı termal kutuyla**.
  Hizalama kaydığında maskeyi düşüren şey tam bu ikinci kutu.

CLI: `tools/make_mask_pool.py` (defterlerin çağırdığı yüzey; `make_masklets.py`
video rotası için zaten var).

### Termal rota seçimi bir iddia değil, bir ölçüm

Kust4K'nın 4 024 hizalı çifti **gerçek maskeli** — `decompose` örnekleri
GT'dir. Kalibrasyon geçişi her örnek için iki şey yapar:

1. örneğin kutusuyla öğretmeni **TIR karede** promptla → IoU(GT)
2. aynı kutuyla **hizalı RGB karede** promptla → IoU(GT)

Sınıf ve boyut kovası başına iki sütun; hangi rotanın nerede kazandığı
(gündüz/gece dahil — Kust4K stem'inde `D`/`N` var) sayıyla görülür,
sonra hasat o rotayla koşulur. Video tarafında aynı ölçüm zaten var:
`make_masklets.py --calibrate` VTUAV VIS'in çizilmiş maskelerine karşı
`--frames rgb` ile `--frames ir`'ı yarıştırır.

RGB dalının kalibrasyonu aynı kalıpla **Kust4K'nın RGB yarısında** (1,66 GB)
— VisDrone'un yoğun maskesi yok, ama öğretmenin "havadan küçük araç RGB'de
kutudan maske" başarısı orada da ölçülür.

### Kapılar

`labels.Gates` aynen: `teacher_iou` (SAM'ın kendi skoru — küçük hedefte
zayıf kapı olduğu ölçülmüştü, eşiği bilgi taşımaz ama sıfır maliyetli),
`box_iou` (yük taşıyan kapı: maske verilen kutuya oturuyor mu),
`area` (parça/taşma), `component` (tek nesne tek parça). Reddedilen kare
depoda **yok** — okuyucular "burada maske süpervizyonu yok" diye okur,
boş maske diye değil. Kabul oranları rapora, ret sebepleri kapı kapı tabloya.

### Depo düzeni

```
POOL_ROOT/
  visdrone/train/<görüntü-anahtarı>/pseudo_masks.npz   # anahtar=örnek idx
  hituav/train/...
  dronevehicle/train/...
  pool_index.jsonl                                     # kare×örnek kaydı
  rgbt234/<dizi>/pseudo_masks.npz                      # anahtar=kare idx
  rgbt234/<dizi>/masklets.json
```

Drive'a `edgetam-pool/{rgb,thermal}` altına, **set biter bitmez** zip'lenip
aynalanır (074f0f2'nin dersi: hepsi bitince değil, biten bitince).

## 4. Defterlerin iskeleti

Ortak (07–11 kalıbı): stamp + repo kontrolü → GPU/kernel kontrolü →
bağımlılıklar (SAM 3 için `transformers>=5`, yoksa 2.1 için 4.56 tabanı) →
ayarlar (TEACHER tek string) → Drive mount + `mirror_now` → indirme →
`describe_layout` + sınıf probu → öğretmen → **önce bak**: birkaç karede
maske çiz, kabul/ret örnekleri yan yana → kalibrasyon → hasat (resume'lu) →
özet tablolar → Drive'a zip.

Farkları:

| | 13 (RGB) | 14 (termal) |
|---|---|---|
| kaynaklar | VisDrone-DET (+ DroneVehicle-RGB isteğe bağlı) | HIT-UAV + DroneVehicle-TIR (b) + RGBT234 (a, video) |
| kalibrasyon | Kust4K-RGB örnekleri | Kust4K TIR **ve** RGB — rota kararının kendisi |
| video hücresi | yok | RGBT234 masklet'leri (+ VTUAV için mevcut CLI'a işaretçi, LasHeR belgeli seçenek) |
| disk | ~4 GB (VisDrone train+val ~1,7 GB, Kust4K-RGB 1,7 GB) | ~22 GB (DroneVehicle 8,9 + RGBT234 7,7 + Kust4K 2,7 + HIT-UAV 0,4) |

## 5. Sonrası: A + B + C karşılaştırması nasıl kurulur

Kullanıcının iki kolu — (1) stok EdgeTAM, (2) EdgeTAM + A + B + C — bugünkü
defterlerle şöyle hizalanır:

- **A (ön-eğitim):** 07–11 zaten koşuyor (distilasyon; ConvNeXt V2'nin
  FCMAE'si `docs/encoder_arastirma.md`'de MAE-evrişimsel not olarak duruyor;
  distilasyon seçiminin gerekçesi orada). Havuz defterleri A'ya dokunmaz.
- **B (tek kare):** bu havuzlar B'nin veri tarafını büyütür. `pool.py`'nin
  okuyucusu `image_loop`'un beklediği örnek biçimini verir; B defterine
  bağlamak ayrı, küçük bir değişiklik (DATASETS listesine havuz kaynağı).
- **C (video):** RGBT234/VTUAV masklet depoları `labels.py` deposunun
  kendisi — `clip_loop` hazır olduğunda aynı arayüzden okunur.

Ölçüm disiplinini 12 koydu: prompt karışık (`mix`), skor tek promptlu kare
değil takip metriği olduğunda anlamlı. Havuz defterleri **hiçbir şeyi
eğitmez** — ürettikleri havuzun kalitesini (kalibrasyon IoU, kabul oranı)
raporlar; eğitim sayıları 07–11/02'nin işi olarak kalır.

## 6. Riskler, açık uçlar

- **HF aynaları üçüncü şahıs.** `xche32/rgbt234`, `banu4prasad/VisDrone-…`
  yayıncının kendisi değil; defter indirileni **sayarak** doğrular (kare
  sayısı, kutu histogramı) ve raporda kaynağı yazar. Resmî kaynaklar
  (Baidu) belgelendi.
- **RGBT234'te hizalama "yüksek doğrulukta"** deniyor ama piksel garantisi
  yok; `gate_boxes` (termal kutuya kapı) tam bu belirsizliğin sigortası, ve
  kalibrasyon hücresi taşınan maskeyi termal GT kutusuna karşı ölçüyor.
- **DroneVehicle OBB→zarf** dik çerçeveye şişme demek (45°'lik araçta zarf
  alanı ~2×). Zarf yalnız **prompt**; kapılardaki `area` oranı öğretmen
  maskesini zarfa değil kendi kutusuna göre ölçtüğünden şişme kabulü
  bozmaz, ama `box_iou` kapısı OBB doluluğuna göre gevşetilebilir — defterde
  ayrı bir ayar olarak duruyor ve kalibrasyonla seçiliyor.
- **LasHeR 224 GB.** Kod yolu var (akışlı, dizi-seçmeli), varsayılan değil.
  Tam havuz istenirse büyük diskli bir makinede `--parts` ile koşulur.
