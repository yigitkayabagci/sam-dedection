# Çoklu-öğretmen tasarımı: neyi doğru yakalıyor, neyi kaçırıyor

Önerilen tasarım şuydu: **DINOv3 → feature loss** ve **SAM → mask loss**, ikisi
birden tek bir custom encoder'a, ağırlıklı toplam olarak.

Kısa hüküm: **deseni doğru, üç varsayımı yanlış, bir eksiği gerçek.** Aşağısı
her birinin gerekçesi ve neyin koda girdiği. Kurulumun kendisi
`docs/encoder_mimari.md`'de.

---

## 1. Desen gerçek ve yayınlanmış

Çoklu-öğretmen distilasyonu uydurma değil. Bu oturumda okunan iki iş:

- **SigLino** (arXiv 2512.20157) — SigLIP2 + DINOv3'ü **donuk** öğretmen olarak
  tek öğrenciye distile ediyor. Öğretmen başına ayrı MLP projeksiyon başlığı,
  ölçek eşitlemek için **PHI-S normalizasyonu**, entropi-ağırlıklı öğretmen
  birleştirme. Gram-anchoring grounding'de +%3.57.
- **arXiv 2604.27128** — SAM 3'ün Perception Encoder'ını (PE-ViT-L+, 446M)
  40.66M'lik bir öğrenciye distile ediyor, dört terimli
  **"direction-then-scale"** kaybıyla. Edinburgh Pig'de %92.29 MOTA.

Yani "feature zenginliği + mekansal hassasiyet birbirini tamamlar" fikri
sağlam. Ama ikinci makale, ilginç bir şekilde, **SAM 3 ve DINOv3'ü
birleştirmiyor** — ikisi ayrı boru hattı bileşeni olarak kalıyor.

---

## 2. Üç yanlış varsayım

### 2.1 "DINOv3 patch-level 14×14 kaba grid, SAM maskesi piksel-hassas →
çözünürlük uyuşmazlığı"

**Yanlış, ve bizde zaten yok.** DINOv3'ün **bütün** varyantları patch **16**
(ViT-S 21M / S+ 29M / B 86M / L 300M / H+ 840M / 7B 6716M — hepsi /16, hepsi 4
register token). 14 olan DINOv**2**.

Bizde bunun sonucu:

```
DINOv3 /16 @ 512  →  32×32 ızgara
öğrencinin vision_features @ 512  →  32×32 ızgara      ← birebir aynı
```

`patch_aligned` bunu bir kuralla veriyor: "öğrencinin girdi çözünürlüğü,
öğretmenin ızgarasına yukarı yuvarlanmış". Öğretmenin haritası kayıptan önce
**hiç yeniden örneklenmiyor**.

Mask tarafı da: "encoder'ın üstüne küçük bir upsampling head ekle" önerisi
gereksiz — **SAM 2'nin mask decoder'ı zaten var** ve `pred_masks_high_res`'i
tam 512'de veriyor. Kayıp zaten piksel çözünürlüğünde.

### 2.2 "custom encoder"

Sıfırdan encoder eğitmiyoruz. **EdgeTAM'ın encoder'ını** ince ayarlıyoruz — ki
o zaten SAM 2'nin distilasyonu ve zaten bir mask decoder'ı var. Bu, analizi
değiştiriyor: öğretmenlerden birinin (SAM) verdiği şeyin çoğu checkpoint'in
içinde zaten var.

### 2.3 "Domain gap: DINOv3 web görüntüsüyle eğitildi, aerial değil"

**Doğru gözlem, yanlış sonuç.** DINOv3'ün *iki* ön-eğitim seti var:

| set | ne | çıkan checkpoint |
|---|---|---|
| **LVD-1689M** | 1.689 milyar web görüntüsü (Instagram) | S, S+, B, L, H+, 7B + ConvNeXt T/S/B/L |
| **SAT-493M** | 493 milyon **512×512 Maxar uydu ortho, 0.6 m/px** | sadece **ViT-L** ve **ViT-7B** |

Ama DINOv3'ün kendi sonucu şu: **alan-özel uydu ön-eğitimi *metrik* görevlerde
(derinlik) fayda veriyor; segmentasyon ve tespitte web ile eğitilen model
kazanıyor** — ve web modeli uzaktan algılamada yüksek çözünürlüklü
segmentasyon/tespitte SOTA kuruyor.

Bizim görevimiz **promptable segmentasyon**. Yani **LVD (web) doğru varsayılan**
— ki zaten öyle. İkinci bir sebep: SAT-493M **0.6 m/px** ortho; bizim UAV
görüntümüz birkaç **cm**/px. "Yukarıdan bakış" ortak, ölçek ~50 kat farklı.

Yine de tek string uzakta duruyor, çünkü ölçmek bedava:

```python
TEACHER = "facebook/dinov3-vitl16-pretrain-sat493m"   # aynı sınıf, aynı kod yolu
```

### Bonus: "RepViT'e distile olur mu" sorusu artık kanıtlı

Geçen turda gerekçeyle cevaplamıştım. Şimdi kanıt var: **DINOv3'ün kendisi
ViT-7B öğretmeninden ConvNeXt öğrenciler distile ediyor** (Tiny 29M → Large
198M). Yani ViT öğretmen → evrişimsel öğrenci, DINOv3 ekibinin kendi yaptığı
şey.

---

## 3. Gerçek eksik: aşama B, aşama A'yı geri alabilir

Önerinin **iki kaybı toplama** içgüdüsü boşuna değil — ama çare toplamak değil.

Bizde iki kayıp **ayrı aşamada**:

```
aşama A:  ~1 700 000 etiketsiz çift   →  kosinüs kaybı
aşama B:  ~10 000–50 000 etiketli kare →  focal + dice + IoU
```

Toplamak neden yanlış: veri iki mertebe farklı. Ortak batch ya aşama A'nın
%97'sini çöpe atar ya da iki akışlı karmaşık bir sampler ister. Ayrıca
aşamalamak **loss-ağırlıklandırma problemini tamamen ortadan kaldırıyor** ve
SAM 2'nin kendi tarifi de aşamalı.

**Ama gerçek risk şu:** aşama A encoder'ı büyük etiketsiz sette oynatıyor,
sonra aşama B onu iki mertebe küçük etiketli sette eğitiyor ve **geri
sürüklenmesini engelleyen hiçbir şey yok.**

Çare çoklu-öğretmen değil, **çapa (anchor)**:

```
aşama B kaybı = focal + dice + IoU  +  λ · (1 − cos(şimdiki öznitelik, aşama A'nın önitelikleri))
```

Yani feature distilasyonu **öğretmen olarak değil, düzenleyici olarak**. Çapa
donuk bir foundation model değil, **modelin aşama B başındaki kendi kopyası**:

- korunmaya değer olan şey aşama A'nın *öğrendiği* şey, DINOv3'ün bildiği değil
- donuk kopya 4.92 M parametre, 300 M değil
- öznitelikler zaten öğrencinin kendi uzayında → **projeksiyon gerekmiyor**
- ve **ekstra encoder geçişi yok**: `propagate_image` zaten hesapladığı
  haritayı geri veriyor

`--anchor-weight 0.5`. Sadece `--base` bir ön-eğitilmiş checkpoint'i
gösteriyorsa anlamlı.

---

## 4. Loss'a giren ikinci bulgu: direction-then-scale

arXiv 2604.27128'in dört terimli kaybı ve ağırlıkları (1.0 / 0.5 / 0.3 / 0.1),
ve asıl değerli kısmı **negatif bulgu**:

> ölçek eşleme, yön terimleriyle karşılaştırılabilir ağırlık aldığında
> **collapsed-variance failure mode** — ağ, öğretmenin varyansını üreten ama
> yönüyle hizalanmayan bir çözüme yakınsıyor.

Bizim kaybımız saf yöndü (kosinüs). Bir **moment terimi** eklendi
(`--moments`), ama **kasten farklı**: kanal başına ortalama/std, **normalize
edilmiş** haritalar üzerinde.

Neden ham büyüklüğü eşlemiyoruz: öğrencinin ham ölçeği, EdgeTAM'ın **kendi mask
decoder'ının** okumaya alıştığı şey; aradaki projektör zaten atılıyor. Yani
öğretmenin mutlak büyüklüğüne çekmek, iskeleyi decoder'ın pahasına eşlemek
olurdu. Normalizasyondan sağ çıkan şey — **hangi kanal iş yapıyor ve nerede** —
kopyalanmaya değer olan. Yakaladığı hata: yarısı harita boyunca sabitlenmiş
kanallar; pozisyon bazlı kosinüs onu hiç görmüyor.

Varsayılan **kapalı**. Makale ağırlıklarını pilot koşulardan seçmiş ve
**ablation raporlamıyor**.

---

## 5. SAM 3'ün asıl yeri: mask öğretmeni değil, **örnek kaynağı**

Öneride "SAM maskeleri mekansal hassasiyet verir" deniyor. Bizde bu **çoğu yerde
gereksiz**: VTUAV VIS **gerçek** instance maskesi veriyor. Gerçek etiket varken
SAM'ın sahte maskesini distile etmek kesinlikle daha kötü.

SAM-öğretmen'in doğru yeri, deponun zaten kullandığı yer: **kutu var maske yok**
(Anti-UAV410, `src/training/labels.py`, notebook 01/02).

**Ama SAM 3 bunu değiştiriyor** — ve bu, önerinin gerçekten haklı olduğu tek
yer, kendi belirttiğinden farklı bir sebeple. SAM 3 (Kasım 2025) **promptable
concept segmentation** yapıyor: metin prompt'u → o kavramın **bütün örnekleri**,
ayrı ayrı. LVIS zero-shot mask AP 47.0 (önceki en iyi 38.5).

Yani semantik setler için:

```
Kust4K termal kare  +  metin "car"  →  SAM 3  →  her araba ayrı maske
```

Bu, **bağlantılı bileşenlerden de, watershed'den de, geçen tur önerdiğim SAM 2
nokta-prompt ayırıcısından da iyi** — çünkü karar **görünüşe** göre veriliyor,
maske geometrisine göre değil. Bitişik iki araba ona hâlâ iki araba.

`transformers` 5.15.1'de hazır: `Sam3Model`, `Sam3Processor` (metin prompt'u
alıyor), `post_process_instance_segmentation`. Çevrimdışı koşup diske
cache'lenmeli — her iki makale de öğretmeni eğitim döngüsünün dışında tutuyor.

**Henüz yazılmadı.** Tetikleyici aynı: hücre 14'ün kaynaşma sayısı %15'i
geçerse. Ama artık çare watershed değil, bu.

---

## 6. Aerial tracking için doğrudan bulgular

- **SAM 2 ince ayarında image encoder + mask decoder'ı *birlikte* açmak en iyi
  sonucu veriyor.** Bizim `encoder` aşamamız tam olarak bu (`HEAD_MODULES +
  image_encoder`).
- **MemLoTrack** (TIR anti-UAV, Anti-UAV410) — donuk DINOv2 gövdesi + LoRA
  (r=64, bütün attention/MLP linear projeksiyonlarına), artı **kapılı FIFO
  hafıza bankası** (kapasite 7): bir kare hafızaya ancak (1) güven > 0.8 **ve**
  (2) Kalman hareket tutarlılığı geçerse yazılıyor. Anti-UAV410'da AUC 63.6 /
  SA 64.0, LoRAT-B'ye karşı +1.4 AUC.

  **Aşama C için doğrudan malzeme:** kazanç en çok **oklüzyon (+8.2 AUC)** ve
  **görüş dışı (+4.1 AUC)** senaryolarında — yani bizim düzeltmeye çalıştığımız
  hata modunun tam olarak orada. Ve çift kapı kritik: tek başına güven ya da tek
  başına hareket, ikisinin birlikteliğinin altında kalıyor (SA 62.8–63.0 vs
  64.0). Bizim `object_score` + `no_obj_ptr` problemimizin literatürdeki karşılığı
  bu, ve çözümü "hafızaya yazmayı kapıla".

  Not: LoRA'yı tam ince ayarla **karşılaştırmıyorlar**. O soru hâlâ notebook
  06 ve 07'nin.

---

## 7. Karar özeti

| öneri | hüküm | ne yapıldı |
|---|---|---|
| çoklu-öğretmen deseni | gerçek ve yayınlanmış | — |
| DINOv3 14×14 kaba grid | **yanlış**, hepsi /16 | zaten 32×32 birebir eşleşme |
| upsampling head ekle | gereksiz | mask decoder zaten 512'de |
| iki kaybı topla | bizde yanlış (veri iki mertebe farklı) | aşamalı kaldı |
| ama sürüklenme riski | **haklı** | `--anchor-weight` |
| loss dengesi | gerçek endişe | `--moments`, kapalı varsayılan + collapsed-variance uyarısı |
| domain gap → uydu modeli | **sonuç ters**: segmentasyonda web kazanıyor | LVD varsayılan, SAT tek string |
| 7B yerine daha küçük | doğru | ViT-B varsayılan (ızgara birebir), L bir string |
| SAM'ı mask öğretmeni yap | gerçek etiket varken yanlış | — |
| ama SAM 3 örnek kaynağı olarak | **en iyi çare** | pipeline hazır: `tools/make_masklets.py`, aşağıda §8 |
| teacher'ı çevrimdışı koştur | doğru | zaten öyle |

---

## 8. 2026-08 taraması: aşama C öncesi literatür, ve masklet hattının kuruluşu

İki soruya bakıldı: aşama A/B tarifinde ölçülebilir bir iyileştirme yayınlandı
mı, ve SAM 3'ü "örnek kaynağı" yapmanın (§5) somut hâli ne. Her satırın hükmü
üçlü: **şimdi** (koda girdi), **koşudan sonra** (07/08'in sayıları gelince
denenecek hipotez), **hayır**.

| bulgu | ne diyor | hüküm |
|---|---|---|
| **SAM 3 video API'si** (arXiv 2511.16719) | `transformers>=5.0.0`'da `Sam3TrackerVideo*` kutu-prompt'lu masklet üretiyor; SAM 2.1'e karşı fark kolay videoda küçük (MOSEv1 +0.5), zor/uzun videoda büyük (SA-V +5.6, LVOSv2 +8.9, MOSEv2 +12.4 J&F). Depo **gated**; SAM lisansı ticarete açık ama Apache olmayan kullanım kısıtları taşıyor (askerî kullanım hariç tutuluyor — projenin bağlamına göre okunmalı) | **şimdi** — `src/training/masklets.py` + `tools/make_masklets.py`: varsayılan öğretmen **SAM 2.1** (gated değil, Apache-2.0, `Sam2Video*` sınıfları), SAM 3 tek string (`--teacher facebook/sam3`). VIS bölümünün çizilmiş maskeleri kalibrasyon seti: `--calibrate` masklet-vs-çizim IoU'sunu basıyor, 400 etiketsiz diziye harcamadan önce o sayı okunuyor |
| **Parça parça, kutuya yeniden demirlenmiş yayılım** (AeroTrack şablonu, arXiv 2607.08075) | uzun videoda tek yayılım drift biriktirir; periyodik yeniden-tespit + kısa yayılım daha iyi | **şimdi** — masklet hattı `--chunk` (varsayılan 200) parçalarında koşuyor ve her parçayı o karenin **kendi GT kutusuyla** yeniden prompt'luyor; VTUAV her karede kutu verdiği için her kare `box_iou` kapısından geçiyor — sıradan masklet madenciliğinin sahip olmadığı bir referans |
| **DINOv3'ün ConvNeXt öğrencileri** (arXiv 2508.10104) | Meta, 7B ViT'ten distile edilmiş ConvNeXt-T/S/B/L yayınladı; conv-öğretmen → RepViT-öğrenci, geometri olarak ViT'ten daha yakın | **şimdi (ucuz kısmı)** — `FeatureTeacher` artık 4-D harita döndüren (token'sız) öğretmenleri de okuyor, yani `TEACHER = "facebook/dinov3-convnext-…"` bir string uzakta. Kazanıp kazanmadığı 07'nin üçüncü koşusuyla ölçülür, varsayılan değişmedi |
| **FreqKD** (arXiv 2606.11572) | RGB→IR distilasyonda yüksek-frekans sapması düşük-frekansın ~2.4 katı; kaybı banda bölmek (+2.4 mAP50 KAIST) ablation'lı | **koşudan sonra** — kosinüs kaybını FFT ile banda bölmek küçük bir ek; önce mevcut tarifin taban sayısı alınmalı |
| **CanKD** (arXiv 2511.21503) / maskeli-öznitelik terimi (Proteus, arXiv 2407.10366) | piksel-eşleme yerine çapraz-dikkat eşleme; ve mask-feature ek hedefi — ikisi de yoğun tahminde ablation'lı kazanç | **koşudan sonra** — ikisi de `distill_loss`'a eklenebilir terimler; hizasızlığa toleransları `--tolerance 1`'in yaptığı işi kısmen üstlenebilir |
| **SAM2LoRA / FS-SAM2** (arXiv 2510.10288, 2509.12105) | rank 16–32 tatlı nokta; az-veri rejiminde tam fine-tune hemen overfit ederken LoRA etmiyor | **koşudan sonra** — `LORA_R = 16` zaten tatlı noktada; 07/08'in LoRA-vs-finetune sütunu bu bulgunun bizim verimizdeki testi. LoRA kazanırsa bu literatürle tutarlı, sürpriz değil |
| **DAM4SAM** (arXiv 2509.13864) | SAM2-tipi hafızada dikkat dağıtıcı-farkında güncelleme; benzer küçük hedeflerde büyük kazanım | **aşama C** — hafıza yolu eğitimi başladığında bellek-seçim politikası olarak değerlendirilecek; encoder işini değiştirmiyor |
| **CST Anti-UAV** (arXiv 2507.23473) | 220 termal dizi, minik İHA'lar; en iyi takipçi %35.9 — Anti-UAV410'dan çok daha sert | **aşama C** — zor değerlendirme seti + masklet hattından geçirilecek ek termal video adayı |
| CosPress, TRACER, DGP vb. | özet ötesi doğrulanamadı ya da yoğun-tahmin kanıtı yok | **hayır** — kanıt gelene kadar |

Masklet hattının çıktısı `labels.py`'ın run-length deposunun aynısı
(`pseudo_masks.npz`), yani aşama C'nin okuyucusu iki kaynağı — Anti-UAV410
sahte maskeleri ve VTUAV masklet'leri — tek arayüzden okuyor. Kapılar da aynı
aile: `teacher_iou` kapısı videoda hiç ateşlemiyor (yayılım kare başına güven
bildirmiyor), yükü `box_iou` taşıyor.

---

## Kaynaklar

- DINOv3 — arXiv 2508.10104, <https://github.com/facebookresearch/dinov3>
- SAM 3 — <https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/>
- SigLino, çoklu-öğretmen distilasyon — arXiv 2512.20157
- SAM 3 encoder distilasyonu, direction-then-scale — arXiv 2604.27128
- MemLoTrack, TIR anti-UAV + LoRA + kapılı hafıza — PMC12694105
- AnyThermal — arXiv 2602.06203
