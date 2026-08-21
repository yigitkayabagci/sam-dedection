# Rapor: EdgeTAM'ın Jetson AGX Orin Üzerinde TensorRT ile Hızlandırılması

`docs/EXPERIMENT_LOG.md` ve `docs/tensorrt_fp16.md`'ye dayanan, Türkçe,
LaTeX kaynaklı bir rapor. Tamamen yerelde (internet gerekmeden) derlenir.

## Kurulum (minimal)

XeLaTeX + KOMA-Script + polyglossia (Türkçe) + biblatex/biber + TeX Gyre
fontları gerekir. Aşağıdaki liste, bu raporun derlenirken gerçekten
açtığı dosyalardan (`main.fls`) çıkarılmış ve paketler tek tek
kaldırılıp yeniden derlenerek doğrulanmıştır. **Diskte toplam ~560 MB.**
`texlive-full` (~5 GB) gerekmez.

**Ubuntu / Debian:**
```bash
sudo apt-get install --no-install-recommends \
    texlive-xetex texlive-latex-recommended texlive-latex-extra \
    texlive-bibtex-extra texlive-pictures texlive-plain-generic \
    texlive-fonts-recommended texlive-lang-european \
    fonts-texgyre biber latexmk
```

`--no-install-recommends` önemli: onsuz `texlive-latex-extra` bir JRE,
`texlive-pictures` ise ruby ve tk çeker; hiçbiri bu rapor için gerekli
değildir (kaldırılıp derleme tekrar denenerek doğrulandı).

| paket | ne için | boyut |
|---|---|---|
| `texlive-xetex` | `xelatex`, fontspec | 15 MB |
| `texlive-latex-recommended` | KOMA-Script, polyglossia, booktabs, listings, subcaption | 25 MB |
| `texlive-latex-extra` | tcolorbox, cleveref, titling | 80 MB |
| `texlive-bibtex-extra` | biblatex | 150 MB |
| `texlive-pictures` | TikZ/PGF (Model Optimizer şeması) | 79 MB |
| `texlive-lang-european` | Türkçe tireleme | 25 MB |
| `texlive-plain-generic` + `texlive-fonts-recommended` | tek tek dosyalar, yukarıdakilerin ihtiyacı | 82 MB |
| `fonts-texgyre` | Pagella / Heros / Cursor OTF | 13 MB |
| `biber`, `latexmk` | kaynakça, derleme sürücüsü | 1 MB |

(Üstüne `texlive-base` + `texlive-binaries` + `texlive-latex-base`
bağımlılık olarak gelir, ~86 MB.)

> **Yer sıkıntısı varsa dikkat edilecek tek paket `texlive-fonts-extra`:**
> tek başına **1,7 GB** ve bu rapor ondan hiçbir dosya kullanmıyor.
> Listeye yazmayın; hiçbir paket onu bağımlılık olarak da çekmiyor.

**macOS:** tam MacTeX yerine **BasicTeX** (~100 MB) + eksikleri
`tlmgr`'la kurmak yeterlidir:
```bash
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install koma-script polyglossia fontspec booktabs caption \
    subcaption tcolorbox cleveref biblatex biber pgf listings titling \
    enumitem environ trimspaces etoolbox float latexmk tex-gyre
```

**Windows (MiKTeX):** kurulduktan sonra eksik paketleri ilk derlemede
otomatik indirir (`Settings → Install packages on the fly` açık olmalı);
`biber`'i ayrıca MiKTeX Console'dan kurun.

## Derleme

```bash
cd report
make
```

Bu, `latexmk -xelatex` çalıştırır -- XeLaTeX, biber ve gereken kadar
tekrar geçişi otomatik sırayla yapar. İki PDF üretir:

| çıktı | ne |
|---|---|
| `report/main.pdf` | tam rapor, 8 bölüm |
| `report/bolum6.pdf` | yalnızca encoder eğitimi bölümü, ayrı basım |

Tek başına biri isteniyorsa: `make main.pdf` ya da `make bolum6.pdf`.

**`bolum6.pdf` ayrı bir kopya değil**, aynı `bolumler/egitim.tex`'i aynı
stille derliyor; bölüme yapılan bir düzeltme ikisine birden giriyor. Bölüm
numarası da 6 kalıyor (`\setcounter{chapter}{5}`), böylece iki PDF'te aynı
şekle aynı numarayla atıf yapılabiliyor. Tek fark diğer bölümlere yapılan
atıflarda: ana raporda `Bölüm 4`, ayrı basımda `ana raporun 4. bölümü`.
Bunu `\diger` makrosu yapıyor (tanımı `preamble.tex`'te, ayrı basımdaki
karşılığı `bolum6.tex`'in başında).

Elle derlemek isterseniz:
```bash
xelatex main.tex
biber main
xelatex main.tex
xelatex main.tex
```

Temizlik: `make clean` (yalnızca ara dosyalar) / `make cleanall`
(+ iki PDF).

## Yapı

```
report/
├── main.tex               -- ana dosya, başlık sayfası, bölüm sırası
├── bolum6.tex             -- yalnızca encoder eğitimi bölümü, ayrı basım
├── preamble.tex           -- stil: fontlar, renkler, kutu/tablo tanımları
├── eskiz.tex              -- el çizimi ("Excalidraw") TikZ stilleri
├── kaynaklar.bib          -- SAM / SAM 2 / EdgeTAM / TensorRT kaynakları
├── bolumler/
│   ├── ozet.tex
│   ├── giris.tex
│   ├── arka_plan.tex      -- SAM→SAM2→EdgeTAM, token matematiği
│   ├── sistem_yontem.tex  -- 4 motor kararı, ölçüm metodolojisi
│   ├── deneyler.tex       -- EXPERIMENT_LOG'daki tüm ölçülmüş sonuçlar
│   ├── veri_setleri.tex   -- aday veri setleri tablosu + J&F metriği
│   ├── egitim.tex         -- encoder eğitimi: aşama A/B/C, görsel ağırlıklı
│   ├── gelecek_calismalar.tex
│   └── sonuc.tex
├── fonts/                 -- Patrick Hand (SIL OFL 1.1), eskiz etiketleri için
└── figures/               -- kendi görselleriniz (bkz. figures/README.md)
```

## Encoder eğitimi bölümü ve el çizimi şekiller

Bölüm 6 (`bolumler/egitim.tex`) raporun geri kalanından **kasten farklı**
görünüyor: on dört şeklin hepsi elle çizilmiş gibi duran, Excalidraw
tarzı TikZ kutularıyla çiziliyor. Sebebi anlatılan şeyin türü. Orada bir
ölçüm değil bir **kurulum** anlatılıyor.

Stiller tek dosyada, `eskiz.tex`'te:

| stil | ne için |
|---|---|
| `kmavi` / `kyesil` / `ksari` / `kkirmizi` / `kmor` / `kgri` / `kbos` | renkli kutular; anlam sözleşmesi dosyanın başında |
| `kdonuk` | taralı kutu = frozen modül |
| `esok` / `esokr=<renk>` / `esink` / `escizik` | oklar ve çizgiler |
| `esnot` / `esvurgu` / `esiyi` / `esbaslik` | etiketler |
| `esisaret` | bir grubu elle daire içine alma (`fit=` ile) |
| `\eskizbasla[tohum]` | her şeklin başına: titremeyi sabitler |

İki not, kurcalayacak olan için:

- **`rounded corners` ile `decorate` birlikte çalışmıyor** (``Dimension too
  large``). Bütün kutular bu yüzden keskin köşeli; titremenin kendisi zaten
  köşeyi mekanik olmaktan çıkarıyor.
- **Titreme sabit tohumlu.** `\eskizbasla` her şeklin başında
  `\pgfmathsetseed` çağırıyor, yani aynı kaynak her derlemede aynı PDF'i
  veriyor. Şekiller derlemeden derlemeye "kıpırdamıyor". Bir şeklin
  titremesi beğenilmezse tek yapılacak tohum sayısını değiştirmek.

**Yazı tipi.** Şekil etiketleri **Patrick Hand** (SIL OFL 1.1) ile
diziliyor; `report/fonts/` içinde, lisansıyla birlikte depoda duruyor, yani
ek bir kurulum gerekmiyor. Ubuntu'nun paketlediği iki el yazısı fontu
(`fonts-humor-sans`, `fonts-comic-neue`) `ğ`/`Ğ`/`İ` harflerini içermiyor,
yani Türkçe bir raporda kullanılamıyor. Seçim bu yüzden.

Font dosyası silinirse rapor **yine de hatasız derlenir**: `\elyazi`
sessizce TeX Gyre Heros'a düşer, şekiller aynı çizilir, yalnızca etiketler
düz sans olur.

## Renkli kutuların anlamı

Deneyler bölümünde (Bölüm 4) her sonuç iki kutudan biriyle işaretlidir:

- **Mavi (Ölçüldü, cihazda)**: Jetson AGX Orin'de gerçek bir koşudan,
  alıntılanabilir.
- **Gri (CPU'da, yapısal)**: CUDA'sız bir makinede ölçülmüş; bir
  ilişkinin var olduğunu kanıtlar, cihazdaki mutlak maliyeti değil.

Rapordaki **her sayı** bu iki kategoriden birine girer; bekleyen (TODO) bir
ölçüm kalmamıştır.


## Görsel eklemek

`figures/README.md`'ye bakın -- her "GÖRSEL BURAYA" kutusu, gerçek bir
`\includegraphics` ile değiştirilmeyi bekleyen bir yer tutucudur ve
onlar olmadan da rapor hatasız derlenir.
