# Rapor: EdgeTAM'ın Jetson AGX Orin Üzerinde TensorRT ile Hızlandırılması

`docs/EXPERIMENT_LOG.md` ve `docs/tensorrt_fp16.md`'ye dayanan, Türkçe,
LaTeX kaynaklı bir rapor. Tamamen yerelde (internet gerekmeden) derlenir.

## Kurulum

XeLaTeX + KOMA-Script + polyglossia (Türkçe) + biblatex/biber + TeX Gyre
fontları gerekir. Bunların hepsi standart bir TeX Live kurulumunda vardır.

**Ubuntu / Debian:**
```bash
sudo apt-get install texlive-xetex texlive-latex-extra texlive-fonts-extra \
    texlive-fonts-recommended texlive-lang-european texlive-bibtex-extra \
    biber latexmk fonts-texgyre
```

**macOS (MacTeX):** [tug.org/mactex](https://tug.org/mactex/) tam kurulumu
yeterlidir, hepsini içerir.

**Windows (MiKTeX):** MiKTeX kurulduktan sonra eksik paketleri ilk
derlemede otomatik indirir (`Settings → Install packages on the fly`
açık olmalı); `biber`'i ayrıca MiKTeX Console'dan kurun.

## Derleme

```bash
cd report
make
```

Bu, `latexmk -xelatex` çalıştırır -- XeLaTeX, biber ve gereken kadar
tekrar geçişi otomatik sırayla yapar. Çıktı: `report/main.pdf`
(boş bir `report/` klasöründen test edilmiştir).

Elle derlemek isterseniz:
```bash
xelatex main.tex
biber main
xelatex main.tex
xelatex main.tex
```

Temizlik: `make clean` (yalnızca ara dosyalar) / `make cleanall`
(+ `main.pdf`).

## Yapı

```
report/
├── main.tex              -- ana dosya, başlık sayfası, bölüm sırası
├── preamble.tex           -- stil: fontlar, renkler, kutu/tablo tanımları
├── kaynaklar.bib          -- SAM / SAM 2 / EdgeTAM / TensorRT kaynakları
├── bolumler/
│   ├── ozet.tex
│   ├── giris.tex
│   ├── arka_plan.tex      -- SAM→SAM2→EdgeTAM, token matematiği
│   ├── sistem_yontem.tex  -- 4 motor kararı, ölçüm metodolojisi
│   ├── deneyler.tex       -- EXPERIMENT_LOG'daki tüm ölçülmüş sonuçlar
│   ├── gelecek_calismalar.tex
│   └── sonuc.tex
└── figures/                -- kendi görselleriniz (bkz. figures/README.md)
```

## Renkli kutuların anlamı

Deneyler bölümünde her sonuç iki kutudan biriyle işaretlidir:

- **Mavi (Ölçüldü, cihazda)**: Jetson AGX Orin'de gerçek bir koşudan,
  alıntılanabilir.
- **Gri (CPU'da, yapısal)**: CUDA'sız bir makinede ölçülmüş; bir
  ilişkinin var olduğunu kanıtlar, cihazdaki mutlak maliyeti değil.

Rapordaki tüm sayılar bu iki kategoriden birine girer; bekleyen (TODO)
bir ölçüm kalmamıştır.

## Görsel eklemek

`figures/README.md`'ye bakın -- her "GÖRSEL BURAYA" kutusu, gerçek bir
`\includegraphics` ile değiştirilmeyi bekleyen bir yer tutucudur ve
onlar olmadan da rapor hatasız derlenir.
