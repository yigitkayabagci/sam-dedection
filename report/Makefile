
# Varsayılan hedef ikisini birden üretiyor: tam rapor ve encoder eğitimi
# bölümünün ayrı basımı. İkisi de aynı kaynaktan (bolumler/egitim.tex)
# beslendiği için bölümdeki bir düzeltme ikisine birden giriyor.
all: main.pdf bolum6.pdf

ORTAK := preamble.tex eskiz.tex kaynaklar.bib $(wildcard bolumler/*.tex)

main.pdf: main.tex $(ORTAK)
	latexmk -xelatex -interaction=nonstopmode main.tex

bolum6.pdf: bolum6.tex bolumler/egitim.tex preamble.tex eskiz.tex kaynaklar.bib
	latexmk -xelatex -interaction=nonstopmode bolum6.tex

.PHONY: all clean cleanall
clean:
	latexmk -c main.tex bolum6.tex

cleanall:
	latexmk -C main.tex bolum6.tex
	rm -f main.bbl main.run.xml bolum6.bbl bolum6.run.xml
