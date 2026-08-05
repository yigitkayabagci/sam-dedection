
main.pdf: main.tex preamble.tex kaynaklar.bib $(wildcard bolumler/*.tex)
	latexmk -xelatex -interaction=nonstopmode main.tex

.PHONY: clean cleanall
clean:
	latexmk -c

cleanall:
	latexmk -C
	rm -f main.bbl main.run.xml
